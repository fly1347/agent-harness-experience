"""
文件作用：
执行 Step 9 的完整 T01～T10 Scenario Replay。Full History、Last-N、Managed Context
三种策略使用同一 Scenario，并在 T08 后真正结束当前 Python worker，再由新 worker
从 SQLite 恢复 Session / Memory，继续运行 T09～T10。

整体结构：
1）母调度流程：依次为三种策略启动 pre-restart / post-restart 两个独立 worker；
2）pre-restart：新建 Session / MemoryManager，运行 T01～T08，每轮由 AgentLoop 自动持久化；
3）post-restart：仅从 SQLite 恢复 Session / Memory，继续运行 T09～T10；
4）逐轮校验：记录答案关键词、预期工具、Memory 动作、restart recovery、token、耗时和 Trace；
5）归档：每种策略同时生成带时间戳 JSONL + 人类可读 MD，并生成三策略总对比 MD；完整执行细节仍留在 Trace。

职责边界：
- 本脚本只负责完整 Replay，不修改 AgentLoop / ContextBuilder 的运行语义；
- T09 主要测跨 restart Memory，T10 主要测早期 Conversation / Context；
- T10 单独记录 no_refetch_success，用来观察“是否无需重新读取工具即可回答”；
  该字段不是总体答案质量分，也不等同于 Managed Context 成败；
- 报告只根据本次真实 JSONL 结果生成，不在运行前预写结论。
"""

from __future__ import annotations

import argparse
import io
import json
import os
import subprocess
import sys
from contextlib import redirect_stdout
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from dotenv import load_dotenv

from mini_agent_harness.core.agent_loop import AgentLoop
from mini_agent_harness.core.context import ContextBuilder
from mini_agent_harness.core.memory import MemoryManager
from mini_agent_harness.core.session import Session
from mini_agent_harness.core.tools import ToolRegistry
from mini_agent_harness.core.tracing import TraceLogger
from mini_agent_harness.providers.openai_compatible import OpenAICompatibleProvider
from mini_agent_harness.storage.sqlite_store import SQLiteStore


STRATEGIES = ("full_history", "last_n", "managed")
PRE_RESTART_TURNS = tuple(f"T{i:02d}" for i in range(1, 9))
POST_RESTART_TURNS = ("T09", "T10")
EXPECTED_FOCUS = "Context Management + 评估信度"


# 返回实验项目根目录。
def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


# 读取并校验冻结的 T01～T10 / T08 restart Scenario。
def _load_scenario(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    turns = data.get("turns")
    if not isinstance(turns, list) or len(turns) < 10:
        raise ValueError("Scenario must contain T01..T10.")

    expected_ids = [f"T{i:02d}" for i in range(1, 11)]
    actual_ids = [turn.get("turn_id") for turn in turns[:10]]
    if actual_ids != expected_ids:
        raise ValueError(f"Expected frozen turns {expected_ids}; got {actual_ids}")
    if data.get("restart_after_turn") != "T08":
        raise ValueError("Step 9 requires restart_after_turn = T08.")
    return data


# 建立 turn_id -> turn 映射。
def _turns_by_id(scenario: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {turn["turn_id"]: turn for turn in scenario["turns"]}


# 按策略创建 ContextBuilder；参数与 Step 6/7 保持同一口径。
def _builder(
    strategy: str,
    *,
    last_n: int,
    summary_trigger: int,
) -> ContextBuilder:
    if strategy == "full_history":
        return ContextBuilder("full_history")
    if strategy == "last_n":
        return ContextBuilder("last_n", last_n=last_n)
    if strategy == "managed":
        return ContextBuilder(
            "managed",
            managed_recent_n=last_n,
            summary_trigger_messages=summary_trigger,
        )
    raise ValueError(f"Unknown strategy: {strategy}")


# 创建真实模型 Provider 与固定三个 Tool。
def _runtime(project_root: Path) -> tuple[OpenAICompatibleProvider, ToolRegistry]:
    load_dotenv(project_root / ".env")
    return (
        OpenAICompatibleProvider(),
        ToolRegistry(
            fixtures_dir=project_root / "fixtures",
            notes_file=project_root / "artifacts" / "notes.md",
        ),
    )


# 生成不会覆盖旧结果的 Step 9 Trace 文件名。
def _trace_path(
    project_root: Path,
    *,
    strategy: str,
    session_id: str,
    turn_id: str,
) -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    return (
        project_root
        / "artifacts"
        / "traces"
        / f"step9-{strategy}-{session_id[:8]}-{turn_id.lower()}-{stamp}.jsonl"
    )


# 读取 Trace 中某类事件的 payload。
def _events(tracer: TraceLogger, event: str) -> list[dict[str, Any]]:
    return [
        record["payload"]
        for record in tracer.records
        if record["event"] == event
    ]


# 读取本轮 RUN_SUMMARY。
def _run_summary(tracer: TraceLogger) -> dict[str, Any]:
    rows = _events(tracer, "RUN_SUMMARY")
    return rows[-1] if rows else {}


# 抽取本轮 working context 的关键统计。
def _context_stats(tracer: TraceLogger) -> dict[str, Any]:
    rows = _events(tracer, "CONTEXT_BUILD")
    if not rows:
        return {
            "history_messages": 0,
            "request_messages": 0,
            "summary_injected": False,
            "summary": "",
            "active_memory_count": 0,
        }

    summaries = [row for row in rows if row.get("summary_injected")]
    return {
        "history_messages": int(rows[0].get("history_message_count") or 0),
        "request_messages": int(rows[0].get("request_message_count") or 0),
        "summary_injected": bool(summaries),
        "summary": summaries[-1].get("summary", "") if summaries else "",
        "active_memory_count": max(
            int(row.get("active_memory_count") or 0) for row in rows
        ),
    }


# 抽取本轮工具调用。
def _tool_calls(tracer: TraceLogger) -> list[dict[str, Any]]:
    return [
        {
            "name": payload.get("name"),
            "arguments": payload.get("arguments") or {},
        }
        for payload in _events(tracer, "TOOL_CALL")
    ]


# 校验答案关键词。
def _contains_check(answer: str, expected: list[str]) -> tuple[bool, list[str]]:
    folded = answer.casefold()
    missing = [item for item in expected if item.casefold() not in folded]
    return not missing, missing


# 校验固定工具；expected_tool=null 时不强制某个工具。
def _tool_check(
    calls: list[dict[str, Any]],
    expected: dict[str, Any] | None,
) -> bool | None:
    if expected is None:
        return None

    for call in calls:
        if call.get("name") != expected.get("name"):
            continue
        document = expected.get("document")
        if document is None or call.get("arguments", {}).get("name") == document:
            return True
    return False


# 校验 T04 write、T07 supersede、T09 read/inject。
def _memory_check(
    tracer: TraceLogger,
    expected: dict[str, Any] | None,
) -> bool | None:
    if expected is None:
        return None

    action = expected.get("action")
    key = expected.get("key")
    fragments = expected.get("value") or []

    if action == "write":
        for payload in _events(tracer, "MEMORY_WRITE"):
            record = payload.get("record") or {}
            value = str(record.get("value") or "")
            if record.get("key") == key and all(x in value for x in fragments):
                return True
        return False

    if action == "supersede":
        for payload in _events(tracer, "MEMORY_SUPERSEDE"):
            old = payload.get("previous") or {}
            new = payload.get("record") or {}
            value = str(new.get("value") or "")
            if (
                old.get("key") == key
                and old.get("status") == "superseded"
                and new.get("key") == key
                and all(x in value for x in fragments)
            ):
                return True
        return False

    if action == "read":
        read_ok = any(
            any(item.get("key") == key for item in payload.get("records", []))
            for payload in _events(tracer, "MEMORY_READ")
        )
        inject_ok = any(
            any(item.get("key") == key for item in payload.get("records", []))
            for payload in _events(tracer, "MEMORY_INJECT")
        )
        return read_ok and inject_ok

    raise ValueError(f"Unsupported memory action: {action}")


# 确认新 worker 恢复了 Conversation 与完整 Memory Lifecycle。
def _recovery_ok(session: Session, memory: MemoryManager) -> bool:
    records = [
        item
        for item in memory.records
        if item.kind == "decision" and item.key == "experiment_focus"
    ]
    active = memory.retrieve_active(kind="decision", key="experiment_focus")
    return (
        len(session.messages) > 1
        and len(records) >= 2
        and {item.status for item in records}.issuperset({"active", "superseded"})
        and len(active) == 1
        and active[0].value == EXPECTED_FOCUS
    )


# 抑制 TraceLogger 的逐事件终端打印；异常时再显示捕获内容。
def _run_quiet(agent: AgentLoop, user_input: str, session: Session) -> str:
    buffer = io.StringIO()
    try:
        with redirect_stdout(buffer):
            return agent.run(user_input, session=session)
    except Exception:
        captured = buffer.getvalue().strip()
        if captured:
            print(captured)
        raise


# 把单轮结果追加到策略 JSONL。
def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")


# 运行一个 Turn，并立即把验收结果落盘。
def _run_turn(
    *,
    project_root: Path,
    strategy: str,
    turn: dict[str, Any],
    session: Session,
    memory: MemoryManager,
    store: SQLiteStore,
    provider: OpenAICompatibleProvider,
    tools: ToolRegistry,
    builder: ContextBuilder,
    output_path: Path,
    restart_ok: bool | None = None,
) -> None:
    turn_id = turn["turn_id"]
    tracer = TraceLogger(
        _trace_path(
            project_root,
            strategy=strategy,
            session_id=session.session_id,
            turn_id=turn_id,
        )
    )

    if turn_id == "T09":
        tracer.log(
            "STATE_RECOVER",
            {
                "session_id": session.session_id,
                "message_count": len(session.messages),
                "memory_record_count": len(memory.records),
                "recovery_check": restart_ok,
                "reason": (
                    "新 Python worker 仅从 SQLite 恢复 Session 与 Memory，"
                    "随后继续固定 Scenario。"
                ),
            },
            console={"recovery_check": restart_ok},
        )

    agent = AgentLoop(
        provider=provider,
        tools=tools,
        tracer=tracer,
        context_builder=builder,
        memory_manager=memory,
        state_store=store,
    )
    answer = _run_quiet(agent, turn["user_input"], session)

    calls = _tool_calls(tracer)
    summary = _run_summary(tracer)
    tokens = summary.get("tokens") or {}
    timing = summary.get("timing_ms") or {}

    expected_contains = turn.get("expected_contains") or []
    contains_ok, missing = _contains_check(answer, expected_contains)
    tool_ok = _tool_check(calls, turn.get("expected_tool"))
    memory_ok = _memory_check(tracer, turn.get("expected_memory_action"))

    turn_success = (
        contains_ok
        and tool_ok is not False
        and memory_ok is not False
        and restart_ok is not False
    )

    # T10 额外观察：是否无需重新读取工具即可回答。
    # 这是交互/成本信号，不作为 Context Strategy 总体质量分。
    no_refetch_success = None
    if turn_id == "T10":
        no_refetch_success = contains_ok and not calls

    record = {
        "strategy": strategy,
        "session_id": session.session_id,
        "turn_id": turn_id,
        "user_input": turn["user_input"],
        "answer": answer,
        "turn_success": turn_success,
        "contains_match": contains_ok,
        "missing_contains": missing,
        "expected_tool_match": tool_ok,
        "tool_calls": calls,
        "tool_call_count": int(summary.get("tool_calls") or 0),
        "memory_action_success": memory_ok,
        "restart_recovery_success": restart_ok if turn_id == "T09" else None,
        "no_refetch_success": no_refetch_success,
        "model_calls": int(summary.get("model_calls") or 0),
        "input_tokens": int(tokens.get("prompt") or 0),
        "output_tokens": int(tokens.get("completion") or 0),
        "latency_ms": float(timing.get("end_to_end") or 0.0),
        "estimated_cost_cny": float(summary.get("estimated_cost_cny") or 0.0),
        "messages_after_turn": len(session.messages),
        "context": _context_stats(tracer),
        "trace_jsonl": str(tracer.path),
        "trace_md": str(tracer.md_path),
    }
    _append_jsonl(output_path, record)

    status = "PASS" if turn_success else "FAIL"
    detail = ""
    if turn_id == "T10":
        detail = (
            " | no_refetch="
            + ("YES" if no_refetch_success else "NO")
        )
    if missing:
        detail += " | missing=" + ",".join(missing)
    print(
        f"[{strategy:<12}] {turn_id} {status} | "
        f"model={record['model_calls']} tool={record['tool_call_count']} | "
        f"input={record['input_tokens']:,} | "
        f"e2e={record['latency_ms'] / 1000:.3f}s{detail}"
    )


# worker A：T01～T08。函数返回后该 Python 进程由 subprocess 正常退出。
def _pre_restart_worker(
    *,
    project_root: Path,
    scenario_path: Path,
    strategy: str,
    db_path: Path,
    output_path: Path,
    session_id: str,
    last_n: int,
    summary_trigger: int,
) -> None:
    turns = _turns_by_id(_load_scenario(scenario_path))
    provider, tools = _runtime(project_root)
    session = Session(session_id=session_id)
    memory = MemoryManager()
    builder = _builder(
        strategy,
        last_n=last_n,
        summary_trigger=summary_trigger,
    )

    print(f"\n===== {strategy} PRE | pid={os.getpid()} | T01..T08 =====")
    with SQLiteStore(db_path) as store:
        for turn_id in PRE_RESTART_TURNS:
            _run_turn(
                project_root=project_root,
                strategy=strategy,
                turn=turns[turn_id],
                session=session,
                memory=memory,
                store=store,
                provider=provider,
                tools=tools,
                builder=builder,
                output_path=output_path,
            )

        active = memory.retrieve_active(kind="decision", key="experiment_focus")
        if not active or active[0].value != EXPECTED_FOCUS:
            raise RuntimeError("T08 pre-restart active Memory is not the latest value.")
        if store.load_session(session_id) is None:
            raise RuntimeError("Session was not persisted before restart.")

    print(f"===== {strategy} PRE EXIT | pid={os.getpid()} =====")


# worker B：只凭 SQLite 恢复状态并运行 T09～T10。
def _post_restart_worker(
    *,
    project_root: Path,
    scenario_path: Path,
    strategy: str,
    db_path: Path,
    output_path: Path,
    session_id: str,
    last_n: int,
    summary_trigger: int,
) -> None:
    turns = _turns_by_id(_load_scenario(scenario_path))
    provider, tools = _runtime(project_root)
    builder = _builder(
        strategy,
        last_n=last_n,
        summary_trigger=summary_trigger,
    )

    print(f"\n===== {strategy} POST | pid={os.getpid()} | T09..T10 =====")
    with SQLiteStore(db_path) as store:
        session = store.load_session(session_id)
        if session is None:
            raise RuntimeError(f"Session not found after restart: {session_id}")
        memory = MemoryManager.from_records(store.load_memory_records())
        restart_ok = _recovery_ok(session, memory)

        for turn_id in POST_RESTART_TURNS:
            _run_turn(
                project_root=project_root,
                strategy=strategy,
                turn=turns[turn_id],
                session=session,
                memory=memory,
                store=store,
                provider=provider,
                tools=tools,
                builder=builder,
                output_path=output_path,
                restart_ok=restart_ok if turn_id == "T09" else None,
            )

    print(f"===== {strategy} POST EXIT | pid={os.getpid()} =====")


# 构造内部 worker 命令。
def _worker_command(
    *,
    script: Path,
    phase: str,
    strategy: str,
    scenario: Path,
    db_path: Path,
    output_path: Path,
    session_id: str,
    last_n: int,
    summary_trigger: int,
) -> list[str]:
    return [
        sys.executable,
        str(script),
        "--worker-phase",
        phase,
        "--strategy",
        strategy,
        "--scenario",
        str(scenario),
        "--db-path",
        str(db_path),
        "--output-path",
        str(output_path),
        "--session-id",
        session_id,
        "--last-n",
        str(last_n),
        "--summary-trigger",
        str(summary_trigger),
    ]


# 读取一份策略结果，用于母调度流程打印最终摘要。
def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]



# 把三态校验值转成适合 Markdown 展示的文本。
def _pass_text(value: bool | None) -> str:
    if value is None:
        return "—"
    return "PASS" if value else "FAIL"


# 汇总一套 Context Strategy 的模型调用、工具、token、耗时与成本。
def _strategy_totals(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "turn_passed": sum(1 for row in rows if row["turn_success"]),
        "model_calls": sum(int(row.get("model_calls") or 0) for row in rows),
        "tool_calls": sum(int(row.get("tool_call_count") or 0) for row in rows),
        "input_tokens": sum(int(row.get("input_tokens") or 0) for row in rows),
        "output_tokens": sum(int(row.get("output_tokens") or 0) for row in rows),
        "latency_ms": sum(float(row.get("latency_ms") or 0.0) for row in rows),
        "estimated_cost_cny": sum(
            float(row.get("estimated_cost_cny") or 0.0) for row in rows
        ),
        "summary_turns": [
            row["turn_id"]
            for row in rows
            if (row.get("context") or {}).get("summary_injected")
        ],
    }


# 兼容旧 Step 9 JSONL 的 context_recall_success，并统一解释为“无需重新读取工具”。
def _no_refetch_value(row: dict[str, Any]) -> bool | None:
    if "no_refetch_success" in row:
        return row.get("no_refetch_success")
    return row.get("context_recall_success")


# 从本批真实结果还原 T01～T10 核心脚本，便于报告随时对照实验目的。
def _scenario_block(rows: list[dict[str, Any]]) -> list[str]:
    lines = ["## Core Scenario", "", "```text"]
    for row in rows:
        lines.append(f"{row['turn_id']}  {row['user_input']}")
        if row["turn_id"] == "T08":
            lines.extend(["", "--- restart ---", ""])
    lines.extend(["```", ""])
    return lines


# 为单套 Context Strategy 生成与 JSONL 配套的人类可读 MD。
def _write_strategy_report(
    path: Path,
    *,
    strategy: str,
    rows: list[dict[str, Any]],
    jsonl_path: Path,
) -> None:
    totals = _strategy_totals(rows)
    t09 = next(row for row in rows if row["turn_id"] == "T09")
    t10 = next(row for row in rows if row["turn_id"] == "T10")

    lines = [
        f"# Step 9 Full Replay — {strategy}",
        "",
        f"- Session: `{rows[0]['session_id']}`",
        "- Scenario: `T01 -> ... -> T08 -> Python process exit -> new Python process -> T09 -> T10`",
        f"- Automated checks: **{totals['turn_passed']}/10 PASS**",
        f"- Restart recovery: **{_pass_text(t09.get('restart_recovery_success'))}**",
        f"- T10 answer check: **{_pass_text(t10.get('contains_match'))}**",
        f"- T10 no-refetch: **{'YES' if _no_refetch_value(t10) else 'NO'}**",
        f"- JSONL: `{jsonl_path}`",
        "",
    ]
    lines.extend(_scenario_block(rows))
    lines.extend([
        "## Aggregate",
        "",
        f"- Model calls: **{totals['model_calls']}**",
        f"- Tool calls: **{totals['tool_calls']}**",
        f"- Input tokens: **{totals['input_tokens']:,}**",
        f"- Output tokens: **{totals['output_tokens']:,}**",
        f"- End-to-end total: **{totals['latency_ms'] / 1000:.3f} s**",
        f"- Estimated cost: **¥{totals['estimated_cost_cny']:.6f}**",
        "- Summary injected turns: "
        + (
            "`" + ", ".join(totals["summary_turns"]) + "`"
            if totals["summary_turns"]
            else "**none**"
        ),
        "",
        "## Turn Results",
        "",
        "| Turn | Check | Model | Tool | Input | Output | E2E | Memory | Restart | No refetch |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |",
    ])

    for row in rows:
        lines.append(
            "| {turn} | {check} | {model} | {tool} | {input:,} | {output:,} | "
            "{e2e:.3f}s | {memory} | {restart} | {recall} |".format(
                turn=row["turn_id"],
                check="PASS" if row["turn_success"] else "FAIL",
                model=int(row.get("model_calls") or 0),
                tool=int(row.get("tool_call_count") or 0),
                input=int(row.get("input_tokens") or 0),
                output=int(row.get("output_tokens") or 0),
                e2e=float(row.get("latency_ms") or 0.0) / 1000,
                memory=_pass_text(row.get("memory_action_success")),
                restart=_pass_text(row.get("restart_recovery_success")),
                recall=("YES" if _no_refetch_value(row) else "NO") if row["turn_id"] == "T10" else "—",
            )
        )

    t10_tools = int(t10.get("tool_call_count") or 0)
    if _no_refetch_value(t10):
        t10_note = (
            "T10 在 **不重新调用工具** 的情况下回答了最早 T01/T02 的信息，"
            "因此本次记录为 no-refetch=YES。它只表示无需重新取证，不代表总体质量评分。"
        )
    else:
        t10_note = (
            f"T10 最终答案关键词检查通过，但重新调用了 **{t10_tools}** 次工具；"
            "因此本次记录为 no-refetch=NO。最终答案仍可正确；这里记录的是重新取证成本，"
            "不是把该 Context Strategy 判失败。"
        )

    lines.extend(
        [
            "",
            "## Restart / Recall",
            "",
            "### T09 — Memory after restart",
            "",
            f"- Recovery: **{_pass_text(t09.get('restart_recovery_success'))}**",
            f"- Memory action: **{_pass_text(t09.get('memory_action_success'))}**",
            f"- Tool calls: **{int(t09.get('tool_call_count') or 0)}**",
            "",
            t09.get("answer", ""),
            "",
            "### T10 — Early conversation / source re-access",
            "",
            f"- Keyword answer check: **{_pass_text(t10.get('contains_match'))}**",
            f"- No-refetch: **{'YES' if _no_refetch_value(t10) else 'NO'}**",
            f"- Tool calls: **{t10_tools}**",
            "",
            t10_note,
            "",
            t10.get("answer", ""),
            "",
            "## Trace Index",
            "",
        ]
    )

    for row in rows:
        lines.append(
            f"- {row['turn_id']}: `{row.get('trace_md', '')}`"
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


# 根据三份真实 JSONL 生成 Context Strategy 总对比 MD。
def _write_comparison_report(
    path: Path,
    *,
    outputs: dict[str, Path],
) -> None:
    rows_by_strategy = {
        strategy: _read_jsonl(jsonl_path)
        for strategy, jsonl_path in outputs.items()
    }
    totals = {
        strategy: _strategy_totals(rows)
        for strategy, rows in rows_by_strategy.items()
    }

    full_input = totals.get("full_history", {}).get("input_tokens", 0)

    lines = [
        "# Step 9 Context Strategy Comparison",
        "",
        "本报告由本次 Step 9 Full Replay 的真实 JSONL 自动生成。",
        "",
    ]
    first_rows = next(iter(rows_by_strategy.values()))
    lines.extend(_scenario_block(first_rows))
    lines.extend([
        "## Result",
        "",
        "| Strategy | Automated checks | Restart | T10 answer | T10 no-refetch | Model calls | Tool calls | Input tokens | Output tokens | E2E total | Cost |",
        "| --- | ---: | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])

    for strategy, rows in rows_by_strategy.items():
        total = totals[strategy]
        t09 = next(row for row in rows if row["turn_id"] == "T09")
        t10 = next(row for row in rows if row["turn_id"] == "T10")
        lines.append(
            "| {strategy} | {passed}/10 | {restart} | {answer} | {refetch} | {model} | "
            "{tool} | {input:,} | {output:,} | {e2e:.3f}s | ¥{cost:.6f} |".format(
                strategy=strategy,
                passed=total["turn_passed"],
                restart=_pass_text(t09.get("restart_recovery_success")),
                answer=_pass_text(t10.get("contains_match")),
                refetch=("YES" if _no_refetch_value(t10) else "NO"),
                model=total["model_calls"],
                tool=total["tool_calls"],
                input=total["input_tokens"],
                output=total["output_tokens"],
                e2e=total["latency_ms"] / 1000,
                cost=total["estimated_cost_cny"],
            )
        )

    lines.extend(["", "## Context Behavior", ""])
    for strategy, rows in rows_by_strategy.items():
        total = totals[strategy]
        t10 = next(row for row in rows if row["turn_id"] == "T10")
        saving = None
        if full_input and strategy != "full_history":
            saving = 1 - (total["input_tokens"] / full_input)

        detail = [
            f"- **{strategy}**: T10 answer **{_pass_text(t10.get('contains_match'))}**; "
            f"no-refetch **{'YES' if _no_refetch_value(t10) else 'NO'}**",
            f"  - T10 tool calls: **{int(t10.get('tool_call_count') or 0)}**",
            f"  - total input tokens: **{total['input_tokens']:,}**",
        ]
        if saving is not None:
            detail.append(
                f"  - input-token reduction vs full_history: **{saving * 100:.1f}%**"
            )
        detail.append(
            "  - summary injected turns: "
            + (
                "`" + ", ".join(total["summary_turns"]) + "`"
                if total["summary_turns"]
                else "**none**"
            )
        )
        lines.extend(detail)

    t09_by_strategy = {
        strategy: next(row for row in rows if row["turn_id"] == "T09")
        for strategy, rows in rows_by_strategy.items()
    }
    t10_by_strategy = {
        strategy: next(row for row in rows if row["turn_id"] == "T10")
        for strategy, rows in rows_by_strategy.items()
    }

    recovery_text = ", ".join(
        f"{strategy}={_pass_text(t09.get('restart_recovery_success'))}"
        for strategy, t09 in t09_by_strategy.items()
    )
    t10_observation_text = ", ".join(
        f"{strategy}: no-refetch={'YES' if _no_refetch_value(t10) else 'NO'}, "
        f"tool={int(t10.get('tool_call_count') or 0)}"
        for strategy, t10 in t10_by_strategy.items()
    )
    input_observation_text = ", ".join(
        f"{strategy}={totals[strategy]['input_tokens']:,}"
        for strategy in rows_by_strategy
    )

    lines.extend(
        [
            "",
            "## What this run demonstrates",
            "",
            f"1. **本批 Persistence / Recovery 结果：** {recovery_text}。T08 后真实更换 Python worker，再从 SQLite 恢复 Session 与 Memory。",
            f"2. **本批 input-token 观测：** {input_observation_text}。这些是本次 Replay 的实测值，不把单批数值外推为策略的固定性能。",
            f"3. **本批 T10 观测：** {t10_observation_text}。no-refetch 与 Tool Call 数只描述本轮实际路径。",
            "4. **Tool Calling 路径由模型动态决策。** 即使 Scenario、Context Strategy 与 Tool Schema 不变，模型仍可能在不同运行中选择不同的 list / read / save 路径；因此 Tool Call、Model Call 与 E2E 存在 run-to-run variation。",
            "5. **Context Strategy 与 Tool Path 要分开解释。** Context Strategy 决定模型能看到什么；模型是否立即调用工具、先 list 还是直接 read、是否做冗余 save，则属于本次模型决策，不能由单次 Tool Call 数反推成策略固有性质。",
            "",
            "## Evaluation Boundary",
            "",
            "- T10 的 `no-refetch` 只表示‘本次是否无需重新读取工具就回答’，它是交互 / 成本信号，不是总体答案质量分；",
            "- 当前 System Prompt 要求不能在未读取 fixture 的情况下声称知道其内容；Full History 保留旧 Tool Result，而 Managed Summary 不保留原始 Tool Result，因此 no-refetch 的结构条件并不完全对称；",
            "- Tool Call、Model Call、E2E、Cost 都是**单次运行观测值**。若要比较策略的稳定性能，需要重复运行并报告分布，而不能用某一批路径作为固定结论；",
            "- `temperature=0` 也不应被理解为整个 Agent Tool Path 必然完全确定；模型服务、采样实现和工具决策仍可能产生运行间差异；",
            "- 原始 Scenario 对 Managed Context 的目标是观察摘要保留、token 压缩、重新取证与状态一致性；不能只用一个 no-refetch 或单次 Tool Call 数给 Managed 下结论。",
            "",
            "## Source JSONL",
            "",
        ]
    )

    for strategy, jsonl_path in outputs.items():
        lines.append(f"- {strategy}: `{jsonl_path}`")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


# 母调度流程：每种策略严格经过 worker A 退出 -> worker B 新启动。
def _orchestrate(
    *,
    project_root: Path,
    scenario_path: Path,
    strategies: list[str],
    last_n: int,
    summary_trigger: int,
) -> None:
    _load_scenario(scenario_path)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    script = Path(__file__).resolve()
    scenario_dir = project_root / "artifacts" / "scenarios"
    report_dir = project_root / "artifacts" / "reports"
    state_dir = project_root / "artifacts" / "state"
    scenario_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    state_dir.mkdir(parents=True, exist_ok=True)

    print("\n===== STEP 9 FULL REPLAY =====")
    print("T01..T10; real restart after T08")
    print("strategies:", ", ".join(strategies))

    outputs: dict[str, Path] = {}
    reports: dict[str, Path] = {}

    for strategy in strategies:
        session_id = uuid4().hex
        db_path = state_dir / f"step9-{strategy}-{stamp}.db"
        output_path = scenario_dir / f"step9-{strategy}-{stamp}.jsonl"
        db_path.unlink(missing_ok=True)
        output_path.unlink(missing_ok=True)

        subprocess.run(
            _worker_command(
                script=script,
                phase="pre_restart",
                strategy=strategy,
                scenario=scenario_path,
                db_path=db_path,
                output_path=output_path,
                session_id=session_id,
                last_n=last_n,
                summary_trigger=summary_trigger,
            ),
            cwd=project_root,
            check=True,
        )

        print(f"----- {strategy}: REAL RESTART -----")

        subprocess.run(
            _worker_command(
                script=script,
                phase="post_restart",
                strategy=strategy,
                scenario=scenario_path,
                db_path=db_path,
                output_path=output_path,
                session_id=session_id,
                last_n=last_n,
                summary_trigger=summary_trigger,
            ),
            cwd=project_root,
            check=True,
        )

        rows = _read_jsonl(output_path)
        if len(rows) != 10:
            raise RuntimeError(f"{strategy}: expected 10 records, got {len(rows)}")
        outputs[strategy] = output_path
        report_path = report_dir / f"step9-{strategy}-{stamp}.md"
        _write_strategy_report(
            report_path,
            strategy=strategy,
            rows=rows,
            jsonl_path=output_path,
        )
        reports[strategy] = report_path

    comparison_path = report_dir / f"step9-context-strategy-comparison-{stamp}.md"
    _write_comparison_report(
        comparison_path,
        outputs=outputs,
    )

    print("\n===== STEP 9 RUN COMPLETE =====")
    for strategy in strategies:
        rows = _read_jsonl(outputs[strategy])
        t09 = next(row for row in rows if row["turn_id"] == "T09")
        t10 = next(row for row in rows if row["turn_id"] == "T10")
        passed = sum(1 for row in rows if row["turn_success"])
        print(
            f"{strategy}: {passed}/10 automated checks | "
            f"restart={'PASS' if t09['restart_recovery_success'] else 'FAIL'} | "
            f"T10 answer={'PASS' if t10['contains_match'] else 'FAIL'} | "
            f"no_refetch={'YES' if _no_refetch_value(t10) else 'NO'}"
        )
        print(f"  JSONL: {outputs[strategy]}")
        print(f"  MD:    {reports[strategy]}")

    print(f"comparison MD: {comparison_path}")


# 默认一次跑三策略；隐藏 worker 参数仅由母调度流程内部使用。
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Step 9 full T01-T10 replay with real restart after T08."
    )
    parser.add_argument(
        "--scenario",
        default="scenarios/multi_turn_context.json",
    )
    parser.add_argument(
        "--strategy",
        choices=["all", *STRATEGIES],
        default="all",
    )
    parser.add_argument("--last-n", type=int, default=6)
    parser.add_argument("--summary-trigger", type=int, default=8)

    # 内部 worker 参数，不需要人工填写。
    parser.add_argument(
        "--worker-phase",
        choices=["pre_restart", "post_restart"],
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--db-path", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--output-path", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--session-id", default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.last_n < 1 or args.summary_trigger < 1:
        raise ValueError("--last-n and --summary-trigger must be >= 1")

    project_root = _project_root()
    raw_scenario = Path(args.scenario)
    scenario_path = (
        raw_scenario
        if raw_scenario.is_absolute()
        else project_root / raw_scenario
    )

    if args.worker_phase is None:
        strategies = list(STRATEGIES) if args.strategy == "all" else [args.strategy]
        _orchestrate(
            project_root=project_root,
            scenario_path=scenario_path,
            strategies=strategies,
            last_n=args.last_n,
            summary_trigger=args.summary_trigger,
        )
        return

    if args.strategy == "all":
        raise ValueError("Internal worker needs one concrete strategy.")
    if not args.db_path or not args.output_path or not args.session_id:
        raise ValueError("Internal worker is missing db/output/session arguments.")

    kwargs = {
        "project_root": project_root,
        "scenario_path": scenario_path,
        "strategy": args.strategy,
        "db_path": Path(args.db_path),
        "output_path": Path(args.output_path),
        "session_id": args.session_id,
        "last_n": args.last_n,
        "summary_trigger": args.summary_trigger,
    }

    if args.worker_phase == "pre_restart":
        _pre_restart_worker(**kwargs)
    else:
        _post_restart_worker(**kwargs)


if __name__ == "__main__":
    main()
