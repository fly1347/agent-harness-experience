"""
文件作用：
运行 Step 7 的固定 Memory Lifecycle 实验，直接复用 scenarios/multi_turn_context.json 中的 T04～T08，观察长期记忆从写入、跨轮读取/注入到显式替换的完整过程。

整体结构：
1）读取固定 Scenario，并只选择 T04～T08，避免提前跨入 Step 8 的 restart 边界；
2）创建同一进程内的 Session、MemoryManager、Managed Context 和 AgentLoop；
3）逐轮真实调用模型和工具，分别收集 Memory Trace、工具调用、token 与最终 active memory；
4）按 Scenario 的 expected_memory_action 做结果校验，但 expected 字段只用于验收，不参与 Memory 动作执行；
5）输出带时间戳的 JSONL / Markdown 报告，并在终端打印最小 PASS/FAIL 汇总。
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from mini_agent_harness.core.agent_loop import AgentLoop
from mini_agent_harness.core.context import ContextBuilder
from mini_agent_harness.core.memory import MemoryManager
from mini_agent_harness.core.session import Session
from mini_agent_harness.core.tools import ToolRegistry
from mini_agent_harness.core.tracing import TraceLogger
from mini_agent_harness.providers.openai_compatible import OpenAICompatibleProvider


_START_TURN = "T04"
_END_TURN = "T08"


def _project_root() -> Path:
    """返回当前实验项目根目录。"""
    return Path(__file__).resolve().parents[1]


def _load_turns(path: Path) -> list[dict[str, Any]]:
    """从固定 Scenario 中截取 Step 7 使用的 T04～T08。"""
    data = json.loads(path.read_text(encoding="utf-8"))
    selected: list[dict[str, Any]] = []
    active = False

    for turn in data["turns"]:
        turn_id = turn["turn_id"]
        if turn_id == _START_TURN:
            active = True
        if active:
            selected.append(turn)
        if turn_id == _END_TURN:
            break

    if not selected or selected[0]["turn_id"] != _START_TURN:
        raise ValueError("Scenario 中未找到 Step 7 起始轮 T04。")
    if selected[-1]["turn_id"] != _END_TURN:
        raise ValueError("Scenario 中未找到 Step 7 结束轮 T08。")
    return selected


def _trace_path(project_root: Path, session_id: str, turn_id: str) -> Path:
    """为每轮 Step 7 Trace 生成独立带时间戳文件名。"""
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    return (
        project_root
        / "artifacts"
        / "traces"
        / f"step7-memory-{session_id[:8]}-{turn_id.lower()}-{stamp}.jsonl"
    )


def _event_payloads(tracer: TraceLogger, event: str) -> list[dict[str, Any]]:
    """从当前轮 Trace 中取出指定事件的 payload 列表。"""
    return [
        record["payload"]
        for record in tracer.records
        if record["event"] == event
    ]


def _run_summary(tracer: TraceLogger) -> dict[str, Any]:
    """读取当前轮最后生成的 RUN_SUMMARY。"""
    items = _event_payloads(tracer, "RUN_SUMMARY")
    return items[-1] if items else {}


def _memory_check(
    turn: dict[str, Any],
    tracer: TraceLogger,
    memory: MemoryManager,
) -> tuple[bool, str]:
    """按 Scenario 的 expected_memory_action 校验本轮 Memory 生命周期结果。"""
    expected = turn.get("expected_memory_action")
    if expected is None:
        active = memory.retrieve_active()
        if not active:
            return True, "—"
        read_ok = bool(_event_payloads(tracer, "MEMORY_READ"))
        inject_ok = bool(_event_payloads(tracer, "MEMORY_INJECT"))
        return read_ok and inject_ok, "read+inject"

    action = expected.get("action")
    key = expected.get("key")
    expected_values = expected.get("value") or []
    active = memory.retrieve_active(key=key)

    if action == "write":
        event_ok = bool(_event_payloads(tracer, "MEMORY_WRITE"))
    elif action == "supersede":
        event_ok = bool(_event_payloads(tracer, "MEMORY_SUPERSEDE"))
    else:
        return False, f"unsupported:{action}"

    value_ok = bool(active) and all(
        value in active[0].value for value in expected_values
    )
    return event_ok and value_ok, action


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """把逐轮实验结果写成 JSONL，供后续 Step 9 汇总复用。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    """生成 Step 7 Memory Lifecycle 的简洁人工阅读报告。"""
    passed = sum(1 for row in rows if row["memory_success"])
    lines = [
        "# Step 7 Memory Lifecycle",
        "",
        f"- Turns: `{_START_TURN}..{_END_TURN}`",
        f"- Memory checks: **{passed}/{len(rows)} PASS**",
        "- Restart boundary: `T08 -> T09`，本脚本不跨入 Step 8。",
        "",
        "## Turn Overview",
        "",
        "| Turn | Memory | Active memory | Model calls | Tools | Input | E2E | Trace |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | --- |",
    ]

    for row in rows:
        result = "PASS" if row["memory_success"] else "FAIL"
        active = "<br>".join(row["active_memory"]) or "—"
        lines.append(
            "| {turn} | **{result}** ({action}) | {active} | {model} | {tools} | "
            "{input:,} | {e2e:.3f} s | `{trace}` |".format(
                turn=row["turn_id"],
                result=result,
                action=row["memory_action"],
                active=active,
                model=row["model_calls"],
                tools=row["tool_calls"],
                input=row["input_tokens"],
                e2e=row["e2e_ms"] / 1000,
                trace=row["trace_md"],
            )
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    """真实运行 T04～T08，并归档 Memory Write / Read / Inject / Supersede 结果。"""
    project_root = _project_root()
    load_dotenv(project_root / ".env")

    turns = _load_turns(project_root / "scenarios" / "multi_turn_context.json")
    provider = OpenAICompatibleProvider()
    tools = ToolRegistry(
        fixtures_dir=project_root / "fixtures",
        notes_file=project_root / "artifacts" / "notes.md",
    )
    memory = MemoryManager()
    session = Session()
    context_builder = ContextBuilder(
        "managed",
        managed_recent_n=6,
        summary_trigger_messages=8,
    )

    rows: list[dict[str, Any]] = []

    print("\n===== STEP 7 MEMORY LIFECYCLE =====")
    print(f"session_id: {session.session_id}")
    print("turns: T04..T08")

    for turn in turns:
        turn_id = turn["turn_id"]
        tracer = TraceLogger(
            _trace_path(project_root, session.session_id, turn_id)
        )
        agent = AgentLoop(
            provider=provider,
            tools=tools,
            tracer=tracer,
            context_builder=context_builder,
            memory_manager=memory,
        )

        print(f"\n===== {turn_id} =====")
        print(turn["user_input"])
        answer = agent.run(turn["user_input"], session=session)

        memory_success, memory_action = _memory_check(turn, tracer, memory)
        summary = _run_summary(tracer)
        tokens = summary.get("tokens") or {}
        timing = summary.get("timing_ms") or {}
        active = memory.retrieve_active()

        row = {
            "turn_id": turn_id,
            "user_input": turn["user_input"],
            "answer": answer,
            "memory_success": memory_success,
            "memory_action": memory_action,
            "active_memory": [
                f"{item.kind}:{item.key} = {item.value}"
                for item in active
            ],
            "memory_events": [
                record["event"]
                for record in tracer.records
                if record["event"].startswith("MEMORY_")
            ],
            "model_calls": int(summary.get("model_calls") or 0),
            "tool_calls": int(summary.get("tool_calls") or 0),
            "input_tokens": int(tokens.get("prompt") or 0),
            "output_tokens": int(tokens.get("completion") or 0),
            "e2e_ms": float(timing.get("end_to_end") or 0.0),
            "trace_jsonl": str(tracer.path),
            "trace_md": str(tracer.md_path),
        }
        rows.append(row)

        print(
            f"[{turn_id}] memory={'PASS' if memory_success else 'FAIL'} "
            f"action={memory_action}"
        )
        print("events:", " -> ".join(row["memory_events"]) or "—")
        print("active:", row["active_memory"] or ["—"])

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    jsonl_path = (
        project_root
        / "artifacts"
        / "scenarios"
        / f"step7-memory-lifecycle-{stamp}.jsonl"
    )
    report_path = (
        project_root
        / "artifacts"
        / "reports"
        / f"step7-memory-lifecycle-{stamp}.md"
    )
    _write_jsonl(jsonl_path, rows)
    _write_markdown(report_path, rows)

    passed = sum(1 for row in rows if row["memory_success"])
    print("\n===== STEP 7 RESULT =====")
    print(f"Memory checks: {passed}/{len(rows)} PASS")
    print("JSONL:", jsonl_path)
    print("MD:   ", report_path)
    print("Step 7 stops at T08; T09 restart recall belongs to Step 8.")


if __name__ == "__main__":
    main()
