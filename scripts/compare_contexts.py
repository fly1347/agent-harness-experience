"""
文件作用：
执行 Step 6 上下文策略对比。让同一固定 Scenario 分别运行 Full History、Last-N、Managed Context，比较上下文消息数、token、工具调用和耗时，并归档带时间戳的 JSONL / Markdown 结果。

整体结构：
1）Scenario 辅助函数：加载并截取 T01～目标轮次，限制 Step 6 不越界进入 Memory Lifecycle；
2）Trace 提取与校验：读取工具调用、RUN_SUMMARY、CONTEXT_BUILD，整理逐轮正确性和上下文统计；
3）策略构建：为 Full History、Last-N、Managed Context 创建对应 ContextBuilder；
4）_write_report：生成三策略总览、逐轮差异和 Managed Summary 快照；
5）main：依次回放三种策略，并用同一 run_stamp 输出不覆盖旧结果的 JSONL / Markdown 报告。
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from mini_agent_harness.core.agent_loop import AgentLoop
from mini_agent_harness.core.context import ContextBuilder
from mini_agent_harness.core.session import Session
from mini_agent_harness.core.tools import ToolRegistry
from mini_agent_harness.core.tracing import TraceLogger
from mini_agent_harness.providers.openai_compatible import (
    OpenAICompatibleProvider,
)


STRATEGIES = (
    ("full_history", "Full History"),
    ("last_n", "Last-N"),
    ("managed", "Managed Context"),
)


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_scenario(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _select_turns(
    turns: list[dict[str, Any]],
    through: str,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for turn in turns:
        selected.append(turn)
        if turn.get("turn_id") == through:
            return selected
    raise ValueError(f"Unknown --through turn: {through}")


def _guard_step6_scope(turns: list[dict[str, Any]]) -> None:
    blocked = [
        turn["turn_id"]
        for turn in turns
        if turn.get("expected_memory_action") is not None
    ]
    if blocked:
        raise RuntimeError(
            "Step 6 compares context strategies before Memory Lifecycle. "
            "Selected turns require memory: " + ", ".join(blocked)
        )


def _contains_check(answer: str, expected: list[str]) -> tuple[bool, list[str]]:
    folded = answer.casefold()
    missing = [x for x in expected if x.casefold() not in folded]
    return not missing, missing


def _tool_calls(tracer: TraceLogger) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    for record in tracer.records:
        if record["event"] != "TOOL_CALL":
            continue
        payload = record["payload"]
        calls.append(
            {
                "name": payload.get("name"),
                "arguments": payload.get("arguments") or {},
            }
        )
    return calls


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
        if document is None:
            return True
        if call.get("arguments", {}).get("name") == document:
            return True
    return False


def _run_summary(tracer: TraceLogger) -> dict[str, Any]:
    for record in reversed(tracer.records):
        if record["event"] == "RUN_SUMMARY":
            return record["payload"]
    return {}


def _context_stats(tracer: TraceLogger) -> dict[str, Any]:
    """从 CONTEXT_BUILD 事件提取首轮/最大消息数和 Managed Summary 状态。"""
    builds = [
        r["payload"]
        for r in tracer.records
        if r["event"] == "CONTEXT_BUILD"
    ]
    if not builds:
        return {
            "context_first": 0,
            "context_max": 0,
            "history_first": 0,
            "summary_injected": False,
            "summary": "",
        }

    summaries = [x for x in builds if x.get("summary_injected")]
    latest_summary = summaries[-1].get("summary", "") if summaries else ""
    return {
        "context_first": int(builds[0].get("request_message_count") or 0),
        "context_max": max(int(x.get("request_message_count") or 0) for x in builds),
        "history_first": int(builds[0].get("history_message_count") or 0),
        "summary_injected": bool(summaries),
        "summary": latest_summary,
    }


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
        / f"step6-{strategy}-{session_id[:8]}-{turn_id.lower()}-{stamp}.jsonl"
    )


def _builder(strategy: str, last_n: int, summary_trigger: int) -> ContextBuilder:
    """把策略名称和实验参数转换成对应的 ContextBuilder 配置。"""
    if strategy == "full_history":
        return ContextBuilder("full_history")
    if strategy == "last_n":
        return ContextBuilder("last_n", last_n=last_n)
    return ContextBuilder(
        "managed",
        managed_recent_n=last_n,
        summary_trigger_messages=summary_trigger,
    )


def _write_report(
    path: Path,
    *,
    scenario_id: str,
    through: str,
    model: str,
    last_n: int,
    summary_trigger: int,
    records: list[dict[str, Any]],
) -> None:
    """汇总三种策略的逐轮记录，生成上下文、token、工具调用和耗时对比报告。"""
    strategy_summary: list[dict[str, Any]] = []
    for strategy, label in STRATEGIES:
        rows = [r for r in records if r["strategy"] == strategy]
        strategy_summary.append(
            {
                "strategy": strategy,
                "label": label,
                "passed": sum(1 for r in rows if r["correct"]),
                "turns": len(rows),
                "input": sum(r["input_tokens"] for r in rows),
                "output": sum(r["output_tokens"] for r in rows),
                "latency": sum(r["end_to_end_ms"] for r in rows),
            }
        )

    lines = [
        "# Context Strategy Comparison",
        "",
        f"- Scenario: `{scenario_id}`",
        f"- Turns: `T01..{through}`",
        f"- Model: `{model}`",
        "- Temperature: `0` (from the frozen experiment config)",
        f"- Last-N window: **{last_n} messages**",
        f"- Managed summary trigger: **>{summary_trigger} history messages**",
        "- Managed summary: **本地确定性压缩，不额外调用 LLM**",
        "- Active memory: **Step 6 尚未实现，本轮注入 0 条**",
        "",
        "## Strategy Overview",
        "",
        "| Strategy | Pass | Input tokens | Output tokens | E2E total |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]

    for row in strategy_summary:
        lines.append(
            f"| {row['label']} | **{row['passed']}/{row['turns']}** | "
            f"{row['input']:,} | {row['output']:,} | {row['latency'] / 1000:.3f} s |"
        )

    lines.extend(
        [
            "",
            "## Turn Comparison",
            "",
            "| Strategy | Turn | Result | History msgs | Context msgs first/max | Input | Tools | E2E | Summary |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )

    for record in records:
        lines.append(
            f"| {record['label']} | {record['turn_id']} | "
            f"**{'PASS' if record['correct'] else 'FAIL'}** | "
            f"{record['history_first']} | "
            f"{record['context_first']}/{record['context_max']} | "
            f"{record['input_tokens']:,} | {record['tool_calls']} | "
            f"{record['end_to_end_ms'] / 1000:.3f} s | "
            f"{'yes' if record['summary_injected'] else '—'} |"
        )

    managed_summaries = [
        r for r in records
        if r["strategy"] == "managed" and r["summary_injected"]
    ]
    if managed_summaries:
        lines.extend(["", "## Managed Context Summary Snapshots", ""])
        seen: set[str] = set()
        for record in managed_summaries:
            summary = record["summary"].strip()
            if not summary or summary in seen:
                continue
            seen.add(summary)
            lines.extend(
                [
                    f"### {record['turn_id']}",
                    "",
                    "```text",
                    summary,
                    "```",
                    "",
                ]
            )

    failures = [r for r in records if not r["correct"]]
    lines.extend(["", "## Observed Differences", ""])
    totals = {x["strategy"]: x for x in strategy_summary}
    full_input = totals["full_history"]["input"]
    for strategy in ("last_n", "managed"):
        current = totals[strategy]["input"]
        delta = current - full_input
        pct = (delta / full_input * 100) if full_input else 0.0
        lines.append(
            f"- `{strategy}` 相比 `full_history`：Input {delta:+,} tokens（{pct:+.1f}%）。"
        )
    if failures:
        lines.append(
            "- 正确性或工具预期出现差异的轮次："
            + ", ".join(f"{r['label']} {r['turn_id']}" for r in failures)
            + "。"
        )
    else:
        lines.append(
            "- 所选轮次全部通过；本次 smoke 主要观察 Context 大小、token 使用和请求组成差异。"
        )

    lines.extend(
        [
            "",
            "## Scope Boundary",
            "",
            "Step 6 在 T03 结束。固定 Scenario 从 T04 开始进入长期 Memory Lifecycle，因此本阶段不提前引入 Memory。Step 7 再加入主动记忆，并避免把 01/02 的普通会话事实直接写入长期记忆。",
            "",
        ]
    )

    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    """用同一 Scenario 依次运行三种上下文策略，并归档同一时间戳的一组对比结果。"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scenario",
        default="scenarios/multi_turn_context.json",
    )
    parser.add_argument("--through", default="T03")
    parser.add_argument("--last-n", type=int, default=6)
    parser.add_argument("--summary-trigger", type=int, default=8)
    args = parser.parse_args()

    project_root = _project_root()
    run_stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    scenario_path = project_root / args.scenario
    scenario = _load_scenario(scenario_path)
    turns = _select_turns(scenario["turns"], args.through)
    _guard_step6_scope(turns)

    load_dotenv(project_root / ".env")
    provider = OpenAICompatibleProvider()
    tools = ToolRegistry(
        fixtures_dir=project_root / "fixtures",
        notes_file=project_root / "artifacts" / "notes.md",
    )

    scenario_id = scenario.get("scenario_id", scenario_path.stem)
    records: list[dict[str, Any]] = []

    print("\n===== CONTEXT STRATEGY COMPARISON =====")
    print(f"scenario: {scenario_id}")
    print(f"turns: T01..{args.through}")
    print(f"last_n: {args.last_n}")
    print(f"summary_trigger: {args.summary_trigger}")

    for strategy, label in STRATEGIES:
        session = Session()
        builder = _builder(strategy, args.last_n, args.summary_trigger)

        print(f"\n===== {label.upper()} =====")
        for turn in turns:
            turn_id = turn["turn_id"]
            print(f"\n--- {turn_id} ---")

            tracer = TraceLogger(
                _trace_path(
                    project_root,
                    strategy=strategy,
                    session_id=session.session_id,
                    turn_id=turn_id,
                )
            )
            agent = AgentLoop(
                provider=provider,
                tools=tools,
                tracer=tracer,
                context_builder=builder,
            )
            answer = agent.run(turn["user_input"], session=session)

            calls = _tool_calls(tracer)
            expected_contains = turn.get("expected_contains") or []
            contains_ok, missing = _contains_check(answer, expected_contains)
            tool_ok = _tool_check(calls, turn.get("expected_tool"))
            correct = contains_ok and tool_ok is not False

            summary = _run_summary(tracer)
            tokens = summary.get("tokens") or {}
            timing = summary.get("timing_ms") or {}
            ctx = _context_stats(tracer)

            failure_parts: list[str] = []
            if missing:
                failure_parts.append("missing_contains=" + ",".join(missing))
            if tool_ok is False:
                failure_parts.append("expected_tool_not_observed")

            record = {
                "scenario_id": scenario_id,
                "strategy": strategy,
                "label": label,
                "turn_id": turn_id,
                "correct": correct,
                "failure_reason": "; ".join(failure_parts),
                "missing_contains": missing,
                "expected_tool_match": tool_ok,
                "model_calls": int(summary.get("model_calls") or 0),
                "tool_calls": int(summary.get("tool_calls") or 0),
                "input_tokens": int(tokens.get("prompt") or 0),
                "output_tokens": int(tokens.get("completion") or 0),
                "latency_ms": float(timing.get("end_to_end") or 0.0),
                "end_to_end_ms": float(timing.get("end_to_end") or 0.0),
                **ctx,
                "trace_md": str(tracer.md_path),
            }
            records.append(record)

            print(
                f"{turn_id}: {'PASS' if correct else 'FAIL'} | "
                f"history={ctx['history_first']} | "
                f"context={ctx['context_first']}/{ctx['context_max']} | "
                f"input={record['input_tokens']} | "
                f"summary={'yes' if ctx['summary_injected'] else 'no'}"
            )

    jsonl_path = (
        project_root
        / "artifacts"
        / "scenarios"
        / f"context_strategy_comparison-{run_stamp}.jsonl"
    )
    report_path = (
        project_root
        / "artifacts"
        / "reports"
        / f"context_strategy_comparison-{run_stamp}.md"
    )
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    with jsonl_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    _write_report(
        report_path,
        scenario_id=scenario_id,
        through=args.through,
        model=provider.model,
        last_n=args.last_n,
        summary_trigger=args.summary_trigger,
        records=records,
    )

    print("\n===== COMPARISON OUTPUT =====")
    print(f"JSONL: {jsonl_path}")
    print(f"MD:    {report_path}")


if __name__ == "__main__":
    main()
