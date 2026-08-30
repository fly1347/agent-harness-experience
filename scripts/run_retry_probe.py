"""
文件作用：
运行 Step 10.1 的最小 Tool Retry 实验，真实制造一次 read_document 瞬时失败，并验证 max_retry=1 时 Harness 能自行重试；随后再制造持续失败，验证重试预算耗尽后当前 AgentLoop 终止。

整体结构：
1）FailOnceToolRegistry：仅第一次 read_document 抛出人工瞬时错误，第二次执行原工具；
2）AlwaysFailToolRegistry：每次 read_document 都抛错，用于验证 exhausted 分支；
3）两个 case 都使用真实 Provider，但不引入新的 Scenario、Memory、SQLite 或 Context 机制；
4）成功 case 验证 TOOL_ERROR -> TOOL_RETRY -> TOOL_RESULT -> Final Answer；
5）持续失败 case 验证两次 TOOL_ERROR + 一次 TOOL_RETRY -> TOOL_RETRY_EXHAUSTED -> AgentLoop 终止；
6）输出带时间戳的 Step 10.1 汇总报告，原始事件保存在各自 JSONL Trace 中。
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from mini_agent_harness.core.agent_loop import AgentLoop
from mini_agent_harness.core.session import Session
from mini_agent_harness.core.tools import ToolRegistry
from mini_agent_harness.core.tracing import TraceLogger
from mini_agent_harness.providers.openai_compatible import OpenAICompatibleProvider


_PROMPT = (
    "读取 01-检索层与证据层评估.md，"
    "并用一句话说明检索层和证据层最核心的区别。"
)


class _FailOnceToolRegistry(ToolRegistry):
    """只让第一次 read_document 失败一次，其她工具保持原行为。"""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._read_failures_remaining = 1

    def execute(self, name: str, arguments: dict[str, Any]) -> str:
        """在 read_document 的第一次实际执行点注入瞬时错误。"""
        if name == "read_document" and self._read_failures_remaining:
            self._read_failures_remaining -= 1
            raise OSError("Step 10.1 injected transient read failure")
        return super().execute(name, arguments)


class _AlwaysFailToolRegistry(ToolRegistry):
    """让 read_document 持续失败，用于验证 retry exhausted 后终止。"""

    def execute(self, name: str, arguments: dict[str, Any]) -> str:
        """除 read_document 外不改动原工具行为。"""
        if name == "read_document":
            raise OSError("Step 10.1 injected persistent read failure")
        return super().execute(name, arguments)


def _project_root() -> Path:
    """返回当前实验项目根目录。"""
    return Path(__file__).resolve().parents[1]


def _trace_path(project_root: Path, case: str) -> Path:
    """生成 Step 10.1 单 case 的带时间戳 Trace 路径。"""
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    return (
        project_root
        / "artifacts"
        / "traces"
        / f"step10-retry-{case}-{stamp}.jsonl"
    )


def _events(tracer: TraceLogger) -> list[str]:
    """按发生顺序提取事件名，便于校验 Harness retry 链。"""
    return [record["event"] for record in tracer.records]


def _run_success_case(
    project_root: Path,
    provider: OpenAICompatibleProvider,
) -> dict[str, Any]:
    """第一次 read 失败、第二次成功；验证 Agent Loop 能继续到 Final Answer。"""
    tracer = TraceLogger(_trace_path(project_root, "success"))
    tools = _FailOnceToolRegistry(
        fixtures_dir=project_root / "fixtures",
        notes_file=project_root / "artifacts" / "notes.md",
    )
    agent = AgentLoop(
        provider=provider,
        tools=tools,
        tracer=tracer,
        max_retry=1,
    )

    answer = agent.run(_PROMPT, session=Session())
    events = _events(tracer)
    retry_results = [
        record["payload"]
        for record in tracer.records
        if record["event"] == "TOOL_RESULT"
        and record["payload"].get("name") == "read_document"
    ]

    checks = {
        "tool_error_seen": "TOOL_ERROR" in events,
        "one_retry_seen": events.count("TOOL_RETRY") == 1,
        "retry_succeeded": bool(retry_results)
        and retry_results[-1].get("retry_count") == 1,
        "final_answer_seen": "FINAL_ANSWER" in events,
        "not_exhausted": "TOOL_RETRY_EXHAUSTED" not in events,
    }
    return {
        "name": "transient failure -> retry -> continue",
        "passed": all(checks.values()),
        "checks": checks,
        "events": events,
        "answer": answer,
        "trace_jsonl": tracer.path,
        "trace_md": tracer.md_path,
    }


def _run_exhausted_case(
    project_root: Path,
    provider: OpenAICompatibleProvider,
) -> dict[str, Any]:
    """read 始终失败；验证 max_retry=1 只允许第二次尝试，然后终止当前 Turn。"""
    tracer = TraceLogger(_trace_path(project_root, "exhausted"))
    tools = _AlwaysFailToolRegistry(
        fixtures_dir=project_root / "fixtures",
        notes_file=project_root / "artifacts" / "notes.md",
    )
    agent = AgentLoop(
        provider=provider,
        tools=tools,
        tracer=tracer,
        max_retry=1,
    )

    error = ""
    try:
        agent.run(_PROMPT, session=Session())
    except RuntimeError as exc:
        error = str(exc)

    events = _events(tracer)
    read_errors = [
        record
        for record in tracer.records
        if record["event"] == "TOOL_ERROR"
        and record["payload"].get("name") == "read_document"
    ]
    checks = {
        "two_attempts_failed": len(read_errors) == 2,
        "one_retry_seen": events.count("TOOL_RETRY") == 1,
        "retry_exhausted_seen": "TOOL_RETRY_EXHAUSTED" in events,
        "agent_loop_terminated": bool(error),
        "no_final_answer": "FINAL_ANSWER" not in events,
    }
    return {
        "name": "persistent failure -> retry exhausted -> terminate",
        "passed": all(checks.values()),
        "checks": checks,
        "events": events,
        "error": error,
        "trace_jsonl": tracer.path,
        # 终止分支没有 RUN_SUMMARY，因此不会生成单轮 Markdown；原始证据保留在 JSONL。
        "trace_md": None,
    }


def _write_report(path: Path, cases: list[dict[str, Any]]) -> None:
    """写出 Step 10.1 的人工验收摘要，不复制大段模型请求。"""
    passed = sum(1 for case in cases if case["passed"])
    lines = [
        "# Step 10.1 Tool Retry",
        "",
        "- Retry scope: local Tool Execution only",
        "- max_retry: `1`",
        "- Injected tool: `read_document`",
        f"- Cases: **{passed}/{len(cases)} PASS**",
        "",
    ]

    for index, case in enumerate(cases, start=1):
        lines.extend([
            f"## Case {index} — {case['name']}",
            "",
            f"- Result: **{'PASS' if case['passed'] else 'FAIL'}**",
            f"- Event chain: `{' -> '.join(case['events'])}`",
        ])
        for name, ok in case["checks"].items():
            lines.append(f"- {'PASS' if ok else 'FAIL'} `{name}`")
        if case.get("answer"):
            lines.extend(["", "### Final Answer", "", case["answer"]])
        if case.get("error"):
            lines.extend(["", "### Terminal Error", "", f"`{case['error']}`"])
        lines.extend([
            "",
            f"- JSONL: `{case['trace_jsonl']}`",
        ])
        if case.get("trace_md"):
            lines.append(f"- Markdown Trace: `{case['trace_md']}`")
        lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    """依次运行成功重试与预算耗尽两个真实 case，并输出总验收结果。"""
    project_root = _project_root()
    load_dotenv(project_root / ".env")
    provider = OpenAICompatibleProvider()

    print("\n===== STEP 10.1 TOOL RETRY =====")
    print("max_retry: 1")

    success = _run_success_case(project_root, provider)
    exhausted = _run_exhausted_case(project_root, provider)
    cases = [success, exhausted]

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    report_path = (
        project_root
        / "artifacts"
        / "reports"
        / f"step10-retry-{stamp}.md"
    )
    _write_report(report_path, cases)

    passed = sum(1 for case in cases if case["passed"])
    print("\n===== STEP 10.1 RESULT =====")
    for case in cases:
        print(f"{'PASS' if case['passed'] else 'FAIL'}  {case['name']}")
    print(f"Checks: {passed}/{len(cases)} cases PASS")
    print("Report:", report_path)

    if passed != len(cases):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
