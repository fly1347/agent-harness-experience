"""
文件作用：
运行 Step 10.3 的真实 Reviewer Subagent Probe：母 Agent 先读取 01 文档形成一句话草案，再把草案和必要证据委派给隔离 Reviewer，最后吸收审阅意见形成用户答案。

整体结构：
1）母 Agent 仍使用原本 AgentLoop 和本地 read_document；
2）review_answer 作为一个 function tool 暴露给母 Agent，实际执行时触发独立 ReviewerSubagent 模型调用；
3）Reviewer 只收到 system + delegated user 两条消息，没有母 Session、Memory 或工具；
4）验证 delegation / context isolation / result return / Mother Agent finalization 四条链，并生成 Step 10.3 汇总报告。
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from mini_agent_harness.core.agent_loop import AgentLoop
from mini_agent_harness.core.session import Session
from mini_agent_harness.core.subagent import ReviewerSubagent, SubagentToolRegistry
from mini_agent_harness.core.tools import ToolRegistry
from mini_agent_harness.core.tracing import TraceLogger
from mini_agent_harness.providers.openai_compatible import OpenAICompatibleProvider


_PROMPT = (
    "读取 01-检索层与证据层评估.md。先形成一句话草案，然后必须调用 review_answer，"
    "让独立 Reviewer 只根据你提供的草案和必要证据检查是否忠于文档；"
    "收到审阅结果后再给最终答案。最终答案只输出一句话。"
)


def _project_root() -> Path:
    """返回实验项目根目录。"""
    return Path(__file__).resolve().parents[1]


def _trace_path(project_root: Path) -> Path:
    """生成带微秒时间戳的 Step 10.3 Trace 路径，避免覆盖旧实验。"""
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    return project_root / "artifacts" / "traces" / f"step10-subagent-{stamp}.jsonl"


def _find_payload(tracer: TraceLogger, event: str) -> dict[str, Any] | None:
    """返回某事件第一次出现的 payload，供 Probe 做简洁验收。"""
    for record in tracer.records:
        if record["event"] == event:
            return record["payload"]
    return None


def _write_report(*, path: Path, tracer: TraceLogger, answer: str) -> bool:
    """写出 Step 10.3 人工验收报告，重点展示委派边界与 Reviewer 上下文隔离。"""
    events = [record["event"] for record in tracer.records]
    context = _find_payload(tracer, "SUBAGENT_CONTEXT") or {}
    review = _find_payload(tracer, "SUBAGENT_RESULT") or {}

    checks = {
        "mother_read_document_seen": any(
            record["event"] == "TOOL_CALL"
            and record["payload"].get("name") == "read_document"
            for record in tracer.records
        ),
        "delegation_seen": "SUBAGENT_DELEGATE" in events,
        "reviewer_context_isolated": (
            context.get("mother_history_inherited") is False
            and context.get("mother_memory_inherited") is False
            and context.get("message_count") == 2
            and context.get("roles") == ["system", "user"]
        ),
        "reviewer_has_no_tools": context.get("tool_count") == 0,
        "reviewer_model_called": "SUBAGENT_MODEL_RESPONSE" in events,
        "review_returned_to_mother": bool(review.get("result")),
        "mother_final_answer_seen": "FINAL_ANSWER" in events,
    }
    passed = all(checks.values())

    lines = [
        "# Step 10.3 Reviewer Subagent",
        "",
        "- Subagent: `reviewer`",
        "- Delegation interface: `review_answer` function tool",
        "- Mother Agent tools: local `list_documents`, `read_document`, `save_note` + `review_answer`",
        "- Reviewer tools: **none**",
        "- Mother Agent Conversation inherited by Reviewer: **no**",
        "- Mother Agent Memory inherited by Reviewer: **no**",
        f"- Result: **{'PASS' if passed else 'FAIL'}**",
        "",
        "## Path",
        "",
        "```text",
        "User",
        "→ Mother Agent",
        "→ read_document",
        "→ Mother Agent draft",
        "→ review_answer",
        "→ Reviewer Subagent (isolated system + delegated payload only)",
        "→ Review result",
        "→ Mother Agent",
        "→ Final Answer",
        "```",
        "",
        "## Checks",
        "",
    ]
    for name, ok in checks.items():
        lines.append(f"- {'PASS' if ok else 'FAIL'} `{name}`")

    lines.extend([
        "",
        "## Reviewer Context",
        "",
        f"- Messages: **{context.get('message_count', 0)}**",
        f"- Roles: `{ ' → '.join(context.get('roles', [])) }`",
        f"- Mother Agent history inherited: **{context.get('mother_history_inherited')}**",
        f"- Mother Agent memory inherited: **{context.get('mother_memory_inherited')}**",
        f"- Tools: **{context.get('tool_count', 0)}**",
        f"- Draft chars: **{context.get('draft_chars', 0)}**",
        f"- Evidence chars: **{context.get('evidence_chars', 0)}**",
        "",
        "## Reviewer Result",
        "",
        str(review.get("result") or ""),
        "",
        "## Mother Agent Final Answer",
        "",
        answer,
        "",
        "## Event Chain",
        "",
        f"`{' -> '.join(events)}`",
        "",
        f"- JSONL: `{tracer.path}`",
        f"- Markdown Trace: `{tracer.md_path}`",
    ])

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return passed


def main() -> None:
    """运行一次真实母 Agent → Reviewer → 母 Agent 链，并输出最终验收结果。"""
    project_root = _project_root()
    load_dotenv(project_root / ".env")

    tracer = TraceLogger(_trace_path(project_root))
    provider = OpenAICompatibleProvider()
    local_tools = ToolRegistry(
        fixtures_dir=project_root / "fixtures",
        notes_file=project_root / "artifacts" / "notes.md",
    )
    reviewer = ReviewerSubagent(provider=provider, tracer=tracer)
    tools = SubagentToolRegistry(
        local_tools=local_tools,
        reviewer=reviewer,
        tracer=tracer,
    )
    agent = AgentLoop(
        provider=provider,
        tools=tools,
        tracer=tracer,
    )

    print("\n===== STEP 10.3 REVIEWER SUBAGENT =====")
    answer = agent.run(_PROMPT, session=Session())

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    report_path = project_root / "artifacts" / "reports" / f"step10-subagent-{stamp}.md"
    passed = _write_report(path=report_path, tracer=tracer, answer=answer)

    print("\n===== STEP 10.3 RESULT =====")
    print(f"{'PASS' if passed else 'FAIL'}  Mother Agent → Reviewer → Mother Agent")
    print("Report:", report_path)

    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
