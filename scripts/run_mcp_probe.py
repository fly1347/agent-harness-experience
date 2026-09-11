"""
文件作用：
运行 Step 10.2 的真实 MCP Probe：保持与此前相同的“读 01 文档并回答一句话”任务，只把 read_document 从本地函数调用替换为 stdio MCP 调用。

整体结构：
1）启动一个独立 MCP Server 子进程并完成握手 / list_tools；
2）用 MCP Server 动态返回的 read_document schema 构造混合 ToolRegistry；
3）AgentLoop 放到 worker thread 运行，使同步 execute 能安全调用主事件循环中的 MCP Client；
4）验证 Trace 中确实出现 MCP_CONNECT / LIST_TOOLS / CALL_TOOL / RESULT，并生成 Step 10.2 汇总报告。
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

from mini_agent_harness.core.agent_loop import AgentLoop
from mini_agent_harness.core.mcp_tools import MCPStdioBridge, MCPToolRegistry
from mini_agent_harness.core.session import Session
from mini_agent_harness.core.tools import ToolRegistry
from mini_agent_harness.core.tracing import TraceLogger
from mini_agent_harness.providers.openai_compatible import OpenAICompatibleProvider


_PROMPT = (
    "读取 01-检索层与证据层评估.md，"
    "并用一句话说明检索层和证据层最核心的区别。"
)


# 返回实验项目根目录。
def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


# 生成带微秒时间戳的 Step 10.2 JSONL Trace 路径，避免覆盖旧实验。
def _trace_path(project_root: Path) -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    return (
        project_root
        / "artifacts"
        / "traces"
        / f"step10-mcp-{stamp}.jsonl"
    )


# 写出 MCP Probe 的人工验收摘要，突出本地 Tool 与 MCP Tool 的路径变化。
def _write_report(
    *,
    path: Path,
    tracer: TraceLogger,
    answer: str,
    protocol_version: str | None,
) -> None:
    events = [record["event"] for record in tracer.records]
    required = {
        "mcp_connected": "MCP_CONNECT" in events,
        "tool_discovered_via_mcp": "MCP_LIST_TOOLS" in events,
        "read_called_via_mcp": "MCP_CALL_TOOL" in events,
        "mcp_result_returned": "MCP_RESULT" in events,
        "agent_received_tool_result": "TOOL_RESULT" in events,
        "final_answer_seen": "FINAL_ANSWER" in events,
    }
    passed = all(required.values())

    lines = [
        "# Step 10.2 MCP",
        "",
        "- Transport: `stdio`",
        "- Server: independent Python subprocess",
        "- Migrated tool: `read_document` only",
        "- Local tools retained: `list_documents`, `save_note`",
        f"- MCP protocol version: `{protocol_version or 'unknown'}`",
        f"- Result: **{'PASS' if passed else 'FAIL'}**",
        "",
        "## Path",
        "",
        "```text",
        "Model",
        "→ AgentLoop",
        "→ MCPToolRegistry",
        "→ MCP Client",
        "→ stdio",
        "→ MCP Server subprocess",
        "→ read_document",
        "→ MCP result",
        "→ AgentLoop",
        "→ Model",
        "```",
        "",
        "## Checks",
        "",
    ]
    for name, ok in required.items():
        lines.append(f"- {'PASS' if ok else 'FAIL'} `{name}`")

    lines.extend([
        "",
        "## Event Chain",
        "",
        f"`{' -> '.join(events)}`",
        "",
        "## Final Answer",
        "",
        answer,
        "",
        f"- JSONL: `{tracer.path}`",
        f"- Markdown Trace: `{tracer.md_path}`",
        "",
        "> 本 Probe 为了不重写现有同步 AgentLoop，让 AgentLoop 在 worker thread 中运行；",
        "> MCP Client 与 stdio Server 会话本身在主 async event loop 中保持连接。",
    ])

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


# 保持一个 MCP stdio 会话贯穿工具发现和实际调用，并执行一次真实 Agent Turn。
async def _run() -> tuple[Path, bool]:
    project_root = _project_root()
    load_dotenv(project_root / ".env")

    tracer = TraceLogger(_trace_path(project_root))
    provider = OpenAICompatibleProvider()
    local_tools = ToolRegistry(
        fixtures_dir=project_root / "fixtures",
        notes_file=project_root / "artifacts" / "notes.md",
    )

    server_script = (
        project_root
        / "src"
        / "mini_agent_harness"
        / "mcp"
        / "read_document_server.py"
    )

    async with MCPStdioBridge(
        server_script=server_script,
        fixtures_dir=project_root / "fixtures",
        tracer=tracer,
    ) as bridge:
        tools = MCPToolRegistry(
            local_tools=local_tools,
            bridge=bridge,
        )
        agent = AgentLoop(
            provider=provider,
            tools=tools,
            tracer=tracer,
        )

        # AgentLoop 仍是同步教学实现；放到 worker thread 后，read_document
        # 可安全地通过 run_coroutine_threadsafe 回到主事件循环中的 MCP Client。
        answer = await asyncio.to_thread(
            agent.run,
            _PROMPT,
            Session(),
        )
        protocol_version = bridge.protocol_version

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    report_path = (
        project_root
        / "artifacts"
        / "reports"
        / f"step10-mcp-{stamp}.md"
    )
    _write_report(
        path=report_path,
        tracer=tracer,
        answer=answer,
        protocol_version=protocol_version,
    )

    events = [record["event"] for record in tracer.records]
    passed = all(
        event in events
        for event in (
            "MCP_CONNECT",
            "MCP_LIST_TOOLS",
            "MCP_CALL_TOOL",
            "MCP_RESULT",
            "TOOL_RESULT",
            "FINAL_ANSWER",
        )
    )
    return report_path, passed


# 运行真实 MCP Probe，并在终端只输出最终验收结论与报告位置。
def main() -> None:
    print("\n===== STEP 10.2 MCP =====")
    report_path, passed = asyncio.run(_run())

    print("\n===== STEP 10.2 RESULT =====")
    print(f"{'PASS' if passed else 'FAIL'}  stdio MCP read_document")
    print("Report:", report_path)

    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
