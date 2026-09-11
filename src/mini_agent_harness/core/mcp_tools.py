"""
文件作用：
实现 Step 10.2 的 MCP Client / ToolRegistry 适配层，让现有同步 AgentLoop 无需重写成 async，也能通过一个长期存活的 stdio MCP 会话调用 read_document。

整体结构：
1）MCPStdioBridge：在 async 主线程中启动 MCP Server 子进程、完成协议握手并发现工具；
2）call_tool_sync：AgentLoop 放到 worker thread 后，通过 run_coroutine_threadsafe 把同步工具调用送回 MCP Client 所在事件循环；
3）MCPToolRegistry：保留原 list_documents / save_note 本地执行，只把 read_document 路由到 MCP；
4）Trace：显式记录 MCP_CONNECT / MCP_LIST_TOOLS / MCP_CALL_TOOL / MCP_RESULT，便于与原本 Local Tool 路径比较。

职责边界：
- MCP 负责“工具如何标准化暴露、发现与调用”；
- AgentLoop 仍负责“模型何时决定调用工具以及拿到结果后如何继续”；
- 本文件不实现 Retry、Memory、Context Management，也不改变原 ToolRegistry 的业务语义。
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path
from typing import Any

from mini_agent_harness.core.tools import ToolRegistry
from mini_agent_harness.core.tracing import TraceLogger


class MCPStdioBridge:
    """维护一个 stdio MCP Client 会话，并把 async MCP 调用桥接给同步 AgentLoop。"""

    # 保存服务与文档路径，初始化尚未连接的 MCP 会话状态。
    def __init__(
        self,
        *,
        server_script: Path,
        fixtures_dir: Path,
        tracer: TraceLogger | None = None,
    ) -> None:
        self.server_script = server_script.resolve()
        self.fixtures_dir = fixtures_dir.resolve()
        self.tracer = tracer
        self._client_cm: Any | None = None
        self._client: Any | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._schemas: dict[str, dict[str, Any]] = {}
        self.server_info: dict[str, Any] = {}
        self.protocol_version: str | None = None

    # 启动 stdio Server 子进程、完成 MCP 握手，并缓存 Server 暴露的工具 schema。
    async def __aenter__(self) -> "MCPStdioBridge":
        # MCP SDK 只在真正运行 Step 10.2 时导入；旧实验和离线单测不因此强依赖 MCP。
        from mcp import Client, StdioServerParameters

        if not self.server_script.is_file():
            raise FileNotFoundError(
                f"MCP server script not found: {self.server_script}"
            )

        self._loop = asyncio.get_running_loop()
        params = StdioServerParameters(
            command=sys.executable,
            args=[str(self.server_script)],
            env={
                "MINI_AGENT_FIXTURES_DIR": str(self.fixtures_dir),
            },
        )
        self._client_cm = Client(params)
        self._client = await self._client_cm.__aenter__()

        info = getattr(self._client, "server_info", None)
        self.server_info = {
            "name": getattr(info, "name", None),
            "version": getattr(info, "version", None),
        }
        self.protocol_version = getattr(
            self._client,
            "protocol_version",
            None,
        )

        self._log(
            "MCP_CONNECT",
            {
                "transport": "stdio",
                "command": sys.executable,
                "server_script": str(self.server_script),
                "server_info": self.server_info,
                "protocol_version": self.protocol_version,
            },
            console={
                "transport": "stdio",
                "server": self.server_info.get("name"),
                "protocol_version": self.protocol_version,
            },
        )

        listed = await self._client.list_tools()
        self._schemas = {
            tool.name: self._to_openai_schema(tool)
            for tool in listed.tools
        }
        self._log(
            "MCP_LIST_TOOLS",
            {
                "count": len(self._schemas),
                "tools": list(self._schemas.values()),
            },
            console={
                "count": len(self._schemas),
                "names": sorted(self._schemas),
            },
        )
        return self

    # 关闭 MCP Client 会话，并让 SDK 按 stdio 生命周期回收 Server 子进程。
    async def __aexit__(self, exc_type, exc, tb) -> None:
        if self._client_cm is not None:
            await self._client_cm.__aexit__(exc_type, exc, tb)
        self._client = None
        self._client_cm = None
        self._loop = None

    # 返回 MCP Server 在握手后动态发现到的某个工具 schema。
    def tool_schema(self, name: str) -> dict[str, Any]:
        if name not in self._schemas:
            raise KeyError(f"MCP tool not discovered: {name}")
        return self._schemas[name]

    # 从 AgentLoop worker thread 同步等待一次 MCP call_tool 结果。
    def call_tool_sync(
        self,
        name: str,
        arguments: dict[str, Any],
    ) -> str:
        if self._client is None or self._loop is None:
            raise RuntimeError("MCP bridge is not connected.")

        self._log(
            "MCP_CALL_TOOL",
            {
                "transport": "stdio",
                "name": name,
                "arguments": arguments,
            },
        )

        future = asyncio.run_coroutine_threadsafe(
            self._client.call_tool(name, arguments),
            self._loop,
        )
        result = future.result()
        text = self._result_text(result)
        is_error = bool(getattr(result, "is_error", False))

        self._log(
            "MCP_RESULT",
            {
                "transport": "stdio",
                "name": name,
                "is_error": is_error,
                "result_chars": len(text),
                "result": text,
            },
            console={
                "name": name,
                "is_error": is_error,
                "result_chars": len(text),
            },
        )

        if is_error:
            raise RuntimeError(f"MCP tool {name!r} failed: {text}")
        return text

    # 把 MCP list_tools 返回的工具定义转换为现有模型 Provider 使用的 function schema。
    @staticmethod
    def _to_openai_schema(tool: Any) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description or "",
                "parameters": tool.input_schema,
            },
        }

    # 提取 MCP CallToolResult 中给模型阅读的文本 content。
    @staticmethod
    def _result_text(result: Any) -> str:
        chunks: list[str] = []
        for block in getattr(result, "content", []) or []:
            text = getattr(block, "text", None)
            if text is not None:
                chunks.append(str(text))
        return "\n".join(chunks)

    # 有 TraceLogger 时记录 MCP 边界事件；无 tracer 时保持适配层可独立测试。
    def _log(
        self,
        event: str,
        payload: dict[str, Any],
        console: dict[str, Any] | None = None,
    ) -> None:
        if self.tracer is not None:
            self.tracer.log(event, payload, console=console)


class MCPToolRegistry:
    """混合工具注册表：read_document 走 MCP，其她工具继续走原本本地实现。"""

    # 组合本地工具与 MCP 桥接器，建立供模型使用的工具列表。
    def __init__(
        self,
        *,
        local_tools: ToolRegistry,
        bridge: MCPStdioBridge,
    ) -> None:
        self.local_tools = local_tools
        self.bridge = bridge
        self._schemas = self._build_schemas()

    # 保留原工具顺序，但用 MCP 动态发现的 read_document schema 替换本地定义。
    @property
    def schemas(self) -> list[dict[str, Any]]:
        return self._schemas

    # 只把 read_document 路由到 MCP；其他工具仍交给 Local ToolRegistry。
    def execute(self, name: str, arguments: dict[str, Any]) -> str:
        if name == "read_document":
            return self.bridge.call_tool_sync(name, arguments)
        return self.local_tools.execute(name, arguments)

    # 把 MCP 发现到的 read_document schema 嵌回现有三工具列表。
    def _build_schemas(self) -> list[dict[str, Any]]:
        mcp_read = self.bridge.tool_schema("read_document")
        schemas: list[dict[str, Any]] = []

        for schema in self.local_tools.schemas:
            name = schema["function"]["name"]
            schemas.append(mcp_read if name == "read_document" else schema)

        return schemas
