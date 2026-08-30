"""
文件作用：
离线验证 Step 10.2 的 MCP ToolRegistry 路由边界，不启动真实 MCP Server、模型或网络。

整体结构：
1）FakeBridge 提供一份“由 MCP 发现”的 read_document schema 和固定调用结果；
2）验证模型看到的 read_document schema 确实来自 MCP，而不是旧本地 schema；
3）验证 read_document 经 MCP Bridge 执行；
4）验证 list_documents / save_note 仍走原 Local ToolRegistry。
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from mini_agent_harness.core.mcp_tools import MCPToolRegistry
from mini_agent_harness.core.tools import ToolRegistry


class _FakeBridge:
    """用最小同步接口代替真实 MCP SDK，专门验证 Registry 路由。"""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, object]]] = []

    def tool_schema(self, name: str):
        if name != "read_document":
            raise KeyError(name)
        return {
            "type": "function",
            "function": {
                "name": "read_document",
                "description": "schema discovered from fake MCP server",
                "parameters": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}},
                    "required": ["name"],
                },
            },
        }

    def call_tool_sync(self, name: str, arguments: dict[str, object]) -> str:
        self.calls.append((name, arguments))
        return "body returned through MCP"


class MCPToolRegistryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        fixtures = root / "fixtures"
        fixtures.mkdir()
        (fixtures / "doc.md").write_text("local body", encoding="utf-8")

        self.local = ToolRegistry(
            fixtures_dir=fixtures,
            notes_file=root / "notes.md",
        )
        self.bridge = _FakeBridge()
        self.registry = MCPToolRegistry(
            local_tools=self.local,
            bridge=self.bridge,
        )

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_read_document_schema_is_replaced_by_mcp_discovery(self) -> None:
        read_schema = next(
            item
            for item in self.registry.schemas
            if item["function"]["name"] == "read_document"
        )
        self.assertEqual(
            read_schema["function"]["description"],
            "schema discovered from fake MCP server",
        )

    def test_read_document_executes_through_mcp_bridge(self) -> None:
        result = self.registry.execute("read_document", {"name": "doc.md"})
        self.assertEqual(result, "body returned through MCP")
        self.assertEqual(
            self.bridge.calls,
            [("read_document", {"name": "doc.md"})],
        )

    def test_other_tools_stay_local(self) -> None:
        result = self.registry.execute("list_documents", {})
        self.assertIn("doc.md", result)
        self.assertEqual(self.bridge.calls, [])


if __name__ == "__main__":
    unittest.main()
