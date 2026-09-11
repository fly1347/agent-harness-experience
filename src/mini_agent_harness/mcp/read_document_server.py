"""
文件作用：
提供 Step 10.2 的独立 MCP stdio Server，只暴露一个 read_document 工具；Harness 不再直接读取文件，而是通过 MCP Client 调用本进程完成读取。

整体结构：
1）从环境变量 MINI_AGENT_FIXTURES_DIR 读取 fixture 根目录；
2）MCPServer 用函数签名与 docstring 自动生成 read_document 的工具描述/schema；
3）read_document 复用原实验的路径安全约束，只允许读取 fixtures 下的纯文件名；
4）main 以 stdio transport 启动 Server，stdin/stdout 专门承载 MCP 协议。

实验边界：
- 这里只迁移 read_document；list_documents / save_note 仍保留本地 ToolRegistry；
- 不启动 HTTP 服务、不做认证、不接外部 MCP Server；
- stdout 是 MCP wire，禁止在 Server 启动前后随意 print。
"""

from __future__ import annotations

import os
from pathlib import Path

from mcp.server import MCPServer


_FIXTURES_ENV = "MINI_AGENT_FIXTURES_DIR"
mcp = MCPServer("mini-agent-harness-document-server")


# 读取并校验 MCP Server 的 fixture 根目录配置。
def _fixtures_dir() -> Path:
    raw = os.environ.get(_FIXTURES_ENV, "").strip()
    if not raw:
        raise RuntimeError(f"Missing environment variable: {_FIXTURES_ENV}")

    path = Path(raw).expanduser().resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"Fixtures directory not found: {path}")
    return path


# 按文件名读取受限目录内的文档；下方英文 docstring 是 MCP 工具描述，需保留。
@mcp.tool()
def read_document(name: str) -> str:
    """Read one fixture document by exact filename."""
    fixtures_dir = _fixtures_dir()

    if Path(name).name != name:
        raise ValueError("Document name must be a plain filename.")

    path = (fixtures_dir / name).resolve()
    if path.parent != fixtures_dir:
        raise ValueError("Document path escapes fixtures directory.")
    if not path.is_file():
        raise FileNotFoundError(f"Fixture not found: {name}")

    return path.read_text(encoding="utf-8")


# 以 stdio transport 启动最小 MCP Server，并阻塞等待 Client 请求。
def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
