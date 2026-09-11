"""
文件作用：
提供 Mini Agent Harness 的本地工具注册与执行层：一边把工具 schema 暴露给模型做原生 function calling，一边在本地执行对应操作并返回字符串结果。

整体结构：
1）ToolRegistry 初始化：保存 fixtures 文档目录和 notes 输出文件位置；
2）schemas：定义 list_documents、read_document、save_note 三个工具的名称、说明和参数 schema；
3）execute：按工具名分发实际执行逻辑；
4）文件安全检查：read_document 只允许读取 fixtures_dir 下的纯文件名，阻止路径越界。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class ToolRegistry:
    # 解析文档根目录和笔记文件的绝对路径，供本地工具执行时使用。
    def __init__(self, fixtures_dir: Path, notes_file: Path) -> None:
        self.fixtures_dir = fixtures_dir.resolve()
        self.notes_file = notes_file.resolve()

    # 返回本实验提供给模型的三个原生 function calling 工具定义。
    @property
    def schemas(self) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "list_documents",
                    "description": "List available fixture documents.",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "additionalProperties": False,
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "read_document",
                    "description": "Read one fixture document by exact filename.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "name": {
                                "type": "string",
                                "description": "Exact fixture filename.",
                            }
                        },
                        "required": ["name"],
                        "additionalProperties": False,
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "save_note",
                    "description": "Append a short note to the experiment notes file.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "text": {
                                "type": "string",
                                "description": "Note text to save.",
                            }
                        },
                        "required": ["text"],
                        "additionalProperties": False,
                    },
                },
            },
        ]

    # 按工具名执行 list/read/save 对应逻辑，并返回可写回模型上下文的字符串结果。
    def execute(self, name: str, arguments: dict[str, Any]) -> str:
        if name == "list_documents":
            docs = sorted(
                p.name
                for p in self.fixtures_dir.iterdir()
                if p.is_file()
            )
            return json.dumps(docs, ensure_ascii=False)

        if name == "read_document":
            filename = arguments["name"]

            if Path(filename).name != filename:
                raise ValueError("Document name must be a plain filename.")

            path = (self.fixtures_dir / filename).resolve()

            if path.parent != self.fixtures_dir:
                raise ValueError("Document path escapes fixtures directory.")
            if not path.is_file():
                raise FileNotFoundError(f"Fixture not found: {filename}")

            return path.read_text(encoding="utf-8")

        if name == "save_note":
            text = str(arguments["text"]).strip()
            self.notes_file.parent.mkdir(parents=True, exist_ok=True)

            with self.notes_file.open("a", encoding="utf-8") as f:
                f.write(text + "\n")

            return "Note saved."

        raise ValueError(f"Unknown tool: {name}")
