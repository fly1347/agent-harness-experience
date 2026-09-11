"""
文件作用：
验证 Step 8.1 SQLiteStore 的最小持久化能力，不访问真实模型、工具或网络；重点确认 Session、Message 和 Memory 在数据库关闭并重新打开以后仍能无损恢复。

整体结构：
1）Session round-trip：保存 Session 后关闭数据库，再打开并恢复完整消息顺序与时间字段；
2）Session snapshot update：同一 session_id 再次保存时应得到最新完整快照，不重复旧消息；
3）Memory round-trip：保存 active / superseded 多版本 Memory，重开数据库后状态链仍保持；
4）Missing session：读取不存在的 Session 时返回 None。
"""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from mini_agent_harness.core.memory import MemoryManager
from mini_agent_harness.core.session import Session
from mini_agent_harness.storage.sqlite_store import SQLiteStore


class SQLiteStoreTests(unittest.TestCase):
    def test_session_survives_close_and_reopen(self) -> None:
        """数据库关闭再打开后，应恢复同一 Session 的完整消息顺序和内容。"""
        with TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "harness.db"

            session = Session(
                session_id="session-a"
            )
            session.add_message(
                {
                    "role": "system",
                    "content": "system",
                }
            )
            session.add_message(
                {
                    "role": "user",
                    "content": "你好",
                }
            )
            session.add_message(
                {
                    "role": "assistant",
                    "content": "我去读取资料。",
                    "tool_calls": [
                        {
                            "id": "call-1",
                            "type": "function",
                            "function": {
                                "name": "read_document",
                                "arguments": (
                                    '{"name":"01.md"}'
                                ),
                            },
                        }
                    ],
                }
            )
            session.add_message(
                {
                    "role": "tool",
                    "tool_call_id": "call-1",
                    "content": "文档内容",
                }
            )

            expected_messages = (
                session.messages.copy()
            )
            expected_created_at = (
                session.created_at
            )
            expected_updated_at = (
                session.updated_at
            )

            with SQLiteStore(db_path) as store:
                store.save_session(session)

            with SQLiteStore(db_path) as reopened:
                restored = reopened.load_session(
                    "session-a"
                )

            self.assertIsNotNone(restored)
            self.assertEqual(
                restored.session_id,
                "session-a",
            )
            self.assertEqual(
                restored.messages,
                expected_messages,
            )
            self.assertEqual(
                restored.created_at,
                expected_created_at,
            )
            self.assertEqual(
                restored.updated_at,
                expected_updated_at,
            )

    def test_resaving_session_replaces_message_snapshot(
        self,
    ) -> None:
        """同一 Session 再次保存时，应更新为最新消息快照而不是重复追加。"""
        with TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "harness.db"

            session = Session(
                session_id="session-b"
            )
            session.add_message(
                {
                    "role": "user",
                    "content": "first",
                }
            )

            with SQLiteStore(db_path) as store:
                store.save_session(session)

                session.add_message(
                    {
                        "role": "assistant",
                        "content": "second",
                    }
                )

                store.save_session(session)
                restored = store.load_session(
                    "session-b"
                )

            self.assertIsNotNone(restored)
            self.assertEqual(
                [
                    item["content"]
                    for item in restored.messages
                ],
                [
                    "first",
                    "second",
                ],
            )

    def test_memory_lifecycle_survives_close_and_reopen(
        self,
    ) -> None:
        """Memory 的多版本状态在重开数据库后应保持不变。"""
        with TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "harness.db"

            memory = MemoryManager()

            memory.write(
                kind="decision",
                key="experiment_focus",
                value="评估对象分层 + 指标边界",
            )

            memory.supersede(
                kind="decision",
                key="experiment_focus",
                value=(
                    "Context Management + 评估信度"
                ),
            )

            memory.write(
                kind="preference",
                key="report_language",
                value="中文",
            )

            with SQLiteStore(db_path) as store:
                store.save_memory_records(
                    memory.records
                )

            with SQLiteStore(db_path) as reopened:
                restored = (
                    reopened.load_memory_records()
                )

            self.assertEqual(
                len(restored),
                3,
            )
            self.assertEqual(
                [
                    item.status
                    for item in restored
                ],
                [
                    "superseded",
                    "active",
                    "active",
                ],
            )
            self.assertEqual(
                [
                    item.value
                    for item in restored
                ],
                [
                    "评估对象分层 + 指标边界",
                    "Context Management + 评估信度",
                    "中文",
                ],
            )
            self.assertEqual(
                [
                    item.memory_id
                    for item in restored
                ],
                [
                    item.memory_id
                    for item in memory.records
                ],
            )

    def test_missing_session_returns_none(
        self,
    ) -> None:
        """读取不存在的 session_id 时应明确返回 None。"""
        with TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "harness.db"

            with SQLiteStore(db_path) as store:
                self.assertIsNone(
                    store.load_session("missing")
                )


if __name__ == "__main__":
    unittest.main()
