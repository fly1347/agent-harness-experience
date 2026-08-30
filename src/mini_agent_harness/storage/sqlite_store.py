"""
文件作用：
提供 Step 8.1 的最小 SQLite 持久化层，把 Session、Conversation Messages 和 Memory Records 写入本地数据库文件，并支持关闭进程后重新打开数据库再恢复这些对象。

整体结构：
1）SQLiteStore.__init__ / _create_schema：打开数据库并初始化 sessions、messages、memories 三张表；
2）save_session / load_session：按 Session 快照保存与恢复完整 canonical conversation history；
3）save_memory_records / load_memory_records：按 MemoryManager 当前 records 快照保存与恢复完整 Memory Lifecycle；
4）close / 上下文管理协议：显式关闭 SQLite 连接，便于测试“关闭后重新打开仍可恢复”。

职责边界：
- 本文件只负责“对象 <-> SQLite”的持久化，不负责决定何时保存；
- 当前采用小实验最容易审阅的 snapshot 写法：保存 Session 时重写该 Session 的 messages，保存 Memory 时重写 memories 表；
- AgentLoop / Session / MemoryManager 的自动接入，以及真正的 restart recovery 留到 Step 8.2 / 8.3。
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Iterable

from mini_agent_harness.core.memory import MemoryRecord
from mini_agent_harness.core.session import Session


class SQLiteStore:
    """把 Session 与 Memory 的当前状态保存到一个本地 SQLite 数据库文件。"""

    def __init__(self, db_path: str | Path) -> None:
        """打开数据库文件；父目录不存在时自动创建，并确保基础表已经存在。"""
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self.connection = sqlite3.connect(self.db_path)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA foreign_keys = ON")

        self._create_schema()

    def _create_schema(self) -> None:
        """创建 Step 8.1 所需的 sessions、messages、memories 三张表。"""
        with self.connection:
            self.connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS messages (
                    session_id TEXT NOT NULL,
                    sequence_index INTEGER NOT NULL,
                    payload_json TEXT NOT NULL,
                    PRIMARY KEY (session_id, sequence_index),
                    FOREIGN KEY (session_id)
                        REFERENCES sessions(session_id)
                        ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS memories (
                    memory_id TEXT PRIMARY KEY,
                    kind TEXT NOT NULL,
                    key TEXT NOT NULL,
                    value TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_memories_status
                    ON memories(status);

                CREATE INDEX IF NOT EXISTS idx_memories_kind_key
                    ON memories(kind, key);
                """
            )

    def save_session(self, session: Session) -> None:
        """把一个 Session 及其全部 messages 作为当前快照写入数据库。"""
        with self.connection:
            self.connection.execute(
                """
                INSERT INTO sessions(session_id, created_at, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    created_at = excluded.created_at,
                    updated_at = excluded.updated_at
                """,
                (
                    session.session_id,
                    session.created_at.isoformat(),
                    session.updated_at.isoformat(),
                ),
            )

            # Step 8.1 先采用整段快照，
            # 避免 append / 去重逻辑干扰持久化机制本身。
            self.connection.execute(
                "DELETE FROM messages WHERE session_id = ?",
                (session.session_id,),
            )

            self.connection.executemany(
                """
                INSERT INTO messages(
                    session_id,
                    sequence_index,
                    payload_json
                )
                VALUES (?, ?, ?)
                """,
                [
                    (
                        session.session_id,
                        index,
                        json.dumps(
                            message,
                            ensure_ascii=False,
                            separators=(",", ":"),
                        ),
                    )
                    for index, message in enumerate(session.messages)
                ],
            )

    def load_session(self, session_id: str) -> Session | None:
        """按 session_id 恢复 Session；数据库中不存在时返回 None。"""
        row = self.connection.execute(
            """
            SELECT session_id, created_at, updated_at
            FROM sessions
            WHERE session_id = ?
            """,
            (session_id,),
        ).fetchone()

        if row is None:
            return None

        message_rows = self.connection.execute(
            """
            SELECT payload_json
            FROM messages
            WHERE session_id = ?
            ORDER BY sequence_index ASC
            """,
            (session_id,),
        ).fetchall()

        return Session(
            session_id=row["session_id"],
            messages=[
                json.loads(message_row["payload_json"])
                for message_row in message_rows
            ],
            created_at=datetime.fromisoformat(
                row["created_at"]
            ),
            updated_at=datetime.fromisoformat(
                row["updated_at"]
            ),
        )

    def save_memory_records(
        self,
        records: Iterable[MemoryRecord],
    ) -> None:
        """把 MemoryManager.records 的完整生命周期快照写入数据库。"""
        snapshot = list(records)

        with self.connection:
            # MemoryManager 当前是单一实验级状态，
            # 因此这一阶段直接保存整张生命周期快照。
            self.connection.execute(
                "DELETE FROM memories"
            )

            self.connection.executemany(
                """
                INSERT INTO memories(
                    memory_id,
                    kind,
                    key,
                    value,
                    status,
                    created_at,
                    updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        record.memory_id,
                        record.kind,
                        record.key,
                        record.value,
                        record.status,
                        record.created_at.isoformat(),
                        record.updated_at.isoformat(),
                    )
                    for record in snapshot
                ],
            )

    def load_memory_records(self) -> list[MemoryRecord]:
        """恢复全部 Memory 版本，包括 superseded 历史记录。"""
        rows = self.connection.execute(
            """
            SELECT
                memory_id,
                kind,
                key,
                value,
                status,
                created_at,
                updated_at
            FROM memories
            ORDER BY created_at ASC, rowid ASC
            """
        ).fetchall()

        return [
            MemoryRecord(
                memory_id=row["memory_id"],
                kind=row["kind"],
                key=row["key"],
                value=row["value"],
                status=row["status"],
                created_at=datetime.fromisoformat(
                    row["created_at"]
                ),
                updated_at=datetime.fromisoformat(
                    row["updated_at"]
                ),
            )
            for row in rows
        ]

    def close(self) -> None:
        """关闭 SQLite 连接；数据库文件仍保留在磁盘上。"""
        self.connection.close()

    def __enter__(self) -> "SQLiteStore":
        """支持 with SQLiteStore(...) as store 的使用方式。"""
        return self

    def __exit__(
        self,
        exc_type,
        exc_value,
        traceback,
    ) -> None:
        """离开 with 代码块时自动关闭 SQLite 连接。"""
        self.close()
