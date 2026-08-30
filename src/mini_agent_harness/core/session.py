"""
文件作用：
提供多轮实验的最小内存 Session，保存一条会话的完整 canonical history，供 AgentLoop 在后续用户轮次继续读取和追加。

整体结构：
1）_utc_now：统一生成 UTC 时间戳；
2）Session：保存 session_id、messages、created_at、updated_at；
3）add_message：深拷贝写入 user / assistant / tool 等消息，并刷新 updated_at。

职责边界：
当前 Session 只存在于进程内，不负责 SQLite / 文件持久化；持久化在后续实验阶段单独实现。
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class Session:
    session_id: str = field(default_factory=lambda: uuid4().hex)
    messages: list[dict[str, Any]] = field(default_factory=list)
    created_at: datetime = field(default_factory=_utc_now)
    updated_at: datetime = field(default_factory=_utc_now)

    def add_message(self, message: dict[str, Any]) -> None:
        """把消息深拷贝追加到完整会话历史，并刷新 Session 更新时间。"""
        self.messages.append(deepcopy(message))
        self.updated_at = _utc_now()
