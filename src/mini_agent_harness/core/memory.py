"""
文件作用：
提供 Step 7 的最小结构化长期记忆管理器，负责把需要跨轮保留的状态独立于 Conversation History 保存，并支持写入、读取和显式替换；旧版本不会被直接删除，便于后续 Trace 和持久化阶段追溯 Memory Lifecycle。

整体结构：
1）MemoryKind / MemoryStatus：限定第一版长期记忆的类型和生命周期状态；
2）MemoryRecord：保存 memory_id、kind、key、value、status 与创建/更新时间；
3）MemoryChange：描述一次 write / supersede 的结果，供 AgentLoop 记录 Trace；
4）MemoryManager.write / retrieve_active / supersede：完成最小长期记忆生命周期；
5）apply_explicit_instruction：识别本实验固定 Scenario 中明确的“记住 / 更新实验重点”指令，并映射到结构化 Memory 操作；
6）from_records：把 SQLite 恢复出的 MemoryRecord 列表重新装配成 MemoryManager；
7）context_items：把当前 active memory 转成 ContextBuilder 可直接注入的简洁文本。

职责边界：
- 本文件只处理结构化 Memory 与最小显式指令规则，不负责 Conversation Summary；
- Memory 是否进入某次模型请求由 ContextBuilder 决定，Trace 由 AgentLoop / TraceLogger 负责；
- MemoryManager 本身仍只管理进程内对象；Step 8 由 SQLiteStore 负责落盘，重启后通过 from_records 重新装配。
"""

from __future__ import annotations

import re
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Literal
from uuid import uuid4


MemoryKind = Literal["preference", "fact", "decision"]
MemoryStatus = Literal["active", "superseded"]
MemoryAction = Literal["write", "supersede"]
_ALLOWED_KINDS: set[str] = {"preference", "fact", "decision"}
_QUOTED_VALUE_RE = re.compile(r"[“\"]([^”\"]+)[”\"]")


def _utc_now() -> datetime:
    """统一生成带时区的 UTC 时间戳。"""
    return datetime.now(timezone.utc)


@dataclass
class MemoryRecord:
    memory_id: str = field(default_factory=lambda: uuid4().hex)
    kind: MemoryKind = "fact"
    key: str = ""
    value: str = ""
    status: MemoryStatus = "active"
    created_at: datetime = field(default_factory=_utc_now)
    updated_at: datetime = field(default_factory=_utc_now)


@dataclass
class MemoryChange:
    action: MemoryAction
    record: MemoryRecord
    previous: MemoryRecord | None = None


class MemoryManager:
    """维护进程内长期记忆，并提供写入、读取、替换和显式指令处理。"""

    def __init__(self) -> None:
        self.records: list[MemoryRecord] = []

    @classmethod
    def from_records(
        cls,
        records: list[MemoryRecord],
    ) -> "MemoryManager":
        """从持久化恢复出的记录重建 MemoryManager，并与调用方数据隔离。"""
        manager = cls()
        manager.records = deepcopy(records)
        return manager

    def write(
        self,
        *,
        kind: MemoryKind,
        key: str,
        value: str,
    ) -> MemoryRecord:
        """新增一条 active memory；已有同 kind + key 记录时拒绝静默覆盖。"""
        self._validate(kind=kind, key=key, value=value)
        if self._find_active(kind=kind, key=key) is not None:
            raise ValueError(
                f"Active memory already exists for {kind}:{key}; use supersede()."
            )

        record = MemoryRecord(
            kind=kind,
            key=key.strip(),
            value=value.strip(),
        )
        self.records.append(record)
        return deepcopy(record)

    def retrieve_active(
        self,
        *,
        kind: MemoryKind | None = None,
        key: str | None = None,
    ) -> list[MemoryRecord]:
        """读取当前 active memory，可按 kind、key 或两者组合过滤。"""
        if kind is not None and kind not in _ALLOWED_KINDS:
            raise ValueError(f"Unsupported memory kind: {kind}")

        normalized_key = key.strip() if isinstance(key, str) else None
        matched = [
            record
            for record in self.records
            if record.status == "active"
            and (kind is None or record.kind == kind)
            and (normalized_key is None or record.key == normalized_key)
        ]
        return deepcopy(matched)

    def supersede(
        self,
        *,
        kind: MemoryKind,
        key: str,
        value: str,
    ) -> MemoryRecord:
        """把指定 active memory 标记为 superseded，并追加新的 active 版本。"""
        self._validate(kind=kind, key=key, value=value)
        normalized_key = key.strip()
        current = self._find_active(kind=kind, key=normalized_key)
        if current is None:
            raise ValueError(f"No active memory found for {kind}:{normalized_key}")

        current.status = "superseded"
        current.updated_at = _utc_now()

        replacement = MemoryRecord(
            kind=kind,
            key=normalized_key,
            value=value.strip(),
        )
        self.records.append(replacement)
        return deepcopy(replacement)

    def apply_explicit_instruction(self, user_input: str) -> MemoryChange | None:
        """识别固定实验中的明确记忆指令，并执行对应的 write 或 supersede。"""
        values = _QUOTED_VALUE_RE.findall(user_input)
        if not values:
            return None

        if "记住" in user_input and "实验" in user_input and "重点" in user_input:
            value = " + ".join(values)
            created = self.write(
                kind="decision",
                key="experiment_focus",
                value=value,
            )
            return MemoryChange(action="write", record=created)

        if "实验重点更新为" in user_input:
            current = self.retrieve_active(
                kind="decision",
                key="experiment_focus",
            )
            if not current:
                raise ValueError(
                    "Cannot update experiment_focus before an active memory exists."
                )
            value = " + ".join(values)
            replacement = self.supersede(
                kind="decision",
                key="experiment_focus",
                value=value,
            )
            previous = current[0]
            previous.status = "superseded"
            return MemoryChange(
                action="supersede",
                record=replacement,
                previous=previous,
            )

        return None

    def context_items(self) -> list[str]:
        """把当前 active memory 转成供 ContextBuilder 注入模型请求的文本列表。"""
        return [
            f"{record.kind}:{record.key} = {record.value}"
            for record in self.retrieve_active()
        ]

    def _find_active(
        self,
        *,
        kind: MemoryKind,
        key: str,
    ) -> MemoryRecord | None:
        """在内部记录中查找指定 kind + key 的 active memory。"""
        normalized_key = key.strip()
        for record in reversed(self.records):
            if (
                record.status == "active"
                and record.kind == kind
                and record.key == normalized_key
            ):
                return record
        return None

    @staticmethod
    def _validate(
        *,
        kind: MemoryKind,
        key: str,
        value: str,
    ) -> None:
        """校验 Memory 类型以及 key / value 的最小输入约束。"""
        if kind not in _ALLOWED_KINDS:
            raise ValueError(f"Unsupported memory kind: {kind}")
        if not isinstance(key, str) or not key.strip():
            raise ValueError("Memory key must be a non-empty string.")
        if not isinstance(value, str) or not value.strip():
            raise ValueError("Memory value must be a non-empty string.")
