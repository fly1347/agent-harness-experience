"""
文件作用：
从 Session 的完整历史中构造“本次 Model Call 实际看到的 working context”，同时负责把检索出的 active memory 插入请求；构建过程只生成本次请求视图，不删除、不改写 Session 原始历史。

整体结构：
1）ContextStrategy / ContextBuild：定义策略名称和一次上下文构建的统计结果；
2）ContextBuilder.build：统一入口，分别实现 full_history、last_n、managed 三种历史选择策略，并在 system 之后注入 active memory；
3）_recent_window：裁剪最近消息时尽量从 user 消息开始，避免拆断一组 tool call / tool result；
4）_summarize：对较早的 user / final assistant 消息做本地确定性压缩，不额外调用 LLM；
5）_memory_message：把 active memory 组合成独立 system message，使长期记忆与 Conversation History 保持分层。
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Literal


ContextStrategy = Literal[
    "full_history",
    "last_n",
    "managed",
]


@dataclass
class ContextBuild:
    strategy: ContextStrategy
    messages: list[dict[str, Any]]
    history_message_count: int
    request_message_count: int
    recent_message_count: int
    summary_injected: bool = False
    summary: str = ""
    active_memory_count: int = 0


class ContextBuilder:
    """按指定策略从完整 Session 历史生成本次模型请求使用的 working context。"""

    # 校验上下文策略与窗口大小，保存摘要触发和长度配置。
    def __init__(
        self,
        strategy: ContextStrategy = "full_history",
        *,
        last_n: int = 6,
        managed_recent_n: int = 6,
        summary_trigger_messages: int = 8,
        summary_max_chars: int = 1800,
    ) -> None:
        if strategy not in {"full_history", "last_n", "managed"}:
            raise ValueError(f"Unknown context strategy: {strategy}")
        if last_n < 1 or managed_recent_n < 1:
            raise ValueError("Context window sizes must be >= 1.")

        self.strategy = strategy
        self.last_n = last_n
        self.managed_recent_n = managed_recent_n
        self.summary_trigger_messages = summary_trigger_messages
        self.summary_max_chars = summary_max_chars

    # 根据当前策略构造一次 Model Call 的消息列表，并返回对应上下文统计。
    def build(
        self,
        history: list[dict[str, Any]],
        *,
        active_memory: list[str] | None = None,
    ) -> ContextBuild:
        if not history:
            return ContextBuild(
                strategy=self.strategy,
                messages=[],
                history_message_count=0,
                request_message_count=0,
                recent_message_count=0,
            )

        system = deepcopy(history[0])
        body = history[1:]
        memory = [x for x in (active_memory or []) if x]

        if self.strategy == "full_history":
            messages = deepcopy(history)
            if memory:
                messages.insert(1, self._memory_message(memory))
            return ContextBuild(
                strategy=self.strategy,
                messages=messages,
                history_message_count=len(history),
                request_message_count=len(messages),
                recent_message_count=len(body),
                active_memory_count=len(memory),
            )

        if self.strategy == "last_n":
            recent, _ = self._recent_window(body, self.last_n)
            messages = [system]
            if memory:
                messages.append(self._memory_message(memory))
            messages.extend(deepcopy(recent))
            return ContextBuild(
                strategy=self.strategy,
                messages=messages,
                history_message_count=len(history),
                request_message_count=len(messages),
                recent_message_count=len(recent),
                active_memory_count=len(memory),
            )

        # 历史较短时 Managed Context 与 Full History 保持一致；
        # 达到阈值后再压缩，便于观察策略差异。
        if len(body) <= self.summary_trigger_messages:
            messages = deepcopy(history)
            if memory:
                messages.insert(1, self._memory_message(memory))
            return ContextBuild(
                strategy=self.strategy,
                messages=messages,
                history_message_count=len(history),
                request_message_count=len(messages),
                recent_message_count=len(body),
                active_memory_count=len(memory),
            )

        recent, start = self._recent_window(
            body,
            self.managed_recent_n,
        )
        older = body[:start]
        summary = self._summarize(older)

        messages: list[dict[str, Any]] = [system]
        if summary:
            messages.append(
                {
                    "role": "system",
                    "content": (
                        "Conversation summary from earlier messages:\n"
                        + summary
                    ),
                }
            )
        if memory:
            messages.append(self._memory_message(memory))
        messages.extend(deepcopy(recent))

        return ContextBuild(
            strategy=self.strategy,
            messages=messages,
            history_message_count=len(history),
            request_message_count=len(messages),
            recent_message_count=len(recent),
            summary_injected=bool(summary),
            summary=summary,
            active_memory_count=len(memory),
        )

    # 选取最近一段完整对话窗口，尽量不从孤立的 assistant/tool 消息中间开始。
    @staticmethod
    def _recent_window(
        body: list[dict[str, Any]],
        n: int,
    ) -> tuple[list[dict[str, Any]], int]:
        if len(body) <= n:
            return body, 0

        tail_start = len(body) - n

        # 尽量从最近 N 条中的 user 消息开始，避免请求以孤立的 tool result 开头。
        # 若当前工具事务本身超过 N 条，则向前扩到最近的 user 消息，保持消息链合法。
        user_in_tail = next(
            (
                i
                for i in range(tail_start, len(body))
                if body[i].get("role") == "user"
            ),
            None,
        )
        if user_in_tail is not None:
            start = user_in_tail
        else:
            earlier_users = [
                i
                for i in range(0, tail_start)
                if body[i].get("role") == "user"
            ]
            start = earlier_users[-1] if earlier_users else tail_start

        return body[start:], start

    # 提取较早的 user / final assistant 内容做确定性压缩，生成 Managed Context 摘要。
    def _summarize(self, messages: list[dict[str, Any]]) -> str:
        pieces: list[str] = []
        used = 0

        for message in messages:
            role = message.get("role")
            content = message.get("content")
            if not isinstance(content, str) or not content.strip():
                continue

            if role == "user":
                label = "User"
                limit = 260
            elif role == "assistant" and not message.get("tool_calls"):
                label = "Assistant"
                limit = 520
            else:
                continue

            compact = " ".join(content.split())
            if len(compact) > limit:
                compact = compact[: limit - 1] + "…"

            piece = f"{label}: {compact}"
            remaining = self.summary_max_chars - used
            if remaining <= 0:
                break
            if len(piece) > remaining:
                piece = piece[: max(0, remaining - 1)] + "…"

            pieces.append(piece)
            used += len(piece) + 1

        return "\n".join(pieces)

    # 把当前有效长期记忆作为“当前状态层”注入，并明确它与历史摘要的冲突优先级。
    @staticmethod
    def _memory_message(memory: list[str]) -> dict[str, str]:
        return {
            "role": "system",
            "content": (
                "Active long-term memory (authoritative current state):\n"
                "- These records represent the currently valid structured state.\n"
                "- If an earlier conversation summary or older conversation text conflicts "
                "with the same item, use this active memory as the current value.\n"
                "- Older values may still be used only as historical context when the user "
                "asks about how the state changed.\n"
                "- " + "\n- ".join(memory)
            ),
        }
