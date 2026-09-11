"""
文件作用：
实现一次用户轮次内部的 Agent 主循环，把 Session、ContextBuilder、MemoryManager、模型、工具和 Trace 串成完整执行链；同一轮内可经历多次 Model Call / Tool Call，直到模型给出最终回答。

整体结构：
1）SYSTEM_PROMPT：规定实验 Agent 的基础行为与工具使用约束；
2）AgentLoop 初始化：接收 provider、ToolRegistry、TraceLogger、ContextBuilder，以及可选 MemoryManager / SQLiteStore；
3）run 起始：写入本轮 user message，并从 MemoryManager 读取当前 active memory；
4）Model/Tool 循环：ContextBuilder 组装 working context，必要时注入 active memory，再调用模型、执行工具并回写 Session；
5）Memory 收尾：最终回答形成后识别明确的“记住 / 更新”指令，执行 write 或 supersede，并写入 Memory Trace；
6）持久化与收尾：最终回答后把 Session / Memory 当前快照自动写入 SQLite，再累计统计并生成 RUN_SUMMARY。

职责边界：
- Session 保存完整会话历史，MemoryManager 保存独立的长期状态，ContextBuilder 决定两者如何进入当前模型请求；
- SQLiteStore 为可选依赖；配置后由本文件在正常 Turn 结束时自动保存状态，但不负责关闭数据库连接；
- 真正的“新 Python 进程重新打开数据库并恢复 Agent”留到 Step 8.3。
"""

from __future__ import annotations

import json
import time
from typing import TYPE_CHECKING

from mini_agent_harness.core.context import ContextBuilder
from mini_agent_harness.core.memory import MemoryManager, MemoryRecord
from mini_agent_harness.core.session import Session
from mini_agent_harness.core.tools import ToolRegistry
from mini_agent_harness.core.tracing import TraceLogger
if TYPE_CHECKING:
    from mini_agent_harness.providers.openai_compatible import (
        OpenAICompatibleProvider,
    )
    from mini_agent_harness.storage.sqlite_store import SQLiteStore


SYSTEM_PROMPT = """You are a minimal research assistant used to study an Agent Harness.

Rules:
- Use the provided tools when the user asks you to inspect fixture documents.
- Never claim to know a fixture's contents without reading it through a tool.
- When a document filename is provided, prefer read_document directly.
- Answer concisely from the tool result.
"""


class AgentLoop:
    # 组装模型、工具、上下文和持久化组件，并校验可选重试次数。
    def __init__(
        self,
        provider: OpenAICompatibleProvider,
        tools: ToolRegistry,
        tracer: TraceLogger,
        max_steps: int = 6,
        max_retry: int | None = None,
        context_builder: ContextBuilder | None = None,
        memory_manager: MemoryManager | None = None,
        state_store: SQLiteStore | None = None,
    ) -> None:
        self.provider = provider
        self.tools = tools
        self.tracer = tracer
        self.max_steps = max_steps
        if max_retry is not None and max_retry < 0:
            raise ValueError("max_retry must be >= 0 or None")
        # None 保留 Step 1～9 的旧行为：工具报错写回模型，由模型自行恢复。
        # Step 10.1 显式传 1 时，Harness 才接管本地 Tool Execution 重试。
        self.max_retry = max_retry
        self.context_builder = context_builder or ContextBuilder()
        self.memory_manager = memory_manager
        self.state_store = state_store

    # 执行一个完整用户轮次，在模型与工具之间循环，直到得到最终回答并完成统计。
    def run(
        self,
        user_input: str,
        session: Session | None = None,
    ) -> str:
        run_started = time.perf_counter()

        model_calls = 0
        tool_calls_count = 0

        prompt_tokens = 0
        completion_tokens = 0
        cache_hit_tokens = 0
        cache_miss_tokens = 0

        model_ms = 0.0
        tool_ms = 0.0
        last_model = self.provider.model

        active_session = session or Session()

        if not active_session.messages:
            active_session.add_message(
                {"role": "system", "content": SYSTEM_PROMPT}
            )

        active_session.add_message(
            {"role": "user", "content": user_input}
        )

        active_memory_records = (
            self.memory_manager.retrieve_active()
            if self.memory_manager is not None
            else []
        )
        active_memory = (
            self.memory_manager.context_items()
            if self.memory_manager is not None
            else []
        )

        if self.memory_manager is not None:
            self.tracer.log(
                "MEMORY_READ",
                {
                    "count": len(active_memory_records),
                    "records": [
                        self._memory_record_payload(record)
                        for record in active_memory_records
                    ],
                    "reason": (
                        "本轮开始时读取当前有效 Memory；如果存在，就加入本轮每次 Model Call 的输入。"
                    ),
                },
                console={
                    "count": len(active_memory_records),
                    "keys": [
                        record.key for record in active_memory_records
                    ],
                },
            )

        for step in range(1, self.max_steps + 1):
            context = self.context_builder.build(
                active_session.messages,
                active_memory=active_memory,
            )
            messages = context.messages

            if context.active_memory_count:
                self.tracer.log(
                    "MEMORY_INJECT",
                    {
                        "step": step,
                        "count": context.active_memory_count,
                        "records": [
                            self._memory_record_payload(record)
                            for record in active_memory_records
                        ],
                        "reason": (
                            "这些记录当前为 active，作为独立长期状态加入本次 working context。"
                        ),
                    },
                    console={
                        "step": step,
                        "count": context.active_memory_count,
                        "keys": [
                            record.key for record in active_memory_records
                        ],
                    },
                )

            self.tracer.log(
                "CONTEXT_BUILD",
                {
                    "step": step,
                    "strategy": context.strategy,
                    "history_message_count": context.history_message_count,
                    "request_message_count": context.request_message_count,
                    "recent_message_count": context.recent_message_count,
                    "summary_injected": context.summary_injected,
                    "summary": context.summary,
                    "active_memory_count": context.active_memory_count,
                },
                console={
                    "step": step,
                    "strategy": context.strategy,
                    "history_messages": context.history_message_count,
                    "request_messages": context.request_message_count,
                    "summary_injected": context.summary_injected,
                    "summary_chars": len(context.summary),
                },
            )

            result = self.provider.complete(
                messages=messages,
                tools=self.tools.schemas,
            )

            model_calls += 1
            model_ms += result.latency_ms
            last_model = result.model

            usage = result.usage or {}

            prompt_tokens += int(
                usage.get("prompt_tokens") or 0
            )
            completion_tokens += int(
                usage.get("completion_tokens") or 0
            )
            cache_hit_tokens += int(
                usage.get("prompt_cache_hit_tokens") or 0
            )
            cache_miss_tokens += int(
                usage.get("prompt_cache_miss_tokens") or 0
            )

            self.tracer.log(
                "MODEL_REQUEST",
                {
                    "step": step,
                    "request": result.request,
                },
                console={
                    "step": step,
                    "message_count": len(messages),
                    "tools": [
                        t["function"]["name"]
                        for t in self.tools.schemas
                    ],
                },
            )

            self.tracer.log(
                "MODEL_RESPONSE",
                {
                    "step": step,
                    "response": result.response,
                    "latency_ms": result.latency_ms,
                    "usage": result.usage,
                },
                console={
                    "step": step,
                    "model": result.model,
                    "latency_ms": round(result.latency_ms, 2),
                    "usage": result.usage,
                    "content": result.message.content,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        }
                        for tc in (result.message.tool_calls or [])
                    ],
                },
            )

            assistant_message = result.message.model_dump(
                exclude_none=True
            )
            active_session.add_message(assistant_message)

            tool_calls = result.message.tool_calls or []

            if not tool_calls:
                answer = result.message.content or ""

                self.tracer.log(
                    "FINAL_ANSWER",
                    {
                        "step": step,
                        "answer": answer,
                    },
                )

                self._apply_memory_instruction(user_input)
                self._persist_state(active_session)

                end_to_end_ms = (
                    time.perf_counter() - run_started
                ) * 1000

                self.tracer.finalize(
                    user_input=user_input,
                    model=last_model,
                    model_calls=model_calls,
                    tool_calls=tool_calls_count,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    cache_hit_tokens=cache_hit_tokens,
                    cache_miss_tokens=cache_miss_tokens,
                    model_ms=model_ms,
                    tool_ms=tool_ms,
                    end_to_end_ms=end_to_end_ms,
                )

                return answer

            for tool_call in tool_calls:
                tool_calls_count += 1
                name = tool_call.function.name

                arguments = json.loads(
                    tool_call.function.arguments or "{}"
                )

                self.tracer.log(
                    "TOOL_CALL",
                    {
                        "step": step,
                        "tool_call_id": tool_call.id,
                        "name": name,
                        "arguments": arguments,
                    },
                )

                output, retry_ms = self._execute_tool_with_retry(
                    step=step,
                    tool_call_id=tool_call.id,
                    name=name,
                    arguments=arguments,
                )
                tool_ms += retry_ms

                active_session.add_message(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": output,
                    }
                )

        raise RuntimeError(
            f"Agent loop exceeded max_steps={self.max_steps}"
        )

    # 执行工具；默认保留旧错误回写语义，显式 max_retry 时由 Harness 重试。
    def _execute_tool_with_retry(
        self,
        *,
        step: int,
        tool_call_id: str,
        name: str,
        arguments: dict[str, object],
    ) -> tuple[str, float]:
        if self.max_retry is None:
            tool_started = time.perf_counter()
            try:
                output = self.tools.execute(name, arguments)
            except Exception as exc:
                tool_latency_ms = (
                    time.perf_counter() - tool_started
                ) * 1000
                output = f"ERROR: {type(exc).__name__}: {exc}"
                self.tracer.log(
                    "ERROR",
                    {
                        "step": step,
                        "tool_call_id": tool_call_id,
                        "name": name,
                        "error": output,
                    },
                )
                return output, tool_latency_ms

            tool_latency_ms = (
                time.perf_counter() - tool_started
            ) * 1000
            self._log_tool_result(
                step=step,
                tool_call_id=tool_call_id,
                name=name,
                output=output,
                tool_latency_ms=tool_latency_ms,
                attempt=1,
                retry_count=0,
            )
            return output, tool_latency_ms

        total_latency_ms = 0.0
        max_retry = self.max_retry

        for attempt in range(1, max_retry + 2):
            tool_started = time.perf_counter()
            try:
                output = self.tools.execute(name, arguments)
            except Exception as exc:
                attempt_latency_ms = (
                    time.perf_counter() - tool_started
                ) * 1000
                total_latency_ms += attempt_latency_ms
                error = f"{type(exc).__name__}: {exc}"
                will_retry = attempt <= max_retry

                self.tracer.log(
                    "TOOL_ERROR",
                    {
                        "step": step,
                        "tool_call_id": tool_call_id,
                        "name": name,
                        "arguments": arguments,
                        "attempt": attempt,
                        "max_retry": max_retry,
                        "will_retry": will_retry,
                        "error": error,
                        "tool_latency_ms": attempt_latency_ms,
                    },
                )

                if will_retry:
                    self.tracer.log(
                        "TOOL_RETRY",
                        {
                            "step": step,
                            "tool_call_id": tool_call_id,
                            "name": name,
                            "retry_number": attempt,
                            "max_retry": max_retry,
                            "reason": error,
                        },
                    )
                    continue

                self.tracer.log(
                    "TOOL_RETRY_EXHAUSTED",
                    {
                        "step": step,
                        "tool_call_id": tool_call_id,
                        "name": name,
                        "attempts": attempt,
                        "max_retry": max_retry,
                        "error": error,
                    },
                )
                raise RuntimeError(
                    f"Tool {name!r} failed after {attempt} attempt(s): {error}"
                ) from exc

            attempt_latency_ms = (
                time.perf_counter() - tool_started
            ) * 1000
            total_latency_ms += attempt_latency_ms
            self._log_tool_result(
                step=step,
                tool_call_id=tool_call_id,
                name=name,
                output=output,
                tool_latency_ms=attempt_latency_ms,
                attempt=attempt,
                retry_count=attempt - 1,
            )
            return output, total_latency_ms

        raise AssertionError("unreachable retry loop")

    # 统一记录成功 Tool Result，并显式标注它是第几次执行尝试。
    def _log_tool_result(
        self,
        *,
        step: int,
        tool_call_id: str,
        name: str,
        output: str,
        tool_latency_ms: float,
        attempt: int,
        retry_count: int,
    ) -> None:
        self.tracer.log(
            "TOOL_RESULT",
            {
                "step": step,
                "tool_call_id": tool_call_id,
                "name": name,
                "result": output,
                "tool_latency_ms": tool_latency_ms,
                "attempt": attempt,
                "retry_count": retry_count,
            },
            console={
                "step": step,
                "name": name,
                "attempt": attempt,
                "retry_count": retry_count,
                "tool_latency_ms": round(tool_latency_ms, 3),
                "result_chars": len(output),
                "preview": output[:300],
            },
        )

    # 正常 Turn 结束后，把 canonical Session 与完整 Memory Lifecycle 自动保存到 SQLite。
    def _persist_state(self, session: Session) -> None:
        if self.state_store is None:
            return

        self.state_store.save_session(session)

        memory_count = 0
        if self.memory_manager is not None:
            self.state_store.save_memory_records(
                self.memory_manager.records
            )
            memory_count = len(self.memory_manager.records)

        self.tracer.log(
            "STATE_PERSIST",
            {
                "session_id": session.session_id,
                "message_count": len(session.messages),
                "memory_record_count": memory_count,
                "db_path": str(self.state_store.db_path),
                "reason": (
                    "本轮正常结束后保存 Session 与 Memory 当前快照，"
                    "供后续跨进程恢复实验使用。"
                ),
            },
            console={
                "session_id": session.session_id,
                "messages": len(session.messages),
                "memory_records": memory_count,
            },
        )

    # 把 MemoryRecord 转成适合 Trace 展示和 JSONL 固化的基础字段。
    @staticmethod
    def _memory_record_payload(record: MemoryRecord) -> dict[str, str]:
        return {
            "memory_id": record.memory_id,
            "kind": record.kind,
            "key": record.key,
            "value": record.value,
            "status": record.status,
        }

    # 在最终回答后处理本轮明确的长期记忆写入或替换指令。
    def _apply_memory_instruction(self, user_input: str) -> None:
        if self.memory_manager is None:
            return

        change = self.memory_manager.apply_explicit_instruction(user_input)
        if change is None:
            return

        if change.action == "write":
            self.tracer.log(
                "MEMORY_WRITE",
                {
                    "action": "write",
                    "record": self._memory_record_payload(change.record),
                    "reason": "用户明确要求记住该实验重点。",
                },
            )
            return

        self.tracer.log(
            "MEMORY_SUPERSEDE",
            {
                "action": "supersede",
                "previous": (
                    self._memory_record_payload(change.previous)
                    if change.previous is not None
                    else None
                ),
                "record": self._memory_record_payload(change.record),
                "reason": "用户明确要求更新已有实验重点，旧版本保留为 superseded。",
            },
        )
