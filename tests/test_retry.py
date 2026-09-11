"""
文件作用：
验证 Step 10.1 的最小 Tool Retry 机制，不访问真实模型和网络。

整体结构：
1）FakeProvider：第一次让模型调用 read_document，第二次返回最终答案；
2）FailOnceTools：第一次 read_document 抛错，第二次成功；
3）AlwaysFailTools：每次 read_document 都抛错；
4）测试 transient failure 会被 Harness 重试一次并继续 Agent Loop；
5）测试连续失败在 max_retry=1 用尽后终止，不再把控制权交回模型。
"""

from __future__ import annotations

import json
from copy import deepcopy
from types import SimpleNamespace
import unittest

from mini_agent_harness.core.agent_loop import AgentLoop
from mini_agent_harness.core.session import Session


class _Message:
    """构造 AgentLoop 所需的最小 assistant message。"""

    def __init__(self, *, content: str = "", tool_calls=None) -> None:
        self.role = "assistant"
        self.content = content
        self.tool_calls = tool_calls

    # 把测试消息转成运行器消费的字典结构。
    def model_dump(self, exclude_none: bool = False) -> dict[str, object]:
        payload: dict[str, object] = {
            "role": self.role,
            "content": self.content,
        }
        if self.tool_calls:
            payload["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.function.name,
                        "arguments": call.function.arguments,
                    },
                }
                for call in self.tool_calls
            ]
        return payload


class _Provider:
    """第一次固定请求 read_document；工具成功后第二次给出最终回答。"""

    model = "fake-model"

    def __init__(self) -> None:
        self.calls = 0
        self.requests: list[list[dict[str, object]]] = []

    # 首轮返回读取工具调用、次轮返回答案，并记录每次请求。
    def complete(self, messages, tools=None):
        self.calls += 1
        request_messages = deepcopy(messages)
        self.requests.append(request_messages)

        if self.calls == 1:
            tool_call = SimpleNamespace(
                id="call-retry-1",
                function=SimpleNamespace(
                    name="read_document",
                    arguments=json.dumps({"name": "doc.md"}),
                ),
            )
            message = _Message(tool_calls=[tool_call])
            response_message = message.model_dump()
        else:
            message = _Message(content="final answer")
            response_message = message.model_dump()

        return SimpleNamespace(
            request={
                "model": self.model,
                "messages": request_messages,
                "tools": tools or [],
            },
            response={"choices": [{"message": response_message}]},
            message=message,
            latency_ms=1.0,
            model=self.model,
            usage={
                "prompt_tokens": 1,
                "completion_tokens": 1,
                "prompt_cache_hit_tokens": 0,
                "prompt_cache_miss_tokens": 1,
            },
        )


class _FailOnceTools:
    """第一次执行失败，第二次返回固定文档内容。"""

    schemas: list[dict[str, object]] = []

    def __init__(self) -> None:
        self.attempts = 0

    # 第一次执行抛出瞬时错误，后续执行返回固定内容。
    def execute(self, name, arguments):
        self.attempts += 1
        if self.attempts == 1:
            raise OSError("injected transient failure")
        return "document body"


class _AlwaysFailTools:
    """每次执行都失败，用于验证 retry exhausted 的终止语义。"""

    schemas: list[dict[str, object]] = []

    def __init__(self) -> None:
        self.attempts = 0

    # 记录尝试次数并持续抛错，供重试耗尽用例检查。
    def execute(self, name, arguments):
        self.attempts += 1
        raise OSError("injected persistent failure")


class _Tracer:
    """只在内存中收集事件，避免单元测试写 artifact。"""

    def __init__(self) -> None:
        self.records: list[dict[str, object]] = []

    # 将测试事件保存到内存，供用例检查执行顺序和内容。
    def log(self, event, payload, console=None):
        self.records.append({"event": event, "payload": payload})

    # 返回空统计，避免测试替身写入真实运行报告。
    def finalize(self, **kwargs):
        return {}


class ToolRetryTests(unittest.TestCase):
    def test_default_mode_preserves_step1_to_step9_error_writeback(self) -> None:
        """未显式启用 retry 时，保留旧的“错误写回模型再自行恢复”行为。"""
        provider = _Provider()
        tools = _AlwaysFailTools()
        tracer = _Tracer()
        agent = AgentLoop(
            provider=provider,
            tools=tools,
            tracer=tracer,
        )

        answer = agent.run("read doc", session=Session())
        events = [record["event"] for record in tracer.records]

        self.assertEqual(answer, "final answer")
        self.assertEqual(tools.attempts, 1)
        self.assertEqual(provider.calls, 2)
        self.assertIn("ERROR", events)
        self.assertNotIn("TOOL_RETRY", events)

    # 首次工具失败后重试成功，确认循环继续到最终回答。
    def test_fail_once_retries_once_and_continues_agent_loop(self) -> None:
        provider = _Provider()
        tools = _FailOnceTools()
        tracer = _Tracer()
        agent = AgentLoop(
            provider=provider,
            tools=tools,
            tracer=tracer,
            max_retry=1,
        )

        answer = agent.run("read doc", session=Session())
        events = [record["event"] for record in tracer.records]

        self.assertEqual(answer, "final answer")
        self.assertEqual(tools.attempts, 2)
        self.assertEqual(provider.calls, 2)
        self.assertIn("TOOL_ERROR", events)
        self.assertIn("TOOL_RETRY", events)
        self.assertIn("TOOL_RESULT", events)
        self.assertNotIn("TOOL_RETRY_EXHAUSTED", events)

    # 工具持续失败时只允许一次重试，随后停止本轮模型循环。
    def test_persistent_failure_stops_after_one_retry(self) -> None:
        provider = _Provider()
        tools = _AlwaysFailTools()
        tracer = _Tracer()
        agent = AgentLoop(
            provider=provider,
            tools=tools,
            tracer=tracer,
            max_retry=1,
        )

        with self.assertRaisesRegex(RuntimeError, "failed after 2 attempt"):
            agent.run("read doc", session=Session())

        events = [record["event"] for record in tracer.records]
        self.assertEqual(tools.attempts, 2)
        self.assertEqual(provider.calls, 1)
        self.assertEqual(events.count("TOOL_ERROR"), 2)
        self.assertEqual(events.count("TOOL_RETRY"), 1)
        self.assertIn("TOOL_RETRY_EXHAUSTED", events)


if __name__ == "__main__":
    unittest.main()
