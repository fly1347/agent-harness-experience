"""
文件作用：
提供 Session 与 ContextBuilder 的最小回归测试，不访问真实模型和网络；用于确认多轮历史保存、Session 隔离和三种上下文策略的基本行为没有被后续修改破坏。

整体结构：
1）FakeMessage / FakeProvider / FakeTools / FakeTracer：替代真实 provider、工具和 Trace，固定测试输入输出；
2）SessionTests：验证同一 Session 复用历史、新 Session 不继承旧历史、不同 Session 状态互相独立；
3）ContextBuilderTests：验证 full_history、last_n、managed 的上下文结果；
4）AgentLoop + last_n 联合测试：确认 working context 可以裁剪，但 Session 中的完整历史不会被同步裁掉。
"""

from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace
import unittest

from mini_agent_harness.core.agent_loop import AgentLoop, SYSTEM_PROMPT
from mini_agent_harness.core.session import Session


class FakeMessage:
    def __init__(self, content: str) -> None:
        self.role = "assistant"
        self.content = content
        self.tool_calls = None

    def model_dump(self, exclude_none: bool = False) -> dict[str, str]:
        return {
            "role": self.role,
            "content": self.content,
        }


class FakeProvider:
    model = "fake-model"

    def __init__(self) -> None:
        self.requests: list[list[dict[str, object]]] = []

    def complete(self, messages, tools=None):
        request_messages = deepcopy(messages)
        self.requests.append(request_messages)

        message = FakeMessage(
            f"answer-{len(self.requests)}"
        )

        return SimpleNamespace(
            request={
                "model": self.model,
                "messages": request_messages,
                "tools": tools or [],
            },
            response={
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": message.content,
                        }
                    }
                ]
            },
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


class FakeTools:
    schemas: list[dict[str, object]] = []

    def execute(self, name, arguments):
        raise AssertionError("No tool call expected in this test.")


class FakeTracer:
    def log(self, event, payload, console=None):
        return None

    def finalize(self, **kwargs):
        return {}


class SessionTests(unittest.TestCase):
    def make_agent(self, provider: FakeProvider) -> AgentLoop:
        return AgentLoop(
            provider=provider,
            tools=FakeTools(),
            tracer=FakeTracer(),
        )

    def test_same_session_reuses_previous_turn_history(self) -> None:
        provider = FakeProvider()
        agent = self.make_agent(provider)
        session = Session()

        agent.run("first turn", session=session)
        agent.run("second turn", session=session)

        self.assertEqual(
            [m["role"] for m in provider.requests[0]],
            ["system", "user"],
        )
        self.assertEqual(
            [m["role"] for m in provider.requests[1]],
            ["system", "user", "assistant", "user"],
        )
        self.assertEqual(
            provider.requests[1][1]["content"],
            "first turn",
        )
        self.assertEqual(
            provider.requests[1][3]["content"],
            "second turn",
        )

    def test_new_session_does_not_inherit_old_history(self) -> None:
        provider = FakeProvider()
        agent = self.make_agent(provider)

        first = Session()
        agent.run("old history", session=first)

        second = Session()
        agent.run("fresh turn", session=second)

        self.assertEqual(
            [m["role"] for m in provider.requests[1]],
            ["system", "user"],
        )
        self.assertEqual(
            provider.requests[1][1]["content"],
            "fresh turn",
        )

    def test_session_instances_have_independent_state(self) -> None:
        first = Session()
        second = Session()

        first.add_message(
            {"role": "user", "content": "only in first"}
        )

        self.assertNotEqual(first.session_id, second.session_id)
        self.assertEqual(len(first.messages), 1)
        self.assertEqual(second.messages, [])



class ContextBuilderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.history = [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "early question"},
            {"role": "assistant", "content": "early answer about retrieval and evidence"},
            {"role": "user", "content": "middle question"},
            {"role": "assistant", "content": "middle answer about generation and citation"},
            {"role": "user", "content": "current question"},
        ]

    def test_full_history_keeps_canonical_history(self) -> None:
        from mini_agent_harness.core.context import ContextBuilder

        built = ContextBuilder("full_history").build(self.history)
        self.assertEqual(built.messages, self.history)
        self.assertEqual(built.request_message_count, len(self.history))

    def test_last_n_drops_early_turns(self) -> None:
        from mini_agent_harness.core.context import ContextBuilder

        built = ContextBuilder("last_n", last_n=2).build(self.history)
        contents = [m.get("content") for m in built.messages]
        self.assertNotIn("early question", contents)
        self.assertIn("current question", contents)

    def test_managed_context_injects_summary(self) -> None:
        from mini_agent_harness.core.context import ContextBuilder

        built = ContextBuilder(
            "managed",
            managed_recent_n=2,
            summary_trigger_messages=3,
        ).build(self.history)
        self.assertTrue(built.summary_injected)
        self.assertIn("early question", built.summary)
        self.assertEqual(built.messages[1]["role"], "system")
        self.assertIn("Conversation summary", built.messages[1]["content"])

    def test_managed_context_marks_active_memory_as_current_authority(self) -> None:
        """历史摘要与当前记忆冲突时，请求中应明确标出 Active Memory 的当前状态优先级。"""
        from mini_agent_harness.core.context import ContextBuilder

        history = [
            {"role": "system", "content": "system"},
            {
                "role": "user",
                "content": "记住：当前重点是评估对象分层和指标边界。",
            },
            {
                "role": "assistant",
                "content": "已记录旧重点。",
            },
            {"role": "user", "content": "较早问题一"},
            {"role": "assistant", "content": "较早回答一"},
            {"role": "user", "content": "较早问题二"},
            {"role": "assistant", "content": "较早回答二"},
            {"role": "user", "content": "较早问题三"},
            {"role": "assistant", "content": "较早回答三"},
            {"role": "user", "content": "当前实验重点是什么？"},
        ]

        built = ContextBuilder(
            "managed",
            managed_recent_n=2,
            summary_trigger_messages=3,
        ).build(
            history,
            active_memory=[
                "decision:experiment_focus = Context Management + 评估信度"
            ],
        )

        system_contents = [
            str(message.get("content", ""))
            for message in built.messages
            if message.get("role") == "system"
        ]
        summary_index = next(
            i
            for i, content in enumerate(system_contents)
            if "Conversation summary" in content
        )
        memory_index = next(
            i
            for i, content in enumerate(system_contents)
            if "Active long-term memory" in content
        )
        memory_message = system_contents[memory_index]

        self.assertLess(summary_index, memory_index)
        self.assertIn("authoritative current state", memory_message)
        self.assertIn("use this active memory as the current value", memory_message)
        self.assertIn("Context Management + 评估信度", memory_message)

    def test_agent_loop_uses_trimmed_context_without_trimming_session(self) -> None:
        from mini_agent_harness.core.context import ContextBuilder

        provider = FakeProvider()
        agent = AgentLoop(
            provider=provider,
            tools=FakeTools(),
            tracer=FakeTracer(),
            context_builder=ContextBuilder("last_n", last_n=2),
        )
        session = Session()

        agent.run("first turn", session=session)
        agent.run("second turn", session=session)

        self.assertEqual(
            [m["content"] for m in provider.requests[1]],
            [SYSTEM_PROMPT, "second turn"],
        )
        self.assertEqual(
            [m["role"] for m in session.messages],
            ["system", "user", "assistant", "user", "assistant"],
        )


if __name__ == "__main__":
    unittest.main()
