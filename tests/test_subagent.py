"""
文件作用：
离线验证 Step 10.3 Reviewer Subagent 的三个核心语义：母 Agent 能显式委派、Reviewer 不继承母上下文、Reviewer 结果能作为工具结果返回母 Agent。

整体结构：
1）FakeProvider 固定返回 Reviewer 审阅结果并记录请求；
2）ReviewerSubagent 测试只构造 system + delegated user 两条消息，且 tools=None；
3）SubagentToolRegistry 测试 review_answer 会走 Reviewer，而原 read_document 仍走 Local Tool；
4）Trace 事件检查 delegation / context / result 都被固化。
"""

from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace
import unittest

from mini_agent_harness.core.subagent import ReviewerSubagent, SubagentToolRegistry


class _Message:
    """构造 ReviewerSubagent 所需的最小 assistant message。"""

    def __init__(self, content: str) -> None:
        self.content = content


class _Provider:
    """固定返回 PASS，并保存 Reviewer 实际收到的 messages / tools。"""

    model = "fake-reviewer-model"

    def __init__(self) -> None:
        self.requests: list[tuple[list[dict[str, object]], object]] = []

    # 保存请求快照并返回固定回复，供测试检查模型实际收到的上下文。
    def complete(self, messages, tools=None):
        frozen = deepcopy(messages)
        self.requests.append((frozen, tools))
        message = _Message("PASS：草案与证据一致。")
        return SimpleNamespace(
            request={"model": self.model, "messages": frozen},
            response={"choices": [{"message": {"role": "assistant", "content": message.content}}]},
            message=message,
            latency_ms=2.0,
            model=self.model,
            usage={
                "prompt_tokens": 10,
                "completion_tokens": 4,
                "prompt_cache_hit_tokens": 0,
                "prompt_cache_miss_tokens": 10,
            },
        )


class _Tracer:
    """只在内存中保存事件，避免单测写 artifact。"""

    def __init__(self) -> None:
        self.records: list[dict[str, object]] = []

    # 将测试事件保存到内存，供用例检查执行顺序和内容。
    def log(self, event, payload, console=None):
        self.records.append({"event": event, "payload": payload})


class _LocalTools:
    """提供一个最小本地 read_document，验证非 Subagent 工具仍保留原路由。"""

    schemas = [
        {
            "type": "function",
            "function": {
                "name": "read_document",
                "description": "read",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, object]]] = []

    # 记录本地调用并返回文档，检查非审阅工具是否保留本地路由。
    def execute(self, name, arguments):
        self.calls.append((name, arguments))
        return "local document"


class ReviewerSubagentTests(unittest.TestCase):
    # 确认 Reviewer 只收到显式委派消息，不继承母上下文或工具。
    def test_reviewer_receives_only_isolated_delegated_context(self) -> None:
        provider = _Provider()
        tracer = _Tracer()
        reviewer = ReviewerSubagent(provider=provider, tracer=tracer)

        result = reviewer.run(
            task="检查草案是否忠于证据",
            draft="检索层看召回。",
            evidence="检索层看问题能不能找回相关候选。",
        )

        messages, tools = provider.requests[0]
        events = [record["event"] for record in tracer.records]
        context = next(
            record["payload"]
            for record in tracer.records
            if record["event"] == "SUBAGENT_CONTEXT"
        )

        self.assertEqual(result, "PASS：草案与证据一致。")
        self.assertEqual([m["role"] for m in messages], ["system", "user"])
        self.assertEqual(len(messages), 2)
        self.assertIsNone(tools)
        self.assertFalse(context["mother_history_inherited"])
        self.assertFalse(context["mother_memory_inherited"])
        self.assertEqual(context["tool_count"], 0)
        self.assertIn("SUBAGENT_MODEL_REQUEST", events)
        self.assertIn("SUBAGENT_MODEL_RESPONSE", events)
        self.assertIn("SUBAGENT_RESULT", events)

    # 确认审阅请求交给 Reviewer，原有文档工具仍走本地执行。
    def test_review_answer_delegates_and_other_tools_remain_local(self) -> None:
        provider = _Provider()
        tracer = _Tracer()
        reviewer = ReviewerSubagent(provider=provider, tracer=tracer)
        local = _LocalTools()
        registry = SubagentToolRegistry(
            local_tools=local,
            reviewer=reviewer,
            tracer=tracer,
        )

        review = registry.execute(
            "review_answer",
            {
                "task": "check",
                "draft": "draft",
                "evidence": "evidence",
            },
        )
        local_result = registry.execute("read_document", {"name": "doc.md"})
        names = [item["function"]["name"] for item in registry.schemas]
        events = [record["event"] for record in tracer.records]

        self.assertEqual(review, "PASS：草案与证据一致。")
        self.assertEqual(local_result, "local document")
        self.assertIn("review_answer", names)
        self.assertEqual(local.calls, [("read_document", {"name": "doc.md"})])
        self.assertIn("SUBAGENT_DELEGATE", events)


if __name__ == "__main__":
    unittest.main()
