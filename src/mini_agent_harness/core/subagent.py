"""
文件作用：
实现 Step 10.3 的最小 Reviewer Subagent 与母 Agent 工具适配层，用一次独立模型调用体验 delegation、context isolation 和 result return。

整体结构：
1）ReviewerSubagent：接收母 Agent 明确委派的 task / draft / evidence，构造完全独立的 system + user 两消息上下文；
2）ReviewerSubagent.run：不继承母 Session、不注入母 Context / Memory，也不给 Reviewer 任何工具，只执行一次专职审阅模型调用；
3）SubagentToolRegistry：在原本三个本地工具之外追加 review_answer，把母 Agent 的 function call 转成 ReviewerSubagent 调用；
4）Trace：记录 SUBAGENT_DELEGATE / SUBAGENT_CONTEXT / SUBAGENT_MODEL_REQUEST / SUBAGENT_MODEL_RESPONSE / SUBAGENT_RESULT，便于直接检查上下文隔离和结果返回。

职责边界：
- 母 Agent 决定何时读文档、何时形成草案、何时委派 Reviewer，以及如何吸收审阅结果形成最终答案；
- Reviewer 只审阅母 Agent 显式传入的草案与证据，不自动获得母 Agent 历史，也不能自行调用 read_document；
- 本步骤不接 Memory、Persistence、Managed Context、MCP 或多 Subagent 编排。
"""

from __future__ import annotations

from typing import Any, Protocol

from mini_agent_harness.core.tools import ToolRegistry
from mini_agent_harness.core.tracing import TraceLogger


REVIEWER_SYSTEM_PROMPT = """You are Reviewer, a minimal specialist subagent in an Agent Harness experiment.

Your only job is to review the delegated draft against the delegated evidence.
- You do not have access to the mother agent's conversation history.
- You do not have tools.
- Do not assume facts beyond the supplied evidence.
- Return a concise review in Chinese.
- Start with PASS or FAIL, then state the smallest necessary correction or confirmation.
"""


class _ProviderLike(Protocol):
    """限定 Reviewer 需要的最小模型 Provider 接口，便于离线测试替换。"""

    model: str

    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> Any: ...


class ReviewerSubagent:
    """一个只有独立上下文和专职审阅职责的最小 Reviewer Subagent。"""

    def __init__(
        self,
        *,
        provider: _ProviderLike,
        tracer: TraceLogger | Any | None = None,
        name: str = "reviewer",
    ) -> None:
        self.provider = provider
        self.tracer = tracer
        self.name = name

    def run(
        self,
        *,
        task: str,
        draft: str,
        evidence: str,
    ) -> str:
        """只用显式委派内容构造两条消息执行审阅，不继承母 Agent 的任何消息。"""
        delegated_user = (
            "Delegated review task:\n"
            f"{task.strip()}\n\n"
            "Draft to review:\n"
            f"{draft.strip()}\n\n"
            "Evidence supplied by Mother Agent:\n"
            f"{evidence.strip()}"
        )
        messages = [
            {"role": "system", "content": REVIEWER_SYSTEM_PROMPT},
            {"role": "user", "content": delegated_user},
        ]

        self._log(
            "SUBAGENT_CONTEXT",
            {
                "subagent": self.name,
                "mother_history_inherited": False,
                "mother_memory_inherited": False,
                "message_count": len(messages),
                "roles": [item["role"] for item in messages],
                "tool_count": 0,
                "draft_chars": len(draft),
                "evidence_chars": len(evidence),
                "messages": messages,
            },
            console={
                "subagent": self.name,
                "mother_history_inherited": False,
                "message_count": len(messages),
                "roles": ["system", "user"],
                "tool_count": 0,
                "draft_chars": len(draft),
                "evidence_chars": len(evidence),
            },
        )

        result = self.provider.complete(messages=messages, tools=None)
        self._log(
            "SUBAGENT_MODEL_REQUEST",
            {
                "subagent": self.name,
                "request": result.request,
            },
            console={
                "subagent": self.name,
                "message_count": len(messages),
                "tools": [],
            },
        )
        self._log(
            "SUBAGENT_MODEL_RESPONSE",
            {
                "subagent": self.name,
                "response": result.response,
                "latency_ms": result.latency_ms,
                "usage": result.usage,
                "model": result.model,
            },
            console={
                "subagent": self.name,
                "model": result.model,
                "latency_ms": round(result.latency_ms, 2),
                "usage": result.usage,
                "content": result.message.content,
            },
        )

        review = (result.message.content or "").strip()
        self._log(
            "SUBAGENT_RESULT",
            {
                "subagent": self.name,
                "result": review,
                "result_chars": len(review),
            },
        )
        return review

    def _log(
        self,
        event: str,
        payload: dict[str, Any],
        console: dict[str, Any] | None = None,
    ) -> None:
        """有 tracer 时固化 Subagent 边界事件；无 tracer 时保持组件可独立测试。"""
        if self.tracer is not None:
            self.tracer.log(event, payload, console=console)


class SubagentToolRegistry:
    """混合工具注册表：保留本地工具，并追加一个把审阅任务委派给 Reviewer 的工具。"""

    def __init__(
        self,
        *,
        local_tools: ToolRegistry | Any,
        reviewer: ReviewerSubagent,
        tracer: TraceLogger | Any | None = None,
    ) -> None:
        self.local_tools = local_tools
        self.reviewer = reviewer
        self.tracer = tracer
        self._schemas = [*local_tools.schemas, self._review_schema()]

    @property
    def schemas(self) -> list[dict[str, Any]]:
        """返回原本本地工具加 review_answer 委派工具的 function calling schema。"""
        return self._schemas

    def execute(self, name: str, arguments: dict[str, Any]) -> str:
        """review_answer 走 Reviewer Subagent；其她工具继续走 Local ToolRegistry。"""
        if name != "review_answer":
            return self.local_tools.execute(name, arguments)

        task = str(arguments["task"])
        draft = str(arguments["draft"])
        evidence = str(arguments["evidence"])
        self._log(
            "SUBAGENT_DELEGATE",
            {
                "subagent": self.reviewer.name,
                "tool": "review_answer",
                "task": task,
                "draft": draft,
                "evidence": evidence,
                "draft_chars": len(draft),
                "evidence_chars": len(evidence),
                "reason": "母 Agent 显式把草案与必要证据委派给隔离 Reviewer 审阅。",
            },
            console={
                "subagent": self.reviewer.name,
                "tool": "review_answer",
                "draft_chars": len(draft),
                "evidence_chars": len(evidence),
            },
        )
        return self.reviewer.run(
            task=task,
            draft=draft,
            evidence=evidence,
        )

    @staticmethod
    def _review_schema() -> dict[str, Any]:
        """定义母 Agent 用来显式委派 Reviewer 的最小工具 schema。"""
        return {
            "type": "function",
            "function": {
                "name": "review_answer",
                "description": (
                    "Delegate a draft answer and its supporting evidence to an isolated "
                    "Reviewer subagent before finalizing the user-facing answer."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "task": {
                            "type": "string",
                            "description": "What the Reviewer should check.",
                        },
                        "draft": {
                            "type": "string",
                            "description": "The Mother Agent's draft answer to review.",
                        },
                        "evidence": {
                            "type": "string",
                            "description": (
                                "Only the evidence needed for the Reviewer to verify the draft."
                            ),
                        },
                    },
                    "required": ["task", "draft", "evidence"],
                    "additionalProperties": False,
                },
            },
        }

    def _log(
        self,
        event: str,
        payload: dict[str, Any],
        console: dict[str, Any] | None = None,
    ) -> None:
        """统一记录母 Agent 到 Reviewer 的委派边界。"""
        if self.tracer is not None:
            self.tracer.log(event, payload, console=console)
