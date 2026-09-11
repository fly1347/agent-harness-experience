"""
文件作用：
提供项目统一的 OpenAI-compatible 模型调用适配层，把 baseline 和 AgentLoop 都收口到同一套 client，并统一返回请求、响应、usage 和 latency 元数据。

整体结构：
1）ModelCallResult：统一承载 request、response、message、model、usage 和 latency_ms；
2）OpenAICompatibleProvider.__init__：从环境变量读取 API key、base URL、模型和 temperature，并创建 OpenAI client；
3）complete：发送 Chat Completions 请求，可附带 tools，供 AgentLoop 使用；
4）chat：封装单条 user message 的纯 LLM baseline，复用 complete。

环境变量：
- LLM_API_KEY
- LLM_BASE_URL（可选）
- LLM_MODEL
- LLM_TEMPERATURE（默认 0）
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any

from openai import OpenAI


@dataclass
class ModelCallResult:
    request: dict[str, Any]
    response: dict[str, Any]
    message: Any
    latency_ms: float
    model: str
    usage: dict[str, Any] | None


class OpenAICompatibleProvider:
    # 从环境读取模型配置并校验必填项，创建供后续调用使用的客户端。
    def __init__(self) -> None:
        api_key = os.getenv("LLM_API_KEY")
        base_url = os.getenv("LLM_BASE_URL") or None
        model = os.getenv("LLM_MODEL")

        if not api_key:
            raise RuntimeError("LLM_API_KEY is not configured.")
        if not model:
            raise RuntimeError("LLM_MODEL is not configured.")

        self.model = model
        self.temperature = float(os.getenv("LLM_TEMPERATURE", "0"))

        kwargs: dict[str, Any] = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url

        self.client = OpenAI(**kwargs)

    # 发送一次可选带 tools 的 Chat Completions 请求，并统一整理模型返回与 usage 元数据。
    def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> ModelCallResult:
        request: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "extra_body": {
                "thinking": {
                    "type": "disabled",
                }
            },
        }

        if tools:
            request["tools"] = tools
            request["tool_choice"] = "auto"

        started = time.perf_counter()
        response = self.client.chat.completions.create(**request)
        latency_ms = (time.perf_counter() - started) * 1000

        return ModelCallResult(
            request=request,
            response=response.model_dump(),
            message=response.choices[0].message,
            latency_ms=latency_ms,
            model=response.model,
            usage=response.usage.model_dump() if response.usage else None,
        )

    # 把单条 user message 交给 complete，执行不带 AgentLoop 的纯 LLM baseline。
    def chat(self, user_input: str) -> ModelCallResult:
        return self.complete(
            messages=[{"role": "user", "content": user_input}]
        )
