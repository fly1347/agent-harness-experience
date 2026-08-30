"""
文件作用：
提供 Mini Agent Harness 的统一命令行入口，用于手动体验 Step 2～4：纯 LLM baseline、单轮 Agent Loop 和可复用 Session 的多轮交互。

整体结构：
1）baseline：默认直接调用 provider.chat，不进入 AgentLoop；
2）--agent：组装 ToolRegistry、TraceLogger、AgentLoop，执行一次带工具调用的 Agent 轮次；
3）--agent --session：创建一个内存 Session，在交互循环中持续复用历史，并为每轮单独生成 Trace；
4）main：解析参数、加载环境变量，并根据运行模式选择对应链路。
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

from mini_agent_harness.core.agent_loop import AgentLoop
from mini_agent_harness.core.session import Session
from mini_agent_harness.core.tools import ToolRegistry
from mini_agent_harness.core.tracing import TraceLogger
from mini_agent_harness.providers.openai_compatible import (
    OpenAICompatibleProvider,
)


def main() -> None:
    """解析运行模式并组装 baseline、单轮 Agent 或多轮 Session 所需组件。"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "prompt",
        nargs="?",
        default="用一句话解释什么是 Agent Harness。",
    )
    parser.add_argument(
        "--agent",
        action="store_true",
        help="Run through the Agent Loop with native tools.",
    )
    parser.add_argument(
        "--session",
        action="store_true",
        help="Run an interactive in-memory multi-turn session.",
    )
    args = parser.parse_args()

    if args.session and not args.agent:
        parser.error("--session requires --agent")

    load_dotenv()

    provider = OpenAICompatibleProvider()

    if not args.agent:
        result = provider.chat(args.prompt)

        print("\n===== MODEL REQUEST =====")
        print(json.dumps(result.request, ensure_ascii=False, indent=2))

        print("\n===== MODEL RESPONSE =====")
        print(json.dumps(result.response, ensure_ascii=False, indent=2))

        print("\n===== BASELINE SUMMARY =====")
        print(f"model: {result.model}")
        print(f"latency_ms: {result.latency_ms:.2f}")
        print(
            "usage:",
            json.dumps(result.usage, ensure_ascii=False),
        )

        print("\n===== ANSWER =====")
        print(result.message.content or "")
        return

    project_root = Path(__file__).resolve().parents[2]

    tools = ToolRegistry(
        fixtures_dir=project_root / "fixtures",
        notes_file=project_root / "artifacts" / "notes.md",
    )

    if args.session:
        session = Session()
        turn = 0

        print("\n===== SESSION STARTED =====")
        print(f"session_id: {session.session_id}")
        print("Type /exit or /quit to stop.")

        while True:
            try:
                user_input = input("\nYou> ").strip()
            except EOFError:
                break

            if user_input.lower() in {"/exit", "/quit"}:
                break
            if not user_input:
                continue

            turn += 1
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            trace_path = (
                project_root
                / "artifacts"
                / "traces"
                / (
                    f"step4-session-{session.session_id[:8]}"
                    f"-turn{turn:02d}-{timestamp}.jsonl"
                )
            )

            tracer = TraceLogger(trace_path)
            agent = AgentLoop(
                provider=provider,
                tools=tools,
                tracer=tracer,
            )

            answer = agent.run(
                user_input,
                session=session,
            )

            print("\n===== AGENT ANSWER =====")
            print(answer)

            print("\n===== SESSION STATE =====")
            print(f"session_id: {session.session_id}")
            print(f"messages: {len(session.messages)}")

            print("\n===== TRACE FILES =====")
            print(f"JSONL: {trace_path}")
            print(f"MD:    {tracer.md_path}")

        return

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    trace_path = (
        project_root
        / "artifacts"
        / "traces"
        / f"step3-agent-loop-{timestamp}.jsonl"
    )

    tracer = TraceLogger(trace_path)

    agent = AgentLoop(
        provider=provider,
        tools=tools,
        tracer=tracer,
    )

    answer = agent.run(args.prompt)

    print("\n===== AGENT ANSWER =====")
    print(answer)

    print("\n===== TRACE FILES =====")
    print(f"JSONL: {trace_path}")
    print(f"MD:    {tracer.md_path}")


if __name__ == "__main__":
    main()
