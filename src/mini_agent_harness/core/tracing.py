"""
文件作用：
记录 Agent Harness 的完整运行 Trace，并把原始 JSONL 事件整理成便于阅读的 Markdown；同时汇总每轮 Memory、Context、Model、Tool 的执行过程，以及 token、缓存、耗时和估算成本。

整体结构：
1）_pricing_snapshot / _estimate_cost：按当前实验价格口径选择计费档并估算单轮成本；
2）TraceLogger.log：把 MEMORY、CONTEXT_BUILD、MODEL、TOOL，以及 Retry / MCP / Subagent 边界事件即时固化到内存和 JSONL；
3）TraceLogger.finalize：生成 RUN_SUMMARY；Step 10.3 时额外合并 Reviewer 的模型调用、token 和 latency，避免嵌套调用漏计；
4）_execution_rows / _markdown_table：把原始事件转换成执行流程和统计表；
5）_write_markdown：按事件类型展开 Memory、Model、Tool 与最终回答，生成可人工复核的 Trace 报告。
"""

from __future__ import annotations

import json
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo


def _pricing_snapshot() -> dict[str, Any]:
    now_bj = datetime.now(ZoneInfo("Asia/Shanghai"))
    hour = now_bj.hour

    peak = (9 <= hour < 12) or (14 <= hour < 18)

    if peak:
        tier = "peak"
        cache_hit = 0.10
        cache_miss = 3.00
        output = 9.00
    else:
        tier = "off_peak"
        cache_hit = 0.05
        cache_miss = 1.50
        output = 4.50

    return {
        "pricing_date": "2026-08-28",
        "pricing_timezone": "Asia/Shanghai",
        "pricing_tier": tier,
        "cache_hit_cny_per_million": cache_hit,
        "cache_miss_cny_per_million": cache_miss,
        "output_cny_per_million": output,
    }


def _estimate_cost(
    cache_hit_tokens: int,
    cache_miss_tokens: int,
    output_tokens: int,
    pricing: dict[str, Any],
) -> float:
    return (
        cache_hit_tokens
        * pricing["cache_hit_cny_per_million"]
        + cache_miss_tokens
        * pricing["cache_miss_cny_per_million"]
        + output_tokens
        * pricing["output_cny_per_million"]
    ) / 1_000_000


class TraceLogger:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.md_path = path.with_suffix(".md")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.records: list[dict[str, Any]] = []

    def log(
        self,
        event: str,
        payload: dict[str, Any],
        console: dict[str, Any] | None = None,
    ) -> None:
        """深拷贝固化一次运行事件，同时写入内存、JSONL，并按需输出精简终端信息。"""
        # 记录时立即做快照，避免可变消息列表继续增长后污染旧的 MODEL_REQUEST。
        frozen_payload = deepcopy(payload)

        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": event,
            "payload": frozen_payload,
        }
        self.records.append(record)

        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        shown = console if console is not None else frozen_payload
        print(f"\n===== {event} =====")
        print(json.dumps(shown, ensure_ascii=False, indent=2))

    def finalize(
        self,
        *,
        user_input: str,
        model: str,
        model_calls: int,
        tool_calls: int,
        prompt_tokens: int,
        completion_tokens: int,
        cache_hit_tokens: int,
        cache_miss_tokens: int,
        model_ms: float,
        tool_ms: float,
        end_to_end_ms: float,
    ) -> dict[str, Any]:
        """汇总一次用户轮次的 token、耗时和成本，写入 RUN_SUMMARY 并生成 Markdown Trace。"""
        pricing = _pricing_snapshot()

        # Step 10.3 的 Reviewer 是嵌套在 review_answer 工具调用里的独立模型调用。
        # AgentLoop 只统计主 Agent，所以这里从 Trace 事件把 Subagent usage / latency 合并进总计，
        # 同时保留 main/subagent 分项，避免“总模型调用数”和真实 API 请求数不一致。
        subagent_records = [
            record["payload"]
            for record in self.records
            if record["event"] == "SUBAGENT_MODEL_RESPONSE"
        ]
        subagent_model_calls = len(subagent_records)
        subagent_prompt_tokens = sum(
            int((item.get("usage") or {}).get("prompt_tokens") or 0)
            for item in subagent_records
        )
        subagent_completion_tokens = sum(
            int((item.get("usage") or {}).get("completion_tokens") or 0)
            for item in subagent_records
        )
        subagent_cache_hit_tokens = sum(
            int((item.get("usage") or {}).get("prompt_cache_hit_tokens") or 0)
            for item in subagent_records
        )
        subagent_cache_miss_tokens = sum(
            int((item.get("usage") or {}).get("prompt_cache_miss_tokens") or 0)
            for item in subagent_records
        )
        subagent_model_ms = sum(
            float(item.get("latency_ms") or 0.0)
            for item in subagent_records
        )

        total_prompt_tokens = prompt_tokens + subagent_prompt_tokens
        total_completion_tokens = completion_tokens + subagent_completion_tokens
        total_cache_hit_tokens = cache_hit_tokens + subagent_cache_hit_tokens
        total_cache_miss_tokens = cache_miss_tokens + subagent_cache_miss_tokens
        total_model_ms = model_ms + subagent_model_ms

        # review_answer 的 Tool latency 包含 Reviewer 模型等待时间；扣掉嵌套模型时间后，
        # tool_local 才近似表示真正的本地工具/封装开销，避免 timing 双重计算。
        tool_local_ms = max(0.0, tool_ms - subagent_model_ms)
        harness_ms = max(
            0.0,
            end_to_end_ms - total_model_ms - tool_local_ms,
        )

        estimated_cost_cny = _estimate_cost(
            cache_hit_tokens=total_cache_hit_tokens,
            cache_miss_tokens=total_cache_miss_tokens,
            output_tokens=total_completion_tokens,
            pricing=pricing,
        )

        summary = {
            "model": model,
            "thinking": "disabled",
            "model_calls": model_calls + subagent_model_calls,
            "main_model_calls": model_calls,
            "subagent_model_calls": subagent_model_calls,
            "tool_calls": tool_calls,
            "tokens": {
                "prompt": total_prompt_tokens,
                "completion": total_completion_tokens,
                "total": total_prompt_tokens + total_completion_tokens,
                "cache_hit": total_cache_hit_tokens,
                "cache_miss": total_cache_miss_tokens,
            },
            "timing_ms": {
                "model": round(total_model_ms, 2),
                "main_model": round(model_ms, 2),
                "subagent_model": round(subagent_model_ms, 2),
                "tool": round(tool_local_ms, 2),
                "tool_wall_including_subagent": round(tool_ms, 2),
                "harness_local": round(harness_ms, 2),
                "end_to_end": round(end_to_end_ms, 2),
            },
            "pricing": pricing,
            "estimated_cost_cny": round(estimated_cost_cny, 8),
        }

        self.log("RUN_SUMMARY", summary)
        self._write_markdown(user_input, summary)

        return summary

    def _execution_rows(
        self,
        summary: dict[str, Any],
    ) -> list[list[str]]:
        rows: list[list[str]] = []

        request_by_step: dict[int, dict[str, Any]] = {}
        tool_call_by_id: dict[str, dict[str, Any]] = {}

        tool_index = 0
        retry_index = 0
        memory_index = 0
        mcp_index = 0
        subagent_index = 0

        for record in self.records:
            event = record["event"]
            payload = record["payload"]

            if event == "MCP_CONNECT":
                mcp_index += 1
                rows.append([
                    f"MCP{mcp_index}",
                    "MCP",
                    (
                        f"stdio connect → `{payload.get('server_info', {}).get('name') or '?'}` "
                        f"· protocol `{payload.get('protocol_version') or '?'}`"
                    ),
                    "—",
                    "—",
                    "—",
                    "—",
                ])

            elif event == "MCP_LIST_TOOLS":
                mcp_index += 1
                names = [
                    item.get("function", {}).get("name", "?")
                    for item in payload.get("tools", [])
                ]
                rows.append([
                    f"MCP{mcp_index}",
                    "MCP",
                    f"list_tools → `{', '.join(names)}`",
                    "—",
                    "—",
                    "—",
                    "—",
                ])

            elif event == "MCP_CALL_TOOL":
                mcp_index += 1
                rows.append([
                    f"MCP{mcp_index}",
                    "MCP",
                    f"call_tool → `{payload.get('name', '?')}` over stdio",
                    "—",
                    "—",
                    "—",
                    "—",
                ])

            elif event == "MCP_RESULT":
                mcp_index += 1
                rows.append([
                    f"MCP{mcp_index}",
                    "MCP",
                    (
                        f"tool result ← `{payload.get('name', '?')}` · "
                        f"**{int(payload.get('result_chars') or 0):,} chars** · "
                        f"error={payload.get('is_error', False)}"
                    ),
                    "—",
                    "—",
                    "—",
                    "—",
                ])

            elif event == "SUBAGENT_DELEGATE":
                subagent_index += 1
                rows.append([
                    f"D{subagent_index}",
                    "Mother Agent",
                    (
                        f"delegate → `{payload.get('subagent', 'reviewer')}` "
                        f"via `{payload.get('tool', 'review_answer')}` · "
                        f"draft {int(payload.get('draft_chars') or 0):,} chars · "
                        f"evidence {int(payload.get('evidence_chars') or 0):,} chars"
                    ),
                    "—",
                    "—",
                    "—",
                    "—",
                ])

            elif event == "SUBAGENT_MODEL_RESPONSE":
                usage = payload.get("usage") or {}
                hit = int(usage.get("prompt_cache_hit_tokens") or 0)
                miss = int(usage.get("prompt_cache_miss_tokens") or 0)
                prompt = int(usage.get("prompt_tokens") or 0)
                completion = int(usage.get("completion_tokens") or 0)
                rows.append([
                    f"S{subagent_index}",
                    "Reviewer",
                    "`isolated review`",
                    "2",
                    f"{prompt:,} ({hit:,}/{miss:,})",
                    f"{completion:,}",
                    f"{float(payload.get('latency_ms') or 0.0) / 1000:.3f} s",
                ])

            elif event == "SUBAGENT_RESULT":
                rows.append([
                    f"S{subagent_index}R",
                    "Reviewer",
                    f"result → Mother Agent · **{int(payload.get('result_chars') or 0):,} chars**",
                    "—",
                    "—",
                    "—",
                    "—",
                ])

            elif event == "MEMORY_READ":
                memory_index += 1
                keys = [
                    item.get("key", "?")
                    for item in payload.get("records", [])
                ]
                operation = (
                    f"读取当前有效记忆 (`retrieve_active`) → **{payload.get('count', 0)}**"
                    + (f" · `{', '.join(keys)}`" if keys else "")
                )
                rows.append([
                    f"Mem{memory_index}",
                    "Memory",
                    operation,
                    "—",
                    "—",
                    "—",
                    "—",
                ])

            elif event == "MEMORY_INJECT":
                memory_index += 1
                keys = [
                    item.get("key", "?")
                    for item in payload.get("records", [])
                ]
                rows.append([
                    f"Mem{memory_index}",
                    "Memory",
                    f"注入当前有效记忆 (`inject`) → `{', '.join(keys)}`",
                    "—",
                    "—",
                    "—",
                    "—",
                ])

            elif event == "MEMORY_WRITE":
                memory_index += 1
                item = payload.get("record") or {}
                rows.append([
                    f"Mem{memory_index}",
                    "Memory",
                    (
                        f"写入长期记忆 (`write`) → `{item.get('key', '?')}` = "
                        f"`{item.get('value', '')}`"
                    ),
                    "—",
                    "—",
                    "—",
                    "—",
                ])

            elif event == "MEMORY_SUPERSEDE":
                memory_index += 1
                old = payload.get("previous") or {}
                new = payload.get("record") or {}
                rows.append([
                    f"Mem{memory_index}",
                    "Memory",
                    (
                        f"替换旧记忆 (`supersede`) → `{old.get('value', '')}` → "
                        f"`{new.get('value', '')}`"
                    ),
                    "—",
                    "—",
                    "—",
                    "—",
                ])

            elif event == "MODEL_REQUEST":
                request_by_step[payload["step"]] = payload["request"]

            elif event == "TOOL_CALL":
                tool_call_by_id[payload["tool_call_id"]] = payload

            elif event == "MODEL_RESPONSE":
                step = payload["step"]
                usage = payload.get("usage") or {}
                response = payload.get("response") or {}

                request = request_by_step.get(step, {})
                message_count = len(request.get("messages", []))

                hit = int(
                    usage.get("prompt_cache_hit_tokens") or 0
                )
                miss = int(
                    usage.get("prompt_cache_miss_tokens") or 0
                )
                prompt = int(usage.get("prompt_tokens") or 0)
                completion = int(
                    usage.get("completion_tokens") or 0
                )

                message = (
                    response.get("choices", [{}])[0]
                    .get("message", {})
                )
                calls = message.get("tool_calls") or []

                if calls:
                    actions: list[str] = []

                    for call in calls:
                        fn = call.get("function", {})
                        name = fn.get("name", "?")

                        try:
                            args = json.loads(
                                fn.get("arguments") or "{}"
                            )
                        except json.JSONDecodeError:
                            args = {}

                        if name == "read_document":
                            doc = args.get("name", "?")
                            actions.append(
                                f'`tool → read_document("{doc}")`'
                            )
                        elif name == "list_documents":
                            actions.append(
                                "`tool → list_documents`"
                            )
                        elif name == "save_note":
                            actions.append(
                                "`tool → save_note`"
                            )
                        else:
                            actions.append(f"`tool → {name}`")

                    operation = "<br>".join(actions)
                else:
                    operation = "`final answer`"

                rows.append([
                    f"M{step}",
                    "Model",
                    operation,
                    str(message_count),
                    f"{prompt:,} ({hit:,}/{miss:,})",
                    f"{completion:,}",
                    f"{payload['latency_ms'] / 1000:.3f} s",
                ])

            elif event == "TOOL_ERROR":
                tool_index += 1
                rows.append([
                    f"T{tool_index}",
                    "Tool",
                    (
                        f"`{payload.get('name', '?')}` attempt "
                        f"**{payload.get('attempt', '?')}** → ERROR: "
                        f"`{payload.get('error', '')}`"
                    ),
                    "—",
                    "—",
                    "—",
                    f"{float(payload.get('tool_latency_ms') or 0.0) / 1000:.6f} s",
                ])

            elif event == "TOOL_RETRY":
                retry_index += 1
                rows.append([
                    f"R{retry_index}",
                    "Harness",
                    (
                        f"retry `{payload.get('name', '?')}` → "
                        f"**{payload.get('retry_number', '?')}/{payload.get('max_retry', '?')}**"
                    ),
                    "—",
                    "—",
                    "—",
                    "—",
                ])

            elif event == "TOOL_RETRY_EXHAUSTED":
                retry_index += 1
                rows.append([
                    f"R{retry_index}",
                    "Harness",
                    (
                        f"retry exhausted `{payload.get('name', '?')}` → terminate"
                    ),
                    "—",
                    "—",
                    "—",
                    "—",
                ])

            elif event == "TOOL_RESULT":
                tool_index += 1

                name = payload["name"]
                result = payload.get("result", "")
                latency_ms = float(
                    payload.get("tool_latency_ms") or 0.0
                )

                call = tool_call_by_id.get(
                    payload.get("tool_call_id", ""),
                    {},
                )

                if name == "list_documents":
                    try:
                        docs = json.loads(result)
                        operation = (
                            f"`list_documents` → "
                            f"**{len(docs)} docs**"
                        )
                    except Exception:
                        operation = (
                            f"`list_documents` → "
                            f"{len(result):,} chars"
                        )

                elif name == "read_document":
                    filename = (
                        call.get("arguments", {}).get("name")
                    )

                    if filename:
                        operation = (
                            f"`read_document` → "
                            f"`{filename}` · "
                            f"**{len(result):,} chars**"
                        )
                    else:
                        operation = (
                            f"`read_document` → "
                            f"**{len(result):,} chars**"
                        )

                elif name == "save_note":
                    operation = "`save_note` → saved"

                else:
                    operation = (
                        f"`{name}` → {len(result):,} chars"
                    )

                retry_count = int(payload.get("retry_count") or 0)
                if retry_count:
                    operation += f" · retry **{retry_count}**"

                rows.append([
                    f"T{tool_index}",
                    "Tool",
                    operation,
                    "—",
                    "—",
                    "—",
                    f"{latency_ms / 1000:.6f} s",
                ])

        tokens = summary["tokens"]
        timing = summary["timing_ms"]

        rows.append([
            "**Total**",
            "**Run**",
            (
                f"**{summary['model_calls']} model + "
                f"{summary['tool_calls']} tool**"
            ),
            "—",
            (
                f"**{tokens['prompt']:,} "
                f"({tokens['cache_hit']:,}/"
                f"{tokens['cache_miss']:,})**"
            ),
            f"**{tokens['completion']:,}**",
            f"**{timing['end_to_end'] / 1000:.3f} s E2E**",
        ])

        return rows

    @staticmethod
    def _markdown_table(
        headers: list[str],
        rows: list[list[str]],
    ) -> list[str]:
        lines = [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
        ]

        for row in rows:
            lines.append("| " + " | ".join(row) + " |")

        return lines

    def _write_markdown(
        self,
        user_input: str,
        summary: dict[str, Any],
    ) -> None:
        """生成面向人工复核的单轮 Trace；用户问题紧邻 Execution Overview 展示。"""
        t = summary["tokens"]
        tm = summary["timing_ms"]

        execution_rows = self._execution_rows(summary)

        lines: list[str] = [
            "# Agent Harness Trace",
            "",
            "## Run Summary",
            "",
            f"- Model: `{summary['model']}`",
            "- Thinking: `disabled`",
            f"- Model calls: **{summary['model_calls']}** "
            f"(main {summary.get('main_model_calls', summary['model_calls'])} / "
            f"subagent {summary.get('subagent_model_calls', 0)})",
            f"- Tool calls: **{summary['tool_calls']}**",
            f"- Total tokens: **{t['total']:,}** "
            f"(input {t['prompt']:,} / output {t['completion']:,})",
            f"- End-to-end: **{tm['end_to_end'] / 1000:.3f} s**",
            f"- Estimated cost: **¥{summary['estimated_cost_cny']:.6f}**",
            "",
            "## User Input",
            "",
            user_input,
            "",
            "## Execution Overview",
            "",
        ]

        lines.extend(
            self._markdown_table(
                [
                    "Step",
                    "Actor",
                    "Operation / Result",
                    "Msgs",
                    "Input tokens (hit/miss)",
                    "Output",
                    "Time",
                ],
                execution_rows,
            )
        )
        lines.extend([
            "",
            "> `Input tokens (hit/miss)` 中括号内依次为 `cache hit / cache miss`。",
            "",
        ])

        detail_start = len(lines)
        flow: list[str] = ["User"]

        for record in self.records:
            event = record["event"]
            payload = record["payload"]

            if event == "MCP_CONNECT":
                flow.append("MCP stdio connect")
                info = payload.get("server_info") or {}
                lines.extend([
                    "## MCP Connect",
                    "",
                    "- Transport: `stdio`",
                    f"- Server: `{info.get('name') or '?'}`",
                    f"- Protocol version: `{payload.get('protocol_version') or '?'}`",
                    f"- Server script: `{payload.get('server_script', '')}`",
                    "",
                ])

            elif event == "MCP_LIST_TOOLS":
                flow.append("MCP list_tools")
                names = [
                    item.get("function", {}).get("name", "?")
                    for item in payload.get("tools", [])
                ]
                lines.extend([
                    "### MCP Tool Discovery",
                    "",
                    f"- Discovered: **{payload.get('count', 0)}**",
                    f"- Tools: `{', '.join(names)}`",
                    "",
                ])

            elif event == "MCP_CALL_TOOL":
                flow.append(f"MCP call {payload.get('name', 'tool')}")
                lines.extend([
                    "### MCP Call Tool",
                    "",
                    f"- Transport: `{payload.get('transport', 'stdio')}`",
                    f"- Tool: `{payload.get('name', '?')}`",
                    f"- Arguments: `{json.dumps(payload.get('arguments', {}), ensure_ascii=False)}`",
                    "",
                ])

            elif event == "MCP_RESULT":
                flow.append(f"MCP result {payload.get('name', 'tool')}")
                lines.extend([
                    "### MCP Result",
                    "",
                    f"- Tool: `{payload.get('name', '?')}`",
                    f"- Is error: **{payload.get('is_error', False)}**",
                    f"- Result chars: **{int(payload.get('result_chars') or 0):,}**",
                    "",
                ])

            elif event == "SUBAGENT_DELEGATE":
                flow.append(f"delegate {payload.get('subagent', 'reviewer')}")
                lines.extend([
                    "## Subagent Delegation",
                    "",
                    f"- Subagent: `{payload.get('subagent', 'reviewer')}`",
                    f"- Interface: `{payload.get('tool', 'review_answer')}`",
                    f"- Draft chars: **{int(payload.get('draft_chars') or 0):,}**",
                    f"- Evidence chars: **{int(payload.get('evidence_chars') or 0):,}**",
                    f"- Reason: {payload.get('reason', '')}",
                    "",
                ])

            elif event == "SUBAGENT_CONTEXT":
                flow.append("Reviewer isolated context")
                lines.extend([
                    "### Reviewer Context Isolation",
                    "",
                    f"- Mother Agent history inherited: **{payload.get('mother_history_inherited')}**",
                    f"- Mother Agent memory inherited: **{payload.get('mother_memory_inherited')}**",
                    f"- Message count: **{payload.get('message_count', 0)}**",
                    f"- Roles: `{' → '.join(payload.get('roles', []))}`",
                    f"- Available tools: **{payload.get('tool_count', 0)}**",
                    f"- Draft chars: **{int(payload.get('draft_chars') or 0):,}**",
                    f"- Evidence chars: **{int(payload.get('evidence_chars') or 0):,}**",
                    "",
                ])

            elif event == "SUBAGENT_MODEL_REQUEST":
                flow.append("Reviewer Model Call")
                request = payload.get("request") or {}
                messages = request.get("messages", [])
                lines.extend([
                    "### Reviewer Model Request",
                    "",
                    f"- Message count: **{len(messages)}**",
                    f"- Roles: `{' → '.join(m.get('role', '?') for m in messages)}`",
                    "- Available tools: **none**",
                    "",
                ])

            elif event == "SUBAGENT_MODEL_RESPONSE":
                usage = payload.get("usage") or {}
                response = payload.get("response") or {}
                message = response.get("choices", [{}])[0].get("message", {})
                lines.extend([
                    "### Reviewer Model Response",
                    "",
                    f"- Model: `{payload.get('model', '?')}`",
                    f"- Latency: **{float(payload.get('latency_ms') or 0.0) / 1000:.3f} s**",
                    f"- Prompt tokens: **{int(usage.get('prompt_tokens') or 0):,}**",
                    f"- Cache hit / miss: **{int(usage.get('prompt_cache_hit_tokens') or 0):,} / {int(usage.get('prompt_cache_miss_tokens') or 0):,}**",
                    f"- Completion tokens: **{int(usage.get('completion_tokens') or 0):,}**",
                    "",
                    str(message.get("content") or ""),
                    "",
                ])

            elif event == "SUBAGENT_RESULT":
                flow.append("Reviewer Result → Mother Agent")
                lines.extend([
                    "### Reviewer Result Returned",
                    "",
                    str(payload.get("result") or ""),
                    "",
                ])

            elif event == "MEMORY_READ":
                flow.append("读取当前有效记忆")
                lines.extend([
                    "## 读取当前有效记忆",
                    "",
                    f"- 当前有效记忆数: **{payload.get('count', 0)}**",
                    f"- 原因: {payload.get('reason', '')}",
                ])
                for item in payload.get("records", []):
                    lines.append(
                        f"- `{item.get('kind')}:{item.get('key')}` = "
                        f"`{item.get('value')}`"
                    )
                lines.append("")

            elif event == "MEMORY_INJECT":
                flow.append("注入当前有效记忆")
                lines.extend([
                    "### 注入当前有效记忆",
                    "",
                    f"- Model step: **{payload.get('step')}**",
                    f"- 注入数量: **{payload.get('count', 0)}**",
                    f"- 原因: {payload.get('reason', '')}",
                ])
                for item in payload.get("records", []):
                    lines.append(
                        f"- `{item.get('kind')}:{item.get('key')}` = "
                        f"`{item.get('value')}`"
                    )
                lines.append("")

            elif event == "MEMORY_WRITE":
                flow.append("写入长期记忆")
                item = payload.get("record") or {}
                lines.extend([
                    "## 写入长期记忆",
                    "",
                    f"- Key: `{item.get('kind')}:{item.get('key')}`",
                    f"- Value: `{item.get('value')}`",
                    f"- Status: `{item.get('status')}`（当前有效）",
                    f"- 原因: {payload.get('reason', '')}",
                    "",
                ])

            elif event == "MEMORY_SUPERSEDE":
                flow.append("替换长期记忆")
                old = payload.get("previous") or {}
                new = payload.get("record") or {}
                lines.extend([
                    "## 替换长期记忆",
                    "",
                    f"- Key: `{new.get('kind')}:{new.get('key')}`",
                    f"- 旧版本: `{old.get('value')}` → `{old.get('status')}`（已被替换）",
                    f"- 当前版本: `{new.get('value')}` → `{new.get('status')}`（当前有效）",
                    f"- 原因: {payload.get('reason', '')}",
                    "",
                ])

            elif event == "MODEL_REQUEST":
                step = payload["step"]
                request = payload["request"]
                messages = request["messages"]
                roles = [m.get("role", "?") for m in messages]

                lines.extend([
                    f"## Model Call {step}",
                    "",
                    "### Request",
                    "",
                    f"- Message count: **{len(messages)}**",
                    f"- Roles: `{' → '.join(roles)}`",
                    "- Available tools: "
                    + ", ".join(
                        f"`{x['function']['name']}`"
                        for x in request.get("tools", [])
                    ),
                    "",
                ])
                flow.append(f"Model Call {step}")

            elif event == "MODEL_RESPONSE":
                response = payload["response"]
                usage = payload.get("usage") or {}
                message = response.get("choices", [{}])[0].get("message", {})
                calls = message.get("tool_calls") or []

                lines.extend([
                    "### Response",
                    "",
                    f"- Latency: **{payload['latency_ms'] / 1000:.3f} s**",
                    f"- Prompt tokens: **{usage.get('prompt_tokens', 0):,}**",
                    f"- Cache hit / miss: **{usage.get('prompt_cache_hit_tokens', 0):,} / "
                    f"{usage.get('prompt_cache_miss_tokens', 0):,}**",
                    f"- Completion tokens: **{usage.get('completion_tokens', 0):,}**",
                    "",
                ])

                if calls:
                    lines.extend(["Model requested:", ""])
                    for call in calls:
                        fn = call.get("function", {})
                        lines.append(
                            f"- `{fn.get('name')}({fn.get('arguments', '{}')})`"
                        )
                    lines.append("")

            elif event == "TOOL_CALL":
                flow.append(payload["name"])
                lines.extend([
                    "### Tool Call",
                    "",
                    f"- Tool: `{payload['name']}`",
                    f"- Arguments: `{json.dumps(payload['arguments'], ensure_ascii=False)}`",
                    "",
                ])

            elif event == "TOOL_ERROR":
                flow.append(f"{payload.get('name', 'tool')} error")
                lines.extend([
                    "### Tool Error",
                    "",
                    f"- Tool: `{payload.get('name', '?')}`",
                    f"- Attempt: **{payload.get('attempt', '?')}**",
                    f"- Max retry: **{payload.get('max_retry', '?')}**",
                    f"- Will retry: **{payload.get('will_retry', False)}**",
                    f"- Error: `{payload.get('error', '')}`",
                    "",
                ])

            elif event == "TOOL_RETRY":
                flow.append(f"retry {payload.get('name', 'tool')}")
                lines.extend([
                    "### Harness Retry",
                    "",
                    f"- Tool: `{payload.get('name', '?')}`",
                    f"- Retry: **{payload.get('retry_number', '?')}/{payload.get('max_retry', '?')}**",
                    f"- Reason: `{payload.get('reason', '')}`",
                    "",
                ])

            elif event == "TOOL_RETRY_EXHAUSTED":
                flow.append(f"retry exhausted {payload.get('name', 'tool')}")
                lines.extend([
                    "### Retry Exhausted",
                    "",
                    f"- Tool: `{payload.get('name', '?')}`",
                    f"- Attempts: **{payload.get('attempts', '?')}**",
                    f"- Error: `{payload.get('error', '')}`",
                    "- Result: **terminate current AgentLoop turn**",
                    "",
                ])

            elif event == "TOOL_RESULT":
                result = payload["result"]
                tool_time = payload.get("tool_latency_ms", 0.0)
                lines.extend([
                    "### Tool Result",
                    "",
                    f"- Tool: `{payload['name']}`",
                    f"- Execution time: **{tool_time / 1000:.6f} s**",
                    f"- Attempt: **{payload.get('attempt', 1)}**",
                    f"- Retry count: **{payload.get('retry_count', 0)}**",
                    f"- Result chars: **{len(result):,}**",
                    "",
                    "Preview:",
                    "",
                    "```text",
                    result[:800],
                    "```",
                    "",
                ])

            elif event == "FINAL_ANSWER":
                flow.append("Final Answer")
                lines.extend([
                    "## Final Answer",
                    "",
                    payload["answer"],
                    "",
                ])

        lines[detail_start:detail_start] = [
            "## Execution Flow",
            "",
            " → ".join(flow),
            "",
        ]

        self.md_path.write_text(
            "\n".join(lines),
            encoding="utf-8",
        )
