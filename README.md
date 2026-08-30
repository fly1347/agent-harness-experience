# Agent Harness Experience

[English](README.en.md)

一个用于**亲手理解 Agent Harness 运行时机制**的最小可审计实验项目。

它是从普通 LLM 调用开始，逐层加入 Tool Calling、Agent Loop、多轮 Session、Context Management、Memory、SQLite Persistence / Recovery、Trace、Retry、MCP 和 Reviewer Subagent，把框架通常封装起来的运行时机制拆成可运行、可回放、可比较的最小实现。

## 实验内容

```text
Plain LLM
→ Tool Calling / Agent Loop
→ Multi-turn Session
→ Fixed Scenario Replay
→ Context Management
→ Memory Lifecycle
→ SQLite Persistence / Recovery
→ Cross-process Replay
→ Tool Retry
→ MCP
→ Reviewer Subagent
```

核心运行链路：

```text
User
↓
Session + Memory
↓
ContextBuilder
↓
Model
├─ Final Answer
└─ Tool Call
   ↓
   Tool / MCP / Subagent
   ↓
   Tool Result
   └────────────→ Model
↓
Trace + Persistence
```

## 核心实现

| 机制 | 实现 |
| --- | --- |
| Tool Calling | `list_documents` / `read_document` / `save_note` |
| Agent Loop | Model → Tool → Model，支持单轮多次调用 |
| Context | `full_history` / `last_n` / `managed` 三种策略 |
| Memory | active / superseded 结构化状态 |
| Persistence | SQLite 保存 Session 与 Memory，支持跨 Python 进程恢复 |
| Trace | JSONL + Markdown 记录 Model / Tool / Context / Memory 事件 |
| Retry | `max_retry=1` 的最小 Tool Failure 恢复实验 |
| MCP | stdio MCP Server 承载 `read_document` |
| Subagent | Reviewer 委派、上下文隔离与结果返回 |

## 已完成的参考运行

固定 T01～T10 Scenario 在三种 Context Strategy 上均完成，并在 T08 后真实退出 Python worker，再由新进程恢复 Session 与 Memory。

| Strategy | Checks | Restart | T10 Answer | Input Tokens | Tool Calls |
| --- | ---: | --- | --- | ---: | ---: |
| Full History | 10/10 | PASS | PASS | 162,106 | 9 |
| Last-N | 10/10 | PASS | PASS | 40,510 | 19 |
| Managed | 10/10 | PASS | PASS | 68,865 | 21 |

表中数字来自一次固定 Scenario 参考运行。模型驱动的 Tool Calling 存在运行间波动；稳定性能比较需要重复运行与分布统计。

扩展 Probe：

- Retry：2/2 PASS；
- MCP：PASS；
- Reviewer Subagent：PASS；
- 离线测试：31/31 PASS。

## 运行

要求 Python `>= 3.10`，并准备一个 Chat Completions / OpenAI-compatible 模型端点。

```bash
pip install -e .
cp .env.example .env
```

配置 `.env`：

```dotenv
LLM_API_KEY=your_api_key
LLM_BASE_URL=your_openai_compatible_base_url
LLM_MODEL=your_model
LLM_TEMPERATURE=0
```

基础入口：

```bash
python -m mini_agent_harness.cli "用一句话解释什么是 Agent Harness。"
python -m mini_agent_harness.cli "读取 01。检索层和证据层最核心的区别是什么？" --agent
python -m mini_agent_harness.cli --agent --session
```

完整 Scenario 与扩展 Probe：

```bash
python scripts/run_scenario.py --strategy all
python scripts/run_retry_probe.py
python scripts/run_mcp_probe.py
python scripts/run_subagent_probe.py
```

离线测试：

```bash
python -m unittest discover -s tests -v
```

在线脚本会调用配置的模型，并在 `artifacts/` 下生成 Trace、Report 与状态文件。

## 仓库结构

```text
src/mini_agent_harness/   Agent Loop、Context、Memory、Trace、MCP、Storage
scripts/                  Scenario、Memory、Recovery 与扩展 Probe
scenarios/                固定多轮实验 Scenario
fixtures/                 固定本地知识材料与来源说明
tests/                    离线测试
artifacts/                 运行时输出
```

Fixture 来源说明：[fixtures/README.md](fixtures/README.md)  
固定 Scenario：[scenarios/multi_turn_context.json](scenarios/multi_turn_context.json)

## 文档

- [架构](docs/architecture.md)
- [实验报告](docs/experiment-report.md)
- [已知限制](docs/known-limitations.md)

## 项目定位

这是一个面向 Agent Runtime / Harness 机制学习、工程验证与可复现实验的单机最小实现，覆盖 Tool Calling、Context、Memory、Persistence / Recovery、Trace、Retry、MCP 与 Subagent 的完整体验链路。
