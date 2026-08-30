# Agent Harness 架构

[English](en/architecture.md)

## 1. 目标

本项目用最小实现拆解 Agent Harness 的运行时职责。核心对象是模型调用之外的控制层：Session、Working Context、Memory、Tool Execution、Agent Loop、Trace、Persistence / Recovery，以及 Retry、MCP、Subagent 等扩展边界。

## 2. 总体结构

```text
User
↓
Session + Active Memory
↓
ContextBuilder
↓
Model
├─ Final Answer
└─ Tool Call
   ↓
   Tool Registry / MCP / Subagent
   ↓
   Tool Result
   └────────────→ Model
↓
Trace + Persistence
```

主要模块：

```text
src/mini_agent_harness/
├── core/
│   ├── agent_loop.py     # Model / Tool 循环与一轮执行收口
│   ├── context.py        # Working Context 构造
│   ├── memory.py         # 结构化 Memory 生命周期
│   ├── session.py        # 完整 Conversation History
│   ├── tools.py          # 本地 Tool Registry
│   ├── mcp_tools.py      # MCP Tool 适配
│   ├── subagent.py       # Reviewer Subagent
│   └── tracing.py        # JSONL / Markdown Trace
├── mcp/                  # stdio MCP Server
├── providers/            # OpenAI-compatible Provider
└── storage/              # SQLite Session / Memory Store
```

## 3. Agent Loop

一次用户轮次可以包含多次 Model Call：

```text
User Message
→ Build Context
→ Model Call
→ Tool Call
→ Harness Execute Tool
→ Tool Result
→ Build Context
→ Model Call
→ ...
→ Final Answer
```

模型负责生成最终回答或提出 Tool Call；Harness 负责执行工具、保存状态、回注 Tool Result、继续或终止循环。

实验内置三个本地工具：

```text
list_documents()
read_document(name)
save_note(text)
```

`max_steps` 限制单轮循环步数，避免工具链无限展开。

## 4. Session、Working Context 与 Memory

### Session

Session 保存完整 Conversation History，包括 user、assistant、tool 等消息，是权威会话存档。

### Working Context

`ContextBuilder` 从完整 Session 中生成当前一次 Model Call 实际看到的消息。项目实现三种策略：

```text
full_history
→ 完整历史

last_n
→ 最近窗口

managed
→ Historical Summary + Recent Raw Messages
```

Active Memory 会作为独立 system message 注入 Working Context。

### Structured Memory

`MemoryManager` 保存需要跨轮保留的当前状态：

```text
kind: preference / fact / decision
status: active / superseded
```

更新采用 supersede：旧版本保留并标记为 `superseded`，新版本成为 `active`。

当前冲突优先级定义为：

```text
Current User Instruction
>
Active Structured Memory
>
Historical Conversation / Summary
```

## 5. Context Strategy

### Full History

完整 Session 直接进入模型请求，原始 Tool Result 也随历史保留。

### Last-N

只选择最近一段完整消息窗口，并尽量从 user message 开始，避免截断 assistant/tool 事务。

### Managed Context

当历史超过阈值后：

```text
Earlier History
→ 提取 user + final assistant
→ 本地确定性摘要
→ 最多 1800 chars

Recent History
→ 保留最近 raw messages

Active Memory
→ 独立注入
```

该实现的目标是让 Context Construction 本身容易观察与 Trace；具体限制见 [已知限制](known-limitations.md)。

## 6. Persistence / Recovery

SQLite Store 持久化：

```text
Session
Memory Records
```

恢复链路：

```text
Process A
→ Session / Memory 写入 SQLite
→ Python process exit

Process B
→ 打开同一 SQLite
→ Restore Session
→ Restore MemoryManager
→ 继续新的 User Turn
```

Step 8 和 Step 9 都实际使用新 Python 进程验证恢复，而非在同一进程内模拟重载。

## 7. Trace

Trace 分为：

```text
JSONL   # 机器可读
Markdown # 人类可读
```

核心事件包括：

```text
CONTEXT_BUILD
MODEL_REQUEST
MODEL_RESPONSE
TOOL_CALL
TOOL_RESULT
TOOL_ERROR
MEMORY_READ
MEMORY_WRITE
MEMORY_SUPERSEDE
MEMORY_INJECT
FINAL_ANSWER
RUN_SUMMARY
```

Retry、MCP、Subagent Probe 会继续记录各自的边界事件。

## 8. Retry、MCP 与 Subagent

### Retry

Step 10.1 在显式 `max_retry=1` 时由 Harness 接管本地 Tool Execution Retry：

```text
Tool Error
→ Retry
→ Success / Retry Exhausted
```

### MCP

Step 10.2 将 `read_document` 迁移到独立 stdio MCP Server：

```text
AgentLoop
→ MCP Client
→ stdio
→ MCP Server
→ read_document
```

模型看到的是 MCP discovery 得到的 Tool Schema，Tool Result 再回到原 Agent Loop。

### Reviewer Subagent

Step 10.3 通过 `review_answer` 做一次最小委派：

```text
Mother Agent
→ task + draft + evidence
→ Reviewer
→ review result
→ Mother Agent
→ Final Answer
```

Reviewer 使用独立 system prompt，只接收委派 payload，不继承母 Agent 的 Conversation 与 Memory。

## 9. 设计取向

项目保持单机、少依赖、可 Trace、可重复运行。功能围绕 Agent Runtime / Harness 机制展开，复杂应用编排与生产基础设施留在项目范围之外。
