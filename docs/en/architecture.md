# Agent Harness Architecture

[中文](../architecture.md)

## 1. Goal

This project decomposes Agent Harness runtime responsibilities with a minimal implementation. The focus is the control layer around model calls: sessions, working context, memory, tool execution, the agent loop, tracing, persistence/recovery, and small probes for retry, MCP, and subagents.

## 2. Runtime structure

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

Main modules:

```text
src/mini_agent_harness/
├── core/
│   ├── agent_loop.py
│   ├── context.py
│   ├── memory.py
│   ├── session.py
│   ├── tools.py
│   ├── mcp_tools.py
│   ├── subagent.py
│   └── tracing.py
├── mcp/
├── providers/
└── storage/
```

## 3. Agent loop

One user turn may contain multiple model calls:

```text
User Message
→ Build Context
→ Model Call
→ Tool Call
→ Harness Executes Tool
→ Tool Result
→ Build Context
→ Model Call
→ ...
→ Final Answer
```

The model proposes an answer or a tool call. The Harness executes tools, retains state, feeds tool results back, and controls loop continuation or termination.

Local tools used by the experiment:

```text
list_documents()
read_document(name)
save_note(text)
```

## 4. Session, working context, and memory

**Session** stores the canonical conversation history, including user, assistant, and tool messages.

**Working Context** is the request view built by `ContextBuilder` for one model call. Three strategies are implemented:

```text
full_history
last_n
managed
```

**Structured Memory** stores state that should survive turns and process restarts. Records use `preference / fact / decision` kinds and `active / superseded` lifecycle states.

Current conflict priority:

```text
Current User Instruction
>
Active Structured Memory
>
Historical Conversation / Summary
```

## 5. Context strategies

`full_history` sends the complete session history.

`last_n` keeps a recent raw-message window while attempting to preserve complete user/tool transactions.

`managed` combines a deterministic historical summary, recent raw messages, and separately injected active memory. Its current implementation and limitations are documented in [Known Limitations](known-limitations.md).

## 6. Persistence and recovery

SQLite persists Session and Memory records. Recovery is tested with a real process boundary:

```text
Process A
→ persist state
→ exit

Process B
→ reopen SQLite
→ restore Session + Memory
→ continue the scenario
```

## 7. Trace

Trace output is available as machine-readable JSONL and human-readable Markdown. Core events include:

```text
CONTEXT_BUILD
MODEL_REQUEST
MODEL_RESPONSE
TOOL_CALL
TOOL_RESULT
TOOL_ERROR
MEMORY_*
FINAL_ANSWER
RUN_SUMMARY
```

## 8. Retry, MCP, and subagents

**Retry**: an explicit `max_retry=1` probe exercises local tool-execution retry and exhausted termination.

**MCP**: `read_document` is served by an independent stdio MCP Server and discovered/called through an MCP client.

**Reviewer Subagent**: the Mother Agent delegates `task + draft + evidence` to an isolated reviewer with no tools or inherited conversation/memory, then consumes the returned review result.

## 9. Design scope

The repository favors a small, single-machine, traceable implementation that makes runtime mechanics visible. Production-scale orchestration and infrastructure are intentionally left outside the experiment scope.
