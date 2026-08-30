# Agent Harness Experiment Report

[中文](../experiment-report.md)

## 1. Scope

The experiment starts with a plain LLM call and incrementally adds:

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

Fixtures are frozen local documents under `fixtures/`; the multi-turn replay uses `scenarios/multi_turn_context.json`.

## 2. Reference model configuration

```text
Provider: DeepSeek Official API
Model: deepseek-v4-flash
Thinking: disabled
Temperature: 0
```

Model-driven tool use still shows run-to-run variation, so model calls, tool calls, latency, and cost are treated as observations from a specific run rather than fixed strategy properties.

## 3. Core runtime validation

The project validates the transition from a plain request/response call to a tool-driven agent loop, persistent multi-turn sessions, explicit context construction, structured memory, tracing, and cross-process state recovery.

## 4. Full context-strategy replay

The final T01–T10 scenario runs all three context strategies. After T08, the Python worker exits and a new process restores Session and Memory before T09–T10.

| Strategy | Checks | Restart | T10 Answer | T10 no-refetch | Model Calls | Tool Calls | Input Tokens | E2E |
| --- | ---: | --- | --- | --- | ---: | ---: | ---: | ---: |
| Full History | 10/10 | PASS | PASS | YES | 19 | 9 | 162,106 | 33.316 s |
| Last-N | 10/10 | PASS | PASS | NO | 27 | 19 | 40,510 | 36.575 s |
| Managed | 10/10 | PASS | PASS | NO | 29 | 21 | 68,865 | 41.308 s |

Estimated cost for the same run:

```text
Full History: ¥0.067373
Last-N:       ¥0.046628
Managed:      ¥0.060160
```

All strategies pass the final-answer and restart checks. The differences are in information visibility, re-fetch paths, and runtime cost for this particular run.

## 5. Run-to-run tool variation

Repeated runs of the same scenario showed different tool paths. For example, Managed T10 changed from two direct reads in one run to failed short-name reads, `list_documents`, and full-name reads in another.

The experiment therefore separates:

```text
Context Strategy
→ what information the model can currently see

Model Tool Decision
→ whether and how tools are called in that turn
```

Stable performance comparison requires repeated runs and distribution-level reporting.

## 6. Memory lifecycle

Structured Memory exercises:

```text
Identify → Write → Retrieve → Inject → Update / Supersede
```

Step 7 memory checks: **5/5 PASS**.

## 7. Persistence / recovery

SQLite persists Session and Memory. Step 8 recovery checks: **16/16 PASS**.

Additional probes confirm that restored Active Memory can support an answer with a fresh Session, and that Active Memory is treated as the current state when it conflicts with historical summary text.

## 8. Extension probes

- Retry: **2/2 PASS**
- MCP: **PASS**
- Reviewer Subagent: **PASS**

The reviewer receives an isolated `task + draft + evidence` payload, has no tools, and does not inherit Mother Agent conversation or memory.

## 9. Final acceptance

```text
31/31 unittest PASS
compileall PASS
3 context strategies: 10/10 each
Restart Recovery: PASS
Retry: 2/2 PASS
MCP: PASS
Reviewer Subagent: PASS
```

See [Known Limitations](known-limitations.md) for the current implementation boundaries.
