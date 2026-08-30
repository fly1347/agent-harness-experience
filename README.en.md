# Agent Harness Experience

[中文](README.md)

A minimal, auditable experiment for **understanding Agent Harness runtime mechanics by building them directly**.

It starts from a plain LLM call and incrementally adds Tool Calling, an Agent Loop, multi-turn sessions, context management, memory, SQLite persistence/recovery, tracing, retry, MCP, and a reviewer subagent. The goal is to expose runtime mechanisms that frameworks usually package together as small, runnable, replayable experiments.

## Experiment scope

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

Core runtime path:

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

## Core implementation

| Mechanism | Implementation |
| --- | --- |
| Tool Calling | `list_documents` / `read_document` / `save_note` |
| Agent Loop | Model → Tool → Model, with multiple calls inside one user turn |
| Context | `full_history` / `last_n` / `managed` strategies |
| Memory | active / superseded structured state |
| Persistence | SQLite Session and Memory recovery across Python processes |
| Trace | JSONL + Markdown events for Model / Tool / Context / Memory |
| Retry | minimal Tool Failure recovery with `max_retry=1` |
| MCP | `read_document` served through a stdio MCP Server |
| Subagent | reviewer delegation, isolated context, and result return |

## Reference run

The fixed T01–T10 scenario completed under all three context strategies. After T08, the Python worker exited and a new process restored Session and Memory from SQLite.

| Strategy | Checks | Restart | T10 Answer | Input Tokens | Tool Calls |
| --- | ---: | --- | --- | ---: | ---: |
| Full History | 10/10 | PASS | PASS | 162,106 | 9 |
| Last-N | 10/10 | PASS | PASS | 40,510 | 19 |
| Managed | 10/10 | PASS | PASS | 68,865 | 21 |

The table records one fixed-scenario reference run. Model-driven Tool Calling can vary across runs; stable performance comparisons require repeated runs and distribution-level reporting.

Extension probes:

- Retry: 2/2 PASS;
- MCP: PASS;
- Reviewer Subagent: PASS;
- Offline tests: 31/31 PASS.

## Run

Requirements: Python `>= 3.10` and a Chat Completions / OpenAI-compatible model endpoint.

```bash
pip install -e .
cp .env.example .env
```

Configure `.env`:

```dotenv
LLM_API_KEY=your_api_key
LLM_BASE_URL=your_openai_compatible_base_url
LLM_MODEL=your_model
LLM_TEMPERATURE=0
```

Basic entry points:

```bash
python -m mini_agent_harness.cli "Explain Agent Harness in one sentence."
python -m mini_agent_harness.cli "读取 01。检索层和证据层最核心的区别是什么？" --agent
python -m mini_agent_harness.cli --agent --session
```

Full scenario and extension probes:

```bash
python scripts/run_scenario.py --strategy all
python scripts/run_retry_probe.py
python scripts/run_mcp_probe.py
python scripts/run_subagent_probe.py
```

Offline tests:

```bash
python -m unittest discover -s tests -v
```

Online scripts call the configured model and write traces, reports, and state under `artifacts/`.

## Repository structure

```text
src/mini_agent_harness/   Agent Loop, Context, Memory, Trace, MCP, Storage
scripts/                  Scenario, Memory, Recovery, and extension probes
scenarios/                fixed multi-turn experiment scenario
fixtures/                 frozen local knowledge materials and provenance
tests/                    offline tests
artifacts/                 runtime outputs
```

Fixture provenance: [fixtures/README.en.md](fixtures/README.en.md)  
Fixed scenario: [scenarios/multi_turn_context.json](scenarios/multi_turn_context.json)

## Documentation

- [Architecture](docs/en/architecture.md)
- [Experiment Report](docs/en/experiment-report.md)
- [Known Limitations](docs/en/known-limitations.md)

## Positioning

This repository is a minimal single-machine implementation for learning, engineering validation, and reproducible experiments around Agent Runtime / Harness mechanics, covering Tool Calling, Context, Memory, Persistence / Recovery, Trace, Retry, MCP, and Subagents.
