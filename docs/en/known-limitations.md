# Known Limitations

[中文](../known-limitations.md)

This document separates mechanisms validated by the experiment from capabilities that would be required by a more complete runtime.

## 1. Managed Context uses a deterministic simplified summary

The current strategy extracts earlier user/final-assistant messages in chronological order up to a fixed 1,800-character budget and combines them with recent raw messages.

This can create a **Middle Gap**: early history remains summarized, recent history remains raw, while some middle messages are absent from the current model request. The canonical Session still contains them; the loss occurs during context construction.

## 2. Historical summary and Active Memory require explicit priority

The experiment currently uses:

```text
Current User Instruction
>
Active Structured Memory
>
Historical Conversation / Summary
```

More complex temporal state, scoping, and multi-entity memory would require a stronger versioning and invalidation model.

## 3. Tool paths vary across runs

The model dynamically decides whether to call tools, which tool to use, whether to list before reading, and how to recover from failed calls. Model-call count, tool-call count, latency, and cost therefore vary between runs. Stable comparisons require repeated measurements.

## 4. Memory extraction is narrow

`MemoryManager.apply_explicit_instruction()` implements only the fixed experiment's explicit write/update patterns. It does not implement general automatic memory extraction, entity normalization, confidence, conflict merging, or retrieval ranking.

## 5. Retry is a minimal tool-execution probe

Only `max_retry=1` is exercised. Exponential backoff, jitter, circuit breakers, idempotency policies, error-class-specific retry, and fallback routing are outside the current implementation.

## 6. MCP uses one local stdio server

The MCP probe migrates only `read_document`. Remote servers, multiple servers, authentication, authorization, network failures, and server lifecycle governance are outside the current scope.

## 7. Reviewer is a minimal local verifier

The reviewer has no tools and receives evidence selected by the Mother Agent. Independent retrieval, parallel reviewers, Review–Revise loops, and reviewer consensus are not implemented.

## 8. Persistence is single-machine SQLite

The store is sufficient for cross-process Session/Memory recovery experiments. Distributed state, concurrent-write governance, HA, leases, and multi-instance consistency are outside scope.

## 9. Trace is experiment-grade audit data

JSONL/Markdown traces can contain full prompts, tool results, and user inputs. Real deployments would require data classification, redaction, retention policies, and access control.

## 10. Provider compatibility varies

The provider targets Chat Completions / OpenAI-compatible endpoints and the reference experiment uses `thinking: disabled`. Tool schemas, extended request fields, usage fields, and cache metadata can differ across providers.

## 11. Project scope

The repository focuses on Agent Runtime / Harness mechanics. Complex planning, multi-agent swarms, web applications, service/container orchestration, production authorization, and full observability platforms belong to a larger system scope.
