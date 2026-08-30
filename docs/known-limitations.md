# 已知限制

[English](en/known-limitations.md)

本文件记录当前最小实现的边界，便于区分“已经验证的 Harness 机制”和“更完整系统仍需补充的能力”。

## 1. Managed Context 使用确定性简化摘要

当前 `managed` 策略在历史超过阈值后：

```text
Earlier History
→ 提取 user + final assistant
→ 从旧到新累计
→ 最多 1800 chars

Recent History
→ 保留最近 raw messages
```

摘要由本地规则生成，没有 LLM summarizer、importance scoring、conversation retrieval 或 hierarchical summary。

### Middle Gap

当 Earlier Summary 达到字符上限，而某段消息又已经离开 Recent Window 时，中间历史可能完全退出本轮 Working Context：

```text
早期历史：Summary 保留
中间历史：可能不可见
最近历史：Raw Messages 保留
```

这里属于 Context Construction 阶段的信息遗漏：权威 Session 仍然保留完整历史，但当前 Model Request 没有收到那部分内容。

## 2. Summary 与 Active Memory 需要状态优先级

历史摘要可能保留旧状态，Structured Memory 可能已经更新为新状态。当前实现明确：

```text
Current User Instruction
>
Active Structured Memory
>
Historical Conversation / Summary
```

这解决了固定实验中的状态冲突，但更复杂的时间状态、作用域和多实体 Memory 仍需要更正式的版本与失效模型。

## 3. Tool Calling 路径存在运行间波动

模型会动态决定：

```text
是否调用工具
调用哪个工具
先 list 还是直接 read
是否产生冗余 save_note
失败后采用哪条恢复路径
```

因此相同 Scenario 的 Model Calls、Tool Calls、Latency 与 Cost 可能变化。当前参考表适合展示具体运行，严格策略比较需要多次重复并报告分布。

## 4. Memory 指令识别范围有限

当前 `MemoryManager.apply_explicit_instruction()` 为固定实验提供最小显式规则，主要识别：

```text
记住：...
实验重点更新为：...
```

它用于观察 Memory Lifecycle，没有实现通用的自动 Memory Extraction、实体归一化、置信度、冲突合并或 Memory Retrieval Ranking。

## 5. Retry 只覆盖最小 Tool Execution Recovery

Step 10.1 只实现：

```text
max_retry = 1
```

并验证 transient failure 与 retry exhausted。当前没有继续实现：

```text
exponential backoff
jitter
circuit breaker
idempotency policy
error-class-specific retry
fallback routing
```

## 6. MCP Probe 只迁移一个本地工具

MCP 实验采用：

```text
stdio transport
single local server
read_document only
```

当前范围没有覆盖远程 MCP、多 Server、认证、权限、网络故障与 Server 生命周期治理。

## 7. Reviewer 是最小局部验证器

Reviewer：

```text
独立 system prompt
无 Tools
只接收 task + draft + evidence
不继承母 Agent Conversation / Memory
```

Evidence 由母 Agent 选择，因此 Reviewer 的独立性受委派证据质量影响。当前没有实现 Reviewer 独立取证、并行审阅、Review–Revise Loop 或多 Reviewer 共识。

## 8. Persistence 是单机 SQLite 实验

SQLite 用于验证 Session / Memory 跨 Python 进程恢复。项目没有扩展到并发写入治理、分布式状态存储、租约、事务编排、HA 或多实例一致性。

## 9. Trace 是实验级可审计记录

JSONL / Markdown Trace 适合观察 Model、Tool、Context、Memory 和错误链路。运行时 Trace 可能包含完整 Prompt、Tool Result 和用户输入，因此实际应用需要额外的数据分级、脱敏、保留期和访问控制。

## 10. Provider 兼容范围

当前 Provider 面向 Chat Completions / OpenAI-compatible endpoint，并在本次实验配置中使用 `thinking: disabled`。不同服务对扩展字段、Tool Calling Schema、usage 与 cache 字段的支持可能存在差异，需要按目标 Provider 调整。

## 11. 项目范围

当前仓库定位为 Agent Runtime / Harness 机制体验和可审计实验。复杂 Planning、Multi-Agent Swarm、Web UI、服务化、容器编排、生产权限与 Observability 平台属于更大的系统工程范围。
