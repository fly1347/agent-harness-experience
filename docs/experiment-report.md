# Agent Harness 实验报告

[English](en/experiment-report.md)

## 1. 实验范围

实验从 Plain LLM 开始，按同一代码基线逐层加入运行时能力：

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

核心实验材料固定为 `fixtures/` 中的 6 篇本地文档；多轮实验使用 `scenarios/multi_turn_context.json`。

## 2. 参考模型配置

最终参考运行使用：

```text
Provider: DeepSeek Official API
Model: deepseek-v4-flash
Thinking: disabled
Temperature: 0
```

模型驱动的 Tool Calling 即使在固定 Scenario 和低随机配置下仍存在运行间波动，因此本文把 Model Call、Tool Call、Latency 与 Cost 作为具体批次的观测值。

## 3. 核心链路验证

### Plain LLM

建立只有 User → Model → Answer 的基线。

### Tool Calling / Agent Loop

真实跑通：

```text
Model proposes Tool Call
→ Harness executes Python tool
→ Tool Result returned to model
→ Model produces answer or next Tool Call
```

### Multi-turn / Session

同一 Session 中，后续轮次可以利用前序 Conversation 和 Tool Result；Session 由 Harness 保存并在下一轮重新进入 Context Construction。

### Trace

Model Request / Response、Tool、Context、Memory 和运行统计均进入 JSONL / Markdown Trace，用于回放和定位执行路径。

## 4. Context Strategy 完整 Replay

最终固定 Scenario 为 T01～T10，并在 T08 后真实退出 Python worker，再由新进程恢复 Session 与 Memory后继续 T09～T10。

最终参考批次：

| Strategy | Checks | Restart | T10 Answer | T10 no-refetch | Model Calls | Tool Calls | Input Tokens | E2E |
| --- | ---: | --- | --- | --- | ---: | ---: | ---: | ---: |
| Full History | 10/10 | PASS | PASS | YES | 19 | 9 | 162,106 | 33.316 s |
| Last-N | 10/10 | PASS | PASS | NO | 27 | 19 | 40,510 | 36.575 s |
| Managed | 10/10 | PASS | PASS | NO | 29 | 21 | 68,865 | 41.308 s |

本批同时记录：

```text
Full History estimated cost: ¥0.067373
Last-N estimated cost:       ¥0.046628
Managed estimated cost:      ¥0.060160
```

这些数字描述本次参考运行。三种策略均通过最终答案检查和 Restart Recovery；差异体现在模型当轮可见信息、重新取证路径以及运行成本。

## 5. Tool Calling 的运行间波动

相同 Scenario、Context Strategy 和 Tool Schema 在先后两批运行中出现了不同 Tool Path。例如：

```text
Last-N T09
一批：save_note
最终批：0 Tool

Managed T09
一批：list_documents
最终批：save_note

Managed T10
一批：read 01 → read 02
最终批：短名 read 失败 → list → 完整文件名重读
```

因此实验报告把两类信息分开：

```text
Context Strategy
→ 决定当前 Model Call 可见的信息

Model Tool Decision
→ 决定当轮是否调用工具、调用顺序与恢复路径
```

单次 Tool Call 数适合描述一次运行；稳定性能比较需要重复运行并报告分布。

## 6. Memory Lifecycle

结构化 Memory 完成：

```text
Identify
→ Write
→ Retrieve
→ Inject
→ Update / Supersede
```

固定状态从：

```text
experiment_focus = 评估对象分层 + 指标边界
```

更新为：

```text
experiment_focus = Context Management + 评估信度
```

Step 7 Memory checks：**5/5 PASS**。

旧记录保留为 `superseded`，新记录成为 `active`。

## 7. Persistence / Recovery

SQLite 保存 Session 与 Memory，并用新 Python 进程完成恢复。

Step 8 最终检查：**16/16 PASS**。

额外 Probe 验证：

- 只恢复 Active Memory、使用全新 Session 时，模型仍能回答当前 `experiment_focus`；
- Historical Summary 与 Active Memory 冲突时，当前实现按 Active Memory 作为当前状态。

## 8. 扩展 Probe

### Retry

`max_retry=1` 两个 Case：

```text
transient failure → retry → continue
persistent failure → retry exhausted → terminate
```

结果：**2/2 PASS**。

### MCP

`read_document` 通过独立 stdio MCP Server 提供，验证：连接、Tool Discovery、Call Tool、Result Return 与 Agent Final Answer。

结果：**PASS**。

### Reviewer Subagent

Reviewer 使用独立上下文，只收到 `task + draft + evidence`，没有 Tool，也没有继承母 Agent 的 Conversation / Memory；Review Result 返回母 Agent后生成最终答案。

结果：**PASS**。

## 9. 最终验收

公开候选目录离线验证：

```text
31/31 unittest PASS
compileall PASS
```

最终在线 smoke：

```text
3 Context Strategies: 10/10 each
Restart Recovery: PASS
Retry: 2/2 PASS
MCP: PASS
Reviewer Subagent: PASS
```

实验主体到此完成。当前实现的限制与后续工程空间见 [已知限制](known-limitations.md)。
