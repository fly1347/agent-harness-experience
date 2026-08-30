# ragas 指标
2026-08-05

ragas官网当前 PyPI 最新版是 **ragas 0.4.3，2026-01-13 发布**。
官网 stable 的 available metrics 按任务分成 **RAG / Nvidia Metrics / Agent or Tool / Natural Language Comparison / SQL / General Purpose / Other tasks** 这些类。

**`LLMContextPrecisionWithReference`** 是 legacy API 里的名字；官网现在更推荐 collections API 里的 `ContextPrecision`，但 legacy 口径仍明确写着：有 `user_input`、`reference`、`retrieved_contexts` 时，用 LLM 比较每个 retrieved context 和 reference 来估计相关性。

## RAGAS 最新指标口径补充

截至 `ragas 0.4.3`，RAGAS 官网的 available metrics 已经不只包含传统 RAG 五项指标，而是按应用场景分为几类：

```text
Retrieval Augmented Generation；
Nvidia Metrics；
Agents or Tool use cases；
Natural Language Comparison；
SQL；
General purpose；
Other tasks。
```

其中 RAG 相关指标包括：

```text
Context Precision；
Context Recall；
Context Entities Recall；
Noise Sensitivity；
Response Relevancy；
Faithfulness；
Multimodal Faithfulness；
Multimodal Relevance。
```

自然语言对比、Agent、SQL、通用评估等指标也在 RAGAS 中提供，但它们不一定都是“RAG 主链路指标”。在项目文档中可以区分：

```text
RAG 主链路质量指标；
答案对比指标；
Agent / Tool 指标；
SQL 指标；
通用 Judge 指标；
传统文本匹配指标。
```

---

### LLMContextPrecisionWithReference

**LLMContextPrecisionWithReference = 带参考答案的 LLM 上下文精确率**

`LLMContextPrecisionWithReference` 是 RAGAS legacy API 中的指标名。

它用于在已有参考答案时，判断检索出来的 contexts 是否相关，并且相关内容是否排得靠前。

它主要比较：

```text
user_input；
reference；
retrieved_contexts。
```

计算时，评估模型会逐个判断 `retrieved_contexts` 中的片段是否有助于支撑 `reference`。

直白理解：

```text
这道题有参考答案；
检索器返回了一组上下文；
RAGAS 判断这些上下文里，哪些真的能支撑参考答案；
相关上下文越靠前，分数越高。
```

它适合观察：

```text
检索排序是否合理；
关键证据是否靠前；
无关 chunk 是否挤占了前排位置；
进入 prompt 的上下文质量是否足够好。
```

注意：

```text
这个指标不是单纯的 source 命中率；
它会用 LLM 判断 context 和 reference 的语义关系；
所以结果会受 judge model、prompt、reference 质量影响。
```

在本项目中，CSV 字段名是：

```text
llm_context_precision_with_reference
```

可以解释为：

```text
带参考答案的 LLM 上下文精确率。
```

它比泛称 `context_precision` 更具体。

---

### ContextPrecision

**ContextPrecision = 上下文精确率**

`ContextPrecision` 是新版 collections API 中的推荐写法。

它评估检索器是否把相关 chunk 排在无关 chunk 前面。

它主要比较：

```text
user_input；
reference；
retrieved_contexts。
```

它关注的是排序质量：

```text
相关 chunk 越靠前，分数越高；
无关 chunk 排到前面，分数下降。
```

如果项目里实际使用的是 legacy API 的 `LLMContextPrecisionWithReference`，文档中可以写：

```text
本项目实际记录字段为 llm_context_precision_with_reference，
对应 RAGAS legacy API 的 LLMContextPrecisionWithReference；
其核心含义属于 context precision 家族。
```

---

### ContextUtilization

**ContextUtilization = 上下文利用率 / 上下文使用精确率**

`ContextUtilization` 也是 context precision 家族里的一个口径。

它和 `ContextPrecision` 的区别是：

```text
ContextPrecision：比较 retrieved_contexts 和 reference；
ContextUtilization：比较 retrieved_contexts 和 response。
```

也就是说：

```text
有参考答案时，看 context 是否支撑 reference；
没有参考答案但有生成答案时，看 context 是否支撑 response。
```

它适合观察：

```text
模型生成答案实际使用了哪些上下文；
检索结果是否对最终回答有用；
答案是否主要来自有用 context。
```

注意：

```text
ContextUtilization 高，不代表答案一定正确；
它只说明 retrieved contexts 和 response 的关系较强。
如果 response 本身错了，仍然需要看 correctness / factual correctness / faithfulness。
```

---

### NonLLMContextPrecisionWithReference

**NonLLMContextPrecisionWithReference = 带参考上下文的非 LLM 上下文精确率**

这个指标不用 LLM 判断 context 是否相关，而是用字符串相似度等非 LLM 方法比较：

```text
retrieved_contexts；
reference_contexts。
```

它适合在有标准参考上下文时，做更稳定、更便宜的检索评估。

直白理解：

```text
我知道标准证据片段是什么；
检索器返回了一批片段；
不用 LLM，只看返回片段和标准片段像不像、是否匹配。
```

适合观察：

```text
检索结果是否接近标准证据；
固定回归集下的检索稳定性；
不想引入 judge model 波动时的检索评估。
```

注意：

```text
它依赖 reference_contexts；
如果没有人工标好的标准证据片段，就不好用。
```

---

### IDBasedContextPrecision

**IDBasedContextPrecision = 基于 ID 的上下文精确率**

这个指标直接比较：

```text
retrieved_context_ids；
reference_context_ids。
```

它不看文本内容，只看 ID 是否命中。

直白理解：

```text
标准答案需要 doc_1、doc_4；
检索结果返回 doc_1、doc_2、doc_3、doc_4；
命中 2 个标准 ID。
```

它适合本项目这种已经有 source_id / chunk_id / doc_id 的工程场景。

适合观察：

```text
检索是否命中标准 source；
chunk_id 是否命中标准证据；
source_id 是否命中预期文档；
不依赖 LLM 的检索命中情况。
```

注意：

```text
ID 命中只说明“找到了预期来源”；
不说明片段内容是否真的足够支撑答案；
也不说明生成答案是否正确。
```

---

### NonLLMContextRecall

**NonLLMContextRecall = 非 LLM 上下文召回率**

这个指标比较：

```text
retrieved_contexts；
reference_contexts。
```

它看的是标准参考上下文中有多少被检索结果覆盖。

直白理解：

```text
标准证据一共有 3 段；
检索结果覆盖了其中 2 段；
召回率就是 2 / 3。
```

它适合观察：

```text
关键证据是否漏召回；
expected evidence 是否被覆盖；
检索 topk 是否足够；
chunk 切分或索引策略是否导致漏召回。
```

注意：

```text
它需要 reference_contexts；
如果只有 reference answer，没有标准证据片段，就更适合用 LLM Context Recall。
```

---

### IDBasedContextRecall

**IDBasedContextRecall = 基于 ID 的上下文召回率**

这个指标直接比较：

```text
retrieved_context_ids；
reference_context_ids。
```

它看标准 ID 中有多少被检索结果命中。

直白理解：

```text
期望命中 4 个 source_id；
检索结果命中了 3 个；
IDBasedContextRecall = 3 / 4。
```

它适合本项目这种 expected source / expected evidence 已经结构化的场景。

适合观察：

```text
期望 source 是否被召回；
expected evidence 是否完整进入 topk；
某次改动是否导致标准证据漏召回。
```

它和项目自定义字段关系很近：

```text
exp_count        = 期望源数；
overlap_count    = 实际命中源数；
coverage_ratio   = overlap_count / exp_count。
```

区别是：

```text
IDBasedContextRecall 是 RAGAS 指标名；
coverage_ratio 是项目内部更直白的工程统计字段。
```

---

### ContextEntityRecall

**ContextEntityRecall = 上下文实体召回率**

这个指标看 retrieved contexts 是否覆盖 reference 中的重要实体。

它主要比较：

```text
reference；
retrieved_contexts。
```

它会抽取 reference 中的实体，再看这些实体有多少也出现在 retrieved contexts 中。

直白理解：

```text
参考答案里有若干关键实体；
检索上下文里覆盖了多少；
覆盖越多，实体召回越好。
```

适合观察：

```text
人物、地点、机构、时间、产品名是否被召回；
事实型问答的实体覆盖；
检索结果是否漏掉关键实体。
```

它适合实体密集型场景，例如：

```text
历史问答；
旅游问答；
企业制度问答；
产品文档问答；
项目阶段和文件名问答。
```

注意：

```text
实体召回高，不代表答案完整；
实体出现了，也不代表关系、条件、因果都正确。
```

---

### NoiseSensitivity

**NoiseSensitivity = 噪声敏感度**

这个指标观察系统在有噪声 context 时，是否容易生成错误答案。

它主要比较：

```text
user_input；
reference；
response；
retrieved_contexts。
```

它会检查 response 中的 claims 是否正确，以及这些错误是否受到相关或无关 retrieved documents 的影响。

直白理解：

```text
检索结果里混进了干扰信息；
系统有没有被干扰信息带偏；
答案里有多少错误 claim。
```

分数范围通常是：

```text
0 到 1；
越低越好。
```

适合观察：

```text
无关 chunk 对生成的干扰；
模型是否被噪声带偏；
topk 过大是否引入噪声；
rerank 或 evidence gate 是否减少噪声影响。
```

注意：

```text
NoiseSensitivity 和 faithfulness 不一样；
faithfulness 看答案是否受上下文支撑；
NoiseSensitivity 更关注噪声上下文是否诱发错误。
```

---

### ResponseRelevancy / AnswerRelevancy

**ResponseRelevancy = 回答相关性**

新版文档中常写 `Response Relevancy`，代码类名常见为 `AnswerRelevancy`。

它观察 response 是否回应 user_input。

它主要比较：

```text
user_input；
response。
```

它不判断事实正确性，也不判断证据支撑。

直白理解：

```text
用户问 A；
答案有没有正面回应 A；
有没有跑题、空泛、答非所问。
```

适合观察：

```text
答案是否贴题；
答案是否回应问题意图；
答案是否缺少用户真正问的部分；
答案是否塞入无关内容。
```

注意：

```text
相关性高，不代表事实正确；
相关性高，不代表有证据支撑；
它需要和 faithfulness、factual correctness、context recall 一起看。
```

---

### MultimodalFaithfulness

**MultimodalFaithfulness = 多模态忠实性**

这个指标用于图像 + 文本上下文场景。

它比较：

```text
response；
retrieved text contexts；
retrieved visual contexts。
```

它判断 response 中的 claims 是否能从视觉或文本上下文中推出。

直白理解：

```text
答案说的内容，能不能被图片或文本证据支撑。
```

适合观察：

```text
多模态 RAG；
图文问答；
截图分析；
图片证据 + 文本证据混合问答。
```

注意：

```text
需要 vision-capable LLM；
普通文本模型不能完成这个评估。
```

---

### MultimodalRelevance

**MultimodalRelevance = 多模态相关性**

这个指标用于判断 response 是否和视觉 / 文本上下文相关。

它主要比较：

```text
user_input；
response；
retrieved text contexts；
retrieved visual contexts。
```

直白理解：

```text
用户问图里的东西；
答案有没有围绕图像或文本上下文回答；
有没有答到无关内容。
```

适合观察：

```text
多模态问答是否跑题；
图片上下文是否被正确利用；
图文混合证据是否和答案相关。
```

注意：

```text
它看相关性；
不等于完整事实正确性；
也不等于所有细节都被证据支撑。
```

---

## Nvidia Metrics

RAGAS 中还提供一组 Nvidia Metrics：

```text
Answer Accuracy；
Context Relevance；
Response Groundedness。
```

这组指标也是 LLM-as-a-Judge 风格，更偏轻量评分。

---

### AnswerAccuracy

**AnswerAccuracy = 答案准确率 / 答案准确性**

AnswerAccuracy 比较 response 和 reference ground truth 是否一致。

它主要比较：

```text
user_input；
response；
reference。
```

它通过两个 LLM judge prompt 分别打分，再归一化到 0-1。

直白理解：

```text
生成答案和标准答案是否一致；
越一致，分数越高。
```

适合观察：

```text
最终答案是否接近标准答案；
固定问题集下的答案质量；
轻量 answer correctness 替代指标。
```

它和 AnswerCorrectness 的区别：

```text
AnswerAccuracy 更像两个 judge 直接给准确度分；
AnswerCorrectness 更强调 factual similarity + semantic similarity 的组合。
```

---

### ContextRelevance

**ContextRelevance = 上下文相关性**

ContextRelevance 判断 retrieved_contexts 是否和 user_input 相关。

它主要比较：

```text
user_input；
retrieved_contexts。
```

直白理解：

```text
检索出来的上下文，和用户问题有没有关系。
```

适合观察：

```text
检索结果整体是否相关；
无关 chunk 是否过多；
retrieval / rerank 是否改善上下文质量。
```

它和 ContextPrecision / ContextRecall 的区别：

```text
ContextPrecision 更看相关 context 是否排在前面；
ContextRecall 更看参考答案需要的信息是否被覆盖；
ContextRelevance 更像整体相关性判断。
```

---

### ResponseGroundedness

**ResponseGroundedness = 回答扎根度 / 回答证据扎根性**

ResponseGroundedness 判断 response 是否被 retrieved_contexts 支撑。

它主要比较：

```text
response；
retrieved_contexts。
```

直白理解：

```text
答案有没有扎在检索上下文里；
答案里的说法能不能从上下文中找到依据。
```

适合观察：

```text
答案是否脱离证据；
答案是否出现上下文外扩展；
生成模型是否遵守 evidence。
```

它和 Faithfulness 很接近。

可以这样区分：

```text
Faithfulness：RAGAS 经典忠实性指标，按 claim 支撑比例计算；
ResponseGroundedness：Nvidia Metrics 里的扎根性判断，偏轻量 judge 打分。
```

---

## Natural Language Comparison

自然语言对比类指标主要比较 response 和 reference。

它们不一定是 RAG 专用，但常用于 RAG 最终答案评估。

---

### FactualCorrectness

**FactualCorrectness = 事实正确性**

FactualCorrectness 比较 response 和 reference 的事实是否一致。

它主要比较：

```text
response；
reference。
```

它会把 response 和 reference 拆成 claims，再用自然语言推理判断事实重叠。

常见模式包括：

```text
precision；
recall；
f1。
```

直白理解：

```text
答案里说的事实，有多少能在参考答案中找到；
参考答案里的事实，又有多少被答案覆盖。
```

适合观察：

```text
答案事实是否正确；
答案有没有多编事实；
答案有没有漏掉参考答案关键事实；
多点答案的事实覆盖情况。
```

它和 AnswerCorrectness 的关系：

```text
AnswerCorrectness 更像综合正确性；
FactualCorrectness 更聚焦事实 claim 的重叠。
```

---

### SemanticSimilarity

**SemanticSimilarity = 语义相似度**

SemanticSimilarity 使用 embedding 和 cosine similarity 比较 response 和 reference 的语义接近程度。

它主要比较：

```text
response；
reference。
```

直白理解：

```text
两段话意思像不像。
```

适合观察：

```text
生成答案和参考答案是否语义接近；
措辞不同但意思相近的答案；
开放式答案的相似程度。
```

注意：

```text
语义相似不等于事实完全正确；
数字、时间、实体、条件仍要单独检查。
```

---

### NonLLMStringSimilarity

**NonLLMStringSimilarity = 非 LLM 字符串相似度**

这个指标不用 LLM，而是用传统字符串距离比较 response 和 reference。

可用距离包括：

```text
Levenshtein；
Hamming；
Jaro；
Jaro-Winkler。
```

直白理解：

```text
两段文本字面上有多像。
```

适合观察：

```text
格式固定答案；
短答案；
代码参数；
名称字段；
不想引入 LLM judge 的场景。
```

注意：

```text
字面相似高，不代表语义正确；
字面相似低，也可能只是表达方式不同。
```

---

### BleuScore

**BleuScore = BLEU 分数**

BLEU 用 n-gram precision 和 brevity penalty 比较 response 和 reference。

直白理解：

```text
生成文本里有多少短语片段和参考答案重合。
```

适合：

```text
机器翻译；
固定表达较强的生成任务；
传统 NLP 对比。
```

在 RAG 问答里要谨慎使用：

```text
自然语言问答允许改写；
BLEU 可能低估合理改写答案。
```

---

### RougeScore

**RougeScore = ROUGE 分数**

ROUGE 用 n-gram overlap、最长公共子序列等方式比较 response 和 reference。

它可以看：

```text
precision；
recall；
fmeasure。
```

直白理解：

```text
生成答案和参考答案有多少词片段重合。
```

适合：

```text
摘要；
参考答案覆盖检查；
传统文本生成评估。
```

注意：

```text
ROUGE 偏字面重合；
它不能可靠判断事实是否正确。
```

---

### CHRFScore

**CHRFScore = 字符 n-gram F 分数**

CHRF 用字符级 n-gram F-score 比较 response 和 reference。

直白理解：

```text
两段文本在字符片段层面有多接近。
```

它相比 BLEU 更关注字符层面的 precision 和 recall。

适合：

```text
形态变化丰富的语言；
短文本；
允许一定改写但仍希望看字面接近度的场景。
```

注意：

```text
它仍然是非 LLM 字面指标；
不能替代事实正确性和证据支撑检查。
```

---

### StringPresence

**StringPresence = 字符串存在检查**

StringPresence 检查 response 是否包含 reference 指定的字符串。

输出通常是：

```text
包含：1；
不包含：0。
```

直白理解：

```text
答案里有没有出现某个关键词或短语。
```

适合：

```text
关键词必须出现；
特定实体必须出现；
格式检查；
工具参数检查。
```

注意：

```text
出现关键词不代表答案正确；
它只适合做窄口径检查。
```

---

### ExactMatch

**ExactMatch = 完全匹配**

ExactMatch 检查 response 是否和 reference 完全一致。

输出通常是：

```text
完全一致：1；
不一致：0。
```

适合：

```text
固定答案；
分类标签；
工具参数；
结构化字段；
严格格式输出。
```

不适合：

```text
开放式问答；
允许改写的自然语言答案；
需要语义判断的答案。
```

---

## Agent / Tool Metrics

Agent / Tool 类指标用于评估工具调用和多轮任务完成情况。

---

### TopicAdherence

**TopicAdherence = 主题遵守度**

TopicAdherence 判断系统是否只在预定义主题范围内回答。

它主要比较：

```text
user_input；
reference_topics。
```

它可以计算：

```text
precision；
recall；
f1。
```

直白理解：

```text
该答的主题内问题有没有答；
不该答的主题外问题有没有乱答。
```

适合：

```text
客服机器人；
企业知识库问答；
有业务边界的助手；
需要拒答域外问题的系统。
```

---

### ToolCallAccuracy

**ToolCallAccuracy = 工具调用准确率**

ToolCallAccuracy 判断 agent 调用的工具是否和 expected tool calls 一致。

它主要比较：

```text
user_input 中的实际 tool calls；
reference_tool_calls。
```

它检查：

```text
工具名是否正确；
参数是否正确；
调用顺序是否正确。
```

直白理解：

```text
该调用哪个工具；
参数该怎么填；
顺序该不该严格一致。
```

适合：

```text
多步 agent 工作流；
工具选择测试；
工具参数回归；
流程编排验收。
```

---

### ToolCallF1

**ToolCallF1 = 工具调用 F1 分数**

ToolCallF1 用 precision / recall / F1 来评估工具调用。

它比较：

```text
实际 tool calls；
reference_tool_calls。
```

直白理解：

```text
该调用的工具有没有调用；
不该调用的工具有没有多调；
调用名称和参数是否匹配。
```

它比 ToolCallAccuracy 更柔和：

```text
ToolCallAccuracy 更看严格匹配；
ToolCallF1 更适合看部分正确。
```

适合：

```text
工具调用调试；
agent 早期迭代；
多工具流程的部分命中观察。
```

---

### AgentGoalAccuracy

**AgentGoalAccuracy = Agent 目标完成准确率**

AgentGoalAccuracy 判断 agent 是否完成了用户目标。

它关注的是最终任务结果，而不是中间用了哪些工具。

输出通常是：

```text
完成：1；
未完成：0。
```

直白理解：

```text
用户要订餐；
最终有没有订成。
用户要查资料并总结；
最终有没有完成目标。
```

它分为：

```text
AgentGoalAccuracyWithReference；
AgentGoalAccuracyWithoutReference。
```

适合：

```text
端到端 agent 验收；
多步任务结果判断；
不关心工具细节、只关心最终是否成功的场景。
```

---

## SQL Metrics

SQL 类指标用于评估 text-to-SQL 或 SQL agent。

---

### DataCompyScore

**DataCompyScore = SQL 执行结果对比分数**

DataCompyScore 是 execution-based metric。

它先执行 SQL，再比较 response 和 reference 的结果表。

它可以按：

```text
row；
column。
```

并计算：

```text
precision；
recall；
f1。
```

直白理解：

```text
不只看 SQL 写得像不像；
直接看 SQL 跑出来的结果对不对。
```

适合：

```text
text-to-SQL；
数据查询 agent；
报表问答；
数据库问答验收。
```

---

### SQLSemanticEquivalence

**SQLSemanticEquivalence = SQL 语义等价性**

SQLSemanticEquivalence 不执行 SQL，而是用 LLM 判断生成 SQL 和参考 SQL 在 schema 语境下是否语义等价。

输出通常是：

```text
等价：1；
不等价：0。
```

直白理解：

```text
SQL 写法可以不同；
只要在这个数据库 schema 下结果等价，就算通过。
```

适合：

```text
无法实际执行 SQL；
只想快速判断 SQL 意图是否一致；
SQL 写法不同但语义可能相同的场景。
```

注意：

```text
这是 LLM 判断；
关键数据库任务最好结合执行结果评估。
```

---

## General Purpose Metrics

通用指标适合自定义评估口径。

---

### AspectCritic

**AspectCritic = 方面评判器**

AspectCritic 用自然语言定义一个评价方面，然后让 LLM 判断样本是否符合。

输出通常是二元结果：

```text
符合；
不符合。
```

直白理解：

```text
我定义一个检查点；
评估器判断答案有没有满足这个检查点。
```

适合：

```text
安全性；
有害性；
礼貌性；
是否遵守格式；
是否符合某条业务规则。
```

---

### SimpleCriteriaScoring

**SimpleCriteriaScoring = 简单标准评分**

SimpleCriteriaScoring 用预定义评分标准给 response 打分。

分数可以是：

```text
0-10；
1-5；
自定义离散等级。
```

直白理解：

```text
按我写好的标准，让 judge 给一个分。
```

适合：

```text
清晰度评分；
完整度评分；
可读性评分；
自定义质量分。
```

---

### RubricsBasedScoring

**RubricsBasedScoring = 基于评分规程的评分**

RubricsBasedScoring 用统一 rubric 对所有样本评分。

例如：

```text
1 分：完全错误；
3 分：部分正确；
5 分：完全正确。
```

适合：

```text
固定问题集；
人工评分对齐；
LLM judge 口径固定；
前后版本可比。
```

注意：

```text
rubric 一旦修改，前后分数不宜直接比较。
```

---

### InstanceSpecificRubricsScoring

**InstanceSpecificRubricsScoring = 单样本专属评分规程**

这个指标允许每个样本有自己的 rubric。

直白理解：

```text
不同题目用不同评分标准。
```

适合：

```text
混合任务数据集；
每道题验收条件不同；
不同业务问题有不同判断口径。
```

注意：

```text
灵活性更高；
但横向对比会更难。
```

---

## Other Tasks

### SummaryScore / SummarizationScore

**SummaryScore = 摘要分数**

SummaryScore 判断 summary 是否覆盖 reference_contexts 中的重要信息。

它会先从原文中抽取关键短语，再生成问题，然后检查摘要是否能回答这些问题。

它还可以加入 conciseness score，避免摘要直接复制原文也拿高分。

直白理解：

```text
摘要有没有覆盖原文重点；
摘要是不是太长、像复制原文。
```

适合：

```text
摘要任务；
长文压缩；
会议纪要；
文档摘要；
知识库内容压缩。
```

---

## 使用口径建议

在本项目文档中，可以这样分层：

```text
1. RAG 主链路核心指标：
   faithfulness；
   response_relevancy / answer_relevancy；
   context_precision；
   context_recall；
   context_entity_recall；
   noise_sensitivity。

2. 本项目实际使用过的 RAGAS 字段：
   llm_context_precision_with_reference；
   faithfulness；
   answer_relevancy；
   context_recall / context_precision 相关字段。

3. 检索命中类工程自定义指标：
   any_hit；
   full_hit；
   overlap_count；
   coverage_ratio；
   expected_first_rank。

4. 答案对比类指标：
   answer_accuracy；
   factual_correctness；
   semantic_similarity；
   answer_correctness。

5. Agent / Tool 指标：
   topic_adherence；
   tool_call_accuracy；
   tool_call_f1；
   agent_goal_accuracy。

6. 通用 Judge 指标：
   aspect_critic；
   simple_criteria_scoring；
   rubrics_based_scoring；
   instance_specific_rubrics_scoring。

7. 传统非 LLM 文本指标：
   non_llm_string_similarity；
   bleu；
   rouge；
   chrf；
   string_presence；
   exact_match。

8. SQL 指标：
   datacompy_score；
   sql_semantic_equivalence。
```

核心提醒：

```text
RAGAS 指标名会随版本演进；
文档里最好同时写：
英文指标名；
中文名称；
所属类别；
输入字段；
观察对象；
适用场景；
和本项目字段的对应关系。
```

```

你这个 `11-rag-evaluation-tools-and-scores.md` 现在第 3 节确实只列了 5 个常见输出：`faithfulness / answer_relevancy / answer_correctness / context_precision / context_recall`，而官网当前 available metrics 已经明显扩展了。:contentReference[oaicite:4]{index=4}  
另外，`answer_correctness` 现在在 stable available metrics 主列表里不作为主项列出，官网主列表更偏向 `AnswerAccuracy`、`FactualCorrectness`、`SemanticSimilarity` 这些拆分后的口径；但旧文档和搜索页仍能找到 `AnswerCorrectness` 的说明。:contentReference[oaicite:5]{index=5}
```

[1]: https://pypi.org/project/ragas/ "ragas · PyPI"
[2]: https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/ "List of available metrics - Ragas"
[3]: https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/context_precision/ "Context Precision - Ragas"
