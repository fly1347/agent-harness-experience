# Fixture provenance

These six fixtures are the frozen knowledge materials used by the Agent Harness experiments.
They come from the project's existing RAG evaluation knowledge system; the public repository keeps the same fixture content used during the experiments rather than silently replacing the data after the runs.

## Source mapping

| Public fixture | Original project source |
| --- | --- |
| `01-检索层与证据层评估.md` | `RAG评估知识系统/RAG评估知识系统-04-检索层与证据层评估.md` |
| `02-生成层与引用层评估.md` | `RAG评估知识系统/RAG评估知识系统-05-生成层与引用层评估.md` |
| `03-实验变量与回归集设计.md` | `RAG评估知识系统/RAG评估知识系统-06-实验变量与回归集设计.md` |
| `04-评估工具与RAGAS指标拆解.md` | `RAG评估知识系统/RAG评估知识系统-08-评估工具与RAGAS指标拆解.md` |
| `05-评估信度与阶段映射.md` | `RAG评估知识系统/RAG评估知识系统-09-评估信度与阶段映射.md` |
| `06-ragas-metrics.md` | `glossary/12-ragas-metrics.md` |

## Publication boundary

The fixtures are used only as deterministic local documents for tool reading, comparison, multi-turn reference, and context-pressure experiments. They do not contain recruiting JDs, private notes, personal preferences, API credentials, or runtime traces.

Files `04` and `06` contain links to public RAGAS documentation as references. Publication cleanup is limited to provenance metadata and removal of tracking-only URL query parameters; the experimental knowledge content is otherwise kept frozen.
