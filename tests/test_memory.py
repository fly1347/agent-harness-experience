"""
文件作用：
验证 Step 7A 的最小 Memory Lifecycle，不访问真实模型、工具或网络；重点确认写入、读取、显式替换和实例隔离行为稳定。

整体结构：
1）MemoryManager 单元测试：验证 write、retrieve_active、supersede、实例隔离和显式记忆指令解析；
2）AgentLoop 联合测试：验证下一轮能读取并注入 active memory，且 Trace 出现 MEMORY_READ / MEMORY_INJECT；
3）更新测试：验证明确更新指令生成 MEMORY_SUPERSEDE，旧记录保留、新记录成为唯一 active；
4）History 分离测试：换成全新 Session 后，旧 Conversation 不存在，但同一 MemoryManager 的 active memory 仍能进入模型请求；
5）Step 8.2 持久化联合测试：AgentLoop 正常结束后自动把 Session 与 Memory 快照写入 SQLite；
6）Step 8.3 恢复装配测试：SQLite 读出的生命周期记录可重建独立 MemoryManager。
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

from mini_agent_harness.core.agent_loop import AgentLoop
from mini_agent_harness.core.memory import MemoryManager
from mini_agent_harness.core.session import Session
from mini_agent_harness.storage.sqlite_store import SQLiteStore


class _FakeMessage:
    def __init__(self, content: str) -> None:
        self.role = "assistant"
        self.content = content
        self.tool_calls = None

    # 把测试消息转成运行器消费的字典结构。
    def model_dump(self, exclude_none: bool = False) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


class _FakeProvider:
    model = "fake-model"

    def __init__(self) -> None:
        self.requests: list[list[dict[str, object]]] = []

    # 保存请求快照并返回固定回复，供测试检查模型实际收到的上下文。
    def complete(self, messages, tools=None):
        request_messages = deepcopy(messages)
        self.requests.append(request_messages)
        message = _FakeMessage(f"answer-{len(self.requests)}")
        return SimpleNamespace(
            request={
                "model": self.model,
                "messages": request_messages,
                "tools": tools or [],
            },
            response={
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": message.content,
                        }
                    }
                ]
            },
            message=message,
            latency_ms=1.0,
            model=self.model,
            usage={
                "prompt_tokens": 1,
                "completion_tokens": 1,
                "prompt_cache_hit_tokens": 0,
                "prompt_cache_miss_tokens": 1,
            },
        )


class _FakeTools:
    schemas: list[dict[str, object]] = []

    # 意外收到工具调用时立即报错，暴露测试场景偏离。
    def execute(self, name, arguments):
        raise AssertionError("本测试不应调用工具。")


class _RecordingTracer:
    def __init__(self) -> None:
        self.records: list[dict[str, object]] = []

    # 将测试事件保存到内存，供用例检查执行顺序和内容。
    def log(self, event, payload, console=None):
        self.records.append({"event": event, "payload": deepcopy(payload)})

    # 返回空统计，避免测试替身写入真实运行报告。
    def finalize(self, **kwargs):
        return {}


class MemoryManagerTests(unittest.TestCase):
    def test_write_creates_active_memory(self) -> None:
        """write 应新增一条 active 结构化记忆并返回其副本。"""
        memory = MemoryManager()

        created = memory.write(
            kind="decision",
            key="experiment_focus",
            value="评估对象分层和指标边界",
        )

        self.assertEqual(created.kind, "decision")
        self.assertEqual(created.key, "experiment_focus")
        self.assertEqual(created.value, "评估对象分层和指标边界")
        self.assertEqual(created.status, "active")
        self.assertEqual(len(memory.records), 1)

    def test_retrieve_active_filters_kind_and_key(self) -> None:
        """retrieve_active 应只返回符合筛选条件的 active 记忆。"""
        memory = MemoryManager()
        memory.write(
            kind="decision",
            key="experiment_focus",
            value="评估对象分层和指标边界",
        )
        memory.write(
            kind="preference",
            key="report_language",
            value="中文",
        )

        by_kind = memory.retrieve_active(kind="decision")
        by_key = memory.retrieve_active(key="report_language")

        self.assertEqual([item.key for item in by_kind], ["experiment_focus"])
        self.assertEqual([item.value for item in by_key], ["中文"])

    def test_supersede_preserves_old_record_and_activates_new_value(self) -> None:
        """supersede 应保留旧版本，同时让新版本成为唯一 active 记录。"""
        memory = MemoryManager()
        old = memory.write(
            kind="decision",
            key="experiment_focus",
            value="评估对象分层和指标边界",
        )

        new = memory.supersede(
            kind="decision",
            key="experiment_focus",
            value="Context Management + 评估信度",
        )

        self.assertEqual(len(memory.records), 2)
        self.assertEqual(memory.records[0].memory_id, old.memory_id)
        self.assertEqual(memory.records[0].status, "superseded")
        self.assertEqual(memory.records[1].memory_id, new.memory_id)
        self.assertEqual(memory.records[1].status, "active")
        self.assertEqual(
            [item.value for item in memory.retrieve_active(key="experiment_focus")],
            ["Context Management + 评估信度"],
        )

    def test_write_rejects_silent_overwrite(self) -> None:
        """已有同 kind + key 的 active 记录时，write 不应静默覆盖。"""
        memory = MemoryManager()
        memory.write(
            kind="fact",
            key="current_step",
            value="Step 7A",
        )

        with self.assertRaises(ValueError):
            memory.write(
                kind="fact",
                key="current_step",
                value="Step 7B",
            )

    def test_manager_instances_have_independent_state(self) -> None:
        """不同 MemoryManager 实例应各自维护独立的进程内状态。"""
        first = MemoryManager()
        second = MemoryManager()
        first.write(
            kind="fact",
            key="current_step",
            value="Step 7A",
        )

        self.assertEqual(len(first.retrieve_active()), 1)
        self.assertEqual(second.retrieve_active(), [])

    def test_explicit_write_instruction_creates_experiment_focus(self) -> None:
        """固定 Scenario 的“记住”指令应映射成 experiment_focus 结构化记忆。"""
        memory = MemoryManager()

        change = memory.apply_explicit_instruction(
            '记住：本次实验当前重点是“评估对象分层”和“指标边界”。'
        )

        self.assertIsNotNone(change)
        self.assertEqual(change.action, "write")
        self.assertEqual(change.record.key, "experiment_focus")
        self.assertEqual(change.record.value, "评估对象分层 + 指标边界")

    def test_agent_loop_reads_and_injects_active_memory_next_turn(self) -> None:
        """Memory 写入后的下一轮应出现读取、注入事件，并进入真实模型请求。"""
        provider = _FakeProvider()
        tracer = _RecordingTracer()
        memory = MemoryManager()
        agent = AgentLoop(
            provider=provider,
            tools=_FakeTools(),
            tracer=tracer,
            memory_manager=memory,
        )
        session = Session()

        agent.run(
            '记住：本次实验当前重点是“评估对象分层”和“指标边界”。',
            session=session,
        )
        tracer.records.clear()
        agent.run("当前实验重点是什么？", session=session)

        memory_messages = [
            item["content"]
            for item in provider.requests[1]
            if item.get("role") == "system"
            and "Active long-term memory" in str(item.get("content"))
        ]
        events = [record["event"] for record in tracer.records]

        self.assertEqual(len(memory_messages), 1)
        self.assertIn("experiment_focus", memory_messages[0])
        self.assertIn("评估对象分层 + 指标边界", memory_messages[0])
        self.assertIn("MEMORY_READ", events)
        self.assertIn("MEMORY_INJECT", events)

    def test_agent_loop_supersedes_memory_from_update_instruction(self) -> None:
        """明确更新实验重点时应保留旧版本，并在 Trace 中记录 MEMORY_SUPERSEDE。"""
        provider = _FakeProvider()
        tracer = _RecordingTracer()
        memory = MemoryManager()
        agent = AgentLoop(
            provider=provider,
            tools=_FakeTools(),
            tracer=tracer,
            memory_manager=memory,
        )
        session = Session()

        agent.run(
            '记住：本次实验当前重点是“评估对象分层”和“指标边界”。',
            session=session,
        )
        tracer.records.clear()
        agent.run(
            '把实验重点更新为“Context Management”和“评估信度”。',
            session=session,
        )

        active = memory.retrieve_active(key="experiment_focus")
        events = [record["event"] for record in tracer.records]

        self.assertEqual(active[0].value, "Context Management + 评估信度")
        self.assertEqual(memory.records[0].status, "superseded")
        self.assertIn("MEMORY_SUPERSEDE", events)

    def test_active_memory_survives_fresh_session_without_old_history(self) -> None:
        """换成全新 Session 后，旧会话历史消失但 active memory 仍应进入模型请求。"""
        provider = _FakeProvider()
        tracer = _RecordingTracer()
        memory = MemoryManager()
        memory.write(
            kind="decision",
            key="experiment_focus",
            value="Context Management + 评估信度",
        )
        agent = AgentLoop(
            provider=provider,
            tools=_FakeTools(),
            tracer=tracer,
            memory_manager=memory,
        )

        agent.run("当前实验重点是什么？", session=Session())

        request = provider.requests[0]
        contents = [str(item.get("content", "")) for item in request]

        self.assertEqual(
            [item.get("role") for item in request],
            ["system", "system", "user"],
        )
        self.assertTrue(
            any("Context Management + 评估信度" in item for item in contents)
        )


    def test_memory_manager_rebuilds_from_persisted_records(self) -> None:
        """SQLite 恢复出的记录应能重建独立 MemoryManager，并保留 active / superseded 状态。"""
        original = MemoryManager()
        original.write(
            kind="decision",
            key="experiment_focus",
            value="评估对象分层 + 指标边界",
        )
        original.supersede(
            kind="decision",
            key="experiment_focus",
            value="Context Management + 评估信度",
        )

        restored = MemoryManager.from_records(original.records)

        self.assertIsNot(restored.records, original.records)
        self.assertEqual(
            [item.status for item in restored.records],
            ["superseded", "active"],
        )
        self.assertEqual(
            [item.value for item in restored.retrieve_active()],
            ["Context Management + 评估信度"],
        )

    def test_agent_loop_persists_session_and_memory_to_sqlite(self) -> None:
        """配置 SQLiteStore 后，AgentLoop 正常结束应自动保存本轮 Session 与 Memory。"""
        with TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "harness.db"
            provider = _FakeProvider()
            tracer = _RecordingTracer()
            memory = MemoryManager()
            session = Session(session_id="persist-session")

            with SQLiteStore(db_path) as store:
                agent = AgentLoop(
                    provider=provider,
                    tools=_FakeTools(),
                    tracer=tracer,
                    memory_manager=memory,
                    state_store=store,
                )
                agent.run(
                    '记住：本次实验当前重点是“评估对象分层”和“指标边界”。',
                    session=session,
                )

            with SQLiteStore(db_path) as reopened:
                restored_session = reopened.load_session(
                    "persist-session"
                )
                restored_memory = reopened.load_memory_records()

            self.assertIsNotNone(restored_session)
            self.assertEqual(
                restored_session.messages,
                session.messages,
            )
            self.assertEqual(len(restored_memory), 1)
            self.assertEqual(
                restored_memory[0].key,
                "experiment_focus",
            )
            self.assertEqual(
                restored_memory[0].value,
                "评估对象分层 + 指标边界",
            )
            self.assertEqual(restored_memory[0].status, "active")
            self.assertIn(
                "STATE_PERSIST",
                [item["event"] for item in tracer.records],
            )


if __name__ == "__main__":
    unittest.main()
