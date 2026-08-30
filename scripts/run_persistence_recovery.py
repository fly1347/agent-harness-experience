"""
文件作用：
运行 Step 8.3 的真实 Persistence / Recovery 实验：由一个调度进程连续启动两个彼此独立的 Python worker；前一个进程运行 T04～T08 并把 Session / Memory 写入 SQLite，进程退出后，后一个新进程只凭同一个数据库文件恢复状态并运行 T09。

整体结构：
1）orchestrate：创建干净的 Step 8 数据库与固定 session_id，依次启动 pre_restart / post_restart 两个独立 Python 进程；
2）pre_restart：复用 Managed Context 跑 T04～T08，AgentLoop 每轮自动把 Session / Memory 快照写入 SQLite；
3）post_restart：重新打开 SQLite，恢复完整 Memory Lifecycle；依次执行 Memory-only Probe、Conversation/Memory 冲突 Probe，再恢复原 Session 执行常规 T09；
4）验收：分别确认“长期 Memory 独立恢复”“冲突时 Active Memory 作为当前状态”“完整会话 + Memory 联合恢复”；
5）产物：保留两类 T09 Trace，并输出带时间戳的 Step 8 Persistence / Recovery Markdown 报告。

职责边界：
- 调度进程只负责启动两个 worker 并传递 db_path / session_id，不持有 Agent 的 Session 或 Memory 对象；
- 真正的运行状态只通过 SQLite 跨越 pre_restart 进程退出边界；
- 本脚本只完成 T08 -> restart -> T09，并额外提供 Memory-only 隔离 Probe；T01～T10 的完整连续 Replay 留给 Step 9。
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from dotenv import load_dotenv

from mini_agent_harness.core.agent_loop import AgentLoop, SYSTEM_PROMPT
from mini_agent_harness.core.context import ContextBuilder
from mini_agent_harness.core.memory import MemoryManager
from mini_agent_harness.core.session import Session
from mini_agent_harness.core.tools import ToolRegistry
from mini_agent_harness.core.tracing import TraceLogger
from mini_agent_harness.providers.openai_compatible import OpenAICompatibleProvider
from mini_agent_harness.storage.sqlite_store import SQLiteStore


_PRE_RESTART_TURNS = ("T04", "T05", "T06", "T07", "T08")
_RECOVERY_TURN = "T09"
_EXPECTED_RECOVERED_FOCUS = "Context Management + 评估信度"


class _NoTools:
    """Memory-only probe 使用的空工具集，确保答案不能靠 fixture 工具重新取得。"""

    schemas: list[dict[str, Any]] = []

    def execute(self, name: str, arguments: dict[str, Any]) -> str:
        """理论上不会被调用；若模型仍构造工具调用则立即暴露实验异常。"""
        raise RuntimeError(f"Memory-only probe 不允许工具调用: {name}")


def _project_root() -> Path:
    """返回当前实验项目根目录。"""
    return Path(__file__).resolve().parents[1]


def _load_turns(path: Path) -> dict[str, dict[str, Any]]:
    """按 turn_id 建立 Scenario 索引，供两个独立 worker 精确读取固定轮次。"""
    data = json.loads(path.read_text(encoding="utf-8"))
    return {turn["turn_id"]: turn for turn in data["turns"]}


def _trace_path(
    project_root: Path,
    *,
    phase: str,
    session_id: str,
    turn_id: str,
) -> Path:
    """生成带阶段、Session、Turn 和微秒时间戳的独立 Trace 文件名。"""
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    return (
        project_root
        / "artifacts"
        / "traces"
        / f"step8-{phase}-{session_id[:8]}-{turn_id.lower()}-{stamp}.jsonl"
    )


def _event_payloads(tracer: TraceLogger, event: str) -> list[dict[str, Any]]:
    """读取当前 Trace 中指定事件的 payload。"""
    return [
        record["payload"]
        for record in tracer.records
        if record["event"] == event
    ]


def _runtime_objects(project_root: Path) -> tuple[OpenAICompatibleProvider, ToolRegistry]:
    """创建两个 worker 共用配置口径的真实模型 Provider 与工具注册表。"""
    load_dotenv(project_root / ".env")
    provider = OpenAICompatibleProvider()
    tools = ToolRegistry(
        fixtures_dir=project_root / "fixtures",
        notes_file=project_root / "artifacts" / "notes.md",
    )
    return provider, tools


def _run_pre_restart(
    *,
    project_root: Path,
    db_path: Path,
    session_id: str,
) -> None:
    """第一个 Python 进程：运行 T04～T08，并让 AgentLoop 自动把最终状态写入 SQLite。"""
    turns = _load_turns(project_root / "scenarios" / "multi_turn_context.json")
    provider, tools = _runtime_objects(project_root)
    session = Session(session_id=session_id)
    memory = MemoryManager()
    context_builder = ContextBuilder(
        "managed",
        managed_recent_n=6,
        summary_trigger_messages=8,
    )

    print("\n===== STEP 8 PRE-RESTART PROCESS =====")
    print(f"pid: {os.getpid()}")
    print(f"session_id: {session_id}")
    print(f"db: {db_path}")
    print("turns: T04..T08")

    with SQLiteStore(db_path) as store:
        for turn_id in _PRE_RESTART_TURNS:
            turn = turns[turn_id]
            tracer = TraceLogger(
                _trace_path(
                    project_root,
                    phase="pre-restart",
                    session_id=session_id,
                    turn_id=turn_id,
                )
            )
            agent = AgentLoop(
                provider=provider,
                tools=tools,
                tracer=tracer,
                context_builder=context_builder,
                memory_manager=memory,
                state_store=store,
            )

            print(f"\n===== {turn_id} =====")
            print(turn["user_input"])
            agent.run(turn["user_input"], session=session)

        active = memory.retrieve_active(key="experiment_focus")
        if not active or active[0].value != _EXPECTED_RECOVERED_FOCUS:
            raise RuntimeError(
                "Pre-restart active memory mismatch: "
                f"{[item.value for item in active]}"
            )

        persisted_session = store.load_session(session_id)
        if persisted_session is None:
            raise RuntimeError("Session was not persisted before restart.")

        print("\n===== PRE-RESTART READY =====")
        print(f"messages_persisted: {len(persisted_session.messages)}")
        print(f"memory_records_persisted: {len(memory.records)}")
        print(f"active_memory: {active[0].value}")


def _write_recovery_report(
    path: Path,
    *,
    session_id: str,
    db_path: Path,
    recovered_message_count: int,
    recovered_memory_count: int,
    recovered_focus: str,
    answer: str,
    memory_only_answer: str,
    conflict_answer: str,
    checks: dict[str, bool],
    trace_path: Path,
    memory_only_trace_path: Path,
    conflict_trace_path: Path,
) -> None:
    """生成 Step 8 的简洁人工验收报告。"""
    passed = sum(1 for value in checks.values() if value)
    lines = [
        "# Step 8 Persistence / Recovery",
        "",
        f"- Session: `{session_id}`",
        f"- SQLite: `{db_path}`",
        "- Boundary: `T08 -> Python process exit -> new Python process -> T09`",
        f"- Checks: **{passed}/{len(checks)} PASS**",
        "",
        "## Recovered State Before T09",
        "",
        f"- Conversation messages: **{recovered_message_count}**",
        f"- Memory records: **{recovered_memory_count}**",
        f"- Active experiment_focus: `{recovered_focus}`",
        "",
        "## Memory-only Restart Probe",
        "",
        "新 Python 进程中创建全新 Session，不恢复旧 Conversation History；只注入从 SQLite 恢复的 Active Memory。",
        "",
        memory_only_answer,
        "",
        f"- Trace: `{memory_only_trace_path.with_suffix('.md')}`",
        "",
        "## Conversation / Memory Conflict Probe",
        "",
        "人为构造旧 Summary 状态与 Active Memory 新状态冲突；要求当前值只采用 Active Memory。",
        "",
        conflict_answer,
        "",
        f"- Trace: `{conflict_trace_path.with_suffix('.md')}`",
        "",
        "## T09 Checks",
        "",
    ]
    for name, ok in checks.items():
        lines.append(f"- {'PASS' if ok else 'FAIL'} `{name}`")

    lines.extend(
        [
            "",
            "## T09 Answer",
            "",
            answer,
            "",
            "## Trace",
            "",
            f"- Memory-only: `{memory_only_trace_path.with_suffix('.md')}`",
            f"- Memory-only JSONL: `{memory_only_trace_path}`",
            f"- Conflict probe: `{conflict_trace_path.with_suffix('.md')}`",
            f"- Conflict probe JSONL: `{conflict_trace_path}`",
            f"- Restored conversation T09: `{trace_path.with_suffix('.md')}`",
            f"- Restored conversation T09 JSONL: `{trace_path}`",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _run_post_restart(
    *,
    project_root: Path,
    db_path: Path,
    session_id: str,
) -> None:
    """第二个全新 Python 进程：只从 SQLite 恢复状态，然后运行 T09。"""
    turns = _load_turns(project_root / "scenarios" / "multi_turn_context.json")
    provider, tools = _runtime_objects(project_root)

    print("\n===== STEP 8 POST-RESTART PROCESS =====")
    print(f"pid: {os.getpid()}")
    print(f"session_id: {session_id}")
    print(f"db: {db_path}")

    with SQLiteStore(db_path) as store:
        session = store.load_session(session_id)
        if session is None:
            raise RuntimeError(f"Cannot recover session: {session_id}")

        persisted_records = store.load_memory_records()
        memory = MemoryManager.from_records(persisted_records)
        active = memory.retrieve_active(key="experiment_focus")
        recovered_focus = active[0].value if active else ""
        recovered_message_count = len(session.messages)

        if recovered_focus != _EXPECTED_RECOVERED_FOCUS:
            raise RuntimeError(
                "Recovered active memory mismatch: "
                f"{recovered_focus!r}"
            )

        # 独立验证长期 Memory：新进程 + 全新 Session + 无工具。
        # 这里故意不恢复旧 Conversation History，使回答来源与会话历史解耦。
        memory_only_tracer = TraceLogger(
            _trace_path(
                project_root,
                phase="memory-only-post-restart",
                session_id=session_id,
                turn_id=_RECOVERY_TURN,
            )
        )
        memory_only_agent = AgentLoop(
            provider=provider,
            tools=_NoTools(),
            tracer=memory_only_tracer,
            context_builder=ContextBuilder("managed"),
            memory_manager=memory,
            state_store=None,
        )
        memory_only_session = Session()
        memory_only_answer = memory_only_agent.run(
            turns[_RECOVERY_TURN]["user_input"],
            session=memory_only_session,
        )
        memory_only_events = [
            record["event"]
            for record in memory_only_tracer.records
        ]

        # 独立验证冲突优先级：故意制造“历史摘要旧值 A + Active Memory 新值 B”。
        # 这个 Probe 只测同一事项发生冲突时的当前值判定，不测试摘要完整性。
        conflict_session = Session()
        conflict_session.add_message(
            {"role": "system", "content": SYSTEM_PROMPT}
        )
        conflict_session.add_message(
            {
                "role": "user",
                "content": "记住：本次实验当前重点是‘评估对象分层’和‘指标边界’。",
            }
        )
        conflict_session.add_message(
            {"role": "assistant", "content": "已记录旧实验重点。"}
        )
        for index in range(1, 5):
            conflict_session.add_message(
                {"role": "user", "content": f"较早的占位问题 {index}"}
            )
            conflict_session.add_message(
                {"role": "assistant", "content": f"较早的占位回答 {index}"}
            )

        conflict_tracer = TraceLogger(
            _trace_path(
                project_root,
                phase="memory-conflict-post-restart",
                session_id=session_id,
                turn_id=_RECOVERY_TURN,
            )
        )
        conflict_agent = AgentLoop(
            provider=provider,
            tools=_NoTools(),
            tracer=conflict_tracer,
            context_builder=ContextBuilder(
                "managed",
                managed_recent_n=2,
                summary_trigger_messages=3,
            ),
            memory_manager=memory,
            state_store=None,
        )
        conflict_answer = conflict_agent.run(
            turns[_RECOVERY_TURN]["user_input"],
            session=conflict_session,
        )
        conflict_events = [
            record["event"]
            for record in conflict_tracer.records
        ]
        conflict_contexts = _event_payloads(
            conflict_tracer,
            "CONTEXT_BUILD",
        )
        conflict_summary = (
            str(conflict_contexts[0].get("summary", ""))
            if conflict_contexts
            else ""
        )

        turn = turns[_RECOVERY_TURN]
        tracer = TraceLogger(
            _trace_path(
                project_root,
                phase="post-restart",
                session_id=session_id,
                turn_id=_RECOVERY_TURN,
            )
        )
        tracer.log(
            "STATE_RECOVER",
            {
                "pid": os.getpid(),
                "session_id": session_id,
                "db_path": str(db_path),
                "message_count_before_t09": recovered_message_count,
                "memory_record_count": len(persisted_records),
                "active_memory": [
                    {
                        "kind": item.kind,
                        "key": item.key,
                        "value": item.value,
                        "status": item.status,
                    }
                    for item in active
                ],
                "reason": (
                    "新 Python 进程重新打开 SQLite；从磁盘恢复完整会话存档（canonical Session） "
                    "与完整 Memory Lifecycle 后再执行 T09。"
                ),
            },
            console={
                "pid": os.getpid(),
                "messages": recovered_message_count,
                "memory_records": len(persisted_records),
                "active_focus": recovered_focus,
            },
        )

        agent = AgentLoop(
            provider=provider,
            tools=tools,
            tracer=tracer,
            context_builder=ContextBuilder(
                "managed",
                managed_recent_n=6,
                summary_trigger_messages=8,
            ),
            memory_manager=memory,
            state_store=store,
        )

        print(f"\n===== {_RECOVERY_TURN} =====")
        print(turn["user_input"])
        answer = agent.run(turn["user_input"], session=session)

        events = [record["event"] for record in tracer.records]
        expected_contains = turn.get("expected_contains") or []
        checks = {
            "session_restored": recovered_message_count > 0,
            "memory_lifecycle_restored": len(persisted_records) >= 2,
            "active_memory_is_latest": recovered_focus == _EXPECTED_RECOVERED_FOCUS,
            "memory_only_fresh_session": len(memory_only_session.messages) == 3,
            "memory_only_read_after_restart": "MEMORY_READ" in memory_only_events,
            "memory_only_inject_after_restart": "MEMORY_INJECT" in memory_only_events,
            "memory_only_answer_contains_expected": all(
                text.lower() in memory_only_answer.lower()
                for text in expected_contains
            ),
            "conflict_summary_contains_old_state": (
                "评估对象分层" in conflict_summary
                and "指标边界" in conflict_summary
            ),
            "conflict_memory_read": "MEMORY_READ" in conflict_events,
            "conflict_memory_inject": "MEMORY_INJECT" in conflict_events,
            "conflict_answer_uses_latest_state": all(
                text.lower() in conflict_answer.lower()
                for text in expected_contains
            ),
            "conflict_answer_excludes_old_state": (
                "评估对象分层" not in conflict_answer
                and "指标边界" not in conflict_answer
            ),
            "memory_read_after_restart": "MEMORY_READ" in events,
            "memory_inject_after_restart": "MEMORY_INJECT" in events,
            "t09_answer_contains_expected": all(
                text.lower() in answer.lower()
                for text in expected_contains
            ),
            "t09_no_tool_call": not _event_payloads(tracer, "TOOL_CALL"),
        }

        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        report_path = (
            project_root
            / "artifacts"
            / "reports"
            / f"step8-persistence-recovery-{stamp}.md"
        )
        _write_recovery_report(
            report_path,
            session_id=session_id,
            db_path=db_path,
            recovered_message_count=recovered_message_count,
            recovered_memory_count=len(persisted_records),
            recovered_focus=recovered_focus,
            answer=answer,
            memory_only_answer=memory_only_answer,
            conflict_answer=conflict_answer,
            checks=checks,
            trace_path=tracer.path,
            memory_only_trace_path=memory_only_tracer.path,
            conflict_trace_path=conflict_tracer.path,
        )

        passed = sum(1 for value in checks.values() if value)
        print("\n===== STEP 8 RECOVERY RESULT =====")
        print(f"Recovery checks: {passed}/{len(checks)} PASS")
        print(f"Report: {report_path}")
        print(f"Trace:  {tracer.md_path}")

        if not all(checks.values()):
            failed = [name for name, ok in checks.items() if not ok]
            raise RuntimeError(f"Step 8 recovery checks failed: {failed}")


def _orchestrate(project_root: Path, db_path: Path) -> None:
    """启动两个真正独立的 Python worker，让状态只能通过 SQLite 穿过重启边界。"""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()

    session_id = uuid4().hex
    script = Path(__file__).resolve()
    base = [
        sys.executable,
        str(script),
        "--db",
        str(db_path),
        "--session-id",
        session_id,
    ]

    print("\n===== STEP 8 ORCHESTRATOR =====")
    print(f"orchestrator_pid: {os.getpid()}")
    print(f"session_id: {session_id}")
    print(f"db: {db_path}")
    print("下面会先启动 pre_restart worker；它退出后，再启动新的 post_restart worker。")

    subprocess.run(
        [*base, "--phase", "pre_restart"],
        cwd=project_root,
        check=True,
    )
    print("\n----- pre_restart worker 已退出；现在启动全新的 Python 进程 -----")
    subprocess.run(
        [*base, "--phase", "post_restart"],
        cwd=project_root,
        check=True,
    )


def main() -> None:
    """按 phase 运行 Step 8 worker；默认 orchestrate 自动跨越真实进程边界。"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase",
        choices=("orchestrate", "pre_restart", "post_restart"),
        default="orchestrate",
    )
    parser.add_argument(
        "--db",
        default="artifacts/state/step8-persistence-recovery.db",
        help="SQLite path; relative paths are resolved from project root.",
    )
    parser.add_argument("--session-id", default=None)
    args = parser.parse_args()

    project_root = _project_root()
    db_path = Path(args.db)
    if not db_path.is_absolute():
        db_path = project_root / db_path

    if args.phase == "orchestrate":
        _orchestrate(project_root, db_path)
        return

    if not args.session_id:
        raise SystemExit("--session-id is required for worker phases.")

    if args.phase == "pre_restart":
        _run_pre_restart(
            project_root=project_root,
            db_path=db_path,
            session_id=args.session_id,
        )
        return

    _run_post_restart(
        project_root=project_root,
        db_path=db_path,
        session_id=args.session_id,
    )


if __name__ == "__main__":
    main()
