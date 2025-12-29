from __future__ import annotations

import importlib
import json
import sys
import types
from pathlib import Path

import pytest


class _DummySignature:
    def __init__(self, name: str, args: tuple, kwargs: dict, capture: dict):
        self.name = name
        self.args = args
        self.kwargs = kwargs
        capture.setdefault("signatures", []).append(
            {"name": name, "args": args, "kwargs": kwargs}
        )


class _DummyTask:
    def __init__(self, name: str, capture: dict):
        self._name = name
        self._capture = capture

    def s(self, *args, **kwargs):
        return _DummySignature(self._name, args, kwargs, self._capture)


class _DummyChain:
    def __init__(self, capture: dict, *tasks):
        self._capture = capture
        self._tasks = tasks
        capture.setdefault("tasks", []).append(tasks)

    def apply_async(self, *, queue: str | None = None, **kwargs):
        self._capture.setdefault("apply_async_calls", []).append(
            {"queue": queue, "kwargs": kwargs}
        )
        self._capture.setdefault("queues", []).append(queue)
        return None


@pytest.fixture
def decider_stub(monkeypatch):
    capture: dict = {}

    celery_module = types.ModuleType("celery")

    def _fake_chain(*tasks):
        return _DummyChain(capture, *tasks)

    celery_module.chain = _fake_chain
    monkeypatch.setitem(sys.modules, "celery", celery_module)

    auto_ai_tasks_module = types.ModuleType("backend.pipeline.auto_ai_tasks")
    auto_ai_tasks_module.validation_build_packs = _DummyTask(
        "validation_build_packs", capture
    )
    auto_ai_tasks_module.validation_send = _DummyTask("validation_send", capture)
    auto_ai_tasks_module.validation_compact = _DummyTask("validation_compact", capture)
    monkeypatch.setitem(
        sys.modules,
        "backend.pipeline.auto_ai_tasks",
        auto_ai_tasks_module,
    )

    import backend.runflow.decider as decider

    decider = importlib.reload(decider)

    return decider, capture


def _write_runflow(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _install_runflow_step(decider, monkeypatch, events_path: Path, event_log: list[dict]):
    events_path.parent.mkdir(parents=True, exist_ok=True)

    def _fake_runflow_step(sid: str, stage: str, step: str, *, status: str, out=None):
        record = {"sid": sid, "stage": stage, "step": step, "status": status, "out": out}
        event_log.append(record)
        with events_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record))
            handle.write("\n")

    monkeypatch.setattr(decider, "runflow_step", _fake_runflow_step)


def _build_zero_pack_snapshot(sid: str) -> dict:
    return {
        "sid": sid,
        "run_state": "VALIDATING",
        "snapshot_version": 3,
        "stages": {
            "merge": {
                "status": "success",
                "metrics": {
                    "merge_zero_packs": True,
                    "skip_reason_top": "missing_original_creditor",
                    "skip_counts": {"missing_original_creditor": 3},
                },
                "summary": {
                    "merge_zero_packs": True,
                    "metrics": {
                        "merge_zero_packs": True,
                        "skip_reason_top": "missing_original_creditor",
                        "skip_counts": {"missing_original_creditor": 3},
                    },
                },
            },
            "validation": {
                "status": "pending",
                "sent": False,
                "results": {"results_total": 3, "completed": 1},
            },
        },
        "umbrella_barriers": {},
    }


def test_auto_enqueue_zero_pack_when_merge_zero_flagged(decider_stub, tmp_path, monkeypatch):
    decider, capture = decider_stub
    monkeypatch.setenv("VALIDATION_AUTOSEND", "1")

    sid = "zero-pack-sid"
    run_root = tmp_path / "runs"
    run_dir = run_root / sid
    index_path = run_dir / "ai_packs" / "validation" / "index.json"
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text("{" + "x" * 48 + "}", encoding="utf-8")

    runflow_path = run_dir / "runflow.json"
    snapshot = _build_zero_pack_snapshot(sid)
    _write_runflow(runflow_path, snapshot)

    events_path = run_dir / "runflow_events.jsonl"
    event_log: list[dict] = []
    _install_runflow_step(decider, monkeypatch, events_path, event_log)

    statuses = decider.reconcile_umbrella_barriers(sid, runs_root=run_root)
    assert isinstance(statuses, dict)

    lock_path = run_dir / ".locks" / "validation_fastpath.lock"
    assert lock_path.exists()

    queues = capture.get("queues", [])
    assert queues and queues[-1] == "validation"

    updated_snapshot = json.loads(runflow_path.read_text(encoding="utf-8"))
    validation_stage = updated_snapshot["stages"]["validation"]
    assert validation_stage.get("sent") is True
    assert validation_stage.get("status") == "in_progress"

    results_payload = validation_stage.get("results", {})
    assert isinstance(results_payload, dict)
    assert results_payload.get("results_total") == 3

    metrics_payload = validation_stage.get("metrics", {})
    assert "merge_zero_packs" not in metrics_payload

    summary_payload = validation_stage.get("summary", {})
    assert "merge_zero_packs" not in summary_payload
    merge_context = validation_stage.get("merge_context")
    assert merge_context and merge_context.get("merge_zero_packs") is True
    assert summary_payload.get("merge_context") == merge_context

    assert event_log
    fastpath_events = [event for event in event_log if event["step"] == "fastpath_send"]
    assert fastpath_events, "expected fastpath_send event to be recorded"

    assert updated_snapshot["snapshot_version"] > snapshot["snapshot_version"]


def test_no_double_enqueue_when_lock_present(decider_stub, tmp_path, monkeypatch):
    decider, capture = decider_stub
    monkeypatch.setenv("VALIDATION_AUTOSEND", "1")

    sid = "locked-sid"
    run_root = tmp_path / "runs"
    run_dir = run_root / sid

    index_path = run_dir / "ai_packs" / "validation" / "index.json"
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text("{" + "y" * 48 + "}", encoding="utf-8")

    lock_path = run_dir / ".locks" / "validation_fastpath.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_text("existing", encoding="utf-8")

    snapshot = _build_zero_pack_snapshot(sid)
    _write_runflow(run_dir / "runflow.json", snapshot)

    events_path = run_dir / "runflow_events.jsonl"
    event_log: list[dict] = []
    _install_runflow_step(decider, monkeypatch, events_path, event_log)

    decider.reconcile_umbrella_barriers(sid, runs_root=run_root)

    apply_calls = capture.get("apply_async_calls", [])
    assert not apply_calls, "auto-enqueue should not fire when lock exists"
    assert event_log == []


def test_persist_without_run_state_change(decider_stub, tmp_path, monkeypatch):
    decider, capture = decider_stub
    monkeypatch.setenv("VALIDATION_AUTOSEND", "1")

    sid = "persist-sid"
    run_root = tmp_path / "runs"
    run_dir = run_root / sid

    index_path = run_dir / "ai_packs" / "validation" / "index.json"
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text("{" + "z" * 48 + "}", encoding="utf-8")

    snapshot = _build_zero_pack_snapshot(sid)
    snapshot["run_state"] = "VALIDATING"
    original_version = snapshot["snapshot_version"]
    _write_runflow(run_dir / "runflow.json", snapshot)

    events_path = run_dir / "runflow_events.jsonl"
    event_log: list[dict] = []
    _install_runflow_step(decider, monkeypatch, events_path, event_log)

    decider.reconcile_umbrella_barriers(sid, runs_root=run_root)

    updated_snapshot = json.loads((run_dir / "runflow.json").read_text(encoding="utf-8"))

    assert updated_snapshot["run_state"] == "VALIDATING"
    assert updated_snapshot["snapshot_version"] > original_version
    assert capture.get("queues", [None])[-1] == "validation"
    assert event_log, "expected fastpath_send event to persist"
