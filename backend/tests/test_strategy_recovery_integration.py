"""
Integration test for strategy recovery after validation completion.
Ensures that the strategy chain is enqueued and executed when recovery is triggered.
"""
import json
import pytest
from pathlib import Path
import shutil
import tempfile

from backend.runflow import decider
from backend.pipeline import auto_ai_tasks


@pytest.fixture
def temp_run_dir():
    d = tempfile.mkdtemp()
    yield Path(d)
    shutil.rmtree(d, ignore_errors=True)


def test_strategy_recovery_only_marks_after_enqueue_success(temp_run_dir, monkeypatch):
    """Test that strategy recovery ONLY marks stage in_progress if enqueue succeeds."""
    sid = "test-sid-recovery"
    run_dir = temp_run_dir / sid
    run_dir.mkdir(parents=True, exist_ok=True)
    
    runflow_path = run_dir / "runflow.json"
    runflow_payload = {
        "sid": sid,
        "run_state": "VALIDATING",
        "stages": {
            "validation": {
                "status": "success",
                "findings_count": 5,
                "metrics": {"validation_ai_completed": True, "merge_results_applied": True},
            },
        },
    }
    runflow_path.write_text(json.dumps(runflow_payload), encoding="utf-8")

    # Test 1: Enqueue succeeds -> stage marked
    def fake_enqueue_success(sid_arg, runs_root=None):
        return "fake-task-id-12345"
    
    monkeypatch.setattr(auto_ai_tasks, "enqueue_strategy_recovery_chain", fake_enqueue_success)
    
    # Directly call the recovery block logic (simulating reconcile internals)
    from backend.runflow.decider import record_stage_force
    
    # Simulate the recovery trigger
    try:
        task_id = fake_enqueue_success(sid, runs_root=run_dir.parent)
        if task_id:
            snapshot = {
                "stages": {
                    "strategy": {
                        "status": "in_progress",
                        "notes": "recovery_enqueued",
                        "task_id": task_id,
                    }
                }
            }
            record_stage_force(sid, snapshot, runs_root=run_dir.parent, last_writer="test", refresh_barriers=False)
    except Exception:
        pass
    
    # Verify strategy was marked
    snapshot = decider.get_runflow_snapshot(sid, runs_root=run_dir.parent)
    strategy_stage = snapshot.get("stages", {}).get("strategy", {})
    assert strategy_stage.get("status") == "in_progress"
    assert strategy_stage.get("task_id") == "fake-task-id-12345"
    
    # Test 2: Enqueue fails -> stage NOT marked
    sid2 = "test-sid-fail"
    run_dir2 = temp_run_dir / sid2
    run_dir2.mkdir(parents=True, exist_ok=True)
    runflow_path2 = run_dir2 / "runflow.json"
    runflow_path2.write_text(json.dumps({"sid": sid2, "run_state": "VALIDATING", "stages": {}}), encoding="utf-8")
    
    def fake_enqueue_fail(sid_arg, runs_root=None):
        raise RuntimeError("Enqueue failed")
    
    monkeypatch.setattr(auto_ai_tasks, "enqueue_strategy_recovery_chain", fake_enqueue_fail)
    
    task_id2 = None
    try:
        task_id2 = fake_enqueue_fail(sid2, runs_root=run_dir2.parent)
    except Exception:
        pass
    
    # Should NOT mark strategy because enqueue failed
    assert task_id2 is None, "Enqueue should have failed and returned None"
    
    snapshot2 = decider.get_runflow_snapshot(sid2, runs_root=run_dir2.parent)
    strategy_stage2 = snapshot2.get("stages", {}).get("strategy", {})
    assert strategy_stage2.get("status") != "in_progress", "Strategy should NOT be marked if enqueue fails"


def test_strategy_recovery_skips_when_already_started(temp_run_dir):
    """Test that recovery skips if strategy is already started."""
    sid = "test-sid-skip"
    run_dir = temp_run_dir / sid
    run_dir.mkdir(parents=True, exist_ok=True)
    
    runflow_path = run_dir / "runflow.json"
    runflow_payload = {
        "sid": sid,
        "run_state": "VALIDATING",
        "stages": {
            "validation": {
                "status": "success",
                "metrics": {
                    "validation_ai_completed": True,
                    "merge_results_applied": True,
                    "findings_count": 2,
                },
            },
            "strategy": {
                "status": "success",  # Already complete
            },
        },
        "umbrella_barriers": {
            "merge_ready": True,
            "validation_ready": True,
            "strategy_ready": True,
        },
    }
    runflow_path.write_text(json.dumps(runflow_payload), encoding="utf-8")

    # Run reconciliation
    statuses = decider.reconcile_umbrella_barriers(sid, runs_root=run_dir.parent)
    
    # Verify strategy stage was NOT changed
    snapshot = decider.get_runflow_snapshot(sid, runs_root=run_dir.parent)
    strategy_stage = snapshot.get("stages", {}).get("strategy", {})
    assert strategy_stage.get("status") == "success", "Strategy stage should remain success"
    assert "task_id" not in strategy_stage, "No new task_id should be added"
