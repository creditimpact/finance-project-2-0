"""
Tests for merge_ai_applied flag and merge_ready barrier timing fix.

These tests verify:
1. _compute_umbrella_barriers correctly checks merge_ai_applied for non-zero-packs
2. Zero-packs fast path remains unaffected (empty_ok takes precedence)
3. Validation orchestrator defers when merge_ready=False
4. Regression test reproducing the original bug pattern (SID 83830ae4)
"""
import json
import tempfile
from pathlib import Path
from datetime import datetime

import pytest

from backend.runflow.decider import (
    _compute_umbrella_barriers,
    finalize_merge_stage,
    _load_runflow,
)
from backend.pipeline.validation_orchestrator import ValidationOrchestrator


class TestMergeAiAppliedBarrier:
    """Test merge_ai_applied flag in barrier logic."""

    def test_zero_packs_fast_path_unaffected(self, tmp_path: Path):
        """Zero-packs case: merge_ready=True even without merge_ai_applied."""
        run_dir = tmp_path / "zero-packs-sid"
        run_dir.mkdir()
        
        runflow = {
            "sid": "zero-packs-sid",
            "stages": {
                "merge": {
                    "status": "success",
                    "empty_ok": True,
                    "merge_zero_packs": True,
                    "results": {"result_files": 0},
                    "metrics": {"created_packs": 0, "pairs_scored": 100},
                    # Intentionally NO merge_ai_applied flag
                }
            }
        }
        runflow_path = run_dir / "runflow.json"
        runflow_path.write_text(json.dumps(runflow), encoding="utf-8")

        # Create minimal merge artifacts structure
        ai_packs = run_dir / "ai_packs" / "merge"
        ai_packs.mkdir(parents=True)
        (ai_packs / "results").mkdir()
        (ai_packs / "packs").mkdir()
        (ai_packs / "pairs_index.json").write_text('{"pairs":[],"totals":{"merge_zero_packs":true}}')

        barriers = _compute_umbrella_barriers(run_dir)
        
        # Zero-packs should be ready even without merge_ai_applied
        assert barriers["merge_ready"] is True, "Zero-packs should bypass merge_ai_applied check"

    def test_non_zero_packs_requires_merge_ai_applied(self, tmp_path: Path):
        """Non-zero-packs case: merge_ready=False without merge_ai_applied."""
        run_dir = tmp_path / "non-zero-packs-sid"
        run_dir.mkdir()
        
        runflow = {
            "sid": "non-zero-packs-sid",
            "stages": {
                "merge": {
                    "status": "success",
                    "empty_ok": False,
                    "results": {"result_files": 5},
                    "metrics": {"created_packs": 5, "pairs_scored": 100},
                    # Intentionally NO merge_ai_applied flag
                }
            }
        }
        runflow_path = run_dir / "runflow.json"
        runflow_path.write_text(json.dumps(runflow), encoding="utf-8")

        # Create merge artifacts
        ai_packs = run_dir / "ai_packs" / "merge"
        results_dir = ai_packs / "results"
        packs_dir = ai_packs / "packs"
        results_dir.mkdir(parents=True)
        packs_dir.mkdir(parents=True)
        
        # Create 5 result files
        for i in range(5):
            (results_dir / f"pair-{i}.result.json").write_text("{}")
            (packs_dir / f"pair_{i}.jsonl").write_text("{}")
        
        (ai_packs / "pairs_index.json").write_text('{"pairs":[1,2,3,4,5],"totals":{"created_packs":5}}')

        barriers = _compute_umbrella_barriers(run_dir)
        
        # Non-zero-packs WITHOUT merge_ai_applied should NOT be ready
        assert barriers["merge_ready"] is False, "Non-zero-packs should require merge_ai_applied"

    def test_non_zero_packs_with_merge_ai_applied_ready(self, tmp_path: Path):
        """Non-zero-packs case: merge_ready=True WITH merge_ai_applied."""
        run_dir = tmp_path / "applied-sid"
        run_dir.mkdir()
        
        runflow = {
            "sid": "applied-sid",
            "stages": {
                "merge": {
                    "status": "success",
                    "empty_ok": False,
                    "results": {"result_files": 5},
                    "metrics": {"created_packs": 5, "pairs_scored": 100},
                    "merge_ai_applied": True,  # KEY: flag is set
                    "merge_ai_applied_at": "2025-01-15T10:30:00Z",
                }
            }
        }
        runflow_path = run_dir / "runflow.json"
        runflow_path.write_text(json.dumps(runflow), encoding="utf-8")

        # Create merge artifacts
        ai_packs = run_dir / "ai_packs" / "merge"
        results_dir = ai_packs / "results"
        packs_dir = ai_packs / "packs"
        results_dir.mkdir(parents=True)
        packs_dir.mkdir(parents=True)
        
        for i in range(5):
            (results_dir / f"pair-{i}.result.json").write_text("{}")
            (packs_dir / f"pair_{i}.jsonl").write_text("{}")
        
        (ai_packs / "pairs_index.json").write_text('{"pairs":[1,2,3,4,5],"totals":{"created_packs":5}}')

        barriers = _compute_umbrella_barriers(run_dir)
        
        # With merge_ai_applied=True, should be ready
        assert barriers["merge_ready"] is True, "Non-zero-packs WITH merge_ai_applied should be ready"


class TestValidationOrchestratorGating:
    """Test validation orchestrator respects merge_ready barrier."""

    def test_orchestrator_defers_when_merge_not_ready(self, tmp_path: Path):
        """Validation orchestrator should return deferred when merge_ready=False."""
        runs_root = tmp_path / "runs"
        runs_root.mkdir()
        sid = "deferred-sid"
        run_dir = runs_root / sid
        run_dir.mkdir()

        # Create runflow with merge NOT ready (no merge_ai_applied)
        runflow = {
            "sid": sid,
            "stages": {
                "merge": {
                    "status": "success",
                    "empty_ok": False,
                    "results": {"result_files": 3},
                    "metrics": {"created_packs": 3},
                    # NO merge_ai_applied
                }
            }
        }
        (run_dir / "runflow.json").write_text(json.dumps(runflow), encoding="utf-8")

        # Create merge artifacts
        ai_packs = run_dir / "ai_packs" / "merge"
        results_dir = ai_packs / "results"
        packs_dir = ai_packs / "packs"
        results_dir.mkdir(parents=True)
        packs_dir.mkdir(parents=True)
        for i in range(3):
            (results_dir / f"pair-{i}.result.json").write_text("{}")
            (packs_dir / f"pair_{i}.jsonl").write_text("{}")
        (ai_packs / "pairs_index.json").write_text('{"pairs":[1,2,3],"totals":{"created_packs":3}}')

        # Run orchestrator
        orchestrator = ValidationOrchestrator(runs_root=runs_root)
        result = orchestrator.run_for_sid(sid)

        # Should be deferred
        assert result.get("deferred") is True, "Should defer when merge_ready=False"
        assert result.get("reason") == "merge_not_ready", "Should cite merge_not_ready reason"

    def test_orchestrator_proceeds_when_merge_ready(self, tmp_path: Path):
        """Validation orchestrator should proceed when merge_ready=True."""
        runs_root = tmp_path / "runs"
        runs_root.mkdir()
        sid = "ready-sid"
        run_dir = runs_root / sid
        run_dir.mkdir()

        # Create runflow with merge READY (merge_ai_applied=True)
        runflow = {
            "sid": sid,
            "stages": {
                "merge": {
                    "status": "success",
                    "empty_ok": False,
                    "results": {"result_files": 3},
                    "metrics": {"created_packs": 3},
                    "merge_ai_applied": True,
                    "merge_ai_applied_at": "2025-01-15T10:30:00Z",
                }
            }
        }
        (run_dir / "runflow.json").write_text(json.dumps(runflow), encoding="utf-8")

        # Create merge artifacts
        ai_packs = run_dir / "ai_packs" / "merge"
        results_dir = ai_packs / "results"
        packs_dir = ai_packs / "packs"
        results_dir.mkdir(parents=True)
        packs_dir.mkdir(parents=True)
        for i in range(3):
            (results_dir / f"pair-{i}.result.json").write_text("{}")
            (packs_dir / f"pair_{i}.jsonl").write_text("{}")
        (ai_packs / "pairs_index.json").write_text('{"pairs":[1,2,3],"totals":{"created_packs":3}}')

        # Create minimal manifest and validation index to prevent build errors
        (run_dir / "manifest.json").write_text('{"ai":{"validation":{}}}', encoding="utf-8")
        val_ai = run_dir / "ai_packs" / "validation"
        val_ai.mkdir(parents=True)
        (val_ai / "index.json").write_text('{"packs":[]}', encoding="utf-8")
        (val_ai / "packs").mkdir()
        (val_ai / "results").mkdir()

        # Run orchestrator
        orchestrator = ValidationOrchestrator(runs_root=runs_root)
        result = orchestrator.run_for_sid(sid)

        # Should NOT be deferred (might be partial/finalized depending on validation content)
        assert result.get("deferred") is not True, "Should proceed when merge_ready=True"


class TestRegressionBugPattern:
    """Regression test reproducing SID 83830ae4 bug pattern."""

    def test_bug_pattern_validation_before_merge_applied(self, tmp_path: Path):
        """
        Reproduce bug: validation completes before merge_ai_applied is set.
        
        Original timeline (SID 83830ae4):
        - T+17s: validation_ai_applied=true (validation done)
        - T+25s: merge_ready=true (8 second gap!)
        
        This test verifies the fix prevents this scenario.
        """
        run_dir = tmp_path / "bug-sid"
        run_dir.mkdir()

        # Simulate state AFTER merge result files appear but BEFORE finalize_merge_stage runs
        runflow = {
            "sid": "bug-sid",
            "stages": {
                "merge": {
                    "status": "success",  # Stage marked success from result file count
                    "empty_ok": False,
                    "results": {"result_files": 10},
                    "metrics": {"created_packs": 10},
                    # BUG: merge_ai_applied NOT YET SET (finalize_merge_stage hasn't run)
                }
            }
        }
        (run_dir / "runflow.json").write_text(json.dumps(runflow), encoding="utf-8")

        # Create merge artifacts (files exist on disk)
        ai_packs = run_dir / "ai_packs" / "merge"
        results_dir = ai_packs / "results"
        packs_dir = ai_packs / "packs"
        results_dir.mkdir(parents=True)
        packs_dir.mkdir(parents=True)
        for i in range(10):
            (results_dir / f"pair-{i}.result.json").write_text("{}")
            (packs_dir / f"pair_{i}.jsonl").write_text("{}")
        (ai_packs / "pairs_index.json").write_text('{"pairs":list(range(10)),"totals":{"created_packs":10}}')

        # Check barriers in this buggy state
        barriers = _compute_umbrella_barriers(run_dir)

        # THE FIX: merge_ready should be FALSE because merge_ai_applied is missing
        assert barriers["merge_ready"] is False, (
            "BUG FIX: merge_ready must be False when merge_ai_applied missing. "
            "This prevents validation from starting before merge results are applied."
        )

        # Now simulate finalize_merge_stage completing (would happen after send_ai_merge_packs.py runs)
        runflow["stages"]["merge"]["merge_ai_applied"] = True
        runflow["stages"]["merge"]["merge_ai_applied_at"] = datetime.utcnow().isoformat() + "Z"
        (run_dir / "runflow.json").write_text(json.dumps(runflow), encoding="utf-8")

        # Re-check barriers
        barriers = _compute_umbrella_barriers(run_dir)

        # NOW merge_ready should be True
        assert barriers["merge_ready"] is True, (
            "After merge_ai_applied=True, merge_ready should be True"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
