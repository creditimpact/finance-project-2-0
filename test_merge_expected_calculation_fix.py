"""
Test for merge expected calculation fix (pairs_count bug).

Verifies that finalize_merge_stage and _merge_artifacts_progress
correctly handle bidirectional pairs representation without treating
len(pairs) as the expected physical pack count.
"""

import json
import tempfile
from pathlib import Path
from typing import Any

import pytest

from backend.runflow.decider import finalize_merge_stage, _merge_artifacts_progress


def _setup_merge_scenario(
    base_dir: Path,
    sid: str,
    *,
    created_packs: int = 1,
    pairs_count: int = 2,
    num_physical_files: int = 1,
) -> Path:
    """
    Create a test scenario with bidirectional pairs representation.
    
    Args:
        base_dir: Base directory for the run
        sid: Session ID
        created_packs: Value for totals.created_packs (physical pack count)
        pairs_count: Length of pairs array (bidirectional, typically 2x physical)
        num_physical_files: Number of physical pack/result files to create
    
    Returns:
        Path to the run directory
    """
    run_dir = base_dir / sid
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Create manifest
    manifest_path = run_dir / "manifest.json"
    manifest = {
        "sid": sid,
        "ai": {
            "merge": {
                "index_path": "ai_packs/merge/pairs_index.json"
            }
        }
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    
    # Create merge directories
    merge_base = run_dir / "ai_packs" / "merge"
    packs_dir = merge_base / "packs"
    results_dir = merge_base / "results"
    packs_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Create pairs_index.json with bidirectional representation
    pairs_payload = []
    packs_payload = []
    for i in range(num_physical_files):
        a_idx = 7 + i * 3
        b_idx = 10 + i * 3
        pack_file = f"pair_{a_idx:03d}_{b_idx:03d}.jsonl"
        
        # Bidirectional entries (this is the key: 2 entries per physical file)
        for pair in ([a_idx, b_idx], [b_idx, a_idx]):
            pairs_payload.append({
                "pair": pair,
                "pack_file": pack_file,
                "score": 49
            })
            packs_payload.append({
                "a": pair[0],
                "b": pair[1],
                "pack_file": pack_file
            })
        
        # Create physical files
        pack_path = packs_dir / pack_file
        pack_path.write_text(json.dumps({"accounts": []}), encoding="utf-8")
        
        result_file = f"pair_{a_idx:03d}_{b_idx:03d}.result.json"
        result_path = results_dir / result_file
        result_path.write_text(json.dumps({"decision": "duplicate"}), encoding="utf-8")
    
    index_payload = {
        "sid": sid,
        "totals": {
            "scored_pairs": 3,
            "created_packs": created_packs,
            "packs_built": created_packs,
            "skipped": 2
        },
        "pairs": pairs_payload,
        "packs": packs_payload,
        "pairs_count": pairs_count
    }
    
    index_path = merge_base / "pairs_index.json"
    index_path.write_text(json.dumps(index_payload, indent=2), encoding="utf-8")
    
    # Create runflow.json
    runflow_path = run_dir / "runflow.json"
    runflow = {
        "sid": sid,
        "merge": {
            "status": "success"
        }
    }
    runflow_path.write_text(json.dumps(runflow), encoding="utf-8")
    
    return run_dir


def test_finalize_merge_stage_with_bidirectional_pairs(tmp_path: Path) -> None:
    """
    Test that finalize_merge_stage succeeds with bidirectional pairs.
    
    This is the core bug scenario:
    - created_packs = 1 (physical pack count)
    - pairs_count = 2 (bidirectional: [7,10] and [10,7])
    - Physical files: 1 pack, 1 result
    
    Before fix: RuntimeError("merge stage artifacts not ready: results=1 packs=1 expected=2")
    After fix: Success, merge_ai_applied=True
    """
    sid = "test-bidirectional-pairs"
    run_dir = _setup_merge_scenario(
        tmp_path,
        sid,
        created_packs=1,
        pairs_count=2,  # Bidirectional representation
        num_physical_files=1
    )
    
    # This should NOT raise RuntimeError
    result = finalize_merge_stage(sid, runs_root=tmp_path)
    
    # Verify merge_ai_applied was set
    assert result.get("merge_ai_applied") is True, "merge_ai_applied should be True"
    
    # Verify counts
    counts = result.get("counts", {})
    assert counts.get("packs_created") == 1, "packs_created should be 1"
    
    # Verify metrics
    metrics = result.get("metrics", {})
    assert metrics.get("result_files") == 1, "result_files should be 1"
    assert metrics.get("pack_files") == 1, "pack_files should be 1"
    
    # Verify runflow.json was updated
    runflow_path = run_dir / "runflow.json"
    runflow_data = json.loads(runflow_path.read_text(encoding="utf-8"))
    assert runflow_data["merge"]["merge_ai_applied"] is True


def test_finalize_merge_stage_with_multiple_packs_bidirectional(tmp_path: Path) -> None:
    """
    Test with multiple packs (3 physical files = 6 bidirectional entries).
    
    Should use created_packs=3, not pairs_count=6.
    """
    sid = "test-multiple-bidirectional"
    run_dir = _setup_merge_scenario(
        tmp_path,
        sid,
        created_packs=3,
        pairs_count=6,  # 3 packs * 2 directions
        num_physical_files=3
    )
    
    result = finalize_merge_stage(sid, runs_root=tmp_path)
    
    assert result.get("merge_ai_applied") is True
    
    counts = result.get("counts", {})
    assert counts.get("packs_created") == 3, "packs_created should be 3"
    
    metrics = result.get("metrics", {})
    assert metrics.get("result_files") == 3, "result_files should be 3"
    assert metrics.get("pack_files") == 3, "pack_files should be 3"


def test_merge_artifacts_progress_with_bidirectional_pairs(tmp_path: Path) -> None:
    """
    Test _merge_artifacts_progress doesn't use len(pairs) as fallback.
    
    Before fix: expected_total = len(pairs) = 2
    After fix: expected_total from created_packs = 1 (or None if missing)
    """
    sid = "test-progress-bidirectional"
    run_dir = _setup_merge_scenario(
        tmp_path,
        sid,
        created_packs=1,
        pairs_count=2,
        num_physical_files=1
    )
    
    result_files, pack_files, expected_total, ready = _merge_artifacts_progress(
        run_dir=run_dir,
        manifest_payload=None
    )
    
    # Verify counts
    assert result_files == 1, "result_files should be 1"
    assert pack_files == 1, "pack_files should be 1"
    
    # expected_total should be 1 (from created_packs), not 2 (from len(pairs))
    assert expected_total == 1, f"expected_total should be 1, got {expected_total}"
    
    # Should be ready
    assert ready is True, "ready should be True when all files present"


def test_merge_artifacts_progress_no_totals_fallback(tmp_path: Path) -> None:
    """
    Test _merge_artifacts_progress when totals are missing.
    
    After fix: Should NOT fall back to len(pairs), expected_total should be None.
    Should still consider ready if result_files == pack_files.
    """
    sid = "test-no-totals-fallback"
    run_dir = _setup_merge_scenario(
        tmp_path,
        sid,
        created_packs=1,
        pairs_count=2,
        num_physical_files=1
    )
    
    # Remove totals from index to simulate missing metadata
    index_path = run_dir / "ai_packs" / "merge" / "pairs_index.json"
    index_data = json.loads(index_path.read_text(encoding="utf-8"))
    index_data["totals"] = {}  # Empty totals
    index_path.write_text(json.dumps(index_data), encoding="utf-8")
    
    result_files, pack_files, expected_total, ready = _merge_artifacts_progress(
        run_dir=run_dir,
        manifest_payload=None
    )
    
    # expected_total should be None (not 2 from len(pairs))
    assert expected_total is None, f"expected_total should be None when totals missing, got {expected_total}"
    
    # Should still be ready because result_files == pack_files
    assert result_files == pack_files == 1
    assert ready is True, "ready should be True when result_files == pack_files even without expected_total"


def test_zero_packs_scenario_unaffected(tmp_path: Path) -> None:
    """
    Test that zero-packs scenario (merge_zero_packs=true) still works.
    
    This should be unaffected by the pairs_count fix.
    """
    sid = "test-zero-packs"
    run_dir = tmp_path / sid
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Create manifest
    manifest_path = run_dir / "manifest.json"
    manifest = {
        "sid": sid,
        "ai": {
            "merge": {
                "index_path": "ai_packs/merge/pairs_index.json"
            }
        }
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    
    # Create merge directories (empty)
    merge_base = run_dir / "ai_packs" / "merge"
    packs_dir = merge_base / "packs"
    results_dir = merge_base / "results"
    packs_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Create index with zero packs flag
    index_payload = {
        "sid": sid,
        "totals": {
            "scored_pairs": 3,
            "created_packs": 0,
            "packs_built": 0,
            "merge_zero_packs": True
        },
        "pairs": [],
        "pairs_count": 0
    }
    index_path = merge_base / "pairs_index.json"
    index_path.write_text(json.dumps(index_payload), encoding="utf-8")
    
    # Create runflow.json
    runflow_path = run_dir / "runflow.json"
    runflow = {
        "sid": sid,
        "merge": {
            "status": "success"
        }
    }
    runflow_path.write_text(json.dumps(runflow), encoding="utf-8")
    
    # Should succeed with zero packs
    result = finalize_merge_stage(sid, runs_root=tmp_path)
    
    assert result.get("merge_ai_applied") is True
    counts = result.get("counts", {})
    assert counts.get("packs_created") == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
