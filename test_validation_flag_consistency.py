#!/usr/bin/env python3
"""Test validation_ai_applied flag consistency across stage structure."""

import json
import sys
from pathlib import Path


def assert_validation_flags_consistent(stage):
    """Assert that validation_ai_applied is consistent across stage structure.
    
    The canonical flag is at stage["validation_ai_applied"].
    Any copies in metrics/summary must either be absent or match the canonical value.
    """
    canon = stage.get("validation_ai_applied")
    
    if canon is None:
        # If canonical is absent, all copies should also be absent
        print("⚠️  No canonical validation_ai_applied flag at root level")
        canon_bool = False
    else:
        canon_bool = bool(canon)
        print(f"✓ Canonical validation_ai_applied: {canon_bool}")
    
    copies = []
    locations = []
    
    # Check metrics
    metrics = stage.get("metrics") or {}
    if "validation_ai_applied" in metrics:
        copies.append(bool(metrics["validation_ai_applied"]))
        locations.append("metrics")
    
    # Check summary
    summary = stage.get("summary") or {}
    if "validation_ai_applied" in summary:
        copies.append(bool(summary["validation_ai_applied"]))
        locations.append("summary")
    
    # Check summary.metrics
    summary_metrics = (summary.get("metrics") or {})
    if "validation_ai_applied" in summary_metrics:
        copies.append(bool(summary_metrics["validation_ai_applied"]))
        locations.append("summary.metrics")
    
    if not copies:
        print("✓ No nested copies found (clean structure)")
        return True
    
    # Check for contradictions
    errors = []
    for i, (value, location) in enumerate(zip(copies, locations)):
        if canon is not None and value != canon_bool:
            errors.append(f"  ❌ {location}.validation_ai_applied = {value} (canonical is {canon_bool})")
        else:
            print(f"  ⚠️  Redundant copy at {location}.validation_ai_applied = {value}")
    
    if errors:
        print("\n❌ INCONSISTENT FLAGS DETECTED:")
        for err in errors:
            print(err)
        raise AssertionError(f"Inconsistent validation_ai_applied: canonical={canon_bool}, contradictions found")
    
    if copies:
        print("  ⚠️  Redundant but consistent copies exist (should be removed)")
    
    return True


def check_sid(sid: str, runs_root: Path = Path("runs")) -> bool:
    """Check validation flag consistency for a SID."""
    
    run_dir = runs_root / sid
    runflow_path = run_dir / "runflow.json"
    
    if not runflow_path.exists():
        print(f"❌ Runflow not found: {runflow_path}")
        return False
    
    with runflow_path.open("r", encoding="utf-8") as f:
        runflow = json.load(f)
    
    validation_stage = runflow.get("stages", {}).get("validation", {})
    
    print(f"\n📋 SID: {sid}")
    print("="*60)
    
    try:
        assert_validation_flags_consistent(validation_stage)
        print("\n✅ VALIDATION FLAG CONSISTENCY CHECK PASSED")
        return True
    except AssertionError as e:
        print(f"\n❌ VALIDATION FLAG CONSISTENCY CHECK FAILED: {e}")
        return False


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_validation_flag_consistency.py <sid>")
        sys.exit(1)
    
    sid = sys.argv[1]
    success = check_sid(sid)
    sys.exit(0 if success else 1)
