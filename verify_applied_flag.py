#!/usr/bin/env python3
"""Verify validation_ai_applied flag is correctly set in runflow after V2 apply."""

import json
import sys
from pathlib import Path

def check_sid(sid: str, runs_root: Path = Path("runs")) -> bool:
    """Check if validation_ai_applied is correctly set for a SID."""
    
    run_dir = runs_root / sid
    manifest_path = run_dir / "manifest.json"
    runflow_path = run_dir / "runflow.json"
    
    if not manifest_path.exists():
        print(f"❌ Manifest not found: {manifest_path}")
        return False
    
    if not runflow_path.exists():
        print(f"❌ Runflow not found: {runflow_path}")
        return False
    
    # Load manifest
    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)
    
    # Load runflow
    with runflow_path.open("r", encoding="utf-8") as f:
        runflow = json.load(f)
    
    # Check manifest V2 apply status
    ai_validation = manifest.get("ai", {}).get("status", {}).get("validation", {})
    manifest_applied = ai_validation.get("validation_ai_applied", False)
    manifest_results_total = ai_validation.get("results_total", 0)
    manifest_results_applied = ai_validation.get("results_applied", 0)
    manifest_apply_ok = ai_validation.get("results_apply_ok", False)
    
    print(f"\n📋 SID: {sid}")
    print(f"\n🔍 Manifest ai.status.validation:")
    print(f"   results_total: {manifest_results_total}")
    print(f"   results_applied: {manifest_results_applied}")
    print(f"   results_apply_ok: {manifest_apply_ok}")
    print(f"   validation_ai_applied: {manifest_applied}")
    
    # Check runflow validation stage
    validation_stage = runflow.get("stages", {}).get("validation", {})
    stage_applied = validation_stage.get("validation_ai_applied", False)
    stage_expected = validation_stage.get("expected_results", 0)
    stage_received = validation_stage.get("results_received", 0)
    stage_status = validation_stage.get("status", "unknown")
    
    metrics = validation_stage.get("metrics", {})
    metrics_applied = metrics.get("validation_ai_applied", False)
    metrics_results_total = metrics.get("results_total", 0)
    metrics_missing = metrics.get("missing_results", 0)
    
    summary = validation_stage.get("summary", {})
    summary_applied = summary.get("validation_ai_applied", False)
    
    print(f"\n🔍 Runflow stages.validation:")
    print(f"   status: {stage_status}")
    print(f"   expected_results: {stage_expected}")
    print(f"   results_received: {stage_received}")
    print(f"   validation_ai_applied: {stage_applied}")
    print(f"   metrics.validation_ai_applied: {metrics_applied}")
    print(f"   metrics.results_total: {metrics_results_total}")
    print(f"   metrics.missing_results: {metrics_missing}")
    print(f"   summary.validation_ai_applied: {summary_applied}")
    
    # Validation checks
    errors = []
    
    if manifest_results_total > 0 and manifest_applied:
        # V2 apply claims success
        if not stage_applied:
            errors.append(f"❌ Stage validation_ai_applied={stage_applied} but manifest shows {manifest_applied}")
        if not metrics_applied:
            errors.append(f"❌ Metrics validation_ai_applied={metrics_applied} but manifest shows {manifest_applied}")
        if not summary_applied:
            errors.append(f"❌ Summary validation_ai_applied={summary_applied} but manifest shows {manifest_applied}")
        if stage_received != manifest_results_applied:
            errors.append(f"❌ Stage results_received={stage_received} but manifest shows results_applied={manifest_results_applied}")
        if metrics_results_total != manifest_results_total:
            errors.append(f"❌ Metrics results_total={metrics_results_total} but manifest shows {manifest_results_total}")
    
    if errors:
        print("\n❌ VALIDATION FAILED:")
        for err in errors:
            print(f"   {err}")
        return False
    else:
        print("\n✅ VALIDATION PASSED: All fields consistent!")
        return True


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python verify_applied_flag.py <sid>")
        sys.exit(1)
    
    sid = sys.argv[1]
    success = check_sid(sid)
    sys.exit(0 if success else 1)
