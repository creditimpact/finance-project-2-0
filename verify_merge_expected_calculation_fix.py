"""
Simple verification script for the merge expected calculation fix.

This script can be run manually to verify the fix works for SID bf94cced.
"""

import json
from pathlib import Path

from backend.runflow.decider import finalize_merge_stage
from backend.core.runflow import runflow_refresh_umbrella_barriers


def verify_fix_for_sid(sid: str, runs_root: Path | str = Path("runs")) -> None:
    """
    Verify that merge finalization now works for the given SID.
    
    Args:
        sid: Session ID to verify
        runs_root: Root directory for runs (default: "runs")
    """
    print(f"\n{'='*70}")
    print(f"VERIFYING MERGE EXPECTED CALCULATION FIX FOR SID: {sid}")
    print(f"{'='*70}\n")
    
    runs_root = Path(runs_root)
    run_dir = runs_root / sid
    
    if not run_dir.exists():
        print(f"❌ ERROR: Run directory not found: {run_dir}")
        return
    
    # Check pairs_index.json
    pairs_index_path = run_dir / "ai_packs" / "merge" / "pairs_index.json"
    if pairs_index_path.exists():
        pairs_index = json.loads(pairs_index_path.read_text(encoding="utf-8"))
        totals = pairs_index.get("totals", {})
        pairs = pairs_index.get("pairs", [])
        
        print("📊 pairs_index.json metrics:")
        print(f"   - totals.created_packs: {totals.get('created_packs')}")
        print(f"   - totals.packs_built: {totals.get('packs_built')}")
        print(f"   - len(pairs): {len(pairs)} (bidirectional)")
        print(f"   - pairs_count: {pairs_index.get('pairs_count')}")
    
    # Count physical files
    packs_dir = run_dir / "ai_packs" / "merge" / "packs"
    results_dir = run_dir / "ai_packs" / "merge" / "results"
    
    pack_files = list(packs_dir.glob("pair_*.jsonl")) if packs_dir.exists() else []
    result_files = list(results_dir.glob("*.result.json")) if results_dir.exists() else []
    
    print(f"\n📁 Physical files:")
    print(f"   - Pack files: {len(pack_files)}")
    print(f"   - Result files: {len(result_files)}")
    
    # Try to finalize
    print(f"\n🔧 Running finalize_merge_stage...")
    try:
        result = finalize_merge_stage(sid, runs_root=runs_root)
        print(f"✅ SUCCESS: finalize_merge_stage completed without RuntimeError!")
        
        merge_ai_applied = result.get("merge_ai_applied")
        print(f"\n📋 Result:")
        print(f"   - merge_ai_applied: {merge_ai_applied}")
        
        counts = result.get("counts", {})
        metrics = result.get("metrics", {})
        print(f"   - packs_created: {counts.get('packs_created')}")
        print(f"   - result_files: {metrics.get('result_files')}")
        print(f"   - pack_files: {metrics.get('pack_files')}")
        
        if merge_ai_applied:
            print(f"\n✅ merge_ai_applied = True (EXPECTED BEHAVIOR)")
        else:
            print(f"\n⚠️  merge_ai_applied = False (UNEXPECTED - should be True)")
            
    except RuntimeError as e:
        print(f"❌ FAILED: RuntimeError raised!")
        print(f"   Error: {e}")
        print(f"\n   This suggests the fix did NOT work correctly.")
        print(f"   Expected behavior: Should NOT raise RuntimeError")
        return
    except Exception as e:
        print(f"❌ ERROR: Unexpected exception!")
        print(f"   {type(e).__name__}: {e}")
        return
    
    # Check runflow.json
    print(f"\n🔄 Checking runflow.json...")
    runflow_path = run_dir / "runflow.json"
    if runflow_path.exists():
        runflow = json.loads(runflow_path.read_text(encoding="utf-8"))
        merge_stage = runflow.get("merge", {})
        
        merge_ai_applied_flag = merge_stage.get("merge_ai_applied")
        merge_ai_applied_at = merge_stage.get("merge_ai_applied_at")
        
        print(f"   - merge.merge_ai_applied: {merge_ai_applied_flag}")
        print(f"   - merge.merge_ai_applied_at: {merge_ai_applied_at}")
        
        if merge_ai_applied_flag:
            print(f"\n✅ runflow.json updated correctly!")
        else:
            print(f"\n⚠️  merge_ai_applied not set in runflow.json")
    
    # Check barriers
    print(f"\n🚧 Refreshing barriers...")
    try:
        runflow_refresh_umbrella_barriers(sid)
        print(f"✅ Barriers refreshed successfully")
        
        # Re-read runflow to check merge_ready
        runflow = json.loads(runflow_path.read_text(encoding="utf-8"))
        umbrella_barriers = runflow.get("umbrella_barriers", {})
        merge_ready = umbrella_barriers.get("merge_ready")
        
        print(f"\n🎯 Final State:")
        print(f"   - umbrella_barriers.merge_ready: {merge_ready}")
        
        if merge_ready:
            print(f"\n✅✅✅ ALL CHECKS PASSED! ✅✅✅")
            print(f"   merge_ready = True")
            print(f"   Validation should now be able to proceed!")
        else:
            print(f"\n⚠️  merge_ready = False")
            print(f"   Expected: True (so validation can proceed)")
            
    except Exception as e:
        print(f"❌ ERROR refreshing barriers: {e}")
    
    print(f"\n{'='*70}\n")


if __name__ == "__main__":
    # Verify the fix for SID bf94cced (the original reported issue)
    verify_fix_for_sid("bf94cced-01d4-479a-b03b-ebf92623aa03")
