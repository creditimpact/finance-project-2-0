#!/usr/bin/env python3
"""Test validation V2 in production pipeline mode.

This script simulates the production flow:
1. Build validation requirements
2. Build validation packs (now triggers V2 autosend)
3. Check results applied

Unlike manual tests, this uses the REAL production code path.
"""
import os
import sys
from pathlib import Path

# Set orchestrator mode flags BEFORE any imports
os.environ["VALIDATION_ORCHESTRATOR_MODE"] = "1"
os.environ["VALIDATION_AUTOSEND_ENABLED"] = "1"

from backend.pipeline.auto_ai import run_validation_requirements_for_all_accounts
from backend.ai.validation_builder import build_validation_packs_for_run
from backend.pipeline.runs import RunManifest


def test_production_v2_flow(sid: str):
    """Test V2 integration using production pipeline flow."""
    
    print(f"\n{'='*80}")
    print(f"Testing Production V2 Flow for SID: {sid}")
    print(f"{'='*80}\n")
    
    runs_root = Path("runs")
    run_dir = runs_root / sid
    
    if not run_dir.exists():
        print(f"❌ Run directory not found: {run_dir}")
        return False
    
    # Step 1: Build validation requirements (production entry point)
    print("\n[STEP 1] Building validation requirements (production entry)...")
    try:
        stats = run_validation_requirements_for_all_accounts(sid)
        print(f"✅ Requirements built: {stats}")
        print(f"   - Processed accounts: {stats.get('processed_accounts', 0)}")
        print(f"   - Findings: {stats.get('findings', 0)}")
    except Exception as e:
        print(f"❌ Requirements build failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Step 2: Build validation packs (NEW - this triggers V2 autosend)
    print("\n[STEP 2] Building validation packs (triggers V2 autosend)...")
    try:
        pack_results = build_validation_packs_for_run(sid, runs_root=runs_root)
        packs_written = sum(len(entries or []) for entries in pack_results.values())
        print(f"✅ Validation packs built: {packs_written} packs")
        print(f"   Pack results: {pack_results}")
    except Exception as e:
        print(f"❌ Pack build failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Step 3: Check that V2 autosend ran
    print("\n[STEP 3] Checking V2 autosend execution...")
    
    # Check for results directory
    results_dir = run_dir / "ai_packs" / "validation" / "results"
    if not results_dir.exists():
        print(f"❌ Results directory not found: {results_dir}")
        return False
    
    result_files = list(results_dir.glob("*.json"))
    print(f"✅ Results directory exists with {len(result_files)} files")
    
    # Step 4: Check manifest for results_applied flag
    print("\n[STEP 4] Checking manifest for results_applied flag...")
    try:
        manifest = RunManifest.for_sid(sid, runs_root=runs_root, allow_create=False)
        validation_status = manifest.get_ai_stage_status("validation")
        results_applied = validation_status.get("results_applied", False)
        
        print(f"   Validation status: {validation_status}")
        print(f"   Results applied: {results_applied}")
        
        if results_applied:
            print("✅ Results applied flag is set")
        else:
            print("❌ Results applied flag is NOT set")
            return False
    except Exception as e:
        print(f"❌ Manifest check failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Step 5: Check summary.json for AI fields
    print("\n[STEP 5] Checking summary.json for AI enrichment...")
    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        print(f"❌ Summary file not found: {summary_path}")
        return False
    
    import json
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    
    # Find first account with validation requirements
    ai_enriched_count = 0
    for account_data in summary.get("accounts", {}).values():
        requirements = account_data.get("validation_requirements", [])
        for req in requirements:
            if req.get("ai_validated"):
                ai_enriched_count += 1
                print(f"   ✅ Found AI-enriched requirement: {req.get('reason_code')}")
                print(f"      - ai_validated: {req.get('ai_validated')}")
                print(f"      - ai_review_status: {req.get('ai_review_status')}")
                print(f"      - ai_explanation: {req.get('ai_explanation', 'N/A')[:80]}...")
                break
        if ai_enriched_count > 0:
            break
    
    if ai_enriched_count == 0:
        print("❌ No AI-enriched requirements found in summary.json")
        return False
    
    print(f"\n✅ Found {ai_enriched_count} AI-enriched requirements")
    
    # Success!
    print(f"\n{'='*80}")
    print("✅ PRODUCTION V2 FLOW TEST PASSED")
    print(f"{'='*80}\n")
    print("Summary:")
    print(f"  ✅ Requirements built via production pipeline")
    print(f"  ✅ Validation packs built ({packs_written} packs)")
    print(f"  ✅ V2 autosend triggered")
    print(f"  ✅ Results written to {results_dir}")
    print(f"  ✅ Results applied flag set in manifest")
    print(f"  ✅ AI fields merged into summary.json")
    print("\n🎉 V2 is now integrated into production pipeline!")
    
    return True


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_production_v2_flow.py <SID>")
        print("\nExample:")
        print("  python test_production_v2_flow.py dcc2ee6f-3457-426f-b385-b884da0f223b")
        sys.exit(1)
    
    sid = sys.argv[1]
    success = test_production_v2_flow(sid)
    sys.exit(0 if success else 1)
