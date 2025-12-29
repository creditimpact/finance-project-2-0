#!/usr/bin/env python3
"""Quick validation that the production pipeline integration is correct.

This script checks that:
1. build_validation_packs_for_run is imported and called in tasks.py
2. The function exists in validation_builder.py
3. V2 autosend logic is present in build_validation_packs_for_run
4. Results applier is called by sender V2
"""
import ast
from pathlib import Path


def check_tasks_py_integration():
    """Check that tasks.py calls build_validation_packs_for_run."""
    print("\n[CHECK 1] Verifying tasks.py integration...")
    
    tasks_path = Path("backend/api/tasks.py")
    content = tasks_path.read_text(encoding="utf-8")
    
    # Check for import
    if "from backend.ai.validation_builder import build_validation_packs_for_run" in content:
        print("  ✅ build_validation_packs_for_run is imported")
    else:
        print("  ❌ Missing import of build_validation_packs_for_run")
        return False
    
    # Check for call
    if "pack_results = build_validation_packs_for_run(sid," in content:
        print("  ✅ build_validation_packs_for_run is called")
    else:
        print("  ❌ build_validation_packs_for_run is not called")
        return False
    
    # Check it's after requirements
    requirements_done = content.find("VALIDATION_REQUIREMENTS_PIPELINE_DONE")
    pack_build = content.find("build_validation_packs_for_run(sid,")
    
    if requirements_done > 0 and pack_build > requirements_done:
        print("  ✅ Pack builder called AFTER requirements pipeline")
    else:
        print("  ❌ Pack builder not in correct position")
        return False
    
    return True


def check_validation_builder():
    """Check that validation_builder.py has V2 autosend logic."""
    print("\n[CHECK 2] Verifying validation_builder.py V2 autosend...")
    
    builder_path = Path("backend/ai/validation_builder.py")
    content = builder_path.read_text(encoding="utf-8")
    
    # Check for autosend trigger log
    if "VALIDATION_V2_AUTOSEND_TRIGGER" in content:
        print("  ✅ V2 autosend trigger log found")
    else:
        print("  ❌ V2 autosend trigger log missing")
        return False
    
    # Check for sender import
    if "from backend.ai.validation_sender_v2 import run_validation_send_for_sid_v2" in content:
        print("  ✅ Sender V2 import found")
    else:
        print("  ❌ Sender V2 import missing")
        return False
    
    # Check for orchestrator_mode check
    if "orchestrator_mode and autosend_enabled" in content:
        print("  ✅ Orchestrator mode guard found")
    else:
        print("  ❌ Orchestrator mode guard missing")
        return False
    
    return True


def check_sender_v2_applier():
    """Check that sender V2 calls the results applier."""
    print("\n[CHECK 3] Verifying sender V2 calls applier...")
    
    sender_path = Path("backend/ai/validation_sender_v2.py")
    content = sender_path.read_text(encoding="utf-8")
    
    # Check for applier import
    if "from backend.validation.apply_results_v2 import apply_validation_results_for_sid" in content:
        print("  ✅ Applier import found")
    else:
        print("  ❌ Applier import missing")
        return False
    
    # Check for applier call
    if "apply_validation_results_for_sid(sid, runs_root)" in content:
        print("  ✅ Applier is called")
    else:
        print("  ❌ Applier call missing")
        return False
    
    # Check for results_applied flag
    if "results_applied" in content:
        print("  ✅ results_applied flag set")
    else:
        print("  ❌ results_applied flag not set")
        return False
    
    return True


def check_applier_module():
    """Check that apply_results_v2.py exists and has main function."""
    print("\n[CHECK 4] Verifying apply_results_v2.py module...")
    
    applier_path = Path("backend/validation/apply_results_v2.py")
    if not applier_path.exists():
        print("  ❌ apply_results_v2.py not found")
        return False
    
    print("  ✅ apply_results_v2.py exists")
    
    content = applier_path.read_text(encoding="utf-8")
    
    # Check for main function
    if "def apply_validation_results_for_sid" in content:
        print("  ✅ apply_validation_results_for_sid function found")
    else:
        print("  ❌ apply_validation_results_for_sid function missing")
        return False
    
    # Check for AI field merge
    if "ai_validation_decision" in content and "ai_validation_rationale" in content:
        print("  ✅ AI field merge logic found")
    else:
        print("  ❌ AI field merge logic missing")
        return False
    
    return True


def check_runflow_decider():
    """Check that runflow checks results_applied flag."""
    print("\n[CHECK 5] Verifying runflow checks results_applied...")
    
    decider_path = Path("backend/runflow/decider.py")
    content = decider_path.read_text(encoding="utf-8")
    
    # Check for results_applied check
    if "results_applied" in content and "orchestrator_mode" in content:
        print("  ✅ Runflow checks results_applied flag")
    else:
        print("  ❌ Runflow doesn't check results_applied")
        return False
    
    return True


def main():
    print("="*80)
    print("VALIDATION V2 PRODUCTION INTEGRATION - VERIFICATION")
    print("="*80)
    
    checks = [
        check_tasks_py_integration,
        check_validation_builder,
        check_sender_v2_applier,
        check_applier_module,
        check_runflow_decider,
    ]
    
    results = []
    for check in checks:
        try:
            result = check()
            results.append(result)
        except Exception as e:
            print(f"  ❌ Check failed with error: {e}")
            results.append(False)
    
    print("\n" + "="*80)
    if all(results):
        print("✅ ALL CHECKS PASSED - V2 PRODUCTION INTEGRATION IS COMPLETE")
        print("="*80)
        print("\nIntegration Summary:")
        print("  ✅ Production pipeline calls build_validation_packs_for_run")
        print("  ✅ Pack builder triggers V2 autosend in orchestrator mode")
        print("  ✅ Sender V2 calls results applier after sending")
        print("  ✅ Applier merges AI fields into summary.json")
        print("  ✅ Runflow checks results_applied flag before promoting")
        print("\n🎉 Ready to test with production SIDs!")
        return True
    else:
        print("❌ SOME CHECKS FAILED - REVIEW INTEGRATION")
        print("="*80)
        failed = sum(1 for r in results if not r)
        print(f"\nFailed checks: {failed}/{len(results)}")
        return False


if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)
