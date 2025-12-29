"""
Prove the merge_ai_applied fix on real SIDs.

This script:
1. Takes before snapshots of runflow and barriers
2. Runs the backfill script to set merge_ai_applied
3. Takes after snapshots
4. Verifies the fix resolved the timing bug

Test SIDs:
- 83830ae4-6406-4a7e-ad80-a3f721a3787b: Bug case (validation before merge)
- 61e8cb38-8a58-42e3-9477-58485d43cb52: Zero-packs case (should be unaffected)
- 9d4c385b-4688-46f1-b545-56ebb5ffff06: Healthy case with packs

Usage:
    python scripts/prove_merge_fix.py --runs-root runs
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from backend.runflow.decider import _compute_umbrella_barriers, reconcile_umbrella_barriers

# Test SIDs
BUG_SID = "83830ae4-6406-4a7e-ad80-a3f721a3787b"
ZERO_PACKS_SID = "61e8cb38-8a58-42e3-9477-58485d43cb52"
HEALTHY_SID = "9d4c385b-4688-46f1-b545-56ebb5ffff06"


def _load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        return {"_error": str(e)}


def _extract_snapshot(run_dir: Path) -> dict[str, Any]:
    """Extract key fields for before/after comparison."""
    runflow = _load_json(run_dir / "runflow.json")
    stages = runflow.get("stages", {})
    merge_stage = stages.get("merge", {})
    validation_stage = stages.get("validation", {})

    barriers = _compute_umbrella_barriers(run_dir)

    return {
        "merge": {
            "status": merge_stage.get("status"),
            "empty_ok": merge_stage.get("empty_ok"),
            "result_files": (merge_stage.get("results") or {}).get("result_files"),
            "created_packs": (merge_stage.get("metrics") or {}).get("created_packs"),
            "merge_ai_applied": merge_stage.get("merge_ai_applied"),
            "merge_ai_applied_at": merge_stage.get("merge_ai_applied_at"),
        },
        "validation": {
            "status": validation_stage.get("status"),
            "validation_ai_applied": (validation_stage.get("metrics") or {}).get("validation_ai_applied"),
        },
        "barriers": {
            "merge_ready": barriers.get("merge_ready"),
            "validation_ready": barriers.get("validation_ready"),
            "all_ready": barriers.get("all_ready"),
        },
    }


def test_sid(
    sid: str,
    runs_root: Path,
    expected_category: str,
) -> dict[str, Any]:
    """Test a single SID: take snapshots, run backfill, verify."""
    run_dir = runs_root / sid
    if not run_dir.exists():
        return {
            "sid": sid,
            "category": expected_category,
            "error": "run_dir_not_found",
        }

    print(f"\n{'='*80}")
    print(f"Testing SID: {sid}")
    print(f"Category: {expected_category}")
    print(f"{'='*80}")

    # BEFORE snapshot
    before = _extract_snapshot(run_dir)
    print(f"\n📸 BEFORE:")
    print(json.dumps(before, indent=2, ensure_ascii=False))

    # Run backfill (idempotent - safe to run multiple times)
    print(f"\n🔧 Running backfill...")
    from scripts.repair_merge_ai_applied_from_run import repair_runs
    repair_runs(runs_root, only_sid=sid, dry_run=False)

    # AFTER snapshot
    after = _extract_snapshot(run_dir)
    print(f"\n📸 AFTER:")
    print(json.dumps(after, indent=2, ensure_ascii=False))

    # Verify expectations
    print(f"\n✅ VERIFICATION:")
    results = {
        "sid": sid,
        "category": expected_category,
        "before": before,
        "after": after,
        "checks": {},
    }

    if expected_category == "bug":
        # Bug case: merge_ai_applied should now be True, merge_ready should be True
        checks = {
            "merge_ai_applied_set": after["merge"]["merge_ai_applied"] is True,
            "merge_ready_true": after["barriers"]["merge_ready"] is True,
            "validation_status_unchanged": (
                before["validation"]["status"] == after["validation"]["status"]
            ),
        }
        results["checks"] = checks
        results["pass"] = all(checks.values())
        
        for check, passed in checks.items():
            status = "✅" if passed else "❌"
            print(f"  {status} {check}: {passed}")

    elif expected_category == "zero-packs":
        # Zero-packs: should be unaffected (merge_ready should stay True)
        checks = {
            "empty_ok_true": after["merge"]["empty_ok"] is True,
            "merge_ready_before": before["barriers"]["merge_ready"] is True,
            "merge_ready_after": after["barriers"]["merge_ready"] is True,
            "no_change_needed": (
                before["merge"]["merge_ai_applied"] == after["merge"]["merge_ai_applied"]
            ),
        }
        results["checks"] = checks
        results["pass"] = all(checks.values())
        
        for check, passed in checks.items():
            status = "✅" if passed else "❌"
            print(f"  {status} {check}: {passed}")

    elif expected_category == "healthy":
        # Healthy case: should already have merge_ai_applied=True or get it from backfill
        checks = {
            "merge_ai_applied_after": after["merge"]["merge_ai_applied"] is True,
            "merge_ready_true": after["barriers"]["merge_ready"] is True,
        }
        results["checks"] = checks
        results["pass"] = all(checks.values())
        
        for check, passed in checks.items():
            status = "✅" if passed else "❌"
            print(f"  {status} {check}: {passed}")

    return results


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prove merge_ai_applied fix on real SIDs"
    )
    parser.add_argument(
        "--runs-root",
        type=str,
        required=True,
        help="Path to runs root directory",
    )
    parser.add_argument(
        "--sid",
        type=str,
        help="Optional: test only this SID (default: test all 3)",
    )
    args = parser.parse_args()

    runs_root = Path(args.runs_root)
    if not runs_root.exists():
        print(f"❌ Runs root not found: {runs_root}", file=sys.stderr)
        return 2

    print(f"🧪 MERGE_AI_APPLIED FIX VERIFICATION")
    print(f"Runs Root: {runs_root}")
    print(f"Timestamp: {datetime.utcnow().isoformat()}Z")

    test_cases = [
        (BUG_SID, "bug"),
        (ZERO_PACKS_SID, "zero-packs"),
        (HEALTHY_SID, "healthy"),
    ]

    if args.sid:
        # Test only specified SID
        category = "custom"
        for sid, cat in test_cases:
            if sid == args.sid:
                category = cat
                break
        test_cases = [(args.sid, category)]

    all_results = []
    for sid, category in test_cases:
        result = test_sid(sid, runs_root, category)
        all_results.append(result)

    # Summary
    print(f"\n{'='*80}")
    print(f"📊 SUMMARY")
    print(f"{'='*80}")
    
    passed = sum(1 for r in all_results if r.get("pass"))
    total = len(all_results)
    
    print(f"Passed: {passed}/{total}")
    
    for result in all_results:
        sid = result["sid"]
        category = result["category"]
        status = "✅ PASS" if result.get("pass") else "❌ FAIL"
        print(f"  {status} {sid[:8]}... ({category})")

    # Write detailed results to file
    output_file = runs_root / "merge_fix_verification_results.json"
    output_file.write_text(
        json.dumps(
            {
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "passed": passed,
                "total": total,
                "results": all_results,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"\n📄 Detailed results written to: {output_file}")

    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
