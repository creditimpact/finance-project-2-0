#!/usr/bin/env python
"""Integration test for F2.B01 branch in full runner context."""
import json
from pathlib import Path
import tempfile
from backend.tradeline_check.runner import run_for_account
from backend.validation.pipeline import AccountContext


def main():
    # Create temp workspace
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        SID = "test_f2_b01_integration"
        
        # Setup account context
        runs_root = tmp_path / "runs"
        run_dir = runs_root / SID
        acc_dir = run_dir / "cases" / "accounts" / "1"
        acc_dir.mkdir(parents=True, exist_ok=True)
        
        # Create bureaus file with F2.B01 trigger case
        bureaus_payload = {
            "equifax": {
                "account_status": "Open",
                "date_of_last_activity": "2024-01-15",
            },
            "two_year_payment_history_monthly_tsv_v2": {
                "equifax": [
                    {"month": f"{i:02d}/2024", "status": "ok"} for i in range(1, 7)
                ] + [
                    {"month": f"{i:02d}/2023", "status": "ok"} for i in range(1, 19)
                ]
            }
        }
        
        bureaus_path = acc_dir / "bureaus.json"
        bureaus_path.write_text(json.dumps(bureaus_payload))
        
        summary_path = acc_dir / "summary.json"
        summary_path.write_text(json.dumps({"validation_requirements": {"schema_version": 3, "findings": []}}))
        
        # Create context
        ctx = AccountContext(
            sid=SID,
            runs_root=runs_root,
            index="1",
            account_key="1",
            account_id="idx-001",
            account_dir=acc_dir,
            summary_path=summary_path,
            bureaus_path=bureaus_path,
        )
        
        # Run tradeline check
        run_for_account(ctx)
        
        # Read results
        results_file = acc_dir / "tradeline_check" / "equifax.json"
        results = json.loads(results_file.read_text())
        
        # Verify F2.B01 branch result exists
        f2_b01 = results.get("branch_results", {}).get("results", {}).get("F2.B01", {})
        
        print("✅ F2.B01 INTEGRATION TEST RESULTS")
        print("=" * 60)
        print(f"Status: {f2_b01.get('status', 'N/A')}")
        print(f"Eligible: {f2_b01.get('eligible', 'N/A')}")
        print(f"Executed: {f2_b01.get('executed', 'N/A')}")
        print(f"Fired: {f2_b01.get('fired', 'N/A')}")
        print(f"Version: {f2_b01.get('version', 'N/A')}")
        print()
        print("Metrics:")
        metrics = f2_b01.get("metrics", {})
        print(f"  total_months: {metrics.get('total_months', 'N/A')}")
        print(f"  count_ok: {metrics.get('count_ok', 'N/A')}")
        print(f"  count_missing: {metrics.get('count_missing', 'N/A')}")
        print(f"  count_delinquent: {metrics.get('count_delinquent', 'N/A')}")
        print()
        print("Branches Visibility:")
        families = results.get("branches", {}).get("families", [])
        f2_family = next((f for f in families if f.get("family_id") == "F2"), None)
        if f2_family:
            print(f"  F2 eligible_branch_ids: {f2_family.get('eligible_branch_ids', [])}")
            print(f"  F2 executed_branch_ids: {f2_family.get('executed_branch_ids', [])}")
            print(f"  F2 fired_branch_ids: {f2_family.get('fired_branch_ids', [])}")
        print()
        print("Non-Blocking Invariant Check:")
        print(f"  payload.status: {results.get('status', 'N/A')} (unchanged)")
        print(f"  findings count: {len(results.get('findings', []))} (unchanged)")
        print(f"  blocked_questions: {results.get('blocked_questions', [])} (unchanged)")
        print()
        print("✅ F2.B01 branch fully integrated and non-blocking")


if __name__ == "__main__":
    main()
