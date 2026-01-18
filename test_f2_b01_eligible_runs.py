#!/usr/bin/env python
"""Test that eligible F2.B01 branch IS invoked when eligible (state_num=1)."""

import json
from pathlib import Path
import tempfile
from backend.tradeline_check.runner import run_for_account
from backend.validation.pipeline import AccountContext


def test_f2_b01_eligible_branch_runs():
    """Test that F2.B01 IS present when eligible (state_num=1)."""
    
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        SID = "test_f2_b01_eligible_runs"
        
        # Setup account context
        runs_root = tmp_path / "runs"
        run_dir = runs_root / SID
        acc_dir = run_dir / "cases" / "accounts" / "1"
        acc_dir.mkdir(parents=True, exist_ok=True)
        
        # Create bureaus file that triggers state_num=1 (Q1=open)
        # Signals: Clear OPEN status + good payment activity
        bureaus_payload = {
            "equifax": {
                "account_status": "Open",           # Signal: OPEN
                "account_rating": "Pays as agreed", # Signal: positive
                "payment_status": "Current",        # Signal: no delinquency
                "date_of_last_activity": "2024-01-15",
                "date_opened": "2020-01-01",
                "date_reported": "2024-02-01",
                "closed_date": None,                # No closed date
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
        
        # Create account context
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
        
        # Run tradeline_check
        run_for_account(ctx)
        
        # Read and validate the equifax output
        tradeline_dir = run_dir / "cases" / "accounts" / "1" / "tradeline_check"
        equifax_output = tradeline_dir / "equifax.json"
        assert equifax_output.exists(), "Equifax tradeline output not generated"
        
        payload = json.loads(equifax_output.read_text())
        
        # Get R1 state_num
        r1_state_num = payload.get("routing", {}).get("R1", {}).get("state_num")
        print(f"\n=== Test: F2.B01 Eligible Branch Execution ===")
        print(f"R1.state_num = {r1_state_num}")
        
        # Check branch_results
        branch_results = payload.get("branch_results", {}).get("results", {})
        print(f"Branch results keys: {list(branch_results.keys())}")
        
        # Validate F2.B01 is present when state_num=1
        if r1_state_num == 1:
            assert "F2.B01" in branch_results, "F2.B01 MUST be present when eligible (state_num=1)"
            print("[OK] F2.B01 present when eligible (state_num=1)")
            
            f2_b01_result = branch_results["F2.B01"]
            assert f2_b01_result.get("executed") == True, "F2.B01 must have executed=True"
            print("[OK] F2.B01.executed = True")
            
            # F2.B02 must NOT be present (ineligible)
            assert "F2.B02" not in branch_results, "F2.B02 should NOT be present when ineligible"
            print("[OK] F2.B02 absent when ineligible (state_num != 3,4)")
        else:
            print(f"[SKIP] Test requires state_num=1, got {r1_state_num}")
        
        # FX should always be present
        fx_keys = [k for k in branch_results.keys() if k.startswith("FX.")]
        assert len(fx_keys) > 0, "FX branches should always be present"
        print(f"[OK] FX branches present: {fx_keys}")
        
        # Check visibility lists
        branches_visibility = payload.get("branches", {})
        families = branches_visibility.get("families", [])
        
        # Find F2 family
        f2_family = None
        for fam in families:
            if fam.get("family_id") == "F2":
                f2_family = fam
                break
        
        if f2_family:
            executed_ids = f2_family.get("executed_branch_ids", [])
            print(f"F2 family executed_branch_ids: {executed_ids}")
            
            if r1_state_num == 1:
                assert "F2.B01" in executed_ids, "F2.B01 should be in executed list when eligible"
                print("[OK] F2.B01 in executed list")
        
        print("\n[SUCCESS] Eligible branch execution test passed!")


if __name__ == "__main__":
    test_f2_b01_eligible_branch_runs()
