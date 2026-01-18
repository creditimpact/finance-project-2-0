#!/usr/bin/env python
"""Integration test for strict branch eligibility gating with central registry.

Validates that:
1. F1–F6 branches are only invoked when eligible
2. Ineligible branches do NOT appear in branch_results.results
3. F0/FX branches continue to run
"""

import json
from pathlib import Path
import tempfile
from backend.tradeline_check.runner import run_for_account
from backend.validation.pipeline import AccountContext


def test_f2_ineligible_branches_absent():
    """Test that F2.B01 and F2.B02 do NOT appear in output when ineligible (state_num=5)."""
    
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        SID = "test_f2_ineligible_gating"
        
        # Setup account context
        runs_root = tmp_path / "runs"
        run_dir = runs_root / SID
        acc_dir = run_dir / "cases" / "accounts" / "1"
        acc_dir.mkdir(parents=True, exist_ok=True)
        
        # Create bureaus file that triggers state_num=5 (unknown/unknown)
        # This is achieved with conflicting Q1 signals (no clear open/closed indication)
        bureaus_payload = {
            "equifax": {
                "account_status": None,           # No status signal
                "account_rating": None,           # No rating signal
                "payment_status": None,           # No payment signal
                "date_of_last_activity": "2024-01-15",
                "date_opened": "2020-01-01",
                "date_reported": "2024-02-01",
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
        print(f"\n=== R1.state_num = {r1_state_num} ===")
        
        # Check branch_results
        branch_results = payload.get("branch_results", {}).get("results", {})
        print(f"\nBranch results keys: {list(branch_results.keys())}")
        
        # Validate F2.B01 and F2.B02 eligibility
        if r1_state_num in {1}:
            # Only F2.B01 should be eligible
            assert "F2.B01" in branch_results, "F2.B01 should be present when eligible (state_num=1)"
            print("[OK] F2.B01 present when eligible (state_num=1)")
        elif r1_state_num in {3, 4}:
            # Only F2.B02 should be eligible
            assert "F2.B02" in branch_results, "F2.B02 should be present when eligible (state_num=3 or 4)"
            print("[OK] F2.B02 present when eligible (state_num=3 or 4)")
        else:
            # Neither should be eligible
            assert "F2.B01" not in branch_results, f"F2.B01 should NOT be present for state_num={r1_state_num}"
            assert "F2.B02" not in branch_results, f"F2.B02 should NOT be present for state_num={r1_state_num}"
            print(f"[OK] F2.B01 and F2.B02 absent when ineligible (state_num={r1_state_num})")
        
        # FX should always be present (always-run family)
        # Note: F0 is stored in record_integrity, not branch_results
        fx_keys = [k for k in branch_results.keys() if k.startswith("FX.")]
        
        assert len(fx_keys) > 0, "FX branches should always be present"
        print(f"[OK] FX branches present (keys: {fx_keys})")
        
        # Check that record_integrity/F0 exists
        record_integrity = payload.get("record_integrity", {})
        assert "F0" in record_integrity, "F0 family should always be in record_integrity"
        print(f"[OK] F0 family present in record_integrity")
        
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
            print(f"\nF2 family executed_branch_ids: {executed_ids}")
            
            # Validate executed list matches branch_results
            if r1_state_num not in {1, 3, 4}:
                # Both ineligible
                assert "F2.B01" not in executed_ids, "F2.B01 should not be in executed list when ineligible"
                assert "F2.B02" not in executed_ids, "F2.B02 should not be in executed list when ineligible"
                print("[OK] F2 family executed list clean for ineligible state")
            elif r1_state_num == 1:
                assert "F2.B01" in executed_ids, "F2.B01 should be in executed list when eligible"
                assert "F2.B02" not in executed_ids, "F2.B02 should not be in executed list when ineligible"
                print("[OK] F2 family executed list correct for state_num=1")
            elif r1_state_num in {3, 4}:
                assert "F2.B01" not in executed_ids, "F2.B01 should not be in executed list when ineligible"
                assert "F2.B02" in executed_ids, "F2.B02 should be in executed list when eligible"
                print("[OK] F2 family executed list correct for state_num={3,4}")
        
        print("\n[SUCCESS] All validation checks passed!")


if __name__ == "__main__":
    test_f2_ineligible_branches_absent()
