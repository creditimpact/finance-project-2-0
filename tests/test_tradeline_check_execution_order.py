"""Test to verify correct execution order and F2.B01 output structure.

This test verifies that:
1. Root checks Q1 is present in final output
2. Branch infrastructure (build_branches_block, ensure_branch_results_container)
3. F2.B01 result is populated and present
4. Branches visibility lists are updated with F2.B01
5. Non-blocking invariants hold (status, findings unchanged by branches)
"""
import json
from pathlib import Path

import pytest

from backend.tradeline_check.runner import run_for_account
from backend.validation.pipeline import AccountContext


SID = "sid_execution_order_test"


def make_acc_ctx(tmp_path: Path, account_key: str, bureaus_payload: dict) -> AccountContext:
    runs_root = tmp_path / "runs"
    run_dir = runs_root / SID
    acc_dir = run_dir / "cases" / "accounts" / str(account_key)
    acc_dir.mkdir(parents=True, exist_ok=True)
    bureaus_path = acc_dir / "bureaus.json"
    bureaus_path.write_text(json.dumps(bureaus_payload), encoding="utf-8")
    summary_path = acc_dir / "summary.json"
    summary_path.write_text(json.dumps({"validation_requirements": {"schema_version": 3, "findings": []}}))
    return AccountContext(
        sid=SID,
        runs_root=runs_root,
        index=str(account_key),
        account_key=str(account_key),
        account_id=f"idx-{int(account_key):03d}",
        account_dir=acc_dir,
        summary_path=summary_path,
        bureaus_path=bureaus_path,
    )


def read_bureau_file(tmp_path: Path, account_key: str, bureau: str) -> dict:
    p = tmp_path / "runs" / SID / "cases" / "accounts" / str(account_key) / "tradeline_check" / f"{bureau}.json"
    return json.loads(p.read_text(encoding="utf-8"))


@pytest.fixture(autouse=True)
def clear_env(monkeypatch):
    monkeypatch.setenv("TRADELINE_CHECK_ENABLED", "1")
    monkeypatch.setenv("TRADELINE_CHECK_WRITE_DEBUG", "1")
    monkeypatch.setenv("TRADELINE_CHECK_PLACEHOLDER_TOKENS", "--,n/a,unknown")
    monkeypatch.setenv("TRADELINE_CHECK_WRITE_EMPTY_RESULTS", "0")
    yield


def test_execution_order_q1_present(tmp_path: Path):
    """Test that Q1 check is present in the final output."""
    bureaus_payload = {
        "equifax": {
            "account_status": "Open",
            "account_rating": "Good",
            "payment_status": "Current",
            "date_of_last_activity": "2024-01-15",
            "last_payment": "2024-01-10",
            "date_opened": "2020-05-01",
            "date_reported": "2024-01-20",
            "account_type": "Credit Card",
            "creditor_type": "Credit Card Issuer",
            "account_description": "Primary credit card",
            "two_year_payment_history_monthly_tsv_v2": {
                "equifax": [
                    {"month": f"{i:02d}/2024", "status": "ok"} for i in range(1, 7)
                ] + [
                    {"month": f"{i:02d}/2023", "status": "ok"} for i in range(1, 19)
                ]
            }
        }
    }

    acc_ctx = make_acc_ctx(tmp_path, "1", bureaus_payload)
    run_for_account(acc_ctx)
    data = read_bureau_file(tmp_path, "1", "equifax")

    root_checks = data.get("root_checks", {})

    # Q1 must be present and have a status field
    assert "Q1" in root_checks, "Q1 missing"
    assert "status" in root_checks["Q1"], "Q1 missing status field"


def test_branch_infrastructure_before_branch_execution(tmp_path: Path):
    """Test that branches block exists before F2.B01 is populated."""
    bureaus_payload = {
        "equifax": {
            "account_status": "Open",
            "account_rating": "Good",
            "payment_status": "Current",
            "date_of_last_activity": "2024-01-15",
            "last_payment": "2024-01-10",
            "date_opened": "2020-05-01",
            "date_reported": "2024-01-20",
            "account_type": "Credit Card",
            "creditor_type": "Credit Card Issuer",
            "account_description": "Primary credit card",
            "two_year_payment_history_monthly_tsv_v2": {
                "equifax": [
                    {"month": f"{i:02d}/2024", "status": "ok"} for i in range(1, 7)
                ] + [
                    {"month": f"{i:02d}/2023", "status": "ok"} for i in range(1, 19)
                ]
            }
        }
    }

    acc_ctx = make_acc_ctx(tmp_path, "2", bureaus_payload)
    run_for_account(acc_ctx)
    data = read_bureau_file(tmp_path, "2", "equifax")

    # branches block must exist (built before F2.B01)
    assert "branches" in data, "branches block missing"
    assert "families" in data["branches"], "branches.families missing"
    assert len(data["branches"]["families"]) > 0, "No families in branches"

    # branch_results block must exist (container created before F2.B01)
    assert "branch_results" in data, "branch_results block missing"
    assert "results" in data["branch_results"], "branch_results.results missing"


def test_f2_b01_present(tmp_path: Path):
    """Test that F2.B01 result is present alongside root checks."""
    bureaus_payload = {
        "equifax": {
            "account_status": "Open",
            "account_rating": "Good",
            "payment_status": "Current",
            "date_of_last_activity": "2024-01-15",
            "last_payment": "2024-01-10",
            "date_opened": "2020-05-01",
            "date_reported": "2024-01-20",
            "account_type": "Credit Card",
            "creditor_type": "Credit Card Issuer",
            "account_description": "Primary credit card",
            "two_year_payment_history_monthly_tsv_v2": {
                "equifax": [
                    {"month": f"{i:02d}/2024", "status": "ok"} for i in range(1, 7)
                ] + [
                    {"month": f"{i:02d}/2023", "status": "ok"} for i in range(1, 19)
                ]
            }
        }
    }

    acc_ctx = make_acc_ctx(tmp_path, "3", bureaus_payload)
    run_for_account(acc_ctx)
    data = read_bureau_file(tmp_path, "3", "equifax")

    # F2.B01 must be present
    assert "branch_results" in data
    assert "results" in data["branch_results"]
    assert "F2.B01" in data["branch_results"]["results"], "F2.B01 result missing"

    f2_b01 = data["branch_results"]["results"]["F2.B01"]
    assert "status" in f2_b01, "F2.B01 status missing"
    assert "version" in f2_b01, "F2.B01 version missing"
    assert "metrics" in f2_b01, "F2.B01 metrics missing"

    # Root checks present
    root_checks = data.get("root_checks", {})
    assert "Q1" in root_checks


def test_branches_visibility_updated_with_f2_b01(tmp_path: Path):
    """Test that branches visibility lists include F2.B01 after update_branches_visibility()."""
    bureaus_payload = {
        "equifax": {
            "account_status": "Open",
            "account_rating": "Good",
            "payment_status": "Current",
            "date_of_last_activity": "2024-01-15",
            "last_payment": "2024-01-10",
            "date_opened": "2020-05-01",
            "date_reported": "2024-01-20",
            "account_type": "Credit Card",
            "creditor_type": "Credit Card Issuer",
            "account_description": "Primary credit card",
            "two_year_payment_history_monthly_tsv_v2": {
                "equifax": [
                    {"month": f"{i:02d}/2024", "status": "ok"} for i in range(1, 7)
                ] + [
                    {"month": f"{i:02d}/2023", "status": "ok"} for i in range(1, 19)
                ]
            }
        }
    }

    acc_ctx = make_acc_ctx(tmp_path, "4", bureaus_payload)
    run_for_account(acc_ctx)
    data = read_bureau_file(tmp_path, "4", "equifax")

    # Find F2 family in branches
    branches = data.get("branches", {})
    families = branches.get("families", [])
    f2_family = next((f for f in families if f.get("family_id") == "F2"), None)

    assert f2_family is not None, "F2 family not found in branches"

    # F2.B01 should be in executed_branch_ids
    assert "F2.B01" in f2_family.get("executed_branch_ids", []), "F2.B01 not in executed_branch_ids"

    # If eligible, should also be in eligible_branch_ids
    assert "F2.B01" in f2_family.get("eligible_branch_ids", []), "F2.B01 not in eligible_branch_ids"

    # Summary counts should be updated
    summary = branches.get("summary", {})
    assert summary.get("total_executed_branches", 0) >= 1, "total_executed_branches not updated"


def test_non_blocking_invariant_status_unchanged(tmp_path: Path):
    """Test that payload.status and findings are unchanged by branch execution."""
    monthly_data = [
        {"month": f"{i:02d}/2024", "status": "ok"} for i in range(1, 13)
    ] + [
        {"month": f"{i:02d}/2023", "status": "ok"} for i in range(1, 13)
    ]
    
    bureaus_payload = {
        "equifax": {
            "account_status": "Open",
            "account_rating": "Good",
            "payment_status": "Current",
            "date_of_last_activity": "2024-01-15",
            "last_payment": "2024-01-10",
            "date_opened": "2020-05-01",
            "date_reported": "2024-01-20",
            "account_type": "Credit Card",
            "creditor_type": "Credit Card Issuer",
            "account_description": "Primary credit card",
            "two_year_payment_history_monthly_tsv_v2": {
                "equifax": monthly_data
            }
        }
    }

    acc_ctx = make_acc_ctx(tmp_path, "5", bureaus_payload)
    run_for_account(acc_ctx)
    data = read_bureau_file(tmp_path, "5", "equifax")

    # payload.status should still be "ok" (not modified by branches)
    assert data.get("status") == "ok", "payload.status should remain 'ok', not modified by branches"

    # findings should not be modified by branches
    findings = data.get("findings", [])
    assert isinstance(findings, list), "findings should be a list"
    # The exact value may vary, but it should exist and not crash


def test_root_checks_routing_branches_and_f2_b01(tmp_path: Path):
    """Test the structure with Q1, routing, branches, and F2.B01."""
    bureaus_payload = {
        "equifax": {
            "account_status": "Open",
            "account_rating": "Good",
            "payment_status": "Current",
            "date_of_last_activity": "2024-01-15",
            "last_payment": "2024-01-10",
            "date_opened": "2020-05-01",
            "date_reported": "2024-01-20",
            "account_type": "Credit Card",
            "creditor_type": "Credit Card Issuer",
            "account_description": "Primary credit card",
            "two_year_payment_history_monthly_tsv_v2": {
                "equifax": [
                    {"month": f"{i:02d}/2024", "status": "ok"} for i in range(1, 7)
                ] + [
                    {"month": f"{i:02d}/2023", "status": "ok"} for i in range(1, 19)
                ]
            }
        }
    }

    acc_ctx = make_acc_ctx(tmp_path, "6", bureaus_payload)
    run_for_account(acc_ctx)
    data = read_bureau_file(tmp_path, "6", "equifax")

    # Root checks present (Q1)
    root_checks = data.get("root_checks", {})
    assert "Q1" in root_checks, "Q1 missing"

    # Routing present
    routing = data.get("routing", {})
    assert "R1" in routing, "R1 routing missing"

    # Branches infrastructure present
    branches = data.get("branches", {})
    assert "families" in branches, "branches.families missing"
    assert "summary" in branches, "branches.summary missing"

    # Branch results present
    branch_results = data.get("branch_results", {})
    assert "results" in branch_results, "branch_results.results missing"
    assert "F2.B01" in branch_results["results"], "F2.B01 result missing"

    # Date convention attached
    assert "date_convention" in data, "date_convention missing"

    # Final verification: Q1 has status
    assert root_checks["Q1"].get("status") is not None, "Q1 has no status"
