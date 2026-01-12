import json
from pathlib import Path

import pytest

from backend.tradeline_check.runner import run_for_account
from backend.validation.pipeline import AccountContext


SID = "sid_f2_b01_test"


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


def make_monthly_history(entries: list[tuple[str, str]]) -> list[dict]:
    """Helper to build monthly history entries.
    
    Args:
        entries: List of (month_str, status_str) tuples
    
    Returns:
        List of {"month": ..., "status": ...} dicts
    """
    return [{"month": month, "status": status} for month, status in entries]


def test_f2_b01_eligible_open_q2_ok_monthly_has_activity(tmp_path: Path):
    """Test F2.B01: eligible open + Q2 ok + monthly has ok/late -> status=ok."""
    # Monthly history with mix of ok and delinquency
    monthly_data = make_monthly_history([
        ("01/2024", "ok"),
        ("02/2024", "ok"),
        ("03/2024", "90"),
        ("04/2024", "ok"),
        ("05/2024", "--"),
        ("06/2024", "ok"),
        ("07/2024", "120"),
        ("08/2024", "ok"),
        ("09/2024", "ok"),
        ("10/2024", "--"),
        ("11/2024", "ok"),
        ("12/2024", "ok"),
        ("01/2023", "ok"),
        ("02/2023", "ok"),
        ("03/2023", "ok"),
        ("04/2023", "ok"),
        ("05/2023", "ok"),
        ("06/2023", "ok"),
        ("07/2023", "ok"),
        ("08/2023", "ok"),
        ("09/2023", "ok"),
        ("10/2023", "ok"),
        ("11/2023", "ok"),
        ("12/2023", "ok"),
    ])
    
    acc_ctx = make_acc_ctx(
        tmp_path,
        account_key="1",
        bureaus_payload={
            "equifax": {
                "account_status": "Open",
                "date_of_last_activity": "2024-01-15",
            },
            "two_year_payment_history_monthly_tsv_v2": {
                "equifax": monthly_data,
            }
        },
    )
    
    run_for_account(acc_ctx)
    data = read_bureau_file(tmp_path, "1", "equifax")
    
    assert "branch_results" in data
    assert "F2.B01" in data["branch_results"]["results"]
    
    f2_b01 = data["branch_results"]["results"]["F2.B01"]
    assert f2_b01["version"] == "f2_b01_activity_vs_monthly_history_v1"
    assert f2_b01["status"] == "ok"
    assert f2_b01["eligible"] is True
    assert f2_b01["executed"] is True
    assert f2_b01["fired"] is False
    
    # Verify metrics
    metrics = f2_b01["metrics"]
    assert metrics["total_months"] == 24
    assert metrics["count_ok"] > 0
    assert metrics["count_delinquent"] > 0
    assert metrics["has_any_activity_in_monthly"] is True
    assert metrics["has_only_missing"] is False


def test_f2_b01_eligible_open_q2_ok_monthly_all_missing_conflict(tmp_path: Path):
    """Test F2.B01: eligible open + Q2 ok + monthly all "--" -> status=conflict."""
    # Monthly history all missing
    monthly_data = make_monthly_history([
        ("01/2024", "--"),
        ("02/2024", "--"),
        ("03/2024", "--"),
        ("04/2024", "--"),
        ("05/2024", "--"),
        ("06/2024", "--"),
        ("07/2024", "--"),
        ("08/2024", "--"),
        ("09/2024", "--"),
        ("10/2024", "--"),
        ("11/2024", "--"),
        ("12/2024", "--"),
        ("01/2023", "--"),
        ("02/2023", "--"),
        ("03/2023", "--"),
        ("04/2023", "--"),
        ("05/2023", "--"),
        ("06/2023", "--"),
        ("07/2023", "--"),
        ("08/2023", "--"),
        ("09/2023", "--"),
        ("10/2023", "--"),
        ("11/2023", "--"),
        ("12/2023", "--"),
    ])
    
    acc_ctx = make_acc_ctx(
        tmp_path,
        account_key="2",
        bureaus_payload={
            "experian": {
                "account_status": "Open",
                "date_of_last_activity": "2024-01-15",
            },
            "two_year_payment_history_monthly_tsv_v2": {
                "experian": monthly_data,
            }
        },
    )
    
    run_for_account(acc_ctx)
    data = read_bureau_file(tmp_path, "2", "experian")
    
    f2_b01 = data["branch_results"]["results"]["F2.B01"]
    assert f2_b01["status"] == "conflict"
    assert f2_b01["fired"] is True
    assert f2_b01["metrics"]["has_only_missing"] is True


def test_f2_b01_eligible_open_q2_skipped_monthly_has_activity_conflict(tmp_path: Path):
    """Test F2.B01: open + Q2 skipped + monthly has activity -> status=skipped (not eligible).
    
    After narrowing eligibility to state 1 only (Q1=open, Q2=ok), this scenario should be skipped
    because Q2=skipped_missing_data (state 2) is no longer eligible.
    """
    monthly_data = make_monthly_history([
        ("01/2024", "ok"),
        ("02/2024", "ok"),
        ("03/2024", "ok"),
        ("04/2024", "ok"),
        ("05/2024", "ok"),
        ("06/2024", "ok"),
        ("07/2024", "ok"),
        ("08/2024", "ok"),
        ("09/2024", "ok"),
        ("10/2024", "ok"),
        ("11/2024", "ok"),
        ("12/2024", "ok"),
        ("01/2023", "ok"),
        ("02/2023", "ok"),
        ("03/2023", "ok"),
        ("04/2023", "ok"),
        ("05/2023", "ok"),
        ("06/2023", "ok"),
        ("07/2023", "ok"),
        ("08/2023", "ok"),
        ("09/2023", "ok"),
        ("10/2023", "ok"),
        ("11/2023", "ok"),
        ("12/2023", "ok"),
    ])
    
    # Q2 will be skipped_missing_data because no activity fields
    acc_ctx = make_acc_ctx(
        tmp_path,
        account_key="3",
        bureaus_payload={
            "transunion": {
                "account_status": "Open",
                # No activity fields (date_of_last_activity, last_payment, etc.)
            },
            "two_year_payment_history_monthly_tsv_v2": {
                "transunion": monthly_data,
            }
        },
    )
    
    run_for_account(acc_ctx)
    data = read_bureau_file(tmp_path, "3", "transunion")
    
    f2_b01 = data["branch_results"]["results"]["F2.B01"]
    assert f2_b01["status"] == "skipped"
    assert f2_b01["eligible"] is False
    assert f2_b01["fired"] is False


def test_f2_b01_eligible_monthly_missing_unknown(tmp_path: Path):
    """Test F2.B01: eligible open + monthly missing/empty -> status=unknown."""
    acc_ctx = make_acc_ctx(
        tmp_path,
        account_key="4",
        bureaus_payload={
            "equifax": {
                "account_status": "Open",
                "date_of_last_activity": "2024-01-15",
            },
            # No two_year_payment_history_monthly_tsv_v2
        },
    )
    
    run_for_account(acc_ctx)
    data = read_bureau_file(tmp_path, "4", "equifax")
    
    f2_b01 = data["branch_results"]["results"]["F2.B01"]
    assert f2_b01["status"] == "unknown"
    assert f2_b01["fired"] is False
    assert f2_b01["metrics"]["total_months"] == 0


def test_f2_b01_non_eligible_closed_skipped(tmp_path: Path):
    """Test F2.B01: non-eligible state (closed) -> status=skipped."""
    # Closed account -> R1 state_num will be 3+ (not state 1)
    acc_ctx = make_acc_ctx(
        tmp_path,
        account_key="5",
        bureaus_payload={
            "experian": {
                "account_status": "Closed",
                "date_of_last_activity": "2023-01-15",
            }
        },
    )
    
    run_for_account(acc_ctx)
    data = read_bureau_file(tmp_path, "5", "experian")
    
    f2_b01 = data["branch_results"]["results"]["F2.B01"]
    assert f2_b01["status"] == "skipped"
    assert f2_b01["eligible"] is False
    assert f2_b01["executed"] is True
    assert f2_b01["fired"] is False


def test_f2_b01_metrics_correctness_mixed_statuses(tmp_path: Path):
    """Test F2.B01: metrics computed correctly with mix of ok/--/numeric."""
    test_entries = [
        ("01/2024", "ok"),
        ("02/2024", "--"),
        ("03/2024", "90"),
        ("04/2024", "ok"),
        ("05/2024", "--"),
        ("06/2024", "ok"),
        ("07/2024", "120"),
        ("08/2024", "ok"),
        ("09/2024", "ok"),
        ("10/2024", "--"),
        ("11/2024", "ok"),
        ("12/2024", "ok"),
        ("01/2023", "ok"),
        ("02/2023", "--"),
        ("03/2023", "60"),
        ("04/2023", "ok"),
        ("05/2023", "--"),
        ("06/2023", "ok"),
        ("07/2023", "150"),
        ("08/2023", "ok"),
        ("09/2023", "ok"),
        ("10/2023", "--"),
        ("11/2023", "ok"),
        ("12/2023", "ok"),
    ]
    monthly_data = make_monthly_history(test_entries)
    
    acc_ctx = make_acc_ctx(
        tmp_path,
        account_key="6",
        bureaus_payload={
            "transunion": {
                "account_status": "Open",
                "date_of_last_activity": "2024-01-15",
            },
            "two_year_payment_history_monthly_tsv_v2": {
                "transunion": monthly_data,
            }
        },
    )
    
    run_for_account(acc_ctx)
    data = read_bureau_file(tmp_path, "6", "transunion")
    
    f2_b01 = data["branch_results"]["results"]["F2.B01"]
    metrics = f2_b01["metrics"]
    
    # Count from test_entries source
    count_ok = sum(1 for _, status in test_entries if status == "ok")
    count_missing = sum(1 for _, status in test_entries if status == "--")
    count_delinquent = sum(1 for _, status in test_entries if status not in ("ok", "--"))
    
    assert metrics["total_months"] == 24
    assert metrics["count_ok"] == count_ok
    assert metrics["count_missing"] == count_missing
    assert metrics["count_delinquent"] == count_delinquent
    assert metrics["has_any_activity_in_monthly"] == (count_ok + count_delinquent > 0)
    assert metrics["has_only_missing"] == (count_missing == 24)


def test_f2_b01_non_blocking_invariant(tmp_path: Path):
    """Test F2.B01: does not modify payload status, gate, root_checks, routing."""
    monthly_data = make_monthly_history([
        ("01/2024", "--"),
        ("02/2024", "--"),
        ("03/2024", "--"),
        ("04/2024", "--"),
        ("05/2024", "--"),
        ("06/2024", "--"),
        ("07/2024", "--"),
        ("08/2024", "--"),
        ("09/2024", "--"),
        ("10/2024", "--"),
        ("11/2024", "--"),
        ("12/2024", "--"),
        ("01/2023", "--"),
        ("02/2023", "--"),
        ("03/2023", "--"),
        ("04/2023", "--"),
        ("05/2023", "--"),
        ("06/2023", "--"),
        ("07/2023", "--"),
        ("08/2023", "--"),
        ("09/2023", "--"),
        ("10/2023", "--"),
        ("11/2023", "--"),
        ("12/2023", "--"),
    ])
    
    acc_ctx = make_acc_ctx(
        tmp_path,
        account_key="7",
        bureaus_payload={
            "equifax": {
                "account_status": "Open",
                "date_of_last_activity": "2024-01-15",
            },
            "two_year_payment_history_monthly_tsv_v2": {
                "equifax": monthly_data,
            }
        },
    )
    
    run_for_account(acc_ctx)
    data = read_bureau_file(tmp_path, "7", "equifax")
    
    # Verify non-blocking invariants
    # - status is still "ok" (not modified by branch)
    assert data["status"] == "ok"
    
    # - findings list unchanged
    assert data.get("findings", []) == []
    
    # - blocked_questions removed from schema
    assert "blocked_questions" not in data
    
    # - root_checks present (Q1, Q2)
    assert "Q1" in data["root_checks"]
    assert "Q2" in data["root_checks"]
    
    # - routing not modified
    assert "R1" in data["routing"]
    
    # - Only branch_results and branches visibility changed
    assert "branch_results" in data
    assert "F2.B01" in data["branch_results"]["results"]
    
    # - branches visibility updated with F2.B01
    branches = data.get("branches", {})
    families = branches.get("families", [])
    f2_family = next((f for f in families if f.get("family_id") == "F2"), None)
    assert f2_family is not None
    assert "F2.B01" in f2_family.get("executed_branch_ids", [])


def test_f2_b01_evidence_extraction_first_and_last_six(tmp_path: Path):
    """Test F2.B01: evidence contains first 6 and last 6 months."""
    test_entries = [
        ("01/2024", "ok"),    # index 0-5: first 6
        ("02/2024", "ok"),
        ("03/2024", "ok"),
        ("04/2024", "ok"),
        ("05/2024", "ok"),
        ("06/2024", "ok"),
        ("07/2024", "ok"),
        ("08/2024", "ok"),
        ("09/2024", "ok"),
        ("10/2024", "ok"),
        ("11/2024", "ok"),
        ("12/2024", "ok"),
        ("01/2023", "ok"),
        ("02/2023", "ok"),
        ("03/2023", "ok"),
        ("04/2023", "ok"),
        ("05/2023", "ok"),
        ("06/2023", "ok"),
        ("07/2023", "ok"),    # index 18-23: last 6
        ("08/2023", "ok"),
        ("09/2023", "ok"),
        ("10/2023", "ok"),
        ("11/2023", "ok"),
        ("12/2023", "ok"),
    ]
    monthly_data = make_monthly_history(test_entries)
    
    acc_ctx = make_acc_ctx(
        tmp_path,
        account_key="8",
        bureaus_payload={
            "transunion": {
                "account_status": "Open",
                "date_of_last_activity": "2024-01-15",
            },
            "two_year_payment_history_monthly_tsv_v2": {
                "transunion": monthly_data,
            }
        },
    )
    
    run_for_account(acc_ctx)
    data = read_bureau_file(tmp_path, "8", "transunion")
    
    f2_b01 = data["branch_results"]["results"]["F2.B01"]
    evidence = f2_b01["evidence"]
    
    first_6 = evidence["first_6_months"]
    last_6 = evidence["last_6_months"]
    
    assert len(first_6) == 6
    assert first_6[0]["month"] == "01/2024"
    assert first_6[5]["month"] == "06/2024"
    
    assert len(last_6) == 6
    assert last_6[0]["month"] == "07/2023"
    assert last_6[5]["month"] == "12/2023"


def test_f2_b01_visibility_lists_updated_f2_family(tmp_path: Path):
    """Test F2.B01: branches visibility lists updated for F2 family."""
    monthly_data = make_monthly_history([
        ("01/2024", "ok"),
        ("02/2024", "ok"),
        ("03/2024", "ok"),
        ("04/2024", "ok"),
        ("05/2024", "ok"),
        ("06/2024", "ok"),
        ("07/2024", "ok"),
        ("08/2024", "ok"),
        ("09/2024", "ok"),
        ("10/2024", "ok"),
        ("11/2024", "ok"),
        ("12/2024", "ok"),
        ("01/2023", "ok"),
        ("02/2023", "ok"),
        ("03/2023", "ok"),
        ("04/2023", "ok"),
        ("05/2023", "ok"),
        ("06/2023", "ok"),
        ("07/2023", "ok"),
        ("08/2023", "ok"),
        ("09/2023", "ok"),
        ("10/2023", "ok"),
        ("11/2023", "ok"),
        ("12/2023", "ok"),
    ])
    
    acc_ctx = make_acc_ctx(
        tmp_path,
        account_key="9",
        bureaus_payload={
            "equifax": {
                "account_status": "Open",
                "date_of_last_activity": "2024-01-15",
            },
            "two_year_payment_history_monthly_tsv_v2": {
                "equifax": monthly_data,
            }
        },
    )
    
    run_for_account(acc_ctx)
    data = read_bureau_file(tmp_path, "9", "equifax")
    
    branches = data.get("branches", {})
    families = branches.get("families", [])
    f2_family = next((f for f in families if f.get("family_id") == "F2"), None)
    
    assert f2_family is not None
    assert "F2.B01" in f2_family["executed_branch_ids"]
    assert "F2.B01" in f2_family["eligible_branch_ids"]
    # Status is "ok" so not fired
    assert "F2.B01" not in f2_family["fired_branch_ids"]
    
    # Summary counts updated
    summary = branches.get("summary", {})
    assert summary["total_eligible_branches"] >= 1
    assert summary["total_executed_branches"] >= 1
    assert summary["total_fired_branches"] >= 0
