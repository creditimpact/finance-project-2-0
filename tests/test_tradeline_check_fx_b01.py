"""
tests/test_tradeline_check_fx_b01.py

Unit tests for FX.B01: last_payment monotonicity check.

Tests FX.B01 behavioral validation logic:
- Calendar-based severity monotonicity after last_payment month
- English month name parsing (jan/feb/.../dec)
- Year extraction from monthly entries
- Severity mapping (OK=0, 30=30, 60=60, 90=90, 120=120, 150=150, 180/CO=999)
- Missing data handling (skipped_missing_data)
- Ungated execution (no R1 dependency)
"""
import json
from pathlib import Path

import pytest

from backend.tradeline_check.runner import run_for_account
from backend.validation.pipeline import AccountContext


SID = "sid_fx_b01_test"


def make_acc_ctx(tmp_path: Path, account_key: str, bureaus_payload: dict) -> AccountContext:
    """Creates test account context with bureaus.json and summary.json"""
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
    """Reads generated bureau JSON from tradeline_check output"""
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
        entries: List of (month_str, status_str) tuples like ("01/2023", "OK")
    
    Returns:
        List of {"month": ..., "status": ...} dicts
    """
    return [{"month": month, "status": status} for month, status in entries]

def test_fx_b01_monotonic_worsening_returns_ok(tmp_path: Path):
    """
    Scenario: last_payment in Jan 2023, severity worsens (OK → 30 → 60).
    Expected: FX.B01 status=ok (monotonic worsening is valid).
    """
    acc_ctx = make_acc_ctx(
        tmp_path,
        account_key="1",
        bureaus_payload={
            "equifax": {
                "account_status": "Open",
                "date_reported": "2024-01-15",
                "last_payment": "2023-01-15",
            },
            "two_year_payment_history_monthly_tsv_v2": {
                "equifax": make_monthly_history([
                    ("01/2023", "OK"),
                    ("02/2023", "30"),
                    ("03/2023", "60"),
                    ("04/2023", "90"),
                ])
            }
        }
    )
    
    run_for_account(acc_ctx)
    data = read_bureau_file(tmp_path, "1", "equifax")
    
    assert "branch_results" in data
    assert "FX.B01" in data["branch_results"]["results"]
    
    fx_b01 = data["branch_results"]["results"]["FX.B01"]
    assert fx_b01["status"] == "ok"
    assert fx_b01["ungated"] is True
    assert fx_b01["eligible"] is True
    assert fx_b01["executed"] is True
    assert fx_b01["fired"] is False
    assert fx_b01["evidence"]["detected_violation"] is None


def test_fx_b01_severity_improvement_returns_conflict(tmp_path: Path):
    """
    Scenario: last_payment in Jan 2023, severity improves (90 → 30 in Mar).
    Expected: FX.B01 status=conflict (severity improvement violates monotonicity).
    """
    acc_ctx = make_acc_ctx(
        tmp_path,
        account_key="2",
        bureaus_payload={
            "equifax": {
                "account_status": "Open",
                "date_reported": "2024-01-15",
                "last_payment": "2023-01-15",
            },
            "two_year_payment_history_monthly_tsv_v2": {
                "equifax": make_monthly_history([
                    ("01/2023", "90"),
                    ("02/2023", "90"),
                    ("03/2023", "30"),  # improvement: conflict
                    ("04/2023", "60"),
                ])
            }
        }
    )
    
    run_for_account(acc_ctx)
    data = read_bureau_file(tmp_path, "2", "equifax")
    
    fx_b01 = data["branch_results"]["results"]["FX.B01"]
    assert fx_b01["status"] == "conflict"
    assert fx_b01["fired"] is True
    assert fx_b01["evidence"]["detected_violation"] is not None
    assert fx_b01["evidence"]["detected_violation"] is not None
    assert fx_b01["evidence"]["detected_violation"] is not None
    assert fx_b01["evidence"]["detected_violation"] is not None


def test_fx_b01_missing_last_payment_returns_skipped(tmp_path: Path):
    """
    Scenario: last_payment field missing.
    Expected: FX.B01 status=skipped_missing_data (insufficient anchor).
    """
    acc_ctx = make_acc_ctx(
        tmp_path,
        account_key="3",
        bureaus_payload={
            "equifax": {
                "account_status": "Open",
                "date_reported": "2024-01-15",
            },
            "two_year_payment_history_monthly_tsv_v2": {
                "equifax": make_monthly_history([
                    ("01/2023", "OK"),
                    ("02/2023", "30"),
                ])
            }
        }
    )
    
    run_for_account(acc_ctx)
    data = read_bureau_file(tmp_path, "3", "equifax")
    
    fx_b01 = data["branch_results"]["results"]["FX.B01"]
    assert fx_b01["status"] == "skipped_missing_data"
    assert fx_b01["fired"] is False


def test_fx_b01_unparseable_last_payment_returns_skipped(tmp_path: Path):
    """
    Scenario: last_payment is unparseable garbage.
    Expected: FX.B01 status=skipped_missing_data.
    """
    acc_ctx = make_acc_ctx(
        tmp_path,
        account_key="4",
        bureaus_payload={
            "equifax": {
                "account_status": "Open",
                "date_reported": "2024-01-15",
                "last_payment": "NOT-A-DATE",
            },
            "two_year_payment_history_monthly_tsv_v2": {
                "equifax": make_monthly_history([
                    ("01/2023", "OK"),
                ])
            }
        }
    )
    
    run_for_account(acc_ctx)
    data = read_bureau_file(tmp_path, "4", "equifax")
    
    fx_b01 = data["branch_results"]["results"]["FX.B01"]
    assert fx_b01["status"] == "skipped_missing_data"
    assert fx_b01["fired"] is False


def test_fx_b01_last_payment_month_not_in_history_returns_skipped(tmp_path: Path):
    """
    Scenario: last_payment month not found in two_year_payment_history_monthly_tsv_v2.
    Expected: FX.B01 status=skipped_missing_data (cannot validate without anchor).
    """
    acc_ctx = make_acc_ctx(
        tmp_path,
        account_key="5",
        bureaus_payload={
            "equifax": {
                "account_status": "Open",
                "date_reported": "2024-01-15",
                "last_payment": "2022-12-15",
            },
            "two_year_payment_history_monthly_tsv_v2": {
                "equifax": make_monthly_history([
                    ("01/2023", "OK"),
                    ("02/2023", "30"),
                ])
            }
        }
    )
    
    run_for_account(acc_ctx)
    data = read_bureau_file(tmp_path, "5", "equifax")
    
    fx_b01 = data["branch_results"]["results"]["FX.B01"]
    assert fx_b01["status"] == "skipped_missing_data"
    assert fx_b01["fired"] is False


def test_fx_b01_missing_monthly_history_returns_skipped(tmp_path: Path):
    """
    Scenario: two_year_payment_history_monthly_tsv_v2 missing entirely.
    Expected: FX.B01 status=skipped_missing_data.
    """
    acc_ctx = make_acc_ctx(
        tmp_path,
        account_key="6",
        bureaus_payload={
            "equifax": {
                "account_status": "Open",
                "date_reported": "2024-01-15",
                "last_payment": "2023-01-15",
            }
        }
    )
    
    run_for_account(acc_ctx)
    data = read_bureau_file(tmp_path, "6", "equifax")
    
    fx_b01 = data["branch_results"]["results"]["FX.B01"]
    assert fx_b01["status"] == "skipped_missing_data"
    assert fx_b01["fired"] is False


def test_fx_b01_ignores_missing_status_months(tmp_path: Path):
    """
    Scenario: Some months have status="--" (missing), should be ignored.
    Severity: 60 → -- → 90 (worsening when ignoring --).
    Expected: FX.B01 status=ok.
    """
    acc_ctx = make_acc_ctx(
        tmp_path,
        account_key="7",
        bureaus_payload={
            "equifax": {
                "account_status": "Open",
                "date_reported": "2024-01-15",
                "last_payment": "2023-01-15",
            },
            "two_year_payment_history_monthly_tsv_v2": {
                "equifax": make_monthly_history([
                    ("01/2023", "60"),
                    ("02/2023", "--"),
                    ("03/2023", "90"),
                ])
            }
        }
    )
    
    run_for_account(acc_ctx)
    data = read_bureau_file(tmp_path, "7", "equifax")
    
    fx_b01 = data["branch_results"]["results"]["FX.B01"]
    assert fx_b01["status"] == "ok"
    assert fx_b01["fired"] is False


def test_fx_b01_chargeoff_highest_severity(tmp_path: Path):
    """
    Scenario: last_payment in Jan, chargeoff in Feb (severity 999).
    Expected: FX.B01 status=ok (chargeoff is terminal severity).
    """
    acc_ctx = make_acc_ctx(
        tmp_path,
        account_key="8",
        bureaus_payload={
            "equifax": {
                "account_status": "Open",
                "date_reported": "2024-01-15",
                "last_payment": "2023-01-15",
            },
            "two_year_payment_history_monthly_tsv_v2": {
                "equifax": make_monthly_history([
                    ("01/2023", "90"),
                    ("02/2023", "CO"),
                ])
            }
        }
    )
    
    run_for_account(acc_ctx)
    data = read_bureau_file(tmp_path, "8", "equifax")
    
    fx_b01 = data["branch_results"]["results"]["FX.B01"]
    assert fx_b01["status"] == "ok"
    assert fx_b01["fired"] is False


def test_fx_b01_chargeoff_followed_by_improvement_conflict(tmp_path: Path):
    """
    Scenario: CO (999) → 30 (improvement from terminal severity).
    Expected: FX.B01 status=conflict.
    """
    acc_ctx = make_acc_ctx(
        tmp_path,
        account_key="9",
        bureaus_payload={
            "equifax": {
                "account_status": "Open",
                "date_reported": "2024-01-15",
                "last_payment": "2023-01-15",
            },
            "two_year_payment_history_monthly_tsv_v2": {
                "equifax": make_monthly_history([
                    ("01/2023", "CO"),
                    ("02/2023", "30"),  # 30 < 999: conflict
                ])
            }
        }
    )
    
    run_for_account(acc_ctx)
    data = read_bureau_file(tmp_path, "9", "equifax")
    
    fx_b01 = data["branch_results"]["results"]["FX.B01"]
    assert fx_b01["status"] == "conflict"
    assert fx_b01["fired"] is True


def test_fx_b01_same_severity_maintained_ok(tmp_path: Path):
    """
    Scenario: Severity stays constant (60 → 60 → 60).
    Expected: FX.B01 status=ok (no improvement = ok).
    """
    acc_ctx = make_acc_ctx(
        tmp_path,
        account_key="10",
        bureaus_payload={
            "equifax": {
                "account_status": "Open",
                "date_reported": "2024-01-15",
                "last_payment": "2023-01-15",
            },
            "two_year_payment_history_monthly_tsv_v2": {
                "equifax": make_monthly_history([
                    ("01/2023", "60"),
                    ("02/2023", "60"),
                    ("03/2023", "60"),
                ])
            }
        }
    )
    
    run_for_account(acc_ctx)
    data = read_bureau_file(tmp_path, "10", "equifax")
    
    fx_b01 = data["branch_results"]["results"]["FX.B01"]
    assert fx_b01["status"] == "ok"
    assert fx_b01["fired"] is False


def test_fx_b01_year_rollover_monotonic(tmp_path: Path):
    """
    Scenario: last_payment in Dec 2022, severity worsens into Jan 2023.
    Expected: FX.B01 status=ok (handles year rollover correctly).
    """
    acc_ctx = make_acc_ctx(
        tmp_path,
        account_key="11",
        bureaus_payload={
            "equifax": {
                "account_status": "Open",
                "date_reported": "2024-01-15",
                "last_payment": "2022-12-15",
            },
            "two_year_payment_history_monthly_tsv_v2": {
                "equifax": make_monthly_history([
                    ("12/2022", "30"),
                    ("01/2023", "60"),
                    ("02/2023", "90"),
                ])
            }
        }
    )
    
    run_for_account(acc_ctx)
    data = read_bureau_file(tmp_path, "11", "equifax")
    
    fx_b01 = data["branch_results"]["results"]["FX.B01"]
    assert fx_b01["status"] == "ok"
    assert fx_b01["fired"] is False


def test_fx_b01_year_rollover_improvement_conflict(tmp_path: Path):
    """
    Scenario: Dec 2022: 90 → Jan 2023: 30 (improvement across year boundary).
    Expected: FX.B01 status=conflict.
    """
    acc_ctx = make_acc_ctx(
        tmp_path,
        account_key="12",
        bureaus_payload={
            "equifax": {
                "account_status": "Open",
                "date_reported": "2024-01-15",
                "last_payment": "2022-12-15",
            },
            "two_year_payment_history_monthly_tsv_v2": {
                "equifax": make_monthly_history([
                    ("12/2022", "90"),
                    ("01/2023", "30"),  # improvement
                ])
            }
        }
    )
    
    run_for_account(acc_ctx)
    data = read_bureau_file(tmp_path, "12", "equifax")
    
    fx_b01 = data["branch_results"]["results"]["FX.B01"]
    assert fx_b01["status"] == "conflict"
    assert fx_b01["fired"] is True


def test_fx_b01_only_last_payment_month_in_history(tmp_path: Path):
    """
    Scenario: Only last_payment month present in history, no subsequent months.
    Expected: FX.B01 status=ok (no violations possible with zero post-anchor months).
    """
    acc_ctx = make_acc_ctx(
        tmp_path,
        account_key="13",
        bureaus_payload={
            "equifax": {
                "account_status": "Open",
                "date_reported": "2024-01-15",
                "last_payment": "2023-03-15",
            },
            "two_year_payment_history_monthly_tsv_v2": {
                "equifax": make_monthly_history([
                    ("03/2023", "60"),
                ])
            }
        }
    )
    
    run_for_account(acc_ctx)
    data = read_bureau_file(tmp_path, "13", "equifax")
    
    fx_b01 = data["branch_results"]["results"]["FX.B01"]
    assert fx_b01["status"] == "ok"
    assert fx_b01["fired"] is False
    assert fx_b01["evidence"]["monthly_entries_checked"] == 0


def test_fx_b01_30_to_ok_improvement_conflict(tmp_path: Path):
    """
    Scenario: 30 → OK (improvement from delinquent to current).
    Expected: FX.B01 status=conflict.
    """
    acc_ctx = make_acc_ctx(
        tmp_path,
        account_key="14",
        bureaus_payload={
            "equifax": {
                "account_status": "Open",
                "date_reported": "2024-01-15",
                "last_payment": "2023-01-15",
            },
            "two_year_payment_history_monthly_tsv_v2": {
                "equifax": make_monthly_history([
                    ("01/2023", "30"),
                    ("02/2023", "OK"),  # improvement
                ])
            }
        }
    )
    
    run_for_account(acc_ctx)
    data = read_bureau_file(tmp_path, "14", "equifax")
    
    fx_b01 = data["branch_results"]["results"]["FX.B01"]
    assert fx_b01["status"] == "conflict"
    assert fx_b01["fired"] is True


def test_fx_b01_empty_monthly_history(tmp_path: Path):
    """
    Scenario: two_year_payment_history_monthly_tsv_v2 is empty list.
    Expected: FX.B01 status=skipped_missing_data.
    """
    acc_ctx = make_acc_ctx(
        tmp_path,
        account_key="15",
        bureaus_payload={
            "equifax": {
                "account_status": "Open",
                "date_reported": "2024-01-15",
                "last_payment": "2023-01-15",
            },
            "two_year_payment_history_monthly_tsv_v2": {
                "equifax": []
            }
        }
    )
    
    run_for_account(acc_ctx)
    data = read_bureau_file(tmp_path, "15", "equifax")
    
    fx_b01 = data["branch_results"]["results"]["FX.B01"]
    assert fx_b01["status"] == "skipped_missing_data"
    assert fx_b01["fired"] is False
