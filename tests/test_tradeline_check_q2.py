import json
from pathlib import Path

import pytest

from backend.tradeline_check.runner import run_for_account
from backend.validation.pipeline import AccountContext


SID = "sid_q2_test"


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
    monkeypatch.setenv("TRADELINE_CHECK_GATE_STRICT", "0")
    monkeypatch.setenv("TRADELINE_CHECK_PLACEHOLDER_TOKENS", "--,n/a,unknown")
    monkeypatch.setenv("TRADELINE_CHECK_WRITE_EMPTY_RESULTS", "0")
    yield


def test_q2_open_with_activity_ok(tmp_path: Path):
    acc_ctx = make_acc_ctx(
        tmp_path,
        account_key="1",
        bureaus_payload={
            "equifax": {
                "account_status": "Open",
                "date_of_last_activity": "2025-01-10",
            }
        },
    )
    run_for_account(acc_ctx)
    data = read_bureau_file(tmp_path, "1", "equifax")
    q2 = data["root_checks"]["Q2"]
    assert q2["version"] == "q2_activity_v1"
    assert q2["expected_activity"] is True
    assert q2["observed_activity"] is True
    assert q2["status"] == "ok"


def test_q2_open_no_activity_skipped(tmp_path: Path):
    acc_ctx = make_acc_ctx(
        tmp_path,
        account_key="2",
        bureaus_payload={
            "experian": {
                "account_status": "Open",
                "date_of_last_activity": "--",
                "last_payment": None,
                "date_reported": "",
            }
        },
    )
    run_for_account(acc_ctx)
    data = read_bureau_file(tmp_path, "2", "experian")
    q2 = data["root_checks"]["Q2"]
    assert q2["expected_activity"] is True
    assert q2["observed_activity"] is None
    assert q2["status"] == "skipped_missing_data"


def test_q2_closed_with_activity_conflict(tmp_path: Path):
    acc_ctx = make_acc_ctx(
        tmp_path,
        account_key="3",
        bureaus_payload={
            "transunion": {
                "account_status": "Closed",
                "last_payment": "2024-12-01",
            }
        },
    )
    run_for_account(acc_ctx)
    data = read_bureau_file(tmp_path, "3", "transunion")
    q2 = data["root_checks"]["Q2"]
    assert q2["expected_activity"] is False
    assert q2["observed_activity"] is True
    assert q2["status"] == "conflict"


def test_q2_closed_no_activity_ok(tmp_path: Path):
    acc_ctx = make_acc_ctx(
        tmp_path,
        account_key="4",
        bureaus_payload={
            "equifax": {
                "account_status": "Closed",
                "date_of_last_activity": "--",
                "last_payment": None,
            }
        },
    )
    run_for_account(acc_ctx)
    data = read_bureau_file(tmp_path, "4", "equifax")
    q2 = data["root_checks"]["Q2"]
    assert q2["expected_activity"] is False
    assert q2["observed_activity"] is None
    assert q2["status"] == "ok"


def test_q2_q1_unknown_returns_unknown(tmp_path: Path):
    acc_ctx = make_acc_ctx(
        tmp_path,
        account_key="5",
        bureaus_payload={
            "experian": {
                "account_status": "--",
                "date_of_last_activity": "2024-01-01",
            }
        },
    )
    run_for_account(acc_ctx)
    data = read_bureau_file(tmp_path, "5", "experian")
    q2 = data["root_checks"]["Q2"]
    assert q2["status"] == "unknown"
    assert q2["expected_activity"] is None
    assert q2["observed_activity"] is None


def test_q2_placeholders_honored(tmp_path: Path):
    acc_ctx = make_acc_ctx(
        tmp_path,
        account_key="6",
        bureaus_payload={
            "transunion": {
                "account_status": "Open",
                "last_payment": "--",
                "date_reported": "unknown",
            }
        },
    )
    run_for_account(acc_ctx)
    data = read_bureau_file(tmp_path, "6", "transunion")
    q2 = data["root_checks"]["Q2"]
    assert q2["expected_activity"] is True
    assert q2["observed_activity"] is None
    assert q2["status"] == "skipped_missing_data"


def test_q2_fallback_ordering(tmp_path: Path):
    acc_ctx = make_acc_ctx(
        tmp_path,
        account_key="7",
        bureaus_payload={
            "equifax": {
                "account_status": "Open",
                "date_of_last_activity": "--",
                "last_payment": "2024-10-01",
                "date_reported": "2024-11-01",
            }
        },
    )
    run_for_account(acc_ctx)
    data = read_bureau_file(tmp_path, "7", "equifax")
    q2 = data["root_checks"]["Q2"]
    assert q2["observed_activity"] is True
    assert q2["status"] == "ok"
    # last_payment should be the recorded evidence before date_reported
    assert q2["evidence_fields"][0] == "last_payment"


def test_q2_bureau_isolation(tmp_path: Path):
    acc_ctx = make_acc_ctx(
        tmp_path,
        account_key="8",
        bureaus_payload={
            "equifax": {
                "account_status": "Open",
                "date_of_last_activity": "2025-02-01",
            },
            "transunion": {
                "account_status": "Closed",
                "date_of_last_activity": "--",
            },
        },
    )
    run_for_account(acc_ctx)
    eq = read_bureau_file(tmp_path, "8", "equifax")
    tu = read_bureau_file(tmp_path, "8", "transunion")
    assert eq["root_checks"]["Q2"]["status"] == "ok"
    assert tu["root_checks"]["Q2"]["status"] == "ok"
    assert eq["root_checks"]["Q2"]["expected_activity"] is True
    assert tu["root_checks"]["Q2"]["expected_activity"] is False


def test_q2_non_blocking_payload(tmp_path: Path):
    acc_ctx = make_acc_ctx(
        tmp_path,
        account_key="9",
        bureaus_payload={
            "experian": {
                "account_status": "Closed",
                "date_of_last_activity": "--",
                "last_payment": None,
            }
        },
    )
    run_for_account(acc_ctx)
    data = read_bureau_file(tmp_path, "9", "experian")
    assert data["status"] == "ok"
    assert data.get("findings", []) == []
    assert data["gate"]["version"] == "q6_presence_v1"
    assert data["coverage"]["version"] == "coverage_v1"
