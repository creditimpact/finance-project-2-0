import json
from pathlib import Path

import pytest

from backend.tradeline_check.runner import run_for_account
from backend.validation.pipeline import AccountContext


SID = "sid_q3_test"


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


def test_q3_coherent_timeline_ok(tmp_path: Path):
    acc_ctx = make_acc_ctx(
        tmp_path,
        account_key="1",
        bureaus_payload={
            "equifax": {
                "date_opened": "2020-01-01",
                "date_reported": "2020-02-01",
                "date_of_last_activity": "2020-03-01",
                "last_payment": "2020-03-15",
            }
        },
    )
    run_for_account(acc_ctx)
    data = read_bureau_file(tmp_path, "1", "equifax")
    q3 = data["root_checks"]["Q3"]
    assert q3["version"] == "q3_timeline_v1"
    assert q3["status"] == "ok"
    assert q3["declared_timeline"] == "coherent"
    assert q3["conflicts"] == []
    assert q3["confidence"] == 1.0


def test_q3_report_before_open_conflict(tmp_path: Path):
    acc_ctx = make_acc_ctx(
        tmp_path,
        account_key="2",
        bureaus_payload={
            "experian": {
                "date_opened": "2020-05-01",
                "date_reported": "2020-04-01",
            }
        },
    )
    run_for_account(acc_ctx)
    data = read_bureau_file(tmp_path, "2", "experian")
    q3 = data["root_checks"]["Q3"]
    assert "report_before_open" in q3["conflicts"]
    assert q3["status"] == "conflict"
    assert q3["declared_timeline"] == "conflict"


def test_q3_activity_after_close_conflict(tmp_path: Path):
    acc_ctx = make_acc_ctx(
        tmp_path,
        account_key="3",
        bureaus_payload={
            "transunion": {
                "date_opened": "2020-01-01",
                "closed_date": "2020-03-01",
                "date_of_last_activity": "2020-04-01",
            }
        },
    )
    run_for_account(acc_ctx)
    data = read_bureau_file(tmp_path, "3", "transunion")
    q3 = data["root_checks"]["Q3"]
    assert "activity_after_close" in q3["conflicts"]
    assert q3["status"] == "conflict"


def test_q3_multiple_conflicts(tmp_path: Path):
    acc_ctx = make_acc_ctx(
        tmp_path,
        account_key="4",
        bureaus_payload={
            "equifax": {
                "date_opened": "2020-03-01",
                "date_reported": "2020-02-01",
                "last_payment": "2020-04-01",
                "date_of_last_activity": "2020-04-15",
                "closed_date": "2020-03-15",
            }
        },
    )
    run_for_account(acc_ctx)
    data = read_bureau_file(tmp_path, "4", "equifax")
    q3 = data["root_checks"]["Q3"]
    assert set(q3["conflicts"]) >= {
        "report_before_open",
        "payment_after_close",
        "activity_after_close",
    }
    assert q3["status"] == "conflict"


def test_q3_requires_opened_and_secondary(tmp_path: Path):
    acc_ctx = make_acc_ctx(
        tmp_path,
        account_key="5",
        bureaus_payload={
            "experian": {
                "date_opened": "2020-01-01",
                "date_reported": "--",
                "date_of_last_activity": None,
                "last_payment": "",
                "closed_date": "--",
            }
        },
    )
    run_for_account(acc_ctx)
    data = read_bureau_file(tmp_path, "5", "experian")
    q3 = data["root_checks"]["Q3"]
    assert q3["status"] == "skipped_missing_data"
    assert q3["conflicts"] == []
    assert q3["confidence"] == 0.0


def test_q3_unparseable_dates_unknown(tmp_path: Path):
    acc_ctx = make_acc_ctx(
        tmp_path,
        account_key="6",
        bureaus_payload={
            "transunion": {
                "date_opened": "not-a-date",
                "date_reported": "2024-01-01",
                "last_payment": "2024-02-02",
            }
        },
    )
    run_for_account(acc_ctx)
    data = read_bureau_file(tmp_path, "6", "transunion")
    q3 = data["root_checks"]["Q3"]
    assert q3["status"] == "unknown"
    assert q3["conflicts"] == []
    assert q3["confidence"] == 0.5


def test_q3_placeholders_respected(tmp_path: Path):
    acc_ctx = make_acc_ctx(
        tmp_path,
        account_key="7",
        bureaus_payload={
            "equifax": {
                "date_opened": "--",
                "date_reported": "unknown",
                "last_payment": "",
            }
        },
    )
    run_for_account(acc_ctx)
    data = read_bureau_file(tmp_path, "7", "equifax")
    q3 = data["root_checks"]["Q3"]
    assert q3["status"] == "skipped_missing_data"
    assert q3["evidence_fields"] == []


def test_q3_last_verified_context_only(tmp_path: Path):
    acc_ctx = make_acc_ctx(
        tmp_path,
        account_key="8",
        bureaus_payload={
            "experian": {
                "date_opened": "2020-01-01",
                "last_payment": "2020-02-01",
                "last_verified": "2019-12-15",
            }
        },
    )
    run_for_account(acc_ctx)
    data = read_bureau_file(tmp_path, "8", "experian")
    q3 = data["root_checks"]["Q3"]
    assert q3["status"] == "ok"
    assert "last_verified" in q3["evidence"]
    assert "last_verified" in q3["evidence_fields"]
    assert q3["conflicts"] == []


def test_q3_bureau_isolation(tmp_path: Path):
    acc_ctx = make_acc_ctx(
        tmp_path,
        account_key="9",
        bureaus_payload={
            "equifax": {
                "date_opened": "2020-03-01",
                "date_reported": "2020-02-01",
            },
            "transunion": {
                "date_opened": "2020-01-01",
                "date_reported": "2020-02-01",
            },
        },
    )
    run_for_account(acc_ctx)
    eq = read_bureau_file(tmp_path, "9", "equifax")
    tu = read_bureau_file(tmp_path, "9", "transunion")
    assert eq["root_checks"]["Q3"]["status"] == "conflict"
    assert tu["root_checks"]["Q3"]["status"] == "ok"


def test_q3_non_blocking_payload(tmp_path: Path):
    acc_ctx = make_acc_ctx(
        tmp_path,
        account_key="10",
        bureaus_payload={
            "experian": {
                "date_opened": "2020-01-01",
                "date_of_last_activity": "2020-02-02",
            }
        },
    )
    run_for_account(acc_ctx)
    data = read_bureau_file(tmp_path, "10", "experian")
    assert data["status"] == "ok"
    assert data.get("findings", []) == []
    assert data["gate"]["version"] == "q6_presence_v1"
    assert data["coverage"]["version"] == "coverage_v1"
