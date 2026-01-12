import json
import os
from pathlib import Path

import pytest

from backend.tradeline_check.runner import run_for_account
from backend.validation.pipeline import AccountContext


SID = "sid_q1_test"


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
    # Ensure deterministic flags for tests
    monkeypatch.setenv("TRADELINE_CHECK_ENABLED", "1")
    monkeypatch.setenv("TRADELINE_CHECK_WRITE_DEBUG", "1")
    monkeypatch.setenv("TRADELINE_CHECK_PLACEHOLDER_TOKENS", "--,n/a,unknown")
    monkeypatch.setenv("TRADELINE_CHECK_WRITE_EMPTY_RESULTS", "0")
    yield


def test_q1_open_detection(tmp_path: Path):
    acc_ctx = make_acc_ctx(
        tmp_path,
        account_key="1",
        bureaus_payload={
            "equifax": {
                "account_status": "Open",
            }
        },
    )
    run_for_account(acc_ctx)
    data = read_bureau_file(tmp_path, "1", "equifax")
    q1 = data["root_checks"]["Q1"]
    assert q1["version"] == "q1_state_v1"
    assert q1["declared_state"] == "open"
    assert q1["status"] == "ok"
    assert "OPEN" in q1["signals"]
    assert "account_status" in q1["contributing_fields"]
    assert q1["confidence"] > 0


def test_q1_closed_detection(tmp_path: Path):
    acc_ctx = make_acc_ctx(
        tmp_path,
        account_key="2",
        bureaus_payload={
            "experian": {
                "account_status": "Closed",
            }
        },
    )
    run_for_account(acc_ctx)
    data = read_bureau_file(tmp_path, "2", "experian")
    q1 = data["root_checks"]["Q1"]
    assert q1["declared_state"] == "closed"
    assert q1["status"] == "ok"
    assert "CLOSED" in q1["signals"]


def test_q1_derog_signal_does_not_override_open(tmp_path: Path):
    acc_ctx = make_acc_ctx(
        tmp_path,
        account_key="3",
        bureaus_payload={
            "transunion": {
                "account_status": "Open",
                "payment_status": "Late 120 Days",
            }
        },
    )
    run_for_account(acc_ctx)
    data = read_bureau_file(tmp_path, "3", "transunion")
    q1 = data["root_checks"]["Q1"]
    assert q1["declared_state"] == "open"
    assert q1["status"] == "ok"
    assert set(q1["signals"]) >= {"OPEN", "DEROG"}


def test_q1_conflict_detection(tmp_path: Path):
    acc_ctx = make_acc_ctx(
        tmp_path,
        account_key="4",
        bureaus_payload={
            "equifax": {
                "account_status": "Open",
                "account_rating": "Closed",
            }
        },
    )
    run_for_account(acc_ctx)
    data = read_bureau_file(tmp_path, "4", "equifax")
    q1 = data["root_checks"]["Q1"]
    assert q1["declared_state"] == "conflict"
    assert q1["status"] == "conflict"
    assert set(q1["signals"]) >= {"OPEN", "CLOSED"}


def test_q1_all_fields_missing_skipped(tmp_path: Path):
    acc_ctx = make_acc_ctx(
        tmp_path,
        account_key="5",
        bureaus_payload={
            "experian": {
                "account_status": "--",
                "account_rating": "",
                "payment_status": None,
            }
        },
    )
    run_for_account(acc_ctx)
    data = read_bureau_file(tmp_path, "5", "experian")
    q1 = data["root_checks"]["Q1"]
    assert q1["declared_state"] == "unknown"
    assert q1["status"] == "skipped_missing_data"
    assert q1["confidence"] == 0.0


def test_q1_placeholder_config(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("TRADELINE_CHECK_PLACEHOLDER_TOKENS", "--,N/A,UNKNOWN")
    acc_ctx = make_acc_ctx(
        tmp_path,
        account_key="6",
        bureaus_payload={
            "transunion": {
                "account_status": "N/A",
            }
        },
    )
    run_for_account(acc_ctx)
    data = read_bureau_file(tmp_path, "6", "transunion")
    q1 = data["root_checks"]["Q1"]
    # No signals, treated as missing due to placeholder
    assert q1["declared_state"] == "unknown"
    assert q1["status"] in {"ok", "skipped_missing_data"}


def test_q1_bureau_isolation(tmp_path: Path):
    acc_ctx = make_acc_ctx(
        tmp_path,
        account_key="7",
        bureaus_payload={
            "equifax": {"account_status": "Open"},
            "transunion": {"account_status": "Closed"},
        },
    )
    run_for_account(acc_ctx)
    eq = read_bureau_file(tmp_path, "7", "equifax")
    tu = read_bureau_file(tmp_path, "7", "transunion")
    assert eq["root_checks"]["Q1"]["declared_state"] == "open"
    assert tu["root_checks"]["Q1"]["declared_state"] == "closed"


def test_q1_non_blocking_payload(tmp_path: Path):
    acc_ctx = make_acc_ctx(
        tmp_path,
        account_key="8",
        bureaus_payload={
            "experian": {
                "account_status": "Open",
                "payment_status": "Late",
            }
        },
    )
    run_for_account(acc_ctx)
    data = read_bureau_file(tmp_path, "8", "experian")
    # payload-level status remains ok; no findings added; coverage present
    assert data["status"] == "ok"
    assert data.get("findings", []) == []
    assert data["coverage"]["version"] == "coverage_v1"
