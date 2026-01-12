import json
from pathlib import Path

import pytest

from backend.tradeline_check.runner import run_for_account
from backend.validation.pipeline import AccountContext


SID = "sid_f0_a01_test"


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


def test_f0_a01_ok_ceiling_covers_all(tmp_path: Path):
    acc_ctx = make_acc_ctx(
        tmp_path,
        account_key="1",
        bureaus_payload={
            "equifax": {
                "account_status": "Open",
                "date_reported": "2024-01-15",
                "last_payment": "2023-12-01",
                "date_of_last_activity": "2023-11-01",
            }
        },
    )

    run_for_account(acc_ctx)
    data = read_bureau_file(tmp_path, "1", "equifax")

    f0_a01 = data["record_integrity"]["F0"]["A01"]
    assert f0_a01["status"] == "ok"
    assert f0_a01["ceiling"]["conflict"] is False
    assert f0_a01["ceiling"]["effective_ceiling_date"] == "2024-01-15"
    assert "date_reported" in f0_a01["ceiling"]["ceiling_sources"]
    assert f0_a01["ceiling"]["violations"] == []


def test_f0_a01_conflict_event_after_ceiling(tmp_path: Path):
    acc_ctx = make_acc_ctx(
        tmp_path,
        account_key="2",
        bureaus_payload={
            "experian": {
                "account_status": "Open",
                "date_reported": "2024-01-15",
                "last_payment": "2024-02-01",
            }
        },
    )

    run_for_account(acc_ctx)
    data = read_bureau_file(tmp_path, "2", "experian")

    f0_a01 = data["record_integrity"]["F0"]["A01"]
    assert f0_a01["status"] == "conflict"
    assert f0_a01["ceiling"]["conflict"] is True
    violations = f0_a01["ceiling"]["violations"]
    assert any(v.get("field") == "last_payment" for v in violations)


def test_f0_a01_unknown_when_no_ceiling_candidates(tmp_path: Path):
    acc_ctx = make_acc_ctx(
        tmp_path,
        account_key="3",
        bureaus_payload={
            "transunion": {
                "account_status": "Open",
                "date_reported": "--",
                "last_verified": None,
                "last_payment": "2024-01-01",
            }
        },
    )

    run_for_account(acc_ctx)
    data = read_bureau_file(tmp_path, "3", "transunion")

    f0_a01 = data["record_integrity"]["F0"]["A01"]
    assert f0_a01["status"] == "unknown"
    assert f0_a01["ceiling"]["effective_ceiling_date"] is None
    assert f0_a01["ceiling"]["conflict"] is False


def test_f0_a01_runs_for_closed_accounts(tmp_path: Path):
    acc_ctx = make_acc_ctx(
        tmp_path,
        account_key="4",
        bureaus_payload={
            "equifax": {
                "account_status": "Closed",
                "date_reported": "2024-01-10",
                "closed_date": "2023-12-31",
            }
        },
    )

    run_for_account(acc_ctx)
    data = read_bureau_file(tmp_path, "4", "equifax")

    f0_a01 = data["record_integrity"]["F0"]["A01"]
    assert f0_a01["status"] == "ok"
    assert f0_a01["ceiling"]["conflict"] is False
    assert f0_a01["ceiling"]["effective_ceiling_date"] == "2024-01-10"
