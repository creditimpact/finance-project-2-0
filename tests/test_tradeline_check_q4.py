import json
from pathlib import Path

import pytest

from backend.tradeline_check.runner import run_for_account
from backend.validation.pipeline import AccountContext


SID = "sid_q4_test"


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


def test_q4_ok_via_account_type_student_loan(tmp_path: Path):
    acc_ctx = make_acc_ctx(
        tmp_path,
        account_key="1",
        bureaus_payload={
            "equifax": {
                "account_type": "Educational",
                "creditor_type": "Miscellaneous Finance",
            }
        },
    )
    run_for_account(acc_ctx)
    data = read_bureau_file(tmp_path, "1", "equifax")
    q4 = data["root_checks"]["Q4"]
    assert q4["version"] == "q4_type_v1"
    assert q4["status"] == "ok"
    assert q4["declared_type"] == "student_loan"
    assert "ACCOUNT_TYPE:STUDENT_LOAN" in q4["signals"]
    assert q4["confidence"] == 1.0


def test_q4_ok_via_creditor_type_fallback(tmp_path: Path):
    acc_ctx = make_acc_ctx(
        tmp_path,
        account_key="2",
        bureaus_payload={
            "experian": {
                "account_type": "--",
                "creditor_type": "Student Loans",
            }
        },
    )
    run_for_account(acc_ctx)
    data = read_bureau_file(tmp_path, "2", "experian")
    q4 = data["root_checks"]["Q4"]
    assert q4["declared_type"] == "student_loan"
    assert q4["status"] == "ok"
    assert "CREDITOR_TYPE:STUDENT_LOAN" in q4["signals"]


def test_q4_conflict_revolving_vs_student_loan(tmp_path: Path):
    acc_ctx = make_acc_ctx(
        tmp_path,
        account_key="3",
        bureaus_payload={
            "transunion": {
                "account_type": "Revolving",
                "creditor_type": "Student Loans",
            }
        },
    )
    run_for_account(acc_ctx)
    data = read_bureau_file(tmp_path, "3", "transunion")
    q4 = data["root_checks"]["Q4"]
    assert q4["status"] == "conflict"
    assert q4["declared_type"] == "conflict"
    assert "type_mismatch_account_vs_creditor" in q4["conflicts"]
    assert q4["confidence"] == 1.0


def test_q4_unknown_misc_finance(tmp_path: Path):
    acc_ctx = make_acc_ctx(
        tmp_path,
        account_key="4",
        bureaus_payload={
            "equifax": {
                "account_type": "--",
                "creditor_type": "Miscellaneous Finance",
            }
        },
    )
    run_for_account(acc_ctx)
    data = read_bureau_file(tmp_path, "4", "equifax")
    q4 = data["root_checks"]["Q4"]
    assert q4["status"] == "unknown"
    assert q4["declared_type"] == "unknown"
    assert q4["confidence"] == 0.5


def test_q4_skipped_missing_both(tmp_path: Path):
    acc_ctx = make_acc_ctx(
        tmp_path,
        account_key="5",
        bureaus_payload={
            "experian": {
                "account_type": "--",
                "creditor_type": "--",
            }
        },
    )
    run_for_account(acc_ctx)
    data = read_bureau_file(tmp_path, "5", "experian")
    q4 = data["root_checks"]["Q4"]
    assert q4["status"] == "skipped_missing_data"
    assert q4["declared_type"] == "unknown"
    assert q4["confidence"] == 0.0


def test_q4_revolving_brand_rule(tmp_path: Path):
    acc_ctx = make_acc_ctx(
        tmp_path,
        account_key="6",
        bureaus_payload={
            "transunion": {
                "account_type": "--",
                "creditor_type": "Visa",
            }
        },
    )
    run_for_account(acc_ctx)
    data = read_bureau_file(tmp_path, "6", "transunion")
    q4 = data["root_checks"]["Q4"]
    assert q4["status"] == "unknown"

    acc_ctx2 = make_acc_ctx(
        tmp_path,
        account_key="7",
        bureaus_payload={
            "transunion": {
                "account_type": "--",
                "creditor_type": "Visa Credit Card",
            }
        },
    )
    run_for_account(acc_ctx2)
    data2 = read_bureau_file(tmp_path, "7", "transunion")
    q4b = data2["root_checks"]["Q4"]
    assert q4b["status"] == "ok"
    assert q4b["declared_type"] == "revolving"


def test_q4_structure_flags_do_not_change_status(tmp_path: Path):
    acc_ctx = make_acc_ctx(
        tmp_path,
        account_key="8",
        bureaus_payload={
            "equifax": {
                "account_type": "Installment",
                "creditor_type": "Miscellaneous Finance",
                # structural all missing
                "term_length": "--",
                "payment_amount": "--",
                "payment_frequency": "--",
            }
        },
    )
    run_for_account(acc_ctx)
    data = read_bureau_file(tmp_path, "8", "equifax")
    q4 = data["root_checks"]["Q4"]
    assert q4["status"] == "ok"
    assert "installment_missing_terms_and_payment" in q4["structure_flags"]


def test_q4_bureau_isolation(tmp_path: Path):
    acc_ctx = make_acc_ctx(
        tmp_path,
        account_key="9",
        bureaus_payload={
            "equifax": {
                "account_type": "--",
                "creditor_type": "Miscellaneous Finance",
            },
            "experian": {
                "account_type": "Educational",
                "creditor_type": "Student Loans",
            },
            "transunion": {
                "account_type": "Revolving",
                "creditor_type": "Visa Credit Card",
            },
        },
    )
    run_for_account(acc_ctx)
    eq = read_bureau_file(tmp_path, "9", "equifax")
    ex = read_bureau_file(tmp_path, "9", "experian")
    tu = read_bureau_file(tmp_path, "9", "transunion")
    assert eq["root_checks"]["Q4"]["status"] == "unknown"
    assert ex["root_checks"]["Q4"]["declared_type"] == "student_loan"
    assert tu["root_checks"]["Q4"]["declared_type"] == "revolving"


def test_q4_non_blocking_invariants(tmp_path: Path):
    acc_ctx = make_acc_ctx(
        tmp_path,
        account_key="10",
        bureaus_payload={
            "experian": {
                "account_type": "--",
                "creditor_type": "Miscellaneous Finance",
            }
        },
    )
    run_for_account(acc_ctx)
    data = read_bureau_file(tmp_path, "10", "experian")
    assert data["status"] == "ok"
    assert data.get("findings", []) == []
    assert data["coverage"]["version"] == "coverage_v1"
    # Q1, Q2, Q4, Q5 should still exist
    assert "Q1" in data["root_checks"]
    assert "Q2" in data["root_checks"]
    assert "Q4" in data["root_checks"]
