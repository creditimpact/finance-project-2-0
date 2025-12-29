import json
import os
from pathlib import Path

import pytest

from backend.tradeline_check.runner import run_for_account
from backend.validation.pipeline import AccountContext


SID = "sid_q6_test"


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
    monkeypatch.setenv("TRADELINE_CHECK_GATE_STRICT", "0")
    monkeypatch.setenv("TRADELINE_CHECK_PLACEHOLDER_TOKENS", "--,")
    monkeypatch.setenv("TRADELINE_CHECK_WRITE_EMPTY_RESULTS", "0")
    yield


def test_q1_presence_only(tmp_path: Path):
    acc_ctx = make_acc_ctx(
        tmp_path,
        account_key="9",
        bureaus_payload={
            "transunion": {
                "account_status": "Open",
                "account_rating": "--",
                "payment_status": "",
            }
        },
    )
    run_for_account(acc_ctx)
    data = read_bureau_file(tmp_path, "9", "transunion")
    assert data["gate"]["eligible"]["Q1"] is True
    # If all status fields missing -> ineligible and list all
    acc_ctx_all_missing = make_acc_ctx(
        tmp_path,
        account_key="10",
        bureaus_payload={
            "transunion": {
                "account_status": "--",
                "account_rating": "",
                "payment_status": None,
            }
        },
    )
    run_for_account(acc_ctx_all_missing)
    data2 = read_bureau_file(tmp_path, "10", "transunion")
    assert data2["gate"]["eligible"]["Q1"] is False
    assert set(data2["gate"]["missing_fields"]["Q1"]) == {
        "account_status",
        "account_rating",
        "payment_status",
    }


def test_q2_requires_status_and_activity(tmp_path: Path):
    # status present only -> not eligible
    acc_ctx = make_acc_ctx(
        tmp_path,
        account_key="11",
        bureaus_payload={
            "experian": {
                "account_status": "Open",
                "date_of_last_activity": "--",
                "last_payment": "--",
            }
        },
    )
    run_for_account(acc_ctx)
    d = read_bureau_file(tmp_path, "11", "experian")
    assert d["gate"]["eligible"]["Q2"] is False
    assert set(d["gate"]["missing_fields"]["Q2"]) == {"date_of_last_activity", "last_payment"}

    # status present + activity present -> eligible
    acc_ctx2 = make_acc_ctx(
        tmp_path,
        account_key="12",
        bureaus_payload={
            "experian": {
                "account_status": "Closed",
                "last_payment": "2024-01-01",
            }
        },
    )
    run_for_account(acc_ctx2)
    d2 = read_bureau_file(tmp_path, "12", "experian")
    assert d2["gate"]["eligible"]["Q2"] is True


def test_q3_requires_opened_and_secondary(tmp_path: Path):
    # missing date_opened -> ineligible
    acc_ctx = make_acc_ctx(
        tmp_path,
        account_key="13",
        bureaus_payload={
            "equifax": {
                "date_reported": "2023-12-01",
            }
        },
    )
    run_for_account(acc_ctx)
    d = read_bureau_file(tmp_path, "13", "equifax")
    assert d["gate"]["eligible"]["Q3"] is False
    assert "date_opened" in d["gate"]["missing_fields"]["Q3"]

    # has date_opened + secondary -> eligible
    acc_ctx2 = make_acc_ctx(
        tmp_path,
        account_key="14",
        bureaus_payload={
            "equifax": {
                "date_opened": "2020-05-05",
                "last_payment": "2024-02-02",
            }
        },
    )
    run_for_account(acc_ctx2)
    d2 = read_bureau_file(tmp_path, "14", "equifax")
    assert d2["gate"]["eligible"]["Q3"] is True

    # has date_opened but no secondary -> ineligible and list secondaries
    acc_ctx3 = make_acc_ctx(
        tmp_path,
        account_key="15",
        bureaus_payload={
            "equifax": {
                "date_opened": "2019-01-01",
                "date_reported": "--",
                "date_of_last_activity": "",
            }
        },
    )
    run_for_account(acc_ctx3)
    d3 = read_bureau_file(tmp_path, "15", "equifax")
    assert d3["gate"]["eligible"]["Q3"] is False
    assert set(d3["gate"]["missing_fields"]["Q3"]) >= {
        "date_reported",
        "date_of_last_activity",
        "last_payment",
        "closed_date",
    }


def test_q4_type_integrity(tmp_path: Path):
    acc_ctx = make_acc_ctx(
        tmp_path,
        account_key="16",
        bureaus_payload={
            "transunion": {
                "account_type": "revolving",
            }
        },
    )
    run_for_account(acc_ctx)
    d = read_bureau_file(tmp_path, "16", "transunion")
    assert d["gate"]["eligible"]["Q4"] is True

    acc_ctx2 = make_acc_ctx(
        tmp_path,
        account_key="17",
        bureaus_payload={
            "transunion": {
                "account_type": "--",
                "creditor_type": "--",
            }
        },
    )
    run_for_account(acc_ctx2)
    d2 = read_bureau_file(tmp_path, "17", "transunion")
    assert d2["gate"]["eligible"]["Q4"] is False
    assert set(d2["gate"]["missing_fields"]["Q4"]) == {"account_type", "creditor_type"}


def test_placeholder_config(tmp_path: Path, monkeypatch):
    # Treat N/A as missing when configured
    monkeypatch.setenv("TRADELINE_CHECK_PLACEHOLDER_TOKENS", "--,N/A,UNKNOWN")
    acc_ctx = make_acc_ctx(
        tmp_path,
        account_key="18",
        bureaus_payload={
            "experian": {
                "account_status": "N/A",
                "payment_status": "UNKNOWN",
                "account_rating": "--",
            }
        },
    )
    run_for_account(acc_ctx)
    d = read_bureau_file(tmp_path, "18", "experian")
    assert d["gate"]["eligible"]["Q1"] is False


def test_cross_bureau_isolation(tmp_path: Path):
    # TU has status, EX missing -> outputs should differ per bureau
    acc_ctx = make_acc_ctx(
        tmp_path,
        account_key="19",
        bureaus_payload={
            "transunion": {"account_status": "Open"},
            "experian": {"account_status": "--"},
        },
    )
    run_for_account(acc_ctx)
    tu = read_bureau_file(tmp_path, "19", "transunion")
    ex = read_bureau_file(tmp_path, "19", "experian")
    assert tu["gate"]["eligible"]["Q1"] is True
    assert ex["gate"]["eligible"]["Q1"] is False


def test_strict_flag_blocks(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("TRADELINE_CHECK_GATE_STRICT", "1")
    acc_ctx = make_acc_ctx(
        tmp_path,
        account_key="20",
        bureaus_payload={
            "equifax": {
                # Missing date_opened -> Q3 fails
                "date_reported": "2024-08-01",
            }
        },
    )
    run_for_account(acc_ctx)
    d = read_bureau_file(tmp_path, "20", "equifax")
    assert d["status"] == "blocked"
    assert "Q3" in d.get("blocked_questions", [])

# ════════════════════════════════════════════════════════════════════════════
# COVERAGE TESTS (non-blocking, capability awareness)
# ════════════════════════════════════════════════════════════════════════════


def test_coverage_block_exists_and_schema(tmp_path: Path):
    """Verify coverage block exists with correct schema keys."""
    acc_ctx = make_acc_ctx(
        tmp_path,
        account_key="30",
        bureaus_payload={
            "equifax": {
                "account_status": "Open",
            }
        },
    )
    run_for_account(acc_ctx)
    d = read_bureau_file(tmp_path, "30", "equifax")
    
    assert "coverage" in d
    coverage = d["coverage"]
    assert coverage["version"] == "coverage_v1"
    assert "placeholders" in coverage
    assert "missing_core_fields" in coverage
    assert "missing_branch_fields" in coverage
    
    # Should have keys for Q1-Q5
    for q in ("Q1", "Q2", "Q3", "Q4", "Q5"):
        assert q in coverage["missing_core_fields"]
        assert q in coverage["missing_branch_fields"]


def test_coverage_q1_core_fields(tmp_path: Path):
    """Coverage Q1 core: account_status, account_rating, payment_status."""
    acc_ctx = make_acc_ctx(
        tmp_path,
        account_key="31",
        bureaus_payload={
            "equifax": {
                "account_status": "Open",
                # account_rating missing
                # payment_status missing
            }
        },
    )
    run_for_account(acc_ctx)
    d = read_bureau_file(tmp_path, "31", "equifax")
    
    coverage = d["coverage"]
    assert "account_rating" in coverage["missing_core_fields"]["Q1"]
    assert "payment_status" in coverage["missing_core_fields"]["Q1"]
    assert "account_status" not in coverage["missing_core_fields"]["Q1"]


def test_coverage_q1_branch_fields(tmp_path: Path):
    """Coverage Q1 branch: dispute_status, creditor_remarks, date_reported, date_of_last_activity."""
    acc_ctx = make_acc_ctx(
        tmp_path,
        account_key="32",
        bureaus_payload={
            "equifax": {
                "account_status": "Open",
                "date_reported": "2024-01-01",
                # dispute_status, creditor_remarks, date_of_last_activity missing
            }
        },
    )
    run_for_account(acc_ctx)
    d = read_bureau_file(tmp_path, "32", "equifax")
    
    coverage = d["coverage"]
    assert "dispute_status" in coverage["missing_branch_fields"]["Q1"]
    assert "creditor_remarks" in coverage["missing_branch_fields"]["Q1"]
    assert "date_of_last_activity" in coverage["missing_branch_fields"]["Q1"]
    assert "date_reported" not in coverage["missing_branch_fields"]["Q1"]


def test_coverage_q2_core_fields(tmp_path: Path):
    """Coverage Q2 core: date_of_last_activity, last_payment, date_reported, closed_date."""
    acc_ctx = make_acc_ctx(
        tmp_path,
        account_key="33",
        bureaus_payload={
            "experian": {
                "account_status": "Open",
                "date_of_last_activity": "2024-06-01",
                "last_payment": "2024-06-15",
                # date_reported, closed_date missing
            }
        },
    )
    run_for_account(acc_ctx)
    d = read_bureau_file(tmp_path, "33", "experian")
    
    coverage = d["coverage"]
    assert "date_reported" in coverage["missing_core_fields"]["Q2"]
    assert "closed_date" in coverage["missing_core_fields"]["Q2"]
    assert "date_of_last_activity" not in coverage["missing_core_fields"]["Q2"]
    assert "last_payment" not in coverage["missing_core_fields"]["Q2"]


def test_coverage_q2_branch_fields_and_history(tmp_path: Path):
    """Coverage Q2 branch: payment_amount, past_due_amount, balance_owed, history blocks."""
    acc_ctx = make_acc_ctx(
        tmp_path,
        account_key="34",
        bureaus_payload={
            "transunion": {
                "account_status": "Open",
                "date_of_last_activity": "2024-06-01",
                "balance_owed": "1500.00",
                # payment_amount, past_due_amount missing
                # two_year_payment_history missing (as a block entry)
                # seven_year_history missing (as a block entry)
            }
        },
    )
    run_for_account(acc_ctx)
    d = read_bureau_file(tmp_path, "34", "transunion")
    
    coverage = d["coverage"]
    assert "payment_amount" in coverage["missing_branch_fields"]["Q2"]
    assert "past_due_amount" in coverage["missing_branch_fields"]["Q2"]
    assert "balance_owed" not in coverage["missing_branch_fields"]["Q2"]
    assert "two_year_payment_history" in coverage["missing_branch_fields"]["Q2"]
    assert "seven_year_history" in coverage["missing_branch_fields"]["Q2"]


def test_coverage_q2_with_history_blocks(tmp_path: Path):
    """Coverage Q2 branch when history blocks are present for a bureau."""
    acc_ctx = make_acc_ctx(
        tmp_path,
        account_key="35",
        bureaus_payload={
            "equifax": {
                "account_status": "Open",
                "date_of_last_activity": "2024-06-01",
            },
            "two_year_payment_history": {
                "equifax": "24 months of payment history",
                # transunion missing
            },
            "seven_year_history": {
                "equifax": "7 year historical data",
                # transunion missing
            },
        },
    )
    run_for_account(acc_ctx)
    d = read_bureau_file(tmp_path, "35", "equifax")
    
    coverage = d["coverage"]
    # equifax has both history blocks
    assert "two_year_payment_history" not in coverage["missing_branch_fields"]["Q2"]
    assert "seven_year_history" not in coverage["missing_branch_fields"]["Q2"]


def test_coverage_history_isolation_per_bureau(tmp_path: Path):
    """History blocks are bureau-scoped: missing in TU but present in EQ yields different coverage."""
    acc_ctx = make_acc_ctx(
        tmp_path,
        account_key="36",
        bureaus_payload={
            "transunion": {
                "account_status": "Open",
                "date_of_last_activity": "2024-06-01",
            },
            "equifax": {
                "account_status": "Open",
                "date_of_last_activity": "2024-06-01",
            },
            "two_year_payment_history": {
                "equifax": "EQ has 24 months",
                # TU missing
            },
        },
    )
    run_for_account(acc_ctx)
    tu = read_bureau_file(tmp_path, "36", "transunion")
    eq = read_bureau_file(tmp_path, "36", "equifax")
    
    # TU missing the history, EQ has it
    assert "two_year_payment_history" in tu["coverage"]["missing_branch_fields"]["Q2"]
    assert "two_year_payment_history" not in eq["coverage"]["missing_branch_fields"]["Q2"]


def test_coverage_q3_core_fields(tmp_path: Path):
    """Coverage Q3 core: date_opened, date_reported, date_of_last_activity, last_payment, closed_date."""
    acc_ctx = make_acc_ctx(
        tmp_path,
        account_key="37",
        bureaus_payload={
            "experian": {
                "account_status": "Open",
                "date_opened": "2020-05-05",
                "last_payment": "2024-06-01",
                # date_reported, date_of_last_activity, closed_date missing
            }
        },
    )
    run_for_account(acc_ctx)
    d = read_bureau_file(tmp_path, "37", "experian")
    
    coverage = d["coverage"]
    assert "date_reported" in coverage["missing_core_fields"]["Q3"]
    assert "date_of_last_activity" in coverage["missing_core_fields"]["Q3"]
    assert "closed_date" in coverage["missing_core_fields"]["Q3"]
    assert "date_opened" not in coverage["missing_core_fields"]["Q3"]
    assert "last_payment" not in coverage["missing_core_fields"]["Q3"]


def test_coverage_q3_branch_history(tmp_path: Path):
    """Coverage Q3 branch: two_year_payment_history, seven_year_history."""
    acc_ctx = make_acc_ctx(
        tmp_path,
        account_key="38",
        bureaus_payload={
            "transunion": {
                "account_status": "Open",
                "date_opened": "2020-05-05",
                "last_payment": "2024-06-01",
            },
            "seven_year_history": {
                "transunion": "7 years of data",
                # two_year missing
            },
        },
    )
    run_for_account(acc_ctx)
    d = read_bureau_file(tmp_path, "38", "transunion")
    
    coverage = d["coverage"]
    assert "two_year_payment_history" in coverage["missing_branch_fields"]["Q3"]
    assert "seven_year_history" not in coverage["missing_branch_fields"]["Q3"]


def test_coverage_q4_core_fields(tmp_path: Path):
    """Coverage Q4 core: account_type, creditor_type, term_length, payment_frequency."""
    acc_ctx = make_acc_ctx(
        tmp_path,
        account_key="39",
        bureaus_payload={
            "equifax": {
                "account_status": "Open",
                "account_type": "revolving",
                # creditor_type, term_length, payment_frequency missing
            }
        },
    )
    run_for_account(acc_ctx)
    d = read_bureau_file(tmp_path, "39", "equifax")
    
    coverage = d["coverage"]
    assert "creditor_type" in coverage["missing_core_fields"]["Q4"]
    assert "term_length" in coverage["missing_core_fields"]["Q4"]
    assert "payment_frequency" in coverage["missing_core_fields"]["Q4"]
    assert "account_type" not in coverage["missing_core_fields"]["Q4"]


def test_coverage_q4_branch_fields(tmp_path: Path):
    """Coverage Q4 branch: credit_limit, high_balance, payment_amount, original_creditor."""
    acc_ctx = make_acc_ctx(
        tmp_path,
        account_key="40",
        bureaus_payload={
            "experian": {
                "account_status": "Open",
                "account_type": "revolving",
                "credit_limit": "5000.00",
                # high_balance, payment_amount, original_creditor missing
            }
        },
    )
    run_for_account(acc_ctx)
    d = read_bureau_file(tmp_path, "40", "experian")
    
    coverage = d["coverage"]
    assert "high_balance" in coverage["missing_branch_fields"]["Q4"]
    assert "payment_amount" in coverage["missing_branch_fields"]["Q4"]
    assert "original_creditor" in coverage["missing_branch_fields"]["Q4"]
    assert "credit_limit" not in coverage["missing_branch_fields"]["Q4"]


def test_coverage_placeholder_behavior(tmp_path: Path, monkeypatch):
    """Placeholder tokens are treated as missing in coverage."""
    monkeypatch.setenv("TRADELINE_CHECK_PLACEHOLDER_TOKENS", "--,N/A,UNKNOWN")
    acc_ctx = make_acc_ctx(
        tmp_path,
        account_key="41",
        bureaus_payload={
            "transunion": {
                "account_status": "N/A",
                "account_rating": "--",
                "payment_status": "UNKNOWN",
            }
        },
    )
    run_for_account(acc_ctx)
    d = read_bureau_file(tmp_path, "41", "transunion")
    
    coverage = d["coverage"]
    # All three placeholders should be reported in coverage.placeholders
    assert set(coverage["placeholders"]) >= {"--", "n/a", "unknown"}
    # All three should be in missing_core_fields["Q1"]
    assert set(coverage["missing_core_fields"]["Q1"]) == {
        "account_status",
        "account_rating",
        "payment_status",
    }


def test_coverage_does_not_affect_gate_eligibility(tmp_path: Path):
    """Coverage is non-blocking: missing branch fields do NOT affect gate.eligible."""
    acc_ctx = make_acc_ctx(
        tmp_path,
        account_key="42",
        bureaus_payload={
            "equifax": {
                "account_status": "Open",
                "account_rating": "Good",
                "payment_status": "Current",
                # Branch fields missing, but should not affect Q1 eligibility
            }
        },
    )
    run_for_account(acc_ctx)
    d = read_bureau_file(tmp_path, "42", "equifax")
    
    # Q1 should be eligible (core present)
    assert d["gate"]["eligible"]["Q1"] is True
    
    # But coverage shows missing branches
    assert len(d["coverage"]["missing_branch_fields"]["Q1"]) > 0


def test_coverage_does_not_affect_status(tmp_path: Path):
    """Coverage is non-blocking: missing fields do NOT change status to 'blocked'."""
    acc_ctx = make_acc_ctx(
        tmp_path,
        account_key="43",
        bureaus_payload={
            "experian": {
                "account_status": "Open",
                # Missing many core and branch fields
            }
        },
    )
    run_for_account(acc_ctx)
    d = read_bureau_file(tmp_path, "43", "experian")
    
    # Status should still be "ok" even with missing coverage
    assert d["status"] == "ok"


def test_coverage_with_strict_mode_independent(tmp_path: Path, monkeypatch):
    """Coverage presence is independent of strict mode; strict only affects gate.eligible blocking."""
    monkeypatch.setenv("TRADELINE_CHECK_GATE_STRICT", "1")
    acc_ctx = make_acc_ctx(
        tmp_path,
        account_key="44",
        bureaus_payload={
            "equifax": {
                # Missing many core fields -> Q1, Q3, Q4 will fail
                "account_status": "--",
                "date_reported": "2024-01-01",
            }
        },
    )
    run_for_account(acc_ctx)
    d = read_bureau_file(tmp_path, "44", "equifax")
    
    # Strict mode should block due to Q1, Q3, Q4 failures
    assert d["status"] == "blocked"
    assert "Q1" in d.get("blocked_questions", [])
    
    # But coverage block should still exist and show missing fields
    assert "coverage" in d
    assert len(d["coverage"]["missing_core_fields"]["Q1"]) > 0


def test_coverage_q5_empty(tmp_path: Path):
    """Q5 coverage is not yet measured; missing_core_fields and missing_branch_fields should be empty."""
    acc_ctx = make_acc_ctx(
        tmp_path,
        account_key="45",
        bureaus_payload={
            "transunion": {
                "account_status": "Open",
            }
        },
    )
    run_for_account(acc_ctx)
    d = read_bureau_file(tmp_path, "45", "transunion")
    
    coverage = d["coverage"]
    assert coverage["missing_core_fields"]["Q5"] == []
    assert coverage["missing_branch_fields"]["Q5"] == []