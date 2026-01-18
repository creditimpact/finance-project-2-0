"""Tests for FX.B03 — Last Payment vs Monthly Coverage."""
import pytest

from backend.tradeline_check.fx_b03_last_payment_vs_monthly_coverage import evaluate_fx_b03

PLACEHOLDERS = {"", "--", "n/a", "unknown"}


def _base_bureau(last_payment: str = "2024-01-15"):
    return {"last_payment": last_payment}


def _base_bureaus_data(entries):
    return {"two_year_payment_history_monthly_tsv_v2": {"equifax": entries}}


# ── Missing/Unparseable Anchor Tests ──────────────────────────────────────────

def test_skipped_when_last_payment_missing():
    """last_payment missing => skipped_missing_data."""
    result = evaluate_fx_b03({}, {}, "equifax", PLACEHOLDERS)
    assert result["status"] == "skipped_missing_data"
    assert result["fired"] is False
    assert result["eligible"] is True
    assert result["ungated"] is True


def test_skipped_when_last_payment_placeholder():
    """last_payment placeholder => skipped_missing_data."""
    result = evaluate_fx_b03({"last_payment": "n/a"}, {}, "equifax", PLACEHOLDERS)
    assert result["status"] == "skipped_missing_data"
    assert "missing or placeholder" in result["explanation"]


def test_skipped_when_last_payment_unparseable():
    """last_payment unparseable => skipped_missing_data."""
    result = evaluate_fx_b03({"last_payment": "not-a-date"}, {}, "equifax", PLACEHOLDERS)
    assert result["status"] == "skipped_missing_data"
    assert "unparseable" in result["explanation"]


# ── Missing Monthly History Tests ─────────────────────────────────────────────

def test_skipped_when_monthly_history_missing():
    """No monthly history block => skipped_missing_data."""
    bureau_obj = _base_bureau()
    result = evaluate_fx_b03(bureau_obj, {}, "equifax", PLACEHOLDERS)
    assert result["status"] == "skipped_missing_data"
    assert "missing or empty" in result["explanation"]


def test_skipped_when_monthly_history_empty():
    """Empty monthly list => skipped_missing_data."""
    bureau_obj = _base_bureau()
    bureaus_data = _base_bureaus_data([])
    result = evaluate_fx_b03(bureau_obj, bureaus_data, "equifax", PLACEHOLDERS)
    assert result["status"] == "skipped_missing_data"


def test_skipped_when_no_parseable_month_keys():
    """Monthly history present but no parseable month_year_key => skipped_missing_data."""
    bureau_obj = _base_bureau()
    bureaus_data = _base_bureaus_data([
        {"month_year_key": "invalid", "status": "ok"},
        {"month_year_key": None, "status": "30"},
    ])
    result = evaluate_fx_b03(bureau_obj, bureaus_data, "equifax", PLACEHOLDERS)
    assert result["status"] == "skipped_missing_data"
    assert "no parseable month_year_key" in result["explanation"]


# ── Conflict Detection Tests ──────────────────────────────────────────────────

def test_conflict_when_last_payment_after_max_month():
    """last_payment month > max monthly coverage month => conflict."""
    bureau_obj = _base_bureau("2024-05-15")
    bureaus_data = _base_bureaus_data([
        {"month_year_key": "2024-01", "status": "ok"},
        {"month_year_key": "2024-02", "status": "30"},
        {"month_year_key": "2024-03", "status": "--"},
    ])
    result = evaluate_fx_b03(bureau_obj, bureaus_data, "equifax", PLACEHOLDERS)
    assert result["status"] == "conflict"
    assert result["fired"] is True
    assert result["evidence"]["last_payment_month_key"] == "2024-05"
    assert result["evidence"]["max_month_key"] == "2024-03"
    assert "exceeds max monthly coverage" in result["explanation"]


def test_conflict_when_last_payment_far_future():
    """last_payment significantly beyond coverage => conflict."""
    bureau_obj = _base_bureau("2025-12-31")
    bureaus_data = _base_bureaus_data([
        {"month_year_key": "2024-06", "status": "ok"},
    ])
    result = evaluate_fx_b03(bureau_obj, bureaus_data, "equifax", PLACEHOLDERS)
    assert result["status"] == "conflict"
    assert result["fired"] is True


# ── OK (No Conflict) Tests ────────────────────────────────────────────────────

def test_ok_when_last_payment_within_coverage():
    """last_payment month <= max monthly coverage => ok."""
    bureau_obj = _base_bureau("2024-02-15")
    bureaus_data = _base_bureaus_data([
        {"month_year_key": "2024-01", "status": "ok"},
        {"month_year_key": "2024-02", "status": "30"},
        {"month_year_key": "2024-03", "status": "--"},
    ])
    result = evaluate_fx_b03(bureau_obj, bureaus_data, "equifax", PLACEHOLDERS)
    assert result["status"] == "ok"
    assert result["fired"] is False
    assert result["evidence"]["last_payment_month_key"] == "2024-02"
    assert result["evidence"]["max_month_key"] == "2024-03"


def test_ok_when_last_payment_equals_max_month():
    """last_payment month == max monthly coverage => ok (exact boundary)."""
    bureau_obj = _base_bureau("2024-03-01")
    bureaus_data = _base_bureaus_data([
        {"month_year_key": "2024-03", "status": "ok"},
    ])
    result = evaluate_fx_b03(bureau_obj, bureaus_data, "equifax", PLACEHOLDERS)
    assert result["status"] == "ok"
    assert result["fired"] is False


def test_ok_when_last_payment_before_max_month():
    """last_payment month < max monthly coverage => ok."""
    bureau_obj = _base_bureau("2023-12-15")
    bureaus_data = _base_bureaus_data([
        {"month_year_key": "2024-01", "status": "ok"},
        {"month_year_key": "2024-02", "status": "30"},
    ])
    result = evaluate_fx_b03(bureau_obj, bureaus_data, "equifax", PLACEHOLDERS)
    assert result["status"] == "ok"
    assert result["fired"] is False


# ── Evidence Field Tests ──────────────────────────────────────────────────────

def test_evidence_fields_populated():
    """Verify all evidence fields are present and correct."""
    bureau_obj = _base_bureau("2024-02-10")
    bureaus_data = _base_bureaus_data([
        {"month_year_key": "2024-01", "status": "ok"},
        {"month_year_key": "2024-02", "status": "30"},
        {"month_year_key": "invalid", "status": "60"},  # Should be ignored
    ])
    result = evaluate_fx_b03(bureau_obj, bureaus_data, "equifax", PLACEHOLDERS)
    
    assert result["evidence"]["last_payment_raw"] == "2024-02-10"
    assert result["evidence"]["last_payment_month_key"] == "2024-02"
    assert result["evidence"]["max_month_key"] == "2024-02"
    assert result["evidence"]["monthly_entries_total"] == 3
    assert result["evidence"]["monthly_entries_parseable_count"] == 2


def test_evidence_monthly_counts_correct():
    """Verify monthly entry counts are accurate."""
    bureau_obj = _base_bureau("2024-01-01")
    bureaus_data = _base_bureaus_data([
        {"month_year_key": "2023-10", "status": "ok"},
        {"month_year_key": "2023-11", "status": "ok"},
        {"month_year_key": "bad_format", "status": "ok"},
        {"status": "ok"},  # Missing month_year_key
    ])
    result = evaluate_fx_b03(bureau_obj, bureaus_data, "equifax", PLACEHOLDERS)
    
    assert result["evidence"]["monthly_entries_total"] == 4
    assert result["evidence"]["monthly_entries_parseable_count"] == 2


# ── Status Space Validation ───────────────────────────────────────────────────

def test_no_unknown_status_emitted():
    """FX.B03 must NEVER return 'unknown' status."""
    test_cases = [
        ({}, {}),  # Missing everything
        ({"last_payment": "invalid"}, {}),  # Unparseable anchor
        ({"last_payment": "2024-01-01"}, _base_bureaus_data([])),  # Empty monthly
        ({"last_payment": "2024-01-01"}, _base_bureaus_data([{"month_year_key": "2024-06", "status": "ok"}])),  # Conflict
        ({"last_payment": "2024-06-01"}, _base_bureaus_data([{"month_year_key": "2024-06", "status": "ok"}])),  # OK
    ]
    
    for bureau_obj, bureaus_data in test_cases:
        result = evaluate_fx_b03(bureau_obj, bureaus_data, "equifax", PLACEHOLDERS)
        assert result["status"] in {"ok", "conflict", "skipped_missing_data"}, \
            f"Invalid status '{result['status']}' for case {bureau_obj}, {bureaus_data}"


def test_ungated_and_eligible_always_true():
    """FX.B03 is always ungated and eligible."""
    result = evaluate_fx_b03({}, {}, "equifax", PLACEHOLDERS)
    assert result["ungated"] is True
    assert result["eligible"] is True


# ── Edge Cases ────────────────────────────────────────────────────────────────

def test_handles_missing_status_field_in_monthly():
    """Monthly entries without status field are processed correctly."""
    bureau_obj = _base_bureau("2024-01-15")
    bureaus_data = _base_bureaus_data([
        {"month_year_key": "2024-01"},  # No status
        {"month_year_key": "2024-02", "status": "ok"},
    ])
    result = evaluate_fx_b03(bureau_obj, bureaus_data, "equifax", PLACEHOLDERS)
    assert result["status"] == "ok"
    assert result["evidence"]["max_month_key"] == "2024-02"


def test_handles_duplicate_month_keys():
    """Max is correctly computed even with duplicate month keys."""
    bureau_obj = _base_bureau("2024-02-15")
    bureaus_data = _base_bureaus_data([
        {"month_year_key": "2024-01", "status": "ok"},
        {"month_year_key": "2024-01", "status": "30"},  # Duplicate
        {"month_year_key": "2024-03", "status": "ok"},
    ])
    result = evaluate_fx_b03(bureau_obj, bureaus_data, "equifax", PLACEHOLDERS)
    assert result["status"] == "ok"
    assert result["evidence"]["max_month_key"] == "2024-03"
