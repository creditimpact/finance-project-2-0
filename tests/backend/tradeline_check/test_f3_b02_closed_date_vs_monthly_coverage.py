"""Tests for F3.B02 — Closed Date vs Monthly Coverage."""
import pytest

from backend.tradeline_check.f3_b02_closed_date_vs_monthly_coverage import evaluate_f3_b02

PLACEHOLDERS = {"", "--", "n/a", "unknown"}


def _base_payload(r1_state_num: int = 2):
    """Create base payload with routing and root_checks."""
    return {
        "routing": {"R1": {"state_num": r1_state_num}},
        "root_checks": {"Q1": {"declared_state": "closed"}},
    }


def _base_bureau(closed_date: str = "2024-01-15"):
    return {"closed_date": closed_date}


def _base_bureaus_data(entries):
    return {"two_year_payment_history_monthly_tsv_v2": {"equifax": entries}}


# ── Eligibility Gating Tests ──────────────────────────────────────────────────

def test_skipped_when_ineligible_state_1():
    """R1.state_num=1 (open) => skipped (ineligible)."""
    payload = _base_payload(r1_state_num=1)
    bureau_obj = _base_bureau()
    bureaus_data = _base_bureaus_data([{"month_year_key": "2024-01", "status": "ok"}])
    
    result = evaluate_f3_b02(bureau_obj, bureaus_data, "equifax", payload, PLACEHOLDERS)
    
    assert result["status"] == "skipped"
    assert result["eligible"] is False
    assert result["executed"] is False
    assert result["fired"] is False
    assert "not eligible" in result["explanation"]


def test_eligible_for_state_2_closed():
    """R1.state_num=2 (closed) => eligible."""
    payload = _base_payload(r1_state_num=2)
    bureau_obj = _base_bureau()
    bureaus_data = _base_bureaus_data([{"month_year_key": "2024-01", "status": "ok"}])
    
    result = evaluate_f3_b02(bureau_obj, bureaus_data, "equifax", payload, PLACEHOLDERS)
    
    assert result["eligible"] is True
    assert result["status"] in {"ok", "conflict", "skipped_missing_data"}


def test_eligible_for_state_3_unknown():
    """R1.state_num=3 (unknown) => eligible."""
    payload = _base_payload(r1_state_num=3)
    bureau_obj = _base_bureau()
    bureaus_data = _base_bureaus_data([{"month_year_key": "2024-01", "status": "ok"}])
    
    result = evaluate_f3_b02(bureau_obj, bureaus_data, "equifax", payload, PLACEHOLDERS)
    
    assert result["eligible"] is True


def test_eligible_for_state_4_conflict():
    """R1.state_num=4 (conflict) => eligible."""
    payload = _base_payload(r1_state_num=4)
    bureau_obj = _base_bureau()
    bureaus_data = _base_bureaus_data([{"month_year_key": "2024-01", "status": "ok"}])
    
    result = evaluate_f3_b02(bureau_obj, bureaus_data, "equifax", payload, PLACEHOLDERS)
    
    assert result["eligible"] is True


# ── Missing/Unparseable Anchor Tests ──────────────────────────────────────────

def test_skipped_when_closed_date_missing():
    """closed_date missing => skipped_missing_data."""
    payload = _base_payload()
    result = evaluate_f3_b02({}, {}, "equifax", payload, PLACEHOLDERS)
    
    assert result["status"] == "skipped_missing_data"
    assert result["eligible"] is True
    assert result["executed"] is True
    assert result["fired"] is False


def test_skipped_when_closed_date_placeholder():
    """closed_date placeholder => skipped_missing_data."""
    payload = _base_payload()
    bureau_obj = {"closed_date": "n/a"}
    
    result = evaluate_f3_b02(bureau_obj, {}, "equifax", payload, PLACEHOLDERS)
    
    assert result["status"] == "skipped_missing_data"
    assert "missing or placeholder" in result["explanation"]


def test_skipped_when_closed_date_unparseable():
    """closed_date unparseable => skipped_missing_data."""
    payload = _base_payload()
    bureau_obj = {"closed_date": "not-a-date"}
    
    result = evaluate_f3_b02(bureau_obj, {}, "equifax", payload, PLACEHOLDERS)
    
    assert result["status"] == "skipped_missing_data"
    assert "unparseable" in result["explanation"]


# ── Missing Monthly History Tests ─────────────────────────────────────────────

def test_skipped_when_monthly_history_missing():
    """No monthly history block => skipped_missing_data."""
    payload = _base_payload()
    bureau_obj = _base_bureau()
    
    result = evaluate_f3_b02(bureau_obj, {}, "equifax", payload, PLACEHOLDERS)
    
    assert result["status"] == "skipped_missing_data"
    assert "missing or empty" in result["explanation"]


def test_skipped_when_monthly_history_empty():
    """Empty monthly list => skipped_missing_data."""
    payload = _base_payload()
    bureau_obj = _base_bureau()
    bureaus_data = _base_bureaus_data([])
    
    result = evaluate_f3_b02(bureau_obj, bureaus_data, "equifax", payload, PLACEHOLDERS)
    
    assert result["status"] == "skipped_missing_data"


def test_skipped_when_no_parseable_month_keys():
    """Monthly history present but no parseable month_year_key => skipped_missing_data."""
    payload = _base_payload()
    bureau_obj = _base_bureau()
    bureaus_data = _base_bureaus_data([
        {"month_year_key": "invalid", "status": "ok"},
        {"month_year_key": None, "status": "30"},
    ])
    
    result = evaluate_f3_b02(bureau_obj, bureaus_data, "equifax", payload, PLACEHOLDERS)
    
    assert result["status"] == "skipped_missing_data"
    assert "no parseable month_year_key" in result["explanation"]


# ── Conflict Detection Tests ──────────────────────────────────────────────────

def test_conflict_when_closed_date_after_max_month():
    """closed_date month > max monthly coverage month => conflict."""
    payload = _base_payload()
    bureau_obj = _base_bureau("2024-05-15")
    bureaus_data = _base_bureaus_data([
        {"month_year_key": "2024-01", "status": "ok"},
        {"month_year_key": "2024-02", "status": "30"},
        {"month_year_key": "2024-03", "status": "--"},
    ])
    
    result = evaluate_f3_b02(bureau_obj, bureaus_data, "equifax", payload, PLACEHOLDERS)
    
    assert result["status"] == "conflict"
    assert result["fired"] is True
    assert result["evidence"]["closed_month_key"] == "2024-05"
    assert result["evidence"]["max_month_key"] == "2024-03"
    assert "exceeds max monthly coverage" in result["explanation"]


def test_conflict_when_closed_date_far_future():
    """closed_date significantly beyond coverage => conflict."""
    payload = _base_payload()
    bureau_obj = _base_bureau("2025-12-31")
    bureaus_data = _base_bureaus_data([
        {"month_year_key": "2024-06", "status": "ok"},
    ])
    
    result = evaluate_f3_b02(bureau_obj, bureaus_data, "equifax", payload, PLACEHOLDERS)
    
    assert result["status"] == "conflict"
    assert result["fired"] is True


# ── OK (No Conflict) Tests ────────────────────────────────────────────────────

def test_ok_when_closed_date_within_coverage():
    """closed_date month <= max monthly coverage => ok."""
    payload = _base_payload()
    bureau_obj = _base_bureau("2024-02-15")
    bureaus_data = _base_bureaus_data([
        {"month_year_key": "2024-01", "status": "ok"},
        {"month_year_key": "2024-02", "status": "30"},
        {"month_year_key": "2024-03", "status": "--"},
    ])
    
    result = evaluate_f3_b02(bureau_obj, bureaus_data, "equifax", payload, PLACEHOLDERS)
    
    assert result["status"] == "ok"
    assert result["fired"] is False
    assert result["evidence"]["closed_month_key"] == "2024-02"
    assert result["evidence"]["max_month_key"] == "2024-03"


def test_ok_when_closed_date_equals_max_month():
    """closed_date month == max monthly coverage => ok (exact boundary)."""
    payload = _base_payload()
    bureau_obj = _base_bureau("2024-03-01")
    bureaus_data = _base_bureaus_data([
        {"month_year_key": "2024-03", "status": "ok"},
    ])
    
    result = evaluate_f3_b02(bureau_obj, bureaus_data, "equifax", payload, PLACEHOLDERS)
    
    assert result["status"] == "ok"
    assert result["fired"] is False


def test_ok_when_closed_date_before_max_month():
    """closed_date month < max monthly coverage => ok."""
    payload = _base_payload()
    bureau_obj = _base_bureau("2023-12-15")
    bureaus_data = _base_bureaus_data([
        {"month_year_key": "2024-01", "status": "ok"},
        {"month_year_key": "2024-02", "status": "30"},
    ])
    
    result = evaluate_f3_b02(bureau_obj, bureaus_data, "equifax", payload, PLACEHOLDERS)
    
    assert result["status"] == "ok"
    assert result["fired"] is False


# ── Evidence Field Tests ──────────────────────────────────────────────────────

def test_evidence_fields_populated():
    """Verify all evidence fields are present and correct."""
    payload = _base_payload()
    bureau_obj = _base_bureau("2024-02-10")
    bureaus_data = _base_bureaus_data([
        {"month_year_key": "2024-01", "status": "ok"},
        {"month_year_key": "2024-02", "status": "30"},
        {"month_year_key": "invalid", "status": "60"},  # Should be ignored
    ])
    
    result = evaluate_f3_b02(bureau_obj, bureaus_data, "equifax", payload, PLACEHOLDERS)
    
    assert result["evidence"]["closed_date_raw"] == "2024-02-10"
    assert result["evidence"]["closed_month_key"] == "2024-02"
    assert result["evidence"]["max_month_key"] == "2024-02"
    assert result["evidence"]["monthly_entries_total"] == 3
    assert result["evidence"]["monthly_entries_parseable_count"] == 2


def test_trigger_fields_populated():
    """Verify trigger fields contain routing info."""
    payload = _base_payload(r1_state_num=3)
    payload["root_checks"]["Q1"]["declared_state"] = "unknown"
    bureau_obj = _base_bureau()
    bureaus_data = _base_bureaus_data([{"month_year_key": "2024-01", "status": "ok"}])
    
    result = evaluate_f3_b02(bureau_obj, bureaus_data, "equifax", payload, PLACEHOLDERS)
    
    assert result["trigger"]["r1_state_num"] == 3
    assert result["trigger"]["q1_declared_state"] == "unknown"


# ── Status Space Validation ───────────────────────────────────────────────────

def test_no_unknown_status_emitted():
    """F3.B02 must NEVER return 'unknown' status."""
    test_cases = [
        (1, {}, {}),  # Ineligible
        (2, {}, {}),  # Missing closed_date
        (2, {"closed_date": "invalid"}, {}),  # Unparseable anchor
        (2, {"closed_date": "2024-01-01"}, _base_bureaus_data([])),  # Empty monthly
        (2, {"closed_date": "2024-01-01"}, _base_bureaus_data([{"month_year_key": "2024-06", "status": "ok"}])),  # Conflict
        (2, {"closed_date": "2024-06-01"}, _base_bureaus_data([{"month_year_key": "2024-06", "status": "ok"}])),  # OK
    ]
    
    for r1_state_num, bureau_obj, bureaus_data in test_cases:
        payload = _base_payload(r1_state_num)
        result = evaluate_f3_b02(bureau_obj, bureaus_data, "equifax", payload, PLACEHOLDERS)
        assert result["status"] in {"ok", "conflict", "skipped_missing_data", "skipped"}, \
            f"Invalid status '{result['status']}' for state={r1_state_num}, bureau={bureau_obj}"


def test_status_set_compliance():
    """F3.B02 returns only allowed statuses: ok, conflict, skipped_missing_data, skipped."""
    # Run through various scenarios
    scenarios = [
        (_base_payload(1), _base_bureau(), _base_bureaus_data([{"month_year_key": "2024-01", "status": "ok"}])),
        (_base_payload(2), {}, {}),
        (_base_payload(2), _base_bureau("2024-05-01"), _base_bureaus_data([{"month_year_key": "2024-01", "status": "ok"}])),
        (_base_payload(2), _base_bureau("2024-01-01"), _base_bureaus_data([{"month_year_key": "2024-06", "status": "ok"}])),
    ]
    
    for payload, bureau_obj, bureaus_data in scenarios:
        result = evaluate_f3_b02(bureau_obj, bureaus_data, "equifax", payload, PLACEHOLDERS)
        assert result["status"] in {"ok", "conflict", "skipped_missing_data", "skipped"}


# ── Edge Cases ────────────────────────────────────────────────────────────────

def test_handles_missing_r1_state():
    """Missing R1 state => ineligible (skipped)."""
    payload = {"routing": {}, "root_checks": {}}
    bureau_obj = _base_bureau()
    bureaus_data = _base_bureaus_data([{"month_year_key": "2024-01", "status": "ok"}])
    
    result = evaluate_f3_b02(bureau_obj, bureaus_data, "equifax", payload, PLACEHOLDERS)
    
    assert result["status"] == "skipped"
    assert result["eligible"] is False


def test_handles_malformed_routing():
    """Malformed routing structure => ineligible."""
    payload = {"routing": "not_a_dict", "root_checks": {}}
    bureau_obj = _base_bureau()
    bureaus_data = _base_bureaus_data([{"month_year_key": "2024-01", "status": "ok"}])
    
    result = evaluate_f3_b02(bureau_obj, bureaus_data, "equifax", payload, PLACEHOLDERS)
    
    assert result["status"] == "skipped"
    assert result["eligible"] is False


def test_handles_duplicate_month_keys():
    """Max is correctly computed even with duplicate month keys."""
    payload = _base_payload()
    bureau_obj = _base_bureau("2024-02-15")
    bureaus_data = _base_bureaus_data([
        {"month_year_key": "2024-01", "status": "ok"},
        {"month_year_key": "2024-01", "status": "30"},  # Duplicate
        {"month_year_key": "2024-03", "status": "ok"},
    ])
    
    result = evaluate_f3_b02(bureau_obj, bureaus_data, "equifax", payload, PLACEHOLDERS)
    
    assert result["status"] == "ok"
    assert result["evidence"]["max_month_key"] == "2024-03"
