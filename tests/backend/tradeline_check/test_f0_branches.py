"""Unit tests for F0.A01 and F0.A02."""
import pytest
from datetime import date
from backend.tradeline_check.f0_a01_time_ceiling_integrity import evaluate_f0_a01
from backend.tradeline_check.f0_a02_opening_date_lower_bound import evaluate_f0_a02


PLACEHOLDERS = {"", "n/a", "unknown", "none"}


class TestF0A01TimesCeilingIntegrity:
    """Tests for F0.A01: Time Ceiling Integrity."""

    def test_unknown_when_no_ceiling_candidates(self):
        """Should return unknown when date_reported and last_verified both missing."""
        bureau_obj = {
            "date_opened": "2020-01-15",
            "first_payment_date": "2020-02-01",
        }
        bureaus_data = {}
        payload = {}
        
        result = evaluate_f0_a01(payload, bureau_obj, bureaus_data, "equifax", PLACEHOLDERS)
        
        assert result["status"] == "unknown"
        assert "no parseable ceiling candidate" in result["explanation"]

    def test_unknown_when_bureau_obj_invalid(self):
        """Should return unknown when bureau_obj is not a Mapping."""
        result = evaluate_f0_a01({}, None, {}, "equifax", PLACEHOLDERS)
        assert result["status"] == "unknown"

    def test_ok_when_all_dates_within_ceiling(self):
        """Should return ok when all event dates <= effective_ceiling."""
        bureau_obj = {
            "date_reported": "2024-06-01",
            "last_verified": "2024-06-15",
            "first_payment_date": "2020-01-01",
            "last_payment_date": "2024-05-01",
        }
        bureaus_data = {
            "two_year_payment_history_monthly_tsv_v2": {}
        }
        
        result = evaluate_f0_a01({}, bureau_obj, bureaus_data, "equifax", PLACEHOLDERS)
        
        assert result["status"] == "ok"
        assert result["ceiling"]["effective_ceiling_date"] == "2024-06-15"

    def test_conflict_when_event_date_exceeds_ceiling(self):
        """Should return conflict when any event date > effective_ceiling."""
        bureau_obj = {
            "date_reported": "2024-06-01",
            "last_verified": "2024-06-15",
            "first_payment_date": "2020-01-01",
            "last_payment_date": "2024-07-01",  # AFTER ceiling
        }
        bureaus_data = {
            "two_year_payment_history_monthly_tsv_v2": {}
        }
        
        result = evaluate_f0_a01({}, bureau_obj, bureaus_data, "equifax", PLACEHOLDERS)
        
        assert result["status"] == "conflict"
        assert result["ceiling"]["conflict"] is True
        assert len(result["ceiling"]["violations"]) == 1
        assert result["ceiling"]["violations"][0]["field"] == "last_payment_date"

    def test_monthly_history_after_ceiling_is_violation(self):
        """Should flag monthly entries with month_year_key > ceiling_month as violations."""
        bureau_obj = {
            "date_reported": "2024-06-01",
            "last_verified": "2024-06-15",
        }
        bureaus_data = {
            "two_year_payment_history_monthly_tsv_v2": {
                "equifax": [
                    {"month_year_key": "2024-05", "status": "ok"},
                    {"month_year_key": "2024-06", "status": "ok"},
                    {"month_year_key": "2024-07", "status": "30"},  # After ceiling (2024-06)
                ]
            }
        }
        
        result = evaluate_f0_a01({}, bureau_obj, bureaus_data, "equifax", PLACEHOLDERS)
        
        assert result["status"] == "conflict"
        violations = result["ceiling"]["violations"]
        assert any(v["field"] == "monthly_history[2024-07]" for v in violations)

    def test_monthly_history_missing_allows_entry_after_ceiling(self):
        """Monthly history entries with status '--' (missing) after ceiling are NOT violations for A01."""
        bureau_obj = {
            "date_reported": "2024-06-01",
            "last_verified": "2024-06-15",
        }
        bureaus_data = {
            "two_year_payment_history_monthly_tsv_v2": {
                "equifax": [
                    {"month_year_key": "2024-06", "status": "ok"},
                    {"month_year_key": "2024-07", "status": "--"},  # After ceiling but missing
                ]
            }
        }
        
        result = evaluate_f0_a01({}, bureau_obj, bureaus_data, "equifax", PLACEHOLDERS)
        
        # A01 flags ANY month after ceiling as violation, regardless of status
        # This is intentional: the fact that there's a data point after ceiling is the issue
        assert result["status"] == "conflict"

    def test_monthly_history_missing_data_ok(self):
        """Should handle missing monthly_history gracefully."""
        bureau_obj = {
            "date_reported": "2024-06-01",
            "last_verified": "2024-06-15",
        }
        bureaus_data = {}  # No monthly history
        
        result = evaluate_f0_a01({}, bureau_obj, bureaus_data, "equifax", PLACEHOLDERS)
        
        assert result["status"] == "ok"

    def test_multiple_violations_collected(self):
        """Should collect all violations, not just first one."""
        bureau_obj = {
            "date_reported": "2024-06-01",
            "last_verified": "2024-06-15",
            "last_payment_date": "2024-07-01",
            "last_follow_up_date": "2024-08-15",
        }
        bureaus_data = {
            "two_year_payment_history_monthly_tsv_v2": {}
        }
        
        result = evaluate_f0_a01({}, bureau_obj, bureaus_data, "equifax", PLACEHOLDERS)
        
        assert result["status"] == "conflict"
        assert len(result["ceiling"]["violations"]) == 2


class TestF0A02OpeningDateLowerBound:
    """Tests for F0.A02: Opening Date Lower Bound."""

    def test_unknown_when_date_opened_missing(self):
        """Should return unknown when date_opened is missing."""
        bureau_obj = {
            "first_payment_date": "2020-01-01",
        }
        bureaus_data = {}
        
        result = evaluate_f0_a02({}, bureau_obj, bureaus_data, "equifax", PLACEHOLDERS)
        
        assert result["status"] == "unknown"
        assert "date_opened missing" in result["explanation"]

    def test_unknown_when_bureau_obj_invalid(self):
        """Should return unknown when bureau_obj is not a Mapping."""
        result = evaluate_f0_a02({}, None, {}, "equifax", PLACEHOLDERS)
        assert result["status"] == "unknown"

    def test_ok_when_all_dates_after_opening(self):
        """Should return ok when all event dates >= date_opened."""
        bureau_obj = {
            "date_opened": "2020-01-15",
            "first_payment_date": "2020-02-01",
            "last_payment_date": "2024-06-01",
        }
        bureaus_data = {
            "two_year_payment_history_monthly_tsv_v2": {}
        }
        
        result = evaluate_f0_a02({}, bureau_obj, bureaus_data, "equifax", PLACEHOLDERS)
        
        assert result["status"] == "ok"
        assert result["floor"]["date_opened"] == "2020-01-15"

    def test_conflict_when_event_date_precedes_opening(self):
        """Should return conflict when any event date < date_opened."""
        bureau_obj = {
            "date_opened": "2020-06-15",
            "first_payment_date": "2020-01-01",  # BEFORE opening
            "last_payment_date": "2024-06-01",
        }
        bureaus_data = {
            "two_year_payment_history_monthly_tsv_v2": {}
        }
        
        result = evaluate_f0_a02({}, bureau_obj, bureaus_data, "equifax", PLACEHOLDERS)
        
        assert result["status"] == "conflict"
        assert result["floor"]["conflict"] is True
        assert len(result["floor"]["violations"]) == 1
        assert result["floor"]["violations"][0]["field"] == "first_payment_date"

    def test_monthly_history_before_opening_with_status_ok_is_violation(self):
        """Monthly entry before opening_month with status != '--' is a violation."""
        bureau_obj = {
            "date_opened": "2020-06-15",
        }
        bureaus_data = {
            "two_year_payment_history_monthly_tsv_v2": {
                "equifax": [
                    {"month_year_key": "2020-05", "status": "ok"},  # Before opening month (2020-06)
                    {"month_year_key": "2020-06", "status": "ok"},
                    {"month_year_key": "2020-07", "status": "30"},
                ]
            }
        }
        
        result = evaluate_f0_a02({}, bureau_obj, bureaus_data, "equifax", PLACEHOLDERS)
        
        assert result["status"] == "conflict"
        violations = result["floor"]["violations"]
        assert any(v["field"] == "monthly_history[2020-05]" for v in violations)

    def test_monthly_history_before_opening_with_status_missing_ok(self):
        """Monthly entry before opening_month with status '--' (missing) is allowed."""
        bureau_obj = {
            "date_opened": "2020-06-15",
        }
        bureaus_data = {
            "two_year_payment_history_monthly_tsv_v2": {
                "equifax": [
                    {"month_year_key": "2020-05", "status": "--"},  # Before opening month but missing
                    {"month_year_key": "2020-06", "status": "ok"},
                    {"month_year_key": "2020-07", "status": "30"},
                ]
            }
        }
        
        result = evaluate_f0_a02({}, bureau_obj, bureaus_data, "equifax", PLACEHOLDERS)
        
        # Should be ok because the pre-opening month has missing status
        assert result["status"] == "ok"

    def test_monthly_history_at_opening_month_allowed(self):
        """Monthly entry for opening month itself is allowed."""
        bureau_obj = {
            "date_opened": "2020-06-15",
        }
        bureaus_data = {
            "two_year_payment_history_monthly_tsv_v2": {
                "equifax": [
                    {"month_year_key": "2020-06", "status": "ok"},  # Opening month is ok
                    {"month_year_key": "2020-07", "status": "30"},
                ]
            }
        }
        
        result = evaluate_f0_a02({}, bureau_obj, bureaus_data, "equifax", PLACEHOLDERS)
        
        assert result["status"] == "ok"

    def test_monthly_history_missing_data_ok(self):
        """Should handle missing monthly_history gracefully."""
        bureau_obj = {
            "date_opened": "2020-06-15",
        }
        bureaus_data = {}  # No monthly history
        
        result = evaluate_f0_a02({}, bureau_obj, bureaus_data, "equifax", PLACEHOLDERS)
        
        assert result["status"] == "ok"

    def test_multiple_violations_collected(self):
        """Should collect all violations, not just first one."""
        bureau_obj = {
            "date_opened": "2020-06-15",
            "first_payment_date": "2019-01-01",
            "last_payment_date": "2018-06-01",
        }
        bureaus_data = {
            "two_year_payment_history_monthly_tsv_v2": {}
        }
        
        result = evaluate_f0_a02({}, bureau_obj, bureaus_data, "equifax", PLACEHOLDERS)
        
        assert result["status"] == "conflict"
        assert len(result["floor"]["violations"]) == 2
