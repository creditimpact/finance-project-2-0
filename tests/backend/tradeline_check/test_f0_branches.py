"""Unit tests for F0.A01, F0.A02, F0.A03, and F0.A04."""
import pytest
from datetime import date
from backend.tradeline_check.f0_a01_time_ceiling_integrity import evaluate_f0_a01
from backend.tradeline_check.f0_a02_opening_date_lower_bound import evaluate_f0_a02
from backend.tradeline_check.f0_a03_monthly_ceiling_integrity import evaluate_f0_a03
from backend.tradeline_check.f0_a04_monthly_floor_integrity import evaluate_f0_a04


PLACEHOLDERS = {"", "n/a", "unknown", "none"}


class TestF0A01TimesCeilingIntegrity:
    """Tests for F0.A01: Time Ceiling Integrity."""

    def test_skipped_when_no_ceiling_candidates(self):
        """Should return skipped_missing_data when date_reported and last_verified both missing."""
        bureau_obj = {
            "date_opened": "2020-01-15",
            "first_payment_date": "2020-02-01",
        }
        bureaus_data = {}
        payload = {}
        
        result = evaluate_f0_a01(payload, bureau_obj, bureaus_data, "equifax", PLACEHOLDERS)
        
        assert result["status"] == "skipped_missing_data"
        assert "no parseable ceiling candidate" in result["explanation"]

    def test_skipped_when_bureau_obj_invalid(self):
        """Should return skipped_missing_data when bureau_obj is not a Mapping."""
        result = evaluate_f0_a01({}, None, {}, "equifax", PLACEHOLDERS)
        assert result["status"] == "skipped_missing_data"

    def test_ok_when_all_dates_within_ceiling(self):
        """Should return ok when all bureau date fields <= effective_ceiling."""
        bureau_obj = {
            "date_reported": "2024-06-01",
            "last_verified": "2024-06-15",
            "first_payment_date": "2020-01-01",
            "last_payment_date": "2024-05-01",
        }
        bureaus_data = {}
        
        result = evaluate_f0_a01({}, bureau_obj, bureaus_data, "equifax", PLACEHOLDERS)
        
        assert result["status"] == "ok"
        assert result["ceiling"]["effective_ceiling_date"] == "2024-06-15"

    def test_conflict_when_event_date_exceeds_ceiling(self):
        """Should return conflict when any bureau date field > effective_ceiling."""
        bureau_obj = {
            "date_reported": "2024-06-01",
            "last_verified": "2024-06-15",
            "first_payment_date": "2020-01-01",
            "last_payment_date": "2024-07-01",  # AFTER ceiling
        }
        bureaus_data = {}
        
        result = evaluate_f0_a01({}, bureau_obj, bureaus_data, "equifax", PLACEHOLDERS)
        
        assert result["status"] == "conflict"
        assert result["ceiling"]["conflict"] is True
        assert len(result["ceiling"]["violations"]) == 1
        assert result["ceiling"]["violations"][0]["field"] == "last_payment_date"



    def test_multiple_violations_collected(self):
        """Should collect all violations, not just first one."""
        bureau_obj = {
            "date_reported": "2024-06-01",
            "last_verified": "2024-06-15",
            "last_payment_date": "2024-07-01",
            "last_follow_up_date": "2024-08-15",
        }
        bureaus_data = {}
        
        result = evaluate_f0_a01({}, bureau_obj, bureaus_data, "equifax", PLACEHOLDERS)
        
        assert result["status"] == "conflict"
        assert len(result["ceiling"]["violations"]) == 2


class TestF0A02OpeningDateLowerBound:
    """Tests for F0.A02: Opening Date Lower Bound."""

    def test_skipped_when_date_opened_missing(self):
        """Should return skipped_missing_data when date_opened is missing."""
        bureau_obj = {
            "first_payment_date": "2020-01-01",
        }
        bureaus_data = {}
        
        result = evaluate_f0_a02({}, bureau_obj, bureaus_data, "equifax", PLACEHOLDERS)
        
        assert result["status"] == "skipped_missing_data"
        assert "date_opened missing" in result["explanation"]

    def test_skipped_when_bureau_obj_invalid(self):
        """Should return skipped_missing_data when bureau_obj is not a Mapping."""
        result = evaluate_f0_a02({}, None, {}, "equifax", PLACEHOLDERS)
        assert result["status"] == "skipped_missing_data"

    def test_ok_when_all_dates_after_opening(self):
        """Should return ok when all bureau date fields >= date_opened."""
        bureau_obj = {
            "date_opened": "2020-01-15",
            "first_payment_date": "2020-02-01",
            "last_payment_date": "2024-06-01",
        }
        bureaus_data = {}
        
        result = evaluate_f0_a02({}, bureau_obj, bureaus_data, "equifax", PLACEHOLDERS)
        
        assert result["status"] == "ok"
        assert result["floor"]["date_opened"] == "2020-01-15"

    def test_conflict_when_event_date_precedes_opening(self):
        """Should return conflict when any bureau date field < date_opened."""
        bureau_obj = {
            "date_opened": "2020-06-15",
            "first_payment_date": "2020-01-01",  # BEFORE opening
            "last_payment_date": "2024-06-01",
        }
        bureaus_data = {}
        
        result = evaluate_f0_a02({}, bureau_obj, bureaus_data, "equifax", PLACEHOLDERS)
        
        assert result["status"] == "conflict"
        assert result["floor"]["conflict"] is True
        assert len(result["floor"]["violations"]) == 1
        assert result["floor"]["violations"][0]["field"] == "first_payment_date"

    def test_monthly_history_before_opening_with_status_ok_is_violation(self):
        """Should collect all violations from bureau date fields."""
        """Should collect all violations, not just first one."""
        bureau_obj = {
            "date_opened": "2020-06-15",
            "first_payment_date": "2019-01-01",
            "last_payment_date": "2018-06-01",
        }
        bureaus_data = {}
        
        result = evaluate_f0_a02({}, bureau_obj, bureaus_data, "equifax", PLACEHOLDERS)
        
        assert result["status"] == "conflict"
        assert len(result["floor"]["violations"]) == 2


class TestF0A03MonthlyCeilingIntegrity:
    """Tests for F0.A03: Monthly Ceiling Integrity."""

    def test_skipped_when_no_ceiling_candidates(self):
        """Should return skipped_missing_data when date_reported and last_verified both missing."""
        bureau_obj = {
            "date_opened": "2020-01-15",
        }
        bureaus_data = {}
        payload = {"record_integrity": {"F0": {}}}
        
        result = evaluate_f0_a03(payload, bureau_obj, bureaus_data, "equifax", PLACEHOLDERS)
        
        assert result["status"] == "skipped_missing_data"
        assert "no parseable ceiling candidate" in result["explanation"]

    def test_skipped_when_monthly_data_missing(self):
        """Should return skipped_missing_data when monthly_history is missing."""
        bureau_obj = {
            "date_reported": "2024-06-01",
            "last_verified": "2024-06-15",
        }
        bureaus_data = {}
        payload = {"record_integrity": {"F0": {}}}
        
        result = evaluate_f0_a03(payload, bureau_obj, bureaus_data, "equifax", PLACEHOLDERS)
        
        assert result["status"] == "skipped_missing_data"
        assert "monthly history data missing or invalid" in result["explanation"]

    def test_skipped_when_monthly_data_empty(self):
        """Should return skipped_missing_data when monthly_history is empty."""
        bureau_obj = {
            "date_reported": "2024-06-01",
            "last_verified": "2024-06-15",
        }
        bureaus_data = {
            "two_year_payment_history_monthly_tsv_v2": {
                "equifax": []
            }
        }
        payload = {"record_integrity": {"F0": {}}}
        
        result = evaluate_f0_a03(payload, bureau_obj, bureaus_data, "equifax", PLACEHOLDERS)
        
        assert result["status"] == "skipped_missing_data"
        assert "monthly history data missing or invalid" in result["explanation"]

    def test_ok_when_all_monthly_entries_within_ceiling(self):
        """Should return ok when all monthly entries <= ceiling_month."""
        bureau_obj = {
            "date_reported": "2024-06-01",
            "last_verified": "2024-06-15",
        }
        bureaus_data = {
            "two_year_payment_history_monthly_tsv_v2": {
                "equifax": [
                    {"month_year_key": "2024-05", "status": "ok"},
                    {"month_year_key": "2024-06", "status": "ok"},
                ]
            }
        }
        payload = {"record_integrity": {"F0": {}}}
        
        result = evaluate_f0_a03(payload, bureau_obj, bureaus_data, "equifax", PLACEHOLDERS)
        
        assert result["status"] == "ok"
        assert result["monthly_ceiling"]["effective_ceiling_date"] == "2024-06-15"
        assert result["monthly_ceiling"]["ceiling_month_key"] == "2024-06"

    def test_conflict_when_monthly_entry_exceeds_ceiling(self):
        """Should return conflict when any monthly entry > ceiling_month."""
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
        payload = {"record_integrity": {"F0": {}}}
        
        result = evaluate_f0_a03(payload, bureau_obj, bureaus_data, "equifax", PLACEHOLDERS)
        
        assert result["status"] == "conflict"
        assert result["monthly_ceiling"]["conflict"] is True
        violations = result["monthly_ceiling"]["violations"]
        assert len(violations) == 1
        assert violations[0]["field"] == "monthly_history[2024-07]"
        assert violations[0]["date"] == "2024-07-01"
        assert result["monthly_ceiling"]["ceiling_month_key"] == "2024-06"

    def test_monthly_entry_after_ceiling_with_missing_status_still_violation(self):
        """Monthly entry after ceiling is violation regardless of status."""
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
        payload = {"record_integrity": {"F0": {}}}
        
        result = evaluate_f0_a03(payload, bureau_obj, bureaus_data, "equifax", PLACEHOLDERS)
        
        # A03 flags ANY month after ceiling as violation, regardless of status
        assert result["status"] == "conflict"
        assert any(v["field"] == "monthly_history[2024-07]" for v in result["monthly_ceiling"]["violations"])

    def test_multiple_monthly_violations_collected(self):
        """Should collect all monthly violations."""
        bureau_obj = {
            "date_reported": "2024-06-01",
            "last_verified": "2024-06-15",
        }
        bureaus_data = {
            "two_year_payment_history_monthly_tsv_v2": {
                "equifax": [
                    {"month_year_key": "2024-06", "status": "ok"},
                    {"month_year_key": "2024-07", "status": "30"},
                    {"month_year_key": "2024-08", "status": "60"},
                ]
            }
        }
        payload = {"record_integrity": {"F0": {}}}
        
        result = evaluate_f0_a03(payload, bureau_obj, bureaus_data, "equifax", PLACEHOLDERS)
        
        assert result["status"] == "conflict"
        assert len(result["monthly_ceiling"]["violations"]) == 2


class TestF0A04MonthlyFloorIntegrity:
    """Tests for F0.A04: Monthly Floor Integrity."""

    def test_skipped_when_date_opened_missing(self):
        """Should return skipped_missing_data when date_opened is missing."""
        bureau_obj = {
            "first_payment_date": "2020-01-01",
        }
        bureaus_data = {}
        payload = {"record_integrity": {"F0": {}}}
        
        result = evaluate_f0_a04(payload, bureau_obj, bureaus_data, "equifax", PLACEHOLDERS)
        
        assert result["status"] == "skipped_missing_data"
        assert "date_opened missing" in result["explanation"]

    def test_skipped_when_monthly_data_missing(self):
        """Should return skipped_missing_data when monthly_history is missing."""
        bureau_obj = {
            "date_opened": "2020-06-15",
        }
        bureaus_data = {}
        payload = {"record_integrity": {"F0": {}}}
        
        result = evaluate_f0_a04(payload, bureau_obj, bureaus_data, "equifax", PLACEHOLDERS)
        
        assert result["status"] == "skipped_missing_data"
        assert "monthly history data missing or invalid" in result["explanation"]

    def test_skipped_when_monthly_data_empty(self):
        """Should return skipped_missing_data when monthly_history is empty."""
        bureau_obj = {
            "date_opened": "2020-06-15",
        }
        bureaus_data = {
            "two_year_payment_history_monthly_tsv_v2": {
                "equifax": []
            }
        }
        payload = {"record_integrity": {"F0": {}}}
        
        result = evaluate_f0_a04(payload, bureau_obj, bureaus_data, "equifax", PLACEHOLDERS)
        
        assert result["status"] == "skipped_missing_data"
        assert "monthly history data missing or invalid" in result["explanation"]

    def test_ok_when_all_pre_opening_months_have_missing_status(self):
        """Should return ok when pre-opening months have status '--'."""
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
        payload = {"record_integrity": {"F0": {}}}
        
        result = evaluate_f0_a04(payload, bureau_obj, bureaus_data, "equifax", PLACEHOLDERS)
        
        assert result["status"] == "ok"
        assert result["monthly_floor"]["date_opened"] == "2020-06-15"
        assert result["monthly_floor"]["opened_month_key"] == "2020-06"

    def test_conflict_when_pre_opening_month_has_non_missing_status(self):
        """Should return conflict when pre-opening month has status != '--'."""
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
        payload = {"record_integrity": {"F0": {}}}
        
        result = evaluate_f0_a04(payload, bureau_obj, bureaus_data, "equifax", PLACEHOLDERS)
        
        assert result["status"] == "conflict"
        assert result["monthly_floor"]["conflict"] is True
        violations = result["monthly_floor"]["violations"]
        assert len(violations) == 1
        assert violations[0]["field"] == "monthly_history[2020-05]"
        assert violations[0]["date"] == "2020-05-01"
        assert violations[0]["status"] == "ok"
        assert result["monthly_floor"]["opened_month_key"] == "2020-06"

    def test_opening_month_entry_allowed(self):
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
        payload = {"record_integrity": {"F0": {}}}
        
        result = evaluate_f0_a04(payload, bureau_obj, bureaus_data, "equifax", PLACEHOLDERS)
        
        assert result["status"] == "ok"

    def test_multiple_pre_opening_violations_collected(self):
        """Should collect all pre-opening violations."""
        bureau_obj = {
            "date_opened": "2020-06-15",
        }
        bureaus_data = {
            "two_year_payment_history_monthly_tsv_v2": {
                "equifax": [
                    {"month_year_key": "2020-04", "status": "30"},
                    {"month_year_key": "2020-05", "status": "ok"},
                    {"month_year_key": "2020-06", "status": "ok"},
                ]
            }
        }
        payload = {"record_integrity": {"F0": {}}}
        
        result = evaluate_f0_a04(payload, bureau_obj, bureaus_data, "equifax", PLACEHOLDERS)
        
        assert result["status"] == "conflict"
        assert len(result["monthly_floor"]["violations"]) == 2
