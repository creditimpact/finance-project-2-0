"""Unit tests for F2.B02 Post-Closure Activity Contradiction."""

import pytest
from backend.tradeline_check.f2_b02_post_closure_activity import evaluate_f2_b02


PLACEHOLDERS = {"", "n/a", "unknown", "none"}


class TestF2B02Eligibility:
    """Test F2.B02 eligibility gating."""

    def test_ineligible_state_1_open(self):
        """F2.B02 must be skipped when R1.state_num=1 (open)."""
        payload = {
            "routing": {"R1": {"state_num": 1}},
            "root_checks": {"Q1": {"declared_state": "open"}},
        }
        bureau_obj = {"closed_date": "2023-01-15"}

        result = evaluate_f2_b02(bureau_obj, payload, PLACEHOLDERS)

        assert result["status"] == "skipped"
        assert result["eligible"] is False
        assert result["executed"] is False

    def test_ineligible_state_3_unknown(self):
        """F2.B02 must be skipped when R1.state_num=3 (unknown)."""
        payload = {
            "routing": {"R1": {"state_num": 3}},
            "root_checks": {"Q1": {"declared_state": "unknown"}},
        }
        bureau_obj = {"closed_date": "2023-01-15"}

        result = evaluate_f2_b02(bureau_obj, payload, PLACEHOLDERS)

        assert result["status"] == "skipped"
        assert result["eligible"] is False

    def test_ineligible_state_4_conflict(self):
        """F2.B02 must be skipped when R1.state_num=4 (conflict)."""
        payload = {
            "routing": {"R1": {"state_num": 4}},
            "root_checks": {"Q1": {"declared_state": "conflict"}},
        }
        bureau_obj = {"closed_date": "2023-01-15"}

        result = evaluate_f2_b02(bureau_obj, payload, PLACEHOLDERS)

        assert result["status"] == "skipped"
        assert result["eligible"] is False

    def test_eligible_state_2_closed(self):
        """F2.B02 must be eligible when R1.state_num=2 (closed)."""
        payload = {
            "routing": {"R1": {"state_num": 2}},
            "root_checks": {"Q1": {"declared_state": "closed"}},
        }
        bureau_obj = {"closed_date": "2023-01-15"}

        result = evaluate_f2_b02(bureau_obj, payload, PLACEHOLDERS)

        assert result["eligible"] is True
        assert result["executed"] is True


class TestF2B02ClosedDateMissing:
    """Test scenarios where closed_date is missing/unparseable → skipped_missing_data."""

    def test_skipped_missing_data_closed_date_none(self):
        """closed_date=None → skipped_missing_data."""
        payload = {
            "routing": {"R1": {"state_num": 2}},
            "root_checks": {"Q1": {"declared_state": "closed"}},
        }
        bureau_obj = {
            "closed_date": None,
            "last_payment": "2023-06-01",
        }

        result = evaluate_f2_b02(bureau_obj, payload, PLACEHOLDERS)

        assert result["status"] == "skipped_missing_data"
        assert result["eligible"] is True
        assert result["executed"] is True
        assert result["fired"] is False
        assert "closed_date is missing or placeholder" in result["explanation"]
        assert result["evidence"]["reason"] == "closed_date missing or placeholder"

    def test_skipped_missing_data_closed_date_placeholder(self):
        """closed_date='n/a' → skipped_missing_data."""
        payload = {
            "routing": {"R1": {"state_num": 2}},
            "root_checks": {"Q1": {"declared_state": "closed"}},
        }
        bureau_obj = {
            "closed_date": "n/a",
            "last_payment": "2023-06-01",
        }

        result = evaluate_f2_b02(bureau_obj, payload, PLACEHOLDERS)

        assert result["status"] == "skipped_missing_data"
        assert result["evidence"]["reason"] == "closed_date missing or placeholder"

    def test_skipped_missing_data_closed_date_empty(self):
        """closed_date='' → skipped_missing_data."""
        payload = {
            "routing": {"R1": {"state_num": 2}},
            "root_checks": {"Q1": {"declared_state": "closed"}},
        }
        bureau_obj = {
            "closed_date": "",
            "last_payment": "2023-06-01",
        }

        result = evaluate_f2_b02(bureau_obj, payload, PLACEHOLDERS)

        assert result["status"] == "skipped_missing_data"

    def test_skipped_missing_data_closed_date_unparseable(self):
        """closed_date='not-a-date' → skipped_missing_data."""
        payload = {
            "routing": {"R1": {"state_num": 2}},
            "root_checks": {"Q1": {"declared_state": "closed"}},
        }
        bureau_obj = {
            "closed_date": "invalid-date-string",
            "last_payment": "2023-06-01",
        }

        result = evaluate_f2_b02(bureau_obj, payload, PLACEHOLDERS)

        assert result["status"] == "skipped_missing_data"
        assert result["evidence"]["reason"] == "closed_date unparseable"
        assert "closed_date is unparseable" in result["explanation"]


class TestF2B02OkNoActivity:
    """Test ok status when closed_date valid but no activity evidence."""

    def test_ok_both_activity_fields_missing(self):
        """closed_date valid + both activity fields missing → ok."""
        payload = {
            "routing": {"R1": {"state_num": 2}},
            "root_checks": {"Q1": {"declared_state": "closed"}},
        }
        bureau_obj = {
            "closed_date": "2023-06-15",
            "date_of_last_activity": None,
            "last_payment": None,
        }

        result = evaluate_f2_b02(bureau_obj, payload, PLACEHOLDERS)

        assert result["status"] == "ok"
        assert result["fired"] is False
        assert result["evidence"]["checked_fields_present"] == []
        assert result["evidence"]["parsed_fields_count"] == 0

    def test_ok_both_activity_fields_placeholder(self):
        """closed_date valid + both activity fields placeholder → ok."""
        payload = {
            "routing": {"R1": {"state_num": 2}},
            "root_checks": {"Q1": {"declared_state": "closed"}},
        }
        bureau_obj = {
            "closed_date": "2023-06-15",
            "date_of_last_activity": "n/a",
            "last_payment": "unknown",
        }

        result = evaluate_f2_b02(bureau_obj, payload, PLACEHOLDERS)

        assert result["status"] == "ok"
        assert result["evidence"]["checked_fields_present"] == []


class TestF2B02OkActivityUnparseable:
    """Test ok status when activity fields present but unparseable (treated as missing)."""

    def test_ok_activity_unparseable_single_field(self):
        """closed_date valid + single activity field unparseable → ok (not skipped_missing_data)."""
        payload = {
            "routing": {"R1": {"state_num": 2}},
            "root_checks": {"Q1": {"declared_state": "closed"}},
        }
        bureau_obj = {
            "closed_date": "2023-06-15",
            "date_of_last_activity": "not-a-date",  # Unparseable
            "last_payment": None,
        }

        result = evaluate_f2_b02(bureau_obj, payload, PLACEHOLDERS)

        assert result["status"] == "ok"  # NOT skipped_missing_data
        assert result["fired"] is False
        assert result["evidence"]["checked_fields_present"] == ["date_of_last_activity"]
        assert result["evidence"]["parsed_fields_count"] == 0  # Present but unparseable

    def test_ok_both_activity_fields_unparseable(self):
        """closed_date valid + both activity fields unparseable → ok."""
        payload = {
            "routing": {"R1": {"state_num": 2}},
            "root_checks": {"Q1": {"declared_state": "closed"}},
        }
        bureau_obj = {
            "closed_date": "2023-06-15",
            "date_of_last_activity": "invalid",
            "last_payment": "also-invalid",
        }

        result = evaluate_f2_b02(bureau_obj, payload, PLACEHOLDERS)

        assert result["status"] == "ok"
        assert result["evidence"]["checked_fields_present"] == [
            "date_of_last_activity",
            "last_payment",
        ]
        assert result["evidence"]["parsed_fields_count"] == 0

    def test_ok_mixed_one_unparseable_one_valid_before_closure(self):
        """One activity unparseable, one valid before closure → ok."""
        payload = {
            "routing": {"R1": {"state_num": 2}},
            "root_checks": {"Q1": {"declared_state": "closed"}},
        }
        bureau_obj = {
            "closed_date": "2023-06-15",
            "date_of_last_activity": "invalid-date",
            "last_payment": "2023-06-01",  # Before closure
        }

        result = evaluate_f2_b02(bureau_obj, payload, PLACEHOLDERS)

        assert result["status"] == "ok"
        assert result["evidence"]["parsed_fields_count"] == 1


class TestF2B02OkActivityBeforeClosure:
    """Test ok status when activity dates are before or equal to closed_date."""

    def test_ok_activity_before_closed_date(self):
        """Activity before closed_date → ok."""
        payload = {
            "routing": {"R1": {"state_num": 2}},
            "root_checks": {"Q1": {"declared_state": "closed"}},
        }
        bureau_obj = {
            "closed_date": "2023-06-15",
            "date_of_last_activity": "2023-06-01",  # Before
        }

        result = evaluate_f2_b02(bureau_obj, payload, PLACEHOLDERS)

        assert result["status"] == "ok"
        assert result["fired"] is False
        assert result["evidence"]["checked_fields_present"] == ["date_of_last_activity"]
        assert result["evidence"]["parsed_fields_count"] == 1

    def test_ok_activity_equal_to_closed_date(self):
        """Activity equal to closed_date → ok (not >)."""
        payload = {
            "routing": {"R1": {"state_num": 2}},
            "root_checks": {"Q1": {"declared_state": "closed"}},
        }
        bureau_obj = {
            "closed_date": "2023-06-15",
            "last_payment": "2023-06-15",  # Equal
        }

        result = evaluate_f2_b02(bureau_obj, payload, PLACEHOLDERS)

        assert result["status"] == "ok"
        assert result["fired"] is False

    def test_ok_both_activities_before_closure(self):
        """Both activities before closed_date → ok."""
        payload = {
            "routing": {"R1": {"state_num": 2}},
            "root_checks": {"Q1": {"declared_state": "closed"}},
        }
        bureau_obj = {
            "closed_date": "2023-06-15",
            "date_of_last_activity": "2023-06-01",
            "last_payment": "2023-06-10",
        }

        result = evaluate_f2_b02(bureau_obj, payload, PLACEHOLDERS)

        assert result["status"] == "ok"
        assert result["evidence"]["parsed_fields_count"] == 2


class TestF2B02ConflictPostClosureActivity:
    """Test conflict status when activity dates occur AFTER closed_date."""

    def test_conflict_date_of_last_activity_after_closure(self):
        """date_of_last_activity after closed_date → conflict."""
        payload = {
            "routing": {"R1": {"state_num": 2}},
            "root_checks": {"Q1": {"declared_state": "closed"}},
        }
        bureau_obj = {
            "closed_date": "2023-06-15",
            "date_of_last_activity": "2023-07-01",  # AFTER
        }

        result = evaluate_f2_b02(bureau_obj, payload, PLACEHOLDERS)

        assert result["status"] == "conflict"
        assert result["fired"] is True
        assert result["evidence"]["violating_field"] == "date_of_last_activity"
        assert result["evidence"]["violating_date_raw"] == "2023-07-01"
        assert result["evidence"]["violating_date_parsed"] == "2023-07-01"
        assert result["evidence"]["closed_date_parsed"] == "2023-06-15"

    def test_conflict_last_payment_after_closure(self):
        """last_payment after closed_date → conflict."""
        payload = {
            "routing": {"R1": {"state_num": 2}},
            "root_checks": {"Q1": {"declared_state": "closed"}},
        }
        bureau_obj = {
            "closed_date": "2023-01-15",
            "last_payment": "2023-02-20",  # AFTER
        }

        result = evaluate_f2_b02(bureau_obj, payload, PLACEHOLDERS)

        assert result["status"] == "conflict"
        assert result["fired"] is True
        assert result["evidence"]["violating_field"] == "last_payment"

    def test_conflict_early_exit_first_violation_wins(self):
        """Early-exit: date_of_last_activity checked first, if it violates, return immediately."""
        payload = {
            "routing": {"R1": {"state_num": 2}},
            "root_checks": {"Q1": {"declared_state": "closed"}},
        }
        bureau_obj = {
            "closed_date": "2023-01-15",
            "date_of_last_activity": "2023-02-01",  # AFTER (checked first)
            "last_payment": "2023-03-01",  # Also AFTER (but not checked due to early-exit)
        }

        result = evaluate_f2_b02(bureau_obj, payload, PLACEHOLDERS)

        assert result["status"] == "conflict"
        assert result["evidence"]["violating_field"] == "date_of_last_activity"
        # last_payment not mentioned because early-exit

    def test_conflict_mixed_one_before_one_after(self):
        """One activity before, one after → conflict (early-exit on first violation)."""
        payload = {
            "routing": {"R1": {"state_num": 2}},
            "root_checks": {"Q1": {"declared_state": "closed"}},
        }
        bureau_obj = {
            "closed_date": "2023-06-15",
            "date_of_last_activity": "2023-06-01",  # Before (checked first, ok)
            "last_payment": "2023-07-20",  # AFTER (checked second, violates)
        }

        result = evaluate_f2_b02(bureau_obj, payload, PLACEHOLDERS)

        assert result["status"] == "conflict"
        assert result["evidence"]["violating_field"] == "last_payment"


class TestF2B02OutputStructure:
    """Test F2.B02 output structure and required fields."""

    def test_output_has_required_fields_ok(self):
        """ok status must contain all required fields."""
        payload = {
            "routing": {"R1": {"state_num": 2}},
            "root_checks": {"Q1": {"declared_state": "closed"}},
        }
        bureau_obj = {"closed_date": "2023-06-15"}

        result = evaluate_f2_b02(bureau_obj, payload, PLACEHOLDERS)

        required_fields = {
            "version",
            "status",
            "eligible",
            "executed",
            "fired",
            "trigger",
            "evidence",
            "explanation",
        }
        assert required_fields.issubset(result.keys())
        assert result["version"] == "f2_b02_post_closure_activity_v1"

    def test_output_has_required_fields_skipped_missing_data(self):
        """skipped_missing_data status must contain all required fields."""
        payload = {
            "routing": {"R1": {"state_num": 2}},
            "root_checks": {"Q1": {"declared_state": "closed"}},
        }
        bureau_obj = {"closed_date": None}

        result = evaluate_f2_b02(bureau_obj, payload, PLACEHOLDERS)

        assert result["status"] == "skipped_missing_data"
        assert "evidence" in result
        assert "reason" in result["evidence"]

    def test_output_no_unknown_status_exists(self):
        """F2.B02 must NEVER return 'unknown' status."""
        # Test various scenarios that might trigger unknown in old implementation
        test_cases = [
            {"closed_date": None},
            {"closed_date": "invalid"},
            {"closed_date": ""},
            {"closed_date": "n/a"},
        ]

        payload = {
            "routing": {"R1": {"state_num": 2}},
            "root_checks": {"Q1": {"declared_state": "closed"}},
        }

        for bureau_obj in test_cases:
            result = evaluate_f2_b02(bureau_obj, payload, PLACEHOLDERS)
            assert result["status"] != "unknown", f"Got 'unknown' for {bureau_obj}"
            # Should be one of the 3 allowed statuses
            assert result["status"] in {"ok", "conflict", "skipped_missing_data"}


class TestF2B02EvidenceFields:
    """Test evidence field structure and content."""

    def test_evidence_checked_fields_present_populated(self):
        """checked_fields_present must list which activity fields were present (not missing/placeholder)."""
        payload = {
            "routing": {"R1": {"state_num": 2}},
            "root_checks": {"Q1": {"declared_state": "closed"}},
        }
        bureau_obj = {
            "closed_date": "2023-06-15",
            "date_of_last_activity": "2023-06-01",
            "last_payment": None,  # Missing
        }

        result = evaluate_f2_b02(bureau_obj, payload, PLACEHOLDERS)

        assert result["status"] == "ok"
        assert result["evidence"]["checked_fields_present"] == ["date_of_last_activity"]

    def test_evidence_parsed_fields_count_correct(self):
        """parsed_fields_count must reflect how many present fields were parseable."""
        payload = {
            "routing": {"R1": {"state_num": 2}},
            "root_checks": {"Q1": {"declared_state": "closed"}},
        }
        bureau_obj = {
            "closed_date": "2023-06-15",
            "date_of_last_activity": "2023-06-01",  # Present + parseable
            "last_payment": "invalid-date",  # Present but NOT parseable
        }

        result = evaluate_f2_b02(bureau_obj, payload, PLACEHOLDERS)

        assert result["evidence"]["checked_fields_present"] == [
            "date_of_last_activity",
            "last_payment",
        ]
        assert result["evidence"]["parsed_fields_count"] == 1  # Only 1 parsed successfully

    def test_evidence_conflict_includes_violating_field_details(self):
        """conflict evidence must include violating_field, raw, and parsed values."""
        payload = {
            "routing": {"R1": {"state_num": 2}},
            "root_checks": {"Q1": {"declared_state": "closed"}},
        }
        bureau_obj = {
            "closed_date": "2023-06-15",
            "last_payment": "2023-07-20",
        }

        result = evaluate_f2_b02(bureau_obj, payload, PLACEHOLDERS)

        assert result["status"] == "conflict"
        evidence = result["evidence"]
        assert "violating_field" in evidence
        assert "violating_date_raw" in evidence
        assert "violating_date_parsed" in evidence
        assert "closed_date_raw" in evidence
        assert "closed_date_parsed" in evidence


class TestF2B02EdgeCases:
    """Test edge cases and defensive handling."""

    def test_missing_routing_block(self):
        """Missing routing block → skipped."""
        payload = {"root_checks": {"Q1": {"declared_state": "closed"}}}
        bureau_obj = {"closed_date": "2023-06-15"}

        result = evaluate_f2_b02(bureau_obj, payload, PLACEHOLDERS)

        assert result["status"] == "skipped"
        assert result["eligible"] is False

    def test_whitespace_closed_date_treated_as_missing(self):
        """Whitespace-only closed_date → skipped_missing_data."""
        payload = {
            "routing": {"R1": {"state_num": 2}},
            "root_checks": {"Q1": {"declared_state": "closed"}},
        }
        bureau_obj = {
            "closed_date": "   ",  # Whitespace only
        }

        result = evaluate_f2_b02(bureau_obj, payload, PLACEHOLDERS)

        assert result["status"] == "skipped_missing_data"

    def test_case_insensitive_placeholder_matching(self):
        """Placeholder matching must be case-insensitive."""
        payload = {
            "routing": {"R1": {"state_num": 2}},
            "root_checks": {"Q1": {"declared_state": "closed"}},
        }
        bureau_obj = {
            "closed_date": "N/A",  # Uppercase
        }

        result = evaluate_f2_b02(bureau_obj, payload, PLACEHOLDERS)

        assert result["status"] == "skipped_missing_data"

    def test_empty_bureau_obj(self):
        """Empty bureau_obj → skipped_missing_data."""
        payload = {
            "routing": {"R1": {"state_num": 2}},
            "root_checks": {"Q1": {"declared_state": "closed"}},
        }
        bureau_obj = {}

        result = evaluate_f2_b02(bureau_obj, payload, PLACEHOLDERS)

        assert result["status"] == "skipped_missing_data"
