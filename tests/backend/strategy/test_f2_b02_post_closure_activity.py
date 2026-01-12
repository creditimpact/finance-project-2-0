"""Unit tests for F2.B02 — Post-Closure Activity Contradiction."""
from __future__ import annotations

import pytest
from backend.tradeline_check.f2_b02_post_closure_activity import evaluate_f2_b02


class TestF2B02EligibilityGating:
    """Test routing-based eligibility gating."""

    def test_eligible_states_3_and_4(self):
        """F2.B02 should be eligible for R1 states 3-4 (Q1=closed, Q2=ok or conflict)."""
        for state_num in [3, 4]:
            bureau_obj = {
                "closed_date": "2024-06-01",
                "last_payment": "2024-05-15",
            }
            payload = {
                "routing": {
                    "R1": {"state_num": state_num}
                },
                "root_checks": {
                    "Q1": {"declared_state": "closed"},
                    "Q2": {"status": "ok"},
                },
            }
            placeholders = {"--", "n/a", "unknown"}

            result = evaluate_f2_b02(bureau_obj, payload, placeholders)

            assert result["eligible"] is True, f"State {state_num} should be eligible"
            assert result["executed"] is True
            assert result["status"] in {"ok", "conflict", "unknown"}

    def test_ineligible_open_account_states(self):
        """F2.B02 should skip for open account states (1-2)."""
        for state_num in [1, 2, 5, 6, 7]:
            bureau_obj = {
                "closed_date": "2024-06-01",
                "last_payment": "2024-05-15",
            }
            payload = {
                "routing": {
                    "R1": {"state_num": state_num}
                },
                "root_checks": {
                    "Q1": {"declared_state": "open"},
                    "Q2": {"status": "ok"},
                },
            }
            placeholders = {"--", "n/a", "unknown"}

            result = evaluate_f2_b02(bureau_obj, payload, placeholders)

            assert result["status"] == "skipped", f"Open state {state_num} should be skipped"
            assert result["eligible"] is False


class TestF2B02ConflictDetection:
    """Test conflict detection logic."""

    def test_conflict_last_payment_after_closed_date(self):
        """Detect conflict when last_payment is after closed_date."""
        bureau_obj = {
            "closed_date": "2024-06-01",
            "last_payment": "2024-08-15",  # AFTER closure
        }
        payload = {
            "routing": {
                "R1": {"state_num": 3}
            },
            "root_checks": {
                "Q1": {"declared_state": "closed"},
                "Q2": {"status": "ok"},
            },
        }
        placeholders = {"--", "n/a", "unknown"}

        result = evaluate_f2_b02(bureau_obj, payload, placeholders)

        assert result["status"] == "conflict"
        assert result["fired"] is True
        assert result["evidence"]["violating_field"] == "last_payment"
        assert result["evidence"]["violating_date_parsed"] == "2024-08-15"
        assert result["evidence"]["closed_date_parsed"] == "2024-06-01"
        assert "last_payment occurs after closed_date" in result["explanation"]

    def test_conflict_date_of_last_activity_after_closed_date(self):
        """Detect conflict when date_of_last_activity is after closed_date."""
        bureau_obj = {
            "closed_date": "2024-06-01",
            "date_of_last_activity": "2024-07-20",  # AFTER closure
        }
        payload = {
            "routing": {
                "R1": {"state_num": 3}
            },
            "root_checks": {
                "Q1": {"declared_state": "closed"},
                "Q2": {"status": "ok"},
            },
        }
        placeholders = {"--", "n/a", "unknown"}

        result = evaluate_f2_b02(bureau_obj, payload, placeholders)

        assert result["status"] == "conflict"
        assert result["fired"] is True
        assert result["evidence"]["violating_field"] == "date_of_last_activity"
        assert result["evidence"]["violating_date_parsed"] == "2024-07-20"
        assert result["evidence"]["closed_date_parsed"] == "2024-06-01"
        assert "date_of_last_activity occurs after closed_date" in result["explanation"]

    def test_conflict_both_fields_after_closed_date_reports_first(self):
        """When both fields violate, report the first one checked (last_payment)."""
        bureau_obj = {
            "closed_date": "2024-06-01",
            "last_payment": "2024-08-15",  # AFTER
            "date_of_last_activity": "2024-07-20",  # AFTER
        }
        payload = {
            "routing": {
                "R1": {"state_num": 3}
            },
            "root_checks": {
                "Q1": {"declared_state": "closed"},
                "Q2": {"status": "ok"},
            },
        }
        placeholders = {"--", "n/a", "unknown"}

        result = evaluate_f2_b02(bureau_obj, payload, placeholders)

        assert result["status"] == "conflict"
        assert result["fired"] is True
        # Should report first violation (last_payment is checked first)
        assert result["evidence"]["violating_field"] == "last_payment"

    def test_conflict_with_various_date_formats(self):
        """Conflict detection should work with various date formats."""
        # Note: parse_date_any uses heuristics, so we test formats that it can disambiguate
        test_cases = [
            ("06/01/2024", "08/15/2024"),  # MM/DD/YYYY - June vs August
            ("2024-06-01", "2024-08-15"),  # ISO
        ]

        for closed_date, last_payment in test_cases:
            bureau_obj = {
                "closed_date": closed_date,
                "last_payment": last_payment,
            }
            payload = {
                "routing": {
                    "R1": {"state_num": 3}
                },
                "root_checks": {
                    "Q1": {"declared_state": "closed"},
                    "Q2": {"status": "ok"},
                },
            }
            placeholders = {"--", "n/a", "unknown"}

            result = evaluate_f2_b02(bureau_obj, payload, placeholders)

            assert result["status"] == "conflict", f"Failed for {closed_date} vs {last_payment}"
            assert result["fired"] is True


class TestF2B02OkCases:
    """Test OK (non-conflict) cases."""

    def test_ok_last_payment_before_closed_date(self):
        """No conflict when last_payment is before closed_date."""
        bureau_obj = {
            "closed_date": "2024-06-01",
            "last_payment": "2024-05-15",  # BEFORE closure
        }
        payload = {
            "routing": {
                "R1": {"state_num": 3}
            },
            "root_checks": {
                "Q1": {"declared_state": "closed"},
                "Q2": {"status": "conflict"},
            },
        }
        placeholders = {"--", "n/a", "unknown"}

        result = evaluate_f2_b02(bureau_obj, payload, placeholders)

        assert result["status"] == "ok"
        assert result["fired"] is False
        assert result["evidence"]["checked_fields"] == ["last_payment"]
        assert "no activity or payments detected after closed_date" in result["explanation"]

    def test_ok_last_payment_equal_to_closed_date(self):
        """No conflict when last_payment equals closed_date (same day)."""
        bureau_obj = {
            "closed_date": "2024-06-01",
            "last_payment": "2024-06-01",  # SAME day as closure
        }
        payload = {
            "routing": {
                "R1": {"state_num": 3}
            },
            "root_checks": {
                "Q1": {"declared_state": "closed"},
                "Q2": {"status": "conflict"},
            },
        }
        placeholders = {"--", "n/a", "unknown"}

        result = evaluate_f2_b02(bureau_obj, payload, placeholders)

        assert result["status"] == "ok"
        assert result["fired"] is False

    def test_ok_date_of_last_activity_before_closed_date(self):
        """No conflict when date_of_last_activity is before closed_date."""
        bureau_obj = {
            "closed_date": "2024-06-01",
            "date_of_last_activity": "2024-04-20",  # BEFORE closure
        }
        payload = {
            "routing": {
                "R1": {"state_num": 3}
            },
            "root_checks": {
                "Q1": {"declared_state": "closed"},
                "Q2": {"status": "conflict"},
            },
        }
        placeholders = {"--", "n/a", "unknown"}

        result = evaluate_f2_b02(bureau_obj, payload, placeholders)

        assert result["status"] == "ok"
        assert result["fired"] is False
        assert result["evidence"]["checked_fields"] == ["date_of_last_activity"]

    def test_ok_both_fields_before_closed_date(self):
        """No conflict when both activity fields are before closed_date."""
        bureau_obj = {
            "closed_date": "2024-06-01",
            "last_payment": "2024-05-15",  # BEFORE
            "date_of_last_activity": "2024-04-20",  # BEFORE
        }
        payload = {
            "routing": {
                "R1": {"state_num": 3}
            },
            "root_checks": {
                "Q1": {"declared_state": "closed"},
                "Q2": {"status": "conflict"},
            },
        }
        placeholders = {"--", "n/a", "unknown"}

        result = evaluate_f2_b02(bureau_obj, payload, placeholders)

        assert result["status"] == "ok"
        assert result["fired"] is False
        assert set(result["evidence"]["checked_fields"]) == {"last_payment", "date_of_last_activity"}

    def test_ok_no_activity_fields_present(self):
        """No conflict when no activity fields are present."""
        bureau_obj = {
            "closed_date": "2024-06-01",
            # No last_payment or date_of_last_activity
        }
        payload = {
            "routing": {
                "R1": {"state_num": 3}
            },
            "root_checks": {
                "Q1": {"declared_state": "closed"},
                "Q2": {"status": "ok"},
            },
        }
        placeholders = {"--", "n/a", "unknown"}

        result = evaluate_f2_b02(bureau_obj, payload, placeholders)

        assert result["status"] == "ok"
        assert result["fired"] is False
        assert result["evidence"]["checked_fields"] == []

    def test_ok_activity_fields_are_placeholders(self):
        """No conflict when activity fields are placeholders."""
        bureau_obj = {
            "closed_date": "2024-06-01",
            "last_payment": "--",
            "date_of_last_activity": "n/a",
        }
        payload = {
            "routing": {
                "R1": {"state_num": 3}
            },
            "root_checks": {
                "Q1": {"declared_state": "closed"},
                "Q2": {"status": "ok"},
            },
        }
        placeholders = {"--", "n/a", "unknown"}

        result = evaluate_f2_b02(bureau_obj, payload, placeholders)

        assert result["status"] == "ok"
        assert result["fired"] is False
        assert result["evidence"]["checked_fields"] == []

    def test_ok_activity_fields_unparseable(self):
        """No conflict when activity fields are present but unparseable."""
        bureau_obj = {
            "closed_date": "2024-06-01",
            "last_payment": "invalid-date",
            "date_of_last_activity": "not a date",
        }
        payload = {
            "routing": {
                "R1": {"state_num": 3}
            },
            "root_checks": {
                "Q1": {"declared_state": "closed"},
                "Q2": {"status": "ok"},
            },
        }
        placeholders = {"--", "n/a", "unknown"}

        result = evaluate_f2_b02(bureau_obj, payload, placeholders)

        assert result["status"] == "ok"
        assert result["fired"] is False
        # Fields were present but unparseable, so they appear in checked_fields
        # (checked_fields includes all non-missing fields, regardless of parseability)
        assert set(result["evidence"]["checked_fields"]) == {"last_payment", "date_of_last_activity"}


class TestF2B02EdgeCases:
    """Test edge cases and boundary conditions."""

    def test_one_field_after_one_field_before(self):
        """One activity field after, one before - should still conflict."""
        bureau_obj = {
            "closed_date": "2024-06-01",
            "last_payment": "2024-08-15",  # AFTER (violation)
            "date_of_last_activity": "2024-04-20",  # BEFORE (ok)
        }
        payload = {
            "routing": {
                "R1": {"state_num": 3}
            },
            "root_checks": {
                "Q1": {"declared_state": "closed"},
                "Q2": {"status": "ok"},
            },
        }
        placeholders = {"--", "n/a", "unknown"}

        result = evaluate_f2_b02(bureau_obj, payload, placeholders)

        assert result["status"] == "conflict"
        assert result["fired"] is True
        assert result["evidence"]["violating_field"] == "last_payment"

    def test_one_field_parseable_one_unparseable_conflict(self):
        """One parseable field violates, one unparseable - should conflict."""
        bureau_obj = {
            "closed_date": "2024-06-01",
            "last_payment": "invalid-date",  # Unparseable
            "date_of_last_activity": "2024-08-15",  # AFTER (violation)
        }
        payload = {
            "routing": {
                "R1": {"state_num": 3}
            },
            "root_checks": {
                "Q1": {"declared_state": "closed"},
                "Q2": {"status": "conflict"},
            },
        }
        placeholders = {"--", "n/a", "unknown"}

        result = evaluate_f2_b02(bureau_obj, payload, placeholders)

        assert result["status"] == "conflict"
        assert result["fired"] is True
        assert result["evidence"]["violating_field"] == "date_of_last_activity"

    def test_closed_date_with_time_component(self):
        """Handle closed_date with time component."""
        bureau_obj = {
            "closed_date": "2024-06-01 00:00:00",
            "last_payment": "2024-06-02",  # AFTER
        }
        payload = {
            "routing": {
                "R1": {"state_num": 3}
            },
            "root_checks": {
                "Q1": {"declared_state": "closed"},
                "Q2": {"status": "conflict"},
            },
        }
        placeholders = {"--", "n/a", "unknown"}

        result = evaluate_f2_b02(bureau_obj, payload, placeholders)

        assert result["status"] == "conflict"
        assert result["fired"] is True

    def test_missing_routing_information(self):
        """Handle missing routing information gracefully."""
        bureau_obj = {
            "closed_date": "2024-06-01",
            "last_payment": "2024-05-15",
        }
        payload = {
            # No routing section
            "root_checks": {
                "Q1": {"declared_state": "closed"},
                "Q2": {"status": "ok"},
            },
        }
        placeholders = {"--", "n/a", "unknown"}

        result = evaluate_f2_b02(bureau_obj, payload, placeholders)

        assert result["status"] == "skipped"
        assert result["eligible"] is False

    def test_version_field_present(self):
        """Ensure version field is present in all outputs."""
        bureau_obj = {
            "closed_date": "2024-06-01",
            "last_payment": "2024-05-15",
        }
        payload = {
            "routing": {
                "R1": {"state_num": 3}
            },
            "root_checks": {
                "Q1": {"declared_state": "closed"},
                "Q2": {"status": "ok"},
            },
        }
        placeholders = {"--", "n/a", "unknown"}

        result = evaluate_f2_b02(bureau_obj, payload, placeholders)

        assert "version" in result
        assert result["version"] == "f2_b02_post_closure_activity_v1"


class TestF2B02OutputContract:
    """Test output structure matches specification."""

    def test_conflict_output_structure(self):
        """Verify conflict output has all required fields."""
        bureau_obj = {
            "closed_date": "2024-06-01",
            "last_payment": "2024-08-15",
        }
        payload = {
            "routing": {
                "R1": {"state_num": 3}
            },
            "root_checks": {
                "Q1": {"declared_state": "closed"},
                "Q2": {"status": "ok"},
            },
        }
        placeholders = {"--", "n/a", "unknown"}

        result = evaluate_f2_b02(bureau_obj, payload, placeholders)

        # Top-level fields
        assert "version" in result
        assert "status" in result
        assert "eligible" in result
        assert "executed" in result
        assert "fired" in result
        assert "trigger" in result
        assert "evidence" in result
        assert "explanation" in result

        # Trigger fields
        assert "r1_state_num" in result["trigger"]
        assert "q1_declared_state" in result["trigger"]
        assert "q2_status" in result["trigger"]

        # Evidence fields (conflict-specific)
        assert "closed_date_raw" in result["evidence"]
        assert "closed_date_parsed" in result["evidence"]
        assert "violating_field" in result["evidence"]
        assert "violating_date_raw" in result["evidence"]
        assert "violating_date_parsed" in result["evidence"]

    def test_ok_output_structure(self):
        """Verify OK output has all required fields."""
        bureau_obj = {
            "closed_date": "2024-06-01",
            "last_payment": "2024-05-15",
        }
        payload = {
            "routing": {
                "R1": {"state_num": 3}
            },
            "root_checks": {
                "Q1": {"declared_state": "closed"},
                "Q2": {"status": "ok"},
            },
        }
        placeholders = {"--", "n/a", "unknown"}

        result = evaluate_f2_b02(bureau_obj, payload, placeholders)

        # Top-level fields
        assert "version" in result
        assert "status" in result
        assert "eligible" in result
        assert "executed" in result
        assert "fired" in result
        assert "trigger" in result
        assert "evidence" in result
        assert "explanation" in result

        # Evidence fields (OK-specific)
        assert "closed_date_raw" in result["evidence"]
        assert "closed_date_parsed" in result["evidence"]
        assert "checked_fields" in result["evidence"]
        assert isinstance(result["evidence"]["checked_fields"], list)

    def test_skipped_output_structure(self):
        """Verify skipped output has required fields."""
        bureau_obj = {
            "closed_date": "2024-06-01",
            "last_payment": "2024-05-15",
        }
        payload = {
            "routing": {
                "R1": {"state_num": 5}  # Ineligible
            },
            "root_checks": {
                "Q1": {"declared_state": "closed"},
                "Q2": {"status": "skipped_missing_data"},
            },
        }
        placeholders = {"--", "n/a", "unknown"}

        result = evaluate_f2_b02(bureau_obj, payload, placeholders)

        assert "version" in result
        assert "status" in result
        assert "eligible" in result
        assert "executed" in result
        assert "fired" in result
        assert "explanation" in result
        assert result["status"] == "skipped"
        assert result["eligible"] is False
        assert result["executed"] is False
        assert result["fired"] is False
