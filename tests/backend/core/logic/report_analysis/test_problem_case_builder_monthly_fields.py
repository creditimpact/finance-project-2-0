"""Test flag-gated propagation of two-year payment history monthly fields into bureaus.json.

This test verifies that:
1. When HISTORY_MAIN_WIRING_ENABLED=1, the _build_bureaus_payload_from_stagea() function
   includes two_year_payment_history_monthly and two_year_payment_history_months in output.
2. When HISTORY_MAIN_WIRING_ENABLED=0, these fields are NOT included.
3. The output structure remains backward compatible (legacy fields always present).
"""

import pytest
from collections import OrderedDict

from backend.core.logic.report_analysis.problem_case_builder import (
    _build_bureaus_payload_from_stagea,
)
from backend import config


def _make_test_account(include_monthly: bool = True):
    """Create a fake Stage A account dict for testing."""
    account = {
        "heading": "AMEX CARD",
        "account_number": "1234567890",
        "lines": [{"text": "Some line"}],
        "triad_fields": {
            "transunion": {"account_number": "1234", "balance": "100"},
            "experian": {"account_number": "1234", "balance": "100"},
            "equifax": {},
        },
        "two_year_payment_history": {
            "transunion": ["OK", "OK", "30"],
            "experian": ["OK", "OK", "OK"],
            "equifax": [],
        },
        "seven_year_history": {
            "transunion": {"late30": 1, "late60": 0, "late90": 0},
            "experian": {"late30": 0, "late60": 0, "late90": 0},
            "equifax": {"late30": 0, "late60": 0, "late90": 0},
        },
    }

    if include_monthly:
        account["two_year_payment_history_monthly"] = {
            "transunion": [
                {"month": "Jan", "value": "OK"},
                {"month": "Feb", "value": "OK"},
            ],
            "experian": [
                {"month": "Jan", "value": "OK"},
                {"month": "Feb", "value": "--"},
            ],
            "equifax": [],
        }
        account["two_year_payment_history_months"] = ["Jan", "Feb", "Mar", "Apr"]

    return account


def test_bureaus_includes_monthly_when_flag_enabled(monkeypatch):
    """When HISTORY_MAIN_WIRING_ENABLED=1, monthly fields should be in bureaus.json output."""
    monkeypatch.setenv("HISTORY_MAIN_WIRING_ENABLED", "1")
    
    # Re-import config to pick up the monkeypatched env var
    import importlib
    importlib.reload(config)
    from backend.core.logic.report_analysis import problem_case_builder
    importlib.reload(problem_case_builder)
    
    account = _make_test_account(include_monthly=True)
    payload = problem_case_builder._build_bureaus_payload_from_stagea(account)

    # Verify legacy fields are present
    assert "two_year_payment_history" in payload
    assert "seven_year_history" in payload

    # Verify monthly fields ARE included when flag is enabled
    assert "two_year_payment_history_monthly" in payload
    assert "two_year_payment_history_months" in payload

    # Verify the monthly data is correct
    assert payload["two_year_payment_history_monthly"] == account["two_year_payment_history_monthly"]
    assert payload["two_year_payment_history_months"] == account["two_year_payment_history_months"]

    # Verify bureaus are still present
    assert "transunion" in payload
    assert "experian" in payload
    assert "equifax" in payload


def test_bureaus_excludes_monthly_when_flag_disabled(monkeypatch):
    """When HISTORY_MAIN_WIRING_ENABLED=0, monthly fields should NOT be in output."""
    monkeypatch.setenv("HISTORY_MAIN_WIRING_ENABLED", "0")
    
    # Re-import config to pick up the monkeypatched env var
    import importlib
    importlib.reload(config)
    from backend.core.logic.report_analysis import problem_case_builder
    importlib.reload(problem_case_builder)
    
    account = _make_test_account(include_monthly=True)
    payload = problem_case_builder._build_bureaus_payload_from_stagea(account)

    # Verify legacy fields are present
    assert "two_year_payment_history" in payload
    assert "seven_year_history" in payload

    # Verify monthly fields are NOT included when flag is disabled
    assert "two_year_payment_history_monthly" not in payload
    assert "two_year_payment_history_months" not in payload

    # Verify bureaus are still present
    assert "transunion" in payload
    assert "experian" in payload
    assert "equifax" in payload


def test_bureaus_monthly_defaults_to_empty_when_missing(monkeypatch):
    """When flag is enabled but account has no monthly data, output empty defaults."""
    monkeypatch.setenv("HISTORY_MAIN_WIRING_ENABLED", "1")
    
    # Re-import config and problem_case_builder
    import importlib
    importlib.reload(config)
    from backend.core.logic.report_analysis import problem_case_builder
    importlib.reload(problem_case_builder)
    
    account = _make_test_account(include_monthly=False)
    payload = problem_case_builder._build_bureaus_payload_from_stagea(account)

    # Verify monthly fields are present with defaults when flag is enabled
    assert "two_year_payment_history_monthly" in payload
    assert payload["two_year_payment_history_monthly"] == {}

    assert "two_year_payment_history_months" in payload
    assert payload["two_year_payment_history_months"] == []


def test_bureaus_payload_is_ordered_dict(monkeypatch):
    """Verify the output is an OrderedDict for consistent serialization."""
    monkeypatch.setenv("HISTORY_MAIN_WIRING_ENABLED", "1")
    
    import importlib
    importlib.reload(config)
    from backend.core.logic.report_analysis import problem_case_builder
    importlib.reload(problem_case_builder)
    
    account = _make_test_account(include_monthly=True)
    payload = problem_case_builder._build_bureaus_payload_from_stagea(account)

    assert isinstance(payload, OrderedDict)


def test_bureaus_backward_compatibility(monkeypatch):
    """Verify backward compatibility: old accounts without monthly data still work."""
    monkeypatch.setenv("HISTORY_MAIN_WIRING_ENABLED", "0")
    
    import importlib
    importlib.reload(config)
    from backend.core.logic.report_analysis import problem_case_builder
    importlib.reload(problem_case_builder)
    
    # Old-style account without monthly fields
    account = {
        "heading": "OLD ACCOUNT",
        "triad_fields": {
            "transunion": {"balance": "500"},
            "experian": {"balance": "500"},
            "equifax": {},
        },
        "two_year_payment_history": {
            "transunion": ["OK"],
            "experian": ["OK"],
            "equifax": [],
        },
        "seven_year_history": {
            "transunion": {"late30": 0, "late60": 0, "late90": 0},
            "experian": {"late30": 0, "late60": 0, "late90": 0},
            "equifax": {"late30": 0, "late60": 0, "late90": 0},
        },
    }

    payload = problem_case_builder._build_bureaus_payload_from_stagea(account)

    # Legacy fields must be present
    assert "two_year_payment_history" in payload
    assert "seven_year_history" in payload

    # Monthly fields should not be present when flag=0
    assert "two_year_payment_history_monthly" not in payload
    assert "two_year_payment_history_months" not in payload

    # Bureaus must be present
    assert "transunion" in payload
    assert "experian" in payload
    assert "equifax" in payload


def test_bureaus_empty_account(monkeypatch):
    """Test handling of empty or None account input."""
    monkeypatch.setenv("HISTORY_MAIN_WIRING_ENABLED", "1")
    
    import importlib
    importlib.reload(config)
    from backend.core.logic.report_analysis import problem_case_builder
    importlib.reload(problem_case_builder)
    
    # Empty dict
    payload1 = problem_case_builder._build_bureaus_payload_from_stagea({})
    assert isinstance(payload1, OrderedDict)
    assert "two_year_payment_history" in payload1
    assert payload1["two_year_payment_history"] == {}

    # None input
    payload2 = problem_case_builder._build_bureaus_payload_from_stagea(None)
    assert isinstance(payload2, OrderedDict)
    assert "two_year_payment_history" in payload2
    assert payload2["two_year_payment_history"] == {}
