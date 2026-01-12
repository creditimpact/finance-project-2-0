"""Test that removed 2Y history fields are no longer injected into bureaus.json.

This test verifies that:
1. When monthly_v2 data exists, the new two_year_payment_history_monthly_tsv_v2 field is present.
2. Legacy two_year_payment_history field is ABSENT when monthly_v2 is present (primary source of truth).
3. The 4 removed fields are NEVER present in bureaus.json output:
   - two_year_payment_history_monthly (removed; never injected)
   - two_year_payment_history_months (removed; never injected)
   - two_year_payment_history_months_by_bureau (removed; never injected)
   - two_year_payment_history_months_tsv_v2 (removed; never injected)
4. When monthly_v2 data is absent, legacy field is injected for backward compatibility.
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
        
        # Include the new monthly_v2 field (list[dict] format)
        account["two_year_payment_history_monthly_tsv_v2"] = {
            "transunion": [
                {"month": "Jan", "status": "OK"},
                {"month": "Feb", "status": "OK"},
            ],
            "experian": [
                {"month": "Jan", "status": "OK"},
                {"month": "Feb", "status": "--"},
            ],
            "equifax": [],
        }

    return account


def test_bureaus_excludes_removed_fields_always(monkeypatch):
    """Verify the 4 removed fields are NEVER present, regardless of flag state."""
    monkeypatch.setenv("HISTORY_MAIN_WIRING_ENABLED", "1")
    
    # Re-import config to pick up the monkeypatched env var
    import importlib
    importlib.reload(config)
    from backend.core.logic.report_analysis import problem_case_builder
    importlib.reload(problem_case_builder)
    
    account = _make_test_account(include_monthly=True)
    payload = problem_case_builder._build_bureaus_payload_from_stagea(account)

    # When monthly_v2 is present, it is injected (new field), legacy absent
    assert "two_year_payment_history_monthly_tsv_v2" in payload
    assert "two_year_payment_history" not in payload
    assert "seven_year_history" in payload

    # Verify the 4 removed fields are NEVER present (removed entirely)
    assert "two_year_payment_history_monthly" not in payload
    assert "two_year_payment_history_months" not in payload
    assert "two_year_payment_history_months_by_bureau" not in payload
    assert "two_year_payment_history_months_tsv_v2" not in payload

    # Verify bureaus are still present
    assert "transunion" in payload
    assert "experian" in payload
    assert "equifax" in payload


def test_bureaus_removed_fields_with_flag_disabled(monkeypatch):
    """When HISTORY_MAIN_WIRING_ENABLED=0, removed fields still absent (always removed)."""
    monkeypatch.setenv("HISTORY_MAIN_WIRING_ENABLED", "0")
    
    # Re-import config to pick up the monkeypatched env var
    import importlib
    importlib.reload(config)
    from backend.core.logic.report_analysis import problem_case_builder
    importlib.reload(problem_case_builder)
    
    account = _make_test_account(include_monthly=True)
    payload = problem_case_builder._build_bureaus_payload_from_stagea(account)

    # When monthly_v2 is present, it is injected, legacy absent
    assert "two_year_payment_history_monthly_tsv_v2" in payload
    assert "two_year_payment_history" not in payload
    assert "seven_year_history" in payload

    # Verify the 4 removed fields are NEVER present (removed entirely, flag value irrelevant)
    assert "two_year_payment_history_monthly" not in payload
    assert "two_year_payment_history_months" not in payload
    assert "two_year_payment_history_months_by_bureau" not in payload
    assert "two_year_payment_history_months_tsv_v2" not in payload

    # Verify bureaus are still present
    assert "transunion" in payload
    assert "experian" in payload
    assert "equifax" in payload


def test_bureaus_removed_fields_missing_not_defaults(monkeypatch):
    """When account has no monthly data, legacy fallback is used (backward compatibility)."""
    monkeypatch.setenv("HISTORY_MAIN_WIRING_ENABLED", "1")
    
    # Re-import config and problem_case_builder
    import importlib
    importlib.reload(config)
    from backend.core.logic.report_analysis import problem_case_builder
    importlib.reload(problem_case_builder)
    
    account = _make_test_account(include_monthly=False)
    payload = problem_case_builder._build_bureaus_payload_from_stagea(account)

    # When no monthly_v2 data, legacy field is present (fallback)
    assert "two_year_payment_history" in payload
    assert "two_year_payment_history_monthly_tsv_v2" not in payload
    
    # Removed fields are absent (not present with empty defaults)
    assert "two_year_payment_history_monthly" not in payload
    assert "two_year_payment_history_months" not in payload
    assert "two_year_payment_history_months_by_bureau" not in payload
    assert "two_year_payment_history_months_tsv_v2" not in payload


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


def test_bureaus_legacy_backward_compatible(monkeypatch):
    """Verify backward compatibility: old accounts without monthly data use legacy field."""
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

    # No monthly_v2, so legacy field is present (backward compatibility)
    assert "two_year_payment_history" in payload
    assert "two_year_payment_history_monthly_tsv_v2" not in payload
    assert "seven_year_history" in payload

    # Removed fields should never be present
    assert "two_year_payment_history_monthly" not in payload
    assert "two_year_payment_history_months" not in payload
    assert "two_year_payment_history_months_by_bureau" not in payload
    assert "two_year_payment_history_months_tsv_v2" not in payload

    # Bureaus must be present
    assert "transunion" in payload
    assert "experian" in payload
    assert "equifax" in payload


def test_bureaus_empty_account_no_removed_fields(monkeypatch):
    """Test handling of empty or None account input; removed fields never present."""
    monkeypatch.setenv("HISTORY_MAIN_WIRING_ENABLED", "1")
    
    import importlib
    importlib.reload(config)
    from backend.core.logic.report_analysis import problem_case_builder
    importlib.reload(problem_case_builder)
    
    # Empty dict - no monthly_v2 or legacy history; nothing is injected
    payload1 = problem_case_builder._build_bureaus_payload_from_stagea({})
    assert isinstance(payload1, OrderedDict)
    assert "two_year_payment_history" not in payload1
    assert "two_year_payment_history_monthly_tsv_v2" not in payload1
    # Verify removed fields absent
    assert "two_year_payment_history_monthly" not in payload1
    assert "two_year_payment_history_months" not in payload1
    assert "two_year_payment_history_months_by_bureau" not in payload1
    assert "two_year_payment_history_months_tsv_v2" not in payload1

    # None input - no monthly_v2 or legacy history; nothing is injected
    payload2 = problem_case_builder._build_bureaus_payload_from_stagea(None)
    assert isinstance(payload2, OrderedDict)
    assert "two_year_payment_history" not in payload2
    assert "two_year_payment_history_monthly_tsv_v2" not in payload2
    # Verify removed fields absent
    assert "two_year_payment_history_monthly" not in payload2
    assert "two_year_payment_history_months" not in payload2
    assert "two_year_payment_history_months_by_bureau" not in payload2
    assert "two_year_payment_history_months_tsv_v2" not in payload2
