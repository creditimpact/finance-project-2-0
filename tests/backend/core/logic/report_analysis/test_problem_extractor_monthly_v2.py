"""Test problem_extractor migration to use monthly_v2 for derogatory status extraction."""

import pytest
from typing import Dict, Any, List

from backend.core.logic.report_analysis.problem_extractor import build_rule_fields_from_triad


def test_build_rule_fields_derogatory_with_monthly_v2_only():
    """Verify derogatory status is correctly extracted from monthly_v2 field only."""
    
    # Create a mock account with ONLY monthly_v2 data (no legacy two_year_payment_history)
    account: Dict[str, Any] = {
        "account_number": "1234",
        "heading": "Test Account",
        "triad_fields": {
            "transunion": {"account_number": "1234"},
            "experian": {"account_number": "1234"},
            "equifax": {},
        },
        # NO legacy two_year_payment_history
        # Only monthly_v2
        "two_year_payment_history_monthly_tsv_v2": {
            "transunion": [
                {"month": "Jan", "status": "OK"},
                {"month": "Feb", "status": "30"},  # derogatory: 30-day late
                {"month": "Mar", "status": "OK"},
            ],
            "experian": [
                {"month": "Jan", "status": "OK"},
                {"month": "Feb", "status": "--"},
            ],
            "equifax": [],
        },
    }
    
    # Extract rule fields
    fields, prov = build_rule_fields_from_triad(account)
    
    # Verify derogatory flag is set correctly based on monthly_v2 status
    # "30" is a derogatory status (30-day late)
    assert fields is not None, "Rule fields should be extracted"
    assert "has_derog_2y" in fields, "Derogatory flag should be present"
    
    # TransUnion has a "30" status in monthly_v2 -> has_derog_2y=True
    assert fields["has_derog_2y"] is True, "Should detect derogatory status from monthly_v2 (30-day late)"


def test_build_rule_fields_derogatory_monthly_v2_no_late():
    """Verify derogatory is False when monthly_v2 has only OK statuses (no dashes)."""
    
    account: Dict[str, Any] = {
        "account_number": "1234",
        "heading": "Test Account",
        "triad_fields": {
            "transunion": {"account_number": "1234"},
            "experian": {"account_number": "1234"},
            "equifax": {},
        },
        "two_year_payment_history_monthly_tsv_v2": {
            "transunion": [
                {"month": "Jan", "status": "OK"},
                {"month": "Feb", "status": "OK"},
            ],
            "experian": [
                {"month": "Jan", "status": "OK"},
                {"month": "Feb", "status": "OK"},
            ],
            "equifax": [],
        },
    }
    
    fields, prov = build_rule_fields_from_triad(account)
    
    assert fields is not None
    assert "has_derog_2y" in fields
    # Only OK statuses (no "--" or late codes) -> has_derog_2y=False
    assert fields["has_derog_2y"] is False, "Should not detect derogatory when all statuses are OK"


def test_build_rule_fields_derogatory_monthly_v2_fallback_to_legacy():
    """Verify fallback to legacy two_year_payment_history when monthly_v2 is absent."""
    
    account: Dict[str, Any] = {
        "account_number": "1234",
        "heading": "Test Account",
        "triad_fields": {
            "transunion": {"account_number": "1234"},
            "experian": {"account_number": "1234"},
            "equifax": {},
        },
        # Only legacy, no monthly_v2
        "two_year_payment_history": {
            "transunion": ["OK", "30", "OK"],  # has 30-day late
            "experian": ["OK", "OK"],
            "equifax": [],
        },
    }
    
    fields, prov = build_rule_fields_from_triad(account)
    
    assert fields is not None
    assert "has_derog_2y" in fields
    # TransUnion legacy has "30" -> has_derog_2y=True (fallback path)
    assert fields["has_derog_2y"] is True, "Should detect derogatory from legacy fallback when monthly_v2 absent"


def test_build_rule_fields_derogatory_60day_late():
    """Verify 60-day late status is detected as derogatory in monthly_v2."""
    
    account: Dict[str, Any] = {
        "account_number": "1234",
        "heading": "Test Account",
        "triad_fields": {
            "transunion": {"account_number": "1234"},
            "experian": {"account_number": "1234"},
            "equifax": {},
        },
        "two_year_payment_history_monthly_tsv_v2": {
            "transunion": [
                {"month": "Jan", "status": "OK"},
                {"month": "Feb", "status": "60"},  # derogatory: 60-day late
            ],
            "experian": [
                {"month": "Jan", "status": "OK"},
            ],
            "equifax": [],
        },
    }
    
    fields, prov = build_rule_fields_from_triad(account)
    
    assert fields is not None
    assert "has_derog_2y" in fields
    # "60" status -> has_derog_2y=True
    assert fields["has_derog_2y"] is True, "Should detect 60-day late as derogatory"


def test_build_rule_fields_derogatory_90day_late():
    """Verify 90-day late status is detected as derogatory in monthly_v2."""
    
    account: Dict[str, Any] = {
        "account_number": "1234",
        "heading": "Test Account",
        "triad_fields": {
            "transunion": {"account_number": "1234"},
            "experian": {"account_number": "1234"},
            "equifax": {},
        },
        "two_year_payment_history_monthly_tsv_v2": {
            "transunion": [
                {"month": "Jan", "status": "OK"},
                {"month": "Feb", "status": "90"},  # derogatory: 90+ day late
            ],
            "experian": [
                {"month": "Jan", "status": "OK"},
            ],
            "equifax": [],
        },
    }
    
    fields, prov = build_rule_fields_from_triad(account)
    
    assert fields is not None
    assert "has_derog_2y" in fields
    # "90" status -> has_derog_2y=True
    assert fields["has_derog_2y"] is True, "Should detect 90+ day late as derogatory"


def test_build_rule_fields_derogatory_empty_monthly_v2():
    """Verify derogatory is False when monthly_v2 is empty."""
    
    account: Dict[str, Any] = {
        "account_number": "1234",
        "heading": "Test Account",
        "triad_fields": {
            "transunion": {"account_number": "1234"},
            "experian": {"account_number": "1234"},
            "equifax": {},
        },
        "two_year_payment_history_monthly_tsv_v2": {
            "transunion": [],
            "experian": [],
            "equifax": [],
        },
    }
    
    fields, prov = build_rule_fields_from_triad(account)
    
    assert fields is not None
    assert "has_derog_2y" in fields
    # Empty monthly_v2 -> has_derog_2y=False
    assert fields["has_derog_2y"] is False, "Should not detect derogatory when monthly_v2 is empty"
