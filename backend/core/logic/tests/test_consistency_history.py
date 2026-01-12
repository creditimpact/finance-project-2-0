import pytest

from backend.core.logic.consistency import _get_bureau_value, normalize_two_year_history


def test_get_bureau_value_prefers_monthly_tsv_v2():
    bureaus = {
        "transunion": {"account_status": "Open"},
        "two_year_payment_history_monthly_tsv_v2": {
            "transunion": [
                {"month": "2024-01", "status": "ok"},
                {"month": "2024-02", "status": "120"},
            ]
        },
        "two_year_payment_history": {"transunion": ["LEGACY", "90"]},
    }

    value = _get_bureau_value(bureaus, "two_year_payment_history", "transunion")

    assert isinstance(value, list)
    assert value and isinstance(value[0], dict)

    normalized = normalize_two_year_history(value)
    assert normalized["tokens"] == ["OK", "120"]
    assert normalized["counts"]["late90"] == 1


def test_get_bureau_value_fallbacks_to_legacy_when_no_monthly():
    bureaus = {
        "experian": {"account_status": "Open"},
        "two_year_payment_history": {"experian": ["OK", "30"]},
    }

    value = _get_bureau_value(bureaus, "two_year_payment_history", "experian")

    assert value == ["OK", "30"]

    normalized = normalize_two_year_history(value)
    assert normalized["tokens"] == ["OK", "30"]
    assert normalized["counts"]["late30"] == 1


def test_get_bureau_value_other_fields_unchanged():
    bureaus = {
        "equifax": {"account_status": "Open"},
    }

    value = _get_bureau_value(bureaus, "account_status", "equifax")
    assert value == "Open"
