import pytest

from backend.tradeline_check.fx_b02_seven_year_vs_two_year_consistency import evaluate_fx_b02


DEFAULT_PLACEHOLDERS = {"--", "n/a", "unknown"}


def _base_bureaus(seven=None, monthly=None):
    bureaus = {}
    if seven is not None:
        bureaus["seven_year_history"] = seven
    if monthly is not None:
        bureaus["two_year_payment_history_monthly_tsv_v2"] = monthly
    return bureaus


def test_skipped_missing_seven_year():
    bureaus_data = _base_bureaus(seven=None, monthly={"equifax": []})
    res = evaluate_fx_b02({}, bureaus_data, "equifax", DEFAULT_PLACEHOLDERS)
    assert res["status"] == "skipped_missing_data"
    assert res["fired"] is False


def test_skipped_missing_two_year():
    bureaus_data = _base_bureaus(seven={"equifax": {"late30": 0, "late60": 0, "late90": 0}}, monthly=None)
    res = evaluate_fx_b02({}, bureaus_data, "equifax", DEFAULT_PLACEHOLDERS)
    assert res["status"] == "skipped_missing_data"
    assert res["fired"] is False


def test_skipped_all_missing_entries():
    bureaus_data = _base_bureaus(
        seven={"equifax": {"late30": 0, "late60": 0, "late90": 0}},
        monthly={"equifax": [
            {"month_year_key": "2024-01", "status": "--"},
            {"month_year_key": "2024-02", "status": "--"},
        ]},
    )
    res = evaluate_fx_b02({}, bureaus_data, "equifax", DEFAULT_PLACEHOLDERS)
    assert res["status"] == "skipped_missing_data"
    assert res["fired"] is False


def test_ok_equal_counts():
    bureaus_data = _base_bureaus(
        seven={"equifax": {"late30": 1, "late60": 1, "late90": 1}},
        monthly={"equifax": [
            {"month_year_key": "2024-01", "status": "30"},
            {"month_year_key": "2024-02", "status": "60"},
            {"month_year_key": "2024-03", "status": "90"},
        ]},
    )
    res = evaluate_fx_b02({}, bureaus_data, "equifax", DEFAULT_PLACEHOLDERS)
    assert res["status"] == "ok"
    assert res["fired"] is False
    assert res["evidence"]["two_year_counts"] == {"late30": 1, "late60": 1, "late90": 1}


def test_conflict_two_year_exceeds():
    bureaus_data = _base_bureaus(
        seven={"equifax": {"late30": 0, "late60": 0, "late90": 0}},
        monthly={"equifax": [
            {"month_year_key": "2024-01", "status": "30"},
        ]},
    )
    res = evaluate_fx_b02({}, bureaus_data, "equifax", DEFAULT_PLACEHOLDERS)
    assert res["status"] == "conflict"
    assert res["fired"] is True
    assert res["evidence"]["two_year_counts"]["late30"] == 1


def test_conflict_co_counts_as_90():
    bureaus_data = _base_bureaus(
        seven={"equifax": {"late30": 0, "late60": 0, "late90": 0}},
        monthly={"equifax": [
            {"month_year_key": "2024-01", "status": "co"},
        ]},
    )
    res = evaluate_fx_b02({}, bureaus_data, "equifax", DEFAULT_PLACEHOLDERS)
    assert res["status"] == "conflict"
    assert res["fired"] is True
    assert res["evidence"]["two_year_counts"]["late90"] == 1


def test_duplicate_month_worst_severity_once():
    bureaus_data = _base_bureaus(
        seven={"equifax": {"late30": 0, "late60": 0, "late90": 0}},
        monthly={"equifax": [
            {"month_year_key": "2024-01", "status": "30"},
            {"month_year_key": "2024-01", "status": "90"},
            {"month_year_key": "2024-02", "status": "60"},
        ]},
    )
    res = evaluate_fx_b02({}, bureaus_data, "equifax", DEFAULT_PLACEHOLDERS)
    assert res["status"] == "conflict"
    assert res["fired"] is True
    # 2024-01 should count once as late90, 2024-02 as late60
    assert res["evidence"]["two_year_counts"] == {"late30": 0, "late60": 1, "late90": 1}
    # example months should include worst offenders, length capped
    assert "2024-01" in res["evidence"]["example_months"]
