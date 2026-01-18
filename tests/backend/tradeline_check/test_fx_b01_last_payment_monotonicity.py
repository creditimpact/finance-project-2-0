import pytest

from backend.tradeline_check.fx_b01_last_payment_monotonicity import evaluate_fx_b01

PLACEHOLDERS = {"", "--", "n/a", "unknown"}


def _base_bureau(last_payment: str = "2024-01-15"):
    return {"last_payment": last_payment}


def _base_bureaus_data(entries):
    return {"two_year_payment_history_monthly_tsv_v2": {"equifax": entries}}


def test_skipped_when_last_payment_missing():
    result = evaluate_fx_b01({}, {}, "equifax", PLACEHOLDERS)
    assert result["status"] == "skipped_missing_data"
    assert result["fired"] is False


def test_skipped_when_last_payment_unparseable():
    result = evaluate_fx_b01({"last_payment": "not-a-date"}, {}, "equifax", PLACEHOLDERS)
    assert result["status"] == "skipped_missing_data"


def test_skipped_when_history_missing():
    bureau_obj = _base_bureau()
    result = evaluate_fx_b01(bureau_obj, {}, "equifax", PLACEHOLDERS)
    assert result["status"] == "skipped_missing_data"


def test_skipped_when_last_payment_month_not_found():
    bureau_obj = _base_bureau()
    bureaus_data = _base_bureaus_data([
        {"month_year_key": "2024-02", "status": "30"},
    ])
    result = evaluate_fx_b01(bureau_obj, bureaus_data, "equifax", PLACEHOLDERS)
    assert result["status"] == "skipped_missing_data"


def test_conflict_on_severity_improvement():
    bureau_obj = _base_bureau()
    bureaus_data = _base_bureaus_data([
        {"month_year_key": "2024-01", "status": "90"},
        {"month_year_key": "2024-02", "status": "30"},  # improvement
    ])
    result = evaluate_fx_b01(bureau_obj, bureaus_data, "equifax", PLACEHOLDERS)
    assert result["status"] == "conflict"
    assert result["fired"] is True


def test_ok_when_no_improvement():
    bureau_obj = _base_bureau()
    bureaus_data = _base_bureaus_data([
        {"month_year_key": "2024-01", "status": "30"},
        {"month_year_key": "2024-02", "status": "60"},
        {"month_year_key": "2024-03", "status": "--"},
    ])
    result = evaluate_fx_b01(bureau_obj, bureaus_data, "equifax", PLACEHOLDERS)
    assert result["status"] == "ok"
    assert result["fired"] is False


def test_no_unknown_status_emitted():
    bureau_obj = _base_bureau()
    bureaus_data = _base_bureaus_data([
        {"month_year_key": "2024-01", "status": "30"},
        {"month_year_key": "2024-02", "status": "60"},
    ])
    result = evaluate_fx_b01(bureau_obj, bureaus_data, "equifax", PLACEHOLDERS)
    assert result["status"] in {"ok", "conflict", "skipped_missing_data"}

