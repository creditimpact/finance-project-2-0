import pytest

from backend.tradeline_check.f3_b01_post_closure_monthly_ok_detection import evaluate_f3_b01


def _payload(state_num: int = 2, declared_state: str | None = "closed") -> dict:
    return {
        "routing": {"R1": {"state_num": state_num}},
        "root_checks": {"Q1": {"declared_state": declared_state}},
    }


def _bureaus(months: list[dict[str, str]] | None):
    return {"two_year_payment_history_monthly_tsv_v2": {"equifax": months}}


def test_f3_b01_conflict_when_ok_after_closure():
    bureau_obj = {"closed_date": "2024-01-15"}
    monthly = [
        {"month_year_key": "2024-02", "status": "ok"},
        {"month_year_key": "2023-12", "status": "ok"},
    ]

    result = evaluate_f3_b01(
        bureau_obj=bureau_obj,
        bureaus_data=_bureaus(months=monthly),
        bureau="equifax",
        payload=_payload(state_num=2),
        placeholders=set(),
    )

    assert result["status"] == "conflict"
    assert result["eligible"] is True
    assert result["fired"] is True
    assert result["evidence"]["post_closure_ok_months"]


def test_f3_b01_ok_when_no_ok_after_closure():
    bureau_obj = {"closed_date": "2024-01-15"}
    monthly = [
        {"month_year_key": "2023-12", "status": "ok"},
        {"month_year_key": "2024-02", "status": "30"},
    ]

    result = evaluate_f3_b01(
        bureau_obj=bureau_obj,
        bureaus_data=_bureaus(months=monthly),
        bureau="equifax",
        payload=_payload(state_num=2),
        placeholders=set(),
    )

    assert result["status"] == "ok"
    assert result["eligible"] is True
    assert result["fired"] is False


def test_f3_b01_skipped_when_ineligible_state():
    bureau_obj = {"closed_date": "2024-01-15"}

    result = evaluate_f3_b01(
        bureau_obj=bureau_obj,
        bureaus_data=_bureaus(months=[{"month_year_key": "2024-02", "status": "ok"}]),
        bureau="equifax",
        payload=_payload(state_num=1),
        placeholders=set(),
    )

    assert result["status"] == "skipped"
    assert result["eligible"] is False
    assert result["executed"] is False


def test_f3_b01_skipped_missing_when_closed_date_missing():
    bureau_obj = {"closed_date": None}

    result = evaluate_f3_b01(
        bureau_obj=bureau_obj,
        bureaus_data=_bureaus(months=[{"month_year_key": "2024-02", "status": "ok"}]),
        bureau="equifax",
        payload=_payload(state_num=2),
        placeholders={"--"},
    )

    assert result["status"] == "skipped_missing_data"
    assert result["eligible"] is True
    assert result["executed"] is True
