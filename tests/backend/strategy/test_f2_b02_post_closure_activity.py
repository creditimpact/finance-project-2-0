import pytest

from backend.tradeline_check.f2_b02_post_closure_activity import evaluate_f2_b02


def _payload(state_num: int = 2, declared_state: str | None = "closed") -> dict:
    return {
        "routing": {"R1": {"state_num": state_num}},
        "root_checks": {"Q1": {"declared_state": declared_state}},
    }


def test_f2_b02_conflict_when_activity_after_closed_date():
    bureau_obj = {
        "closed_date": "2024-01-01",
        "date_of_last_activity": "2024-02-01",
    }

    result = evaluate_f2_b02(bureau_obj=bureau_obj, payload=_payload(), placeholders=set())

    assert result["status"] == "conflict"
    assert result["eligible"] is True
    assert result["fired"] is True
    assert result["evidence"]["violating_field"] == "date_of_last_activity"


def test_f2_b02_ok_when_activity_before_closed_date():
    bureau_obj = {
        "closed_date": "2024-01-15",
        "date_of_last_activity": "2023-12-01",
    }

    result = evaluate_f2_b02(bureau_obj=bureau_obj, payload=_payload(), placeholders=set())

    assert result["status"] == "ok"
    assert result["eligible"] is True
    assert result["fired"] is False
    assert "checked_fields" in result.get("evidence", {})


def test_f2_b02_skipped_when_not_eligible_state():
    bureau_obj = {"closed_date": "2024-01-01"}

    result = evaluate_f2_b02(bureau_obj=bureau_obj, payload=_payload(state_num=1), placeholders=set())

    assert result["status"] == "skipped"
    assert result["eligible"] is False
    assert result["executed"] is False


def test_f2_b02_skipped_missing_when_closed_date_missing():
    bureau_obj = {"closed_date": None}

    result = evaluate_f2_b02(bureau_obj=bureau_obj, payload=_payload(), placeholders={"--"})

    assert result["status"] == "skipped_missing_data"
    assert result["eligible"] is True
    assert result["executed"] is True
