import copy

import pytest

from ..logic.summary_compact import compact_merge_sections


_ALLOWED_MERGE_SCORING = {
    "best_with",
    "score_total",
    "reasons",
    "conflicts",
    "identity_score",
    "debt_score",
    "acctnum_level",
    "matched_fields",
    "acctnum_digits_len_a",
    "acctnum_digits_len_b",
}

_ALLOWED_MERGE_EXPLANATION = {
    "kind",
    "with",
    "decision",
    "total",
    "parts",
    "matched_fields",
    "reasons",
    "conflicts",
    "strong",
    "acctnum_level",
    "acctnum_digits_len_a",
    "acctnum_digits_len_b",
}

_BANNED_KEYS = {
    "aux",
    "by_field_pairs",
    "matched_pairs",
    "tiebreaker",
    "strong_rank",
    "dates_all",
    "mid",
}


@pytest.mark.parametrize(
    "matched_fields_input",
    [
        {"account_number": 1, "balance_owed": 0, "last_payment": True},
        {"account_number": "1", "balance_owed": None, "last_payment": "false"},
    ],
)
def test_compact_merge_sections_scrubs_merge_sections_and_banned_keys(matched_fields_input):
    summary = {
        "merge_scoring": {
            "best_with": "39",
            "score_total": 52.8,
            "reasons": ["total"],
            "conflicts": ("amount_conflict:balance_owed",),
            "identity_score": "36",
            "debt_score": 12,
            "acctnum_level": "exact_or_known_match",
            "matched_fields": matched_fields_input,
            "acctnum_digits_len_a": "5",
            "acctnum_digits_len_b": 5.0,
            "aux": {"junk": True},
            "tiebreaker": 99,
        },
        "merge_explanations": [
            {
                "kind": "merge_pair",
                "with": "14",
                "decision": "ai",
                "total": "39",
                "parts": {"account_number": "28", "other": "preserved"},
                "matched_fields": matched_fields_input,
                "reasons": ["total"],
                "conflicts": [],
                "strong": "yes",
                "acctnum_level": "exact_or_known_match",
                "acctnum_digits_len_a": "5",
                "acctnum_digits_len_b": 5,
                "aux": {"noise": "x"},
                "by_field_pairs": {"noise": "y"},
                "strong_rank": 1,
                "mid": "123",
            }
        ],
        "other": {
            "deep": {"aux": "remove", "list": [{"mid": 3}, 2]},
            "dates_all": ["2020-01-01"],
        },
    }

    original = copy.deepcopy(summary)
    compacted = compact_merge_sections(summary)

    # ensure function mutates the input mapping in-place but returns same object
    assert compacted is summary

    merge_scoring = compacted["merge_scoring"]
    expected_scoring_keys = _ALLOWED_MERGE_SCORING
    if "identity_score" not in merge_scoring:
        expected_scoring_keys = expected_scoring_keys - {"identity_score"}
    assert set(merge_scoring) == expected_scoring_keys
    assert merge_scoring["best_with"] == 39
    assert merge_scoring["score_total"] == 52
    assert merge_scoring["reasons"] == ["total"]
    assert merge_scoring["conflicts"] == ["amount_conflict:balance_owed"]
    assert merge_scoring["debt_score"] == 12
    assert merge_scoring["acctnum_level"] == "exact_or_known_match"
    assert merge_scoring["acctnum_digits_len_a"] == 5
    assert merge_scoring["acctnum_digits_len_b"] == 5
    if "identity_score" in merge_scoring:
        assert merge_scoring["identity_score"] == 36

    matched_fields = merge_scoring["matched_fields"]
    assert all(isinstance(value, bool) for value in matched_fields.values())
    assert matched_fields["account_number"] is True
    assert matched_fields["balance_owed"] is False
    if "last_payment" in matched_fields:
        if matched_fields_input.get("last_payment") == "false":
            assert matched_fields["last_payment"] is False
        elif matched_fields_input.get("last_payment") is True:
            assert matched_fields["last_payment"] is True

    merge_explanations = compacted["merge_explanations"]
    assert isinstance(merge_explanations, list)
    assert len(merge_explanations) == 1

    explanation = merge_explanations[0]
    expected_explanation_keys = _ALLOWED_MERGE_EXPLANATION
    if "strong" not in explanation:
        expected_explanation_keys = expected_explanation_keys - {"strong"}
    assert set(explanation) == expected_explanation_keys
    assert explanation["kind"] == "merge_pair"
    assert explanation["with"] == 14
    assert explanation["decision"] == "ai"
    assert explanation["total"] == 39
    assert explanation["parts"] == {"account_number": 28, "other": "preserved"}
    assert explanation["reasons"] == ["total"]
    assert explanation["conflicts"] == []
    if "strong" in explanation:
        assert explanation["strong"] is True
    assert explanation["acctnum_level"] == "exact_or_known_match"
    assert explanation["acctnum_digits_len_a"] == 5
    assert explanation["acctnum_digits_len_b"] == 5
    assert all(isinstance(value, bool) for value in explanation["matched_fields"].values())
    assert explanation["matched_fields"]["account_number"] is True
    assert explanation["matched_fields"]["balance_owed"] is False
    if "last_payment" in explanation["matched_fields"]:
        if matched_fields_input.get("last_payment") == "false":
            assert explanation["matched_fields"]["last_payment"] is False
        elif matched_fields_input.get("last_payment") is True:
            assert explanation["matched_fields"]["last_payment"] is True

    def assert_no_banned_keys(value):
        if isinstance(value, dict):
            for key, item in value.items():
                assert key not in _BANNED_KEYS
                assert_no_banned_keys(item)
        elif isinstance(value, list):
            for item in value:
                assert_no_banned_keys(item)

    assert_no_banned_keys(compacted)

    # ensure unrelated content is preserved besides banned keys being removed
    assert compacted["other"]["deep"]["list"] == [{}, 2]
    assert "dates_all" not in compacted["other"]

    # original object should remain unchanged for reference
    assert original["other"]["deep"]["list"][0]["mid"] == 3
