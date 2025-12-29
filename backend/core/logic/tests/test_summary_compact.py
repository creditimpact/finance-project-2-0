import copy

from backend.core.logic.summary_compact import compact_merge_sections


def test_compact_merge_sections_strips_noise_and_keeps_required():
    src = {
        "merge_scoring": {
            "best_with": 39,
            "score_total": 52,
            "reasons": ["total"],
            "conflicts": ["amount_conflict:balance_owed"],
            "identity_score": 36,
            "debt_score": 12,
            "acctnum_level": "exact_or_known_match",
            "matched_fields": {
                "account_number": 1,
                "balance_owed": 0,
                "last_payment": True,
            },
            "acctnum_digits_len_a": 5,
            "acctnum_digits_len_b": 5,
            "aux": {"junk": True},
        },
        "merge_explanations": [
            {
                "kind": "merge_pair",
                "with": 14,
                "decision": "ai",
                "total": 39,
                "parts": {"account_number": 28},
                "matched_fields": {
                    "account_number": 1,
                    "balance_owed": 0,
                },
                "reasons": ["total"],
                "conflicts": [],
                "strong": True,
                "acctnum_level": "exact_or_known_match",
                "acctnum_digits_len_a": 5,
                "acctnum_digits_len_b": 5,
                "aux": {"noise": "x"},
                "by_field_pairs": {"noise": "y"},
            }
        ],
        "something_else": {"keep": "untouched"},
    }

    out = compact_merge_sections(copy.deepcopy(src))

    ms = out["merge_scoring"]
    assert set(ms.keys()) == {
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
    assert isinstance(ms["matched_fields"]["account_number"], bool)

    me = out["merge_explanations"][0]
    assert set(me.keys()) == {
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
    assert isinstance(me["matched_fields"]["balance_owed"], bool)

    assert out["something_else"]["keep"] == "untouched"
