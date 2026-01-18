import copy

from backend.tradeline_check.runner import project_public_payload


def test_project_public_payload_filters_debug_fields_without_mutation():
    original = {
        "schema_version": 1,
        "generated_at": "2024-01-01T00:00:00Z",
        "account_key": "acct-1",
        "bureau": "equifax",
        "status": "ok",
        "date_convention": {
            "convention": "MDY",
            "extra": "keep-internal",
        },
        "coverage": {"placeholders": ["--"], "missing_core_fields": {"Q1": []}},
        "root_checks": {
            "Q1": {
                "declared_state": "open",
                "status": "ok",
                "explanation": "q1 explanation",
                "signals": ["OPEN"],
            },
            # Root checks now only expose Q1
        },
        "routing": {
            "R1": {
                "version": "r1_router_v3",
                "state_id": "S_open",
                "state_num": 1,
                "state_key": {"q1_declared_state": "open"},
                "inputs": {"Q1.declared_state": "open"},
                "explanation": "R1 derived from Q1.declared_state",
            },
        },
        "record_integrity": {"F0": {"A01": {"status": "ok"}}},
    }

    original_snapshot = copy.deepcopy(original)

    public_view = project_public_payload(original)

    # Original payload is untouched
    assert original == original_snapshot

    # Removed debug blocks
    assert "coverage" not in public_view

    # date_convention collapsed to string
    assert public_view.get("date_convention") == "MDY"

    # root_checks reduced to allowed keys
    q1_public = public_view.get("root_checks", {}).get("Q1", {})
    assert set(q1_public.keys()) <= {"declared_state", "status", "explanation"}
    assert q1_public.get("declared_state") == "open"
    assert "signals" not in q1_public

    assert set(public_view.get("root_checks", {}).keys()) == {"Q1"}

    # routing.R1 reduced to allowed keys
    r1_public = public_view.get("routing", {}).get("R1", {})
    assert set(r1_public.keys()) <= {"version", "state_id", "state_num"}
    assert r1_public.get("state_id") == "S_open"
    assert r1_public.get("state_num") == 1
    assert "state_key" not in r1_public
    assert "inputs" not in r1_public
    assert "explanation" not in r1_public

    # Unrelated top-level data is preserved
    assert public_view.get("record_integrity", {}).get("F0", {}).get("A01", {}).get("status") == "ok"
    assert public_view.get("schema_version") == 1
