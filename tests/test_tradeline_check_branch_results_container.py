from copy import deepcopy

from backend.tradeline_check.branch_results import (
    BRANCH_RESULTS_VERSION,
    ensure_branch_results_container,
)


def test_container_present_and_empty():
    payload = {}

    ensure_branch_results_container(payload)

    assert "branch_results" in payload
    assert payload["branch_results"]["version"] == BRANCH_RESULTS_VERSION
    assert payload["branch_results"]["results"] == {}


def test_idempotent_does_not_overwrite_existing_results():
    existing_results = {"F2.B01": {"status": "ok"}}
    payload = {"branch_results": {"version": BRANCH_RESULTS_VERSION, "results": existing_results}}

    ensure_branch_results_container(payload)

    assert payload["branch_results"]["results"] is existing_results
    assert payload["branch_results"]["version"] == BRANCH_RESULTS_VERSION


def test_does_not_mutate_other_keys():
    payload = {
        "root_checks": {"Q1": {"status": "ok"}},
        "routing": {"R1": {"state_num": 3}},
        "branches": {"version": "branches_v1"},
    }
    before = deepcopy(payload)

    ensure_branch_results_container(payload)

    assert payload["root_checks"] == before["root_checks"]
    assert payload["routing"] == before["routing"]
    assert payload["branches"] == before["branches"]
    assert payload["branch_results"]["version"] == BRANCH_RESULTS_VERSION
    assert payload["branch_results"]["results"] == {}
