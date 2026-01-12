from __future__ import annotations

import copy

from backend.tradeline_check.branch_families import BRANCH_FAMILIES
from backend.tradeline_check.branch_visibility import build_branches_block


def test_families_always_present_and_ordered():
    payload = {}
    block = build_branches_block(payload)

    assert block["version"] == "branches_v1"
    assert block["r1_state_num"] == 0

    families = block["families"]
    assert len(families) == 6
    assert [f["family_id"] for f in families] == [f["family_id"] for f in BRANCH_FAMILIES]

    for fam in families:
        assert fam["eligible_branch_ids"] == []
        assert fam["executed_branch_ids"] == []
        assert fam["fired_branch_ids"] == []

    summary = block["summary"]
    assert summary["total_families"] == 6
    assert summary["total_eligible_branches"] == 0
    assert summary["total_executed_branches"] == 0
    assert summary["total_fired_branches"] == 0


def test_r1_state_num_copied_when_present():
    payload = {"routing": {"R1": {"state_num": 5}}}
    block = build_branches_block(payload)
    assert block["r1_state_num"] == 5


def test_r1_state_num_defaults_to_zero_when_missing():
    payload = {"routing": {}}
    block = build_branches_block(payload)
    assert block["r1_state_num"] == 0


def test_input_not_mutated():
    payload = {"routing": {"R1": {"state_num": 3}}}
    original = copy.deepcopy(payload)
    _ = build_branches_block(payload)
    assert payload == original
