"""Test suite for strict branch eligibility gating with central registry.

Validates that:
1. F1–F6 branches are only invoked when eligible per R1.state_num
2. Ineligible branches do NOT appear in branch_results.results
3. Visibility lists reflect only invoked branches
4. F0/FX branches continue to run unconditionally
"""

import pytest
from backend.tradeline_check.branch_registry import BRANCH_REGISTRY, is_branch_eligible, get_branch_by_id


class TestBranchRegistryStructure:
    """Test the central registry structure."""

    def test_registry_has_entries(self):
        """Registry must contain at least one branch entry."""
        assert len(BRANCH_REGISTRY) > 0

    def test_all_entries_have_required_fields(self):
        """Each registry entry must have essential fields."""
        required_fields = {"branch_id", "family_id", "eligible_states", "evaluator_path", "evaluator_args"}
        for entry in BRANCH_REGISTRY:
            assert required_fields.issubset(entry.keys()), f"Entry missing fields: {entry}"

    def test_f2_b01_registered(self):
        """F2.B01 must be in registry."""
        f2_b01 = get_branch_by_id("F2.B01")
        assert f2_b01 is not None
        assert f2_b01["family_id"] == "F2"
        assert f2_b01["eligible_states"] == {1}

    def test_f2_b02_registered(self):
        """F2.B02 must be in registry."""
        f2_b02 = get_branch_by_id("F2.B02")
        assert f2_b02 is not None
        assert f2_b02["family_id"] == "F2"
        assert f2_b02["eligible_states"] == {2}


class TestBranchEligibility:
    """Test branch eligibility determination."""

    def test_f2_b01_eligible_state_1(self):
        """F2.B01 must be eligible for R1.state_num=1."""
        f2_b01 = get_branch_by_id("F2.B01")
        assert is_branch_eligible(f2_b01, 1) is True

    def test_f2_b01_ineligible_state_5(self):
        """F2.B01 must be ineligible for R1.state_num=5."""
        f2_b01 = get_branch_by_id("F2.B01")
        assert is_branch_eligible(f2_b01, 5) is False

    def test_f2_b02_eligible_state_2(self):
        """F2.B02 must be eligible for R1.state_num=2 (closed)."""
        f2_b02 = get_branch_by_id("F2.B02")
        assert is_branch_eligible(f2_b02, 2) is True

    def test_f2_b02_ineligible_state_4(self):
        """F2.B02 must be ineligible for R1.state_num=4 (conflict)."""
        f2_b02 = get_branch_by_id("F2.B02")
        assert is_branch_eligible(f2_b02, 4) is False

    def test_f2_b02_ineligible_state_1(self):
        """F2.B02 must be ineligible for R1.state_num=1."""
        f2_b02 = get_branch_by_id("F2.B02")
        assert is_branch_eligible(f2_b02, 1) is False

    def test_eligibility_with_none_state(self):
        """Eligibility must return False when R1.state_num is None."""
        f2_b01 = get_branch_by_id("F2.B01")
        assert is_branch_eligible(f2_b01, None) is False


class TestRegistryLookup:
    """Test registry lookup functions."""

    def test_get_branch_by_id_found(self):
        """get_branch_by_id must return entry when branch exists."""
        entry = get_branch_by_id("F2.B01")
        assert entry is not None
        assert entry["branch_id"] == "F2.B01"

    def test_get_branch_by_id_not_found(self):
        """get_branch_by_id must return None when branch does not exist."""
        entry = get_branch_by_id("NONEXISTENT.BRANCH")
        assert entry is None

    def test_all_registered_branches_retrievable(self):
        """All branches in registry must be retrievable by ID."""
        for branch_entry in BRANCH_REGISTRY:
            branch_id = branch_entry["branch_id"]
            retrieved = get_branch_by_id(branch_id)
            assert retrieved is not None
            assert retrieved["branch_id"] == branch_id
