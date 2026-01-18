"""Central registry for branch eligibility and dispatch.

This module defines the single source of truth for branch family definitions,
eligibility rules, and evaluator function references. It enables strict
pre-invocation gating: branches are only invoked when eligible for the current
R1 router state.

Families F0 (always-run) and FX (always-run) are NOT in this registry; they
are handled separately in runner.py with unconditional invocation.

Families F1–F6 are defined here with their eligibility rules per R1.state_num.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping

# ─────────────────────────────────────────────────────────────────────────────
# REGISTRY: F1–F6 Branches with Eligibility Rules
# ─────────────────────────────────────────────────────────────────────────────
# Each entry defines:
#   - branch_id: unique identifier (e.g., "F2.B01")
#   - family_id: family identifier (e.g., "F2")
#   - eligible_states: set of R1.state_num values that allow this branch to run
#   - evaluator_path: "module.function" for lazy import
#   - evaluator_args: list of argument names to pass from kwargs


BRANCH_REGISTRY: list[dict[str, Any]] = [
    # ── F2: Payment Performance ──────────────────────────────────────────────
    {
        "branch_id": "F2.B01",
        "family_id": "F2",
        "name": "Activity vs Monthly History",
        "eligible_states": {1},  # Q1=open (Q1-only router)
        "evaluator_path": "backend.tradeline_check.f2_b01_activity_vs_monthly_history.evaluate_f2_b01",
        "evaluator_args": ["payload", "bureau_obj", "bureaus_data", "bureau", "placeholders"],
    },
    {
        "branch_id": "F2.B02",
        "family_id": "F2",
        "name": "Post-Closure Activity",
        "eligible_states": {2},  # Q1=closed (Q1-only router)
        "evaluator_path": "backend.tradeline_check.f2_b02_post_closure_activity.evaluate_f2_b02",
        "evaluator_args": ["bureau_obj", "payload", "placeholders"],
    },
    # ── F3: Timeline Deep Checks ─────────────────────────────────────────────
    {
        "branch_id": "F3.B01",
        "family_id": "F3",
        "name": "Post-Closure Monthly OK Detection",
        "eligible_states": {2, 3, 4},  # Closed + unknown + conflict (Q1-only router)
        "evaluator_path": "backend.tradeline_check.f3_b01_post_closure_monthly_ok_detection.evaluate_f3_b01",
        "evaluator_args": ["bureau_obj", "bureaus_data", "bureau", "payload", "placeholders"],
    },
    {
        "branch_id": "F3.B02",
        "family_id": "F3",
        "name": "Closed Date vs Monthly Coverage",
        "eligible_states": {2, 3, 4},  # Closed + unknown + conflict (Q1-only router)
        "evaluator_path": "backend.tradeline_check.f3_b02_closed_date_vs_monthly_coverage.evaluate_f3_b02",
        "evaluator_args": ["bureau_obj", "bureaus_data", "bureau", "payload", "placeholders"],
    },
    # ── Additional families (F1, F4, F5, F6) can be added here when implemented
]


def is_branch_eligible(branch_entry: Mapping[str, Any], r1_state_num: int | None) -> bool:
    """Check if a branch is eligible for the current R1 router state.

    Parameters
    ----------
    branch_entry : Mapping[str, Any]
        A registry entry containing 'eligible_states'
    r1_state_num : int | None
        Current R1.state_num; None if router result is missing

    Returns
    -------
    bool
        True if the branch should be invoked; False otherwise
    """
    if r1_state_num is None:
        return False
    eligible_states = branch_entry.get("eligible_states", set())
    return r1_state_num in eligible_states


def get_branch_by_id(branch_id: str) -> dict[str, Any] | None:
    """Look up a branch registry entry by branch_id.

    Parameters
    ----------
    branch_id : str
        Branch identifier (e.g., "F2.B01")

    Returns
    -------
    dict or None
        Registry entry if found; None otherwise
    """
    for entry in BRANCH_REGISTRY:
        if entry.get("branch_id") == branch_id:
            return entry
    return None


def invoke_branch_by_path(evaluator_path: str, args_dict: Mapping[str, Any], arg_names: list[str]) -> dict[str, Any]:
    """Dynamically import and invoke a branch evaluator.

    Parameters
    ----------
    evaluator_path : str
        Import path (e.g., "backend.tradeline_check.f2_b01_activity_vs_monthly_history.evaluate_f2_b01")
    args_dict : Mapping[str, Any]
        Dictionary containing all available arguments
    arg_names : list[str]
        Names of arguments to pass to the evaluator (in order)

    Returns
    -------
    dict
        Result from the evaluator function

    Raises
    ------
    ImportError
        If the module or function cannot be imported
    """
    parts = evaluator_path.rsplit(".", 1)
    if len(parts) != 2:
        raise ValueError(f"Invalid evaluator path: {evaluator_path}")

    module_name, func_name = parts

    # Dynamic import
    import importlib
    module = importlib.import_module(module_name)
    evaluator = getattr(module, func_name)

    # Build arguments in order
    args = [args_dict[arg_name] for arg_name in arg_names]

    return evaluator(*args)
