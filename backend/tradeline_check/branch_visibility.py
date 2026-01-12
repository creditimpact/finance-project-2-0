"""Always-on branch families visibility block builder.

This module builds the branches/branching block that is persisted in each
per-bureau tradeline_check output. It is non-blocking and does not mutate the
input payload.
"""
from __future__ import annotations

from typing import Any, Mapping, MutableMapping

from backend.tradeline_check.branch_families import BRANCH_FAMILIES


def build_branches_block(payload: Mapping[str, Any]) -> dict:
    """Build the always-on branches visibility block.

    Parameters
    ----------
    payload: Mapping[str, Any]
        The current bureau payload (read-only usage). Must not be mutated.

    Returns
    -------
    dict
        Branches visibility block with families and summary.
    """
    routing = payload.get("routing", {}) if isinstance(payload, Mapping) else {}
    r1 = routing.get("R1", {}) if isinstance(routing, Mapping) else {}
    r1_state_num = r1.get("state_num", 0) if isinstance(r1, Mapping) else 0

    families = []
    for fam in BRANCH_FAMILIES:
        # Copy the canonical fields; lists are empty in v1
        families.append(
            {
                "family_id": fam.get("family_id"),
                "name": fam.get("name"),
                "eligible_branch_ids": [],
                "executed_branch_ids": [],
                "fired_branch_ids": [],
            }
        )

    summary = {
        "total_families": len(families),
        "total_eligible_branches": 0,
        "total_executed_branches": 0,
        "total_fired_branches": 0,
    }

    return {
        "version": "branches_v1",
        "r1_state_num": r1_state_num if r1_state_num is not None else 0,
        "families": families,
        "summary": summary,
    }


def update_branches_visibility(payload: MutableMapping[str, Any]) -> None:
    """Update branches visibility lists based on branch_results.

    Populates eligible/executed/fired lists for each branch and updates
    summary counts. This function mutates payload in-place (branches block only).

    Parameters
    ----------
    payload : MutableMapping[str, Any]
        The bureau payload (mutable). Must contain both branches and branch_results blocks.
    """
    branches = payload.get("branches")
    branch_results = payload.get("branch_results")

    if not isinstance(branches, dict):
        return
    if not isinstance(branch_results, dict):
        return

    families = branches.get("families")
    results_dict = branch_results.get("results")

    if not isinstance(families, list):
        return
    if not isinstance(results_dict, dict):
        return

    # Build mapping: branch_id -> result
    branch_id_to_result = {}
    for branch_id, result in results_dict.items():
        if isinstance(result, dict):
            branch_id_to_result[branch_id] = result

    # For each branch result, update corresponding family lists
    for branch_id, result in branch_id_to_result.items():
        # Parse branch_id to extract family_id (e.g., "F2.B01" -> family_id="F2")
        parts = str(branch_id).split(".")
        if len(parts) < 1:
            continue
        family_id = parts[0]

        # Find family in families list
        family_obj = None
        for fam in families:
            if isinstance(fam, dict) and fam.get("family_id") == family_id:
                family_obj = fam
                break

        if family_obj is None:
            continue

        # Update lists
        # FX branches are always ungated (always eligible/executed when run)
        is_ungated = result.get("ungated", False)
        
        eligible = result.get("eligible", False) or is_ungated
        executed = result.get("executed", False)
        fired = result.get("fired", False)

        if eligible:
            if branch_id not in family_obj.get("eligible_branch_ids", []):
                family_obj.setdefault("eligible_branch_ids", []).append(branch_id)

        if executed:
            if branch_id not in family_obj.get("executed_branch_ids", []):
                family_obj.setdefault("executed_branch_ids", []).append(branch_id)

        if fired:
            if branch_id not in family_obj.get("fired_branch_ids", []):
                family_obj.setdefault("fired_branch_ids", []).append(branch_id)

    # Update summary counts
    summary = branches.get("summary", {})
    if isinstance(summary, dict):
        total_eligible = 0
        total_executed = 0
        total_fired = 0

        for fam in families:
            if isinstance(fam, dict):
                total_eligible += len(fam.get("eligible_branch_ids", []))
                total_executed += len(fam.get("executed_branch_ids", []))
                total_fired += len(fam.get("fired_branch_ids", []))

        summary["total_eligible_branches"] = total_eligible
        summary["total_executed_branches"] = total_executed
        summary["total_fired_branches"] = total_fired
