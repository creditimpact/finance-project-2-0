"""Branch results container initializer for tradeline_check outputs."""

from __future__ import annotations

from typing import Any, MutableMapping

BRANCH_RESULTS_VERSION = "branch_results_v1"


def ensure_branch_results_container(payload: MutableMapping[str, Any]) -> None:
    """Ensure payload has the branch_results container.

    If the key already exists, leave it untouched. Otherwise, attach a stable
    container for downstream branch writes.
    """

    if "branch_results" in payload:
        return

    payload["branch_results"] = {"version": BRANCH_RESULTS_VERSION, "results": {}}
