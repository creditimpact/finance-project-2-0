from __future__ import annotations

"""Additional validations for the policy rulebook."""

from pathlib import Path
from typing import Any, Mapping, Set

import yaml

# Mismatch types produced by ``compute_mismatches`` in
# ``backend.core.logic.report_analysis.tri_merge``.
TRI_MERGE_MISMATCH_TYPES: Set[str] = {
    "presence",
    "balance",
    "status",
    "dates",
    "remarks",
    "utilization",
    "personal_info",
    "duplicate",
}


def validate_tri_merge_mismatch_rules(
    rulebook: Mapping[str, Any] | None = None,
    mismatch_types: Set[str] | None = None,
) -> None:
    """Ensure every mismatch type has a corresponding rule in the rulebook.

    ``mismatch_types`` defaults to :data:`TRI_MERGE_MISMATCH_TYPES`. If ``rulebook``
    is not provided, ``backend/policy/rulebook.yaml`` is loaded. A ``ValueError``
    is raised if any known mismatch type is missing.
    """

    if mismatch_types is None:
        mismatch_types = TRI_MERGE_MISMATCH_TYPES

    if rulebook is None:
        path = Path(__file__).with_name("rulebook.yaml")
        rulebook = yaml.safe_load(path.read_text(encoding="utf-8"))

    found: Set[str] = set()
    for rule in rulebook.get("rules", []):
        effect = rule.get("effect") or {}
        mismatch = effect.get("source_mismatch")
        if isinstance(mismatch, str) and mismatch.startswith("tri_merge."):
            found.add(mismatch.split(".", 1)[1])

    missing = mismatch_types - found
    if missing:
        raise ValueError(
            "Missing tri-merge rules for mismatch types: " + ", ".join(sorted(missing))
        )


__all__ = ["validate_tri_merge_mismatch_rules", "TRI_MERGE_MISMATCH_TYPES"]
