"""Canonical branch family catalog for tradeline_check.

Defines the fixed, deterministic ordering of branch families used for visibility
and reporting. This is the single source of truth for family IDs and names.
"""
from __future__ import annotations

BRANCH_FAMILIES: list[dict[str, str]] = [
    {"family_id": "F0", "name": "Universal Integrity Context"},
    {"family_id": "F1", "name": "Core Story Alignment"},
    {"family_id": "F2", "name": "Payment Performance"},
    {"family_id": "F3", "name": "Timeline Deep Checks"},
    {"family_id": "F4", "name": "Amounts & Balances Sanity"},
    {"family_id": "F5", "name": "Type-Conditioned Checks"},
    {"family_id": "F6", "name": "Disputes & Remarks Signals"},
    {"family_id": "FX", "name": "Always-Run Behavioral Signals"},
]
