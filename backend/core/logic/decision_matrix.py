"""Static decision matrix for validation findings."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Mapping

import yaml

__all__ = ["get_decision_matrix", "lookup_decision"]


_MATRIX_PATH = Path(__file__).with_name("decision_matrix.yml")


def _normalize_reason_code(reason_code: str | None) -> str | None:
    """Return the normalized reason code prefix (e.g., ``"C3"``)."""

    if not reason_code:
        return None

    normalized = reason_code.strip().upper()
    if not normalized:
        return None

    prefix, _, _ = normalized.partition("_")
    return prefix or normalized


@lru_cache(maxsize=1)
def get_decision_matrix() -> Mapping[str, Mapping[str, str]]:
    """Load the decision matrix from :mod:`decision_matrix.yml`."""

    try:
        raw_text = _MATRIX_PATH.read_text(encoding="utf-8")
    except FileNotFoundError:
        return {}

    loaded = yaml.safe_load(raw_text) or {}
    if not isinstance(loaded, Mapping):
        return {}

    matrix = loaded.get("matrix")
    if not isinstance(matrix, Mapping):
        return {}

    normalized_matrix: dict[str, dict[str, str]] = {}

    for field_name, decisions in matrix.items():
        if not isinstance(field_name, str) or not isinstance(decisions, Mapping):
            continue

        normalized_field = field_name.strip()
        if not normalized_field:
            continue

        normalized_decisions: dict[str, str] = {}
        for reason_code, decision in decisions.items():
            if not isinstance(reason_code, str) or not isinstance(decision, str):
                continue
            normalized_reason = reason_code.strip().upper()
            normalized_decision = decision.strip()
            if normalized_reason and normalized_decision:
                normalized_decisions[normalized_reason] = normalized_decision

        if normalized_decisions:
            normalized_matrix[normalized_field] = normalized_decisions

    return normalized_matrix


def lookup_decision(field: str | None, reason_code: str | None) -> str | None:
    """Return the deterministic decision for ``field`` and ``reason_code``."""

    if not field:
        return None

    normalized_field = field.strip()
    if not normalized_field:
        return None

    normalized_reason = _normalize_reason_code(reason_code)
    if not normalized_reason:
        return None

    matrix = get_decision_matrix()
    field_map = matrix.get(normalized_field)
    if not field_map:
        return None

    return field_map.get(normalized_reason)
