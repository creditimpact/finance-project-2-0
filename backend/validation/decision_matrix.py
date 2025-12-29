"""Decision matrix helpers for validation findings."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Mapping, Optional

import yaml

__all__ = ["decide_default"]


_MATRIX_PATH = Path(__file__).resolve().parents[1] / "core" / "logic" / "decision_matrix.yml"


def _normalize_field(field: str | None) -> str | None:
    if not field:
        return None
    normalized = field.strip()
    return normalized or None


def _normalize_reason_prefix(reason_code: str | None) -> str | None:
    if not reason_code:
        return None
    normalized = reason_code.strip().upper()
    if not normalized:
        return None
    prefix, _, _ = normalized.partition("_")
    return prefix or normalized


@lru_cache(maxsize=1)
def _load_matrix() -> Mapping[str, Mapping[str, str]]:
    try:
        raw = yaml.safe_load(_MATRIX_PATH.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}

    if not isinstance(raw, Mapping):
        return {}

    matrix = raw.get("matrix")
    if not isinstance(matrix, Mapping):
        return {}

    normalized_matrix: dict[str, dict[str, str]] = {}
    for raw_field, raw_decisions in matrix.items():
        if not isinstance(raw_field, str) or not isinstance(raw_decisions, Mapping):
            continue
        field_key = _normalize_field(raw_field)
        if not field_key:
            continue

        decisions: dict[str, str] = {}
        for raw_reason, raw_decision in raw_decisions.items():
            if not isinstance(raw_reason, str) or not isinstance(raw_decision, str):
                continue
            reason_key = _normalize_reason_prefix(raw_reason)
            decision_value = raw_decision.strip()
            if reason_key and decision_value:
                decisions[reason_key] = decision_value

        if decisions:
            normalized_matrix[field_key] = decisions

    return normalized_matrix


def decide_default(field: str, reason_code: str) -> Optional[str]:
    """Return the default decision for ``field`` and ``reason_code``."""

    field_key = _normalize_field(field)
    if not field_key:
        return None

    reason_key = _normalize_reason_prefix(reason_code)
    if not reason_key:
        return None

    decisions = _load_matrix().get(field_key)
    if not decisions:
        return None

    return decisions.get(reason_key)
