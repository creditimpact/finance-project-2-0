from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from copy import deepcopy
from typing import Any


_MERGE_SCORING_ALLOWED = (
    "best_with",
    "score_total",
    "reasons",
    "conflicts",
    "identity_score",
    "debt_score",
    "acctnum_level",
    "matched_fields",
    "acctnum_digits_len_a",
    "acctnum_digits_len_b",
    "points_mode",
)

_MERGE_EXPLANATION_ALLOWED = (
    "kind",
    "with",
    "decision",
    "total",
    "parts",
    "matched_fields",
    "reasons",
    "conflicts",
    "strong",
    "acctnum_level",
    "acctnum_digits_len_a",
    "acctnum_digits_len_b",
)

_BANNED_KEYS = {
    "aux",
    "by_field_pairs",
    "matched_pairs",
    "tiebreaker",
    "strong_rank",
    "dates_all",
    "mid",
}


def _coerce_bool(value: Any) -> bool:
    """Return ``value`` represented as a boolean."""

    if isinstance(value, bool):
        return value

    if isinstance(value, (int, float)):
        return bool(value)

    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"", "0", "false", "f", "no", "n", "off"}:
            return False
        if lowered in {"1", "true", "t", "yes", "y", "on"}:
            return True

    if isinstance(value, (list, tuple, set)):
        return bool(value)

    if isinstance(value, Mapping):
        return bool(value)

    return bool(value)


def _ensure_bool_mapping(value: Any) -> dict[str, bool]:
    """Return a mapping containing only boolean values."""

    if isinstance(value, Mapping):
        return {str(key): _coerce_bool(val) for key, val in value.items()}
    return {}


def _coerce_int(value: Any) -> int | None:
    """Best-effort conversion of ``value`` to an ``int``."""

    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    try:
        text = str(value).strip()
    except Exception:
        return None
    if not text:
        return None
    try:
        return int(text)
    except ValueError:
        try:
            return int(float(text))
        except ValueError:
            return None


def _coerce_str(value: Any) -> str | None:
    """Best-effort conversion of ``value`` to a trimmed string."""

    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _coerce_str_list(value: Any) -> list[str]:
    """Coerce ``value`` into a list of non-empty strings."""

    if value is None:
        return []
    if isinstance(value, Mapping):
        iterable: Iterable[Any] = value.values()
    elif isinstance(value, set):
        iterable = sorted(value, key=lambda item: str(item))
    elif isinstance(value, (list, tuple)):
        iterable = value
    else:
        iterable = (value,)

    result: list[str] = []
    for item in iterable:
        text = _coerce_str(item)
        if text is not None:
            result.append(text)
    return result


def _normalize_parts(value: Any) -> dict[str, Any]:
    """Return a mapping of parts with numeric values when possible."""

    if not isinstance(value, Mapping):
        return {}

    normalized: dict[str, Any] = {}
    for key, item in value.items():
        coerced = _coerce_int(item)
        if coerced is not None:
            normalized[str(key)] = coerced
        elif item not in (None, ""):
            normalized[str(key)] = deepcopy(item)
    return normalized


def _normalize_merge_scoring(source: Mapping[str, Any] | None) -> dict[str, Any]:
    """Return a compact merge scoring payload."""

    if not isinstance(source, Mapping):
        return {}

    normalized: dict[str, Any] = {}

    partner = _coerce_int(source.get("best_with"))
    if partner is not None:
        normalized["best_with"] = partner

    points_mode_active = bool(source.get("points_mode"))
    if points_mode_active:
        try:
            score_total = float(source.get("score_total"))
        except (TypeError, ValueError):
            score_total = None
        if score_total is not None:
            normalized["score_total"] = score_total
            normalized["points_mode"] = True
    else:
        score_total = _coerce_int(source.get("score_total"))
        if score_total is not None:
            normalized["score_total"] = score_total

    reasons_raw = source.get("reasons")
    reasons = _coerce_str_list(reasons_raw)
    if reasons or reasons_raw is not None:
        normalized["reasons"] = reasons

    conflicts_raw = source.get("conflicts")
    conflicts = _coerce_str_list(conflicts_raw)
    if conflicts or conflicts_raw is not None:
        normalized["conflicts"] = conflicts

    if points_mode_active:
        try:
            identity_score = float(source.get("identity_score"))
        except (TypeError, ValueError):
            identity_score = None
        if identity_score is not None:
            normalized["identity_score"] = identity_score
        try:
            debt_score = float(source.get("debt_score"))
        except (TypeError, ValueError):
            debt_score = None
        if debt_score is not None:
            normalized["debt_score"] = debt_score
    else:
        identity_score = _coerce_int(source.get("identity_score"))
        if identity_score is not None:
            normalized["identity_score"] = identity_score

        debt_score = _coerce_int(source.get("debt_score"))
        if debt_score is not None:
            normalized["debt_score"] = debt_score

    acctnum_level = _coerce_str(source.get("acctnum_level"))
    if acctnum_level is not None:
        normalized["acctnum_level"] = acctnum_level

    normalized["matched_fields"] = _ensure_bool_mapping(source.get("matched_fields"))

    for key in ("acctnum_digits_len_a", "acctnum_digits_len_b"):
        digits = _coerce_int(source.get(key))
        if digits is not None:
            normalized[key] = digits

    return {
        key: normalized[key]
        for key in _MERGE_SCORING_ALLOWED
        if key in normalized
    }


def _normalize_merge_explanations(
    entries: Sequence[Any] | None,
) -> list[dict[str, Any]]:
    """Return compact merge explanation entries."""

    if not isinstance(entries, Sequence) or isinstance(entries, (str, bytes, bytearray)):
        return []

    normalized_entries: list[dict[str, Any]] = []
    for entry in entries:
        if not isinstance(entry, Mapping):
            continue

        normalized_entry: dict[str, Any] = {}

        kind = _coerce_str(entry.get("kind"))
        if kind is not None:
            normalized_entry["kind"] = kind

        partner = _coerce_int(entry.get("with"))
        if partner is not None:
            normalized_entry["with"] = partner

        decision = _coerce_str(entry.get("decision"))
        if decision is not None:
            normalized_entry["decision"] = decision

        total = _coerce_int(entry.get("total"))
        if total is None:
            total = _coerce_int(entry.get("score_total"))
        if total is not None:
            normalized_entry["total"] = total

        parts_raw = entry.get("parts")
        parts = _normalize_parts(parts_raw)
        if parts or isinstance(parts_raw, Mapping):
            normalized_entry["parts"] = parts

        matched_fields_raw = entry.get("matched_fields")
        matched_fields = _ensure_bool_mapping(matched_fields_raw)
        if matched_fields or isinstance(matched_fields_raw, Mapping):
            normalized_entry["matched_fields"] = matched_fields

        reasons_raw = entry.get("reasons")
        reasons = _coerce_str_list(reasons_raw)
        if reasons or reasons_raw is not None:
            normalized_entry["reasons"] = reasons

        conflicts_raw = entry.get("conflicts")
        conflicts = _coerce_str_list(conflicts_raw)
        if conflicts or conflicts_raw is not None:
            normalized_entry["conflicts"] = conflicts

        if "strong" in entry:
            normalized_entry["strong"] = bool(entry.get("strong"))

        acctnum_level = _coerce_str(entry.get("acctnum_level"))
        if acctnum_level is not None:
            normalized_entry["acctnum_level"] = acctnum_level

        for key in ("acctnum_digits_len_a", "acctnum_digits_len_b"):
            digits = _coerce_int(entry.get(key))
            if digits is not None:
                normalized_entry[key] = digits

        if normalized_entry:
            filtered_entry = {
                key: normalized_entry[key]
                for key in _MERGE_EXPLANATION_ALLOWED
                if key in normalized_entry
            }
            if filtered_entry:
                normalized_entries.append(filtered_entry)

    return normalized_entries


def _scrub_banned(value: Any) -> Any:
    """Recursively remove banned keys from dictionaries."""

    if isinstance(value, Mapping):
        cleaned: dict[str, Any] = {}
        for key, item in value.items():
            if key in _BANNED_KEYS:
                continue
            cleaned[key] = _scrub_banned(item)
        return cleaned

    if isinstance(value, list):
        return [_scrub_banned(item) for item in value]

    if isinstance(value, tuple):
        return tuple(_scrub_banned(item) for item in value)

    return value


def compact_merge_sections(summary: dict[str, Any]) -> dict[str, Any]:
    """Compact merge sections and scrub banned keys from a summary payload."""

    merge_scoring_payload = summary.get("merge_scoring")
    normalized_scoring = _normalize_merge_scoring(
        merge_scoring_payload if isinstance(merge_scoring_payload, Mapping) else None
    )
    if normalized_scoring:
        summary["merge_scoring"] = normalized_scoring
    elif "merge_scoring" in summary:
        summary.pop("merge_scoring", None)

    merge_explanations_payload = summary.get("merge_explanations")
    normalized_explanations = _normalize_merge_explanations(merge_explanations_payload)
    if normalized_explanations:
        summary["merge_explanations"] = normalized_explanations
    elif "merge_explanations" in summary:
        summary.pop("merge_explanations", None)

    scrubbed = _scrub_banned(summary)
    summary.clear()
    summary.update(scrubbed)
    return summary
