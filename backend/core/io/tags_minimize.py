"""Utilities for minimizing account tags after AI adjudication."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping, MutableMapping, Sequence

from backend.core.io.tags import read_tags, write_tags_atomic
from backend.core.logic.summary_compact import compact_merge_sections
from backend.core.merge.acctnum import normalize_level


def _coerce_int(value: Any) -> Any:
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, int):
        return value
    try:
        return int(str(value))
    except Exception:
        return value


def _coerce_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text else None


def _has_value(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return value != ""
    if isinstance(value, (list, tuple, set, dict)):
        return bool(value)
    return True


def _load_summary(summary_path: Path) -> dict[str, Any]:
    if not summary_path.exists():
        return {}
    try:
        data = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if isinstance(data, dict):
        return dict(data)
    return {}


def _coerce_mapping_list(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, (list, tuple)):
        items = value
    else:
        items = []
    result: list[dict[str, Any]] = []
    for entry in items:
        if isinstance(entry, Mapping):
            result.append(dict(entry))
    return result


def _merge_entries(
    existing: Iterable[Mapping[str, Any]],
    updates: Iterable[Mapping[str, Any]],
    key_fields: Sequence[str],
) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    index: dict[tuple[Any, ...], int] = {}

    for entry in existing:
        data = dict(entry)
        key = tuple(data.get(field) for field in key_fields)
        if key in index:
            merged[index[key]] = data
        else:
            index[key] = len(merged)
            merged.append(data)

    for entry in updates:
        data = dict(entry)
        key = tuple(data.get(field) for field in key_fields)
        if key in index:
            merged[index[key]] = data
        else:
            index[key] = len(merged)
            merged.append(data)

    return merged


def _minimal_issue(tag: Mapping[str, Any]) -> dict[str, Any]:
    minimal: dict[str, Any] = {"kind": "issue"}
    type_value = tag.get("type")
    if type_value is not None:
        minimal["type"] = str(type_value)
    return minimal


def _minimal_merge_best(tag: Mapping[str, Any]) -> dict[str, Any]:
    payload: dict[str, Any] = {"kind": "merge_best"}
    partner = _coerce_int(tag.get("with"))
    if partner is not None:
        payload["with"] = partner
    decision = _coerce_str(tag.get("decision"))
    if decision is not None:
        payload["decision"] = decision
    return payload


def _minimal_ai_decision(tag: Mapping[str, Any]) -> dict[str, Any]:
    payload: dict[str, Any] = {"kind": "ai_decision"}
    partner = _coerce_int(tag.get("with"))
    if partner is not None:
        payload["with"] = partner
    decision = _coerce_str(tag.get("decision"))
    if decision is not None:
        payload["decision"] = decision
    at_value = _coerce_str(tag.get("at"))
    if at_value is not None:
        payload["at"] = at_value
    return payload


def _minimal_same_debt(tag: Mapping[str, Any]) -> dict[str, Any]:
    payload: dict[str, Any] = {"kind": "same_debt_pair"}
    partner = _coerce_int(tag.get("with"))
    if partner is not None:
        payload["with"] = partner
    at_value = _coerce_str(tag.get("at"))
    if at_value is not None:
        payload["at"] = at_value
    return payload


def _minimal_other(tag: Mapping[str, Any]) -> dict[str, Any] | None:
    minimal: dict[str, Any] = {}
    ignored_fields = {
        "tag",
        "source",
        "parts",
        "aux",
        "reasons",
        "conflicts",
        "raw_response",
        "reason",
        "details",
        "score_total",
        "strong_rank",
        "tiebreaker",
        "total",
        "mid",
        "dates_all",
        "strong",
    }
    for key, value in tag.items():
        if key in ignored_fields:
            continue
        if key == "kind":
            minimal["kind"] = str(value)
        elif key == "with":
            partner = _coerce_int(value)
            if partner is not None:
                minimal["with"] = partner
        elif isinstance(value, (str, int, float, bool)) and value != "":
            minimal[key] = value
    if not minimal:
        return None
    minimal.setdefault("kind", str(tag.get("kind", "")))
    return {key: val for key, val in minimal.items() if val is not None}


def _merge_explanation_from_tag(tag: Mapping[str, Any]) -> dict[str, Any] | None:
    kind = str(tag.get("kind", ""))
    if kind not in {"merge_best", "merge_pair"}:
        return None

    payload: dict[str, Any] = {"kind": kind}
    partner = _coerce_int(tag.get("with"))
    if partner is not None:
        payload["with"] = partner
    decision = _coerce_str(tag.get("decision"))
    if decision is not None:
        payload["decision"] = decision

    verbose_fields: MutableMapping[str, Any] = {
        "total": tag.get("total"),
        "mid": tag.get("mid"),
        "dates_all": tag.get("dates_all"),
        "parts": tag.get("parts"),
        "aux": tag.get("aux"),
        "reasons": tag.get("reasons"),
        "conflicts": tag.get("conflicts"),
        "strong": tag.get("strong"),
        "strong_rank": tag.get("strong_rank"),
        "score_total": tag.get("score_total"),
        "tiebreaker": tag.get("tiebreaker"),
    }

    if not any(_has_value(value) for value in verbose_fields.values()):
        return None

    for key, value in verbose_fields.items():
        if _has_value(value):
            payload[key] = value

    aux = tag.get("aux")
    if isinstance(aux, Mapping):
        acct_level = normalize_level(aux.get("acctnum_level"))
        if acct_level != "none":
            payload.setdefault("acctnum_level", acct_level)
        matched_fields = aux.get("matched_fields")
        if isinstance(matched_fields, Mapping) and matched_fields:
            payload.setdefault("matched_fields", dict(matched_fields))

    return payload


def _ai_explanation_from_tag(
    tag: Mapping[str, Any],
    *,
    decision_reason_map: dict[Any, str],
) -> list[dict[str, Any]]:
    kind = str(tag.get("kind", ""))
    partner = _coerce_int(tag.get("with"))
    decision = _coerce_str(tag.get("decision"))
    reason = tag.get("reason")
    raw_response = tag.get("raw_response")
    explanations: list[dict[str, Any]] = []

    if kind == "ai_decision":
        if isinstance(partner, int) and isinstance(reason, str) and reason:
            decision_reason_map[partner] = reason
        if not _has_value(reason) and not _has_value(raw_response):
            return []
        payload: dict[str, Any] = {"kind": kind}
        if partner is not None:
            payload["with"] = partner
        if decision is not None:
            payload["decision"] = decision
        if _has_value(reason):
            payload["reason"] = reason
        if _has_value(raw_response):
            payload["raw_response"] = raw_response
        explanations.append(payload)
        return explanations

    if kind == "same_debt_pair":
        if not _has_value(reason) and isinstance(partner, int):
            reason = decision_reason_map.get(partner)
        if not _has_value(reason):
            return []
        payload = {"kind": kind}
        if partner is not None:
            payload["with"] = partner
        payload["reason"] = reason
        explanations.append(payload)

    return explanations


def compact_account_tags(account_dir: Path) -> None:
    """Rewrite ``tags.json`` to minimal tags and capture explanations."""

    account_path = Path(account_dir)
    tags = read_tags(account_path)
    if not tags:
        return

    minimal_tags: list[dict[str, Any]] = []
    merge_explanations: list[dict[str, Any]] = []
    ai_explanations: list[dict[str, Any]] = []
    reason_by_partner: dict[Any, str] = {}

    for tag in tags:
        if not isinstance(tag, Mapping):
            continue

        kind = str(tag.get("kind", ""))
        if kind == "issue":
            minimal_tags.append(_minimal_issue(tag))
        elif kind == "merge_best":
            minimal_tags.append(_minimal_merge_best(tag))
        elif kind == "ai_decision":
            minimal_tags.append(_minimal_ai_decision(tag))
        elif kind == "same_debt_pair":
            minimal_tags.append(_minimal_same_debt(tag))
        else:
            other = _minimal_other(tag)
            if other is not None and other.get("kind"):
                minimal_tags.append(other)

        merge_payload = _merge_explanation_from_tag(tag)
        if merge_payload is not None:
            merge_explanations.append(merge_payload)

        ai_payloads = _ai_explanation_from_tag(tag, decision_reason_map=reason_by_partner)
        if ai_payloads:
            ai_explanations.extend(ai_payloads)

    deduped_tags: list[dict[str, Any]] = []
    seen_signatures: set[str] = set()
    for tag in minimal_tags:
        signature = json.dumps(tag, sort_keys=True, separators=(",", ":"))
        if signature in seen_signatures:
            continue
        seen_signatures.add(signature)
        deduped_tags.append(tag)

    write_tags_atomic(account_path, deduped_tags)

    summary_path = account_path / "summary.json"
    summary_data = _load_summary(summary_path)

    existing_merge = _coerce_mapping_list(summary_data.get("merge_explanations"))
    existing_ai = _coerce_mapping_list(summary_data.get("ai_explanations"))

    if merge_explanations:
        summary_data["merge_explanations"] = _merge_entries(
            existing_merge, merge_explanations, ("kind", "with")
        )
    elif "merge_explanations" in summary_data:
        summary_data["merge_explanations"] = _merge_entries(
            existing_merge, (), ("kind", "with")
        )

    if ai_explanations:
        summary_data["ai_explanations"] = _merge_entries(
            existing_ai, ai_explanations, ("kind", "with", "decision")
        )
    elif "ai_explanations" in summary_data:
        summary_data["ai_explanations"] = _merge_entries(
            existing_ai, (), ("kind", "with", "decision")
        )

    if merge_explanations or ai_explanations or summary_path.exists():
        if os.getenv("COMPACT_MERGE_SUMMARY", "1") == "1":
            compact_merge_sections(summary_data)
        summary_path.write_text(
            json.dumps(summary_data, ensure_ascii=False, indent=2), encoding="utf-8"
        )


__all__ = ["compact_account_tags"]

