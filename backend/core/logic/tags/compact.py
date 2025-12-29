"""Canonical tag compaction helpers for AI adjudication artifacts."""

from __future__ import annotations

import json
import logging
import os
from os import PathLike
from pathlib import Path
from typing import Any, Iterable, Mapping, MutableMapping, Sequence, Union

from backend.core.io.json_io import _atomic_write_json
from backend.core.logic.report_analysis.account_merge import (
    MergeDecision,
    detect_points_mode_from_payload,
    merge_summary_sections,
    normalize_parts_for_serialization,
    resolve_identity_debt_fields,
)
from backend.core.logic.summary_compact import compact_merge_sections
from backend.core.merge.acctnum import normalize_level

Pathish = Union[str, Path, PathLike[str]]


log = logging.getLogger(__name__)


def _maybe_slice(iterable: Iterable[Path]) -> Iterable[Path]:
    """Return ``iterable`` unchanged so tag compaction covers all accounts."""

    debug_first_n = os.getenv("DEBUG_FIRST_N", "").strip()
    if debug_first_n:
        log.debug("DEBUG_FIRST_N=%s ignored for tag compaction", debug_first_n)
    return iterable


def _coerce_int(value: object) -> int | None:
    try:
        if isinstance(value, bool):
            return int(value)
        return int(str(value))
    except (TypeError, ValueError):
        return None


def _coerce_str(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _has_value(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return value != ""
    if isinstance(value, (list, tuple, set, dict)):
        return bool(value)
    return True


def _sum_parts(
    parts: Mapping[str, object] | None,
    fields: Iterable[str],
    *,
    as_float: bool = False,
) -> Union[int, float]:
    if as_float:
        total_float = 0.0
        if isinstance(parts, Mapping):
            for field in fields:
                try:
                    part_value = parts.get(field, 0.0)
                except AttributeError:
                    part_value = 0.0
                try:
                    total_float += float(part_value or 0.0)
                except (TypeError, ValueError):
                    continue
        return total_float

    total_int = 0
    if isinstance(parts, Mapping):
        for field in fields:
            part_value = parts.get(field)
            coerced = _coerce_int(part_value)
            if coerced is not None:
                total_int += coerced
    return total_int


def _coerce_score(value: object, *, as_float: bool) -> Union[int, float, None]:
    if as_float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return None
    return _coerce_int(value)


def _normalize_matched_fields(raw: Mapping[str, object] | None) -> dict[str, bool]:
    matched: dict[str, bool] = {}
    if not isinstance(raw, Mapping):
        return matched
    for field, flag in raw.items():
        matched[str(field)] = bool(flag)
    return matched


def _build_merge_scoring_summary(
    best_tag: Mapping[str, object] | None,
    existing: Mapping[str, object] | None,
) -> dict[str, object] | None:
    summary: dict[str, object] = {}
    if isinstance(existing, Mapping):
        summary.update(existing)

    if not isinstance(best_tag, Mapping):
        return summary or None

    updates: dict[str, object] = {}

    points_mode_active = detect_points_mode_from_payload(best_tag)

    has_parts = "parts" in best_tag
    parts = best_tag.get("parts") if isinstance(best_tag.get("parts"), Mapping) else None

    if has_parts:
        normalized_parts = normalize_parts_for_serialization(
            parts, points_mode=points_mode_active
        )
        identity_fields, debt_fields = resolve_identity_debt_fields()
        updates["identity_score"] = _sum_parts(
            normalized_parts,
            identity_fields,
            as_float=points_mode_active,
        )
        updates["debt_score"] = _sum_parts(
            normalized_parts,
            debt_fields,
            as_float=points_mode_active,
        )

    aux_payload = best_tag.get("aux") if isinstance(best_tag.get("aux"), Mapping) else None
    if "aux" in best_tag and isinstance(aux_payload, Mapping):
        acctnum_level = normalize_level(aux_payload.get("acctnum_level"))
        if acctnum_level == "none":
            account_number_aux = aux_payload.get("account_number")
            if isinstance(account_number_aux, Mapping):
                acctnum_level = normalize_level(account_number_aux.get("acctnum_level"))
        updates["acctnum_level"] = acctnum_level
        matched_fields = _normalize_matched_fields(aux_payload.get("matched_fields"))
        updates["matched_fields"] = matched_fields
        by_field_pairs = aux_payload.get("by_field_pairs")
        if isinstance(by_field_pairs, Mapping):
            updates["matched_pairs"] = {
                str(field): [str(pair[0]), str(pair[1])]
                for field, pair in by_field_pairs.items()
                if isinstance(pair, (list, tuple)) and len(pair) == 2
            }
        for key in ("acctnum_digits_len_a", "acctnum_digits_len_b"):
            value = aux_payload.get(key)
            if value is None:
                continue
            try:
                updates[key] = int(value)
            except (TypeError, ValueError):
                continue

    conflicts_raw = best_tag.get("conflicts") if "conflicts" in best_tag else None
    if isinstance(conflicts_raw, Iterable) and not isinstance(conflicts_raw, (str, bytes)):
        conflicts: list[str] = []
        for conflict in conflicts_raw:
            if conflict is None:
                continue
            conflict_text = str(conflict)
            if conflict_text:
                conflicts.append(conflict_text)
        updates["conflicts"] = conflicts

    reasons_raw = best_tag.get("reasons") if "reasons" in best_tag else None
    if isinstance(reasons_raw, Iterable) and not isinstance(reasons_raw, (str, bytes)):
        reasons: list[str] = []
        for reason in reasons_raw:
            if reason is None:
                continue
            reason_text = str(reason)
            if reason_text:
                reasons.append(reason_text)
        if reasons:
            updates["reasons"] = reasons

    partner = _coerce_int(best_tag.get("with"))
    score_total = _coerce_score(
        best_tag.get("score_total"), as_float=points_mode_active
    )
    if score_total is None:
        score_total = _coerce_score(best_tag.get("total"), as_float=points_mode_active)
    if score_total is not None and ("score_total" in best_tag or "total" in best_tag):
        updates["score_total"] = score_total

    if updates:
        summary.update(updates)

    if partner is not None:
        summary["best_with"] = partner
    elif "best_with" in summary:
        summary.pop("best_with")

    matched_pairs_payload = summary.get("matched_pairs")
    if isinstance(matched_pairs_payload, Mapping):
        matched_pairs_payload = dict(matched_pairs_payload)
        matched_pairs_payload.setdefault("account_number", [])
        summary["matched_pairs"] = matched_pairs_payload
    else:
        summary["matched_pairs"] = {"account_number": []}

    return summary


def _minimal_issue(tag: Mapping[str, object]) -> dict[str, object]:
    payload = {"kind": "issue"}
    type_value = _coerce_str(tag.get("type"))
    if type_value:
        payload["type"] = type_value
    return payload


def _minimal_merge_best(tag: Mapping[str, object]) -> dict[str, object]:
    payload: dict[str, object] = {"kind": "merge_best"}
    partner = _coerce_int(tag.get("with"))
    if partner is not None:
        payload["with"] = partner
    decision = _coerce_str(tag.get("decision"))
    if decision:
        payload["decision"] = decision
    timestamp = _coerce_str(tag.get("at"))
    if timestamp:
        payload["at"] = timestamp
    return payload


def _normalize_flag(value: object) -> bool | str | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered == "unknown":
            return "unknown"
        if lowered == "true":
            return True
        if lowered == "false":
            return False
    return None


def _coerce_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "y"}:
            return True
        if lowered in {"false", "0", "no", "n"}:
            return False
    return False


def _final_resolution_tag(
    idx_with: int, decision: str, flags: dict, reason: str
) -> dict[str, object]:
    return {
        "kind": "ai_resolution",
        "with": idx_with,
        "decision": decision,
        "flags": flags,
        "reason": reason[:200],
    }


def _update_resolution_candidate(
    resolution_by_partner: MutableMapping[int, dict[str, object]],
    partner: int | None,
    *,
    decision: str | None,
    flags: Mapping[str, object] | None,
    reason: str | None,
    normalized: bool,
) -> None:
    if partner is None or not decision:
        return
    flags_dict = dict(flags) if isinstance(flags, Mapping) else {}
    reason_text = reason or decision
    candidate = {
        "decision": decision,
        "flags": flags_dict,
        "reason": reason_text,
        "normalized": bool(normalized),
    }
    existing = resolution_by_partner.get(partner)
    if existing is None:
        resolution_by_partner[partner] = candidate
        return
    existing_normalized = bool(existing.get("normalized"))
    candidate_normalized = candidate["normalized"]
    if candidate_normalized and not existing_normalized:
        resolution_by_partner[partner] = candidate
        return
    if candidate_normalized == existing_normalized:
        resolution_by_partner[partner] = candidate


def _minimal_ai_decision(tag: Mapping[str, object]) -> dict[str, object]:
    payload: dict[str, object] = {"kind": "ai_decision"}
    partner = _coerce_int(tag.get("with"))
    if partner is not None:
        payload["with"] = partner
    decision = _coerce_str(tag.get("decision"))
    if decision:
        payload["decision"] = decision
    timestamp = _coerce_str(tag.get("at"))
    if timestamp:
        payload["at"] = timestamp
    flags = tag.get("flags")
    if isinstance(flags, Mapping):
        account_flag = _normalize_flag(flags.get("account_match"))
        debt_flag = _normalize_flag(flags.get("debt_match"))
        filtered_flags = {}
        if account_flag is not None:
            filtered_flags["account_match"] = account_flag
        if debt_flag is not None:
            filtered_flags["debt_match"] = debt_flag
        if filtered_flags:
            payload["flags"] = filtered_flags
    return payload


def _minimal_pair(tag: Mapping[str, object], *, kind: str) -> dict[str, object]:
    payload: dict[str, object] = {"kind": kind}
    partner = _coerce_int(tag.get("with"))
    if partner is not None:
        payload["with"] = partner
    timestamp = _coerce_str(tag.get("at"))
    if timestamp:
        payload["at"] = timestamp
    return payload


def _merge_explanation_from_tag(tag: Mapping[str, object]) -> dict[str, object] | None:
    kind = str(tag.get("kind", ""))
    if kind not in {"merge_best", "merge_pair"}:
        return None

    payload: dict[str, object] = {"kind": kind}
    partner = _coerce_int(tag.get("with"))
    if partner is not None:
        payload["with"] = partner
    decision = _coerce_str(tag.get("decision"))
    if decision:
        payload["decision"] = decision

    verbose_fields: dict[str, object | None] = {
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
        "matched_pairs": tag.get("matched_pairs"),
        "acctnum_digits_len_a": tag.get("acctnum_digits_len_a"),
        "acctnum_digits_len_b": tag.get("acctnum_digits_len_b"),
    }

    meaningful = False
    for key, value in verbose_fields.items():
        if _has_value(value):
            payload[key] = value
            meaningful = True

    acct_level_value: str | None = None
    aux = tag.get("aux")
    if isinstance(aux, Mapping):
        acct_level_value = normalize_level(aux.get("acctnum_level"))
        matched_fields = aux.get("matched_fields")
        if isinstance(matched_fields, Mapping) and matched_fields:
            payload.setdefault("matched_fields", dict(matched_fields))
            meaningful = True
        by_field_pairs = aux.get("by_field_pairs")
        if isinstance(by_field_pairs, Mapping):
            non_empty_pairs = {
                str(field): [str(pair[0]), str(pair[1])]
                for field, pair in by_field_pairs.items()
                if isinstance(pair, (list, tuple)) and len(pair) == 2
            }
            if non_empty_pairs:
                payload.setdefault("matched_pairs", non_empty_pairs)
                meaningful = True
        for key in ("acctnum_digits_len_a", "acctnum_digits_len_b"):
            if key not in payload and key in aux:
                value = aux.get(key)
                if _has_value(value):
                    payload[key] = value
                    meaningful = True

    direct_level = normalize_level(tag.get("acctnum_level"))
    if direct_level is not None:
        if direct_level != "none" or acct_level_value in (None, "none"):
            acct_level_value = direct_level
    if acct_level_value is not None:
        payload["acctnum_level"] = acct_level_value
        meaningful = True

    matched_pairs_payload = payload.get("matched_pairs")
    if isinstance(matched_pairs_payload, Mapping):
        matched_pairs_payload = dict(matched_pairs_payload)
        if not any(
            isinstance(pair, (list, tuple)) and len(pair) == 2
            for pair in matched_pairs_payload.values()
        ):
            payload.pop("matched_pairs", None)
            matched_pairs_payload = None
        else:
            matched_pairs_payload.setdefault("account_number", [])
            payload["matched_pairs"] = matched_pairs_payload
    if "matched_pairs" not in payload:
        payload["matched_pairs"] = {"account_number": []}

    return payload if meaningful else None


def _ai_explanations_from_tag(
    tag: Mapping[str, object],
    *,
    decision_reason_map: dict[int, str],
    existing_ai_lookup: Mapping[int, Mapping[str, object]] | None = None,
) -> tuple[list[dict[str, object]], dict[str, object] | None]:
    kind = str(tag.get("kind", ""))
    partner = _coerce_int(tag.get("with"))
    decision = _coerce_str(tag.get("decision"))
    reason = tag.get("reason")
    raw_response = tag.get("raw_response")
    entries: list[dict[str, object]] = []
    merge_entry: dict[str, object] | None = None

    # Summary entries include a normalized flag, e.g. {"kind": "ai_decision", "normalized": true}.
    if kind == "ai_decision":
        existing_entry = (
            existing_ai_lookup.get(partner)
            if existing_ai_lookup is not None and partner is not None
            else None
        )
        reason_text = _coerce_str(reason)
        if reason_text is None and isinstance(existing_entry, Mapping):
            reason_text = _coerce_str(existing_entry.get("reason"))
        if isinstance(partner, int) and reason_text:
            decision_reason_map[partner] = reason_text

        normalized_flag = _coerce_bool(tag.get("normalized"))
        payload: dict[str, object] = {
            "kind": kind,
            "normalized": normalized_flag,
        }
        if partner is not None:
            payload["with"] = partner
        if decision:
            payload["decision"] = decision
        if reason_text is not None:
            payload["reason"] = reason_text
        flags = tag.get("flags")
        normalized_flags: dict[str, object] = {}
        if isinstance(flags, Mapping):
            account_flag = _normalize_flag(flags.get("account_match"))
            debt_flag = _normalize_flag(flags.get("debt_match"))
            if account_flag is not None:
                normalized_flags["account_match"] = account_flag
            if debt_flag is not None:
                normalized_flags["debt_match"] = debt_flag
        canonical_decision = MergeDecision.canonical_value(decision)
        normalized_decision = canonical_decision or (decision or "")
        if not normalized_decision:
            normalized_decision = "different"
        default_flags: dict[str, object] = {}
        lowered_decision = normalized_decision.strip().lower()
        if lowered_decision.startswith("same_account_"):
            default_flags["account_match"] = True
            if lowered_decision.endswith("_same_debt"):
                default_flags["debt_match"] = True
        if lowered_decision.startswith("same_debt_"):
            default_flags["debt_match"] = True
        final_flags: dict[str, object] = {}
        for key in ("account_match", "debt_match"):
            if key in normalized_flags:
                final_flags[key] = normalized_flags[key]
            elif key in default_flags:
                final_flags[key] = default_flags[key]
            else:
                final_flags[key] = "unknown"
        payload["flags"] = dict(final_flags)
        response_payload = raw_response if _has_value(raw_response) else None
        if response_payload is None and isinstance(existing_entry, Mapping):
            existing_response = existing_entry.get("raw_response")
            if _has_value(existing_response):
                response_payload = existing_response
        if _has_value(response_payload):
            payload["raw_response"] = response_payload
        ai_result_decision = normalized_decision if not decision else decision
        ai_result_payload: dict[str, object] = {
            "decision": ai_result_decision,
            "flags": {key: final_flags[key] for key in ("account_match", "debt_match")},
        }
        if reason_text:
            ai_result_payload["reason"] = reason_text
        payload["ai_result"] = dict(ai_result_payload)
        entries.append(payload)

        resolution_entry = {
            "kind": "ai_resolution",
            "normalized": normalized_flag,
            "flags": dict(final_flags),
            "ai_result": dict(ai_result_payload),
        }
        if partner is not None:
            resolution_entry["with"] = partner
        resolution_entry["decision"] = ai_result_decision
        if reason_text is not None:
            resolution_entry["reason"] = reason_text
        entries.append(resolution_entry)

        if decision == "merge":
            merge_entry = {
                "kind": "ai_merge_decision",
                "origin": "ai",
                "normalized": payload["normalized"],
            }
            if partner is not None:
                merge_entry["with"] = partner
            if decision:
                merge_entry["decision"] = decision
            if reason_text is not None:
                merge_entry["reason"] = reason_text
            if "flags" in payload:
                merge_entry["flags"] = payload["flags"]
        return entries, merge_entry

    if kind in {"same_debt_pair", "same_account_pair"}:
        if not _has_value(reason) and isinstance(partner, int):
            reason = decision_reason_map.get(partner)
        if not _has_value(reason):
            return entries, None
        payload = {"kind": kind}
        if partner is not None:
            payload["with"] = partner
        payload["reason"] = reason
        entries.append(payload)

    return entries, merge_entry


def _load_tags(tags_path: Path) -> list[Mapping[str, object]]:
    if not tags_path.exists():
        return []
    try:
        raw = tags_path.read_text(encoding="utf-8")
    except OSError:
        return []
    if not raw.strip():
        return []
    try:
        data = json.loads(raw)
    except Exception:
        return []
    if isinstance(data, list):
        entries = data
    elif isinstance(data, Mapping):
        tags = data.get("tags")
        entries = tags if isinstance(tags, list) else []
    else:
        entries = []
    result: list[Mapping[str, object]] = []
    for entry in entries:
        if isinstance(entry, Mapping):
            result.append(entry)
    return result


def _dedupe(entries: Iterable[dict[str, object]]) -> list[dict[str, object]]:
    unique: list[dict[str, object]] = []
    seen: set[str] = set()
    for entry in entries:
        key = json.dumps(entry, sort_keys=True, separators=(",", ":"))
        if key in seen:
            continue
        seen.add(key)
        unique.append(entry)
    return unique


def _merge_summary_entries(
    existing: Sequence[MutableMapping[str, object]] | None,
    updates: Iterable[Mapping[str, object]],
) -> list[dict[str, object]]:
    combined: list[dict[str, object]] = []
    if existing:
        for entry in existing:
            if isinstance(entry, Mapping):
                combined.append(dict(entry))
    for entry in updates:
        if isinstance(entry, Mapping):
            combined.append(dict(entry))
    return _dedupe(combined)


def _write_tags(tags_path: Path, payload: Iterable[Mapping[str, object]]) -> None:
    entries = [dict(entry) for entry in payload]
    _atomic_write_json(tags_path, entries)
    log.info("TAGS_WRITTEN_ATOMIC path=%s entries=%d", tags_path, len(entries))


def _write_summary(summary_path: Path, payload: Mapping[str, object]) -> None:
    data = dict(payload)
    if os.getenv("COMPACT_MERGE_SUMMARY", "1") == "1":
        compact_merge_sections(data)
    _atomic_write_json(summary_path, data)
    log.info(
        "SUMMARY_WRITTEN_ATOMIC path=%s keys=%d",
        summary_path,
        len(data),
    )


def compact_account_tags(
    account_dir: Pathish,
    *,
    minimal_only: bool = True,
    explanations_to_summary: bool = True,
) -> None:
    """Reduce ``tags.json`` to minimal tags and move verbose data to summary."""

    account_path = Path(account_dir)
    tags_path = account_path / "tags.json"
    tags = _load_tags(tags_path)
    if not tags:
        return

    minimal_tags: list[dict[str, object]] = []
    merge_explanations: list[dict[str, object]] = []
    ai_explanations: list[dict[str, object]] = []
    decision_reasons: dict[int, str] = {}
    existing_ai_lookup: dict[int, Mapping[str, object]] = {}
    best_merge_tag: Mapping[str, object] | None = None
    summary_path = account_path / "summary.json"
    summary_data: dict[str, object] | None = None
    resolution_by_partner: dict[int, dict[str, object]] = {}

    if explanations_to_summary:
        if summary_path.exists():
            try:
                summary_data = json.loads(summary_path.read_text(encoding="utf-8"))
            except Exception:
                summary_data = {}
        else:
            summary_data = {}
        if not isinstance(summary_data, dict):
            summary_data = {}

        existing_ai_entries = summary_data.get("ai_explanations")
        if isinstance(existing_ai_entries, Sequence):
            for entry in existing_ai_entries:
                if not isinstance(entry, Mapping):
                    continue
                kind = str(entry.get("kind", ""))
                if kind != "ai_decision":
                    continue
                partner = _coerce_int(entry.get("with"))
                reason_text = _coerce_str(entry.get("reason"))
                if partner is not None and reason_text:
                    decision_reasons[partner] = reason_text
                if partner is not None:
                    existing_ai_lookup[partner] = entry

    for tag in tags:
        kind = str(tag.get("kind", "")).strip().lower()
        if kind == "ai_resolution":
            partner = _coerce_int(tag.get("with"))
            decision_text = _coerce_str(tag.get("decision"))
            flags_payload = tag.get("flags")
            if not isinstance(flags_payload, Mapping):
                flags_payload = None
            reason_text = _coerce_str(tag.get("reason"))
            normalized_flag = _coerce_bool(tag.get("normalized"))
            _update_resolution_candidate(
                resolution_by_partner,
                partner,
                decision=decision_text,
                flags=flags_payload,
                reason=reason_text,
                normalized=normalized_flag,
            )
            continue
        if kind == "ai_error":
            partner = _coerce_int(tag.get("with"))
            error_kind = _coerce_str(tag.get("error_kind")) or "error"
            message = _coerce_str(tag.get("message"))
            if message and error_kind and message != error_kind:
                reason_text = f"{error_kind}: {message}"
            else:
                reason_text = message or error_kind
            _update_resolution_candidate(
                resolution_by_partner,
                partner,
                decision="error",
                flags=None,
                reason=reason_text,
                normalized=False,
            )
            continue
        if minimal_only:
            if kind == "issue":
                minimal_tags.append(_minimal_issue(tag))
            elif kind == "merge_best":
                minimal_tags.append(_minimal_merge_best(tag))
            elif kind == "ai_decision":
                minimal_tags.append(_minimal_ai_decision(tag))
            elif kind in {"same_debt_pair", "same_account_pair"}:
                minimal_tags.append(_minimal_pair(tag, kind=kind))
        else:
            minimal_tags.append(dict(tag))

        if explanations_to_summary:
            if kind in {"merge_best", "merge_pair"}:
                merge_payload = _merge_explanation_from_tag(tag)
                if merge_payload is not None:
                    merge_explanations.append(merge_payload)
                if kind == "merge_best":
                    best_merge_tag = dict(tag)
            elif kind in {"ai_decision", "same_debt_pair", "same_account_pair"}:
                ai_entries, ai_merge_entry = _ai_explanations_from_tag(
                    tag,
                    decision_reason_map=decision_reasons,
                    existing_ai_lookup=existing_ai_lookup,
                )
                ai_explanations.extend(ai_entries)
                if ai_merge_entry is not None:
                    merge_explanations.append(ai_merge_entry)
        elif kind == "merge_best":
            best_merge_tag = dict(tag)

        if kind == "ai_decision":
            partner = _coerce_int(tag.get("with"))
            decision_text = _coerce_str(tag.get("decision"))
            if partner is not None and decision_text:
                reason_text = _coerce_str(tag.get("reason"))
                if not reason_text and partner in decision_reasons:
                    reason_text = decision_reasons.get(partner)
                if reason_text:
                    decision_reasons[partner] = reason_text
                minimal_entry = _minimal_ai_decision(tag)
                flags_payload = minimal_entry.get("flags")
                normalized_flag = _coerce_bool(tag.get("normalized"))
                _update_resolution_candidate(
                    resolution_by_partner,
                    partner,
                    decision=decision_text,
                    flags=flags_payload if isinstance(flags_payload, Mapping) else None,
                    reason=reason_text,
                    normalized=normalized_flag,
                )

    if minimal_only:
        minimal_tags = [tag for tag in minimal_tags if tag]

    if resolution_by_partner:
        for partner, data in sorted(resolution_by_partner.items()):
            decision_value = _coerce_str(data.get("decision"))
            if not decision_value:
                continue
            flags_obj = data.get("flags")
            flags_value = dict(flags_obj) if isinstance(flags_obj, Mapping) else {}
            reason_value = _coerce_str(data.get("reason")) or decision_value
            resolution_tag = _final_resolution_tag(
                partner,
                decision_value,
                flags_value,
                reason_value,
            )
            minimal_tags.append(resolution_tag)

    minimal_tags = _dedupe(minimal_tags)

    _write_tags(tags_path, minimal_tags)

    if not explanations_to_summary:
        return

    if summary_data is None:
        if summary_path.exists():
            try:
                summary_data = json.loads(summary_path.read_text(encoding="utf-8"))
            except Exception:
                summary_data = {}
        else:
            summary_data = {}
        if not isinstance(summary_data, dict):
            summary_data = {}

    existing_merge = summary_data.get("merge_explanations")
    existing_ai = summary_data.get("ai_explanations")
    existing_scoring = summary_data.get("merge_scoring")

    summary_data["merge_explanations"] = _merge_summary_entries(existing_merge, merge_explanations)
    summary_data["ai_explanations"] = _merge_summary_entries(existing_ai, ai_explanations)

    merge_scoring_summary = _build_merge_scoring_summary(best_merge_tag, existing_scoring)
    if merge_scoring_summary is not None:
        summary_data["merge_scoring"] = merge_scoring_summary
    elif "merge_scoring" in summary_data:
        summary_data.pop("merge_scoring", None)

    merge_summary_sections(summary_data)

    _write_summary(summary_path, summary_data)


def compact_tags_for_run(
    sid: str,
    *,
    runs_root: Pathish | None = None,
    minimal_only: bool = True,
    explanations_to_summary: bool = True,
) -> None:
    """Compact tags for all accounts under ``runs/<sid>/cases/accounts``."""

    base = Path(runs_root) if runs_root is not None else Path("runs")
    accounts_dir = base / sid / "cases" / "accounts"
    if not accounts_dir.exists() or not accounts_dir.is_dir():
        return

    for account_dir in _maybe_slice(sorted(accounts_dir.iterdir())):
        if not account_dir.is_dir():
            continue
        try:
            compact_account_tags(
                account_dir,
                minimal_only=minimal_only,
                explanations_to_summary=explanations_to_summary,
            )
        except Exception:  # pragma: no cover - defensive logging
            log.exception(
                "TAGS_COMPACT_FAILED sid=%s account_dir=%s",
                sid,
                account_dir,
            )
            continue


def compact_tags_for_sid(
    sid: str,
    runs_root: Pathish | None = None,
    *,
    minimal_only: bool = True,
    explanations_to_summary: bool = True,
) -> None:
    """Backwards-compatible alias for run-level compaction."""

    compact_tags_for_run(
        sid,
        runs_root=runs_root,
        minimal_only=minimal_only,
        explanations_to_summary=explanations_to_summary,
    )


__all__ = [
    "compact_account_tags",
    "compact_tags_for_run",
    "compact_tags_for_sid",
]
