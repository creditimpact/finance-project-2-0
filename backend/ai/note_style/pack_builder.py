"""Utilities for building note_style AI packs with contextual metadata."""

from __future__ import annotations

import json
import logging
import math
import re
import unicodedata
from collections import Counter
from datetime import datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from backend.ai.note_style.prompt import build_base_system_prompt
from backend.core.ai.paths import (
    NoteStyleAccountPaths,
    ensure_note_style_account_paths,
    ensure_note_style_paths,
)


log = logging.getLogger(__name__)


_NOTE_VALUE_PATHS: tuple[tuple[str, ...], ...] = (
    ("note",),
    ("note_text",),
    ("explain",),
    ("explanation",),
    ("data", "explain"),
    ("answers", "explain"),
    ("answers", "explanation"),
    ("answers", "note"),
    ("answers", "notes"),
    ("answers", "customer_note"),
)

_BUREAU_FIELDS: tuple[str, ...] = (
    "reported_creditor",
    "account_type",
    "account_status",
    "payment_status",
    "creditor_type",
    "date_opened",
    "date_reported",
    "date_of_last_activity",
    "closed_date",
    "last_verified",
    "balance_owed",
    "high_balance",
    "past_due_amount",
)

_AMOUNT_FIELDS = {"balance_owed", "high_balance", "past_due_amount"}
_DATE_FIELDS = {
    "date_opened",
    "date_reported",
    "date_of_last_activity",
    "closed_date",
    "last_verified",
}

_BUREAU_PRIORITY = ("transunion", "experian", "equifax")
_BUREAU_KEYS = ("transunion", "experian", "equifax")
_BUREAU_CONTEXT_FIELDS = (
    "account_type",
    "account_status",
    "payment_status",
    "creditor_type",
    "date_opened",
    "date_reported",
    "date_of_last_activity",
    "closed_date",
    "last_verified",
    "balance_owed",
    "high_balance",
    "past_due_amount",
)

_CONTEXT_NOISE_KEYS = (
    "hash",
    "salt",
    "debug",
    "raw",
    "blob",
    "token",
    "signature",
    "checksum",
)

def _build_system_message(
    account_context: Mapping[str, Any] | None,
    bureaus_summary: Mapping[str, Any] | None,
) -> str:
    """Return the fixed system prompt for note_style analysis."""

    _ = account_context
    _ = bureaus_summary
    return build_base_system_prompt()


class PackBuilderError(RuntimeError):
    """Raised when a note_style pack cannot be constructed."""


def _build_user_message_content(
    *,
    meta_name: str,
    primary_issue_tag: str | None,
    bureau_data: Mapping[str, Any],
    note_text: str,
) -> dict[str, Any]:
    return {
        "meta_name": meta_name,
        "primary_issue_tag": primary_issue_tag,
        "bureau_data": dict(bureau_data) if isinstance(bureau_data, Mapping) else {},
        "note_text": note_text,
    }


def build_pack(
    sid: str,
    account_id: str,
    *,
    runs_root: Path | str | None = None,
    mirror_debug: bool = False,
) -> Mapping[str, Any]:
    """Build a note_style pack for ``sid``/``account_id``.

    The pack is persisted to the canonical packs directory as a single JSONL line.
    The constructed payload is returned for convenience.
    """

    if not sid:
        raise ValueError("sid is required")
    if not account_id:
        raise ValueError("account_id is required")

    runs_root_path = Path(runs_root or "runs").resolve()
    run_dir = runs_root_path / sid

    response_path = run_dir / "frontend" / "review" / "responses" / f"{account_id}.result.json"
    if not response_path.is_file():
        raise PackBuilderError(f"response note not found: {response_path}")

    account_dir = _locate_account_dir(run_dir / "cases" / "accounts", account_id)
    if account_dir is None:
        raise PackBuilderError(
            f"account artifacts not found for account_id={account_id!r} under {run_dir / 'cases' / 'accounts'}"
        )

    response_payload = _load_json(response_path)
    note_text = _extract_note_text(response_payload)

    bureaus_raw, _bureaus_missing = _load_account_context(
        account_dir / "bureaus.json",
        sid=sid,
        account_id=account_id,
        context_label="bureaus",
    )
    tags_raw, _tags_missing = _load_account_context(
        account_dir / "tags.json",
        sid=sid,
        account_id=account_id,
        context_label="tags",
    )
    meta_raw, _meta_missing = _load_account_context(
        account_dir / "meta.json",
        sid=sid,
        account_id=account_id,
        context_label="meta",
    )

    bureaus_payload = _ensure_mapping(bureaus_raw)
    tags_payload = _ensure_sequence(tags_raw)
    meta_payload = _ensure_mapping(meta_raw)

    meta_name = _extract_meta_name(meta_payload, account_id)
    bureau_data = _with_bureau_defaults(_extract_bureau_data(bureaus_payload))
    primary_issue_tag = _extract_primary_issue_tag(tags_payload)

    bureau_context = dict(bureau_data) if isinstance(bureau_data, Mapping) else {}

    user_message = _build_user_message_content(
        meta_name=meta_name,
        primary_issue_tag=primary_issue_tag,
        bureau_data=bureau_context,
        note_text=note_text,
    )

    system_message = _build_system_message(None, None)

    pack_payload: dict[str, Any] = {
        "meta_name": meta_name,
        "primary_issue_tag": primary_issue_tag,
        "bureau_data": bureau_context,
        "note_text": note_text,
        "messages": [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ],
    }

    paths = ensure_note_style_paths(runs_root_path, sid, create=True)
    account_paths = ensure_note_style_account_paths(paths, account_id, create=True)

    bytes_written = _write_jsonl(account_paths.pack_file, pack_payload)

    try:
        pack_relative = (
            account_paths.pack_file.resolve().relative_to(paths.base.resolve()).as_posix()
        )
    except ValueError:
        pack_relative = account_paths.pack_file.resolve().as_posix()

    log.info(
        "[NOTE_STYLE] PACK_WRITTEN account=%s path=%s bytes=%d",
        account_id,
        pack_relative,
        bytes_written,
    )

    log.info(
        "NOTE_STYLE_PACK_BUILT sid=%s account=%s fields=[meta_name,primary_issue_tag,bureau_data,note_text,messages]",
        sid,
        account_id,
    )

    if mirror_debug:
        _write_debug_snapshot(account_paths, pack_payload)

    return pack_payload


def _write_jsonl(path: Path, payload: Mapping[str, Any]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, ensure_ascii=False)
    data = (text + "\n").encode("utf-8")
    with path.open("wb") as handle:
        handle.write(data)
    return len(data)


def _write_debug_snapshot(account_paths: NoteStyleAccountPaths, payload: Mapping[str, Any]) -> None:
    account_paths.debug_file.parent.mkdir(parents=True, exist_ok=True)
    account_paths.debug_file.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _load_json(path: Path) -> Any:
    if not path.is_file():
        return None
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        return None
    return json.loads(text)


def _ensure_mapping(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    return {}


def _ensure_sequence(value: Any) -> Sequence[Any]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return value
    return []


def _load_account_context(
    path: Path,
    *,
    sid: str,
    account_id: str,
    context_label: str,
) -> tuple[Any, bool]:
    """Load a required account context payload, logging a warning when missing."""

    try:
        payload = _load_json(path)
    except json.JSONDecodeError:
        log.warning(
            "[NOTE_STYLE] PACK_CONTEXT_MISSING sid=%s account=%s context=%s reason=invalid_json path=%s",
            sid,
            account_id,
            context_label,
            path,
        )
        return None, True

    if payload is None:
        reason = "missing" if not path.is_file() else "empty"
        log.warning(
            "[NOTE_STYLE] PACK_CONTEXT_MISSING sid=%s account=%s context=%s reason=%s path=%s",
            sid,
            account_id,
            context_label,
            reason,
            path,
        )
        return None, True

    return payload, False


def _build_account_payload(
    meta: Mapping[str, Any],
    bureaus: Mapping[str, Any],
    tags: Sequence[Any],
) -> dict[str, Any]:
    payload: dict[str, Any] = {}

    sanitized_meta = _sanitize_context_payload(meta)
    if sanitized_meta:
        payload["meta"] = sanitized_meta

    sanitized_bureaus = _sanitize_context_payload(bureaus)
    if sanitized_bureaus:
        payload["bureaus"] = sanitized_bureaus

    sanitized_tags = _sanitize_context_payload(tags)
    if sanitized_tags:
        payload["tags"] = sanitized_tags

    return payload


def _extract_note_text(payload: Any) -> str:
    if isinstance(payload, Mapping):
        for path in _NOTE_VALUE_PATHS:
            current: Any = payload
            for key in path:
                if not isinstance(current, Mapping):
                    break
                current = current.get(key)
            else:
                normalized = _normalize_text(current)
                if normalized:
                    return normalized
    return ""


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return unicodedata.normalize("NFKC", value).strip()
    return unicodedata.normalize("NFKC", str(value)).strip()


def _sanitize_context_payload(value: Any, *, depth: int = 0) -> Any:
    if depth > 6:
        return None
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        sanitized: dict[str, Any] = {}
        for key, raw in value.items():
            if not isinstance(key, str):
                continue
            if _is_noise_key(key):
                continue
            cleaned = _sanitize_context_payload(raw, depth=depth + 1)
            if _is_empty_context_value(cleaned):
                continue
            sanitized[key] = cleaned
        return sanitized
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        sanitized_seq: list[Any] = []
        for entry in value:
            cleaned = _sanitize_context_payload(entry, depth=depth + 1)
            if _is_empty_context_value(cleaned):
                continue
            sanitized_seq.append(cleaned)
        return sanitized_seq
    normalized = _normalize_text(value)
    return normalized if normalized else None


def _is_noise_key(key: str) -> bool:
    lowered = key.lower()
    return any(fragment in lowered for fragment in _CONTEXT_NOISE_KEYS)


def _is_empty_context_value(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, (str, bytes, bytearray)):
        return len(value) == 0
    if isinstance(value, Mapping):
        return all(_is_empty_context_value(v) for v in value.values())
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return len(value) == 0 or all(_is_empty_context_value(v) for v in value)
    return False


def _normalize_amount(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, Decimal):
        decimal_value = value
    elif isinstance(value, (int, float)):
        if isinstance(value, float) and not math.isfinite(value):
            return ""
        decimal_value = Decimal(str(value))
    else:
        text = _normalize_text(value)
        if not text:
            return ""
        stripped = text.replace(",", "").replace("$", "").strip()
        negative = False
        if stripped.startswith("(") and stripped.endswith(")"):
            negative = True
            stripped = stripped[1:-1]
        if stripped.startswith("-"):
            negative = True
            stripped = stripped[1:]
        if stripped.endswith("-"):
            negative = True
            stripped = stripped[:-1]
        stripped = stripped.strip()
        if not stripped:
            return ""
        try:
            decimal_value = Decimal(stripped)
        except InvalidOperation:
            filtered = re.sub(r"[^0-9.]", "", stripped)
            if not filtered:
                return ""
            try:
                decimal_value = Decimal(filtered)
            except InvalidOperation:
                return ""
        if negative:
            decimal_value = -decimal_value
    normalized = format(decimal_value, "f")
    if "." in normalized:
        normalized = normalized.rstrip("0").rstrip(".")
    if not normalized:
        normalized = "0"
    return normalized


def _normalize_date(value: Any) -> str:
    text = _normalize_text(value)
    if not text:
        return ""
    stripped = text.strip()
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%m/%d/%Y", "%m-%d-%Y", "%m.%d.%Y"):
        try:
            parsed = datetime.strptime(stripped, fmt)
        except ValueError:
            continue
        return parsed.date().isoformat()
    for fmt in ("%m/%d/%y", "%m-%d-%y", "%m.%d.%y"):
        try:
            parsed = datetime.strptime(stripped, fmt)
        except ValueError:
            continue
        return parsed.date().isoformat()
    try:
        parsed = datetime.fromisoformat(stripped)
    except ValueError:
        return stripped
    return parsed.date().isoformat()


def _normalize_field(field: str, value: Any) -> str:
    if isinstance(value, Mapping):
        for key in ("value", "raw", "display", "text", "formatted"):
            if key in value:
                candidate = _normalize_field(field, value[key])
                if candidate:
                    return candidate
        return ""
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for entry in value:
            candidate = _normalize_field(field, entry)
            if candidate:
                return candidate
        return ""
    if field in _DATE_FIELDS:
        return _normalize_date(value)
    if field in _AMOUNT_FIELDS:
        return _normalize_amount(value)
    return _normalize_text(value)


def _summarize_bureaus(bureaus: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(bureaus, Mapping):
        return {}

    per_bureau: dict[str, dict[str, str]] = {}
    field_values: dict[str, dict[str, str]] = {field: {} for field in _BUREAU_FIELDS}

    for bureau_name, payload in sorted(bureaus.items(), key=lambda item: item[0]):
        if not isinstance(payload, Mapping):
            continue
        normalized_fields: dict[str, str] = {}
        for field in _BUREAU_FIELDS:
            value = _normalize_field(field, payload.get(field))
            if value:
                normalized_fields[field] = value
                field_values.setdefault(field, {})[bureau_name] = value
        if normalized_fields:
            per_bureau[bureau_name] = normalized_fields

    if not per_bureau:
        return {}

    majority_values: dict[str, str] = {}
    disagreements: dict[str, dict[str, str]] = {}

    for field, bureau_map in field_values.items():
        if not bureau_map:
            continue
        unique_values = {value for value in bureau_map.values() if value}
        if unique_values:
            selected = _select_majority_value(bureau_map)
            if selected:
                majority_values[field] = selected
        if len(unique_values) > 1:
            disagreements[field] = dict(sorted(bureau_map.items(), key=lambda item: item[0]))

    return {
        "per_bureau": per_bureau,
        "majority_values": majority_values,
        "disagreements": disagreements,
    }


def _select_majority_value(bureau_map: Mapping[str, str]) -> str:
    non_empty = {bureau: value for bureau, value in bureau_map.items() if value}
    if not non_empty:
        return ""

    counter = Counter(non_empty.values())
    if counter:
        most_common = counter.most_common()
        if most_common:
            top_count = most_common[0][1]
            candidates = [value for value, count in most_common if count == top_count]
            if len(candidates) == 1:
                return candidates[0]
            for bureau in _BUREAU_PRIORITY:
                candidate = non_empty.get(bureau)
                if candidate:
                    return candidate
            for bureau, value in sorted(non_empty.items(), key=lambda item: item[0]):
                if value:
                    return value
    for bureau in _BUREAU_PRIORITY:
        candidate = non_empty.get(bureau)
        if candidate:
            return candidate
    for value in non_empty.values():
        if value:
            return value
    return ""


def _build_account_context(
    meta: Mapping[str, Any],
    bureaus: Mapping[str, Any],
    tags: Sequence[Any],
    bureaus_summary: Mapping[str, Any],
) -> dict[str, Any]:
    context: dict[str, Any] = {}

    heading_guess = _normalize_text(meta.get("heading_guess"))
    creditor_name = _normalize_text(meta.get("creditor_name"))
    reported_creditor = heading_guess or creditor_name

    per_bureau = bureaus_summary.get("per_bureau") if isinstance(bureaus_summary, Mapping) else None
    if not reported_creditor:
        majority_values = bureaus_summary.get("majority_values") if isinstance(bureaus_summary, Mapping) else None
        if isinstance(majority_values, Mapping):
            reported_creditor = _normalize_text(majority_values.get("reported_creditor"))
        if not reported_creditor and isinstance(per_bureau, Mapping):
            for bureau in _BUREAU_PRIORITY:
                payload = per_bureau.get(bureau)
                if isinstance(payload, Mapping):
                    candidate = _normalize_text(payload.get("reported_creditor"))
                    if candidate:
                        reported_creditor = candidate
                        break
    if reported_creditor:
        context["reported_creditor"] = reported_creditor

    account_tail = _extract_account_tail(meta, bureaus)
    if account_tail:
        context["account_tail"] = account_tail

    issues: list[str] = []
    for tag in tags:
        if not isinstance(tag, Mapping):
            continue
        if _normalize_text(tag.get("kind")).lower() != "issue":
            continue
        issue_value = _normalize_text(tag.get("type"))
        if issue_value and issue_value not in issues:
            issues.append(issue_value)
    if issues:
        context.setdefault("tags", {})["issues"] = issues
        context["primary_issue"] = issues[0]

    if heading_guess:
        context.setdefault("meta", {})["heading_guess"] = heading_guess

    return context


def _extract_account_tail(meta: Mapping[str, Any], bureaus: Mapping[str, Any]) -> str:
    tail = _normalize_text(meta.get("account_number_tail"))
    if tail:
        digits = re.sub(r"\D", "", tail)
        return digits[-4:] if digits else tail

    if isinstance(bureaus, Mapping):
        for payload in bureaus.values():
            if not isinstance(payload, Mapping):
                continue
            candidate = _normalize_text(payload.get("account_number_display"))
            if candidate:
                digits = re.sub(r"\D", "", candidate)
                return digits[-4:] if digits else candidate
    return ""


def _clean_display_text(value: Any) -> str | None:
    if isinstance(value, str):
        text = value.strip()
        return text or None
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _extract_meta_name(meta_payload: Mapping[str, Any] | None, account_id: str) -> str:
    if isinstance(meta_payload, Mapping):
        for key in ("heading_guess", "name"):
            candidate = _clean_display_text(meta_payload.get(key))
            if candidate:
                return candidate
    fallback = _clean_display_text(account_id)
    return fallback or str(account_id)


def _filter_bureau_fields(payload: Mapping[str, Any] | None) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        return {}
    filtered: dict[str, Any] = {}
    for field in _BUREAU_CONTEXT_FIELDS:
        value = payload.get(field)
        if value is None:
            continue
        if isinstance(value, str):
            text = value.strip()
            if not text:
                continue
            filtered[field] = text
        else:
            filtered[field] = value
    return filtered


def _build_majority_values(bureaus_payload: Mapping[str, Any]) -> dict[str, Any]:
    majority_payload = (
        bureaus_payload.get("majority_values")
        if isinstance(bureaus_payload, Mapping)
        else None
    )
    majority_values = _filter_bureau_fields(majority_payload)
    if majority_values:
        return majority_values

    if isinstance(bureaus_payload, Mapping):
        for bureau_key in _BUREAU_KEYS:
            bureau_values = _filter_bureau_fields(bureaus_payload.get(bureau_key))
            if bureau_values:
                return bureau_values
    return {}


def _extract_bureau_data(bureaus_payload: Mapping[str, Any]) -> Mapping[str, Any]:
    if not isinstance(bureaus_payload, Mapping):
        return {}

    majority_values = _build_majority_values(bureaus_payload)
    if majority_values:
        return majority_values

    for bureau_key in _BUREAU_KEYS:
        filtered = _filter_bureau_fields(bureaus_payload.get(bureau_key))
        if filtered:
            return filtered

    return {}


def _with_bureau_defaults(bureau_data: Mapping[str, Any] | None) -> dict[str, Any]:
    defaults = {field: "--" for field in _BUREAU_CONTEXT_FIELDS}
    if isinstance(bureau_data, Mapping):
        defaults.update(bureau_data)
    return defaults


def _issue_type_from_entry(entry: Any) -> str | None:
    if not isinstance(entry, Mapping):
        return None
    if entry.get("kind") != "issue":
        return None
    issue_type = _clean_display_text(entry.get("type"))
    return issue_type


def _extract_primary_issue_tag(tags_payload: Any) -> str | None:
    if isinstance(tags_payload, Mapping):
        issue_type = _issue_type_from_entry(tags_payload)
        if issue_type:
            return issue_type
        entries = tags_payload.get("tags")
        if isinstance(entries, Iterable) and not isinstance(entries, (str, bytes, bytearray)):
            for entry in entries:
                issue_type = _issue_type_from_entry(entry)
                if issue_type:
                    return issue_type
        return None

    if isinstance(tags_payload, Iterable) and not isinstance(tags_payload, (str, bytes, bytearray)):
        for entry in tags_payload:
            issue_type = _issue_type_from_entry(entry)
            if issue_type:
                return issue_type
    return None


def _locate_account_dir(accounts_dir: Path, account_id: str) -> Path | None:
    if not accounts_dir.is_dir():
        return None

    # direct match
    direct = accounts_dir / account_id
    if direct.is_dir():
        return direct.resolve()

    digits = re.findall(r"(\d+)", account_id)
    for piece in digits:
        normalized = piece.lstrip("0") or "0"
        candidate = accounts_dir / normalized
        if candidate.is_dir():
            return candidate.resolve()

    target = _normalize_text(account_id).lower()
    for entry in sorted(accounts_dir.iterdir(), key=lambda p: p.name):
        if not entry.is_dir():
            continue
        for filename in ("summary.json", "meta.json"):
            payload = _load_json(entry / filename)
            if isinstance(payload, Mapping):
                for key in ("account_id", "account_key", "account_identifier"):
                    value = _normalize_text(payload.get(key)).lower()
                    if value and value == target:
                        return entry.resolve()
    return None


__all__ = ["build_pack", "PackBuilderError"]
