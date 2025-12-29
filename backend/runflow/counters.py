"""Helpers for reconciling runflow stage counters with filesystem artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

import json

from typing import TYPE_CHECKING

from backend.core.ai.paths import ensure_merge_paths, ensure_note_style_paths
from backend.frontend.packs.config import load_frontend_stage_config


def _coerce_int(value: Any) -> Optional[int]:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _load_document(path: Path) -> Optional[Mapping[str, Any]]:
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    except OSError:
        return None

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return None

    if isinstance(payload, Mapping):
        return payload

    if isinstance(payload, Sequence):
        # Legacy layouts occasionally serialised the payload as a list.
        return {"items": list(payload)}

    return None


def _normalize_entry_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, (int, float)):
        try:
            return str(int(value))
        except (TypeError, ValueError):
            return str(value)
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def _entry_first_text(entry: Mapping[str, Any], keys: Sequence[str]) -> str:
    for key in keys:
        try:
            value = entry[key]
        except KeyError:
            continue
        text = _normalize_entry_text(value)
        if text:
            return text
    return ""


def _validation_pack_entry_key(entry: Mapping[str, Any]) -> tuple[str, ...]:
    account_key = _normalize_entry_text(entry.get("account_id"))
    pack_key = _entry_first_text(
        entry,
        ("pack", "pack_path", "pack_file", "pack_filename"),
    )
    result_jsonl_key = _entry_first_text(
        entry,
        ("result_jsonl", "result_jsonl_path", "result_jsonl_file"),
    )
    result_json_key = _entry_first_text(
        entry,
        (
            "result_json",
            "result_json_path",
            "result_summary_path",
            "result_path",
        ),
    )
    pack_id_key = _entry_first_text(entry, ("pack_id", "id"))
    source_hash_key = _entry_first_text(entry, ("source_hash",))

    identity = (
        account_key,
        pack_key,
        result_jsonl_key,
        result_json_key,
        pack_id_key,
        source_hash_key,
    )

    if any(identity):
        return identity

    extras: list[str] = []
    for key in sorted(entry.keys()):
        try:
            encoded = json.dumps(entry[key], sort_keys=True)
        except TypeError:
            encoded = json.dumps(str(entry[key]))
        extras.append(f"{key}:{encoded}")
    return tuple(extras)


def validation_findings_count(base_dir: Path) -> Optional[int]:
    """Return the number of validation findings written to disk for ``sid``."""

    index_path = base_dir / "ai_packs" / "validation" / "index.json"
    document = _load_document(index_path)
    if document is None:
        return None

    total = 0
    found = False

    for key in ("packs", "items"):
        entries = document.get(key)
        if not isinstance(entries, Sequence):
            continue
        for entry in entries:
            if not isinstance(entry, Mapping):
                continue
            lines = _coerce_int(entry.get("lines"))
            if lines is None:
                lines = _coerce_int(entry.get("line_count"))
            if lines is not None:
                total += lines
            found = True

    if found:
        return total

    totals = document.get("totals")
    if isinstance(totals, Mapping):
        for candidate in ("findings", "weak_count", "fields_built", "count"):
            value = _coerce_int(totals.get(candidate))
            if value is not None:
                return value

    fallback = _coerce_int(document.get("findings_count"))
    if fallback is not None:
        return fallback

    return None


def validation_packs_count(base_dir: Path) -> Optional[int]:
    """Return the number of unique validation packs written for ``sid``."""

    index_path = base_dir / "ai_packs" / "validation" / "index.json"
    document = _load_document(index_path)
    if document is None:
        return None

    packs_value = document.get("packs")
    if not isinstance(packs_value, Sequence):
        return None

    seen: set[tuple[str, ...]] = set()
    count = 0
    for entry in packs_value:
        if not isinstance(entry, Mapping):
            continue
        key = _validation_pack_entry_key(entry)
        if key in seen:
            continue
        seen.add(key)
        count += 1

    return count


def _has_review_attachments(payload: Mapping[str, Any]) -> bool:
    attachments = payload.get("attachments")
    if isinstance(attachments, Mapping):
        for value in attachments.values():
            if isinstance(value, str) and value.strip():
                return True
            if isinstance(value, Iterable) and not isinstance(
                value, (str, bytes, bytearray)
            ):
                for entry in value:
                    if isinstance(entry, str) and entry.strip():
                        return True

    legacy = payload.get("evidence")
    if isinstance(legacy, Iterable) and not isinstance(legacy, (str, bytes, bytearray)):
        for item in legacy:
            if not isinstance(item, Mapping):
                continue
            docs = item.get("docs")
            if isinstance(docs, Iterable) and not isinstance(
                docs, (str, bytes, bytearray)
            ):
                for doc in docs:
                    if isinstance(doc, Mapping):
                        doc_ids = doc.get("doc_ids")
                        if isinstance(doc_ids, Iterable) and not isinstance(
                            doc_ids, (str, bytes, bytearray)
                        ):
                            for doc_id in doc_ids:
                                if isinstance(doc_id, str) and doc_id.strip():
                                    return True
    return False


def frontend_packs_count(base_dir: Path) -> Optional[int]:
    """Return the number of frontend review packs written for ``sid``."""

    config = load_frontend_stage_config(base_dir)
    packs_dir = config.packs_dir

    if not packs_dir.exists() or not packs_dir.is_dir():
        return None

    try:
        return sum(
            1
            for entry in packs_dir.iterdir()
            if entry.is_file() and entry.suffix == ".json"
        )
    except OSError:
        return None


def frontend_answers_counters(
    base_dir: Path,
    *,
    attachments_required: bool,
) -> dict[str, int]:
    """Return frontend response answer counters rooted at ``base_dir``."""

    required = frontend_packs_count(base_dir) or 0

    config = load_frontend_stage_config(base_dir)
    responses_dir = config.responses_dir

    try:
        entries = sorted(
            path
            for path in responses_dir.iterdir()
            if path.is_file() and path.name.endswith(".result.json")
        )
    except OSError:
        entries = []

    answered_ids: set[str] = set()

    for entry in entries:
        payload = _load_document(entry)
        if payload is None:
            continue

        answers = payload.get("answers")
        if not isinstance(answers, Mapping):
            continue

        explanation = answers.get("explanation")
        if not isinstance(explanation, str) or not explanation.strip():
            continue

        if attachments_required and not _has_review_attachments(answers):
            continue

        received_at = payload.get("received_at")
        if not isinstance(received_at, str) or not received_at.strip():
            continue

        account_id = payload.get("account_id")
        if isinstance(account_id, str) and account_id.strip():
            answered_ids.add(account_id.strip())
        else:
            answered_ids.add(entry.stem)

    return {
        "answers_required": required,
        "answers_received": len(answered_ids),
        "answered_accounts": sorted(answered_ids),
    }


def merge_scored_pairs_count(base_dir: Path) -> Optional[int]:
    """Return the number of merge pairs scored for ``sid``."""

    merge_paths = ensure_merge_paths(base_dir.parent, base_dir.name, create=False)
    index_path = merge_paths.index_file
    document = _load_document(index_path)
    if document is None:
        return None

    totals = document.get("totals")
    if isinstance(totals, Mapping):
        value = _coerce_int(totals.get("scored_pairs"))
        if value is not None:
            return value

    fallback = _coerce_int(document.get("scored_pairs"))
    if fallback is not None:
        return fallback

    return None


def _normalize_note_style_result_name(name: str) -> str:
    text = name
    if text.endswith(".jsonl"):
        text = text[: -len(".jsonl")]
    if text.endswith(".json"):
        text = text[: -len(".json")]
    if text.endswith(".result"):
        text = text[: -len(".result")]
    if text.startswith("acc_"):
        text = text[len("acc_") :]
    return text


def _load_note_style_index_statuses(base_dir: Path) -> dict[str, str]:
    try:
        paths = ensure_note_style_paths(base_dir.parent, base_dir.name, create=False)
        index_path = paths.index_file
    except Exception:
        index_path = base_dir / "ai_packs" / "note_style" / "index.json"
    document = _load_document(index_path)
    if not isinstance(document, Mapping):
        return {}

    entries: Sequence[Mapping[str, Any]] = ()
    packs_payload = document.get("packs")
    if isinstance(packs_payload, Sequence):
        entries = [entry for entry in packs_payload if isinstance(entry, Mapping)]
    else:
        items_payload = document.get("items")
        if isinstance(items_payload, Sequence):
            entries = [entry for entry in items_payload if isinstance(entry, Mapping)]

    statuses: dict[str, str] = {}
    for entry in entries:
        account_value = entry.get("account_id")
        if not isinstance(account_value, str):
            continue
        statuses[account_value.strip()] = str(entry.get("status") or "")

    return statuses


def note_style_stage_counts(base_dir: Path) -> Optional[dict[str, int]]:
    """Return aggregate counters for note_style stage artifacts."""

    from backend.ai.note_style.io import note_style_stage_view

    view = note_style_stage_view(base_dir.name, runs_root=base_dir.parent)

    total = view.total_expected
    completed = view.completed_total
    failed = view.failed_total

    return {
        "packs_total": total,
        "packs_completed": completed,
        "packs_failed": failed,
    }


def runflow_validation_findings_total(base_dir: Path) -> Optional[int]:
    """Return the validation findings total recorded in ``runflow.json``."""

    runflow_path = base_dir / "runflow.json"
    document = _load_document(runflow_path)
    if document is None:
        return None

    stages = document.get("stages")
    if not isinstance(stages, Mapping):
        return None

    validation_stage = stages.get("validation")
    if not isinstance(validation_stage, Mapping):
        return None

    value = _coerce_int(validation_stage.get("findings_count"))
    if value is not None:
        return value

    summary = validation_stage.get("summary")
    if isinstance(summary, Mapping):
        fallback = _coerce_int(summary.get("findings_count"))
        if fallback is not None:
            return fallback

    return None


def stage_counts(stage: str, base_dir: Path) -> dict[str, int]:
    """Return authoritative counter mappings for ``stage`` rooted at ``base_dir``."""

    stage_key = str(stage)
    if stage_key == "validation":
        value = validation_findings_count(base_dir)
        return {"findings_count": value} if value is not None else {}
    if stage_key == "frontend":
        value = frontend_packs_count(base_dir)
        return {"packs_count": value} if value is not None else {}
    if stage_key == "merge":
        value = merge_scored_pairs_count(base_dir)
        return {"scored_pairs": value} if value is not None else {}
    if stage_key == "note_style":
        value = note_style_stage_counts(base_dir)
        return value if value is not None else {}
    return {}


__all__ = [
    "frontend_answers_counters",
    "frontend_packs_count",
    "merge_scored_pairs_count",
    "note_style_stage_counts",
    "stage_counts",
    "runflow_validation_findings_total",
    "validation_packs_count",
    "validation_findings_count",
]
if TYPE_CHECKING:
    from backend.ai.note_style.io import NoteStyleSnapshot

