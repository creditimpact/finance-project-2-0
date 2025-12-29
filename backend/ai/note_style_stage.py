"""Simplified builders for the note_style AI stage."""

from __future__ import annotations

import json
import logging
import os
import re
import time
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, MutableMapping, Sequence

from backend import config
from backend.config import note_style as note_cfg
from backend.ai.manifest import (
    ensure_note_style_section,
    register_note_style_build,
    update_note_style_stage_status,
)
from backend.ai.note_style.io import (
    NoteStyleSnapshot,
    NoteStyleStageView,
    note_style_snapshot,
    note_style_stage_view,
)
from backend.ai.note_style.prompt import NOTE_STYLE_SYSTEM
from backend.ai.note_style_logging import log_note_style_decision
from backend.core.ai.paths import (
    NoteStylePaths,
    ensure_note_style_account_paths,
    note_style_pack_filename,
    note_style_result_filename,
    normalize_note_style_account_id,
)
from backend.core.paths import normalize_stage_path
from backend.runflow.manifest import resolve_note_style_stage_paths
from backend.runflow.decider import record_stage


log = logging.getLogger(__name__)


_NOTE_STYLE_SYSTEM_PROMPT = NOTE_STYLE_SYSTEM

_NOTE_KEYS = {
    "note",
    "notes",
    "customer_note",
    "explain",
    "explanation",
}
_ZERO_WIDTH_TRANSLATION = {
    ord("\u200b"): " ",
    ord("\u200c"): " ",
    ord("\u200d"): " ",
    ord("\ufeff"): " ",
    ord("\u2060"): " ",
}

_MANIFEST_WAIT_ATTEMPTS = 20
_MANIFEST_WAIT_DELAY_SECONDS = 0.1


_LOGGED_PATH_STATES: set[str] = set()


def _log_resolved_paths(sid: str, paths: NoteStylePaths) -> None:
    signature = "|".join(
        [
            sid,
            paths.base.as_posix(),
            paths.packs_dir.as_posix(),
            paths.results_dir.as_posix(),
            paths.index_file.as_posix(),
            paths.log_file.as_posix(),
        ]
    )
    if signature in _LOGGED_PATH_STATES:
        return

    _LOGGED_PATH_STATES.add(signature)
    log.info(
        "NOTE_STYLE_STAGE_PATHS sid=%s base=%s packs=%s results=%s index=%s logs=%s manifest_paths=%s",
        sid,
        paths.base,
        paths.packs_dir,
        paths.results_dir,
        paths.index_file,
        paths.log_file,
        config.NOTE_STYLE_USE_MANIFEST_PATHS,
    )

@dataclass(frozen=True)
class NoteStyleResponseAccount:
    """Details about a frontend response discovered for the stage."""

    account_id: str
    normalized_account_id: str
    response_path: Path
    response_relative: PurePosixPath
    pack_filename: str
    result_filename: str


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _resolve_runs_root(runs_root: Path | str | None) -> Path:
    def _coerce(value: Path | str) -> Path:
        if isinstance(value, Path):
            return value.resolve()

        text = str(value or "").strip()
        if not text:
            return Path("runs").resolve()

        sanitized = text.replace("\\", "/")
        if len(sanitized) >= 2 and sanitized[1] == ":":
            try:
                return normalize_stage_path(Path("/"), sanitized)
            except ValueError:
                return Path("runs").resolve()

        candidate = Path(sanitized)
        if candidate.is_absolute():
            return candidate.resolve()

        return (Path.cwd() / candidate).resolve()

    if runs_root is None:
        env_root = os.getenv("RUNS_ROOT")
        if env_root:
            return _coerce(env_root)
        return Path("runs").resolve()

    return _coerce(runs_root)


def _load_json_data(path: Path) -> Any:
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    except OSError:
        log.warning("NOTE_STYLE_LOAD_JSON_FAILED path=%s", path, exc_info=True)
        return None

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        log.warning("NOTE_STYLE_LOAD_JSON_INVALID path=%s", path, exc_info=True)
        return None


def _extract_note_text(payload: Any) -> str:
    if isinstance(payload, str):
        text = payload.strip()
        return text

    if isinstance(payload, Mapping):
        for key in _NOTE_KEYS:
            candidate = payload.get(key)
            text = _extract_note_text(candidate)
            if text:
                return text
        for value in payload.values():
            if isinstance(value, Mapping) or (
                isinstance(value, Iterable) and not isinstance(value, (str, bytes, bytearray))
            ):
                text = _extract_note_text(value)
                if text:
                    return text
        return ""

    if isinstance(payload, Iterable) and not isinstance(payload, (bytes, bytearray)):
        for item in payload:
            text = _extract_note_text(item)
            if text:
                return text

    return ""


def _sanitize_note_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text or "")
    translated = normalized.translate(_ZERO_WIDTH_TRANSLATION)
    collapsed = " ".join(translated.split())
    return collapsed.strip()


_BUREAU_KEYS = ("transunion", "experian", "equifax")

_BUREAU_CORE_FIELDS = (
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


def _clean_display_text(value: Any) -> str | None:
    if isinstance(value, str):
        text = value.strip()
        return text or None
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _extract_meta_name(
    meta_payload: Mapping[str, Any] | None, account_id: str
) -> str:
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
    for field in _BUREAU_CORE_FIELDS:
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
    majority_payload = bureaus_payload.get("majority_values")
    majority_values = _filter_bureau_fields(majority_payload)
    if majority_values:
        return majority_values

    for bureau_key in _BUREAU_KEYS:
        bureau_values = _filter_bureau_fields(bureaus_payload.get(bureau_key))
        if bureau_values:
            return bureau_values
    return {}


def _extract_bureau_data(bureaus_payload: Mapping[str, Any] | None) -> Mapping[str, Any]:
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


def read_meta_name(path: Path) -> str:
    account_id = ""
    if isinstance(path, Path):
        account_id = path.parent.name
        meta_payload = _load_json_data(path)
    else:  # pragma: no cover - defensive
        meta_payload = None

    if isinstance(meta_payload, Mapping):
        heading_guess = _clean_display_text(meta_payload.get("heading_guess"))
        if heading_guess:
            return heading_guess

        for key in ("account_id", "id"):
            fallback_candidate = _clean_display_text(meta_payload.get(key))
            if fallback_candidate:
                return fallback_candidate

    fallback = _clean_display_text(account_id)
    if fallback:
        return fallback

    if account_id:
        return str(account_id)

    return "--"


def read_primary_issue_tag(path: Path) -> str | None:
    tags_payload = _load_json_data(path) if isinstance(path, Path) else None
    return _extract_primary_issue_tag(tags_payload)


def build_bureau_data(path: Path) -> dict[str, Any]:
    default = {field: "--" for field in _BUREAU_CORE_FIELDS}
    bureaus_payload = _load_json_data(path) if isinstance(path, Path) else None
    if not isinstance(bureaus_payload, Mapping):
        return default

    start_index: int | None = None
    for idx, bureau_key in enumerate(_BUREAU_KEYS):
        if isinstance(bureaus_payload.get(bureau_key), Mapping):
            start_index = idx
            break

    if start_index is None:
        return default

    ordered_bureaus: list[Mapping[str, Any]] = []
    for bureau_key in _BUREAU_KEYS[start_index:]:
        bureau_payload = bureaus_payload.get(bureau_key)
        if isinstance(bureau_payload, Mapping):
            ordered_bureaus.append(bureau_payload)

    result: dict[str, Any] = {}
    for field in _BUREAU_CORE_FIELDS:
        selected_value: Any | None = None
        for bureau_payload in ordered_bureaus:
            candidate = bureau_payload.get(field)
            if candidate is None:
                continue
            if isinstance(candidate, str):
                if candidate.strip():
                    selected_value = candidate
                    break
                continue
            selected_value = candidate
            break

        if selected_value is None:
            result[field] = "--"
        else:
            result[field] = selected_value

    return {field: result.get(field, "--") for field in _BUREAU_CORE_FIELDS}


def _resolve_response_path(sid: str, account_id: str, runs_root: Path) -> Path:
    return (runs_root / sid / "frontend" / "review" / "responses" / f"{account_id}.result.json").resolve()


def _coerce_raw_path(value: Any) -> Path | None:
    if value is None:
        return None
    try:
        raw = os.fspath(value)
    except TypeError:
        return None

    text = str(raw).strip()
    if not text:
        return None

    return Path(text)


def _resolve_path_from_bases(raw_path: Path, bases: Iterable[Path]) -> Path:
    if raw_path.is_absolute():
        try:
            return raw_path.resolve()
        except OSError:
            return raw_path

    candidates: list[Path] = []
    for base in bases:
        if base is None:
            continue
        try:
            candidate = Path(base) / raw_path
        except TypeError:
            continue
        candidates.append(candidate)

    if not candidates:
        candidates.append(raw_path)

    first_candidate = candidates[0]
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except OSError:
            resolved = candidate
        if resolved.exists():
            return resolved
        if candidate.exists():
            return candidate

    try:
        return first_candidate.resolve()
    except OSError:
        return first_candidate


def _await_path_ready(path: Path) -> Path:
    candidate = path
    for attempt in range(_MANIFEST_WAIT_ATTEMPTS):
        if candidate.exists():
            break
        if attempt < _MANIFEST_WAIT_ATTEMPTS - 1:
            time.sleep(_MANIFEST_WAIT_DELAY_SECONDS)
    try:
        return candidate.resolve()
    except OSError:
        return candidate


def _lookup_manifest_account_entry(
    manifest_payload: Any, account_id: str
) -> Mapping[str, Any] | None:
    if not isinstance(manifest_payload, Mapping):
        return None

    artifacts = manifest_payload.get("artifacts")
    if not isinstance(artifacts, Mapping):
        return None

    cases_section = artifacts.get("cases")
    if not isinstance(cases_section, Mapping):
        return None

    accounts_section = cases_section.get("accounts")
    if not isinstance(accounts_section, Mapping):
        return None

    normalized = str(account_id).strip()
    raw_candidates: list[str] = []
    if normalized:
        raw_candidates.append(normalized)
        raw_candidates.append(normalized.lower())

    for piece in re.findall(r"(\d+)", normalized):
        trimmed = piece.lstrip("0") or "0"
        raw_candidates.append(trimmed)
        raw_candidates.append(trimmed.lower())

    candidates = [candidate for candidate in dict.fromkeys(raw_candidates) if candidate]

    for candidate in candidates:
        entry = accounts_section.get(candidate)
        if isinstance(entry, Mapping):
            return entry

        for key, value in accounts_section.items():
            if isinstance(key, str) and key.lower() == candidate.lower() and isinstance(value, Mapping):
                return value

    return None


def _await_manifest_account_entry(
    sid: str, account_id: str, run_dir: Path
) -> Mapping[str, Any]:
    manifest_path = run_dir / "manifest.json"
    manifest_payload: Any = None
    account_entry: Mapping[str, Any] | None = None

    for attempt in range(_MANIFEST_WAIT_ATTEMPTS):
        manifest_payload = _load_json_data(manifest_path)
        if isinstance(manifest_payload, Mapping):
            account_entry = _lookup_manifest_account_entry(manifest_payload, account_id)
            if isinstance(account_entry, Mapping):
                return account_entry
        if attempt < _MANIFEST_WAIT_ATTEMPTS - 1:
            time.sleep(_MANIFEST_WAIT_DELAY_SECONDS)

    if not isinstance(manifest_payload, Mapping):
        log.warning(
            "NOTE_STYLE_MANIFEST_MISSING sid=%s path=%s", sid, manifest_path
        )
        raise RuntimeError(
            f"note_style manifest missing for sid={sid} path={manifest_path}"
        )

    log.warning(
        "NOTE_STYLE_MANIFEST_ACCOUNT_MISSING sid=%s account_id=%s", sid, account_id
    )
    raise RuntimeError(
        f"note_style manifest missing account entry for sid={sid} account_id={account_id}"
    )


def _resolve_account_context_paths(
    sid: str, account_id: str, runs_root: Path
) -> dict[str, Path]:
    run_dir = runs_root / sid
    account_entry = _await_manifest_account_entry(sid, account_id, run_dir)

    resolved: dict[str, Path] = {}
    account_dir: Path | None = None
    raw_dir = _coerce_raw_path(account_entry.get("dir"))
    if raw_dir is not None:
        account_dir = _resolve_path_from_bases(raw_dir, (run_dir,))

    if account_dir is None:
        account_dir = run_dir / "cases" / "accounts" / account_id

    account_dir = _await_path_ready(account_dir)
    resolved["dir"] = account_dir

    for key, filename in (
        ("meta", "meta.json"),
        ("bureaus", "bureaus.json"),
        ("tags", "tags.json"),
    ):
        raw_value = None
        raw_value = _coerce_raw_path(account_entry.get(key))
        if raw_value is not None:
            candidate = _resolve_path_from_bases(raw_value, (account_dir, run_dir))
        else:
            candidate = account_dir / filename
        resolved[key] = _await_path_ready(candidate)

    return resolved


def _load_context_payload(path: Path | None) -> tuple[Any | None, bool]:
    if not isinstance(path, Path):
        return None, True

    payload = _load_json_data(path)
    missing = payload is None
    return payload, missing


def _relative_to_base(path: Path, base: Path) -> str:
    try:
        return path.resolve().relative_to(base.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _ensure_index_entry(
    *,
    paths: NoteStylePaths,
    account_id: str,
    pack_path: Path,
    result_path: Path,
    timestamp: str,
) -> Mapping[str, Any]:
    index_path = paths.index_file
    packs: list[MutableMapping[str, Any]] = []
    index_payload: MutableMapping[str, Any]

    existing = _load_json_data(index_path)
    if isinstance(existing, Mapping):
        existing_packs = existing.get("packs")
        if isinstance(existing_packs, list):
            for entry in existing_packs:
                if isinstance(entry, Mapping) and str(entry.get("account_id")) != account_id:
                    packs.append(dict(entry))

    packs.append(
        {
            "account_id": account_id,
            "status": "built",
            "pack_path": _relative_to_base(pack_path, paths.base),
            "result_path": _relative_to_base(result_path, paths.base),
            "built_at": timestamp,
        }
    )
    packs.sort(key=lambda entry: str(entry.get("account_id")))

    index_payload = {
        "version": 1,
        "updated_at": timestamp,
        "packs": packs,
        "totals": {"packs_total": len(packs)},
    }

    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text(json.dumps(index_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return index_payload


def _write_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    serialized = json.dumps(payload, ensure_ascii=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(serialized + "\n", encoding="utf-8")


def discover_note_style_response_accounts(
    sid: str, *, runs_root: Path | str | None = None
) -> list[NoteStyleResponseAccount]:
    runs_root_path = _resolve_runs_root(runs_root)
    responses_dir = (runs_root_path / sid / "frontend" / "review" / "responses").resolve()

    if not responses_dir.is_dir():
        log.info("NOTE_STYLE_DISCOVERY sid=%s responses=%s usable=%s", sid, 0, 0)
        return []

    discovered: list[NoteStyleResponseAccount] = []
    total = 0
    usable = 0
    for candidate in sorted(responses_dir.glob("*.result.json"), key=lambda item: item.name):
        if not candidate.is_file():
            continue
        total += 1
        payload = _load_json_data(candidate)
        if not isinstance(payload, Mapping):
            continue
        note_text = _sanitize_note_text(_extract_note_text(payload))
        if not note_text:
            continue
        account_id = candidate.stem.replace(".result", "")
        normalized = normalize_note_style_account_id(account_id)
        pack_filename = note_style_pack_filename(account_id)
        result_filename = note_style_result_filename(account_id)
        relative = PurePosixPath(_relative_to_base(candidate, runs_root_path))
        discovered.append(
            NoteStyleResponseAccount(
                account_id=account_id,
                normalized_account_id=normalized,
                response_path=candidate.resolve(),
                response_relative=relative,
                pack_filename=pack_filename,
                result_filename=result_filename,
            )
        )
        usable += 1

    discovered.sort(key=lambda entry: entry.account_id)
    log.info(
        "NOTE_STYLE_DISCOVERY sid=%s responses=%s usable=%s",
        sid,
        total,
        usable,
    )
    return discovered


def _record_stage_snapshot(
    *,
    sid: str,
    runs_root: Path,
    index_payload: Mapping[str, Any],
) -> NoteStyleStageView:
    _ = index_payload  # preserved for compatibility with existing call sites
    snapshot = note_style_snapshot(sid, runs_root=runs_root)
    view = note_style_stage_view(sid, runs_root=runs_root, snapshot=snapshot)

    counts_payload = {"packs_total": view.total_expected}
    metrics_payload = {
        "packs_total": view.total_expected,
        "packs_built": view.built_total,
    }
    results_payload = {
        "results_total": view.total_expected,
        "completed": view.completed_total,
        "failed": view.failed_total,
    }

    try:
        record_stage(
            sid,
            "note_style",
            status=view.state,
            counts=counts_payload,
            empty_ok=view.total_expected == 0,
            metrics=metrics_payload,
            results=results_payload,
            runs_root=runs_root,
        )
    except Exception:  # pragma: no cover - defensive logging
        log.exception("NOTE_STYLE_STAGE_RECORD_FAILED sid=%s", sid)

    log_note_style_decision(
        "NOTE_STYLE_STAGE_SNAPSHOT",
        logger=log,
        sid=sid,
        runs_root=runs_root,
        view=view,
        reason="record_stage_snapshot",
    )

    return view


def build_note_style_pack_for_account(
    sid: str, account_id: str, *, runs_root: Path | str | None = None
) -> Mapping[str, Any]:
    runs_root_path = _resolve_runs_root(runs_root)
    ensure_note_style_section(sid, runs_root=runs_root_path)
    paths = resolve_note_style_stage_paths(runs_root_path, sid, create=True)
    _log_resolved_paths(sid, paths)
    log.info(
        "NOTE_STYLE mode: %s, allow_tool_calls=%s",
        getattr(note_cfg.NOTE_STYLE_RESPONSE_MODE, "value", note_cfg.NOTE_STYLE_RESPONSE_MODE),
        note_cfg.NOTE_STYLE_ALLOW_TOOL_CALLS,
    )

    response_path = _resolve_response_path(sid, account_id, runs_root_path)
    payload = _load_json_data(response_path)
    if not isinstance(payload, Mapping):
        log.info(
            "NOTE_STYLE_BUILD_SKIP sid=%s account_id=%s reason=no_response", sid, account_id
        )
        return {"status": "skipped", "reason": "no_response"}

    note_text = _sanitize_note_text(_extract_note_text(payload))
    if not note_text:
        log.info(
            "NOTE_STYLE_BUILD_SKIP sid=%s account_id=%s reason=no_note", sid, account_id
        )
        return {"status": "skipped", "reason": "no_note"}

    context_paths = _resolve_account_context_paths(sid, account_id, runs_root_path)
    meta_path = context_paths.get("meta")
    bureaus_path = context_paths.get("bureaus")
    tags_path = context_paths.get("tags")

    account_paths = ensure_note_style_account_paths(paths, account_id, create=True)

    _, meta_missing = _load_context_payload(meta_path)
    _, bureaus_missing = _load_context_payload(bureaus_path)
    _, tags_missing = _load_context_payload(tags_path)
    if meta_missing or bureaus_missing or tags_missing:
        missing_parts: list[str] = []
        if meta_missing:
            missing_parts.append("meta")
        if tags_missing:
            missing_parts.append("tags")
        if bureaus_missing:
            missing_parts.append("bureaus")
        log.warning(
            "NOTE_STYLE_WARN: missing context for account %s (%s)",
            account_id,
            "/".join(missing_parts),
        )

    meta_name = read_meta_name(meta_path)
    primary_issue_tag = read_primary_issue_tag(tags_path)
    bureau_data = build_bureau_data(bureaus_path)
    timestamp = _now_iso()

    context_payload: dict[str, Any] = {
        "meta_name": meta_name,
        "primary_issue_tag": primary_issue_tag,
        "bureau_data": bureau_data,
        "note_text": note_text,
    }

    user_content = json.dumps(context_payload, ensure_ascii=False)

    messages = [
        {"role": "system", "content": _NOTE_STYLE_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    pack_payload = {
        "meta_name": meta_name,
        "primary_issue_tag": primary_issue_tag,
        "bureau_data": bureau_data,
        "note_text": note_text,
        "messages": messages,
    }
    _write_jsonl(account_paths.pack_file, pack_payload)
    if account_paths.result_file.exists():
        try:
            account_paths.result_file.unlink()
        except OSError:
            log.warning(
                "NOTE_STYLE_RESULT_CLEANUP_FAILED sid=%s account_id=%s path=%s",
                sid,
                account_id,
                account_paths.result_file,
                exc_info=True,
            )
    if account_paths.debug_file.exists():
        try:
            account_paths.debug_file.unlink()
        except OSError:
            log.warning(
                "NOTE_STYLE_DEBUG_CLEANUP_FAILED sid=%s account_id=%s path=%s",
                sid,
                account_id,
                account_paths.debug_file,
                exc_info=True,
            )
    index_payload = _ensure_index_entry(
        paths=paths,
        account_id=account_id,
        pack_path=account_paths.pack_file,
        result_path=account_paths.result_file,
        timestamp=timestamp,
    )
    view = _record_stage_snapshot(
        sid=sid, runs_root=runs_root_path, index_payload=index_payload
    )

    log.info(
        "✅ Pack built for %s (%s), chars=%s, words=%s",
        account_id,
        meta_name,
        len(note_text),
        len(note_text.split()),
    )

    try:
        register_note_style_build(sid, runs_root=runs_root_path)
    except Exception:  # pragma: no cover - defensive logging
        log.warning(
            "NOTE_STYLE_MANIFEST_BUILD_REGISTER_FAILED sid=%s account_id=%s",
            sid,
            account_id,
            exc_info=True,
        )

    try:
        update_note_style_stage_status(
            sid,
            runs_root=runs_root_path,
            built=view.built_complete,
            sent=False,
            completed_at=None,
            state=view.state,
        )
    except Exception:  # pragma: no cover - defensive logging
        log.warning(
            "NOTE_STYLE_MANIFEST_STAGE_PRIME_FAILED sid=%s account_id=%s",
            sid,
            account_id,
            exc_info=True,
        )

    return {
        "status": "completed",
        "packs_total": index_payload.get("totals", {}).get("packs_total", 1),
    }


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


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return unicodedata.normalize("NFKC", value).strip()
    return unicodedata.normalize("NFKC", str(value)).strip()


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


def schedule_note_style_refresh(
    sid: str, account_id: str, *, runs_root: Path | str | None = None
) -> None:
    if not config.NOTE_STYLE_ENABLED:
        log.info("NOTE_STYLE_DISABLED sid=%s account_id=%s", sid, account_id)
        return
    try:
        build_note_style_pack_for_account(sid, account_id, runs_root=runs_root)
    except Exception:  # pragma: no cover - defensive logging
        log.exception("NOTE_STYLE_REFRESH_FAILED sid=%s account_id=%s", sid, account_id)


__all__ = [
    "NoteStyleResponseAccount",
    "discover_note_style_response_accounts",
    "build_note_style_pack_for_account",
    "schedule_note_style_refresh",
]
