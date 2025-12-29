"""Frontend pack generation helpers."""

from __future__ import annotations

import glob
import hashlib
import json
import logging
import os
import re
import shutil
import tempfile
import time
from collections import Counter
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from backend.core.io.json_io import _atomic_write_json as _shared_atomic_write_json
from backend.core.paths.frontend_review import (
    ensure_frontend_review_dirs,
    get_frontend_review_paths,
)
from backend.core.runflow import (
    record_frontend_responses_progress,
    runflow_account_steps_enabled,
    runflow_step,
)
from backend.core.runflow.io import (
    compose_hint,
    format_exception_tail,
    runflow_stage_end,
    runflow_stage_error,
    runflow_stage_start,
)
from backend.frontend.packs.config import (
    FrontendStageConfig,
    load_frontend_stage_config,
)
from backend.domain.claims import CLAIM_FIELD_LINK_MAP

log = logging.getLogger(__name__)

_BUREAU_BADGES: Mapping[str, Mapping[str, str]] = {
    "transunion": {"id": "transunion", "label": "TransUnion", "short_label": "TU"},
    "equifax": {"id": "equifax", "label": "Equifax", "short_label": "EF"},
    "experian": {"id": "experian", "label": "Experian", "short_label": "EX"},
}

_BUREAU_ORDER: tuple[str, ...] = ("transunion", "experian", "equifax")

_BUREAU_SHORT_CODES: Mapping[str, str] = {
    "transunion": "tu",
    "experian": "ex",
    "equifax": "eq",
}

_DISPLAY_SCHEMA_VERSION = "1.3"

_STAGE_PAYLOAD_MODE_MINIMAL = "minimal"
_STAGE_PAYLOAD_MODE_FULL = "full"
_STAGE_PAYLOAD_MODE_LEGACY = "legacy"
_STAGE_PAYLOAD_MODES: set[str] = {
    _STAGE_PAYLOAD_MODE_MINIMAL,
    _STAGE_PAYLOAD_MODE_FULL,
    _STAGE_PAYLOAD_MODE_LEGACY,
}


_QUESTION_SET = [
    {"id": "ownership", "prompt": "Do you own this account?"},
    {"id": "recognize", "prompt": "Do you recognize this account on your report?"},
    {"id": "explanation", "prompt": "Anything else we should know about this account?"},
    {"id": "identity_theft", "prompt": "Is this account tied to identity theft?"},
]


def _env_flag_enabled(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    lowered = value.strip().lower()
    if lowered in {"", "0", "false", "no", "off"}:
        return False
    return True


def _claim_field_links_payload() -> dict[str, list[str]]:
    return {key: list(values) for key, values in CLAIM_FIELD_LINK_MAP.items()}


def _coerce_question_list(questions: Any) -> list[dict[str, Any]]:
    if not isinstance(questions, Sequence) or isinstance(
        questions, (str, bytes, bytearray)
    ):
        return []

    normalized: list[dict[str, Any]] = []
    for question in questions:
        if isinstance(question, Mapping):
            normalized.append(dict(question))

    return normalized


def _resolve_stage_pack_questions(
    *,
    existing_pack: Mapping[str, Any] | None,
    question_set: Sequence[Mapping[str, Any]] | None,
) -> list[dict[str, Any]]:
    if isinstance(existing_pack, Mapping):
        existing_questions = _coerce_question_list(existing_pack.get("questions"))
        if existing_questions:
            return existing_questions

    if question_set is None:
        return []

    return _coerce_question_list(question_set)


def _normalize_claim_field_links(payload: Any) -> dict[str, list[str]]:
    if not isinstance(payload, Mapping):
        return {}

    normalized: dict[str, list[str]] = {}
    for key, values in payload.items():
        if not isinstance(key, str):
            continue
        collected: list[str] = []
        if isinstance(values, Sequence) and not isinstance(values, (str, bytes, bytearray)):
            for value in values:
                if not isinstance(value, str):
                    continue
                trimmed = value.strip()
                if not trimmed or trimmed in collected:
                    continue
                collected.append(trimmed)
        if collected:
            normalized[key] = collected
    return normalized


def _merge_claim_field_links(
    *sources: Mapping[str, Any] | None,
) -> dict[str, list[str]]:
    merged: dict[str, list[str]] = {}
    for source in sources:
        normalized = _normalize_claim_field_links(source)
        for key, values in normalized.items():
            bucket = merged.setdefault(key, [])
            for value in values:
                if value not in bucket:
                    bucket.append(value)
    return merged


_POINTER_KEYS: tuple[str, ...] = (
    "meta",
    "tags",
    "raw",
    "bureaus",
    "flat",
    "summary",
)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _default_runs_root() -> Path:
    root_env = os.getenv("RUNS_ROOT")
    return Path(root_env) if root_env else Path("runs")


def _resolve_runs_root(runs_root: Path | str | None) -> Path:
    if runs_root is None:
        return _default_runs_root()
    return Path(runs_root)


def _frontend_packs_enabled() -> bool:
    value = os.getenv("ENABLE_FRONTEND_PACKS", "1")
    return value not in {"0", "false", "False"}


def _frontend_packs_lean_enabled() -> bool:
    value = os.getenv("FRONTEND_PACKS_LEAN", "1")
    return value not in {"0", "false", "False"}


def _frontend_packs_debug_mirror_enabled() -> bool:
    value = os.getenv("FRONTEND_PACKS_DEBUG_MIRROR", "0")
    return value not in {"0", "false", "False"}


def _frontend_use_bureaus_only_enabled() -> bool:
    return _env_flag_enabled("FRONTEND_USE_BUREAUS_JSON_ONLY", False)


def _frontend_warn_bureaus_conflict_enabled() -> bool:
    return _env_flag_enabled("FRONTEND_WARN_BUREAUS_CONFLICT", False)


def _resolve_stage_payload_mode() -> str:
    value = os.getenv("FRONTEND_STAGE_PAYLOAD", _STAGE_PAYLOAD_MODE_MINIMAL)
    if not value:
        return _STAGE_PAYLOAD_MODE_MINIMAL

    normalized = value.strip().lower()
    if normalized not in _STAGE_PAYLOAD_MODES:
        log.warning(
            "FRONTEND_STAGE_PAYLOAD_INVALID value=%s", value,
        )
        return _STAGE_PAYLOAD_MODE_MINIMAL

    return normalized


def _frontend_review_create_empty_index_enabled() -> bool:
    return _env_flag_enabled("FRONTEND_REVIEW_CREATE_EMPTY_INDEX", False)


def _log_stage_paths(
    sid: str,
    config: FrontendStageConfig,
    canonical_paths: Mapping[str, str],
) -> None:
    base_path = canonical_paths.get("frontend_base") or config.stage_dir.parent
    log.info(
        "FRONTEND_REVIEW_PATHS sid=%s base=%s dir=%s packs=%s results=%s",
        sid,
        str(base_path),
        str(config.stage_dir),
        str(config.packs_dir),
        str(config.responses_dir),
    )


def _frontend_build_lock_path(run_dir: Path) -> Path:
    return run_dir / "frontend" / ".locks" / "build.lock"


def _acquire_frontend_build_lock(run_dir: Path, sid: str) -> tuple[str, Path | None]:
    lock_path = _frontend_build_lock_path(run_dir)

    try:
        lock_path.parent.mkdir(parents=True, exist_ok=True)
    except Exception:  # pragma: no cover - defensive logging
        log.warning(
            "FRONTEND_BUILD_LOCK_DIR_FAILED sid=%s path=%s",
            sid,
            lock_path.parent,
            exc_info=True,
        )

    try:
        fd = os.open(os.fspath(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return "locked", lock_path
    except Exception:  # pragma: no cover - defensive logging
        log.warning(
            "FRONTEND_BUILD_LOCK_ACQUIRE_FAILED sid=%s path=%s",
            sid,
            lock_path,
            exc_info=True,
        )
        return "error", None

    payload = {
        "sid": sid,
        "acquired_at": time.time(),
        "pid": os.getpid(),
    }

    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False)
    except Exception:  # pragma: no cover - defensive logging
        log.warning(
            "FRONTEND_BUILD_LOCK_WRITE_FAILED sid=%s path=%s",
            sid,
            lock_path,
            exc_info=True,
        )
        try:
            os.close(fd)
        except Exception:
            pass
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass
        except Exception:  # pragma: no cover - defensive logging
            log.warning(
                "FRONTEND_BUILD_LOCK_CLEANUP_FAILED sid=%s path=%s",
                sid,
                lock_path,
                exc_info=True,
            )
        return "error", None

    return "acquired", lock_path


def _release_frontend_build_lock(lock_path: Path, sid: str) -> None:
    try:
        lock_path.unlink()
    except FileNotFoundError:
        return
    except Exception:  # pragma: no cover - defensive logging
        log.warning(
            "FRONTEND_BUILD_LOCK_RELEASE_FAILED sid=%s path=%s",
            sid,
            lock_path,
            exc_info=True,
        )


def _resolve_idempotent_lock_path(run_dir: Path) -> Path | None:
    value = os.getenv("FRONTEND_IDEMPOTENT_LOCK_REL")
    if value:
        candidate = Path(value)
        if not candidate.is_absolute():
            candidate = run_dir / candidate
        return candidate

    return _frontend_build_lock_path(run_dir)


def _lock_mtime(lock_path: Path | None) -> float | None:
    if lock_path is None:
        return None
    try:
        return lock_path.stat().st_mtime
    except FileNotFoundError:
        return None
    except OSError:  # pragma: no cover - defensive logging
        log.warning("FRONTEND_LOCK_STAT_FAILED path=%s", lock_path, exc_info=True)
        return None


def _should_skip_pack_due_to_lock(
    *,
    stage_pack_path: Path,
    lock_path: Path | None,
    lock_mtime: float | None,
) -> bool:
    if lock_path is None or lock_mtime is None:
        return False

    if not stage_pack_path.exists():
        return False

    try:
        pack_mtime = stage_pack_path.stat().st_mtime
    except OSError:  # pragma: no cover - defensive logging
        log.warning(
            "FRONTEND_PACK_STAT_FAILED path=%s lock=%s",
            stage_pack_path,
            lock_path,
            exc_info=True,
        )
        return False

    return pack_mtime > lock_mtime


def _log_build_summary(
    sid: str,
    *,
    packs_count: int,
    last_built_at: str | None,
) -> None:
    log.info(
        "FRONTEND_REVIEW_BUILD_COMPLETE sid=%s packs_count=%s last_built_at=%s",
        sid,
        packs_count,
        last_built_at or "-",
    )


def _count_frontend_responses(responses_dir: Path) -> int:
    if not responses_dir.is_dir():
        return 0

    total = 0
    for entry in responses_dir.iterdir():
        if not entry.is_file():
            continue
        if entry.name.endswith(".tmp"):
            continue
        total += 1
    return total


def _emit_responses_scan(_sid: str, responses_dir: Path) -> int:
    return _count_frontend_responses(responses_dir)


def _account_sort_key(path: Path) -> tuple[int, Any]:
    name = path.name
    if name.isdigit():
        return (0, int(name))
    return (1, name)


def _load_json(path: Path) -> Mapping[str, Any] | None:
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    except OSError:  # pragma: no cover - defensive logging
        log.warning("FRONTEND_PACK_READ_FAILED path=%s", path, exc_info=True)
        return None

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        log.warning("FRONTEND_PACK_PARSE_FAILED path=%s", path, exc_info=True)
        return None

    if not isinstance(payload, Mapping):
        return None
    return payload


def _load_json_payload(path: Path) -> Any:
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    except OSError:  # pragma: no cover - defensive logging
        log.warning("FRONTEND_PACK_READ_FAILED path=%s", path, exc_info=True)
        return None

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        log.warning("FRONTEND_PACK_PARSE_FAILED path=%s", path, exc_info=True)
        return None


def load_bureaus_meta_tags(
    account_dir: Path,
) -> tuple[Mapping[str, Any], Mapping[str, Any] | None, Any, dict[str, str]]:
    """Load bureau, meta, and tag artifacts for ``account_dir``."""

    bureaus_path = account_dir / "bureaus.json"
    bureaus_payload = _load_json(bureaus_path)
    if not isinstance(bureaus_payload, Mapping):
        raise FileNotFoundError(bureaus_path)

    meta_path = account_dir / "meta.json"
    meta_payload = _load_json(meta_path)

    tags_path = account_dir / "tags.json"
    tags_payload = _load_json_payload(tags_path)

    pointers = {
        "bureaus": bureaus_path.as_posix(),
        "meta": meta_path.as_posix(),
        "tags": tags_path.as_posix(),
    }

    return bureaus_payload, meta_payload, tags_payload, pointers


def _is_frontend_review_index(path: Path) -> bool:
    normalized = path.as_posix()
    return normalized.endswith("frontend/review/index.json")


def _atomic_write_frontend_review_index(path: Path, payload: Any) -> None:
    directory = path.parent
    os.makedirs(directory, exist_ok=True)

    attempts = 0
    last_error: OSError | None = None
    while attempts < 2:
        attempts += 1
        fd: int | None = None
        tmp_path: Path | None = None
        try:
            fd, tmp_raw_path = tempfile.mkstemp(
                prefix=f"{path.name}.", dir=directory, text=True
            )
            tmp_path = Path(tmp_raw_path)
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                fd = None  # File descriptor handled by os.fdopen context manager.
                json.dump(payload, handle, ensure_ascii=False, indent=2)
            os.replace(str(tmp_path), str(path))
        except FileNotFoundError as exc:
            last_error = exc
            log.warning(
                "FRONTEND_REVIEW_INDEX_RETRY path=%s tmp=%s error=%s",
                path,
                tmp_path,
                exc,
            )
            if tmp_path is not None:
                try:
                    if tmp_path.exists():
                        tmp_path.unlink()
                except FileNotFoundError:
                    pass
            if attempts >= 2:
                break
            time.sleep(0.05)
            continue
        except OSError as exc:
            last_error = exc
            if attempts >= 2:
                break
            if tmp_path is not None:
                try:
                    if tmp_path.exists():
                        tmp_path.unlink()
                except FileNotFoundError:
                    pass
            time.sleep(0.05)
            continue
        finally:
            if fd is not None:
                try:
                    os.close(fd)
                except OSError:
                    pass
            if tmp_path is not None:
                try:
                    if tmp_path.exists():
                        tmp_path.unlink()
                except FileNotFoundError:
                    pass
        return

    if last_error is not None:
        raise last_error


def _atomic_write_json(path: Path | str, payload: Any) -> None:
    path_obj = Path(path)
    if _is_frontend_review_index(path_obj):
        _atomic_write_frontend_review_index(path_obj, payload)
        return
    _shared_atomic_write_json(path_obj, payload)


def _write_json_if_changed(path: Path, payload: Any) -> bool:
    current = _load_json_payload(path)
    if current == payload:
        return False

    try:
        _atomic_write_json(path, payload)
    except FileNotFoundError as exc:
        log.warning(
            "FRONTEND_STAGE_ATOMIC_WRITE_FALLBACK path=%s error=%s",
            path,
            exc,
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
    return True


def _enrich_stage_payload_with_full(
    candidate: Mapping[str, Any] | None,
    full_payload: Mapping[str, Any] | None,
) -> tuple[dict[str, Any], bool]:
    if isinstance(candidate, dict):
        merged: dict[str, Any] = candidate
    elif isinstance(candidate, Mapping):
        merged = dict(candidate)
    else:
        merged = {}

    if not isinstance(full_payload, Mapping):
        return merged, False

    changed = False

    def _copy_text(key: str, *, treat_unknown: bool = True) -> None:
        nonlocal changed
        candidate_value = merged.get(key)
        if _has_meaningful_text(candidate_value, treat_unknown=treat_unknown):
            return
        source_value = full_payload.get(key)
        if not _has_meaningful_text(source_value, treat_unknown=treat_unknown):
            return
        merged[key] = source_value
        changed = True

    for field_name in ("holder_name", "primary_issue", "creditor_name", "account_type", "status"):
        _copy_text(field_name, treat_unknown=True)

    for key in ("last4", "balance_owed", "dates", "bureau_badges"):
        source_section = full_payload.get(key)
        if not isinstance(source_section, Mapping):
            if key not in merged and source_section is not None:
                merged[key] = source_section
                changed = True
            continue

        candidate_section = merged.get(key)
        if isinstance(candidate_section, Mapping):
            continue
        merged[key] = dict(source_section)
        changed = True

    display_candidate = merged.get("display")
    display_enriched, display_changed = _preserve_stage_display_values(
        full_payload.get("display"), display_candidate
    )
    if display_changed or (display_candidate is None and isinstance(display_enriched, dict)):
        merged["display"] = display_enriched
        if display_changed:
            changed = True

    if "claim_field_links" not in merged and isinstance(
        full_payload.get("claim_field_links"), Mapping
    ):
        merged["claim_field_links"] = dict(full_payload["claim_field_links"])
        changed = True

    if "pointers" not in merged and isinstance(full_payload.get("pointers"), Mapping):
        merged["pointers"] = dict(full_payload["pointers"])
        changed = True

    return merged, changed


def _ensure_frontend_index_redirect_stub(path: Path, *, force: bool = False) -> None:
    """Write the legacy ``frontend/index.json`` redirect if it is missing."""

    if path.exists() and not force:
        return

    # Backward-compatibility stub for clients still reading the legacy path.
    path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(path, {"redirect": "frontend/review/index.json"})


def _relative_to_run_dir(path: Path, run_dir: Path) -> str:
    try:
        return path.relative_to(run_dir).as_posix()
    except ValueError:
        return path.as_posix()


def _relative_to_stage_dir(path: Path, stage_dir: Path) -> str:
    try:
        return path.relative_to(stage_dir).as_posix()
    except ValueError:
        return path.as_posix()


def _optional_str(value: Any) -> str | None:
    if isinstance(value, str):
        trimmed = value.strip()
        return trimmed or None
    return None


def _safe_sha1(path: Path) -> str | None:
    try:
        digest = hashlib.sha1()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8192), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:  # pragma: no cover - defensive logging
        log.warning("FRONTEND_PACK_SHA1_FAILED path=%s", path, exc_info=True)
        return None


def _resolve_pack_output_path(pack_path: str, run_dir: Path) -> Path:
    candidate = Path(pack_path)
    if not candidate.is_absolute():
        candidate = run_dir / candidate
    return candidate


def _pack_requires_pointer_backfill(payload: Mapping[str, Any]) -> bool:
    pointers = payload.get("pointers") if isinstance(payload, Mapping) else None
    if pointers is None:
        return False
    if not isinstance(pointers, Mapping):
        return True

    for key in _POINTER_KEYS:
        value = pointers.get(key)
        if not isinstance(value, str) or not value:
            return True

    return False


def _index_requires_pointer_backfill(
    index_payload: Mapping[str, Any], run_dir: Path
) -> bool:
    candidates: list[Mapping[str, Any]] = []

    accounts = index_payload.get("accounts")
    if isinstance(accounts, Sequence):
        for entry in accounts:
            if isinstance(entry, Mapping):
                candidates.append(entry)

    packs = index_payload.get("packs")
    if isinstance(packs, Sequence):
        for entry in packs:
            if isinstance(entry, Mapping):
                candidates.append(entry)

    if not candidates:
        return False

    seen_paths: set[str] = set()
    for entry in candidates:
        pack_path_value = entry.get("pack_path")
        if not isinstance(pack_path_value, str):
            pack_path_value = entry.get("path") if isinstance(entry, Mapping) else None
        if not isinstance(pack_path_value, str):
            continue
        if pack_path_value in seen_paths:
            continue
        seen_paths.add(pack_path_value)
        if "frontend/accounts/" in pack_path_value:
            return True
        pack_path = _resolve_pack_output_path(pack_path_value, run_dir)
        pack_payload = _load_json_payload(pack_path)
        if not isinstance(pack_payload, Mapping):
            return True
        if _pack_requires_pointer_backfill(pack_payload):
            return True

    return False


def _log_done(sid: str, packs: int, **extras: Any) -> None:
    details = [f"sid={sid}", f"packs={packs}"]
    for key, value in sorted(extras.items()):
        if value is None:
            continue
        details.append(f"{key}={value}")
    log.info("PACKGEN_FRONTEND_DONE %s", " ".join(details))


def _extract_text(value: Any) -> str | None:
    if isinstance(value, str):
        trimmed = value.strip()
        return trimmed or None
    if isinstance(value, Mapping):
        for key in (
            "text",
            "label",
            "display",
            "display_name",
            "value",
            "normalized",
            "name",
        ):
            candidate = value.get(key)
            if isinstance(candidate, str) and candidate.strip():
                return candidate.strip()
    return None


def _extract_summary_labels(summary: Mapping[str, Any]) -> Mapping[str, str | None]:
    labels = summary.get("labels")
    creditor = None
    account_type = None
    status = None

    if isinstance(labels, Mapping):
        creditor = _extract_text(labels.get("creditor") or labels.get("creditor_name"))
        account_type = _extract_text(labels.get("account_type"))
        status = _extract_text(labels.get("status") or labels.get("account_status"))

    if creditor is None:
        creditor = _extract_text(summary.get("creditor") or summary.get("creditor_name"))

    normalized = summary.get("normalized")
    if isinstance(normalized, Mapping):
        account_type = account_type or _extract_text(normalized.get("account_type"))
        status = status or _extract_text(normalized.get("status") or normalized.get("account_status"))

    return {
        "creditor_name": creditor,
        "account_type": account_type,
        "status": status,
    }


def _extract_last4(displays: Iterable[str]) -> Mapping[str, str | None]:
    digits: list[str] = []
    cleaned_display = None
    for display in displays:
        if not display:
            continue
        trimmed = str(display).strip()
        if not trimmed:
            continue
        cleaned_display = cleaned_display or trimmed
        numbers = re.sub(r"\D", "", trimmed)
        if len(numbers) >= 4:
            digits.append(numbers[-4:])

    last4_value = None
    if digits:
        # Prefer the most common last4
        seen: dict[str, int] = {}
        for candidate in digits:
            seen[candidate] = seen.get(candidate, 0) + 1
        last4_value = max(seen.items(), key=lambda item: (item[1], item[0]))[0]

    return {"display": cleaned_display, "last4": last4_value}


def _derive_masked_display(last4_payload: Mapping[str, Any] | None) -> str:
    """Return a masked display string derived from the last-4 payload."""

    display_value: str | None = None
    last4_digits: str | None = None

    if isinstance(last4_payload, Mapping):
        raw_display = last4_payload.get("display")
        if isinstance(raw_display, str):
            display_value = raw_display.strip() or None
        elif raw_display is not None:
            display_value = str(raw_display).strip() or None

        raw_last4 = last4_payload.get("last4")
        if isinstance(raw_last4, str):
            cleaned = re.sub(r"\D", "", raw_last4)
            last4_digits = cleaned[-4:] if cleaned else None
        elif raw_last4 is not None:
            cleaned = re.sub(r"\D", "", str(raw_last4))
            last4_digits = cleaned[-4:] if cleaned else None

    if display_value:
        return display_value

    if last4_digits:
        return f"****{last4_digits}"

    return "****"


def _coerce_display_text(value: Any) -> str:
    """Normalize optional display text into a stable string value."""

    if isinstance(value, str):
        cleaned = value.strip()
        return cleaned if cleaned else ""

    if value is None:
        return ""

    cleaned = str(value).strip()
    return cleaned if cleaned else ""


def _collect_field_per_bureau(
    bureaus: Mapping[str, Mapping[str, Any]], field: str
) -> tuple[dict[str, str], str | None]:
    values: dict[str, str] = {}
    unique: set[str] = set()
    for bureau, payload in bureaus.items():
        value = payload.get(field)
        if isinstance(value, str):
            trimmed = value.strip()
            if trimmed:
                values[bureau] = trimmed
                unique.add(trimmed)
        elif value is not None:
            coerced = str(value)
            if coerced.strip():
                values[bureau] = coerced.strip()
                unique.add(coerced.strip())

    consensus = unique.pop() if len(unique) == 1 else None
    return values, consensus


def _collect_field_text_values(
    bureaus: Mapping[str, Mapping[str, Any]], field: str
) -> dict[str, str]:
    values: dict[str, str] = {}
    for bureau in _BUREAU_ORDER:
        payload = bureaus.get(bureau)
        if not isinstance(payload, Mapping):
            continue
        raw_value = payload.get(field)
        if isinstance(raw_value, str):
            cleaned = raw_value.strip()
        elif raw_value is not None:
            cleaned = str(raw_value).strip()
        else:
            cleaned = ""
        if cleaned:
            values[bureau] = cleaned
    return values


def _normalize_status_text(value: str) -> str:
    normalized = value.strip()
    lowered = normalized.lower()
    simplified = re.sub(r"[^a-z]", "", lowered)
    if simplified == "collectionchargeoff":
        return "Collection"
    return normalized


def _collect_bureau_field_map(
    bureaus: Mapping[str, Mapping[str, Any]],
    key: str,
    *,
    fallback_key: str | None = None,
    transform: Callable[[str], str] | None = None,
) -> dict[str, str]:
    values: dict[str, str] = {}
    for bureau in _BUREAU_ORDER:
        payload = bureaus.get(bureau)
        if not isinstance(payload, Mapping):
            continue
        raw_value = payload.get(key)
        text_value = _coerce_display_text(raw_value)
        if not text_value and fallback_key:
            raw_value = payload.get(fallback_key)
            text_value = _coerce_display_text(raw_value)
        if not text_value:
            continue
        if transform is not None:
            transformed = transform(text_value)
            text_value = _coerce_display_text(transformed)
            if not text_value:
                continue
        values[bureau] = text_value
    return values


def _resolve_meta_holder_name(
    meta_payload: Mapping[str, Any] | None,
) -> tuple[str | None, str]:
    if isinstance(meta_payload, Mapping):
        for key in (
            "heading_guess",
            "furnisher_name",
            "furnisher_display_name",
            "furnisher_display",
            "furnisher",
            "creditor_name",
            "creditor_display_name",
            "creditor_display",
            "creditor",
            "name",
        ):
            candidate = _extract_text(meta_payload.get(key))
            if candidate:
                return candidate, f"meta.{key}"
    return None, "missing"


def build_display_from_bureaus(
    bureaus: Mapping[str, Any],
    meta: Mapping[str, Any] | None,
    tags: Any,
) -> dict[str, Any]:
    """Build a display payload sourced entirely from bureau artifacts.

    When :func:`_frontend_use_bureaus_only_enabled` is active this becomes the
    canonical path for assembling review pack display data.  The legacy
    ``fields_flat``/``summary`` pipeline continues to exist as a fallback, but
    the values returned here should be treated as the source of truth whenever
    ``bureaus.json`` is available.
    """
    bureaus_branches: dict[str, Mapping[str, Any]] = {
        bureau: payload
        for bureau, payload in bureaus.items()
        if bureau in _BUREAU_BADGES and isinstance(payload, Mapping)
    }

    account_number_values = _collect_bureau_field_map(
        bureaus_branches, "account_number_display"
    )
    if not account_number_values:
        account_number_values = _collect_bureau_field_map(
            bureaus_branches, "account_number"
        )
    account_number_per_bureau = _normalize_per_bureau(account_number_values)
    account_number_consensus = _resolve_account_number_consensus(
        account_number_per_bureau
    )

    account_type_values = _collect_bureau_field_map(bureaus_branches, "account_type")
    account_type_per_bureau = _normalize_per_bureau(account_type_values)
    account_type_consensus = _resolve_majority_consensus(account_type_per_bureau)

    status_values = _collect_bureau_field_map(
        bureaus_branches,
        "account_status",
        fallback_key="payment_status",
        transform=_normalize_status_text,
    )
    status_per_bureau = _normalize_per_bureau(status_values)
    status_consensus = _resolve_majority_consensus(status_per_bureau)

    opened_per_bureau = _normalize_per_bureau(
        _collect_bureau_field_map(bureaus_branches, "date_opened")
    )
    closed_date_per_bureau = _normalize_per_bureau(
        _collect_bureau_field_map(bureaus_branches, "closed_date")
    )
    last_payment_per_bureau = _normalize_per_bureau(
        _collect_bureau_field_map(bureaus_branches, "last_payment")
    )
    dofd_per_bureau = _normalize_per_bureau(
        _collect_bureau_field_map(
            bureaus_branches,
            "date_of_first_delinquency",
            fallback_key="date_of_last_activity",
        )
    )
    balance_per_bureau = _normalize_per_bureau(
        _collect_bureau_field_map(bureaus_branches, "balance_owed")
    )
    high_balance_per_bureau = _normalize_per_bureau(
        _collect_bureau_field_map(bureaus_branches, "high_balance")
    )
    limit_per_bureau = _normalize_per_bureau(
        _collect_bureau_field_map(bureaus_branches, "credit_limit")
    )
    remarks_per_bureau = _normalize_per_bureau(
        _collect_bureau_field_map(bureaus_branches, "creditor_remarks")
    )

    holder_values: dict[str, str] = {}
    for key in (
        "creditor_name",
        "creditor",
        "furnisher_name",
        "furnisher",
        "name",
    ):
        values = _collect_bureau_field_map(bureaus_branches, key)
        if not values:
            continue
        for bureau, value in values.items():
            holder_values.setdefault(bureau, value)

    holder_per_bureau = _normalize_per_bureau(holder_values)
    holder_consensus = _resolve_majority_consensus(holder_per_bureau)

    holder_candidate, _holder_resolver = _resolve_meta_holder_name(meta)
    display_holder_name = _coerce_display_text(holder_candidate)
    if not _has_meaningful_text(display_holder_name, treat_unknown=True):
        fallback_holder = None
        if _has_meaningful_text(holder_consensus, treat_unknown=True):
            fallback_holder = holder_consensus
        else:
            for bureau in _BUREAU_ORDER:
                candidate = holder_per_bureau.get(bureau)
                if _has_meaningful_text(candidate, treat_unknown=True):
                    fallback_holder = candidate
                    break
            if not fallback_holder:
                for candidate in holder_values.values():
                    if _has_meaningful_text(candidate, treat_unknown=True):
                        fallback_holder = candidate
                        break
        display_holder_name = fallback_holder or "Unknown"

    issues = _collect_issue_types(tags)
    primary_issue_value = issues[0] if issues else None
    display_primary_issue = _coerce_display_text(primary_issue_value or "unknown")
    if not display_primary_issue:
        display_primary_issue = "unknown"

    balance_payload = {"per_bureau": dict(balance_per_bureau)}

    display_payload: dict[str, Any] = {
        "display_version": _DISPLAY_SCHEMA_VERSION,
        "holder_name": display_holder_name,
        "primary_issue": display_primary_issue,
        "account_number": {
            "per_bureau": dict(account_number_per_bureau),
            "consensus": account_number_consensus,
        },
        "account_type": {
            "per_bureau": dict(account_type_per_bureau),
            "consensus": account_type_consensus,
        },
        "status": {
            "per_bureau": dict(status_per_bureau),
            "consensus": status_consensus,
        },
        "balance": dict(balance_payload),
        "balance_owed": dict(balance_payload),
        "high_balance": {"per_bureau": dict(high_balance_per_bureau)},
        "limit": {"per_bureau": dict(limit_per_bureau)},
        "remarks": {"per_bureau": dict(remarks_per_bureau)},
        "opened": dict(opened_per_bureau),
        "date_opened": dict(opened_per_bureau),
        "closed_date": dict(closed_date_per_bureau),
        "last_payment": dict(last_payment_per_bureau),
        "dofd": dict(dofd_per_bureau),
    }

    return display_payload


def _extract_bureaus_majority_value(
    bureaus_payload: Mapping[str, Any], field: str
) -> str | None:
    if not isinstance(bureaus_payload, Mapping):
        return None

    majority = bureaus_payload.get("majority_values")
    if isinstance(majority, Mapping):
        candidate = _extract_text(majority.get(field))
        if candidate:
            return candidate

    return None


def _resolve_account_number_consensus(per_bureau: Mapping[str, str]) -> str:
    duplicates = set()
    ordered_values: list[str] = []
    for bureau in _BUREAU_ORDER:
        value = per_bureau.get(bureau)
        if value and value != "--":
            ordered_values.append(value)
    if not ordered_values:
        return "--"
    counts = Counter(ordered_values)
    duplicates = {value for value, count in counts.items() if count >= 2}
    if duplicates:
        for bureau in _BUREAU_ORDER:
            value = per_bureau.get(bureau)
            if value in duplicates:
                return value
    for bureau in _BUREAU_ORDER:
        value = per_bureau.get(bureau)
        if value and value != "--":
            return value
    return "--"


def _resolve_majority_consensus(per_bureau: Mapping[str, str]) -> str:
    values: list[str] = []
    for bureau in _BUREAU_ORDER:
        value = per_bureau.get(bureau)
        if value and value != "--":
            values.append(value)
    if not values:
        return "--"
    counts = Counter(values)
    majority_values = {value for value, count in counts.items() if count >= 2}
    if majority_values:
        for bureau in _BUREAU_ORDER:
            value = per_bureau.get(bureau)
            if value in majority_values:
                return value
    for bureau in _BUREAU_ORDER:
        value = per_bureau.get(bureau)
        if value and value != "--":
            return value
    return "--"


def _load_raw_lines(path: Path) -> Sequence[str]:
    payload = _load_json_payload(path)
    if isinstance(payload, Sequence) and not isinstance(payload, (str, bytes, bytearray)):
        lines: list[str] = []
        for entry in payload:
            if isinstance(entry, Mapping):
                text = entry.get("text")
                if isinstance(text, str):
                    lines.append(text)
            elif isinstance(entry, str):
                lines.append(entry)
        return lines
    return []


def holder_name_from_raw_lines(raw_lines: list[str]) -> str | None:
    preferred: list[str] = []
    fallback: list[str] = []
    for candidate in raw_lines:
        if not isinstance(candidate, str):
            continue
        stripped = candidate.strip()
        if not stripped:
            continue
        if not _looks_like_holder_heading(stripped):
            continue

        if re.search(r"[ /]", stripped):
            preferred.append(stripped)
        else:
            fallback.append(stripped)

    if preferred:
        return preferred[0]
    if fallback:
        return fallback[0]
    return None


def _looks_like_holder_heading(text: str) -> bool:
    if not text:
        return False

    if text != text.upper():
        return False

    letters = sum(1 for char in text if char.isalpha())
    if letters < 2:
        return False

    digits = sum(1 for char in text if char.isdigit())
    if digits > max(2, len(text) // 5):
        return False

    if re.search(r"\b(ACCOUNT|BALANCE|PAYMENT|VERIFIED|OPENED|REPORTED)\b", text):
        return False

    if not re.fullmatch(r"[A-Z0-9 &'./-]+", text):
        return False

    return True


def _derive_holder_name_from_summary(
    summary: Mapping[str, Any] | None,
    fields_flat: Mapping[str, Any] | None,
) -> tuple[str | None, str]:
    candidates: list[tuple[Any, str]] = []

    if isinstance(summary, Mapping):
        candidates.extend(
            [
                (summary.get("holder_name"), "summary.holder_name"),
                (summary.get("consumer_name"), "summary.consumer_name"),
                (summary.get("consumer"), "summary.consumer"),
            ]
        )

        labels = summary.get("labels")
        if isinstance(labels, Mapping):
            candidates.append((labels.get("holder_name"), "summary.labels.holder_name"))
            candidates.append((labels.get("consumer_name"), "summary.labels.consumer_name"))

        normalized = summary.get("normalized")
        if isinstance(normalized, Mapping):
            candidates.append((normalized.get("holder_name"), "summary.normalized.holder_name"))

        meta = summary.get("meta")
        if isinstance(meta, Mapping):
            candidates.append((meta.get("holder_name"), "summary.meta.holder_name"))
            candidates.append((meta.get("heading_guess"), "summary.meta.heading_guess"))

    if isinstance(fields_flat, Mapping):
        for key in ("holder_name", "consumer_name"):
            value = fields_flat.get(key)
            if value:
                candidates.append((value, f"fields_flat.{key}"))

    for candidate, resolver in candidates:
        text = _extract_text(candidate)
        if text:
            return text, resolver

    return None, "missing"


def _derive_holder_name_from_meta(
    meta_payload: Mapping[str, Any] | None, account_id: str
) -> tuple[str | None, str]:
    if isinstance(meta_payload, Mapping):
        for key in (
            "heading_guess",
            "name",
            "furnisher_name",
            "creditor_name",
            "creditor",
        ):
            candidate = _extract_text(meta_payload.get(key))
            if candidate:
                return candidate, f"meta.{key}"

    fallback = _extract_text(account_id)
    return fallback, "account_id"


def _collect_issue_types(payload: Any) -> list[str]:
    issues: list[str] = []
    seen: set[str] = set()
    if isinstance(payload, Sequence) and not isinstance(payload, (str, bytes, bytearray)):
        for entry in payload:
            if not isinstance(entry, Mapping):
                continue
            if entry.get("kind") != "issue":
                continue
            issue_type = entry.get("type")
            if not isinstance(issue_type, str):
                continue
            trimmed = issue_type.strip()
            if not trimmed or trimmed in seen:
                continue
            issues.append(trimmed)
            seen.add(trimmed)
    return issues


def _extract_issue_tags(
    tags_path: Path, payload: Any | None = None
) -> tuple[str | None, list[str]]:
    if payload is None:
        payload = _load_json_payload(tags_path)

    issues = _collect_issue_types(payload)
    primary = issues[0] if issues else None
    return primary, issues


def _summarize_balance(balance_payload: Mapping[str, Any] | None) -> str | None:
    if not isinstance(balance_payload, Mapping):
        return None

    consensus = balance_payload.get("consensus")
    if isinstance(consensus, str) and consensus.strip():
        return consensus.strip()

    per_bureau = balance_payload.get("per_bureau")
    if isinstance(per_bureau, Mapping):
        for value in per_bureau.values():
            if isinstance(value, str) and value.strip():
                return value.strip()

    return None


def _prepare_bureau_payload_from_flat(
    *,
    account_number_values: Mapping[str, str],
    balance_values: Mapping[str, str],
    date_opened_values: Mapping[str, str],
    closed_date_values: Mapping[str, str],
    date_reported_values: Mapping[str, str],
    reported_bureaus: Iterable[str],
) -> dict[str, Any]:
    """Normalise bureau details for downstream pack payloads.

    Despite the historical name this helper now operates on *any* normalized
    per-bureau mappings.  Both the legacy ``fields_flat`` pipeline and the
    bureaus-only feature flag reuse this function to produce ``bureau_summary``
    payloads.
    """
    badges = [
        dict(_BUREAU_BADGES[bureau])
        for bureau in reported_bureaus
        if bureau in _BUREAU_BADGES
    ]

    if not badges:
        badges = [dict(_BUREAU_BADGES[bureau]) for bureau in _BUREAU_ORDER]

    displays = [value for value in account_number_values.values() if isinstance(value, str)]
    last4_info = _extract_last4(displays)

    balance_consensus = _resolve_majority_consensus(balance_values)

    return {
        "last4": last4_info,
        "balance_owed": {
            "per_bureau": dict(balance_values),
            **({"consensus": balance_consensus} if balance_consensus else {}),
        },
        "dates": {
            "date_opened": dict(date_opened_values),
            "closed_date": dict(closed_date_values),
            "date_reported": dict(date_reported_values),
        },
        "bureau_badges": badges,
    }


def _per_bureau_conflict(
    existing: Mapping[str, Any] | None,
    candidate: Mapping[str, Any] | None,
) -> bool:
    for bureau in _BUREAU_ORDER:
        existing_value = _coerce_display_text(
            existing.get(bureau) if isinstance(existing, Mapping) else None
        )
        candidate_value = _coerce_display_text(
            candidate.get(bureau) if isinstance(candidate, Mapping) else None
        )
        if not _has_meaningful_text(candidate_value, treat_unknown=True):
            continue
        if not _has_meaningful_text(existing_value, treat_unknown=True):
            return True
        if existing_value != candidate_value:
            return True
    return False


def _maybe_warn_bureaus_conflict(
    *,
    sid: str,
    account_id: str,
    account_dir: Path,
    per_bureau_snapshot: Mapping[str, Mapping[str, Any] | None],
) -> None:
    if not _frontend_warn_bureaus_conflict_enabled():
        return

    bureaus_path = account_dir / "bureaus.json"
    bureaus_payload = _load_json(bureaus_path)
    if not isinstance(bureaus_payload, Mapping):
        return

    display_from_bureaus = build_display_from_bureaus(bureaus_payload, None, None)

    def _section_per_bureau(section: Any) -> Mapping[str, Any]:
        if isinstance(section, Mapping):
            per = section.get("per_bureau")
            if isinstance(per, Mapping):
                return per
            return section
        return {}

    conflicts: list[str] = []
    for field_name, existing_values in per_bureau_snapshot.items():
        display_section = display_from_bureaus.get(field_name)
        candidate_values = _section_per_bureau(display_section)
        if not candidate_values:
            continue
        if _per_bureau_conflict(existing_values, candidate_values):
            conflicts.append(field_name)

    if conflicts:
        log.warning(
            "FRONTEND_PACK_BUREAUS_CONFLICT sid=%s account=%s fields=%s",
            sid,
            account_id,
            ",".join(sorted(conflicts)),
        )


def _stringify_flat_value(value: Any) -> str | None:
    if isinstance(value, (int, float, Decimal)):
        return str(value)

    if isinstance(value, Mapping):
        for key in (
            "display",
            "text",
            "label",
            "normalized",
            "value",
            "amount",
            "raw",
            "name",
        ):
            candidate = value.get(key)
            text = _stringify_flat_value(candidate)
            if text:
                return text
        return None

    if isinstance(value, str):
        trimmed = value.strip()
        return trimmed or None

    return None


def _flat_lookup(
    flat_payload: Mapping[str, Any] | None, field: str, bureau: str | None = None
) -> Any:
    if not isinstance(flat_payload, Mapping):
        return None

    if bureau is not None:
        per_bureau = flat_payload.get("per_bureau")
        if isinstance(per_bureau, Mapping):
            candidate = per_bureau.get(bureau)
            if isinstance(candidate, Mapping):
                inner = candidate.get(field)
                if inner is not None:
                    return inner
            elif candidate is not None:
                return candidate

        bureau_payload = flat_payload.get(bureau)
        if isinstance(bureau_payload, Mapping):
            direct = bureau_payload.get(field)
            if direct is not None:
                return direct
            nested = bureau_payload.get("fields")
            if isinstance(nested, Mapping):
                candidate = nested.get(field)
                if candidate is not None:
                    return candidate

        for key in (
            f"{bureau}_{field}",
            f"{field}_{bureau}",
            f"{bureau}.{field}",
            f"{field}.{bureau}",
        ):
            if key in flat_payload:
                return flat_payload[key]

    field_payload = flat_payload.get(field)
    if isinstance(field_payload, Mapping):
        if bureau is not None:
            candidate = field_payload.get(bureau)
            if candidate is not None:
                return candidate
        per_bureau = field_payload.get("per_bureau")
        if isinstance(per_bureau, Mapping) and bureau is not None:
            candidate = per_bureau.get(bureau)
            if candidate is not None:
                return candidate
        value = field_payload.get("value")
        if value is not None and bureau is None:
            return value

    return None


def _collect_flat_field_per_bureau(
    flat_payload: Mapping[str, Any] | None, field: str
) -> dict[str, str]:
    values: dict[str, str] = {}
    for bureau in _BUREAU_ORDER:
        raw_value = _flat_lookup(flat_payload, field, bureau)
        text = _stringify_flat_value(raw_value)
        if text:
            values[bureau] = text
    return values


def _collect_flat_consensus(flat_payload: Mapping[str, Any] | None, field: str) -> str | None:
    values: set[str] = set()
    if not isinstance(flat_payload, Mapping):
        return None

    value = _stringify_flat_value(flat_payload.get(field))
    if value:
        values.add(value)

    for bureau in _BUREAU_ORDER:
        bureau_value = _stringify_flat_value(_flat_lookup(flat_payload, field, bureau))
        if bureau_value:
            values.add(bureau_value)

    if len(values) == 1:
        return next(iter(values))
    return None


def _determine_reported_bureaus(
    summary: Mapping[str, Any] | None,
    flat_payload: Mapping[str, Any] | None,
) -> list[str]:
    reported: list[str] = []

    if isinstance(summary, Mapping):
        bureaus = summary.get("bureaus")
        if isinstance(bureaus, Sequence):
            for entry in bureaus:
                if isinstance(entry, str):
                    normalized = entry.strip().lower()
                    if normalized in _BUREAU_BADGES and normalized not in reported:
                        reported.append(normalized)

    if isinstance(flat_payload, Mapping):
        for bureau in _BUREAU_ORDER:
            if bureau in reported:
                continue
            candidate = _flat_lookup(flat_payload, "account_number_display", bureau)
            if _stringify_flat_value(candidate):
                reported.append(bureau)

    return reported


def _normalize_per_bureau(source: Mapping[str, Any] | None) -> dict[str, str]:
    normalized: dict[str, str] = {}
    for bureau in _BUREAU_ORDER:
        raw_value: Any | None = None
        if isinstance(source, Mapping):
            raw_value = source.get(bureau)
        if isinstance(raw_value, str):
            value = raw_value.strip() or None
        elif raw_value is not None:
            value = str(raw_value).strip() or None
        else:
            value = None
        normalized[bureau] = value if value else "--"
    return normalized


def _count_filled_fields_per_bureau(
    *per_bureau_sources: Mapping[str, Any] | None,
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for bureau in _BUREAU_ORDER:
        short_code = _BUREAU_SHORT_CODES.get(bureau, bureau)
        counts[short_code] = 0

    for source in per_bureau_sources:
        if not isinstance(source, Mapping):
            continue
        for bureau, raw_value in source.items():
            short_code = _BUREAU_SHORT_CODES.get(bureau)
            if short_code is None:
                continue
            text_value = _coerce_display_text(raw_value)
            if not _has_meaningful_text(text_value, treat_unknown=True):
                continue
            counts[short_code] = counts.get(short_code, 0) + 1

    ordered_counts = {
        _BUREAU_SHORT_CODES[bureau]: counts.get(_BUREAU_SHORT_CODES[bureau], 0)
        for bureau in _BUREAU_ORDER
        if bureau in _BUREAU_SHORT_CODES
    }

    return ordered_counts


def _normalize_consensus_text(value: Any) -> str:
    if isinstance(value, str):
        trimmed = value.strip()
    elif value is None:
        trimmed = ""
    else:
        trimmed = str(value).strip()
    return trimmed if trimmed else "--"


def build_display_payload(
    *,
    holder_name: str,
    primary_issue: str,
    account_number_per_bureau: Mapping[str, str],
    account_number_consensus: str | None,
    account_type_per_bureau: Mapping[str, str],
    account_type_consensus: str | None,
    status_per_bureau: Mapping[str, str],
    status_consensus: str | None,
    balance_per_bureau: Mapping[str, str],
    date_opened_per_bureau: Mapping[str, str],
    closed_date_per_bureau: Mapping[str, str],
    last_payment_per_bureau: Mapping[str, str] | None = None,
    dofd_per_bureau: Mapping[str, str] | None = None,
    high_balance_per_bureau: Mapping[str, str] | None = None,
    limit_per_bureau: Mapping[str, str] | None = None,
    remarks_per_bureau: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    account_number_consensus_text = _normalize_consensus_text(account_number_consensus)
    account_type_consensus_text = _normalize_consensus_text(account_type_consensus)
    status_consensus_text = _normalize_consensus_text(status_consensus)

    return {
        "display_version": _DISPLAY_SCHEMA_VERSION,
        "holder_name": holder_name,
        "primary_issue": primary_issue,
        "account_number": {
            "per_bureau": dict(account_number_per_bureau),
            "consensus": account_number_consensus_text,
        },
        "account_type": {
            "per_bureau": dict(account_type_per_bureau),
            "consensus": account_type_consensus_text,
        },
        "status": {
            "per_bureau": dict(status_per_bureau),
            "consensus": status_consensus_text,
        },
        "balance": {"per_bureau": dict(balance_per_bureau)},
        "balance_owed": {"per_bureau": dict(balance_per_bureau)},
        "high_balance": {
            "per_bureau": dict(_normalize_per_bureau(high_balance_per_bureau))
            if high_balance_per_bureau is not None
            else dict(_normalize_per_bureau({}))
        },
        "limit": {
            "per_bureau": dict(_normalize_per_bureau(limit_per_bureau))
            if limit_per_bureau is not None
            else dict(_normalize_per_bureau({}))
        },
        "remarks": {
            "per_bureau": dict(_normalize_per_bureau(remarks_per_bureau))
            if remarks_per_bureau is not None
            else dict(_normalize_per_bureau({}))
        },
        "opened": dict(date_opened_per_bureau),
        "date_opened": dict(date_opened_per_bureau),
        "closed_date": dict(closed_date_per_bureau),
        "last_payment": dict(
            _normalize_per_bureau(last_payment_per_bureau)
            if last_payment_per_bureau is not None
            else _normalize_per_bureau({})
        ),
        "dofd": dict(
            _normalize_per_bureau(dofd_per_bureau)
            if dofd_per_bureau is not None
            else _normalize_per_bureau({})
        ),
    }


def _build_compact_display(
    *,
    holder_name: str | None,
    primary_issue: str | None,
    display_payload: Mapping[str, Any],
) -> dict[str, Any]:
    def _copy_account_section(
        source: Mapping[str, Any] | None, *, include_consensus: bool
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {"per_bureau": {}}
        if isinstance(source, Mapping):
            per_bureau = source.get("per_bureau")
            if isinstance(per_bureau, Mapping):
                payload["per_bureau"] = dict(per_bureau)
            if include_consensus:
                consensus = source.get("consensus")
                if consensus is not None:
                    payload["consensus"] = consensus if isinstance(consensus, str) else str(consensus)
        return payload

    def _bureau_dates(source: Mapping[str, Any] | None) -> dict[str, Any]:
        return dict(source) if isinstance(source, Mapping) else {}

    return {
        "display_version": display_payload.get(
            "display_version", _DISPLAY_SCHEMA_VERSION
        ),
        "holder_name": holder_name,
        "primary_issue": primary_issue,
        "account_number": _copy_account_section(
            display_payload.get("account_number"), include_consensus=True
        ),
        "account_type": _copy_account_section(
            display_payload.get("account_type"), include_consensus=True
        ),
        "status": _copy_account_section(
            display_payload.get("status"), include_consensus=True
        ),
        "balance": _copy_account_section(
            display_payload.get("balance"), include_consensus=False
        ),
        "balance_owed": _copy_account_section(
            display_payload.get("balance_owed"), include_consensus=False
        ),
        "high_balance": _copy_account_section(
            display_payload.get("high_balance"), include_consensus=False
        ),
        "limit": _copy_account_section(
            display_payload.get("limit"), include_consensus=False
        ),
        "remarks": _copy_account_section(
            display_payload.get("remarks"), include_consensus=False
        ),
        "opened": _bureau_dates(display_payload.get("opened")),
        "date_opened": _bureau_dates(display_payload.get("date_opened")),
        "closed_date": _bureau_dates(display_payload.get("closed_date")),
        "last_payment": _bureau_dates(display_payload.get("last_payment")),
        "dofd": _bureau_dates(display_payload.get("dofd")),
    }


def build_pack_doc(
    *,
    sid: str,
    account_id: str,
    creditor_name: str | None,
    account_type: str | None,
    status: str | None,
    bureau_summary: Mapping[str, Any],
    holder_name: str | None,
    primary_issue: str | None,
    display_payload: Mapping[str, Any],
    pointers: Mapping[str, str],
    issues: Sequence[str] | None,
) -> dict[str, Any]:
    payload = {
        "sid": sid,
        "account_id": account_id,
        "creditor_name": creditor_name,
        "account_type": account_type,
        "status": status,
        "last4": bureau_summary["last4"],
        "balance_owed": bureau_summary["balance_owed"],
        "dates": bureau_summary["dates"],
        "bureau_badges": bureau_summary["bureau_badges"],
        "holder_name": holder_name,
        "primary_issue": primary_issue,
        "display": dict(display_payload),
        "pointers": dict(pointers),
        "questions": list(_QUESTION_SET),
        "claim_field_links": _claim_field_links_payload(),
    }
    if issues:
        payload["issues"] = list(issues)
    return payload


def build_lean_pack_doc(
    *,
    holder_name: str | None,
    primary_issue: str | None,
    display_payload: Mapping[str, Any],
    pointers: Mapping[str, str],
    questions: Sequence[Any],
) -> dict[str, Any]:
    display = _build_compact_display(
        holder_name=holder_name,
        primary_issue=primary_issue,
        display_payload=display_payload,
    )

    questions_payload = _coerce_question_list(questions)

    return {
        "holder_name": holder_name,
        "primary_issue": primary_issue,
        "display": display,
        "questions": questions_payload,
        "pointers": dict(pointers),
        "claim_field_links": _claim_field_links_payload(),
    }


def build_stage_pack_doc(
    *,
    account_id: str,
    holder_name: str | None,
    primary_issue: str | None,
    creditor_name: str | None,
    account_type: str | None,
    status: str | None,
    display_payload: Mapping[str, Any],
    bureau_summary: Mapping[str, Any],
    pointers: Mapping[str, str] | None = None,
    account_number_per_bureau: Mapping[str, Any] | None = None,
    account_number_consensus: str | None = None,
    account_type_per_bureau: Mapping[str, Any] | None = None,
    account_type_consensus: str | None = None,
    status_per_bureau: Mapping[str, Any] | None = None,
    status_consensus: str | None = None,
    balance_per_bureau: Mapping[str, Any] | None = None,
    date_opened_per_bureau: Mapping[str, Any] | None = None,
    closed_date_per_bureau: Mapping[str, Any] | None = None,
    high_balance_per_bureau: Mapping[str, Any] | None = None,
    limit_per_bureau: Mapping[str, Any] | None = None,
    remarks_per_bureau: Mapping[str, Any] | None = None,
    last_payment_per_bureau: Mapping[str, Any] | None = None,
    dofd_per_bureau: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    display = _build_compact_display(
        holder_name=holder_name,
        primary_issue=primary_issue,
        display_payload=display_payload,
    )

    _enrich_stage_display(
        display,
        holder_name=holder_name,
        account_number_per_bureau=account_number_per_bureau,
        account_number_consensus=account_number_consensus,
        account_type_per_bureau=account_type_per_bureau,
        account_type_consensus=account_type_consensus,
        status_per_bureau=status_per_bureau,
        status_consensus=status_consensus,
        balance_per_bureau=balance_per_bureau,
        date_opened_per_bureau=date_opened_per_bureau,
        closed_date_per_bureau=closed_date_per_bureau,
        high_balance_per_bureau=high_balance_per_bureau,
        limit_per_bureau=limit_per_bureau,
        remarks_per_bureau=remarks_per_bureau,
        last_payment_per_bureau=last_payment_per_bureau,
        dofd_per_bureau=dofd_per_bureau,
        bureau_summary=bureau_summary,
    )

    payload: dict[str, Any] = {
        "account_id": account_id,
        "holder_name": holder_name,
        "primary_issue": primary_issue,
        "creditor_name": creditor_name,
        "account_type": account_type,
        "status": status,
        "display": display,
        "claim_field_links": _claim_field_links_payload(),
    }

    if isinstance(pointers, Mapping):
        payload["pointers"] = dict(pointers)

    last4_payload = bureau_summary.get("last4") if isinstance(bureau_summary, Mapping) else None
    if isinstance(last4_payload, Mapping):
        payload["last4"] = dict(last4_payload)
    elif last4_payload is not None:
        payload["last4"] = last4_payload

    balance_payload = (
        bureau_summary.get("balance_owed") if isinstance(bureau_summary, Mapping) else None
    )
    if isinstance(balance_payload, Mapping):
        payload["balance_owed"] = dict(balance_payload)

    dates_payload = (
        bureau_summary.get("dates") if isinstance(bureau_summary, Mapping) else None
    )
    if isinstance(dates_payload, Mapping):
        payload["dates"] = dict(dates_payload)

    badges_payload = (
        bureau_summary.get("bureau_badges") if isinstance(bureau_summary, Mapping) else None
    )
    if isinstance(badges_payload, Sequence):
        payload["bureau_badges"] = [
            dict(badge)
            if isinstance(badge, Mapping)
            else badge
            for badge in badges_payload
        ]

    return payload


def _enrich_stage_display(
    display: Mapping[str, Any] | None,
    *,
    holder_name: str | None = None,
    account_number_per_bureau: Mapping[str, Any] | None = None,
    account_number_consensus: str | None = None,
    account_type_per_bureau: Mapping[str, Any] | None = None,
    account_type_consensus: str | None = None,
    status_per_bureau: Mapping[str, Any] | None = None,
    status_consensus: str | None = None,
    balance_per_bureau: Mapping[str, Any] | None = None,
    date_opened_per_bureau: Mapping[str, Any] | None = None,
    closed_date_per_bureau: Mapping[str, Any] | None = None,
    high_balance_per_bureau: Mapping[str, Any] | None = None,
    limit_per_bureau: Mapping[str, Any] | None = None,
    remarks_per_bureau: Mapping[str, Any] | None = None,
    last_payment_per_bureau: Mapping[str, Any] | None = None,
    dofd_per_bureau: Mapping[str, Any] | None = None,
    bureau_summary: Mapping[str, Any] | None = None,
) -> None:
    if not isinstance(display, dict):
        return

    def _ensure_section(key: str) -> dict[str, Any]:
        section = display.get(key)
        if isinstance(section, dict):
            return section
        if isinstance(section, Mapping):
            converted = dict(section)
            display[key] = converted
            return converted
        converted: dict[str, Any] = {}
        display[key] = converted
        return converted

    def _ensure_per_bureau(section: dict[str, Any]) -> dict[str, Any]:
        per_bureau = section.get("per_bureau")
        if isinstance(per_bureau, dict):
            return per_bureau
        if isinstance(per_bureau, Mapping):
            converted = dict(per_bureau)
            section["per_bureau"] = converted
            return converted
        converted = {}
        section["per_bureau"] = converted
        return converted

    def _apply_per_bureau(
        key: str,
        values: Mapping[str, Any] | None,
        *,
        consensus: str | None = None,
        fill_missing_from_consensus: bool = False,
    ) -> None:
        section = _ensure_section(key)
        per_bureau_section = _ensure_per_bureau(section)

        if isinstance(values, Mapping):
            for bureau, raw_value in values.items():
                if not isinstance(bureau, str):
                    continue
                text_value = _coerce_display_text(raw_value)
                if not _has_meaningful_text(text_value, treat_unknown=True):
                    continue
                per_bureau_section[bureau] = text_value

        consensus_text = _normalize_consensus_text(consensus)
        section["consensus"] = consensus_text
        if fill_missing_from_consensus:
            for bureau in _BUREAU_ORDER:
                existing = per_bureau_section.get(bureau)
                if _has_meaningful_text(existing, treat_unknown=True):
                    continue
                per_bureau_section[bureau] = consensus_text

    def _apply_date_mapping(key: str, values: Mapping[str, Any] | None) -> None:
        section = _ensure_section(key)
        if isinstance(values, Mapping):
            for bureau, raw_value in values.items():
                if not isinstance(bureau, str):
                    continue
                text_value = _coerce_display_text(raw_value)
                if not _has_meaningful_text(text_value, treat_unknown=True):
                    continue
                section[bureau] = text_value

    if _has_meaningful_text(holder_name, treat_unknown=True):
        display["holder_name"] = _coerce_display_text(holder_name)

    _apply_per_bureau(
        "account_number",
        account_number_per_bureau,
        consensus=account_number_consensus,
        fill_missing_from_consensus=False,
    )
    _apply_per_bureau(
        "account_type",
        account_type_per_bureau,
        consensus=account_type_consensus,
        fill_missing_from_consensus=False,
    )
    _apply_per_bureau(
        "status",
        status_per_bureau,
        consensus=status_consensus,
        fill_missing_from_consensus=False,
    )

    balance_source = balance_per_bureau
    dates_payload: Mapping[str, Any] | None = None
    if isinstance(bureau_summary, Mapping):
        balance_payload = bureau_summary.get("balance_owed")
        if isinstance(balance_payload, Mapping):
            per_bureau_payload = balance_payload.get("per_bureau")
            if balance_source is None and isinstance(per_bureau_payload, Mapping):
                balance_source = per_bureau_payload
        dates_payload_candidate = bureau_summary.get("dates")
        if isinstance(dates_payload_candidate, Mapping):
            dates_payload = dates_payload_candidate

    _apply_per_bureau(
        "balance_owed",
        balance_source,
        fill_missing_from_consensus=False,
    )
    _apply_per_bureau(
        "balance",
        balance_source,
        fill_missing_from_consensus=False,
    )
    _apply_per_bureau(
        "high_balance",
        high_balance_per_bureau,
        fill_missing_from_consensus=False,
    )
    _apply_per_bureau(
        "limit",
        limit_per_bureau,
        fill_missing_from_consensus=False,
    )
    _apply_per_bureau(
        "remarks",
        remarks_per_bureau,
        fill_missing_from_consensus=False,
    )

    date_opened_source: Mapping[str, Any] | None = date_opened_per_bureau
    closed_date_source: Mapping[str, Any] | None = closed_date_per_bureau
    if dates_payload is not None:
        if date_opened_source is None:
            candidate = dates_payload.get("date_opened")
            if isinstance(candidate, Mapping):
                date_opened_source = candidate
        if closed_date_source is None:
            candidate = dates_payload.get("closed_date")
            if isinstance(candidate, Mapping):
                closed_date_source = candidate

    _apply_date_mapping("opened", date_opened_source)
    _apply_date_mapping("date_opened", date_opened_source)
    _apply_date_mapping("closed_date", closed_date_source)
    _apply_date_mapping("last_payment", last_payment_per_bureau)
    _apply_date_mapping("dofd", dofd_per_bureau)


def _has_meaningful_text(value: Any, *, treat_unknown: bool = False) -> bool:
    if isinstance(value, str):
        normalized = value.strip()
        if not normalized:
            return False
        if normalized == "--":
            return False
        if treat_unknown and normalized.lower() == "unknown":
            return False
        return True
    return value is not None


def _mapping_has_meaningful_values(
    mapping: Mapping[str, Any] | None, *, treat_unknown: bool = False
) -> bool:
    if not isinstance(mapping, Mapping):
        return False
    for value in mapping.values():
        if _has_meaningful_text(value, treat_unknown=treat_unknown):
            return True
    return False


def _has_meaningful_display(display: Mapping[str, Any] | None) -> bool:
    if not isinstance(display, Mapping):
        return False

    if _has_meaningful_text(display.get("holder_name"), treat_unknown=True):
        return True
    if _has_meaningful_text(display.get("primary_issue"), treat_unknown=True):
        return True

    account_number = display.get("account_number")
    if isinstance(account_number, Mapping):
        if _has_meaningful_text(account_number.get("consensus")):
            return True
        if _mapping_has_meaningful_values(account_number.get("per_bureau")):
            return True

    account_type = display.get("account_type")
    if isinstance(account_type, Mapping):
        if _has_meaningful_text(account_type.get("consensus"), treat_unknown=True):
            return True
        if _mapping_has_meaningful_values(account_type.get("per_bureau"), treat_unknown=True):
            return True

    status = display.get("status")
    if isinstance(status, Mapping):
        if _has_meaningful_text(status.get("consensus"), treat_unknown=True):
            return True
        if _mapping_has_meaningful_values(status.get("per_bureau"), treat_unknown=True):
            return True

    balance = display.get("balance_owed")
    if isinstance(balance, Mapping) and _mapping_has_meaningful_values(
        balance.get("per_bureau")
    ):
        return True
    balance_new = display.get("balance")
    if isinstance(balance_new, Mapping) and _mapping_has_meaningful_values(
        balance_new.get("per_bureau")
    ):
        return True
    high_balance = display.get("high_balance")
    if isinstance(high_balance, Mapping) and _mapping_has_meaningful_values(
        high_balance.get("per_bureau")
    ):
        return True
    credit_limit = display.get("limit")
    if isinstance(credit_limit, Mapping) and _mapping_has_meaningful_values(
        credit_limit.get("per_bureau")
    ):
        return True
    remarks = display.get("remarks")
    if isinstance(remarks, Mapping) and _mapping_has_meaningful_values(
        remarks.get("per_bureau"), treat_unknown=True
    ):
        return True

    if _mapping_has_meaningful_values(display.get("opened")):
        return True
    if _mapping_has_meaningful_values(display.get("date_opened")):
        return True
    if _mapping_has_meaningful_values(display.get("closed_date")):
        return True
    if _mapping_has_meaningful_values(display.get("last_payment")):
        return True
    if _mapping_has_meaningful_values(display.get("dofd")):
        return True

    return False


def _stage_payload_has_meaningful_data(payload: Mapping[str, Any] | None) -> bool:
    if not isinstance(payload, Mapping):
        return False

    account_id = payload.get("account_id")

    if _has_meaningful_text(payload.get("holder_name"), treat_unknown=True):
        holder_name = str(payload.get("holder_name"))
        if isinstance(account_id, str) and holder_name.strip() == account_id.strip():
            pass
        else:
            return True
    if _has_meaningful_text(payload.get("primary_issue"), treat_unknown=True):
        return True

    for key in ("creditor_name", "account_type", "status"):
        value = payload.get(key)
        if _has_meaningful_text(value, treat_unknown=True):
            if (
                key == "creditor_name"
                and isinstance(account_id, str)
                and isinstance(value, str)
                and value.strip() == account_id.strip()
            ):
                continue
            return True

    display_payload = payload.get("display")
    if isinstance(display_payload, Mapping):
        holder_display = display_payload.get("holder_name")
        if (
            isinstance(account_id, str)
            and isinstance(holder_display, str)
            and holder_display.strip() == account_id.strip()
        ):
            trimmed_display = dict(display_payload)
            trimmed_display["holder_name"] = ""
            if _has_meaningful_display(trimmed_display):
                return True
        elif _has_meaningful_display(display_payload):
            return True

    return False


def _preserve_stage_display_values(
    existing: Mapping[str, Any] | None,
    candidate: Mapping[str, Any] | None,
) -> tuple[dict[str, Any], bool]:
    if not isinstance(existing, Mapping):
        if isinstance(candidate, dict):
            return candidate, False
        if isinstance(candidate, Mapping):
            return dict(candidate), False
        return {}, False

    if isinstance(candidate, dict):
        merged: dict[str, Any] = candidate
    elif isinstance(candidate, Mapping):
        merged = dict(candidate)
    else:
        merged = {}

    changed = False

    def _preserve_text(key: str, *, treat_unknown: bool = True) -> None:
        nonlocal changed
        existing_value = existing.get(key)
        if not _has_meaningful_text(existing_value, treat_unknown=treat_unknown):
            return
        candidate_value = merged.get(key)
        if _has_meaningful_text(candidate_value, treat_unknown=treat_unknown):
            return
        merged[key] = existing_value
        changed = True

    def _ensure_dict(value: Any) -> dict[str, Any]:
        if isinstance(value, dict):
            return value
        if isinstance(value, Mapping):
            return dict(value)
        return {}

    def _preserve_section(
        key: str,
        *,
        treat_unknown: bool = True,
        include_consensus: bool = True,
    ) -> None:
        nonlocal changed
        existing_section = existing.get(key)
        if not isinstance(existing_section, Mapping):
            return

        candidate_section = merged.get(key)
        if not isinstance(candidate_section, dict):
            candidate_section = _ensure_dict(candidate_section)
            merged[key] = candidate_section

        if include_consensus:
            existing_consensus = existing_section.get("consensus")
            if _has_meaningful_text(existing_consensus, treat_unknown=treat_unknown):
                candidate_consensus = candidate_section.get("consensus")
                if not _has_meaningful_text(
                    candidate_consensus, treat_unknown=treat_unknown
                ):
                    candidate_section["consensus"] = existing_consensus
                    changed = True

        existing_per = existing_section.get("per_bureau")
        if isinstance(existing_per, Mapping):
            candidate_per = candidate_section.get("per_bureau")
            if not isinstance(candidate_per, dict):
                candidate_per = _ensure_dict(candidate_per)
                candidate_section["per_bureau"] = candidate_per
            for bureau, raw_value in existing_per.items():
                if not isinstance(bureau, str):
                    continue
                if not _has_meaningful_text(raw_value, treat_unknown=treat_unknown):
                    continue
                candidate_value = candidate_per.get(bureau)
                if _has_meaningful_text(candidate_value, treat_unknown=treat_unknown):
                    continue
                candidate_per[bureau] = raw_value
                changed = True

    def _preserve_date_mapping(key: str) -> None:
        nonlocal changed
        existing_section = existing.get(key)
        if not isinstance(existing_section, Mapping):
            return

        candidate_section = merged.get(key)
        if not isinstance(candidate_section, dict):
            candidate_section = _ensure_dict(candidate_section)
            merged[key] = candidate_section

        for bureau, raw_value in existing_section.items():
            if not isinstance(bureau, str):
                continue
            if not _has_meaningful_text(raw_value, treat_unknown=True):
                continue
            candidate_value = candidate_section.get(bureau)
            if _has_meaningful_text(candidate_value, treat_unknown=True):
                continue
            candidate_section[bureau] = raw_value
            changed = True

    if not merged.get("display_version") and existing.get("display_version"):
        merged["display_version"] = existing["display_version"]
        changed = True

    _preserve_text("holder_name")
    _preserve_text("primary_issue")
    _preserve_section("account_number", treat_unknown=False)
    _preserve_section("account_type", treat_unknown=True)
    _preserve_section("status", treat_unknown=True)
    _preserve_section("balance", treat_unknown=False, include_consensus=False)
    _preserve_section("balance_owed", treat_unknown=False, include_consensus=False)
    _preserve_section("high_balance", treat_unknown=False, include_consensus=False)
    _preserve_section("limit", treat_unknown=False, include_consensus=False)
    _preserve_section("remarks", treat_unknown=True, include_consensus=False)
    _preserve_date_mapping("opened")
    _preserve_date_mapping("date_opened")
    _preserve_date_mapping("closed_date")
    _preserve_date_mapping("last_payment")
    _preserve_date_mapping("dofd")

    return merged, changed


def _preserve_stage_pack_payload(
    existing: Mapping[str, Any] | None,
    candidate: Mapping[str, Any] | None,
) -> tuple[dict[str, Any], set[str]]:
    if isinstance(candidate, dict):
        merged: dict[str, Any] = candidate
    elif isinstance(candidate, Mapping):
        merged = dict(candidate)
    else:
        merged = {}

    if not isinstance(existing, Mapping):
        return merged, set()

    preserved_fields: set[str] = set()

    def _preserve_text(key: str, *, treat_unknown: bool = True) -> None:
        existing_value = existing.get(key)
        if not _has_meaningful_text(existing_value, treat_unknown=treat_unknown):
            return
        candidate_value = merged.get(key)
        if _has_meaningful_text(candidate_value, treat_unknown=treat_unknown):
            return
        merged[key] = existing_value
        preserved_fields.add(key)

    for field_name in ("holder_name", "primary_issue", "creditor_name", "account_type", "status"):
        _preserve_text(field_name, treat_unknown=True)

    display_candidate = merged.get("display")
    display_merged, display_changed = _preserve_stage_display_values(
        existing.get("display"), display_candidate
    )
    if display_changed or (display_candidate is None and isinstance(display_merged, dict)):
        merged["display"] = display_merged
    if display_changed:
        preserved_fields.add("display")

    return merged, preserved_fields


def _safe_account_dirname(account_id: str, fallback: str) -> str:
    account_id = account_id.strip()
    if not account_id:
        return fallback
    sanitized = re.sub(r"[^A-Za-z0-9_.-]", "_", account_id)
    return sanitized or fallback


def _build_stage_manifest(
    *,
    sid: str,
    stage_name: str,
    run_dir: Path,
    stage_packs_dir: Path,
    stage_responses_dir: Path,
    stage_index_path: Path,
    question_set: Sequence[Mapping[str, Any]] | None = None,
) -> Mapping[str, Any]:
    stage_dir = stage_index_path.parent
    pack_entries: list[dict[str, Any]] = []
    pack_index_entries: list[dict[str, Any]] = []

    if stage_packs_dir.is_dir():
        pack_paths = sorted(
            (
                path
                for path in stage_packs_dir.iterdir()
                if path.is_file() and path.suffix == ".json"
            ),
            key=_account_sort_key,
        )

        for pack_path in pack_paths:
            payload = _load_json_payload(pack_path)
            holder_name: str | None = None
            primary_issue: str | None = None
            has_questions = False

            if isinstance(payload, Mapping):
                holder_name = _optional_str(payload.get("holder_name"))
                primary_issue = _optional_str(payload.get("primary_issue"))
                questions = payload.get("questions")
                if isinstance(questions, Sequence) and not isinstance(
                    questions, (str, bytes, bytearray)
                ):
                    has_questions = len(questions) > 0

            if not has_questions:
                has_questions = bool(_QUESTION_SET)

            account_id = None
            if isinstance(payload, Mapping):
                account_id = _optional_str(payload.get("account_id"))

            if not account_id:
                account_id = pack_path.stem

            display_payload = None
            if isinstance(payload, Mapping):
                raw_display = payload.get("display")
                if isinstance(raw_display, Mapping):
                    display_payload = raw_display

            stage_relative_path = _relative_to_stage_dir(pack_path, stage_dir)
            run_relative_path = _relative_to_run_dir(pack_path, run_dir)

            pack_entry: dict[str, Any] = {
                "account_id": account_id,
                "holder_name": holder_name,
                "primary_issue": primary_issue,
                "path": run_relative_path,
                "bytes": os.path.getsize(pack_path),
                "has_questions": has_questions,
            }

            if display_payload is not None:
                pack_entry["display"] = display_payload

            pack_entry["pack_path"] = run_relative_path
            pack_entry["pack_path_rel"] = stage_relative_path
            pack_entry["file"] = run_relative_path

            sha1_digest = _safe_sha1(pack_path)
            if sha1_digest:
                pack_entry["sha1"] = sha1_digest

            pack_entries.append(pack_entry)
            pack_index_entries.append({"account": account_id, "file": stage_relative_path})

    responses_count = _count_frontend_responses(stage_responses_dir)
    responses_dir_value = _relative_to_run_dir(stage_responses_dir, run_dir)
    responses_dir_rel = _relative_to_stage_dir(stage_responses_dir, stage_dir)
    packs_dir_value = _relative_to_run_dir(stage_packs_dir, run_dir)
    packs_dir_rel = _relative_to_stage_dir(stage_packs_dir, stage_dir)
    index_path_value = _relative_to_run_dir(stage_index_path, run_dir)
    index_rel_value = _relative_to_stage_dir(stage_index_path, stage_dir)

    questions_payload = list(question_set) if question_set is not None else list(_QUESTION_SET)

    manifest_core: dict[str, Any] = {
        "sid": sid,
        "stage": stage_name,
        "schema_version": "1.0",
        "counts": {
            "packs": len(pack_entries),
            "responses": responses_count,
        },
        "packs": pack_entries,
        "responses_dir": responses_dir_value,
        "responses_dir_rel": responses_dir_rel,
        "packs_dir": packs_dir_value,
        "packs_dir_rel": packs_dir_rel,
        "index_path": index_path_value,
        "index_rel": index_rel_value,
        "packs_count": len(pack_entries),
        "questions": questions_payload,
        "packs_index": pack_index_entries,
    }

    generated_at = _now_iso()
    built_at = generated_at
    existing_manifest = _load_json_payload(stage_index_path)
    if isinstance(existing_manifest, Mapping):
        previous_core = dict(existing_manifest)
        previous_generated = previous_core.pop("generated_at", None)
        previous_built = previous_core.pop("built_at", None)
        if previous_core == manifest_core:
            if isinstance(previous_generated, str):
                generated_at = previous_generated
            if isinstance(previous_built, str):
                built_at = previous_built

    if not built_at:
        built_at = generated_at

    manifest_payload = {**manifest_core, "generated_at": generated_at, "built_at": built_at}
    _write_json_if_changed(stage_index_path, manifest_payload)
    log.info(
        "wrote review index sid=%s (count=%d) path=%s",
        sid,
        len(pack_entries),
        stage_index_path,
    )

    return manifest_payload


def _migrate_legacy_frontend_root_packs(
    *,
    sid: str,
    stage_name: str,
    run_dir: Path,
    stage_dir: Path,
    stage_packs_dir: Path,
    stage_responses_dir: Path,
    stage_index_path: Path,
    redirect_stub_path: Path,
) -> Mapping[str, Any] | None:
    canonical = get_frontend_review_paths(str(run_dir))
    legacy_glob = os.path.join(canonical["frontend_base"], "idx-*.json")

    moved = 0
    for legacy_path in glob.glob(legacy_glob):
        source = Path(legacy_path)
        if not source.is_file():
            continue

        destination = stage_packs_dir / source.name
        if destination.exists():
            log.warning(
                "FRONTEND_LEGACY_MIGRATE_EXISTS sid=%s source=%s target=%s",
                sid,
                source,
                destination,
            )
            continue

        destination.parent.mkdir(parents=True, exist_ok=True)

        try:
            shutil.move(str(source), str(destination))
        except (OSError, shutil.Error):
            log.warning(
                "FRONTEND_LEGACY_MIGRATE_FAILED sid=%s source=%s target=%s",
                sid,
                source,
                destination,
                exc_info=True,
            )
            continue

        moved += 1

    if not moved:
        return None

    stage_dir.mkdir(parents=True, exist_ok=True)
    stage_packs_dir.mkdir(parents=True, exist_ok=True)
    stage_responses_dir.mkdir(parents=True, exist_ok=True)
    stage_index_path.parent.mkdir(parents=True, exist_ok=True)

    manifest_payload = _build_stage_manifest(
        sid=sid,
        stage_name=stage_name,
        run_dir=run_dir,
        stage_packs_dir=stage_packs_dir,
        stage_responses_dir=stage_responses_dir,
        stage_index_path=stage_index_path,
        question_set=_QUESTION_SET,
    )
    _ensure_frontend_index_redirect_stub(redirect_stub_path, force=True)

    log.info("FRONTEND_LEGACY_PACK_MIGRATION sid=%s moved=%d", sid, moved)
    return manifest_payload


def generate_frontend_packs_for_run(
    sid: str,
    *,
    runs_root: Path | str | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Build customer-facing account packs for ``sid``."""

    base_root = _resolve_runs_root(runs_root)
    run_dir = base_root / sid
    accounts_dir = run_dir / "cases" / "accounts"

    def _env_override(*names: str, default: str | None = None) -> str | None:
        for name in names:
            value = os.getenv(name)
            if value:
                return value
        return default

    stage_name = _env_override("FRONTEND_STAGE_NAME", "FRONTEND_STAGE", default="review")

    lock_state, lock_path = _acquire_frontend_build_lock(run_dir, sid)
    if lock_state == "locked":
        canonical_paths = get_frontend_review_paths(str(run_dir))
        packs_dir_candidate = canonical_paths.get("packs_dir")
        packs_dir_path = (
            Path(packs_dir_candidate)
            if isinstance(packs_dir_candidate, str) and packs_dir_candidate
            else run_dir / "frontend" / "review" / "packs"
        )
        packs_dir_str = str(packs_dir_path.absolute())
        log.info("FRONTEND_BUILD_SKIP sid=%s reason=%s", sid, "locked")
        return {
            "status": "locked",
            "packs_count": 0,
            "empty_ok": True,
            "built": False,
            "packs_dir": packs_dir_str,
            "last_built_at": None,
            "skip_reason": "locked",
        }
    lock_acquired = lock_state == "acquired"

    try:
        config = load_frontend_stage_config(run_dir)

        stage_dir = config.stage_dir
        stage_packs_dir = config.packs_dir
        stage_responses_dir = config.responses_dir
        stage_index_path = config.index_path
        debug_packs_dir = stage_dir / "debug"
    
        canonical_paths = ensure_frontend_review_dirs(str(run_dir))

        _log_stage_paths(sid, config, canonical_paths)

        idempotent_lock_path = _resolve_idempotent_lock_path(run_dir)
        idempotent_lock_mtime = _lock_mtime(idempotent_lock_path)

        legacy_accounts_dir = run_dir / "frontend" / "accounts"
        if legacy_accounts_dir.is_dir():
            log.warning(
                "FRONTEND_LEGACY_ACCOUNTS_DIR sid=%s path=%s",
                sid,
                legacy_accounts_dir,
            )
    
        legacy_index_env = _env_override("FRONTEND_INDEX_PATH", "FRONTEND_INDEX")
        if legacy_index_env:
            candidate = Path(legacy_index_env)
            if not candidate.is_absolute():
                redirect_stub_path = run_dir / candidate
            else:
                redirect_stub_path = candidate
        else:
            redirect_stub_path = Path(
                canonical_paths.get("legacy_index", canonical_paths["index"])
            )
    
        _migrate_legacy_frontend_root_packs(
            sid=sid,
            stage_name=stage_name,
            run_dir=run_dir,
            stage_dir=stage_dir,
            stage_packs_dir=stage_packs_dir,
            stage_responses_dir=stage_responses_dir,
            stage_index_path=stage_index_path,
            redirect_stub_path=redirect_stub_path,
        )
        packs_dir_str = str(stage_packs_dir.absolute())
    
        frontend_autorun_enabled = _env_flag_enabled("FRONTEND_STAGE_AUTORUN", True)
        review_autorun_enabled = _env_flag_enabled("REVIEW_STAGE_AUTORUN", True)
        if not (frontend_autorun_enabled and review_autorun_enabled):
            if not frontend_autorun_enabled and not review_autorun_enabled:
                reason = "autorun_disabled"
            elif not frontend_autorun_enabled:
                reason = "frontend_stage_autorun_disabled"
            else:
                reason = "review_stage_autorun_disabled"
            log.info(
                "FRONTEND_AUTORUN_DISABLED sid=%s reason=%s",
                sid,
                reason,
            )
            return {
                "status": reason,
                "packs_count": 0,
                "empty_ok": True,
                "built": False,
                "packs_dir": packs_dir_str,
                "last_built_at": None,
                "autorun_disabled": True,
            }
    
        runflow_stage_start("frontend", sid=sid)
        current_account_id: str | None = None
        try:
            account_dirs: list[Path] = (
                sorted(
                    [path for path in accounts_dir.iterdir() if path.is_dir()],
                    key=_account_sort_key,
                )
                if accounts_dir.is_dir()
                else []
            )
            total_accounts = len(account_dirs)
    
            runflow_step(
                sid,
                "frontend",
                "frontend_review_start",
                metrics={"accounts": total_accounts},
            )
    
            if not _frontend_packs_enabled():
                fallback_manifest: Mapping[str, Any] | None = None
                if _frontend_review_create_empty_index_enabled():
                    stage_dir.mkdir(parents=True, exist_ok=True)
                    os.makedirs(stage_packs_dir, exist_ok=True)
                    os.makedirs(stage_responses_dir, exist_ok=True)
                    stage_index_path.parent.mkdir(parents=True, exist_ok=True)
                    fallback_manifest = _build_stage_manifest(
                        sid=sid,
                        stage_name=stage_name,
                        run_dir=run_dir,
                        stage_packs_dir=stage_packs_dir,
                        stage_responses_dir=stage_responses_dir,
                        stage_index_path=stage_index_path,
                        question_set=_QUESTION_SET,
                    )
                    _ensure_frontend_index_redirect_stub(redirect_stub_path)
                    log.info("FRONTEND_EMPTY_INDEX_FALLBACK sid=%s", sid)

                responses_count = _emit_responses_scan(sid, stage_responses_dir)
                summary: dict[str, Any] = {
                    "packs_count": 0,
                    "responses_received": responses_count,
                    "empty_ok": True,
                    "reason": "disabled",
                }
                if fallback_manifest is not None:
                    summary["fallback_index"] = True
                runflow_step(
                    sid,
                    "frontend",
                    "frontend_review_no_candidates",
                    out={"reason": "disabled"},
                )
                record_frontend_responses_progress(
                    sid,
                    accounts_published=0,
                    answers_received=responses_count,
                    answers_required=0,
                )
                runflow_step(
                    sid,
                    "frontend",
                    "frontend_review_finish",
                    status="skipped",
                    metrics={"packs": 0},
                    out={"reason": "disabled"},
                )
                runflow_stage_end(
                    "frontend",
                    sid=sid,
                    status="skipped",
                    summary=summary,
                    empty_ok=True,
                )
                if isinstance(fallback_manifest, Mapping):
                    generated_at = fallback_manifest.get("generated_at")
                    fallback_last_built = generated_at if isinstance(generated_at, str) else None
                else:
                    fallback_last_built = None
                _log_done(
                    sid,
                    0,
                    status="skipped",
                    reason="disabled",
                    fallback_index=bool(fallback_manifest),
                )
                result = {
                    "status": "skipped",
                    "packs_count": 0,
                    "empty_ok": True,
                    "built": False,
                    "packs_dir": packs_dir_str,
                    "last_built_at": fallback_last_built,
                }
                if fallback_manifest is not None:
                    result["fallback_index"] = True
                _log_build_summary(
                    sid,
                    packs_count=0,
                    last_built_at=fallback_last_built,
                )
                return result
    
            stage_dir.mkdir(parents=True, exist_ok=True)
            os.makedirs(stage_packs_dir, exist_ok=True)
            os.makedirs(stage_responses_dir, exist_ok=True)
            stage_index_path.parent.mkdir(parents=True, exist_ok=True)
    
            lean_enabled = _frontend_packs_lean_enabled()
            debug_mirror_enabled = _frontend_packs_debug_mirror_enabled()
            stage_payload_mode = _resolve_stage_payload_mode()
            stage_payload_full = stage_payload_mode in {
                _STAGE_PAYLOAD_MODE_FULL,
                _STAGE_PAYLOAD_MODE_LEGACY,
            }
    
            if not account_dirs:
                manifest_payload = _build_stage_manifest(
                    sid=sid,
                    stage_name=stage_name,
                    run_dir=run_dir,
                    stage_packs_dir=stage_packs_dir,
                    stage_responses_dir=stage_responses_dir,
                    stage_index_path=stage_index_path,
                    question_set=_QUESTION_SET,
                )
                _ensure_frontend_index_redirect_stub(redirect_stub_path)
                runflow_step(
                    sid,
                    "frontend",
                    "frontend_review_no_candidates",
                    metrics={"accounts": total_accounts},
                )
                runflow_step(
                    sid,
                    "frontend",
                    "frontend_review_finish",
                    metrics={"packs": 0},
                )
                responses_count = _emit_responses_scan(sid, stage_responses_dir)
                summary = {
                    "packs_count": 0,
                    "responses_received": responses_count,
                    "empty_ok": True,
                }
                record_frontend_responses_progress(
                    sid,
                    accounts_published=0,
                    answers_received=responses_count,
                    answers_required=0,
                )
                runflow_stage_end(
                    "frontend",
                    sid=sid,
                    summary=summary,
                    empty_ok=True,
                )
                _log_done(sid, 0, status="success")
                result = {
                    "status": "success",
                    "packs_count": 0,
                    "empty_ok": True,
                    "built": True,
                    "packs_dir": packs_dir_str,
                    "last_built_at": manifest_payload.get("generated_at"),
                }
                _log_build_summary(
                    sid,
                    packs_count=0,
                    last_built_at=manifest_payload.get("generated_at"),
                )
                return result
    
            if not force and stage_index_path.exists():
                existing = _load_json(stage_index_path)
                if existing:
                    pointer_backfill_required = _index_requires_pointer_backfill(
                        existing, run_dir
                    )
                    if pointer_backfill_required:
                        log.info("FRONTEND_PACK_POINTER_BACKFILL sid=%s", sid)
                    else:
                        packs_count = int(existing.get("packs_count", 0) or 0)
                        if not packs_count:
                            accounts = existing.get("accounts")
                            if isinstance(accounts, list):
                                packs_count = len(accounts)
                        log.debug(
                            "FRONTEND_PACKS_EXISTS sid=%s path=%s",
                            sid,
                            stage_index_path,
                        )
                        generated_at = existing.get("generated_at")
                        last_built = (
                            str(generated_at) if isinstance(generated_at, str) else None
                        )
                        _ensure_frontend_index_redirect_stub(redirect_stub_path)
                        if not debug_mirror_enabled and debug_packs_dir.is_dir():
                            for mirror_path in debug_packs_dir.glob("*.full.json"):
                                try:
                                    mirror_path.unlink()
                                except FileNotFoundError:
                                    continue
                                except OSError:  # pragma: no cover - defensive logging
                                    log.warning(
                                        "FRONTEND_PACK_DEBUG_MIRROR_UNLINK_FAILED path=%s",
                                        mirror_path,
                                        exc_info=True,
                                    )
                        if packs_count == 0:
                            runflow_step(
                                sid,
                                "frontend",
                                "frontend_review_no_candidates",
                                out={"reason": "cache"},
                            )
                        runflow_step(
                            sid,
                            "frontend",
                            "frontend_review_finish",
                            status="success",
                            metrics={"packs": packs_count},
                            out={"cache_hit": True},
                        )
                        responses_count = _emit_responses_scan(sid, stage_responses_dir)
                        summary = {
                            "packs_count": packs_count,
                            "responses_received": responses_count,
                            "empty_ok": packs_count == 0,
                            "cache_hit": True,
                        }
                        record_frontend_responses_progress(
                            sid,
                            accounts_published=packs_count,
                            answers_received=responses_count,
                            answers_required=packs_count,
                        )
                        runflow_stage_end(
                            "frontend",
                            sid=sid,
                            summary=summary,
                            empty_ok=packs_count == 0,
                        )
                        _log_done(sid, packs_count, status="success", cache_hit=True)
                        result = {
                            "status": "success",
                            "packs_count": packs_count,
                            "empty_ok": packs_count == 0,
                            "built": True,
                            "packs_dir": packs_dir_str,
                            "last_built_at": last_built,
                        }
                        _log_build_summary(
                            sid,
                            packs_count=packs_count,
                            last_built_at=last_built,
                        )
                        return result
    
            built_docs = 0
            unchanged_docs = 0
            skipped_missing = 0
            skip_reasons = {"missing_summary": 0}
            write_errors: list[tuple[str, Exception]] = []
            pack_count = 0

            use_bureaus_only = _frontend_use_bureaus_only_enabled()

            for account_dir in account_dirs:
                summary_path = account_dir / "summary.json"
                if use_bureaus_only:
                    summary = _load_json(summary_path)
                    if isinstance(summary, Mapping):
                        account_id = str(summary.get("account_id") or account_dir.name)
                    else:
                        summary = {}
                        account_id = account_dir.name
                else:
                    summary = _load_json(summary_path)
                    if not summary:
                        skipped_missing += 1
                        skip_reasons["missing_summary"] = (
                            skip_reasons.get("missing_summary", 0) + 1
                        )
                        log.warning(
                            "FRONTEND_PACK_MISSING_SUMMARY sid=%s path=%s",
                            sid,
                            summary_path,
                        )
                        continue
                    account_id = str(summary.get("account_id") or account_dir.name)

                current_account_id = account_id

                tags_path = account_dir / "tags.json"
                tags_payload_override: Any | None = None
                meta_payload: Mapping[str, Any] | None = None
                bureaus_payload: Mapping[str, Any] | None = None
                loader_pointers: dict[str, str] | None = None

                if use_bureaus_only:
                    try:
                        (
                            bureaus_payload,
                            meta_payload,
                            tags_payload_override,
                            loader_pointers,
                        ) = load_bureaus_meta_tags(account_dir)
                    except FileNotFoundError:
                        bureaus_path = account_dir / "bureaus.json"
                        log.warning(
                            "FRONTEND_PACK_MISSING_BUREAUS sid=%s account=%s path=%s",
                            sid,
                            account_id,
                            bureaus_path,
                        )
                        skip_reasons["missing_bureaus"] = (
                            skip_reasons.get("missing_bureaus", 0) + 1
                        )
                        skipped_missing += 1
                        continue

                if not tags_path.exists():
                    log.warning(
                        "FRONTEND_PACK_MISSING_TAGS sid=%s account=%s path=%s",
                        sid,
                        account_id,
                        tags_path,
                    )

                primary_issue, issues = _extract_issue_tags(
                    tags_path, payload=tags_payload_override
                )
                if not primary_issue:
                    primary_issue = "unknown"

                display_primary_issue = _coerce_display_text(primary_issue or "unknown")

                if use_bureaus_only:
                    bureaus_branches: dict[str, Mapping[str, Any]] = {
                        bureau: payload
                        for bureau, payload in bureaus_payload.items()
                        if bureau in _BUREAU_BADGES and isinstance(payload, Mapping)
                    }

                    display_payload = build_display_from_bureaus(
                        bureaus_payload, meta_payload, tags_payload_override
                    )

                    def _section_per_bureau(source: Any) -> Mapping[str, Any]:
                        if isinstance(source, Mapping):
                            per = source.get("per_bureau")
                            if isinstance(per, Mapping):
                                return per
                            return source
                        return {}

                    account_number_section = display_payload.get("account_number")
                    if isinstance(account_number_section, Mapping):
                        account_number_per_bureau = _normalize_per_bureau(
                            _section_per_bureau(account_number_section)
                        )
                        account_number_consensus = (
                            _coerce_display_text(account_number_section.get("consensus"))
                            or "--"
                        )
                    else:
                        account_number_per_bureau = _normalize_per_bureau({})
                        account_number_consensus = "--"

                    account_type_section = display_payload.get("account_type")
                    if isinstance(account_type_section, Mapping):
                        account_type_per_bureau = _normalize_per_bureau(
                            _section_per_bureau(account_type_section)
                        )
                        account_type_consensus = (
                            _coerce_display_text(account_type_section.get("consensus"))
                            or "--"
                        )
                    else:
                        account_type_per_bureau = _normalize_per_bureau({})
                        account_type_consensus = "--"

                    status_section = display_payload.get("status")
                    if isinstance(status_section, Mapping):
                        status_per_bureau = _normalize_per_bureau(
                            _section_per_bureau(status_section)
                        )
                        status_consensus = (
                            _coerce_display_text(status_section.get("consensus")) or "--"
                        )
                    else:
                        status_per_bureau = _normalize_per_bureau({})
                        status_consensus = "--"

                    balance_section = (
                        display_payload.get("balance")
                        or display_payload.get("balance_owed")
                    )
                    balance_per_bureau = _normalize_per_bureau(
                        _section_per_bureau(balance_section)
                    )

                    date_opened_per_bureau = _normalize_per_bureau(
                        _section_per_bureau(
                            display_payload.get("opened")
                            or display_payload.get("date_opened")
                        )
                    )

                    closed_date_per_bureau = _normalize_per_bureau(
                        _section_per_bureau(display_payload.get("closed_date"))
                    )

                    last_payment_per_bureau = _normalize_per_bureau(
                        _section_per_bureau(display_payload.get("last_payment"))
                    )

                    dofd_per_bureau = _normalize_per_bureau(
                        _section_per_bureau(display_payload.get("dofd"))
                    )

                    high_balance_per_bureau = _normalize_per_bureau(
                        _section_per_bureau(display_payload.get("high_balance"))
                    )

                    limit_per_bureau = _normalize_per_bureau(
                        _section_per_bureau(display_payload.get("limit"))
                    )

                    remarks_per_bureau = _normalize_per_bureau(
                        _section_per_bureau(display_payload.get("remarks"))
                    )

                    holder_values: dict[str, str] = {}
                    for holder_key in (
                        "creditor_name",
                        "creditor",
                        "furnisher_name",
                        "furnisher",
                        "name",
                    ):
                        values = _collect_bureau_field_map(bureaus_branches, holder_key)
                        if not values:
                            continue
                        for bureau, value in values.items():
                            holder_values.setdefault(bureau, value)

                    holder_per_bureau = _normalize_per_bureau(holder_values)

                    date_reported_values_raw = _collect_field_text_values(
                        bureaus_branches, "date_reported"
                    )
                    date_reported_per_bureau = _normalize_per_bureau(date_reported_values_raw)

                    fields_filled_counts = _count_filled_fields_per_bureau(
                        account_number_per_bureau,
                        account_type_per_bureau,
                        status_per_bureau,
                        balance_per_bureau,
                        high_balance_per_bureau,
                        limit_per_bureau,
                        remarks_per_bureau,
                        date_opened_per_bureau,
                        closed_date_per_bureau,
                        last_payment_per_bureau,
                        dofd_per_bureau,
                        date_reported_per_bureau,
                        holder_per_bureau,
                    )

                    reported_bureaus: list[str] = []
                    for bureau in _BUREAU_ORDER:
                        branch = bureaus_branches.get(bureau)
                        if not isinstance(branch, Mapping):
                            continue
                        if any(
                            _has_meaningful_text(branch.get(field), treat_unknown=True)
                            for field in (
                                "account_number_display",
                                "account_number",
                                "account_type",
                                "account_status",
                                "balance_owed",
                            )
                        ):
                            reported_bureaus.append(bureau)
                    if not reported_bureaus:
                        for bureau, payload in bureaus_payload.items():
                            if bureau in _BUREAU_ORDER and isinstance(payload, Mapping):
                                reported_bureaus.append(bureau)

                    bureau_summary = _prepare_bureau_payload_from_flat(
                        account_number_values=account_number_per_bureau,
                        balance_values=balance_per_bureau,
                        date_opened_values=date_opened_per_bureau,
                        closed_date_values=closed_date_per_bureau,
                        date_reported_values=date_reported_per_bureau,
                        reported_bureaus=reported_bureaus,
                    )

                    meta_holder_name, meta_holder_resolver = _resolve_meta_holder_name(
                        meta_payload
                    )
                    meta_holder_text = _coerce_display_text(meta_holder_name)

                    display_holder_name = _coerce_display_text(
                        display_payload.get("holder_name")
                    )
                    if _has_meaningful_text(meta_holder_text, treat_unknown=True):
                        display_holder_name = meta_holder_text
                        display_holder_name_resolver = meta_holder_resolver
                    elif _has_meaningful_text(display_holder_name, treat_unknown=True):
                        display_holder_name_resolver = "bureaus"
                    else:
                        display_holder_name = "Unknown"
                        display_holder_name_resolver = "default"

                    holder_name = (
                        display_holder_name
                        if _has_meaningful_text(display_holder_name, treat_unknown=True)
                        else None
                    )

                    display_primary_issue = _coerce_display_text(
                        display_payload.get("primary_issue") or primary_issue
                    )
                    if not display_primary_issue:
                        display_primary_issue = "unknown"

                    primary_issue = display_primary_issue or "unknown"

                    creditor_name_value = display_holder_name

                    account_type_value = (
                        account_type_consensus
                        if _has_meaningful_text(account_type_consensus, treat_unknown=True)
                        else None
                    )
                    status_value = (
                        status_consensus
                        if _has_meaningful_text(status_consensus, treat_unknown=True)
                        else None
                    )

                    if _has_meaningful_text(account_type_value, treat_unknown=True):
                        account_type_consensus = account_type_value
                    if _has_meaningful_text(status_value, treat_unknown=True):
                        status_consensus = status_value

                    field_resolvers = {
                        "holder_name": display_holder_name_resolver,
                        "creditor_name": display_holder_name_resolver,
                        "account_type": "bureaus.consensus"
                        if account_type_value
                        else "missing",
                        "status": "bureaus.consensus"
                        if status_value
                        else "missing",
                    }

                else:
                    flat_path = account_dir / "fields_flat.json"
                    fields_flat_payload = _load_json(flat_path)
                    if fields_flat_payload is None:
                        log.warning(
                            "FRONTEND_PACK_MISSING_FLAT sid=%s account=%s path=%s",
                            sid,
                            account_id,
                            flat_path,
                        )

                    labels = _extract_summary_labels(summary)
                    holder_name, holder_name_resolver = _derive_holder_name_from_summary(
                        summary, fields_flat_payload
                    )

                    account_number_values_raw = _collect_flat_field_per_bureau(
                        fields_flat_payload, "account_number_display"
                    )
                    account_number_per_bureau = _normalize_per_bureau(account_number_values_raw)
                    account_number_consensus = _resolve_account_number_consensus(
                        account_number_per_bureau
                    )

                    account_type_values_raw = _collect_flat_field_per_bureau(
                        fields_flat_payload, "account_type"
                    )
                    account_type_per_bureau = _normalize_per_bureau(account_type_values_raw)
                    account_type_consensus = _resolve_majority_consensus(
                        account_type_per_bureau
                    )
                    if account_type_consensus == "--":
                        fallback_account_type = labels.get("account_type") or _collect_flat_consensus(
                            fields_flat_payload, "account_type"
                        )
                        if fallback_account_type:
                            account_type_consensus = fallback_account_type

                    status_values_raw = _collect_flat_field_per_bureau(
                        fields_flat_payload, "account_status"
                    )
                    status_per_bureau = _normalize_per_bureau(status_values_raw)
                    status_consensus = _resolve_majority_consensus(status_per_bureau)
                    if status_consensus == "--":
                        fallback_status = labels.get("status") or _collect_flat_consensus(
                            fields_flat_payload, "account_status"
                        )
                        if fallback_status:
                            status_consensus = fallback_status

                    balance_values_raw = _collect_flat_field_per_bureau(
                        fields_flat_payload, "balance_owed"
                    )
                    balance_per_bureau = _normalize_per_bureau(balance_values_raw)

                    date_opened_values_raw = _collect_flat_field_per_bureau(
                        fields_flat_payload, "date_opened"
                    )
                    date_opened_per_bureau = _normalize_per_bureau(date_opened_values_raw)

                    closed_date_values_raw = _collect_flat_field_per_bureau(
                        fields_flat_payload, "closed_date"
                    )
                    closed_date_per_bureau = _normalize_per_bureau(closed_date_values_raw)

                    date_reported_values_raw = _collect_flat_field_per_bureau(
                        fields_flat_payload, "date_reported"
                    )
                    date_reported_per_bureau = _normalize_per_bureau(date_reported_values_raw)
                    last_payment_values_raw = _collect_flat_field_per_bureau(
                        fields_flat_payload, "last_payment"
                    )
                    last_payment_per_bureau = _normalize_per_bureau(last_payment_values_raw)

                    dofd_values_raw = _collect_flat_field_per_bureau(
                        fields_flat_payload, "date_of_first_delinquency"
                    )
                    if not dofd_values_raw:
                        dofd_values_raw = _collect_flat_field_per_bureau(
                            fields_flat_payload, "date_of_last_activity"
                        )
                    dofd_per_bureau = _normalize_per_bureau(dofd_values_raw)

                    high_balance_values_raw = _collect_flat_field_per_bureau(
                        fields_flat_payload, "high_balance"
                    )
                    high_balance_per_bureau = _normalize_per_bureau(high_balance_values_raw)

                    limit_values_raw = _collect_flat_field_per_bureau(
                        fields_flat_payload, "credit_limit"
                    )
                    limit_per_bureau = _normalize_per_bureau(limit_values_raw)

                    remarks_values_raw = _collect_flat_field_per_bureau(
                        fields_flat_payload, "creditor_remarks"
                    )
                    remarks_per_bureau = _normalize_per_bureau(remarks_values_raw)

                    _maybe_warn_bureaus_conflict(
                        sid=sid,
                        account_id=account_id,
                        account_dir=account_dir,
                        per_bureau_snapshot={
                            "account_number": account_number_per_bureau,
                            "account_type": account_type_per_bureau,
                            "status": status_per_bureau,
                            "balance": balance_per_bureau,
                            "balance_owed": balance_per_bureau,
                            "opened": date_opened_per_bureau,
                            "date_opened": date_opened_per_bureau,
                            "closed_date": closed_date_per_bureau,
                            "last_payment": last_payment_per_bureau,
                            "dofd": dofd_per_bureau,
                            "high_balance": high_balance_per_bureau,
                            "limit": limit_per_bureau,
                            "remarks": remarks_per_bureau,
                        },
                    )


                    reported_bureaus = _determine_reported_bureaus(summary, fields_flat_payload)
                    bureau_summary = _prepare_bureau_payload_from_flat(
                        account_number_values=account_number_per_bureau,
                        balance_values=balance_per_bureau,
                        date_opened_values=date_opened_per_bureau,
                        closed_date_values=closed_date_per_bureau,
                        date_reported_values=date_reported_per_bureau,
                        reported_bureaus=reported_bureaus,
                    )

                    field_resolvers: dict[str, str] = {"holder_name": holder_name_resolver}

                    creditor_name_values_raw = _collect_flat_field_per_bureau(
                        fields_flat_payload, "creditor_name"
                    )
                    creditor_name_consensus = _collect_flat_consensus(
                        fields_flat_payload, "creditor_name"
                    )
                    display_holder_name = _coerce_display_text(creditor_name_consensus)
                    display_holder_name_resolver = "flat_consensus"
                    if not _has_meaningful_text(display_holder_name, treat_unknown=True):
                        label_creditor = _coerce_display_text(labels.get("creditor_name"))
                        if _has_meaningful_text(label_creditor, treat_unknown=True):
                            display_holder_name = label_creditor
                            display_holder_name_resolver = "summary.labels.creditor"
                    if not _has_meaningful_text(display_holder_name, treat_unknown=True):
                        bureau_name_value = None
                        bureau_resolver = None
                        for bureau in _BUREAU_ORDER:
                            raw_value = creditor_name_values_raw.get(bureau)
                            text_value = _coerce_display_text(raw_value)
                            if _has_meaningful_text(text_value, treat_unknown=True):
                                bureau_name_value = text_value
                                bureau_resolver = f"flat_bureau.{bureau}"
                                break
                        if bureau_name_value:
                            display_holder_name = bureau_name_value
                            display_holder_name_resolver = bureau_resolver or "flat_bureau"
                    if not _has_meaningful_text(display_holder_name, treat_unknown=True):
                        flat_creditor_value = _flat_lookup(fields_flat_payload, "creditor_name")
                        flat_creditor_text = _coerce_display_text(
                            _stringify_flat_value(flat_creditor_value)
                        )
                        if _has_meaningful_text(flat_creditor_text, treat_unknown=True):
                            display_holder_name = flat_creditor_text
                            display_holder_name_resolver = "flat_root"
                    if not _has_meaningful_text(display_holder_name, treat_unknown=True):
                        summary_creditor = _extract_text(summary.get("creditor_name"))
                        if _has_meaningful_text(summary_creditor, treat_unknown=True):
                            display_holder_name = summary_creditor
                            display_holder_name_resolver = "summary.creditor_name"
                    if not _has_meaningful_text(display_holder_name, treat_unknown=True):
                        holder_name_text = _coerce_display_text(holder_name)
                        if _has_meaningful_text(holder_name_text, treat_unknown=True):
                            display_holder_name = holder_name_text
                            display_holder_name_resolver = field_resolvers.get(
                                "holder_name", "holder"
                            )
                    if not _has_meaningful_text(display_holder_name, treat_unknown=True):
                        display_holder_name = "Unknown"
                        display_holder_name_resolver = "default"

                    holder_name = holder_name or display_holder_name
                    if not _has_meaningful_text(holder_name, treat_unknown=True):
                        holder_name = display_holder_name
                    field_resolvers["holder_name"] = display_holder_name_resolver

                    creditor_name_value = labels.get("creditor_name")
                    if not _has_meaningful_text(creditor_name_value, treat_unknown=True):
                        if isinstance(fields_flat_payload, Mapping):
                            creditor_name_value = _stringify_flat_value(
                                fields_flat_payload.get("creditor_name")
                            )
                            creditor_name_resolver = "flat_root"
                        else:
                            creditor_name_value = None
                            creditor_name_resolver = "missing"
                    else:
                        creditor_name_resolver = "summary.labels.creditor"
                    if not _has_meaningful_text(creditor_name_value, treat_unknown=True):
                        creditor_name_value = _extract_text(summary.get("creditor_name"))
                        creditor_name_resolver = "summary.creditor_name"
                    if not _has_meaningful_text(creditor_name_value, treat_unknown=True):
                        creditor_name_value = _collect_flat_consensus(
                            fields_flat_payload, "creditor_name"
                        )
                        creditor_name_resolver = "flat_consensus"
                    if not _has_meaningful_text(creditor_name_value, treat_unknown=True):
                        for bureau in _BUREAU_ORDER:
                            candidate = creditor_name_values_raw.get(bureau)
                            if _has_meaningful_text(candidate, treat_unknown=True):
                                creditor_name_value = candidate
                                creditor_name_resolver = f"flat_bureau.{bureau}"
                                break
                    if not _has_meaningful_text(creditor_name_value, treat_unknown=True):
                        creditor_name_value = display_holder_name
                        creditor_name_resolver = display_holder_name_resolver

                    field_resolvers["creditor_name"] = creditor_name_resolver or "missing"

                    def _first_meaningful_from_mapping(
                        values: Mapping[str, str]
                    ) -> tuple[str | None, str | None]:
                        for bureau in _BUREAU_ORDER:
                            candidate = values.get(bureau)
                            if _has_meaningful_text(candidate, treat_unknown=True):
                                return candidate, bureau
                        for bureau, candidate in values.items():
                            if _has_meaningful_text(candidate, treat_unknown=True):
                                return candidate, bureau
                        return None, None

                    account_type_value = None
                    account_type_resolver = "missing"
                    if _has_meaningful_text(account_type_consensus, treat_unknown=True):
                        account_type_value = account_type_consensus
                        account_type_resolver = "flat_bureau"
                    else:
                        label_account_type = labels.get("account_type")
                        if _has_meaningful_text(label_account_type, treat_unknown=True):
                            account_type_value = label_account_type
                            account_type_resolver = "summary.labels.account_type"
                        else:
                            consensus_account_type = _collect_flat_consensus(
                                fields_flat_payload, "account_type"
                            )
                            if _has_meaningful_text(
                                consensus_account_type, treat_unknown=True
                            ):
                                account_type_value = consensus_account_type
                                account_type_resolver = "flat_consensus"
                            else:
                                bureau_value, bureau = _first_meaningful_from_mapping(
                                    account_type_per_bureau
                                )
                                if bureau_value:
                                    account_type_value = bureau_value
                                    account_type_resolver = (
                                        f"flat_bureau.{bureau}" if bureau else "flat_bureau"
                                    )

                    field_resolvers["account_type"] = account_type_resolver

                    if _has_meaningful_text(account_type_value, treat_unknown=True):
                        account_type_consensus = account_type_value

                    status_value = None
                    status_resolver = "missing"
                    if _has_meaningful_text(status_consensus, treat_unknown=True):
                        status_value = status_consensus
                        status_resolver = "flat_bureau"
                    else:
                        label_status = labels.get("status")
                        if _has_meaningful_text(label_status, treat_unknown=True):
                            status_value = label_status
                            status_resolver = "summary.labels.status"
                        else:
                            consensus_status = _collect_flat_consensus(
                                fields_flat_payload, "account_status"
                            )
                            if _has_meaningful_text(
                                consensus_status, treat_unknown=True
                            ):
                                status_value = consensus_status
                                status_resolver = "flat_consensus"
                            else:
                                bureau_value, bureau = _first_meaningful_from_mapping(
                                    status_per_bureau
                                )
                                if bureau_value:
                                    status_value = bureau_value
                                    status_resolver = (
                                        f"flat_bureau.{bureau}" if bureau else "flat_bureau"
                                    )

                    field_resolvers["status"] = status_resolver

                    if _has_meaningful_text(status_value, treat_unknown=True):
                        status_consensus = status_value

                    display_payload = build_display_payload(
                        holder_name=display_holder_name,
                        primary_issue=display_primary_issue,
                        account_number_per_bureau=account_number_per_bureau,
                        account_number_consensus=account_number_consensus,
                        account_type_per_bureau=account_type_per_bureau,
                        account_type_consensus=account_type_consensus,
                        status_per_bureau=status_per_bureau,
                        status_consensus=status_consensus,
                        balance_per_bureau=balance_per_bureau,
                        date_opened_per_bureau=date_opened_per_bureau,
                        closed_date_per_bureau=closed_date_per_bureau,
                        high_balance_per_bureau=high_balance_per_bureau,
                        limit_per_bureau=limit_per_bureau,
                        remarks_per_bureau=remarks_per_bureau,
                        last_payment_per_bureau=last_payment_per_bureau,
                        dofd_per_bureau=dofd_per_bureau,
                    )

                def _format_field_value(value: Any) -> str:
                    if value is None:
                        return "<missing>"
                    if isinstance(value, str):
                        return value if value else "<empty>"
                    return str(value)

                resolver_log_parts = []
                for name, resolved_value in (
                    ("holder_name", display_holder_name),
                    ("creditor_name", creditor_name_value),
                    ("account_type", account_type_value),
                    ("status", status_value),
                ):
                    resolver_log_parts.append(
                        f"[resolver={field_resolvers.get(name, 'unknown')}] "
                        f"{name}={_format_field_value(resolved_value)}"
                    )

                log.info(
                    "PACK_FIELD_RESOLUTION sid=%s account=%s %s",
                    sid,
                    account_id,
                    " ".join(resolver_log_parts),
                )

                try:
                    relative_account_dir = account_dir.relative_to(run_dir).as_posix()
                except ValueError:
                    relative_account_dir = account_dir.as_posix()

                if use_bureaus_only:
                    pointers: dict[str, str] = {}
                    if loader_pointers:
                        for key, pointer_path in loader_pointers.items():
                            path_obj = Path(pointer_path)
                            try:
                                account_relative = path_obj.relative_to(account_dir).as_posix()
                            except ValueError:
                                try:
                                    run_relative = path_obj.relative_to(run_dir).as_posix()
                                except ValueError:
                                    relative_value = path_obj.as_posix()
                                else:
                                    relative_value = run_relative
                            else:
                                relative_value = f"{relative_account_dir}/{account_relative}"

                            pointers[key] = relative_value
                    else:
                        for key, filename in (
                            ("bureaus", "bureaus.json"),
                            ("meta", "meta.json"),
                            ("tags", "tags.json"),
                        ):
                            pointers[key] = f"{relative_account_dir}/{filename}"

                    if summary_path.exists():
                        pointers["summary"] = f"{relative_account_dir}/summary.json"

                    log.info(
                        "PACK_BUILD_SOURCE=bureaus_only sid=%s account=%s fields_filled: %s pointers: %s",
                        sid,
                        account_id,
                        fields_filled_counts,
                        pointers,
                    )
                else:
                    pointers = {
                        "summary": f"{relative_account_dir}/summary.json",
                        "tags": f"{relative_account_dir}/tags.json",
                        "flat": f"{relative_account_dir}/fields_flat.json",
                    }

                full_pack_payload: dict[str, Any] | None = None

                def _ensure_full_pack_payload() -> dict[str, Any] | None:
                    nonlocal full_pack_payload
                    if full_pack_payload is None:
                        full_pack_payload = build_pack_doc(
                            sid=sid,
                            account_id=account_id,
                            creditor_name=creditor_name_value,
                            account_type=account_type_value,
                            status=status_value,
                            bureau_summary=bureau_summary,
                            holder_name=holder_name,
                            primary_issue=primary_issue,
                            display_payload=display_payload,
                            pointers=pointers,
                            issues=issues if issues else None,
                        )
                    return full_pack_payload

                need_full_payload = (
                    debug_mirror_enabled or not lean_enabled or stage_payload_full
                )
                if need_full_payload:
                    _ensure_full_pack_payload()

                stage_write_mode = "full" if stage_payload_full else "minimal"
                stage_write_reason = "ok"
                minimal_enriched = False

                if stage_payload_full:
                    full_payload = _ensure_full_pack_payload()
                    stage_pack_payload = dict(full_payload) if full_payload is not None else {}
                else:
                    stage_pack_payload = build_stage_pack_doc(
                        account_id=account_id,
                        holder_name=display_holder_name,
                        primary_issue=display_primary_issue,
                        creditor_name=creditor_name_value,
                        account_type=account_type_value,
                        status=status_value,
                        display_payload=display_payload,
                        bureau_summary=bureau_summary,
                        pointers=pointers,
                        account_number_per_bureau=account_number_per_bureau,
                        account_number_consensus=account_number_consensus,
                        account_type_per_bureau=account_type_per_bureau,
                        account_type_consensus=account_type_consensus,
                        status_per_bureau=status_per_bureau,
                        status_consensus=status_consensus,
                        balance_per_bureau=balance_per_bureau,
                        date_opened_per_bureau=date_opened_per_bureau,
                        closed_date_per_bureau=closed_date_per_bureau,
                        last_payment_per_bureau=last_payment_per_bureau,
                        dofd_per_bureau=dofd_per_bureau,
                        high_balance_per_bureau=high_balance_per_bureau,
                        limit_per_bureau=limit_per_bureau,
                        remarks_per_bureau=remarks_per_bureau,
                    )

                    full_payload = _ensure_full_pack_payload()
                    if full_payload is not None:
                        stage_pack_payload, minimal_enriched = _enrich_stage_payload_with_full(
                            stage_pack_payload, full_payload
                        )

                    if not _has_meaningful_display(stage_pack_payload.get("display")):
                        full_payload = _ensure_full_pack_payload()
                        if full_payload is not None and _has_meaningful_display(
                            full_payload.get("display")
                        ):
                            stage_pack_payload = dict(full_payload)
                            stage_write_mode = "full"
                            stage_write_reason = "failsafe"
                            log.info(
                                "PACKGEN_FAILSAFE_USED_FULL sid=%s account=%s",
                                sid,
                                account_id,
                            )
                    elif minimal_enriched:
                        stage_write_reason = "minimal_enriched"

                account_filename = _safe_account_dirname(account_id, account_dir.name)
                stage_pack_path = stage_packs_dir / f"{account_filename}.json"

                existing_stage_pack: Mapping[str, Any] | None = None
                if stage_pack_path.exists():
                    existing_payload = _load_json_payload(stage_pack_path)
                    if isinstance(existing_payload, Mapping):
                        existing_stage_pack = existing_payload

                existing_has_meaningful_data = _stage_payload_has_meaningful_data(
                    existing_stage_pack
                )
                candidate_has_meaningful_data = _stage_payload_has_meaningful_data(
                    stage_pack_payload
                )

                if existing_has_meaningful_data and not candidate_has_meaningful_data:
                    if stage_payload_full:
                        stage_write_mode = "full"
                    stage_write_reason = "skip_empty_overwrite"
                    log.info(
                        "PACKGEN_SKIP_EMPTY_OVERWRITE sid=%s account=%s",
                        sid,
                        account_id,
                    )
                    skip_reasons["placeholder"] = skip_reasons.get("placeholder", 0) + 1
                    unchanged_docs += 1
                    pack_count += 1
                    log.info(
                        "PACK_WRITE_DECISION sid=%s account=%s mode=%s guarded=skipped reason=%s",
                        sid,
                        account_id,
                        stage_write_mode,
                        stage_write_reason,
                    )
                    continue

                stage_pack_payload, preserved_fields = _preserve_stage_pack_payload(
                    existing_stage_pack, stage_pack_payload
                )
                if preserved_fields:
                    log.info(
                        "PACKGEN_PRESERVED_FIELDS sid=%s account=%s fields=%s",
                        sid,
                        account_id,
                        ",".join(sorted(preserved_fields)),
                    )

                stage_pack_payload["questions"] = _resolve_stage_pack_questions(
                    existing_pack=existing_stage_pack,
                    question_set=_QUESTION_SET,
                )

                merged_claim_links = _merge_claim_field_links(
                    stage_pack_payload.get("claim_field_links"),
                    existing_stage_pack.get("claim_field_links")
                    if isinstance(existing_stage_pack, Mapping)
                    else None,
                )
                if merged_claim_links:
                    stage_pack_payload["claim_field_links"] = merged_claim_links
                elif "claim_field_links" in stage_pack_payload:
                    stage_pack_payload.pop("claim_field_links", None)

                if _should_skip_pack_due_to_lock(
                    stage_pack_path=stage_pack_path,
                    lock_path=idempotent_lock_path,
                    lock_mtime=idempotent_lock_mtime,
                ):
                    log.info(
                        "PACKGEN_SKIP_LOCKED sid=%s account=%s pack=%s lock=%s",
                        sid,
                        account_id,
                        stage_pack_path,
                        idempotent_lock_path,
                    )
                    skip_reasons["locked"] = skip_reasons.get("locked", 0) + 1
                    unchanged_docs += 1
                    pack_count += 1
                    continue

                log.info(
                    "writing pack sid=%s (acct=%s, type=%s, status=%s)",
                    sid,
                    account_id,
                    account_type_value or "unknown",
                    status_value or "unknown",
                )

                try:
                    stage_changed = _write_json_if_changed(
                        stage_pack_path, stage_pack_payload
                    )
                    changed = stage_changed
                    if debug_mirror_enabled and full_pack_payload is not None:
                        debug_packs_dir.mkdir(parents=True, exist_ok=True)
                        mirror_path = debug_packs_dir / f"{account_filename}.full.json"
                        _write_json_if_changed(mirror_path, full_pack_payload)
                    elif not debug_mirror_enabled:
                        mirror_path = debug_packs_dir / f"{account_filename}.full.json"
                        try:
                            mirror_path.unlink()
                        except FileNotFoundError:
                            pass
                        except OSError:  # pragma: no cover - defensive logging
                            log.warning(
                                "FRONTEND_PACK_DEBUG_MIRROR_UNLINK_FAILED path=%s",
                                mirror_path,
                                exc_info=True,
                            )
                except Exception as exc:
                    log.exception(
                        "FRONTEND_PACK_WRITE_FAILED sid=%s account=%s path=%s",
                        sid,
                        account_id,
                        stage_pack_path,
                    )
                    write_errors.append((account_id, exc))
                    continue

                if stage_payload_full:
                    stage_write_mode = "full"
                log.info(
                    "PACK_WRITE_DECISION sid=%s account=%s mode=%s guarded=written reason=%s",
                    sid,
                    account_id,
                    stage_write_mode,
                    stage_write_reason,
                )
    
                if changed:
                    built_docs += 1
                else:
                    unchanged_docs += 1
    
                try:
                    relative_pack = stage_pack_path.relative_to(run_dir).as_posix()
                except ValueError:
                    relative_pack = str(stage_pack_path)
    
                pack_count += 1
    
                try:
                    relative_stage_pack = stage_pack_path.relative_to(run_dir).as_posix()
                except ValueError:
                    relative_stage_pack = str(stage_pack_path)
    
                if stage_changed and runflow_account_steps_enabled():
                    runflow_step(
                        sid,
                        "frontend",
                        "frontend_review_pack_created",
                        out={
                            "account_id": account_id,
                            "bytes": stage_pack_path.stat().st_size,
                            "path": relative_stage_pack,
                        },
                    )
    
            build_metrics = {
                "accounts": total_accounts,
                "built": built_docs,
                "skipped_missing": skipped_missing,
                "unchanged": unchanged_docs,
            }
            skip_summary = {key: value for key, value in skip_reasons.items() if value}
    
            generated_at = _now_iso()
            manifest_payload = _build_stage_manifest(
                sid=sid,
                stage_name=stage_name,
                run_dir=run_dir,
                stage_packs_dir=stage_packs_dir,
                stage_responses_dir=stage_responses_dir,
                stage_index_path=stage_index_path,
                question_set=_QUESTION_SET,
            )
            _ensure_frontend_index_redirect_stub(redirect_stub_path)
            done_status = "error" if write_errors else "success"
            _log_done(sid, pack_count, status=done_status)
    
            finish_out: dict[str, Any] = {
                "skip_reasons": skip_summary or None,
                "write_failures": len(write_errors) if write_errors else None,
            }
            finish_out = {key: value for key, value in finish_out.items() if value is not None}
    
            if pack_count == 0:
                runflow_step(
                    sid,
                    "frontend",
                    "frontend_review_no_candidates",
                )
            runflow_step(
                sid,
                "frontend",
                "frontend_review_finish",
                status=done_status,
                metrics={**build_metrics, "packs": pack_count},
                out=finish_out or None,
                error=(
                    {
                        "type": "PackWriteError",
                        "message": f"{len(write_errors)} pack writes failed",
                    }
                    if write_errors
                    else None
                ),
            )
    
            responses_count = _emit_responses_scan(sid, stage_responses_dir)
            summary = {
                "packs_count": pack_count,
                "responses_received": responses_count,
                "empty_ok": pack_count == 0,
                "skipped_missing": skipped_missing,
            }
            if built_docs:
                summary["built"] = built_docs
            if unchanged_docs:
                summary["unchanged"] = unchanged_docs
            if write_errors:
                summary["write_failures"] = len(write_errors)
            record_frontend_responses_progress(
                sid,
                accounts_published=pack_count,
                answers_received=responses_count,
                answers_required=pack_count,
            )
            runflow_stage_end(
                "frontend",
                sid=sid,
                summary=summary,
                empty_ok=pack_count == 0,
            )
    
            result = {
                "status": "success",
                "packs_count": pack_count,
                "empty_ok": pack_count == 0,
                "built": True,
                "packs_dir": packs_dir_str,
                "last_built_at": manifest_payload.get("generated_at"),
            }
            _log_build_summary(
                sid,
                packs_count=pack_count,
                last_built_at=manifest_payload.get("generated_at"),
            )
            return result
        except Exception as exc:
            runflow_step(
                sid,
                "frontend",
                "frontend_review_finish",
                status="error",
                out={
                    "account_id": current_account_id,
                    "error_class": exc.__class__.__name__,
                    "message": str(exc),
                },
                error={
                    "type": exc.__class__.__name__,
                    "message": str(exc),
                },
            )
            runflow_stage_error(
                "frontend",
                sid=sid,
                error_type=exc.__class__.__name__,
                message=str(exc),
                traceback_tail=format_exception_tail(exc),
                hint=compose_hint("frontend pack generation", exc),
            )
            raise
    finally:
        if lock_acquired and lock_path is not None:
            _release_frontend_build_lock(lock_path, sid)
    
    
__all__ = ["generate_frontend_packs_for_run"]
