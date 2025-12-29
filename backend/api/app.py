# ruff: noqa: E402
import os
import re
import sys

# Ensure the project root is always on sys.path, regardless of the
# working directory from which this module is executed.
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from dotenv import load_dotenv

from backend.api.env_sanitize import sanitize_openai_env

load_dotenv()

sanitize_openai_env()

import hashlib
import json
import logging
import mimetypes
import queue
import secrets
import tempfile
import threading
import time
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from flask import Blueprint, Flask, Response, jsonify, redirect, request, url_for
from flask import stream_with_context
from flask_cors import CORS
from jsonschema import Draft7Validator, ValidationError

import backend.config as config
from backend.analytics.batch_runner import BatchFilters, BatchRunner
from backend.api.admin import admin_bp
from backend.api.ai_endpoints import ai_bp
from backend.api.auth import require_api_key_or_role
from backend.api.config import ENABLE_BATCH_RUNNER, get_app_config
from backend.api.pipeline import run_full_pipeline
from backend.api.session_manager import (
    get_session,
    set_session,
    update_intake,
    update_session,
)
from backend.api.tasks import run_credit_repair_process  # noqa: F401
from backend.api.tasks import app as celery_app, request_frontend_review_build, smoke_task
from backend.pipeline.runs import RunManifest, get_runs_root
from backend.core.logic.compliance.upload_validator import move_uploaded_file
from backend.core.paths.frontend_review import get_frontend_review_paths
from backend.frontend.packs.claim_schema import (
    ClaimSchemaEntry,
    all_claim_keys,
    all_doc_keys_for_claim,
    all_doc_keys,
    auto_attach_base,
    get_claim_entry,
    load_claims_schema,
    resolve_issue_claims,
)
from backend.frontend.packs.config import load_frontend_stage_config
from backend.frontend.review_writer import write_client_response
from backend.core.runflow import runflow_barriers_refresh
from backend.runflow.decider import (
    reconcile_umbrella_barriers,
    refresh_frontend_stage_from_responses,
)
from backend.ai.note_style_stage import build_note_style_pack_for_account
from backend.ai.note_style.tasks import note_style_prepare_and_send_task
from backend.domain.claims import (
    CLAIM_FIELD_LINK_MAP,
    DOC_KEY_ALIAS_TO_CANONICAL,
)
from backend.api.routes_smoke import bp as smoke_bp
from backend.api.routes_run_assets import bp as run_assets_bp
from backend.api.ui_events import ui_event_bp
from backend.core import orchestrators as orch
from backend.core.case_store import api as cs_api
from backend.core.case_store.errors import NOT_FOUND, CaseStoreError
from backend.core.collectors import (
    collect_stageA_logical_accounts,
    collect_stageA_problem_accounts,
)
from backend.core.config.flags import FLAGS
from backend.core.logic.letters.explanations_normalizer import (
    extract_structured,
    sanitize,
)
from backend.core.logic.consistency import compute_field_consistency, normalize_date
from backend.core.io.json_io import update_json_in_place
from backend.core.materialize.casestore_view import build_account_view

logger = logging.getLogger(__name__)
log = logger

api_bp = Blueprint("api", __name__)
review_bp = Blueprint("review", __name__, url_prefix="/api")


@review_bp.get("/runs/<sid>/frontend/index")
def get_review_index_status(sid: str):
    return api_frontend_index(sid)


SCHEMA_DIR = Path(__file__).resolve().parent.parent / "schemas"
with open(SCHEMA_DIR / "problem_account.json") as _f:
    _problem_account_validator = Draft7Validator(json.load(_f))


_CLAIMS_SCHEMA_DATA = load_claims_schema()
_ALL_CLAIM_KEYS = all_claim_keys()
_AUTO_ATTACH_BASE = tuple(auto_attach_base())

_NOTE_STYLE_SEND_DELAY_MS = 300

_CANONICAL_DOC_KEY_TO_ALIASES: dict[str, tuple[str, ...]] = {}
if DOC_KEY_ALIAS_TO_CANONICAL:
    alias_map: dict[str, set[str]] = {}
    for alias, canonical in DOC_KEY_ALIAS_TO_CANONICAL.items():
        alias_map.setdefault(canonical, set()).add(alias)
    _CANONICAL_DOC_KEY_TO_ALIASES = {
        canonical: tuple(sorted(aliases))
        for canonical, aliases in alias_map.items()
    }
_ALL_DOC_KEYS = all_doc_keys()


FRONTEND_ACCOUNT_ID_PATTERN = re.compile(r"^idx-\d{3}$")
FRONTEND_PACK_FILENAME_PATTERN = re.compile(r"^idx-\d{3}\.json$")


_PRIMARY_ISSUE_ALIASES = {
    "late_payment": "delinquency",
    "late_history": "delinquency",
}


_request_counts: dict[str, list[float]] = defaultdict(list)


_REVIEW_STREAM_KEEPALIVE_INTERVAL = 25.0
_REVIEW_STREAM_QUEUE_WAIT_SECONDS = 1.0


class _ReviewStreamBroker:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._subscribers: dict[str, list[queue.Queue]] = defaultdict(list)

    def subscribe(self, sid: str) -> queue.Queue:
        channel: queue.Queue = queue.Queue()
        with self._lock:
            self._subscribers[sid].append(channel)
        return channel

    def unsubscribe(self, sid: str, channel: queue.Queue) -> None:
        with self._lock:
            channels = self._subscribers.get(sid)
            if not channels:
                return
            try:
                channels.remove(channel)
            except ValueError:
                return
            if not channels:
                self._subscribers.pop(sid, None)

    def publish(self, sid: str, event: str, data: Any | None = None) -> None:
        message = {"event": event, "data": data}
        with self._lock:
            subscribers = list(self._subscribers.get(sid, ()))
        for channel in subscribers:
            try:
                channel.put_nowait(message)
            except queue.Full:  # pragma: no cover - unbounded queue
                continue


_review_stream_broker = _ReviewStreamBroker()


def _runs_root_path() -> Path:
    return get_runs_root()


def _validate_sid(sid: str) -> str:
    sid = (sid or "").strip()
    if not sid or sid.startswith("/") or sid.startswith("\\"):
        raise ValueError("invalid sid")
    if "/" in sid or "\\" in sid:
        raise ValueError("invalid sid")
    parts = Path(sid).parts
    if any(part in {"", ".", ".."} for part in parts):
        raise ValueError("invalid sid")
    return sid


def _run_dir_for_sid(sid: str) -> Path:
    validated = _validate_sid(sid)
    return _runs_root_path() / validated


def _env_flag_enabled(name: str, *, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    lowered = raw.strip().lower()
    if lowered in {"", "0", "false", "no", "off"}:
        return False
    return True


def _run_has_inputs(runs_root: Path, sid: str) -> bool:
    base = runs_root / sid
    uploads_dir = base / "uploads"
    try:
        if any((uploads_dir).glob("*.pdf")):
            return True
    except OSError:
        return False

    responses_dir = base / "frontend" / "review" / "responses"
    try:
        if any(responses_dir.glob("*.result.json")):
            return True
    except OSError:
        return False

    return False


def _run_has_cases(run_dir: Path) -> bool:
    """Return ``True`` when the run has materialised frontend cases."""

    accounts_dir = run_dir / "cases" / "accounts"
    try:
        for entry in accounts_dir.iterdir():
            if entry.is_dir():
                return True
    except FileNotFoundError:
        pass
    except OSError:
        logger.warning("FRONTEND_CASES_DISCOVERY_FAILED path=%s", accounts_dir, exc_info=True)

    index_path = run_dir / "cases" / "index.json"
    try:
        if index_path.is_file():
            return True
    except OSError:
        logger.warning("FRONTEND_CASES_INDEX_STAT_FAILED path=%s", index_path, exc_info=True)

    return False


def _iter_run_sids(runs_root: Path) -> Iterable[str]:
    try:
        entries = sorted(runs_root.iterdir())
    except FileNotFoundError:
        return []
    except OSError:
        logger.warning("RUNFLOW_DISCOVER_SIDS_FAILED root=%s", runs_root, exc_info=True)
        return []

    return [entry.name for entry in entries if entry.is_dir()]


def _reconcile_runs_on_boot() -> None:
    runs_root = _runs_root_path()
    ignore_empty = _env_flag_enabled("RUNFLOW_IGNORE_EMPTY_SIDS", default=True)

    for sid in _iter_run_sids(runs_root):
        if ignore_empty and not _run_has_inputs(runs_root, sid):
            continue
        try:
            reconcile_umbrella_barriers(sid, runs_root=runs_root)
        except Exception:  # pragma: no cover - defensive logging
            logger.warning(
                "RUNFLOW_BOOT_RECONCILE_FAILED sid=%s runs_root=%s",
                sid,
                runs_root,
                exc_info=True,
            )


def _note_style_ready_from_barriers(
    barriers: Mapping[str, Any] | None,
) -> bool:
    if not isinstance(barriers, Mapping):
        return True

    merge_ready = bool(barriers.get("merge_ready"))
    review_ready = bool(barriers.get("review_ready"))
    validation_ready = bool(barriers.get("validation_ready"))

    strict_validation = _env_flag_enabled(
        "UMBRELLA_BARRIERS_STRICT_VALIDATION", default=False
    )
    if strict_validation:
        return merge_ready and review_ready and validation_ready
    return merge_ready and review_ready

def _frontend_stage_dir(run_dir: Path) -> Path:
    config = load_frontend_stage_config(run_dir)
    return config.stage_dir


def _frontend_stage_index_candidates(run_dir: Path) -> list[Path]:
    config = load_frontend_stage_config(run_dir)
    candidates: list[Path] = []

    if config.index_path not in candidates:
        candidates.append(config.index_path)

    canonical = get_frontend_review_paths(str(run_dir))
    review_index = Path(canonical["index"])
    if review_index not in candidates:
        candidates.append(review_index)

    legacy_index_value = canonical.get("legacy_index")
    if legacy_index_value:
        legacy_index = Path(legacy_index_value)
        if legacy_index not in candidates:
            candidates.append(legacy_index)

    return candidates


def _frontend_stage_packs_dir(run_dir: Path) -> Path:
    config = load_frontend_stage_config(run_dir)
    return config.packs_dir


def _is_valid_frontend_account_id(account_id: str) -> bool:
    return bool(FRONTEND_ACCOUNT_ID_PATTERN.fullmatch((account_id or "").strip()))


def _safe_relative_path(run_dir: Path, relative_path: str) -> Path:
    """Return ``relative_path`` resolved under ``run_dir`` with guardrails.

    ``relative_path`` can include Windows-style path separators even when the
    backend is running on POSIX.  Converting to a POSIX-style string before
    constructing :class:`Path` objects ensures the separators are interpreted as
    directory boundaries instead of literal ``"\\"`` characters in the final
    URL responses.
    """

    normalized = relative_path.replace("\\", "/")
    rel = Path(normalized)
    if rel.is_absolute():
        return rel

    candidate = (run_dir / rel).resolve(strict=False)
    try:
        base = run_dir.resolve(strict=False)
    except FileNotFoundError:
        base = run_dir

    if base == candidate or base in candidate.parents:
        return candidate

    raise ValueError("path escapes run directory")


def _load_json_file(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _parse_created_at(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def _latest_run_from_index(runs_root: Path) -> str | None:
    index_path = runs_root / "index.json"
    if not index_path.is_file():
        return None

    try:
        payload = _load_json_file(index_path)
    except Exception:  # pragma: no cover - defensive logging
        logger.warning("RUN_INDEX_READ_FAILED path=%s", index_path, exc_info=True)
        return None

    runs = payload.get("runs") if isinstance(payload, Mapping) else None
    if not isinstance(runs, Iterable):
        return None

    latest_sid: str | None = None
    latest_created: datetime | None = None

    for entry in runs:
        if not isinstance(entry, Mapping):
            continue
        sid = entry.get("sid")
        if not isinstance(sid, str) or not sid:
            continue

        created_at = _parse_created_at(entry.get("created_at"))
        if latest_created is None:
            latest_sid = sid
            latest_created = created_at
            continue

        if created_at is None:
            continue

        if latest_created is None or created_at > latest_created:
            latest_sid = sid
            latest_created = created_at

    return latest_sid


def _latest_run_from_directories(runs_root: Path) -> str | None:
    try:
        entries = list(runs_root.iterdir())
    except FileNotFoundError:
        return None

    latest_sid: str | None = None
    latest_mtime: float | None = None

    for entry in entries:
        if not entry.is_dir():
            continue

        try:
            stat = entry.stat()
        except OSError:
            continue

        mtime = stat.st_mtime
        if latest_mtime is None or mtime > latest_mtime:
            latest_mtime = mtime
            latest_sid = entry.name

    return latest_sid


def _format_sse(event: str | None, data: Any | None) -> bytes:
    lines: list[str] = []
    if event:
        lines.append(f"event: {event}")

    payload: str
    if data is None:
        payload = "null"
    else:
        try:
            payload = json.dumps(data, ensure_ascii=False)
        except TypeError:
            payload = json.dumps(str(data), ensure_ascii=False)

    for chunk in payload.splitlines() or [""]:
        lines.append(f"data: {chunk}")

    return ("\n".join(lines) + "\n\n").encode("utf-8")


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _response_filename_for_account(account_id: str) -> str:
    trimmed = (account_id or "").strip()
    match = re.fullmatch(r"idx-(\d+)", trimmed)
    number: int | None = None
    if match:
        number = int(match.group(1))
    else:
        try:
            number = int(trimmed)
        except ValueError:
            number = None

    if number is not None:
        return f"idx-{number:03d}.result.json"

    sanitized = re.sub(r"[^A-Za-z0-9_.-]", "_", trimmed) or "account"
    return f"{sanitized}.result.json"


def _schedule_note_style_send_after_delay(
    sid: str,
    account_id: str,
    *,
    runs_root: Path,
    delay_ms: int = _NOTE_STYLE_SEND_DELAY_MS,
) -> None:
    delay_seconds = max(delay_ms, 0) / 1000.0

    logger.info(
        "NOTE_STYLE_SEND_SCHEDULED sid=%s account_id=%s delay_ms=%s",
        sid,
        account_id,
        int(delay_seconds * 1000),
    )

    if runs_root is None:
        runs_root_arg: str | None = None
    else:
        try:
            runs_root_arg = os.fspath(runs_root)
        except TypeError:
            runs_root_arg = str(runs_root)

    def _invoke() -> None:
        try:
            if runs_root_arg is None:
                note_style_prepare_and_send_task.delay(sid)
            else:
                note_style_prepare_and_send_task.delay(
                    sid, runs_root=runs_root_arg
                )
            logger.info(
                "NOTE_STYLE_SEND_TASK_ENQUEUED sid=%s account_id=%s", sid, account_id
            )
        except Exception:  # pragma: no cover - defensive logging
            logger.warning(
                "NOTE_STYLE_SEND_SCHEDULE_FAILED sid=%s account_id=%s",
                sid,
                account_id,
                exc_info=True,
            )

    if delay_seconds <= 0:
        _invoke()
        return

    timer = threading.Timer(delay_seconds, _invoke)
    timer.daemon = True
    timer.start()


def _sanitize_upload_component(value: str, default: str = "item") -> str:
    trimmed = value.strip() if isinstance(value, str) else ""
    if not trimmed:
        return default
    sanitized = re.sub(r"[^A-Za-z0-9_.-]", "_", trimmed)
    return sanitized or default


_REVIEW_ALLOWED_UPLOAD_EXTENSIONS = {"pdf", "jpg", "jpeg", "png", "doc", "docx"}
_REVIEW_UPLOAD_MAX_BYTES = 10 * 1024 * 1024  # 10 MB
_REVIEW_DOC_KEY_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+$")


def _slugify_upload_name(filename: str) -> str:
    name = Path(filename or "document").name
    stem = Path(name).stem
    slug_source = stem.lower().strip()
    if not slug_source:
        slug_source = "document"
    slug = re.sub(r"[^a-z0-9]+", "-", slug_source).strip("-")
    return slug or "document"


def _extract_upload_extension(filename: str) -> str:
    suffix = Path(filename or "").suffix
    if not suffix:
        return ""
    return suffix.lstrip(".").lower()


def _resolve_review_doc_id(doc_id: str, run_dir: Path, uploads_dir: Path) -> Path | None:
    project_root = _runs_root_path().parent
    try:
        candidate = _safe_relative_path(project_root, doc_id)
    except ValueError:
        return None

    uploads_base = uploads_dir.resolve(strict=False)
    try:
        candidate.relative_to(uploads_base)
    except ValueError:
        return None

    if not candidate.is_file():
        return None

    return candidate


def _doc_id_for_path(path: Path) -> str:
    project_root = _runs_root_path().parent
    resolved = path.resolve(strict=False)
    return resolved.relative_to(project_root).as_posix()


def _merge_attachment_doc_ids(
    target: dict[str, list[str]],
    doc_key: str,
    doc_ids: Sequence[str],
) -> None:
    if not doc_ids:
        return

    canonical = DOC_KEY_ALIAS_TO_CANONICAL.get(doc_key, doc_key)
    related_keys: set[str] = {doc_key, canonical}
    related_keys.update(_CANONICAL_DOC_KEY_TO_ALIASES.get(canonical, ()))

    for key in related_keys:
        existing = target.setdefault(key, [])
        for doc_id in doc_ids:
            if doc_id not in existing:
                existing.append(doc_id)


_FRONTEND_BUREAUS: tuple[str, ...] = ("transunion", "experian", "equifax")


def _load_json_mapping(path: Path) -> Mapping[str, Any] | None:
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    except OSError:
        log.warning("FRONTEND_REVIEW_JSON_READ_FAILED path=%s", path, exc_info=True)
        return None

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        log.warning("FRONTEND_REVIEW_JSON_PARSE_FAILED path=%s", path, exc_info=True)
        return None

    if isinstance(payload, Mapping):
        return payload

    return None


def _account_dir_for_frontend(run_dir: Path, account_id: str) -> Path | None:
    accounts_dir = run_dir / "cases" / "accounts"
    direct = accounts_dir / account_id
    if direct.is_dir():
        return direct

    if not accounts_dir.is_dir():
        return None

    for candidate in accounts_dir.iterdir():
        if not candidate.is_dir():
            continue
        summary_payload = _load_json_mapping(candidate / "summary.json")
        if not isinstance(summary_payload, Mapping):
            continue
        summary_account_id = summary_payload.get("account_id")
        if isinstance(summary_account_id, str) and summary_account_id.strip() == account_id:
            return candidate

    return None


def _load_account_validation_context(
    run_dir: Path, account_id: str
) -> tuple[Mapping[str, Any] | None, dict[str, Mapping[str, Any]]]:
    account_dir = _account_dir_for_frontend(run_dir, account_id)
    if account_dir is None:
        return None, {}

    summary_payload = _load_json_mapping(account_dir / "summary.json")
    bureaus_payload_raw = _load_json_mapping(account_dir / "bureaus.json")

    bureaus: dict[str, Mapping[str, Any]] = {}
    if isinstance(bureaus_payload_raw, Mapping):
        for bureau in _FRONTEND_BUREAUS:
            branch = bureaus_payload_raw.get(bureau)
            if isinstance(branch, Mapping):
                bureaus[bureau] = dict(branch)

    return summary_payload, bureaus


def _extract_validation_block(summary_payload: Mapping[str, Any] | None) -> Mapping[str, Any] | None:
    if not isinstance(summary_payload, Mapping):
        return None
    block = summary_payload.get("validation_requirements")
    return block if isinstance(block, Mapping) else None


def _resolve_field_consistency_map(
    summary_payload: Mapping[str, Any] | None,
    bureaus_payload: Mapping[str, Mapping[str, Any]],
) -> Mapping[str, Any]:
    candidates: list[Mapping[str, Any]] = []
    if isinstance(summary_payload, Mapping):
        validation_block = _extract_validation_block(summary_payload)
        if isinstance(validation_block, Mapping):
            field_consistency = validation_block.get("field_consistency")
            if isinstance(field_consistency, Mapping):
                candidates.append(field_consistency)
        field_consistency = summary_payload.get("field_consistency")
        if isinstance(field_consistency, Mapping):
            candidates.append(field_consistency)

    for candidate in candidates:
        if candidate:
            return candidate

    if bureaus_payload:
        try:
            return compute_field_consistency(dict(bureaus_payload))
        except Exception:  # pragma: no cover - defensive
            log.warning("FRONTEND_REVIEW_COMPUTE_CONSISTENCY_FAILED", exc_info=True)

    return {}


def _collect_reason_codes(validation_block: Mapping[str, Any] | None) -> dict[str, list[str]]:
    if not isinstance(validation_block, Mapping):
        return {}

    findings = validation_block.get("findings")
    if not isinstance(findings, Sequence):
        return {}

    collected: dict[str, set[str]] = {}
    for entry in findings:
        if not isinstance(entry, Mapping):
            continue
        field = entry.get("field")
        reason = entry.get("reason_code")
        if not isinstance(field, str) or not isinstance(reason, str):
            continue
        normalized = reason.strip().upper()
        if not normalized or not normalized.startswith("C"):
            continue
        collected.setdefault(field, set()).add(normalized)

    return {field: sorted(values) for field, values in collected.items()}


def _bureau_value_snapshot(
    bureaus_payload: Mapping[str, Mapping[str, Any]], field: str
) -> dict[str, str]:
    snapshot: dict[str, str] = {}
    for bureau in _FRONTEND_BUREAUS:
        branch = bureaus_payload.get(bureau)
        value = "--"
        if isinstance(branch, Mapping):
            raw_value = branch.get(field)
            if isinstance(raw_value, str):
                cleaned = raw_value.strip()
                if cleaned:
                    value = cleaned
            elif raw_value is not None:
                cleaned = str(raw_value).strip()
                if cleaned:
                    value = cleaned
        snapshot[bureau] = value
    return snapshot


def _build_field_snapshots(
    summary_payload: Mapping[str, Any] | None,
    bureaus_payload: Mapping[str, Mapping[str, Any]],
    fields: Sequence[str],
) -> dict[str, Any]:
    if not fields:
        return {}

    field_consistency = _resolve_field_consistency_map(summary_payload, bureaus_payload)
    validation_block = _extract_validation_block(summary_payload)
    reason_codes_map = _collect_reason_codes(validation_block)

    snapshots: dict[str, Any] = {}
    for field in fields:
        consistency_details = (
            field_consistency.get(field)
            if isinstance(field_consistency, Mapping)
            else None
        )
        consensus_value = "unanimous"
        if isinstance(consistency_details, Mapping):
            consensus_value = str(consistency_details.get("consensus") or "")

        consistent = consensus_value.lower() == "unanimous"
        if not isinstance(consistency_details, Mapping) and not bureaus_payload:
            consistent = True

        codes = reason_codes_map.get(field, [])
        if not codes and not consistent:
            codes = ["INCONSISTENT_BUREAUS"]

        snapshots[field] = {
            "by_bureau": _bureau_value_snapshot(bureaus_payload, field),
            "consistent": consistent,
            "c_codes": codes,
        }

    return snapshots


def _generate_attachment_id() -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"att_{timestamp}_{secrets.token_hex(4)}"


def _sanitize_field_snapshots(snapshot_map: Mapping[str, Any]) -> dict[str, Any]:
    sanitized: dict[str, Any] = {}
    for field, details in snapshot_map.items():
        if not isinstance(field, str) or not isinstance(details, Mapping):
            continue

        by_bureau_raw = details.get("by_bureau")
        by_bureau: dict[str, str] = {}
        if isinstance(by_bureau_raw, Mapping):
            for bureau in _FRONTEND_BUREAUS:
                value = by_bureau_raw.get(bureau)
                if isinstance(value, str):
                    by_bureau[bureau] = value
        for bureau in _FRONTEND_BUREAUS:
            by_bureau.setdefault(bureau, "--")

        consistent = details.get("consistent")
        if isinstance(consistent, bool):
            consistent_value = consistent
        else:
            consistent_value = bool(consistent)

        codes_raw = details.get("c_codes")
        codes: list[str] = []
        if isinstance(codes_raw, Sequence) and not isinstance(codes_raw, (str, bytes)):
            for code in codes_raw:
                if isinstance(code, str) and code:
                    codes.append(code)

        sanitized[field] = {
            "by_bureau": by_bureau,
            "consistent": consistent_value,
            "c_codes": codes,
        }

    return sanitized


def _clone_field_snapshots_payload(snapshot_map: Mapping[str, Any]) -> dict[str, Any]:
    cloned: dict[str, Any] = {}
    for field, details in snapshot_map.items():
        if not isinstance(field, str) or not isinstance(details, Mapping):
            continue

        by_bureau: dict[str, str] = {}
        source_bureaus = details.get("by_bureau")
        if isinstance(source_bureaus, Mapping):
            for bureau in _FRONTEND_BUREAUS:
                value = source_bureaus.get(bureau)
                if isinstance(value, str):
                    by_bureau[bureau] = value
                else:
                    by_bureau[bureau] = "--"
        else:
            for bureau in _FRONTEND_BUREAUS:
                by_bureau[bureau] = "--"

        consistent_value = details.get("consistent")
        if isinstance(consistent_value, bool):
            consistent = consistent_value
        else:
            consistent = bool(consistent_value)

        codes: list[str] = []
        codes_raw = details.get("c_codes")
        if isinstance(codes_raw, Sequence) and not isinstance(codes_raw, (str, bytes)):
            for code in codes_raw:
                if isinstance(code, str) and code:
                    codes.append(code)

        cloned[field] = {
            "by_bureau": by_bureau,
            "consistent": consistent,
            "c_codes": codes,
        }

    return cloned


def _sanitize_attachment_record(entry: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(entry, Mapping):
        return {}

    sanitized: dict[str, Any] = {}

    for key in (
        "id",
        "claim_type",
        "filename",
        "stored_path",
        "mime",
        "sha1",
        "uploaded_at",
        "doc_id",
    ):
        value = entry.get(key)
        if isinstance(value, str) and value:
            sanitized[key] = value

    size_value = entry.get("size")
    if isinstance(size_value, (int, float)):
        sanitized["size"] = int(size_value)

    hot_fields: list[str] = []
    raw_hot_fields = entry.get("hot_fields")
    if isinstance(raw_hot_fields, Sequence) and not isinstance(
        raw_hot_fields, (str, bytes)
    ):
        for field in raw_hot_fields:
            if isinstance(field, str) and field:
                hot_fields.append(field)
    sanitized["hot_fields"] = hot_fields

    field_snapshots_raw = entry.get("field_snapshots")
    if isinstance(field_snapshots_raw, Mapping):
        sanitized["field_snapshots"] = _clone_field_snapshots_payload(
            field_snapshots_raw
        )
    else:
        sanitized["field_snapshots"] = {}

    return sanitized


def _claim_field_links_payload() -> dict[str, list[str]]:
    return {key: list(values) for key, values in CLAIM_FIELD_LINK_MAP.items()}


def _append_attachment_summary(
    responses_dir: Path,
    *,
    sid: str,
    account_id: str,
    attachments: Sequence[Mapping[str, Any]],
) -> None:
    if not attachments:
        return

    summary_path = responses_dir / f"{account_id}.summary.json"

    def _update(current: Any) -> Any:
        payload = current if isinstance(current, Mapping) else {}
        record = dict(payload)
        record["sid"] = sid
        record["account_id"] = account_id

        existing: list[dict[str, Any]] = []
        prior = record.get("attachments")
        if isinstance(prior, Sequence):
            for item in prior:
                sanitized_entry = _sanitize_attachment_record(item)
                if sanitized_entry:
                    existing.append(sanitized_entry)

        for entry in attachments:
            sanitized_entry = _sanitize_attachment_record(entry)
            if sanitized_entry:
                existing.append(sanitized_entry)

        record["attachments"] = existing

        union: list[str] = []
        seen: set[str] = set()
        for entry in existing:
            hot_fields = entry.get("hot_fields")
            if not isinstance(hot_fields, Sequence):
                continue
            for field in hot_fields:
                if not isinstance(field, str):
                    continue
                if field not in seen:
                    seen.add(field)
                    union.append(field)

        record["hot_fields_union"] = union
        return record

    update_json_in_place(summary_path, _update)


def _build_account_attachments_summary(
    responses_dir: Path, account_id: str, *, sid: str
) -> dict[str, Any]:
    summary_path = responses_dir / f"{account_id}.summary.json"
    payload = _load_json_mapping(summary_path)

    attachments: list[dict[str, Any]] = []
    if isinstance(payload, Mapping):
        raw_entries = payload.get("attachments")
        if isinstance(raw_entries, Sequence):
            for entry in raw_entries:
                sanitized = _sanitize_attachment_record(entry)
                if sanitized:
                    attachments.append(sanitized)

    union: list[str] = []
    seen: set[str] = set()
    for entry in attachments:
        hot_fields = entry.get("hot_fields")
        if not isinstance(hot_fields, Sequence) or isinstance(hot_fields, (str, bytes)):
            continue
        for field in hot_fields:
            if not isinstance(field, str) or not field or field in seen:
                continue
            seen.add(field)
            union.append(field)

    return {
        "sid": sid,
        "account_id": account_id,
        "attachments": attachments,
        "hot_fields_union": union,
    }


def _merge_collectors(
    problems: list[Mapping[str, Any]] | None,
    logical: list[Mapping[str, Any]] | None,
) -> list[Mapping[str, Any]]:
    """Merge Stage-A collector outputs while dropping parser artifacts."""
    merged: dict[tuple[str | None, str | None], dict] = {}
    for acc in problems or []:
        key = (acc.get("account_id"), acc.get("bureau"))
        merged[key] = dict(acc)
    for acc in logical or []:
        key = (acc.get("account_id"), acc.get("bureau"))
        if key in merged:
            merged[key].update(acc)
        else:
            merged[key] = dict(acc)
    result: list[Mapping[str, Any]] = []
    for acc in merged.values():
        acc.pop("source_stage", None)
        result.append(acc)
    return result


@api_bp.route("/")
def index():
    return jsonify({"status": "ok", "message": "API is up"})


@api_bp.route("/api/smoke", methods=["GET"])
def smoke():
    """Lightweight health check verifying Celery round-trip."""
    result = smoke_task.delay().get(timeout=10)
    return jsonify({"ok": True, "celery": result})


@api_bp.route("/api/runs/last", methods=["GET"])
def api_runs_last():
    runs_root = _runs_root_path()
    sid = _latest_run_from_index(runs_root) or _latest_run_from_directories(runs_root)
    if not sid:
        return jsonify({"error": "no_runs"}), 404
    return jsonify({"sid": sid})


@api_bp.route("/api/batch-runner", methods=["POST"])
@require_api_key_or_role(roles={"batch_runner"})
def run_batch_job():
    if not ENABLE_BATCH_RUNNER:
        return jsonify({"status": "error", "message": "batch runner disabled"}), 403
    data = request.get_json(force=True)
    filters_data = data.get("filters", {}) or {}
    action_tags = filters_data.get("action_tags")
    if not action_tags:
        return (
            jsonify({"status": "error", "message": "action_tags required"}),
            400,
        )

    cycle_range = filters_data.get("cycle_range")
    if isinstance(cycle_range, list):
        cycle_range = tuple(cycle_range)  # type: ignore[assignment]

    filters = BatchFilters(
        action_tags=action_tags,
        family_ids=filters_data.get("family_ids"),
        cycle_range=cycle_range,
        start_ts=filters_data.get("start_ts"),
        end_ts=filters_data.get("end_ts"),
        page_size=filters_data.get("page_size"),
        page_token=filters_data.get("page_token"),
    )

    fmt = data.get("format", "json")
    runner = BatchRunner()
    job_id = runner.run(filters, fmt)
    return jsonify({"status": "ok", "job_id": job_id})


def _load_frontend_stage_manifest(run_dir: Path) -> tuple[Path, Any] | None:
    for candidate in _frontend_stage_index_candidates(run_dir):
        if not candidate.is_file():
            continue
        try:
            payload = _load_json_file(candidate)
        except json.JSONDecodeError:
            logger.warning(
                "FRONTEND_STAGE_INDEX_DECODE_FAILED path=%s", candidate, exc_info=True
            )
            continue
        except OSError:
            logger.warning(
                "FRONTEND_STAGE_INDEX_READ_FAILED path=%s", candidate, exc_info=True
            )
            continue

        return candidate, payload
    return None


def _load_frontend_pack(pack_path: Path) -> Any:
    try:
        return _load_json_file(pack_path)
    except json.JSONDecodeError as exc:  # pragma: no cover - defensive logging
        logger.warning(
            "FRONTEND_PACK_DECODE_FAILED path=%s error=%s", pack_path, exc, exc_info=True
        )
        raise
    except OSError as exc:  # pragma: no cover - defensive logging
        logger.warning(
            "FRONTEND_PACK_READ_FAILED path=%s error=%s", pack_path, exc, exc_info=True
        )
        raise


def _unwrap_pack_payload(payload: Any) -> Mapping[str, Any] | None:
    if not isinstance(payload, Mapping):
        return None
    pack_value = payload.get("pack")
    if isinstance(pack_value, Mapping):
        return pack_value
    return payload


def _extract_pack_questions(payload: Any, account_id: str) -> list[Any] | None:
    if isinstance(payload, Mapping):
        questions = payload.get("questions")
        if isinstance(questions, list):
            return list(questions)

        for entry in _iter_frontend_pack_entries(payload):
            if entry.get("account_id") != account_id:
                continue
            entry_questions = entry.get("questions")
            if isinstance(entry_questions, list):
                return list(entry_questions)
    return None


EMPTY_TOKENS = {"", "--", "n/a", "na"}
BUREAU_ORDER = ("experian", "transunion", "equifax")
_BUREAU_RANK_DEFAULT = len(BUREAU_ORDER)
_KNOWN_BUREAUS = set(BUREAU_ORDER)


def is_empty(value: Any) -> bool:
    if value is None:
        return True
    text = str(value).strip()
    if not text:
        return True
    return text.lower() in EMPTY_TOKENS


def digits_count(value: Any) -> int:
    if value is None:
        return 0
    return sum(char.isdigit() for char in str(value))


def _bureau_rank(bureau: str) -> int:
    try:
        return BUREAU_ORDER.index(bureau)
    except ValueError:
        return _BUREAU_RANK_DEFAULT


def _stringify(value: Any) -> str:
    if value is None:
        return ""
    return value if isinstance(value, str) else str(value)


def pick_majority(values: Mapping[str, Any] | None) -> tuple[str, str, str]:
    if not isinstance(values, Mapping):
        values = {}

    non_empty: dict[str, str] = {}
    for bureau_key, raw_value in values.items():
        bureau = str(bureau_key)
        if _KNOWN_BUREAUS and bureau not in _KNOWN_BUREAUS:
            continue
        if is_empty(raw_value):
            continue
        non_empty[bureau] = _stringify(raw_value)

    if not non_empty:
        return ("", "", "empty")

    non_empty_count = len(non_empty)
    if non_empty_count == 1:
        bureau, value = next(iter(non_empty.items()))
        return (_stringify(value), bureau, "general")

    buckets: dict[str, dict[str, Any]] = {}
    for bureau, text in non_empty.items():
        key = text.strip().lower()
        entry = buckets.setdefault(
            key,
            {"count": 0, "sources": [], "value": text, "value_source": bureau},
        )
        entry["count"] += 1
        entry["sources"].append(bureau)
        if _bureau_rank(bureau) < _bureau_rank(entry["value_source"]):
            entry["value"] = text
            entry["value_source"] = bureau

    top_entry = max(buckets.values(), key=lambda item: item["count"])
    if top_entry["count"] >= 2 or top_entry["count"] > (non_empty_count / 2):
        source = min(top_entry["sources"], key=_bureau_rank)
        return (_stringify(top_entry["value"]), source, "majority")

    source = min(non_empty.keys(), key=_bureau_rank)
    return (_stringify(non_empty[source]), source, "precedence")


def pick_account_number(values: Mapping[str, Any] | None) -> tuple[str, str]:
    if not isinstance(values, Mapping):
        values = {}

    candidates: list[tuple[str, str, int, int]] = []
    for bureau_key, raw_value in values.items():
        bureau = str(bureau_key)
        if _KNOWN_BUREAUS and bureau not in _KNOWN_BUREAUS:
            continue
        if is_empty(raw_value):
            continue
        text = _stringify(raw_value)
        candidates.append((bureau, text, digits_count(text), len(text)))

    if not candidates:
        return ("", "")

    max_digits = max(candidate[2] for candidate in candidates)
    pool = [candidate for candidate in candidates if candidate[2] == max_digits]
    if len(pool) == 1:
        bureau, value, _, _ = pool[0]
        return (_stringify(value), bureau)

    best_rank = min(_bureau_rank(candidate[0]) for candidate in pool)
    ranked = [candidate for candidate in pool if _bureau_rank(candidate[0]) == best_rank]
    if len(ranked) == 1:
        bureau, value, _, _ = ranked[0]
        return (_stringify(value), bureau)

    bureau, value, _, _ = max(ranked, key=lambda candidate: candidate[3])
    return (_stringify(value), bureau)


def _extract_bureau_values(block: Any) -> dict[str, Any]:
    if not isinstance(block, Mapping):
        return {}

    per_bureau = block.get("per_bureau")
    if isinstance(per_bureau, Mapping):
        return {
            str(bureau): per_bureau[bureau]
            for bureau in per_bureau
            if str(bureau) in _KNOWN_BUREAUS
        }

    return {
        str(bureau): block[bureau]
        for bureau in block
        if str(bureau) in _KNOWN_BUREAUS
    }


def _normalize_resolved_date(value: str) -> str:
    normalized = normalize_date(value)
    return normalized if normalized else value


def resolve_display_fields(display: Mapping[str, Any] | None) -> dict[str, dict[str, str]]:
    display_mapping: dict[str, Any]
    if isinstance(display, Mapping):
        display_mapping = dict(display)
    else:
        display_mapping = {}

    account_number_values = _extract_bureau_values(display_mapping.get("account_number"))
    account_number_value, account_number_source = pick_account_number(account_number_values)

    account_type_value, account_type_source, account_type_method = pick_majority(
        _extract_bureau_values(display_mapping.get("account_type"))
    )
    status_value, status_source, status_method = pick_majority(
        _extract_bureau_values(display_mapping.get("status"))
    )
    balance_value, balance_source, balance_method = pick_majority(
        _extract_bureau_values(display_mapping.get("balance_owed"))
    )

    date_opened_values = _extract_bureau_values(display_mapping.get("date_opened"))
    closed_date_values = _extract_bureau_values(display_mapping.get("closed_date"))

    date_opened_value, date_opened_source, date_opened_method = pick_majority(
        date_opened_values
    )
    if date_opened_value:
        date_opened_value = _normalize_resolved_date(date_opened_value)

    closed_date_value, closed_date_source, closed_date_method = pick_majority(
        closed_date_values
    )
    if closed_date_value:
        closed_date_value = _normalize_resolved_date(closed_date_value)

    return {
        "account_number": {
            "value": account_number_value,
            "source": account_number_source,
            "method": "max_digits",
        },
        "account_type": {
            "value": account_type_value,
            "source": account_type_source,
            "method": account_type_method,
        },
        "status": {
            "value": status_value,
            "source": status_source,
            "method": status_method,
        },
        "balance_owed": {
            "value": balance_value,
            "source": balance_source,
            "method": balance_method,
        },
        "date_opened": {
            "value": date_opened_value,
            "source": date_opened_source,
            "method": date_opened_method,
        },
        "closed_date": {
            "value": closed_date_value,
            "source": closed_date_source,
            "method": closed_date_method,
        },
    }


def _prepare_pack_response(payload: Any, account_id: str) -> dict[str, Any]:
    pack_mapping = _unwrap_pack_payload(payload)
    result: dict[str, Any] = dict(pack_mapping) if isinstance(pack_mapping, Mapping) else {}

    if isinstance(payload, Mapping):
        for key in ("answers", "response"):
            if key in result:
                continue
            value = payload.get(key)
            if isinstance(value, Mapping):
                result[key] = value

    if account_id and not result.get("account_id"):
        result["account_id"] = account_id

    if "claims" not in result:
        primary_issue = result.get("primary_issue")
        if not primary_issue:
            display = result.get("display")
            if isinstance(display, Mapping):
                primary_issue = display.get("primary_issue")
        claims_payload = _resolved_claims_payload(primary_issue)
        if claims_payload:
            result["claims"] = claims_payload

    if "claim_field_links" not in result:
        result["claim_field_links"] = _claim_field_links_payload()

    display_payload = result.get("display")
    if isinstance(display_payload, Mapping):
        display_mapping = dict(display_payload)
        display_mapping["resolved"] = resolve_display_fields(display_mapping)
        result["display"] = display_mapping

    return result


def _resolve_run_manifest(run_dir: Path) -> Path | None:
    candidate = run_dir / "manifest.json"
    return candidate if candidate.is_file() else None


def _load_run_manifest(run_dir: Path) -> Mapping[str, Any] | None:
    manifest_path = _resolve_run_manifest(run_dir)
    if manifest_path is None:
        return None

    try:
        payload = _load_json_file(manifest_path)
    except json.JSONDecodeError:
        logger.warning(
            "RUN_MANIFEST_DECODE_FAILED sid=%s path=%s", run_dir.name, manifest_path
        )
        raise
    except OSError:
        logger.warning(
            "RUN_MANIFEST_READ_FAILED sid=%s path=%s", run_dir.name, manifest_path
        )
        raise

    if isinstance(payload, Mapping):
        return payload
    return None


def _iter_frontend_pack_entries(payload: Mapping[str, Any]) -> Iterable[Mapping[str, Any]]:
    for key in ("items", "packs"):
        entries = payload.get(key)
        if isinstance(entries, list):
            for entry in entries:
                if isinstance(entry, Mapping):
                    yield entry


def _to_posix_path(value: str) -> str:
    return value.replace("\\", "/")


def _normalize_path_value(value: Any) -> Any:
    if isinstance(value, str):
        return _to_posix_path(value)
    if isinstance(value, Mapping):
        return _normalize_path_like_entries(dict(value))
    if isinstance(value, list):
        return [_normalize_path_value(item) for item in value]
    return value


def _is_path_like_key(key: str) -> bool:
    lowered = key.lower()
    if "path" in lowered or "dir" in lowered:
        return True
    return lowered in {"index", "file", "packs"}


def _normalize_path_like_entries(payload: Any) -> Any:
    if isinstance(payload, Mapping):
        for key in list(payload.keys()):
            value = payload[key]
            if _is_path_like_key(key):
                payload[key] = _normalize_path_value(value)
            else:
                payload[key] = _normalize_path_like_entries(value)
        return payload
    if isinstance(payload, list):
        return [_normalize_path_like_entries(item) for item in payload]
    return payload


def _normalize_primary_issue(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip().lower()
    if not normalized:
        return None
    return _PRIMARY_ISSUE_ALIASES.get(normalized, normalized)


def _claim_entry_to_payload(entry: ClaimSchemaEntry) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "key": entry.key,
        "title": entry.title,
        "requires": list(entry.requires),
    }
    if entry.description:
        payload["description"] = entry.description
    if entry.optional:
        payload["optional"] = list(entry.optional)
    if entry.auto_attach:
        payload["autoAttach"] = list(entry.auto_attach)
    if entry.min_uploads is not None:
        payload["minUploads"] = entry.min_uploads
    return payload


def _resolved_claims_payload(primary_issue: Any) -> dict[str, Any] | None:
    normalized_issue = _normalize_primary_issue(primary_issue)
    base_docs, claim_entries = resolve_issue_claims(normalized_issue)
    items = [_claim_entry_to_payload(entry) for entry in claim_entries]
    return {"autoAttachBase": list(base_docs), "items": items}


def _resolve_pack_primary_issue(run_dir: Path, account_id: str) -> str | None:
    stage_pack = _stage_pack_path_for_account(run_dir, account_id)
    if stage_pack is None or not stage_pack.is_file():
        return None
    try:
        payload = _load_frontend_pack(stage_pack)
    except Exception:  # pragma: no cover - defensive
        return None
    pack_mapping = _unwrap_pack_payload(payload)
    if isinstance(pack_mapping, Mapping):
        primary_issue = pack_mapping.get("primary_issue")
        if not primary_issue:
            display = pack_mapping.get("display")
            if isinstance(display, Mapping):
                primary_issue = display.get("primary_issue")
        return _normalize_primary_issue(primary_issue)
    return None


def _normalize_review_listing_path(run_dir: Path, value: str) -> str | None:
    try:
        candidate = _safe_relative_path(run_dir, value)
    except ValueError:
        return None

    try:
        rel = candidate.relative_to(run_dir)
        return _to_posix_path(rel.as_posix())
    except ValueError:
        return _to_posix_path(candidate.as_posix())


def _collect_review_pack_listing(
    run_dir: Path, payload: Mapping[str, Any]
) -> list[dict[str, str]]:
    listing: list[dict[str, str]] = []
    stage_config = load_frontend_stage_config(run_dir)

    packs_dir_hint = payload.get("packs_dir") if isinstance(payload, Mapping) else None
    packs_dir_str: str | None = packs_dir_hint if isinstance(packs_dir_hint, str) else None

    for entry in _iter_frontend_pack_entries(payload):
        account_id = entry.get("account_id")
        if not isinstance(account_id, str):
            continue

        candidates: list[str] = []
        for key in ("file", "path"):
            value = entry.get(key)
            if isinstance(value, str):
                candidates.append(value)

        filename = entry.get("filename")
        if isinstance(filename, str):
            candidates.append(str(stage_config.packs_dir / filename))
            candidates.append(filename)
            dir_hint = entry.get("dir")
            if isinstance(dir_hint, str):
                candidates.append(str(Path(dir_hint) / filename))
            if packs_dir_str:
                candidates.append(str(Path(packs_dir_str) / filename))

        normalized: str | None = None
        for candidate in candidates:
            normalized = _normalize_review_listing_path(run_dir, candidate)
            if normalized:
                break

        if not normalized:
            fallback = stage_config.packs_dir / f"{account_id}.json"
            normalized = _normalize_review_listing_path(run_dir, str(fallback))

        if normalized:
            listing.append({"account_id": account_id, "file": normalized})

    return listing


@api_bp.route("/api/runs/<sid>/frontend/manifest", methods=["GET"])
def api_frontend_manifest(sid: str):
    try:
        run_dir = _run_dir_for_sid(sid)
    except ValueError:
        return jsonify({"error": "invalid_sid"}), 400

    if not run_dir.exists():
        return jsonify({"error": "run_not_found"}), 404

    try:
        manifest = _load_run_manifest(run_dir)
    except Exception:  # pragma: no cover - defensive logging
        return jsonify({"error": "manifest_read_failed"}), 500

    if manifest is None:
        return jsonify({"error": "manifest_not_found"}), 404

    section = (request.args.get("section") or "").strip().lower()
    if section == "frontend":
        frontend_payload = manifest.get("frontend")
        normalized_frontend: Mapping[str, Any] | None = None

        if isinstance(frontend_payload, Mapping):
            normalized_frontend = dict(frontend_payload)
            review_payload = frontend_payload.get("review")
            if isinstance(review_payload, Mapping):
                review_section = dict(review_payload)
            else:
                review_section = dict(frontend_payload)
                review_section.pop("review", None)

            if not isinstance(review_section.get("responses_dir"), str):
                results_dir = review_section.get("results_dir")
                if isinstance(results_dir, str):
                    review_section["responses_dir"] = results_dir

            if not isinstance(review_section.get("packs_dir"), str):
                packs_dir_hint = review_section.get("packs")
                if isinstance(packs_dir_hint, str):
                    review_section["packs_dir"] = packs_dir_hint

            normalized_frontend["review"] = review_section

        subset: dict[str, Any] = {
            "sid": manifest.get("sid"),
            "frontend": normalized_frontend if normalized_frontend is not None else frontend_payload,
        }
        normalized_subset = _normalize_path_like_entries(subset)
        return jsonify(normalized_subset)

    return jsonify(manifest)


@api_bp.route("/api/runs/<sid>/frontend/index", methods=["GET"])
def api_frontend_index(sid: str):
    try:
        run_dir = _run_dir_for_sid(sid)
    except ValueError:
        return jsonify({"error": "invalid_sid"}), 400

    if not run_dir.exists():
        return jsonify({"error": "run_not_found"}), 404

    manifest = _load_frontend_stage_manifest(run_dir)
    if manifest is None:
        has_cases = _run_has_cases(run_dir)
        queued = request_frontend_review_build(sid, run_dir=run_dir) if has_cases else False
        baseline = _normalize_frontend_review_index_payload(
            run_dir,
            {"sid": sid, "packs": [], "counts": {"packs": 0, "responses": 0}},
            sid=sid,
        )
        baseline.setdefault("sid", sid)
        baseline.setdefault("packs", [])
        baseline.setdefault("packs_index", [])
        baseline.setdefault("counts", {"packs": 0, "responses": 0})
        response_payload = {
            "status": "building",
            "queued": bool(queued),
            "has_cases": has_cases,
            "frontend": {"review": baseline},
        }
        response = jsonify(response_payload)
        response.status_code = 202
        response.headers["X-Index-Shape"] = "building"
        response.headers["Retry-After"] = "2"
        return response

    _, payload = manifest
    normalized = _normalize_frontend_review_index_payload(run_dir, payload, sid=sid)
    response_payload = {"frontend": {"review": normalized}}
    response = jsonify(response_payload)
    response.headers["X-Index-Shape"] = "nested"
    return response


@api_bp.route("/api/runs/<sid>/frontend/review/index", methods=["GET"])
def api_frontend_review_index(sid: str):
    return api_frontend_index(sid)


def _coerce_non_negative_int(value: Any) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return 0
    return max(number, 0)


def _extract_packs_count(payload: Mapping[str, Any] | None) -> int:
    if not isinstance(payload, Mapping):
        return 0
    value = payload.get("packs_count", 0)
    count = _coerce_non_negative_int(value)
    if count:
        return count
    counts_payload = payload.get("counts")
    if isinstance(counts_payload, Mapping):
        return _coerce_non_negative_int(counts_payload.get("packs"))
    return 0


def _extract_responses_count(payload: Mapping[str, Any] | None) -> int:
    if not isinstance(payload, Mapping):
        return 0

    counts_payload = payload.get("counts")
    if isinstance(counts_payload, Mapping):
        count = _coerce_non_negative_int(counts_payload.get("responses"))
        if count:
            return count

    return _coerce_non_negative_int(payload.get("responses_count"))


@api_bp.route("/api/runs/<sid>/frontend/review/stream", methods=["GET"])
def api_frontend_review_stream(sid: str):
    try:
        run_dir = _run_dir_for_sid(sid)
    except ValueError:
        return jsonify({"error": "invalid_sid"}), 400

    if not run_dir.exists():
        return jsonify({"error": "run_not_found"}), 404

    keepalive_interval = _REVIEW_STREAM_KEEPALIVE_INTERVAL
    queue_wait = _REVIEW_STREAM_QUEUE_WAIT_SECONDS

    def _generate():
        subscription = _review_stream_broker.subscribe(sid)
        keepalive_deadline = time.monotonic() + keepalive_interval
        packs_ready_sent = False

        def _resolve_packs_ready_event() -> bytes | None:
            manifest = _load_frontend_stage_manifest(run_dir)
            if manifest is None:
                return None

            _, payload = manifest
            normalized = _normalize_frontend_review_index_payload(run_dir, payload, sid=sid)
            packs_count = _extract_packs_count(normalized)
            if packs_count > 0:
                return _format_sse("packs_ready", {"packs_count": packs_count})
            return None

        try:
            initial_event = _resolve_packs_ready_event()
            if initial_event is not None:
                packs_ready_sent = True
                yield initial_event
                keepalive_deadline = time.monotonic() + keepalive_interval

            while True:
                if not packs_ready_sent:
                    ready_event = _resolve_packs_ready_event()
                    if ready_event is not None:
                        packs_ready_sent = True
                        yield ready_event
                        keepalive_deadline = time.monotonic() + keepalive_interval

                timeout = min(queue_wait, keepalive_interval)
                try:
                    message = subscription.get(timeout=timeout)
                except queue.Empty:
                    message = None

                if message is not None:
                    event = message.get("event") if isinstance(message, Mapping) else None
                    data = message.get("data") if isinstance(message, Mapping) else None
                    if event:
                        yield _format_sse(event, data)
                        if event == "packs_ready":
                            packs_ready_sent = True
                        keepalive_deadline = time.monotonic() + keepalive_interval

                now = time.monotonic()
                if now >= keepalive_deadline:
                    yield b": keepalive\n\n"
                    keepalive_deadline = now + keepalive_interval
        finally:
            _review_stream_broker.unsubscribe(sid, subscription)

    headers = {
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",
    }

    return Response(
        stream_with_context(_generate()),
        mimetype="text/event-stream",
        headers=headers,
    )


@api_bp.route("/api/runs/<sid>/frontend/review/packs", methods=["GET"])
def api_frontend_review_packs(sid: str):
    try:
        run_dir = _run_dir_for_sid(sid)
    except ValueError:
        return jsonify({"error": "invalid_sid"}), 400

    manifest = _load_frontend_stage_manifest(run_dir)
    if manifest is None:
        has_cases = _run_has_cases(run_dir)
        queued = (
            request_frontend_review_build(sid, run_dir=run_dir)
            if has_cases
            else False
        )
        payload = {
            "status": "building",
            "queued": bool(queued),
            "has_cases": has_cases,
            "items": [],
        }
        response = jsonify(payload)
        response.status_code = 202
        response.headers["X-Packs-Shape"] = "building"
        response.headers["Retry-After"] = "2"
        return response

    _, payload = manifest
    normalized = _normalize_frontend_review_index_payload(run_dir, payload, sid=sid)
    response_payload = {"items": normalized.get("items", [])}
    response = jsonify(response_payload)
    response.headers["X-Packs-Shape"] = "listing"
    return response


def _normalize_frontend_review_index_payload(
    run_dir: Path, payload: Any, *, sid: str | None = None
) -> dict[str, Any]:
    result: dict[str, Any]
    if isinstance(payload, Mapping):
        result = dict(payload)
    else:
        result = {}

    try:
        stage_config = load_frontend_stage_config(run_dir)
    except Exception:  # pragma: no cover - defensive fallback
        stage_config = None

    stage_prefix = None
    if stage_config is not None:
        stage_name_component = stage_config.stage_dir.name or "review"
        stage_prefix = f"frontend/{stage_name_component.strip('/')}/"

    def _stage_relative_path(value: str | None) -> str:
        normalized = (value or "").replace("\\", "/")
        if not normalized:
            return normalized
        if stage_config is None:
            return normalized.lstrip("/") or normalized
        try:
            candidate = _safe_relative_path(run_dir, normalized)
        except ValueError:
            candidate = Path(normalized)
        try:
            relative = candidate.relative_to(stage_config.stage_dir)
        except ValueError:
            try:
                relative = candidate.relative_to(stage_config.stage_dir.parent)
            except ValueError:
                return candidate.as_posix()
        return relative.as_posix()

    items = _collect_review_pack_listing(run_dir, result) if isinstance(payload, Mapping) else []
    result["items"] = items

    packs_count = _extract_packs_count(result)
    if packs_count <= 0 and items:
        packs_count = len(items)

    result["packs_count"] = packs_count

    packs_index_payload = result.get("packs_index")
    normalized_index: list[dict[str, Any]] = []
    if isinstance(packs_index_payload, list):
        for entry in packs_index_payload:
            if not isinstance(entry, Mapping):
                continue
            account_value = entry.get("account") or entry.get("account_id")
            if not isinstance(account_value, str) or not account_value:
                continue
            file_value = entry.get("file") or entry.get("path")
            file_str = file_value if isinstance(file_value, str) else None
            if file_str and stage_prefix and not file_str.replace("\\", "/").startswith("/"):
                normalized_candidate = file_str.replace("\\", "/")
                if not normalized_candidate.startswith("frontend/"):
                    file_str = f"{stage_prefix}{normalized_candidate.lstrip('/')}"
            stage_relative = _stage_relative_path(file_str)
            if not stage_relative and stage_config is not None:
                fallback_path = stage_config.packs_dir / f"{account_value}.json"
                stage_relative = _stage_relative_path(fallback_path.as_posix())
            if not stage_relative:
                continue
            normalized_index.append({"account": account_value, "file": stage_relative})

    if not normalized_index and items:
        for entry in items:
            if not isinstance(entry, Mapping):
                continue
            account_id = entry.get("account_id")
            file_value = entry.get("file")
            if not isinstance(account_id, str) or not account_id:
                continue
            file_str = file_value if isinstance(file_value, str) else None
            if file_str and stage_prefix and not file_str.replace("\\", "/").startswith("/"):
                normalized_candidate = file_str.replace("\\", "/")
                if not normalized_candidate.startswith("frontend/"):
                    file_str = f"{stage_prefix}{normalized_candidate.lstrip('/')}"
            stage_relative = _stage_relative_path(file_str)
            if not stage_relative and stage_config is not None:
                fallback_path = stage_config.packs_dir / f"{account_id}.json"
                stage_relative = _stage_relative_path(fallback_path.as_posix())
            if not stage_relative:
                continue
            normalized_index.append({"account": account_id, "file": stage_relative})

    result["packs_index"] = normalized_index

    responses_count = _extract_responses_count(result)
    if responses_count <= 0:
        responses_payload = result.get("responses")
        if isinstance(responses_payload, list):
            responses_count = len(responses_payload)

    counts_payload = result.get("counts")
    counts: dict[str, Any]
    if isinstance(counts_payload, Mapping):
        counts = dict(counts_payload)
    else:
        counts = {}

    counts["packs"] = packs_count
    counts["responses"] = responses_count
    result["counts"] = counts

    attachments_overview: dict[str, Any] = {}

    if stage_config is not None:
        for item in items:
            account_id = item.get("account_id") if isinstance(item, Mapping) else None
            if not isinstance(account_id, str):
                continue
            if not _is_valid_frontend_account_id(account_id):
                continue
            summary_payload = _build_account_attachments_summary(
                stage_config.responses_dir,
                account_id,
                sid=sid or "",
            )
            attachments_overview[account_id] = {
                "count": len(summary_payload.get("attachments", ())),
                "hot_fields_union": list(summary_payload.get("hot_fields_union", ())),
            }

        frontend_payload = result.get("frontend") if isinstance(result.get("frontend"), Mapping) else None
        if frontend_payload is not None:
            frontend_map = dict(frontend_payload)
        else:
            frontend_map = {}
        review_payload = frontend_map.get("review") if isinstance(frontend_map.get("review"), Mapping) else None
        if review_payload is not None:
            review_map = dict(review_payload)
        else:
            review_map = {}
        review_map["attachments_summary"] = attachments_overview
        frontend_map["review"] = review_map
        result["frontend"] = frontend_map

    normalized_result = _normalize_path_like_entries(result)

    if normalized_result:
        return normalized_result

    fallback = {"packs_count": packs_count, "items": items}
    return _normalize_path_like_entries(fallback)


def _stage_pack_path_for_account(run_dir: Path, account_id: str) -> Path | None:
    if not _is_valid_frontend_account_id(account_id):
        return None

    stage_dir = _frontend_stage_packs_dir(run_dir)
    candidate = stage_dir / f"{account_id}.json"
    if candidate.is_file():
        return candidate

    manifest_info = _load_frontend_stage_manifest(run_dir)
    if manifest_info is None:
        return candidate if candidate.is_file() else None

    _, manifest_payload = manifest_info
    if not isinstance(manifest_payload, Mapping):
        return candidate if candidate.is_file() else None

    packs_dir_hint = manifest_payload.get("packs_dir")
    packs_dir_value = Path(packs_dir_hint) if isinstance(packs_dir_hint, str) else None

    for entry in _iter_frontend_pack_entries(manifest_payload):
        if entry.get("account_id") != account_id:
            continue

        path_candidates: list[str] = []
        for key in ("path", "file"):
            value = entry.get(key)
            if isinstance(value, str):
                path_candidates.append(value)

        filename_value = entry.get("filename")
        if isinstance(filename_value, str):
            path_candidates.append(filename_value)
            dir_hint = entry.get("dir")
            if isinstance(dir_hint, str):
                path_candidates.append(str(Path(dir_hint) / filename_value))
            if packs_dir_value is not None:
                path_candidates.append(str(packs_dir_value / filename_value))
            path_candidates.append(str(stage_dir / filename_value))

        for value in path_candidates:
            try:
                manifest_candidate = _safe_relative_path(run_dir, value)
            except ValueError:
                continue
            if manifest_candidate.is_file():
                return manifest_candidate

    return candidate if candidate.is_file() else None


@api_bp.route("/api/runs/<sid>/frontend/review/accounts/<account_id>", methods=["GET"])
@api_bp.route("/api/runs/<sid>/frontend/review/pack/<account_id>", methods=["GET"])
def api_frontend_review_pack(sid: str, account_id: str):
    try:
        run_dir = _run_dir_for_sid(sid)
    except ValueError:
        return jsonify({"error": "invalid_sid"}), 400

    stage_pack = _stage_pack_path_for_account(run_dir, account_id)

    if stage_pack is None or not stage_pack.is_file():
        return jsonify({"error": "pack_not_found"}), 404

    try:
        payload = _load_frontend_pack(stage_pack)
    except Exception:  # pragma: no cover - error path
        return jsonify({"error": "pack_read_failed"}), 500

    pack_obj = _prepare_pack_response(payload, account_id)

    manifest_info = _load_frontend_stage_manifest(run_dir)
    if manifest_info is not None:
        _, manifest_payload = manifest_info
        questions = _extract_pack_questions(manifest_payload, account_id)
        if questions is not None and "questions" not in pack_obj:
            pack_obj["questions"] = questions

    return jsonify(pack_obj)


@api_bp.route("/api/review/claims-schema", methods=["GET"])
def api_review_claims_schema():
    return jsonify(_CLAIMS_SCHEMA_DATA)


@api_bp.route(
    "/api/runs/<sid>/frontend/review/accounts/<account_id>/answer",
    methods=["POST"],
)
@api_bp.route(
    "/api/runs/<sid>/frontend/review/response/<account_id>",
    methods=["POST"],
)
def api_frontend_review_answer(sid: str, account_id: str):
    try:
        run_dir = _run_dir_for_sid(sid)
    except ValueError:
        return jsonify({"error": "invalid_sid"}), 400

    if not run_dir.exists():
        return jsonify({"error": "run_not_found"}), 404

    data = request.get_json(force=True, silent=True) or {}
    if not isinstance(data, Mapping):
        return jsonify({"error": "invalid_payload"}), 400

    answers = data.get("answers")
    if not isinstance(answers, Mapping):
        return jsonify({"error": "invalid_answers"}), 400

    client_ts = data.get("client_ts")
    if client_ts is not None and not isinstance(client_ts, str):
        return jsonify({"error": "invalid_client_ts"}), 400

    client_meta = data.get("client_meta")
    if client_meta is not None and not isinstance(client_meta, Mapping):
        return jsonify({"error": "invalid_client_meta"}), 400

    stage_config = load_frontend_stage_config(run_dir)
    uploads_dir = stage_config.uploads_dir

    primary_issue = _resolve_pack_primary_issue(run_dir, account_id)
    base_docs, issue_claim_entries = resolve_issue_claims(primary_issue)
    claim_entry_map: dict[str, ClaimSchemaEntry] = {
        entry.key: entry for entry in issue_claim_entries
    }

    allowed_doc_keys = set(_ALL_DOC_KEYS)
    allowed_doc_keys.update(base_docs)
    for claim_entry in claim_entry_map.values():
        allowed_doc_keys.update(all_doc_keys_for_claim(claim_entry))
    allowed_doc_keys.update(DOC_KEY_ALIAS_TO_CANONICAL.keys())

    selected_claims_source = "answers"
    selected_claims_raw = answers.get("selected_claims")
    if selected_claims_raw is None and isinstance(data.get("claims"), list):
        selected_claims_raw = data.get("claims")
        selected_claims_source = "legacy"

    selected_claims: list[str] = []
    if selected_claims_raw is not None:
        if not isinstance(selected_claims_raw, list):
            return jsonify({"error": "invalid_claims"}), 400
        seen: set[str] = set()
        for value in selected_claims_raw:
            if not isinstance(value, str):
                return jsonify({"error": "invalid_claims"}), 400
            key = value.strip()
            if not key:
                continue
            entry = claim_entry_map.get(key)
            if entry is None:
                entry = get_claim_entry(key)
            if entry is None:
                return jsonify({"error": "invalid_claims"}), 400
            if key in seen:
                continue
            seen.add(key)
            selected_claims.append(key)

    attachments_raw = answers.get("attachments")
    attachments_display_map: dict[str, list[str]] = {}
    attachments_map: dict[str, list[str]] = {}
    if attachments_raw is not None:
        if not isinstance(attachments_raw, Mapping):
            return jsonify({"error": "invalid_attachments"}), 400
        for doc_key_raw, value in attachments_raw.items():
            if not isinstance(doc_key_raw, str):
                return jsonify({"error": "invalid_attachments"}), 400
            doc_key = doc_key_raw.strip()
            if not doc_key or not _REVIEW_DOC_KEY_PATTERN.fullmatch(doc_key):
                return jsonify({"error": "invalid_attachments"}), 400
            canonical_doc_key = DOC_KEY_ALIAS_TO_CANONICAL.get(doc_key, doc_key)
            if canonical_doc_key not in allowed_doc_keys:
                return jsonify({"error": "invalid_attachments"}), 400

            candidates: list[Any]
            if isinstance(value, list):
                candidates = value
            else:
                candidates = [value]

            validated_ids: list[str] = []
            for candidate in candidates:
                if not isinstance(candidate, str):
                    return jsonify({"error": "invalid_attachment_doc"}), 400
                doc_id = candidate.strip()
                if not doc_id:
                    continue
                if _resolve_review_doc_id(doc_id, run_dir, uploads_dir) is None:
                    return jsonify({"error": "invalid_attachment_doc"}), 400
                if doc_id not in validated_ids:
                    validated_ids.append(doc_id)

            if validated_ids:
                _merge_attachment_doc_ids(attachments_map, doc_key, validated_ids)
                display_ids = attachments_display_map.setdefault(doc_key, [])
                for doc_id in validated_ids:
                    if doc_id not in display_ids:
                        display_ids.append(doc_id)

    evidence_entries = data.get("evidence")
    validated_evidence: list[dict[str, Any]] = []
    if isinstance(evidence_entries, list):
        for evidence in evidence_entries:
            if not isinstance(evidence, Mapping):
                continue
            docs_payload = evidence.get("docs")
            if not isinstance(docs_payload, list):
                continue
            claim_value = evidence.get("claim")
            sanitized_evidence_docs: list[dict[str, Any]] = []
            for doc_entry in docs_payload:
                if not isinstance(doc_entry, Mapping):
                    continue
                doc_key_value = doc_entry.get("doc_key")
                doc_ids_value = doc_entry.get("doc_ids")
                if not isinstance(doc_key_value, str):
                    return jsonify({"error": "invalid_evidence_doc"}), 400
                doc_key = doc_key_value.strip()
                if not doc_key or not _REVIEW_DOC_KEY_PATTERN.fullmatch(doc_key):
                    return jsonify({"error": "invalid_evidence_doc"}), 400
                canonical_doc_key = DOC_KEY_ALIAS_TO_CANONICAL.get(doc_key, doc_key)
                if canonical_doc_key not in allowed_doc_keys:
                    return jsonify({"error": "invalid_evidence_doc"}), 400

                candidates: list[Any]
                if isinstance(doc_ids_value, list):
                    candidates = doc_ids_value
                else:
                    candidates = [doc_ids_value]

                validated_ids: list[str] = []
                for candidate in candidates:
                    if not isinstance(candidate, str):
                        return jsonify({"error": "invalid_evidence_doc"}), 400
                    doc_id = candidate.strip()
                    if not doc_id:
                        continue
                    if _resolve_review_doc_id(doc_id, run_dir, uploads_dir) is None:
                        return jsonify({"error": "invalid_evidence_doc"}), 400
                    if doc_id not in validated_ids:
                        validated_ids.append(doc_id)

                if validated_ids:
                    _merge_attachment_doc_ids(attachments_map, doc_key, validated_ids)
                    sanitized_evidence_docs.append(
                        {
                            "doc_key": doc_key,
                            "doc_ids": list(validated_ids),
                        }
                    )
            if sanitized_evidence_docs:
                entry_payload: dict[str, Any] = {"docs": sanitized_evidence_docs}
                if isinstance(claim_value, str):
                    entry_payload["claim"] = claim_value
                validated_evidence.append(entry_payload)
    elif evidence_entries is not None:
        return jsonify({"error": "invalid_evidence"}), 400

    missing_docs: dict[str, list[str]] = {}
    min_upload_failures: dict[str, int] = {}
    enforce_doc_requirements = selected_claims and selected_claims_source == "answers"
    if enforce_doc_requirements:
        for claim_key in selected_claims:
            claim_entry = claim_entry_map.get(claim_key) or get_claim_entry(claim_key)
            if claim_entry is None:
                continue
            required_docs = list(claim_entry.requires)
            missing_for_claim = [doc for doc in required_docs if not attachments_map.get(doc)]
            if missing_for_claim:
                missing_docs[claim_key] = missing_for_claim

            if claim_entry.min_uploads is not None and claim_entry.min_uploads > 0:
                present_count = sum(
                    1
                    for doc_key in all_doc_keys_for_claim(claim_entry)
                    if attachments_map.get(doc_key)
                )
                if present_count < claim_entry.min_uploads:
                    min_upload_failures[claim_key] = claim_entry.min_uploads

    if enforce_doc_requirements and (missing_docs or min_upload_failures):
        logger.info(
            "FRONTEND_REVIEW_MISSING_DOCS sid=%s account_id=%s claims=%s missing=%s min_uploads=%s",
            sid,
            account_id,
            selected_claims,
            missing_docs,
            min_upload_failures,
        )
        return (
            jsonify(
                {
                    "error": "missing_required_docs",
                    "details": {
                        "missing": missing_docs,
                        "min_uploads": min_upload_failures,
                    },
                }
            ),
            400,
        )

    sanitized_answers = dict(answers)

    explanation_value = sanitized_answers.get("explanation")
    if isinstance(explanation_value, str):
        trimmed = explanation_value.strip()
        sanitized_answers["explanation"] = trimmed if trimmed else ""

    if selected_claims:
        logger.info(
            "FRONTEND_REVIEW_SELECTED_CLAIMS sid=%s account_id=%s claims=%s attachments=%s",
            sid,
            account_id,
            selected_claims,
            sorted(attachments_map.keys()),
        )

    if selected_claims:
        sanitized_answers["selected_claims"] = selected_claims
    elif "selected_claims" in sanitized_answers:
        sanitized_answers.pop("selected_claims", None)

    if attachments_display_map:
        formatted_attachments: dict[str, Any] = {}
        for doc_key, doc_ids in attachments_display_map.items():
            if not doc_ids:
                continue
            if len(doc_ids) == 1:
                formatted_attachments[doc_key] = doc_ids[0]
            else:
                formatted_attachments[doc_key] = doc_ids
        if formatted_attachments:
            sanitized_answers["attachments"] = formatted_attachments
        else:
            sanitized_answers.pop("attachments", None)
    elif "attachments" in sanitized_answers:
        sanitized_answers.pop("attachments", None)

    record: dict[str, Any] = {
        "sid": sid,
        "account_id": account_id,
        "answers": sanitized_answers,
        "received_at": _now_utc_iso(),
    }
    if client_ts is not None:
        record["client_ts"] = client_ts
    if isinstance(client_meta, Mapping):
        record["client_meta"] = dict(client_meta)
    if selected_claims:
        record["claims"] = selected_claims
    if validated_evidence:
        record["evidence"] = validated_evidence

    responses_dir = stage_config.responses_dir
    responses_dir.mkdir(parents=True, exist_ok=True)
    filename = _response_filename_for_account(account_id)
    resp_path = responses_dir / filename
    with resp_path.open("w", encoding="utf-8") as handle:
        json.dump(record, handle, ensure_ascii=False, indent=2)

    write_client_response(sid, account_id, record)

    _review_stream_broker.publish(sid, "responses_written", {"account_id": account_id})

    barriers_payload: Mapping[str, Any] | None = None
    try:
        barriers_payload = runflow_barriers_refresh(sid)
    except Exception:  # pragma: no cover - defensive logging
        logger.warning(
            "FRONTEND_BARRIERS_REFRESH_FAILED sid=%s account_id=%s",
            sid,
            account_id,
            exc_info=True,
        )

    try:
        refresh_frontend_stage_from_responses(sid, runs_root=run_dir.parent)
    except Exception:  # pragma: no cover - defensive logging
        logger.warning(
            "FRONTEND_STAGE_REFRESH_FAILED sid=%s account_id=%s",
            sid,
            account_id,
            exc_info=True,
        )
    else:
        try:
            refreshed_barriers = runflow_barriers_refresh(sid)
        except Exception:  # pragma: no cover - defensive logging
            logger.warning(
                "FRONTEND_BARRIERS_REFRESH_AFTER_STAGE_FAILED sid=%s account_id=%s",
                sid,
                account_id,
                exc_info=True,
            )
        else:
            if isinstance(refreshed_barriers, Mapping):
                barriers_payload = refreshed_barriers

    build_result: Mapping[str, Any] | None = None
    send_on_write_enabled = _env_flag_enabled(
        "NOTE_STYLE_SEND_ON_RESPONSE_WRITE",
        default=config.NOTE_STYLE_SEND_ON_RESPONSE_WRITE,
    )

    if config.NOTE_STYLE_ENABLED:
        try:
            build_result = build_note_style_pack_for_account(
                sid, account_id, runs_root=run_dir.parent
            )
        except Exception:  # pragma: no cover - defensive logging
            logger.warning(
                "NOTE_STYLE_BUILD_FAILED sid=%s account_id=%s",
                sid,
                account_id,
                exc_info=True,
            )

        if (
            send_on_write_enabled
            and isinstance(build_result, Mapping)
            and _note_style_ready_from_barriers(barriers_payload)
        ):
            status_text = str(build_result.get("status") or "").strip().lower()
            if status_text == "completed":
                _schedule_note_style_send_after_delay(
                    sid,
                    account_id,
                    runs_root=run_dir.parent,
                    delay_ms=max(_NOTE_STYLE_SEND_DELAY_MS, 300),
                )

    try:
        reconcile_umbrella_barriers(sid, runs_root=run_dir.parent)
    except Exception:  # pragma: no cover - defensive logging
        logger.warning(
            "FRONTEND_BARRIERS_RECONCILE_FAILED sid=%s account_id=%s",
            sid,
            account_id,
            exc_info=True,
        )

    return jsonify(record)


@api_bp.route(
    "/api/runs/<sid>/frontend/review/attachments/<account_id>",
    methods=["GET"],
)
def api_frontend_review_attachments_summary(sid: str, account_id: str):
    try:
        run_dir = _run_dir_for_sid(sid)
    except ValueError:
        return jsonify({"error": "invalid_sid"}), 400

    if not run_dir.exists():
        return jsonify({"error": "run_not_found"}), 404

    if not _is_valid_frontend_account_id(account_id):
        return jsonify({"error": "invalid_account_id"}), 400

    stage_config = load_frontend_stage_config(run_dir)
    summary = _build_account_attachments_summary(
        stage_config.responses_dir,
        account_id,
        sid=sid,
    )

    return jsonify(summary)


@api_bp.route(
    "/api/runs/<sid>/frontend/review/attachments/<account_id>",
    methods=["POST"],
)
def api_frontend_review_attachment_upload(sid: str, account_id: str):
    try:
        run_dir = _run_dir_for_sid(sid)
    except ValueError:
        return jsonify({"error": "invalid_sid"}), 400

    if not run_dir.exists():
        return jsonify({"error": "run_not_found"}), 404

    if not _is_valid_frontend_account_id(account_id):
        return jsonify({"error": "invalid_account_id"}), 400

    claim_type = (request.form.get("claim_type") or "").strip()
    if not claim_type or claim_type not in CLAIM_FIELD_LINK_MAP:
        return jsonify({"error": "invalid_claim_type"}), 400

    uploaded = request.files.get("file")
    if uploaded is None:
        return jsonify({"error": "missing_file"}), 400

    original_name = uploaded.filename or "document"
    extension = _extract_upload_extension(original_name)
    if extension not in _REVIEW_ALLOWED_UPLOAD_EXTENSIONS:
        return jsonify({"error": "invalid_file_type"}), 400

    content_length = getattr(uploaded, "content_length", None)
    if isinstance(content_length, int) and content_length > _REVIEW_UPLOAD_MAX_BYTES:
        return jsonify({"error": "file_too_large"}), 400

    if extension == "pdf":
        head = uploaded.stream.read(4)
        uploaded.stream.seek(0)
        if head != b"%PDF":
            return jsonify({"error": "invalid_pdf"}), 400

    stage_config = load_frontend_stage_config(run_dir)
    uploads_dir = stage_config.uploads_dir
    target_dir = uploads_dir / account_id
    target_dir.mkdir(parents=True, exist_ok=True)

    timestamp_ms = int(time.time() * 1000)
    slug = _slugify_upload_name(original_name)
    stored_name = f"{timestamp_ms}_{slug}"
    if extension:
        stored_name = f"{stored_name}.{extension}"
    target_path = target_dir / stored_name
    counter = 1
    while target_path.exists():
        timestamp_ms += counter
        stored_name = f"{timestamp_ms}_{slug}"
        if extension:
            stored_name = f"{stored_name}.{extension}"
        target_path = target_dir / stored_name
        counter += 1

    try:
        uploaded.save(target_path)
    except Exception:
        log.exception(
            "failed to save frontend attachment upload",
            extra={"sid": sid, "account_id": account_id, "claim_type": claim_type},
        )
        return jsonify({"error": "upload_failed"}), 500

    try:
        size_value = target_path.stat().st_size
    except OSError:
        size_value = None

    if isinstance(size_value, int) and size_value > _REVIEW_UPLOAD_MAX_BYTES:
        try:
            target_path.unlink()
        except OSError:  # pragma: no cover - best effort cleanup
            pass
        return jsonify({"error": "file_too_large"}), 400

    mimetype_hint = getattr(uploaded, "mimetype", None)
    mime_type = mimetype_hint if isinstance(mimetype_hint, str) and mimetype_hint else None
    if not mime_type:
        guess = mimetypes.guess_type(original_name)[0]
        mime_type = guess or "application/octet-stream"

    sha1 = hashlib.sha1()
    try:
        with target_path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(65536), b""):
                sha1.update(chunk)
    except OSError:
        log.warning(
            "FRONTEND_REVIEW_ATTACHMENT_SHA1_FAILED sid=%s account_id=%s path=%s",
            sid,
            account_id,
            target_path,
            exc_info=True,
        )
    sha1_hex = sha1.hexdigest()

    uploaded_at = (
        datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    )

    stored_path = _doc_id_for_path(target_path)

    summary_payload, bureaus_payload = _load_account_validation_context(run_dir, account_id)
    hot_fields = list(CLAIM_FIELD_LINK_MAP.get(claim_type, ()))
    field_snapshots_raw = _build_field_snapshots(
        summary_payload,
        bureaus_payload,
        hot_fields,
    )
    field_snapshots = _sanitize_field_snapshots(field_snapshots_raw)

    attachment_record: dict[str, Any] = {
        "id": _generate_attachment_id(),
        "claim_type": claim_type,
        "filename": original_name,
        "stored_path": stored_path,
        "mime": mime_type,
        "size": int(size_value) if isinstance(size_value, int) else size_value,
        "sha1": sha1_hex,
        "uploaded_at": uploaded_at,
        "hot_fields": list(hot_fields),
        "field_snapshots": field_snapshots,
        "doc_id": stored_path,
    }

    sanitized_record = _sanitize_attachment_record(attachment_record)

    _append_attachment_summary(
        stage_config.responses_dir,
        sid=sid,
        account_id=account_id,
        attachments=[sanitized_record],
    )

    return jsonify(sanitized_record), 201


@api_bp.route("/api/runs/<sid>/frontend/review/uploads", methods=["POST"])
def api_frontend_review_upload(sid: str):
    try:
        run_dir = _run_dir_for_sid(sid)
    except ValueError:
        return jsonify({"error": "invalid_sid"}), 400

    if not run_dir.exists():
        return jsonify({"error": "run_not_found"}), 404

    form = request.form or {}
    account_id_raw = form.get("account_id") or ""
    claim_key = (form.get("claim") or "").strip()
    doc_key_raw = form.get("doc_key") or ""
    sid_from_form = (form.get("sid") or "").strip()

    account_id = account_id_raw.strip()
    if not _is_valid_frontend_account_id(account_id):
        return jsonify({"error": "invalid_account_id"}), 400

    if sid_from_form and sid_from_form != sid:
        return jsonify({"error": "invalid_sid"}), 400

    if not claim_key:
        return jsonify({"error": "invalid_claim"}), 400

    claim_entry = get_claim_entry(claim_key)
    if claim_entry is None:
        return jsonify({"error": "invalid_claim"}), 400

    doc_key = doc_key_raw.strip()
    if not doc_key or not _REVIEW_DOC_KEY_PATTERN.fullmatch(doc_key):
        return jsonify({"error": "invalid_doc_key"}), 400

    allowed_doc_keys = all_doc_keys_for_claim(claim_entry)
    allowed_doc_keys.update(_AUTO_ATTACH_BASE)
    allowed_doc_keys.update(DOC_KEY_ALIAS_TO_CANONICAL.keys())
    canonical_doc_key = DOC_KEY_ALIAS_TO_CANONICAL.get(doc_key, doc_key)
    if canonical_doc_key not in allowed_doc_keys:
        return jsonify({"error": "unsupported_doc_key"}), 400

    uploaded_files = request.files.getlist("files")
    if not uploaded_files:
        return jsonify({"error": "missing_files"}), 400

    stage_config = load_frontend_stage_config(run_dir)
    uploads_dir = stage_config.uploads_dir

    target_dir = uploads_dir / account_id
    target_dir.mkdir(parents=True, exist_ok=True)

    summary_payload, bureaus_payload = _load_account_validation_context(
        run_dir, account_id
    )
    hot_fields = list(CLAIM_FIELD_LINK_MAP.get(claim_key, []))
    field_snapshots_raw = _build_field_snapshots(
        summary_payload, bureaus_payload, hot_fields
    )
    field_snapshots = _sanitize_field_snapshots(field_snapshots_raw)

    doc_ids: list[str] = []
    attachments_to_append: list[dict[str, Any]] = []
    last_timestamp_ms = 0
    for uploaded in uploaded_files:
        if uploaded is None:
            return jsonify({"error": "invalid_file"}), 400
        original_name = uploaded.filename or "document"
        extension = _extract_upload_extension(original_name)
        if extension not in _REVIEW_ALLOWED_UPLOAD_EXTENSIONS:
            return jsonify({"error": "invalid_file_type"}), 400

        content_length = getattr(uploaded, "content_length", None)
        if (
            isinstance(content_length, int)
            and content_length > _REVIEW_UPLOAD_MAX_BYTES
        ):
            return jsonify({"error": "file_too_large"}), 400

        if extension == "pdf":
            head = uploaded.stream.read(4)
            uploaded.stream.seek(0)
            if head != b"%PDF":
                return jsonify({"error": "invalid_pdf"}), 400

        timestamp_ms = int(time.time() * 1000)
        if timestamp_ms <= last_timestamp_ms:
            timestamp_ms = last_timestamp_ms + 1
        last_timestamp_ms = timestamp_ms

        slug = _slugify_upload_name(original_name)
        stored_name = f"{timestamp_ms}_{slug}"
        if extension:
            stored_name = f"{stored_name}.{extension}"
        target_path = target_dir / stored_name

        try:
            uploaded.save(target_path)
        except Exception:  # pragma: no cover - filesystem errors
            log.exception(
                "failed to save frontend review upload",
                extra={
                    "sid": sid,
                    "account_id": account_id,
                    "claim": claim_key,
                    "doc_key": doc_key,
                    "target": str(target_path),
                },
            )
            return jsonify({"error": "upload_failed"}), 500

        try:
            size = target_path.stat().st_size
        except OSError:
            size = None

        if size is not None and size > _REVIEW_UPLOAD_MAX_BYTES:
            try:
                target_path.unlink()
            except OSError:  # pragma: no cover - best effort cleanup
                pass
            return jsonify({"error": "file_too_large"}), 400

        doc_id = _doc_id_for_path(target_path)
        doc_ids.append(doc_id)

        try:
            stored_relative = target_path.relative_to(run_dir).as_posix()
        except ValueError:
            stored_relative = target_path.as_posix()

        mimetype_hint = getattr(uploaded, "mimetype", None)
        mime_type = mimetype_hint if isinstance(mimetype_hint, str) and mimetype_hint else None
        if not mime_type:
            guess = mimetypes.guess_type(original_name)[0]
            mime_type = guess or "application/octet-stream"

        sha1 = hashlib.sha1()
        try:
            with target_path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(65536), b""):
                    sha1.update(chunk)
        except OSError:
            log.warning(
                "FRONTEND_REVIEW_UPLOAD_SHA1_FAILED sid=%s account_id=%s path=%s",
                sid,
                account_id,
                target_path,
                exc_info=True,
            )
        sha1_hex = sha1.hexdigest()

        uploaded_at = (
            datetime.fromtimestamp(timestamp_ms / 1000, tz=timezone.utc)
            .isoformat(timespec="seconds")
            .replace("+00:00", "Z")
        )

        attachment_record = {
            "id": _generate_attachment_id(),
            "claim_type": claim_key,
            "filename": original_name,
            "stored_path": stored_relative,
            "mime": mime_type,
            "size": int(size) if isinstance(size, int) else size,
            "sha1": sha1_hex,
            "uploaded_at": uploaded_at,
            "hot_fields": list(hot_fields),
            "field_snapshots": _clone_field_snapshots_payload(field_snapshots),
        }
        attachments_to_append.append(attachment_record)

    if attachments_to_append:
        _append_attachment_summary(
            stage_config.responses_dir,
            sid=sid,
            account_id=account_id,
            attachments=attachments_to_append,
        )

    return jsonify({"doc_ids": doc_ids})


@api_bp.route(
    "/api/runs/<sid>/frontend/review/complete",
    methods=["POST"],
)
def api_frontend_review_complete(sid: str):
    try:
        run_dir = _run_dir_for_sid(sid)
    except ValueError:
        return jsonify({"error": "invalid_sid"}), 400

    if not run_dir.exists():
        return jsonify({"error": "run_not_found"}), 404

    completed_at = _now_utc_iso().replace("+00:00", "Z")
    payload = {"ok": True, "sid": sid, "completed_at": completed_at}

    marker_path = _frontend_stage_dir(run_dir) / "completed.json"
    marker_payload = {"sid": sid, "completed_at": completed_at}
    try:
        marker_path.parent.mkdir(parents=True, exist_ok=True)
        marker_path.write_text(
            json.dumps(marker_payload, ensure_ascii=False),
            encoding="utf-8",
        )
    except OSError:
        log.exception(
            "failed to write frontend review completion marker",
            extra={"sid": sid, "marker_path": str(marker_path)},
        )

    return jsonify(payload), 200


@api_bp.route("/api/start-process", methods=["POST"])
def start_process():
    try:
        print("Received request to /api/start-process")

        form = request.form or {}
        payload = request.get_json(silent=True) or {}
        session_id_raw = (form.get("session_id") or payload.get("session_id") or "").strip()
        email = (form.get("email") or payload.get("email") or "").strip()

        if not session_id_raw:
            return jsonify({"status": "error", "message": "Missing session_id"}), 400

        try:
            session_id = _validate_sid(session_id_raw)
        except ValueError:
            return (
                jsonify({"status": "error", "message": "Invalid session id"}),
                400,
            )

        try:
            manifest = RunManifest.for_sid(session_id, allow_create=False)
        except FileNotFoundError:
            logger.info("START_PROCESS sid=%s manifest_exists=0", session_id)
            return (
                jsonify({"status": "error", "message": "unknown sid"}),
                400,
            )
        logger.info("START_PROCESS sid=%s manifest_exists=1", session_id)

        pdf_path = manifest.inputs.report_pdf
        if not pdf_path:
            return (
                jsonify(
                    {
                        "status": "error",
                        "message": "Manifest is missing inputs.report_pdf",
                    }
                ),
                422,
            )

        pdf_path = Path(pdf_path).resolve()
        try:
            size = pdf_path.stat().st_size
        except FileNotFoundError:
            return (
                jsonify(
                    {
                        "status": "error",
                        "message": f"Report not found at {pdf_path}",
                    }
                ),
                422,
            )

        if size <= 0:
            return (
                jsonify(
                    {
                        "status": "error",
                        "message": "Report PDF is empty",
                    }
                ),
                422,
            )

        original_name = None
        session_snapshot = get_session(session_id) or {}
        if isinstance(session_snapshot, dict):
            original_name = session_snapshot.get("original_filename")
        if not original_name:
            original_name = pdf_path.name

        session_payload = {
            "file_path": str(pdf_path),
            "original_filename": original_name,
        }
        if email:
            session_payload["email"] = email

        update_session(session_id, **session_payload)

        result = run_full_pipeline(session_id).get(timeout=300)

        try:
            cs_api.load_session_case(session_id)
            problem_accounts = orch.collect_stageA_logical_accounts(session_id)
        except CaseStoreError:
            logger.exception("casestore_unavailable session=%s", session_id)
            return (
                jsonify(
                    {
                        "status": "error",
                        "message": "Case Store unavailable",
                    }
                ),
                503,
            )

        valid_accounts = []
        for acc in problem_accounts:
            to_validate = dict(acc)
            to_validate.pop("aggregation_meta", None)
            try:
                _problem_account_validator.validate(to_validate)
                valid_accounts.append(acc)
            except ValidationError:
                logger.warning(
                    "invalid_problem_account session=%s account=%s",
                    session_id,
                    acc,
                    exc_info=True,
                )
        problem_accounts = valid_accounts
        if config.API_INCLUDE_DECISION_META:
            for acc in problem_accounts:
                meta = orch.get_stageA_decision_meta(session_id, acc.get("account_id"))
                if meta is None:
                    meta = {
                        "decision_source": acc.get("decision_source", "rules"),
                        "confidence": acc.get("confidence", 0.0),
                        "tier": acc.get("tier", "none"),
                    }
                    fields_used = acc.get("fields_used")
                    if fields_used:
                        meta["fields_used"] = fields_used
                fields_used = meta.get("fields_used")
                if fields_used:
                    meta["fields_used"] = list(fields_used)[
                        : config.API_DECISION_META_MAX_FIELDS_USED
                    ]
                acc["decision_meta"] = meta

        legacy = request.args.get("legacy", "").lower() in ("1", "true", "yes")

        accounts = {
            # Primary field
            "problem_accounts": problem_accounts,
        }

        if legacy:
            # Backward compatibility fields for legacy clients
            accounts["negative_accounts"] = problem_accounts
            accounts["open_accounts_with_issues"] = problem_accounts

        accounts["unauthorized_inquiries"] = result.get(
            "unauthorized_inquiries", result.get("inquiries", [])
        )
        accounts["high_utilization_accounts"] = result.get(
            "high_utilization_accounts", result.get("high_utilization", [])
        )

        payload = {
            "status": "awaiting_user_explanations",
            "session_id": session_id,
            "filename": pdf_path.name,
            "original_filename": original_name,
            "accounts": accounts,
        }

        logger.info("start_process payload: %s", payload)

        return jsonify(payload)

    except Exception as e:
        print("Exception occurred:", str(e))
        return jsonify({"status": "error", "message": str(e)}), 500


# ---------------------------------------------------------------------------
# Async upload → queue analysis
# ---------------------------------------------------------------------------


@api_bp.route("/api/upload", methods=["POST"])
def api_upload():
    try:
        form = request.form or {}
        email = (form.get("email") or "").strip()
        file = request.files.get("file")
        if not email or not file:
            return jsonify({"ok": False, "message": "missing fields"}), 400

        first_bytes = file.stream.read(4)
        file.stream.seek(0)
        if first_bytes != b"%PDF":
            return jsonify({"ok": False, "message": "Invalid PDF file"}), 400

        session_id = str(uuid.uuid4())

        tmp_handle = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        tmp_path = Path(tmp_handle.name)
        tmp_handle.close()
        file.save(tmp_path)
        try:
            pdf_path = move_uploaded_file(tmp_path, session_id, allow_create=True)
        finally:
            try:
                tmp_path.unlink(missing_ok=True)
            except FileNotFoundError:
                pass

        if not pdf_path.exists():
            return jsonify({"ok": False, "message": "File upload failed"}), 400

        logger.info("UPLOAD_CANONICALIZED sid=%s dest=%s", session_id, str(pdf_path))

        original_name = file.filename or "report.pdf"

        # Persist initial session state
        set_session(
            session_id,
            {
                "file_path": str(pdf_path),
                "original_filename": original_name,
                "email": email,
                "status": "queued",
            },
        )

        # Queue background extraction (non-blocking)
        task = run_full_pipeline(session_id)
        update_session(session_id, task_id=task.id, status="queued")

        # Return explicit async contract (frontend polls /api/result)
        return (
            jsonify(
                {
                    "ok": True,
                    "status": "queued",
                    "session_id": session_id,
                    "task_id": task.id,
                }
            ),
            202,
        )
    except Exception as e:  # pragma: no cover - defensive
        logger.exception("upload failed")
        return jsonify({"ok": False, "message": str(e)}), 500


@api_bp.route("/api/result", methods=["GET"])
def api_result():
    session_id = (request.args.get("session_id") or "").strip()
    if not session_id:
        return jsonify({"ok": False, "message": "missing session_id"}), 400
    session = get_session(session_id)
    if session is None:
        # Tolerant contract: treat not-found as in-progress to avoid noisy 404s
        return jsonify({"ok": True, "status": "processing"}), 200

    status = session.get("status") or "queued"
    if status in ("queued", "processing"):
        return jsonify({"ok": True, "status": status}), 200
    if status == "error":
        return (
            jsonify(
                {"ok": False, "status": "error", "message": session.get("error") or ""}
            ),
            200,
        )

    # done
    payload = session.get("result") or {}
    return (
        jsonify(
            {
                "ok": True,
                "status": "done",
                "session_id": session_id,
                "result": payload,
            }
        ),
        200,
    )


@api_bp.route("/api/explanations", methods=["POST"])
def explanations_endpoint():
    data = request.get_json(force=True)
    session_id = data.get("session_id")
    explanations = data.get("explanations", [])

    if not session_id or not isinstance(explanations, list):
        return jsonify({"status": "error", "message": "Invalid input"}), 400

    structured: list[dict] = []
    raw_store: list[dict] = []
    for item in explanations:
        text = item.get("text", "")
        ctx = {
            "account_id": item.get("account_id", ""),
            "dispute_type": item.get("dispute_type", ""),
        }
        raw_store.append({"account_id": ctx["account_id"], "text": text})
        safe = sanitize(text)
        structured.append(extract_structured(safe, ctx))

    update_session(session_id, structured_summaries=structured)
    update_intake(session_id, raw_explanations=raw_store)
    return jsonify({"status": "ok", "structured": structured})


@api_bp.route("/api/summaries/<session_id>", methods=["GET"])
def get_summaries(session_id: str):
    session = get_session(session_id)
    if not session:
        return jsonify({"status": "error", "message": "Session not found"}), 404
    raw = session.get("structured_summaries", {}) or {}
    allowed = {
        "account_id",
        "dispute_type",
        "facts_summary",
        "claimed_errors",
        "dates",
        "evidence",
        "risk_flags",
    }
    cleaned: dict[str, dict] = {}
    if isinstance(raw, list):
        for item in raw:
            if isinstance(item, dict):
                key = item.get("account_id") or str(len(cleaned))
                cleaned[key] = {k: item.get(k) for k in allowed if k in item}
    elif isinstance(raw, dict):
        for key, item in raw.items():
            cleaned[key] = {k: item.get(k) for k in allowed if k in item}
    logger.debug("summaries payload for %s: %s", session_id, cleaned)
    return jsonify({"status": "ok", "summaries": cleaned})


@api_bp.route("/api/account-transitions/<session_id>/<account_id>", methods=["GET"])
def account_transitions(session_id: str, account_id: str):
    session = get_session(session_id)
    if not session:
        return jsonify({"status": "error", "message": "Session not found"}), 404
    states = session.get("account_states", {}) or {}
    data = states.get(str(account_id))
    if not data:
        return jsonify({"status": "error", "message": "Account not found"}), 404
    return jsonify({"status": "ok", "history": data.get("history", [])})


@api_bp.route("/api/submit-explanations", methods=["POST"])
def submit_explanations():
    return redirect(url_for("api.explanations_endpoint"), code=307)


def create_app() -> Flask:
    sanitize_openai_env()
    app = Flask(__name__)

    cors_enable = os.getenv("CORS_ENABLE", "").strip().lower()
    if cors_enable in {"1", "true", "yes", "on"}:
        allowed_origins = [
            origin.strip()
            for origin in os.getenv("CORS_ALLOW_ORIGINS", "").split(",")
            if origin.strip()
        ]
        CORS(
            app,
            resources={
                r"/api/*": {"origins": allowed_origins},
                r"/runs/*": {"origins": allowed_origins},
            },
            supports_credentials=True,
        )
    app.register_blueprint(admin_bp)
    app.register_blueprint(review_bp)
    app.register_blueprint(api_bp)
    app.register_blueprint(ai_bp)
    app.register_blueprint(ui_event_bp)
    app.register_blueprint(run_assets_bp)
    app.register_blueprint(smoke_bp, url_prefix="/smoke")

    def _run_boot_reconcile_once() -> None:
        if getattr(app, "_runflow_boot_reconciled", False):
            return
        should_run = _env_flag_enabled("RUNFLOW_RECONCILE_ON_BOOT", default=False)
        if not should_run:
            app._runflow_boot_reconciled = True
            return
        try:
            _reconcile_runs_on_boot()
        except Exception:  # pragma: no cover - defensive logging
            logger.warning("RUNFLOW_BOOT_RECONCILE_ABORTED", exc_info=True)
        finally:
            app._runflow_boot_reconciled = True

    @app.before_request
    def _load_config() -> None:
        if getattr(app, "_config_loaded", False):
            _run_boot_reconcile_once()
            return
        cfg = get_app_config()
        celery_app.conf.update(
            broker_url=cfg.celery_broker_url,
            result_backend=cfg.celery_broker_url,
        )
        os.environ.setdefault("OPENAI_API_KEY", cfg.ai.api_key)
        app.secret_key = cfg.secret_key
        app.auth_tokens = cfg.auth_tokens
        app.rate_limit_per_minute = cfg.rate_limit_per_minute
        logger.info("Flask app starting with OPENAI_BASE_URL=%s", cfg.ai.base_url)
        logger.info("Flask app OPENAI_API_KEY present=%s", bool(cfg.ai.api_key))
        app._config_loaded = True
        _run_boot_reconcile_once()

    @app.before_request
    def _auth_and_throttle() -> tuple[dict, int] | None:
        tokens: list[str] = getattr(app, "auth_tokens", [])
        limit: int = getattr(app, "rate_limit_per_minute", 60)
        identifier = request.remote_addr or "global"
        if tokens:
            auth_header = request.headers.get("Authorization", "")
            token = (
                auth_header[7:].strip() if auth_header.startswith("Bearer ") else None
            )
            if token not in tokens:
                return jsonify({"status": "error", "message": "Unauthorized"}), 401
            identifier = token
        now = time.time()
        recent = [t for t in _request_counts[identifier] if now - t < 60]
        if len(recent) >= limit:
            return jsonify({"status": "error", "message": "Too Many Requests"}), 429
        recent.append(now)
        _request_counts[identifier] = recent

    @app.route("/runs/<sid>/frontend/packs", methods=["GET"])
    def dev_frontend_packs(sid: str):
        try:
            run_dir = _run_dir_for_sid(sid)
        except ValueError:
            return jsonify({"error": "invalid_sid"}), 400

        stage_config = load_frontend_stage_config(run_dir)
        packs_dir = stage_config.packs_dir
        packs: list[str] = []

        if packs_dir.is_dir():
            for entry in sorted(packs_dir.iterdir()):
                if not entry.is_file():
                    continue
                if not FRONTEND_PACK_FILENAME_PATTERN.fullmatch(entry.name):
                    continue
                try:
                    rel_path = entry.relative_to(run_dir).as_posix()
                except ValueError:
                    rel_path = entry.as_posix()
                packs.append(rel_path)

        try:
            relative_dir = packs_dir.relative_to(run_dir).as_posix()
        except ValueError:
            relative_dir = packs_dir.as_posix()

        return jsonify({"dir": relative_dir, "packs": packs})

    @app.route("/runs/<sid>/frontend/pack/<account_id>", methods=["GET"])
    def dev_frontend_pack(sid: str, account_id: str):
        try:
            run_dir = _run_dir_for_sid(sid)
        except ValueError:
            return jsonify({"error": "invalid_sid"}), 400

        stage_pack = _stage_pack_path_for_account(run_dir, account_id)
        if stage_pack is None or not stage_pack.is_file():
            return jsonify({"error": "pack_not_found"}), 404

        try:
            payload = _load_frontend_pack(stage_pack)
        except Exception:  # pragma: no cover - error path
            return jsonify({"error": "pack_read_failed"}), 500

        return jsonify(payload)

    return app


# ---------------------------------------------------------------------------
# Accounts API (reads analyzer-produced artifacts)
# ---------------------------------------------------------------------------


@api_bp.route("/api/account/<session_id>/<account_id>", methods=["GET"])
def account_view_api(session_id: str, account_id: str):
    if not session_id or not account_id:
        return jsonify({"error": "invalid_request"}), 400
    try:
        view = build_account_view(session_id, account_id)
    except CaseStoreError as exc:  # pragma: no cover - error path
        if getattr(exc, "code", "") == NOT_FOUND:
            return jsonify({"error": "account_not_found"}), 404
        logger.exception(
            "account_view_failed session=%s account=%s", session_id, account_id
        )
        return jsonify({"error": "internal_error"}), 500
    return jsonify(view)


@api_bp.route("/api/accounts/<session_id>", methods=["GET"])
def list_accounts_api(session_id: str):
    """Return compact list of problem accounts built from Case Store artifacts."""

    if not session_id:
        return jsonify({"ok": False, "message": "missing session_id"}), 400

    try:
        probs = collect_stageA_problem_accounts(session_id) or []
    except CaseStoreError:
        probs = []
    try:
        logical = collect_stageA_logical_accounts(session_id) or []
    except CaseStoreError:
        logical = []

    accounts = _merge_collectors(probs, logical)

    if FLAGS.case_first_build_required and not accounts:
        return jsonify({"ok": True, "session_id": session_id, "accounts": []})

    return jsonify({"ok": True, "session_id": session_id, "accounts": accounts})


@api_bp.route("/api/problem_accounts")
def api_problem_accounts_legacy():
    """Legacy parser-first endpoint intentionally disabled."""
    if FLAGS.disable_parser_ui_summary:
        return jsonify({"ok": False, "error": "parser_first_disabled"}), 410
    return jsonify({"ok": False, "error": "parser_first_disabled"}), 410


@api_bp.route("/api/cases/<session_id>", methods=["GET"])
def api_list_cases(session_id: str):
    try:
        session_case = cs_api.load_session_case(session_id)
    except Exception as e:  # pragma: no cover - debug endpoint
        return (
            jsonify({"ok": False, "session_id": session_id, "error": str(e)}),
            200,
        )

    accounts = session_case.accounts or {}
    logical_index = session_case.summary.logical_index or {}
    reverse_index = {aid: lk for lk, aid in logical_index.items()}
    items = []
    for aid, account in accounts.items():
        issuer = None
        logical_key = reverse_index.get(aid)
        try:
            by_bureau = getattr(account.fields, "by_bureau", {}) or {}
            for bureau_code in ("EX", "EQ", "TU"):
                bureau_obj = by_bureau.get(bureau_code) or {}
                issuer = (
                    issuer
                    or bureau_obj.get("issuer")
                    or bureau_obj.get("creditor_name")
                )
        except Exception:
            pass
        items.append({"case_id": aid, "issuer": issuer, "logical_key": logical_key})

    return jsonify({"ok": True, "session_id": session_id, "cases": items})


@api_bp.route("/api/session/<session_id>/logical_index", methods=["GET"])
def api_logical_index(session_id: str):
    try:
        session_case = cs_api.load_session_case(session_id)
        idx = session_case.summary.logical_index or {}
        return jsonify({"ok": True, "session_id": session_id, "logical_index": idx})
    except Exception as e:  # pragma: no cover - debug endpoint
        return (
            jsonify({"ok": False, "session_id": session_id, "error": str(e)}),
            200,
        )


if __name__ == "__main__":  # pragma: no cover - manual execution
    debug_mode = os.environ.get("FLASK_DEBUG", "0") in ("1", "true", "True")
    create_app().run(host="0.0.0.0", port=5000, debug=debug_mode)
