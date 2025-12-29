"""Send Validation AI packs to the model and persist the responses."""

from __future__ import annotations

import importlib
import inspect
import json
import logging
import os
import random
import re
import threading
import time
from collections import Counter
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Iterable, Mapping, MutableMapping, Sequence

from jsonschema import Draft7Validator

from backend.ai.validation_index import ValidationPackIndexWriter
from backend.ai.manifest import (
    StageManifestPaths,
    extract_stage_manifest_paths,
)
from backend.telemetry.metrics import emit_counter
from backend.core.runflow import (
    record_validation_results_summary,
    runflow_barriers_refresh,
)
from backend.core.ai.paths import (
    validation_result_error_filename_for_account,
    validation_result_json_filename_for_account,
    validation_result_jsonl_filename_for_account,
    validation_result_summary_filename_for_account,
    validation_write_json_enabled,
)
from backend.core.paths import normalize_stage_path, sanitize_stage_path_value
from backend.core.ai import PROJECT_HEADER_NAME, auth_probe, build_openai_headers
from backend.core.logic.validation_field_sets import (
    ALL_VALIDATION_FIELDS,
    ALWAYS_INVESTIGATABLE_FIELDS,
    CONDITIONAL_FIELDS,
)
from backend.runflow.decider import (
    reconcile_umbrella_barriers,
    refresh_validation_stage_from_index,
)
from backend.validation.index_schema import (
    ValidationIndex,
    ValidationPackRecord,
    load_validation_index,
)
from .io import read_jsonl, write_json, write_jsonl

from backend.pipeline.runs import RunManifest, persist_manifest, _utc_now
from backend.validation.redaction import sanitize_validation_log_payload

_DEFAULT_MODEL = "gpt-4o-mini"
_DEFAULT_TIMEOUT = 30.0
_THROTTLE_SECONDS = 0.05
_DEFAULT_QUEUE_NAME = "validation"
_VALID_DECISIONS = {
    "strong_actionable",
    "supportive_needs_companion",
    "neutral_context_only",
    "no_case",
}
_LEGACY_DECISION_MAP = {
    "strong": "strong_actionable",
    "supportive": "supportive_needs_companion",
    "neutral": "neutral_context_only",
    "no_case": "no_case",
    "no_claim": "no_case",
    "no_claims": "no_case",
}
_REVERSE_LEGACY_DECISION_MAP = {
    "strong_actionable": "strong",
    "supportive_needs_companion": "supportive",
    "neutral_context_only": "neutral",
    "no_case": "no_case",
}


def _remap_legacy_decision_enum(
    values: Sequence[Any] | None,
) -> tuple[list[Any], bool]:
    if values is None:
        return [], False
    if isinstance(values, (str, bytes, bytearray)):
        return [values], False
    mapped: list[Any] = []
    changed = False
    for entry in values:
        if isinstance(entry, str):
            lowered = entry.strip().lower()
            replacement = _LEGACY_DECISION_MAP.get(lowered)
            if replacement is not None:
                mapped.append(replacement)
                if replacement != lowered:
                    changed = True
                continue
        mapped.append(entry)
    return mapped, changed
_ALWAYS_INVESTIGATABLE_FIELDS = ALWAYS_INVESTIGATABLE_FIELDS
_CONDITIONAL_FIELDS = CONDITIONAL_FIELDS
_ALLOWED_FIELDS = frozenset(ALL_VALIDATION_FIELDS)
_CREDITOR_REMARK_KEYWORDS = (
    "charge off",
    "charge-off",
    "consumer dispute",
    "consumer disputes",
    "consumer states",
    "fcra",
    "fraud",
    "fraudulent",
    "repossession",
)

_CONFIDENCE_THRESHOLD_ENV = "VALIDATION_AI_MIN_CONFIDENCE"
_USE_MANIFEST_PATHS_ENV = "VALIDATION_USE_MANIFEST_PATHS"
_MANIFEST_STAGE = "validation"
_MANIFEST_RETRY_ATTEMPTS = 5
_MANIFEST_RETRY_DELAY = 0.5
_INDEX_WAIT_ATTEMPTS = 10
_INDEX_WAIT_DELAY = 0.5
_INDEX_READY_ATTEMPTS = 10
_INDEX_READY_MIN_DELAY = 0.3
_INDEX_READY_MAX_DELAY = 0.5
_INDEX_FILE_WAIT_ATTEMPTS = 10
_INDEX_FILE_WAIT_DELAY = 0.4
_INDEX_FILE_MIN_SIZE = 20
_DEFAULT_CONFIDENCE_THRESHOLD = 0.70
_WRITE_JSON_ENVELOPE_ENV = "VALIDATION_WRITE_JSON_ENVELOPE"
_LOG_PATH_REL_ENV = "VALIDATION_LOG_PATH_REL"
_DEFAULT_LOG_FILENAME = "logs.txt"
_DEBUG_ENV = "VALIDATION_DEBUG"


def _canonical_result_path(path: Path, *, allow_json: bool = False) -> Path:
    """Normalize ``.result.json`` paths to the canonical ``.result.jsonl``."""

    if allow_json or path.suffix.lower() != ".json":
        return path
    if not path.name.endswith(".result.json"):
        return path
    return path.with_suffix(".jsonl")


def _canonical_result_display(display: str | None, *, allow_json: bool = False) -> str | None:
    """Normalize display paths so we never advertise ``.result.json`` outputs."""

    if not display:
        return display
    normalized = display.replace("\\", "/")
    if allow_json:
        return normalized
    if normalized.endswith(".result.json"):
        return normalized[:-5] + "jsonl"
    return normalized
_VALIDATION_MAX_RETRIES_ENV = "VALIDATION_MAX_RETRIES"
_DEFAULT_VALIDATION_MAX_RETRIES = 2
_VALIDATION_REQUEST_GROUP_SIZE_ENV = "VALIDATION_REQUEST_GROUP_SIZE"
_VALIDATION_REQUEST_GROUP_SIZE = 1


log = logging.getLogger(__name__)

requests: Any | None = None

_AUTH_READY = False
_AUTH_LOCK = threading.Lock()


def _key_prefix(value: str) -> str:
    if not value:
        return "<empty>"
    if value.startswith("sk-proj-"):
        return "sk-proj..."
    if value.startswith("sk-"):
        return "sk-..."
    if len(value) <= 4:
        return f"{value}..."
    return f"{value[:4]}..."


def _confidence_threshold() -> float:
    raw = os.getenv(_CONFIDENCE_THRESHOLD_ENV)
    if raw is None:
        return _DEFAULT_CONFIDENCE_THRESHOLD
    try:
        value = float(str(raw).strip())
    except (TypeError, ValueError):
        log.warning(
            "VALIDATION_AI_CONFIDENCE_PARSE_FAILED value=%s default=%s",
            raw,
            _DEFAULT_CONFIDENCE_THRESHOLD,
        )
        return _DEFAULT_CONFIDENCE_THRESHOLD
    if value < 0 or value > 1:
        log.warning(
            "VALIDATION_AI_CONFIDENCE_OUT_OF_RANGE value=%s default=%s",
            value,
            _DEFAULT_CONFIDENCE_THRESHOLD,
        )
        return _DEFAULT_CONFIDENCE_THRESHOLD
    return value


def _coerce_bool_flag(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "y", "on"}:
            return True
        if lowered in {"0", "false", "no", "n", "off"}:
            return False
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return bool(value)
    return False


def _coerce_int_value(value: Any, default: int = 1) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _normalize_bureau_key(value: Any) -> str | None:
    if value is None:
        return None
    try:
        text = str(value).strip()
    except Exception:
        return None
    if not text:
        return None
    return text.lower()


def _resolve_request_group_size() -> int:
    raw = os.getenv(_VALIDATION_REQUEST_GROUP_SIZE_ENV)
    if raw is None:
        return _VALIDATION_REQUEST_GROUP_SIZE
    text = str(raw).strip()
    if not text:
        return _VALIDATION_REQUEST_GROUP_SIZE
    try:
        size = int(text)
    except (TypeError, ValueError):
        log.warning(
            "VALIDATION_REQUEST_GROUP_SIZE_PARSE_FAILED value=%s forcing=%s",
            raw,
            _VALIDATION_REQUEST_GROUP_SIZE,
        )
        return _VALIDATION_REQUEST_GROUP_SIZE
    if size < 1:
        log.warning(
            "VALIDATION_REQUEST_GROUP_SIZE_INVALID size=%s forcing=%s",
            size,
            _VALIDATION_REQUEST_GROUP_SIZE,
        )
        return _VALIDATION_REQUEST_GROUP_SIZE
    if size != _VALIDATION_REQUEST_GROUP_SIZE:
        log.warning(
            "VALIDATION_REQUEST_GROUP_SIZE_OVERRIDE size=%s forcing=%s",
            size,
            _VALIDATION_REQUEST_GROUP_SIZE,
        )
    return _VALIDATION_REQUEST_GROUP_SIZE


def _extract_bureau_records(
    pack_line: Mapping[str, Any]
) -> dict[str, dict[str, Any]]:
    bureaus = pack_line.get("bureaus")
    if not isinstance(bureaus, Mapping):
        return {}
    records: dict[str, dict[str, Any]] = {}
    for bureau, value in bureaus.items():
        key = _normalize_bureau_key(bureau)
        if not key or not isinstance(value, Mapping):
            continue
        records[key] = {
            "raw": value.get("raw"),
            "normalized": value.get("normalized"),
        }
    return records


def _normalize_account_number_token(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, Mapping):
        last4 = value.get("last4")
        if isinstance(last4, str) and last4.strip():
            return last4.strip()
        display = value.get("display")
        if isinstance(display, str) and display.strip():
            digits = re.findall(r"\d", display)
            if len(digits) >= 4:
                return "".join(digits[-4:])
            return display.strip().lower()
        for candidate in ("normalized", "raw", "value", "text"):
            if candidate in value:
                token = _normalize_account_number_token(value[candidate])
                if token:
                    return token
        return None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        digits = re.findall(r"\d", text)
        if len(digits) >= 4:
            return "".join(digits[-4:])
        return text.lower()
    try:
        text = str(value).strip()
    except Exception:
        return None
    if not text:
        return None
    digits = re.findall(r"\d", text)
    if len(digits) >= 4:
        return "".join(digits[-4:])
    return text.lower()


def _normalize_text_fragment(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, Mapping):
        for candidate in ("normalized", "raw", "value", "text", "display"):
            if candidate in value:
                normalized = _normalize_text_fragment(value[candidate])
                if normalized:
                    return normalized
        return None
    text = str(value).strip().lower()
    if not text:
        return None
    return " ".join(text.split())


def _normalize_text_value(data: Mapping[str, Any]) -> str | None:
    for key in ("normalized", "raw"):
        if key in data:
            normalized = _normalize_text_fragment(data[key])
            if normalized:
                return normalized
    return None


def _normalize_account_number_value(data: Mapping[str, Any]) -> str | None:
    for key in ("normalized", "raw"):
        if key in data:
            token = _normalize_account_number_token(data[key])
            if token:
                return token
    return None


def _conditional_mismatch_metrics(
    field: str, pack_line: Mapping[str, Any]
) -> tuple[bool, int, list[str]]:
    records = _extract_bureau_records(pack_line)
    normalized_values: list[str] = []

    if field == "account_number_display":
        for record in records.values():
            token = _normalize_account_number_value(record)
            if token:
                normalized_values.append(token)
    else:
        for record in records.values():
            token = _normalize_text_value(record)
            if token:
                normalized_values.append(token)

    unique_values = sorted(set(normalized_values))
    mismatch = len(unique_values) >= 2
    corroboration = len(unique_values)
    return mismatch, corroboration, normalized_values


def _has_high_signal_creditor_remarks(values: Sequence[str]) -> bool:
    for value in values:
        if any(keyword in value for keyword in _CREDITOR_REMARK_KEYWORDS):
            return True
    return False


def _append_gate_note(rationale: str, reason: str) -> str:
    note = f"[conditional_gate:{reason}]"
    if not rationale:
        return note
    return f"{rationale} {note}"


def _append_guardrail_note(rationale: str, reason: str) -> str:
    note = f"[guardrail:{reason}]"
    if not rationale:
        return note
    return f"{rationale} {note}"


def _normalize_structured_decision(value: Any) -> str | None:
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in _VALID_DECISIONS:
            return lowered
        mapped = _LEGACY_DECISION_MAP.get(lowered)
        if mapped in _VALID_DECISIONS:
            return mapped
    return None


def _normalize_structured_rationale(value: Any) -> str | None:
    if isinstance(value, str):
        text = value.strip()
        if text:
            return text
    return None


def _normalize_structured_citations(value: Any) -> list[str] | None:
    if value is None:
        return []
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return None
    citations: list[str] = []
    for item in value:
        if isinstance(item, str):
            text = item.strip()
            if text:
                citations.append(text)
    return citations


def _normalize_structured_labels(value: Any) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return []
    labels: list[str] = []
    for item in value:
        if isinstance(item, str):
            text = item.strip()
            if text:
                labels.append(text)
    return labels


def _normalize_structured_confidence(value: Any) -> float | None:
    if value is None:
        return None
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return None
    if confidence < 0 or confidence > 1:
        return None
    return round(confidence, 6)


_MISSING = object()


def validate_and_normalize(
    response: Mapping[str, Any] | None, finding: Mapping[str, Any] | None
) -> tuple[dict[str, Any] | None, list[str]]:
    if not isinstance(response, Mapping):
        return None, ["response_not_mapping"]

    required_fields: set[str] = {"decision", "rationale", "citations"}
    if isinstance(finding, Mapping):
        expected_output = finding.get("expected_output")
        if isinstance(expected_output, Mapping):
            raw_required = expected_output.get("required")
            if isinstance(raw_required, Sequence) and not isinstance(
                raw_required, (str, bytes, bytearray)
            ):
                for entry in raw_required:
                    if isinstance(entry, str):
                        text = entry.strip()
                        if text:
                            required_fields.add(text)

    normalized: dict[str, Any] = {}
    errors: list[str] = []

    decision_raw = response.get("decision", _MISSING)
    decision_value = _normalize_structured_decision(decision_raw)
    if decision_value is None:
        if "decision" in required_fields:
            errors.append("decision_invalid")
    else:
        normalized["decision"] = decision_value

    rationale_raw = response.get("rationale", _MISSING)
    rationale_value = _normalize_structured_rationale(rationale_raw)
    if rationale_value is None:
        if "rationale" in required_fields:
            errors.append("rationale_missing")
    else:
        normalized["rationale"] = rationale_value

    citations_raw = response.get("citations", _MISSING)
    if citations_raw is _MISSING and "citations" in required_fields:
        errors.append("citations_missing")
        citations_value: list[str] | None = None
    else:
        citations_value = _normalize_structured_citations(
            None if citations_raw is _MISSING else citations_raw
        )
        if citations_value is None:
            if "citations" in required_fields:
                errors.append("citations_invalid")
        else:
            normalized["citations"] = citations_value

    labels_value = _normalize_structured_labels(response.get("labels"))
    if labels_value:
        normalized["labels"] = labels_value

    confidence_value = _normalize_structured_confidence(response.get("confidence"))
    if confidence_value is not None:
        normalized["confidence"] = confidence_value

    if errors:
        return None, errors

    normalized.setdefault("citations", [])
    return normalized, []


def make_fallback_decision(_: Mapping[str, Any] | None) -> dict[str, Any]:
    return {
        "decision": "no_case",
        "rationale": "schema_mismatch",
        "citations": ["system:none"],
        "checks": {
            "materiality": False,
            "supports_consumer": False,
            "doc_requirements_met": False,
            "mismatch_code": "unknown",
        },
    }


def correction_suffix(errors: list[str]) -> str:
    bullets = "; ".join(errors[:3])
    return (
        "\n\nFIX:\n"
        f"- Your previous output was invalid: {bullets}.\n"
        "- Output ONE JSON object only, strictly matching the schema with non-empty 'citations'."
    )


def _empty_decision_metrics() -> dict[str, dict[str, int]]:
    buckets = {"conditional": 0, "non_conditional": 0}
    return {
        "strong": dict(buckets),
        "supportive": dict(buckets),
        "neutral": dict(buckets),
        "no_case": dict(buckets),
        "weak": dict(buckets),
    }


def _enforce_conditional_gate(
    field: str,
    decision: str,
    rationale: str,
    pack_line: Mapping[str, Any],
) -> tuple[str, str, Mapping[str, Any] | None]:
    if field not in _CONDITIONAL_FIELDS:
        return decision, rationale, None
    if decision != "strong":
        return decision, rationale, None
    if not _coerce_bool_flag(pack_line.get("conditional_gate")):
        return decision, rationale, None

    min_corroboration = max(1, _coerce_int_value(pack_line.get("min_corroboration"), 1))
    mismatch, corroboration, normalized_values = _conditional_mismatch_metrics(
        field, pack_line
    )

    if field == "creditor_remarks" and mismatch:
        if not _has_high_signal_creditor_remarks(normalized_values):
            mismatch = False

    if not mismatch or corroboration < min_corroboration:
        gate_payload: dict[str, Any] = {
            "reason": "insufficient_evidence",
            "corroboration": corroboration,
            "unique_values": sorted(set(normalized_values)),
            "required_corroboration": min_corroboration,
        }
        if field == "creditor_remarks":
            gate_payload["high_signal_keywords"] = _CREDITOR_REMARK_KEYWORDS
        return (
            "no_case",
            _append_gate_note(rationale, "insufficient_evidence"),
            gate_payload,
        )

    return decision, rationale, None


class ValidationPackError(RuntimeError):
    """Raised when the Validation AI sender encounters a fatal error."""


def _ensure_requests_module() -> Any:
    """Load the ``requests`` module lazily so tests can stub it."""

    global requests
    if requests is not None:
        return requests

    try:
        requests = importlib.import_module("requests")
    except ModuleNotFoundError as exc:  # pragma: no cover - defensive import
        raise ValidationPackError(
            "requests library is required to send validation packs"
        ) from exc

    return requests


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


def _validation_max_retries() -> int:
    raw = os.getenv(_VALIDATION_MAX_RETRIES_ENV)
    if raw is None:
        return _DEFAULT_VALIDATION_MAX_RETRIES
    try:
        value = int(str(raw).strip())
    except (TypeError, ValueError):
        return _DEFAULT_VALIDATION_MAX_RETRIES
    if value < 0:
        return 0
    return value


def _truncate_response_body(body: Any, *, limit: int = 300) -> str:
    text = ""
    if isinstance(body, str):
        text = body
    elif body is None:
        text = ""
    else:
        text = str(body)
    text = text.strip()
    if len(text) > limit:
        return text[:limit]
    return text


@dataclass(slots=True)
@dataclass(frozen=True)
class _ChatCompletionResponse:
    """Wrapper around the chat completion response payload."""

    payload: Mapping[str, Any]
    status_code: int
    latency: float
    retries: int


class _ChatCompletionClient:
    """Minimal HTTP client for the OpenAI Chat Completions API."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        timeout: float | int,
        project_id: str | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/") or "https://api.openai.com/v1"
        self.api_key = api_key
        self.timeout: float | int = timeout
        self.project_id = project_id

    def create(
        self,
        payload: Mapping[str, Any],
        *,
        pack_id: str | None = None,
        on_error: Callable[[int, str], None] | None = None,
    ) -> _ChatCompletionResponse:
        url = f"{self.base_url}/chat/completions"
        headers = build_openai_headers(api_key=self.api_key, project_id=self.project_id)

        response_format = {}
        if isinstance(payload, Mapping):
            response_format = payload.get("response_format", {}) or {}

        include_beta_header = (
            isinstance(response_format, Mapping)
            and response_format.get("type") == "json_object"
        )
        if include_beta_header:
            headers["OpenAI-Beta"] = "response_format=v1"

        log.info(
            "VALIDATION_HTTP_HEADERS_SET beta_structured=%s",
            "true" if include_beta_header else "false",
        )

        request_lib = _ensure_requests_module()
        start_time = time.monotonic()
        log.debug(
            "OPENAI_CALL endpoint=%s has_project_header=%s",
            url,
            PROJECT_HEADER_NAME in headers,
        )
        response = request_lib.post(url, headers=headers, json=payload, timeout=self.timeout)
        latency = time.monotonic() - start_time
        status_code = getattr(response, "status_code", 0)
        try:
            body_text = response.text
        except Exception:  # pragma: no cover - defensive logging
            body_text = "<unavailable>"
        snippet = _truncate_response_body(body_text)

        def _record_error(status: int, body: str) -> None:
            if on_error is None:
                return
            try:
                on_error(status, body)
            except Exception:  # pragma: no cover - best effort logging
                log.exception(
                    "VALIDATION_HTTP_ERROR_SIDECAR_FAILED pack=%s",
                    pack_id or "<unknown>",
                )

        if not getattr(response, "ok", False):
            log.error(
                "VALIDATION_HTTP_ERROR pack=%s status=%s body=%s",
                pack_id or "<unknown>",
                status_code,
                snippet or "<empty>",
            )
            try:
                normalized_status = int(status_code)
            except (TypeError, ValueError):
                normalized_status = 0
            _record_error(normalized_status, snippet or "")
        response.raise_for_status()

        return _ChatCompletionResponse(
            payload=self._safe_json(
                response,
                pack_id=pack_id,
                snippet=snippet,
                status_code=status_code,
                on_error=_record_error,
            ),
            status_code=getattr(response, "status_code", 0),
            latency=latency,
            retries=0,
        )

    @staticmethod
    def _safe_json(
        response: Any,
        *,
        pack_id: str | None,
        snippet: str,
        status_code: Any,
        on_error: Callable[[int, str], None],
    ) -> Mapping[str, Any]:
        try:
            payload = response.json()
        except ValueError as exc:
            log.error("VALIDATION_EMPTY_RESPONSE pack=%s", pack_id or "<unknown>")
            try:
                normalized_status = int(status_code)
            except (TypeError, ValueError):
                normalized_status = 0
            on_error(normalized_status, snippet or "")
            raise ValidationPackError("Response JSON parse failed") from exc
        if payload is None:
            log.error("VALIDATION_EMPTY_RESPONSE pack=%s", pack_id or "<unknown>")
            try:
                normalized_status = int(status_code)
            except (TypeError, ValueError):
                normalized_status = 0
            on_error(normalized_status, snippet or "")
            raise ValidationPackError("Response JSON payload is empty")
        return payload


@dataclass(frozen=True)
class _ManifestView:
    """Resolved information about a validation manifest."""

    index: ValidationIndex
    log_path: Path
    stage_paths: StageManifestPaths | None = None


@dataclass(frozen=True)
class _IndexPreparationResult:
    """Outcome of preparing the validation index prior to sending."""

    index: ValidationIndex | None
    view: _ManifestView | None
    skip: bool = False


@dataclass(frozen=True)
class _PreflightAccount:
    """Resolved paths for a single manifest entry during validation."""

    record: ValidationPackRecord
    pack_path: Path
    result_jsonl_path: Path
    result_json_path: Path
    pack_missing: bool


@dataclass(frozen=True)
class _PreflightSummary:
    """Aggregate data produced by :meth:`ValidationPackSender._preflight`."""

    manifest_path: Path
    accounts: tuple[_PreflightAccount, ...]
    missing: int
    results_dir_created: bool
    parent_dirs_created: int


def _coerce_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _coerce_str(value: Any, default: str = "") -> str:
    if isinstance(value, str):
        text = value.strip()
        return text or default
    if value is None:
        return default
    return str(value)


def _count_index_pairs(path: Path) -> int:
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return 0

    packs: Any = None
    if isinstance(payload, Mapping):
        packs = payload.get("packs")
        if packs is None:
            packs = payload.get("items")

    if isinstance(packs, Sequence) and not isinstance(packs, (bytes, bytearray, str)):
        return sum(1 for entry in packs if isinstance(entry, Mapping))
    return 0


def _infer_manifest_index_wait_info(
    manifest: Mapping[str, Any] | ValidationIndex | Path | str,
    *,
    stage: str | None,
) -> tuple[Path, str] | None:
    if isinstance(manifest, ValidationIndex):
        sid = manifest.sid or "<unknown>"
        return manifest.index_path, sid

    if isinstance(manifest, Mapping):
        stage_paths = _sanitize_stage_paths(
            extract_stage_manifest_paths(manifest, stage), stage
        )
        is_index_document = _document_is_index_document(manifest)
        force_stage_only = _is_validation_stage(stage)
        use_manifest_paths = _should_use_manifest_paths() or force_stage_only
        try:
            index_path = _index_path_from_mapping(
                manifest,
                stage_paths=stage_paths,
                use_manifest_paths=use_manifest_paths,
                is_index_document=is_index_document,
                manifest_path=None,
                force_stage_paths_only=force_stage_only,
            )
        except ValidationPackError:
            return None
        sid = _coerce_str(manifest.get("sid")) or "<unknown>"
        return index_path, sid

    manifest_path = Path(manifest)
    try:
        text = manifest_path.read_text(encoding="utf-8")
        document = json.loads(text)
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(document, Mapping):
        return None

    stage_paths = _sanitize_stage_paths(
        extract_stage_manifest_paths(document, stage), stage
    )
    is_index_document = _document_is_index_document(document)
    force_stage_only = _is_validation_stage(stage)
    use_manifest_paths = _should_use_manifest_paths() or force_stage_only
    try:
        index_path = _index_path_from_mapping(
            document,
            stage_paths=stage_paths,
            use_manifest_paths=use_manifest_paths,
            is_index_document=is_index_document,
            manifest_path=manifest_path,
            force_stage_paths_only=force_stage_only,
        )
    except ValidationPackError:
        return None

    sid = _coerce_str(document.get("sid")) or "<unknown>"
    return index_path, sid


def _update_stage_status_after_send(index: ValidationIndex | None, stage: str) -> None:
    if index is None:
        return

    stage_key = stage.strip().lower()
    if stage_key not in {"merge", "validation"}:
        return

    try:
        run_root = index.index_path.parents[2]
    except IndexError:
        run_root = index.index_path.parent

    manifest_path = (run_root / "manifest.json").resolve()
    sid_hint = index.sid or manifest_path.parent.name

    try:
        manifest = RunManifest.load_or_create(manifest_path, sid_hint)
    except Exception:  # pragma: no cover - defensive logging
        log.debug(
            "VALIDATION_STAGE_STATUS_MANIFEST_LOAD_FAILED sid=%s path=%s stage=%s",
            sid_hint,
            manifest_path,
            stage_key,
            exc_info=True,
        )
        return

    stage_status = manifest.ensure_ai_stage_status(stage_key)
    stage_status["sent"] = True
    stage_status["completed_at"] = _utc_now()
    persist_manifest(manifest)


def _wait_for_index_file(index_path: Path, sid: str) -> bool:
    for i in range(_INDEX_FILE_WAIT_ATTEMPTS):
        try:
            if index_path.exists() and index_path.stat().st_size > _INDEX_FILE_MIN_SIZE:
                log.info("VALIDATION_INDEX_READY pairs=%d", _count_index_pairs(index_path))
                return True
        except OSError:
            pass
        log.info("VALIDATION_INDEX_WAIT attempt=%d", i + 1)
        time.sleep(_INDEX_FILE_WAIT_DELAY)
    log.warning("VALIDATION_NO_INDEX_FOUND sid=%s", sid)
    return False


def _is_validation_stage(stage: str | None) -> bool:
    if stage is None:
        return False
    return stage.strip().lower() == _MANIFEST_STAGE


def _stage_path_is_validation(path: Path | None) -> Path | None:
    if path is None:
        return None

    parts = [part.lower() for part in path.parts]
    if "merge" in parts:
        return None
    return path


def _safe_resolve(path: Path) -> Path:
    try:
        return path.resolve()
    except OSError:
        return path


def _manifest_sid(manifest: Mapping[str, Any]) -> str | None:
    for key in ("sid", "run_sid", "run_id"):
        value = manifest.get(key)
        if isinstance(value, str):
            text = value.strip()
            if text:
                return text
    return None


def _validation_base_from_hint(hint: Any) -> Path | None:
    if hint is None:
        return None
    try:
        candidate = Path(str(hint))
    except (TypeError, ValueError):
        return None

    if candidate.suffix:
        candidate = candidate.parent

    parts = [part.lower() for part in candidate.parts]
    for idx in range(len(parts) - 1, -1, -1):
        if parts[idx] == _MANIFEST_STAGE:
            return Path(*candidate.parts[: idx + 1])
    return None


def _default_validation_stage_paths_for_manifest(
    manifest: Mapping[str, Any]
) -> StageManifestPaths:
    hints: list[Any] = []
    for key in (
        "__stage_dir__",
        "__validation_dir__",
        "__manifest_dir__",
        "__base_dir__",
        "__index_dir__",
        "__index_path__",
    ):
        value = manifest.get(key)
        if value is not None:
            hints.append(value)

    for hint in hints:
        base = _validation_base_from_hint(hint)
        if base is not None:
            base_resolved = _safe_resolve(base)
            return StageManifestPaths(
                base_dir=base_resolved,
                packs_dir=_safe_resolve(base_resolved / "packs"),
                results_dir=_safe_resolve(base_resolved / "results"),
                index_file=_safe_resolve(base_resolved / "index.json"),
                log_file=_safe_resolve(base_resolved / "logs.txt"),
            )

    sid = _manifest_sid(manifest)
    if sid is None:
        raise ValidationPackError(
            "Validation manifest is missing a sid for validation paths",
        )

    base_dir = _safe_resolve(Path("runs") / sid / "ai_packs" / _MANIFEST_STAGE)
    return StageManifestPaths(
        base_dir=base_dir,
        packs_dir=_safe_resolve(base_dir / "packs"),
        results_dir=_safe_resolve(base_dir / "results"),
        index_file=_safe_resolve(base_dir / "index.json"),
        log_file=_safe_resolve(base_dir / "logs.txt"),
    )


def _resolve_validation_stage_paths(
    manifest: Mapping[str, Any],
    raw_stage_paths: StageManifestPaths | None,
) -> StageManifestPaths:
    sanitized = _sanitize_stage_paths(raw_stage_paths, _MANIFEST_STAGE)
    if sanitized is None:
        sanitized = StageManifestPaths()

    default: StageManifestPaths | None = None

    def _default() -> StageManifestPaths:
        nonlocal default
        if default is None:
            default = _default_validation_stage_paths_for_manifest(manifest)
        return default

    if sanitized.base_dir is None:
        defaults = _default()
        sanitized = replace(sanitized, base_dir=defaults.base_dir)

    if sanitized.packs_dir is None:
        defaults = _default()
        sanitized = replace(sanitized, packs_dir=defaults.packs_dir)

    if sanitized.results_dir is None:
        defaults = _default()
        sanitized = replace(sanitized, results_dir=defaults.results_dir)

    if sanitized.index_file is None:
        defaults = _default()
        sanitized = replace(sanitized, index_file=defaults.index_file)

    if sanitized.log_file is None:
        defaults = _default()
        sanitized = replace(sanitized, log_file=defaults.log_file)

    return sanitized


def _resolve_stage_paths(
    manifest: Mapping[str, Any],
    stage: str,
    is_index_document: bool,
) -> StageManifestPaths | None:
    raw_stage_paths = extract_stage_manifest_paths(manifest, stage)
    if _is_validation_stage(stage) and not is_index_document:
        return _resolve_validation_stage_paths(manifest, raw_stage_paths)
    return _sanitize_stage_paths(raw_stage_paths, stage)


def _infer_run_root_from_path(path: Path, stage: str | None) -> Path | None:
    stage_lower = stage.strip().lower() if stage else None
    parts = path.parts
    lowered = [part.lower() for part in parts]

    if "ai_packs" in lowered:
        idx = lowered.index("ai_packs")
        if idx > 0:
            return Path(*parts[:idx])

    if stage_lower and stage_lower in lowered:
        idx = lowered.index(stage_lower)
        if idx > 0:
            return Path(*parts[:idx])

    return None


def _infer_run_root(stage_paths: StageManifestPaths, stage: str | None) -> Path | None:
    candidates = (
        stage_paths.base_dir,
        stage_paths.packs_dir,
        stage_paths.results_dir,
        stage_paths.index_file,
        stage_paths.log_file,
    )
    for candidate in candidates:
        if candidate is None:
            continue
        run_root = _infer_run_root_from_path(candidate, stage)
        if run_root is not None:
            return run_root.resolve()
    return None


def _normalize_manifest_stage_path(
    path: Path | None, *, run_root: Path | None
) -> Path | None:
    if path is None:
        return None

    raw = sanitize_stage_path_value(path)
    if not raw:
        return None

    if run_root is not None:
        try:
            return normalize_stage_path(run_root, raw)
        except ValueError:
            pass

    candidate = Path(raw)
    try:
        return candidate.resolve()
    except OSError:
        return candidate


def _sanitize_stage_paths(
    stage_paths: StageManifestPaths | None, stage: str | None
) -> StageManifestPaths | None:
    if stage_paths is None or not _is_validation_stage(stage):
        if stage_paths is None:
            return None

        run_root = _infer_run_root(stage_paths, stage)
        return replace(
            stage_paths,
            base_dir=_normalize_manifest_stage_path(stage_paths.base_dir, run_root=run_root),
            packs_dir=_normalize_manifest_stage_path(stage_paths.packs_dir, run_root=run_root),
            results_dir=_normalize_manifest_stage_path(stage_paths.results_dir, run_root=run_root),
            index_file=_normalize_manifest_stage_path(stage_paths.index_file, run_root=run_root),
            log_file=_normalize_manifest_stage_path(stage_paths.log_file, run_root=run_root),
        )

    run_root = _infer_run_root(stage_paths, stage)
    normalized = replace(
        stage_paths,
        base_dir=_normalize_manifest_stage_path(stage_paths.base_dir, run_root=run_root),
        packs_dir=_normalize_manifest_stage_path(stage_paths.packs_dir, run_root=run_root),
        results_dir=_normalize_manifest_stage_path(stage_paths.results_dir, run_root=run_root),
        index_file=_normalize_manifest_stage_path(stage_paths.index_file, run_root=run_root),
        log_file=_normalize_manifest_stage_path(stage_paths.log_file, run_root=run_root),
    )

    sanitized = replace(
        normalized,
        base_dir=_stage_path_is_validation(normalized.base_dir),
        packs_dir=_stage_path_is_validation(normalized.packs_dir),
        results_dir=_stage_path_is_validation(normalized.results_dir),
        index_file=_stage_path_is_validation(normalized.index_file),
        log_file=_stage_path_is_validation(normalized.log_file),
    )
    return sanitized


def _index_path_from_stage_paths(stage_paths: StageManifestPaths | None) -> Path | None:
    if stage_paths is None:
        return None

    if stage_paths.index_file is not None:
        return stage_paths.index_file

    if stage_paths.base_dir is not None:
        return (stage_paths.base_dir / "index.json").resolve()

    return None


def _index_path_from_mapping(
    document: Mapping[str, Any],
    *,
    stage_paths: StageManifestPaths | None,
    use_manifest_paths: bool,
    is_index_document: bool,
    manifest_path: Path | None = None,
    force_stage_paths_only: bool = False,
) -> Path:
    stage_index_path = _index_path_from_stage_paths(stage_paths)
    if stage_index_path is not None:
        return stage_index_path

    if force_stage_paths_only:
        raise ValidationPackError(
            "Validation manifest is missing validation stage index path",
        )

    if use_manifest_paths and not is_index_document:
        raise ValidationPackError(
            "Validation manifest is missing validation stage paths",
        )

    index_path_override = document.get("__index_path__")
    if index_path_override:
        return Path(str(index_path_override)).resolve()

    manifest_resolved: Path | None = None
    if manifest_path is not None:
        try:
            manifest_resolved = manifest_path.resolve()
        except OSError:
            manifest_resolved = manifest_path

    if manifest_resolved is not None and is_index_document:
        return manifest_resolved

    base_dir_override = (
        document.get("__base_dir__")
        or document.get("__manifest_dir__")
        or document.get("__index_dir__")
    )
    if base_dir_override:
        base_dir = Path(str(base_dir_override)).resolve()
    elif manifest_resolved is not None:
        base_dir = manifest_resolved.parent.resolve()
    else:
        base_dir = Path.cwd()

    filename = _coerce_str(document.get("__index_filename__"), default="index.json")
    return (base_dir / filename).resolve()


def _index_from_document(document: Mapping[str, Any], *, index_path: Path) -> ValidationIndex:
    schema_version = _coerce_int(document.get("schema_version"), default=0)
    if schema_version < 2:
        document = _convert_v1_document(document, index_path=index_path)
        schema_version = 2

    sid = _coerce_str(document.get("sid"))
    if not sid:
        raise ValidationPackError("Validation manifest is missing 'sid'")

    root = _coerce_str(document.get("root"), default=".") or "."
    packs_dir = _coerce_str(document.get("packs_dir"), default="packs") or "packs"
    results_dir = _coerce_str(document.get("results_dir"), default="results") or "results"

    raw_packs = document.get("packs")
    records: list[ValidationPackRecord] = []
    if isinstance(raw_packs, Sequence):
        for entry in raw_packs:
            if isinstance(entry, Mapping):
                records.append(ValidationPackRecord.from_mapping(entry))

    return ValidationIndex(
        index_path=index_path,
        sid=sid,
        root=root,
        packs_dir=packs_dir,
        results_dir=results_dir,
        packs=records,
        schema_version=schema_version,
    )


def _convert_v1_document(
    document: Mapping[str, Any], *, index_path: Path
) -> Mapping[str, Any]:
    """Convert a legacy v1 manifest document to the v2 schema."""

    base_dir = index_path.parent.resolve()
    sid = _coerce_str(document.get("sid"))

    raw_items = document.get("packs") or document.get("items")
    records: list[dict[str, Any]] = []
    if isinstance(raw_items, Sequence):
        for entry in raw_items:
            if not isinstance(entry, Mapping):
                continue
            record: dict[str, Any] = dict(entry)
            record["pack"] = _to_relative(
                entry.get("pack_path")
                or entry.get("pack")
                or entry.get("pack_file")
                or entry.get("pack_filename"),
                base_dir,
            )
            record["result_jsonl"] = _to_relative(
                entry.get("result_jsonl_path")
                or entry.get("result_jsonl")
                or entry.get("result_jsonl_file"),
                base_dir,
            )
            record["result_json"] = _to_relative(
                entry.get("result_path")
                or entry.get("result_summary_path")
                or entry.get("result_json"),
                base_dir,
            )
            records.append(record)

    return {
        "schema_version": 2,
        "sid": sid,
        "root": ".",
        "packs_dir": "packs",
        "results_dir": "results",
        "packs": records,
    }


def _to_relative(path_value: Any, base_dir: Path) -> str:
    """Return a POSIX relative path for ``path_value`` with ``base_dir`` as the anchor."""

    text = _coerce_str(path_value)
    if not text:
        return ""

    candidate = Path(text)
    if not candidate.is_absolute():
        return PurePosixPath(text).as_posix()

    try:
        candidate_resolved = candidate.resolve()
    except OSError:
        candidate_resolved = candidate

    base_resolved = base_dir.resolve()
    try:
        relative = candidate_resolved.relative_to(base_resolved)
    except ValueError:
        try:
            relative = Path(os.path.relpath(candidate_resolved, base_resolved))
        except (OSError, ValueError):
            return candidate_resolved.as_posix()

    return PurePosixPath(relative).as_posix()


def _resolve_log_candidate(index_dir: Path, raw: Any) -> Path | None:
    text = _coerce_str(raw)
    if not text:
        return None

    candidate = Path(text)
    if candidate.is_absolute():
        try:
            return candidate.resolve()
        except OSError:
            return candidate

    relative = index_dir / PurePosixPath(text)
    try:
        return relative.resolve()
    except OSError:
        return relative


def _resolve_log_path(
    index_path: Path,
    document: Mapping[str, Any] | None,
    *,
    stage_paths: StageManifestPaths | None,
) -> Path:
    index_dir = index_path.parent.resolve()

    if stage_paths and stage_paths.log_file:
        return stage_paths.log_file

    env_candidate = _resolve_log_candidate(index_dir, os.getenv(_LOG_PATH_REL_ENV))

    if document:
        logs_section = document.get("logs")
        if isinstance(logs_section, Mapping):
            for key in ("send", "sender", "log", "log_path", "path"):
                value = logs_section.get(key)
                resolved = _resolve_log_candidate(index_dir, value)
                if resolved is not None:
                    return resolved

        fallback_document = document if isinstance(document, Mapping) else None
        if fallback_document:
            for key in ("log", "log_path"):
                value = fallback_document.get(key)
                resolved = _resolve_log_candidate(index_dir, value)
                if resolved is not None:
                    return resolved

    if env_candidate is not None:
        return env_candidate

    fallback = _resolve_log_candidate(index_dir, _DEFAULT_LOG_FILENAME)
    if fallback is not None:
        return fallback

    return index_dir / _DEFAULT_LOG_FILENAME


def _document_is_index_document(document: Mapping[str, Any]) -> bool:
    packs = document.get("packs")
    if isinstance(packs, Sequence) and not isinstance(packs, (str, bytes, bytearray)):
        return True
    items = document.get("items")
    if isinstance(items, Sequence) and not isinstance(items, (str, bytes, bytearray)):
        return True
    return False


def _should_use_manifest_paths() -> bool:
    raw = os.getenv(_USE_MANIFEST_PATHS_ENV)
    if raw is None:
        return True
    value = _coerce_bool_flag(raw)
    if not value:
        log.warning(
            "%s disabled via env=%s; manifest paths remain enforced",
            _USE_MANIFEST_PATHS_ENV,
            raw,
        )
    return True


def _wait_for_path(path: Path, *, attempts: int, delay: float) -> bool:
    if path.exists():
        return True
    for _ in range(max(0, attempts)):
        time.sleep(max(delay, 0.0))
        if path.exists():
            return True
    return False


def _looks_like_validation_packs_dir(path: Path) -> bool:
    parts = [part.lower() for part in path.parts]
    return any("validation" in part for part in parts) and (parts and parts[-1] == "packs")


def _validate_index_ready(
    index: ValidationIndex, stage_paths: StageManifestPaths | None
) -> bool:
    packs_dir_path = index.packs_dir_path

    expected_dir: Path | None = None
    if stage_paths and stage_paths.packs_dir:
        try:
            expected_dir = stage_paths.packs_dir.resolve()
        except OSError:
            expected_dir = stage_paths.packs_dir

    if expected_dir is not None and packs_dir_path != expected_dir:
        log.error(
            "VALIDATION_PACKS_DIR_MISMATCH sid=%s expected=%s actual=%s index=%s",
            index.sid or "<unknown>",
            str(expected_dir),
            str(packs_dir_path),
            str(index.index_path),
        )
        raise ValidationPackError(
            "Validation index packs directory does not match manifest",
        )

    if expected_dir is None and not _looks_like_validation_packs_dir(packs_dir_path):
        log.error(
            "VALIDATION_PACKS_DIR_UNEXPECTED sid=%s packs_dir=%s index=%s",
            index.sid or "<unknown>",
            str(packs_dir_path),
            str(index.index_path),
        )
        raise ValidationPackError(
            "Validation index does not reference validation packs",
        )

    if not packs_dir_path.exists():
        log.error(
            "VALIDATION_PACKS_DIR_MISSING sid=%s packs_dir=%s index=%s",
            index.sid or "<unknown>",
            str(packs_dir_path),
            str(index.index_path),
        )
        raise ValidationPackError("Validation packs directory is missing")

    eligible_files = [
        path
        for path in packs_dir_path.glob("val_acc_*.jsonl")
        if path.is_file()
    ]

    if not eligible_files:
        log.info(
            "VALIDATION_NO_PACKS_ELIGIBLE sid=%s packs_dir=%s index=%s",
            index.sid or "<unknown>",
            str(packs_dir_path),
            str(index.index_path),
        )
        return False

    return True


def _prepare_validation_index(
    manifest: Mapping[str, Any] | ValidationIndex | Path | str,
    *,
    stage: str,
) -> _IndexPreparationResult:
    if isinstance(manifest, ValidationIndex):
        view = _ManifestView(
            index=manifest,
            log_path=_resolve_log_path(manifest.index_path, None, stage_paths=None),
            stage_paths=None,
        )
    else:
        view = _load_manifest_view(manifest, stage=stage)

    index = view.index
    stage_paths = view.stage_paths

    if not _validate_index_ready(index, stage_paths):
        return _IndexPreparationResult(index=index, view=view, skip=True)

    return _IndexPreparationResult(index=index, view=view, skip=False)


def _load_manifest_view(
    manifest: Mapping[str, Any] | ValidationIndex | Path | str,
    *,
    stage: str,
) -> _ManifestView:
    if isinstance(manifest, ValidationIndex):
        index = manifest
        log_path = _resolve_log_path(index.index_path, None, stage_paths=None)
        return _ManifestView(index=index, log_path=log_path, stage_paths=None)

    if isinstance(manifest, Mapping):
        is_index_document = _document_is_index_document(manifest)
        stage_paths = _resolve_stage_paths(manifest, stage, is_index_document)
        force_stage_only = _is_validation_stage(stage) and not is_index_document
        use_manifest_paths = _should_use_manifest_paths() or force_stage_only

        if not is_index_document:
            if stage_paths is None or not stage_paths.has_any():
                raise ValidationPackError(
                    "Validation manifest is missing validation stage paths",
                )
        index_path = _index_path_from_mapping(
            manifest,
            stage_paths=stage_paths,
            use_manifest_paths=use_manifest_paths,
            is_index_document=is_index_document,
            manifest_path=None,
            force_stage_paths_only=force_stage_only,
        )

        if is_index_document:
            index = _index_from_document(manifest, index_path=index_path)
        else:
            if not index_path.exists() and not _wait_for_path(
                index_path,
                attempts=_INDEX_WAIT_ATTEMPTS,
                delay=_INDEX_WAIT_DELAY,
            ):
                raise ValidationPackError(
                    f"Validation index missing: {index_path}",
                )
            index = load_validation_index(index_path)

        log_path = _resolve_log_path(index_path, manifest, stage_paths=stage_paths)
        return _ManifestView(index=index, log_path=log_path, stage_paths=stage_paths)

    manifest_path = Path(manifest)
    use_manifest_paths = _should_use_manifest_paths()

    for attempt in range(max(1, _MANIFEST_RETRY_ATTEMPTS)):
        try:
            text = manifest_path.read_text(encoding="utf-8")
        except FileNotFoundError as exc:
            if attempt < _MANIFEST_RETRY_ATTEMPTS - 1:
                time.sleep(_MANIFEST_RETRY_DELAY)
                continue
            raise ValidationPackError(
                f"Validation index missing: {manifest_path}"
            ) from exc
        except OSError as exc:
            raise ValidationPackError(
                f"Unable to read validation index: {manifest_path}"
            ) from exc

        try:
            document = json.loads(text)
        except json.JSONDecodeError:
            if attempt < _MANIFEST_RETRY_ATTEMPTS - 1:
                time.sleep(_MANIFEST_RETRY_DELAY)
                continue
            raise ValidationPackError(
                f"Validation index is not valid JSON: {manifest_path}"
            )

        if not isinstance(document, Mapping):
            raise ValidationPackError("Validation index root must be an object")

        is_index_document = _document_is_index_document(document)
        stage_paths = _resolve_stage_paths(document, stage, is_index_document)
        force_stage_only = _is_validation_stage(stage) and not is_index_document
        if not is_index_document:
            if stage_paths is None or not stage_paths.has_any():
                if attempt < _MANIFEST_RETRY_ATTEMPTS - 1:
                    log.info(
                        "VALIDATION_MANIFEST_STAGE_WAIT stage=%s attempt=%s path=%s",
                        stage,
                        attempt + 1,
                        str(manifest_path),
                    )
                    time.sleep(_MANIFEST_RETRY_DELAY)
                    continue
                raise ValidationPackError(
                    "Validation manifest is missing validation stage paths",
                )
        index_path = _index_path_from_mapping(
            document,
            stage_paths=stage_paths,
            use_manifest_paths=use_manifest_paths or force_stage_only,
            is_index_document=is_index_document,
            manifest_path=manifest_path,
            force_stage_paths_only=force_stage_only,
        )

        if is_index_document:
            index = _index_from_document(document, index_path=index_path)
        else:
            if not index_path.exists() and not _wait_for_path(
                index_path,
                attempts=_INDEX_WAIT_ATTEMPTS,
                delay=_INDEX_WAIT_DELAY,
            ):
                if attempt < _MANIFEST_RETRY_ATTEMPTS - 1:
                    time.sleep(_MANIFEST_RETRY_DELAY)
                    continue
                raise ValidationPackError(
                    f"Validation index missing: {index_path}",
                )
            index = load_validation_index(index_path)

        log_path = _resolve_log_path(index_path, document, stage_paths=stage_paths)
        return _ManifestView(index=index, log_path=log_path, stage_paths=stage_paths)

    raise ValidationPackError(
        f"Unable to resolve validation manifest: {manifest_path}",
    )


class ValidationPackSender:
    """Send validation packs and store the adjudication results."""

    def __init__(
        self,
        manifest: Mapping[str, Any] | ValidationIndex | Path | str,
        *,
        http_client: _ChatCompletionClient | None = None,
        stage: str | None = None,
        preloaded_view: _ManifestView | None = None,
    ) -> None:
        resolved_stage = stage or _MANIFEST_STAGE
        if preloaded_view is not None:
            view = preloaded_view
        else:
            view = _load_manifest_view(manifest, stage=resolved_stage)
        self._index = view.index
        self.sid = self._index.sid
        self._stage = resolved_stage

        raw_model = os.getenv("AI_MODEL")
        if raw_model is None or not str(raw_model).strip():
            fallback_model = os.getenv("VALIDATION_MODEL")
            if fallback_model is not None:
                raw_model = fallback_model
        if raw_model is None:
            raw_model = _DEFAULT_MODEL
        self.model = str(raw_model).strip()
        self._client = http_client or self._build_client()
        self._throttle = _THROTTLE_SECONDS
        self._results_root: Path | None = None
        self._log_path = view.log_path
        self._debug = _coerce_bool_flag(os.getenv(_DEBUG_ENV))
        self._confidence_threshold = _confidence_threshold()
        self._default_queue = (
            self._infer_queue_hint(self._index.packs) or _DEFAULT_QUEUE_NAME
        )
        self._request_group_size = _resolve_request_group_size()
        envelope_flag = _coerce_bool_flag(os.getenv(_WRITE_JSON_ENVELOPE_ENV))
        if envelope_flag:
            log.info(
                "VALIDATION_JSON_ENVELOPE_DISABLED env_flag=true -> forcing jsonl-only"
            )
        self._write_json_envelope = validation_write_json_enabled()
        self._runs_root = self._infer_runs_root()
        self._index_writer: ValidationPackIndexWriter | None = None
        self._stage_promotion_logged = False

    def _log_run_summary(
        self,
        *,
        packs: int,
        sent: int,
        results_written: int,
        skipped_existing: int,
        errors: int,
    ) -> None:
        """Emit a normalized summary log line for the current send run."""

        log.info(
            "VALIDATION_SEND_SUMMARY sid=%s packs=%s sent=%s results_written=%s "
            "skipped_existing=%s errors=%s",
            self.sid,
            packs,
            sent,
            results_written,
            skipped_existing,
            errors,
        )

    def _await_index_ready(self) -> ValidationIndex:
        index = self._load_index()
        index_path = index.index_path

        def _ready() -> bool:
            try:
                return index_path.exists() and index_path.stat().st_size > 0
            except OSError:
                return False

        if not _ready():
            for attempt in range(1, _INDEX_READY_ATTEMPTS + 1):
                log.info("VALIDATION_INDEX_WAIT attempt=%s", attempt)
                time.sleep(
                    random.uniform(_INDEX_READY_MIN_DELAY, _INDEX_READY_MAX_DELAY)
                )
                if _ready():
                    break

            if not _ready():
                raise ValidationPackError(
                    f"Validation index missing or empty: {index_path}",
                )

            try:
                refreshed = load_validation_index(index_path)
            except Exception as exc:  # pragma: no cover - defensive
                raise ValidationPackError(
                    f"Validation index load failed after wait: {index_path}"
                ) from exc
            self._index = refreshed
            index = refreshed

        log.info("VALIDATION_INDEX_READY count=%s", len(index.packs))
        return index

    # ------------------------------------------------------------------
    # Queue routing helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _queue_from_record(record: ValidationPackRecord) -> str | None:
        extra = record.extra
        if isinstance(extra, Mapping):
            for key in ("queue", "celery_queue", "task_queue", "target_queue"):
                value = extra.get(key)
                if isinstance(value, str):
                    text = value.strip()
                    if text:
                        return text
        return None

    def _infer_queue_hint(
        self, records: Sequence[ValidationPackRecord]
    ) -> str | None:
        counts: dict[str, int] = {}
        for record in records:
            queue = self._queue_from_record(record)
            if not queue:
                continue
            counts[queue] = counts.get(queue, 0) + 1
        if not counts:
            return None
        return max(counts.items(), key=lambda item: (item[1], item[0]))[0]

    def _queue_plan(self, accounts: Sequence[_PreflightAccount]) -> dict[str, int]:
        plan: dict[str, int] = {}
        for account in accounts:
            queue = self._queue_from_record(account.record) or self._default_queue
            plan[queue] = plan.get(queue, 0) + 1
        return plan

    def _select_queue(self, plan: Mapping[str, int]) -> str:
        if not plan:
            return self._default_queue
        if self._default_queue in plan:
            return self._default_queue
        return max(plan.items(), key=lambda item: (item[1], item[0]))[0]

    # ------------------------------------------------------------------
    # Pre-flight validation
    # ------------------------------------------------------------------
    def _preflight(self, index: ValidationIndex) -> _PreflightSummary:
        manifest_path = index.index_path
        accounts: list[_PreflightAccount] = []

        results_dir = index.results_dir_path
        results_dir_exists = results_dir.exists()
        if not results_dir_exists:
            results_dir.mkdir(parents=True, exist_ok=True)

        created_parent_dirs: set[Path] = set()

        for record in index.packs:
            pack_path = index.resolve_pack_path(record)
            try:
                result_jsonl_path = index.resolve_result_jsonl_path(record)
            except ValueError as exc:
                try:
                    normalized_account = int(record.account_id)
                    expected_filename = validation_result_jsonl_filename_for_account(
                        normalized_account
                    )
                except (TypeError, ValueError):
                    normalized_account = record.account_id
                    suffix = (
                        str(record.account_id).strip()
                        if record.account_id is not None
                        else "missing"
                    )
                    try:
                        expected_filename = validation_result_jsonl_filename_for_account(
                            suffix
                        )
                    except Exception:
                        expected_filename = f"acc_{suffix}.result.jsonl"

                expected_relative = Path(index.results_dir) / expected_filename
                expected_absolute = (index.root_dir / expected_relative).resolve()
                account_display = (
                    f"{normalized_account}" if normalized_account is not None else "<unknown>"
                )
                raise ValueError(
                    "Validation preflight could not resolve .result.jsonl path for "
                    f"account_id={account_display}: expected {expected_absolute}"
                ) from exc
            result_json_path = index.resolve_result_json_path(record)

            for candidate in (result_jsonl_path.parent, result_json_path.parent):
                if candidate.exists():
                    continue
                candidate.mkdir(parents=True, exist_ok=True)
                created_parent_dirs.add(candidate.resolve())

            accounts.append(
                _PreflightAccount(
                    record=record,
                    pack_path=pack_path,
                    result_jsonl_path=result_jsonl_path,
                    result_json_path=result_json_path,
                    pack_missing=not pack_path.is_file(),
                )
            )

        missing = sum(1 for account in accounts if account.pack_missing)
        summary = _PreflightSummary(
            manifest_path=manifest_path,
            accounts=tuple(accounts),
            missing=missing,
            results_dir_created=not results_dir_exists,
            parent_dirs_created=len(created_parent_dirs),
        )
        self._print_preflight_summary(index, summary)
        return summary

    def _print_preflight_summary(
        self, index: ValidationIndex, summary: _PreflightSummary
    ) -> None:
        manifest_display = self._display_path(summary.manifest_path)
        print(f"MANIFEST: {manifest_display}")
        print(f"PACKS: {len(summary.accounts)}, missing: {summary.missing}")

        results_status = "ok"
        if summary.results_dir_created:
            results_status = "created"
        elif summary.parent_dirs_created:
            results_status = f"ok (created {summary.parent_dirs_created} dirs)"
        print(f"RESULTS DIR: {results_status}")

        missing_accounts = {
            account.record.account_id for account in summary.accounts if account.pack_missing
        }
        for account in summary.accounts:
            record = account.record
            account_id = record.account_id
            pack_display = record.pack or self._display_path(
                account.pack_path, base=index.index_dir
            )
            jsonl_display = record.result_jsonl or self._display_path(
                account.result_jsonl_path, base=index.index_dir
            )
            summary_display = record.result_json or self._display_path(
                account.result_json_path, base=index.index_dir
            )

            line = (
                f"[acc={account_id:03d}] pack={pack_display} -> "
                f"{jsonl_display}, {summary_display}  (lines={record.lines})"
            )
            if account_id in missing_accounts:
                missing_path = self._display_path(
                    account.pack_path, base=index.index_dir
                )
                line += f"  [MISSING: {missing_path}]"
            print(line)

    @staticmethod
    def _display_path(path: Path, *, base: Path | None = None) -> str:
        candidate = path.resolve()
        anchors: list[Path] = []
        if base is not None:
            anchors.append(base.resolve())
        anchors.append(Path.cwd().resolve())
        for anchor in anchors:
            try:
                relative = candidate.relative_to(anchor)
                return PurePosixPath(relative).as_posix() or "."
            except ValueError:
                try:
                    relpath = Path(os.path.relpath(candidate, anchor))
                except (OSError, ValueError):
                    continue
                return PurePosixPath(relpath).as_posix()
        return candidate.as_posix()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def _log_preflight_line(self, index: ValidationIndex) -> None:
        packs_dir = index.packs_dir_path
        try:
            pack_count = sum(1 for _ in packs_dir.glob("val_acc_*.jsonl")) if packs_dir.exists() else 0
        except OSError as exc:  # pragma: no cover - defensive logging
            log.warning(
                "VALIDATION_SEND_PREFLIGHT_GLOB_FAILED sid=%s packs_dir=%s error=%s",
                self.sid,
                str(packs_dir),
                exc,
            )
            pack_count = 0

        api_key_set = bool(os.getenv("OPENAI_API_KEY"))
        base_url_set = bool(os.getenv("OPENAI_BASE_URL"))

        log.info(
            "VALIDATION_SEND_PREFLIGHT sid=%s model=%s packs_dir=%s pack_files=%s env_api_key=%s env_base_url=%s",
            self.sid,
            self.model or "<missing>",
            str(packs_dir),
            pack_count,
            "set" if api_key_set else "missing",
            "set" if base_url_set else "missing",
        )

        log.info(
            "VALIDATION_PACKS_DIR_USED=%s sid=%s",
            str(index.packs_dir_path),
            self.sid,
        )
        log.info(
            "VALIDATION_RESULTS_DIR_USED=%s sid=%s",
            str(index.results_dir_path),
            self.sid,
        )

    def send(self) -> list[dict[str, Any]]:
        """Send every pack referenced by the manifest index."""

        log.info(
            "VALIDATION_STAGE_STARTED sid=%s stage=%s",
            self.sid,
            self._stage,
        )
        index = self._await_index_ready()
        self._log_preflight_line(index)

        if not self.model:
            log.error(
                "VALIDATION_SEND_MODEL_MISSING sid=%s manifest=%s detail=%s",
                self.sid,
                str(index.index_path),
                "no model configured",
            )
            self._log_run_summary(
                packs=0,
                sent=0,
                results_written=0,
                skipped_existing=0,
                errors=1,
            )
            return []

        if not index.packs:
            log.warning(
                "VALIDATION_SEND_NO_PACKS sid=%s manifest=%s detail=%s",
                self.sid,
                str(index.index_path),
                "no eligible packs found",
            )
            self._log_run_summary(
                packs=0,
                sent=0,
                results_written=0,
                skipped_existing=0,
                errors=0,
            )
            return []

        preflight = self._preflight(index)
        self._results_root = index.results_dir_path

        log.info(
            "VALIDATION_SEND_DISCOVERY sid=%s manifest=%s packs=%s missing=%s results_dir=%s",
            self.sid,
            str(index.index_path),
            len(preflight.accounts),
            preflight.missing,
            str(index.results_dir_path),
        )

        dispatchable_accounts = [
            account for account in preflight.accounts if not account.pack_missing
        ]
        queue_plan = self._queue_plan(dispatchable_accounts)
        total_enqueued = sum(queue_plan.values())
        missing_accounts = sum(1 for account in preflight.accounts if account.pack_missing)
        default_queue = self._select_queue(queue_plan)
        queue_plan_display = ", ".join(
            f"{name}:{count}" for name, count in sorted(queue_plan.items())
        )
        if not queue_plan_display:
            queue_plan_display = "none"

        if total_enqueued:
            log.info(
                "VALIDATION_SEND_QUEUE_PLAN sid=%s queue=%s routed=%s missing=%s routes=%s",
                self.sid,
                default_queue,
                total_enqueued,
                missing_accounts,
                queue_plan_display,
            )
        else:
            log.warning(
                "VALIDATION_SEND_QUEUE_PLAN sid=%s queue=%s routed=%s missing=%s routes=%s",
                self.sid,
                default_queue,
                total_enqueued,
                missing_accounts,
                queue_plan_display,
            )

        self._log(
            "send_queue_plan",
            queue=default_queue,
            total_accounts=len(preflight.accounts),
            missing_accounts=missing_accounts,
            routed_accounts=total_enqueued,
            routes=dict(queue_plan),
        )

        packs_total = 0
        sent_count = 0
        results_written = 0
        skipped_existing = 0
        error_count = 0
        missing_results_accounts: list[str] = []

        results: list[dict[str, Any]] = []
        for account in preflight.accounts:
            record = account.record
            account_id = record.account_id
            try:
                normalized_account = self._normalize_account_id(account_id)
            except ValueError:
                normalized_account = None

            pack_relative = record.pack
            result_jsonl_relative = record.result_jsonl
            result_json_relative = record.result_json

            resolved_pack = account.pack_path
            result_jsonl_path = account.result_jsonl_path
            result_json_path = account.result_json_path

            account_label = (
                f"{normalized_account:03d}"
                if normalized_account is not None
                else str(account_id)
            )

            if account.pack_missing:
                missing_display = pack_relative or self._display_path(
                    resolved_pack, base=index.index_dir
                )
                error_message = f"Pack file missing: {missing_display}"
                log.warning(
                    "VALIDATION_PACK_MISSING sid=%s account_id=%s pack=%s",
                    self.sid,
                    account_label,
                    missing_display,
                )
                log.error(
                    "VALIDATION_SEND_ACCOUNT_FAILED sid=%s account_id=%s pack=%s exc_type=%s message=%s line_ids=%s",
                    self.sid,
                    account_label,
                    str(resolved_pack),
                    "FileNotFoundError",
                    error_message,
                    [],
                )
                if normalized_account is None:
                    self._log(
                        "send_account_failed",
                        account_id=str(account_id),
                        error=error_message,
                    )
                    continue

                self._log(
                    "send_account_failed",
                    account_id=f"{normalized_account:03d}",
                    error=error_message,
                )
                continue

            pack_exists = resolved_pack.exists()
            if not pack_exists:
                missing_display = pack_relative or self._display_path(
                    resolved_pack, base=index.index_dir
                )
                error_message = f"Pack file missing: {missing_display}"
                log.warning(
                    "VALIDATION_PACK_MISSING sid=%s account_id=%s pack=%s",
                    self.sid,
                    account_label,
                    missing_display,
                )
                log.error(
                    "VALIDATION_SEND_ACCOUNT_FAILED sid=%s account_id=%s pack=%s exc_type=%s message=%s line_ids=%s",
                    self.sid,
                    account_label,
                    str(resolved_pack),
                    "FileNotFoundError",
                    error_message,
                    [],
                )
                if normalized_account is None:
                    self._log(
                        "send_account_failed",
                        account_id=str(account_id),
                        error=error_message,
                    )
                else:
                    self._log(
                        "send_account_failed",
                        account_id=f"{normalized_account:03d}",
                        error=error_message,
                    )
                error_count += 1
                continue

            packs_total += 1

            if self._write_json_envelope:
                result_display = (
                    result_json_relative
                    or self._display_path(result_json_path, base=index.index_dir)
                )
            else:
                result_display = (
                    result_jsonl_relative
                    or self._display_path(result_jsonl_path, base=index.index_dir)
                )

            log.info(
                "PROCESSING account_id=%s pack=%s result_json=%s",
                account_label,
                pack_relative
                or self._display_path(resolved_pack, base=index.index_dir),
                result_display,
            )

            account_had_error = False
            try:
                account_summary = self._process_account(
                    account_id,
                    normalized_account,
                    resolved_pack,
                    pack_relative,
                    result_jsonl_path,
                    result_jsonl_relative,
                    result_json_path,
                    result_json_relative,
                )
                account_had_error = (
                    str(account_summary.get("status") or "").lower() == "error"
                )
            except ValidationPackError as exc:
                failed_line_ids = getattr(exc, "line_ids", []) or []
                log.error(
                    "VALIDATION_SEND_ACCOUNT_FAILED sid=%s account_id=%s pack=%s exc_type=%s message=%s line_ids=%s",
                    self.sid,
                    account_label,
                    str(resolved_pack),
                    type(exc).__name__,
                    exc,
                    failed_line_ids,
                )
                if normalized_account is None:
                    self._log(
                        "send_account_failed",
                        account_id=str(account_id),
                        error=str(exc),
                    )
                    continue
                self._log(
                    "send_account_failed",
                    account_id=f"{normalized_account:03d}",
                    error=str(exc),
                )
                account_summary = self._record_account_failure(
                    normalized_account,
                    resolved_pack,
                    pack_relative,
                    result_jsonl_path,
                    result_jsonl_relative,
                    result_json_path,
                    result_json_relative,
                    str(exc),
                )
                account_had_error = True
            except Exception as exc:  # pragma: no cover - defensive logging
                failed_line_ids = getattr(exc, "line_ids", []) or []
                log.error(
                    "VALIDATION_SEND_ACCOUNT_FAILED sid=%s account_id=%s pack=%s exc_type=%s message=%s line_ids=%s",
                    self.sid,
                    account_label,
                    str(resolved_pack),
                    type(exc).__name__,
                    exc,
                    failed_line_ids,
                )
                log.exception(
                    "VALIDATION_PACK_ACCOUNT_UNEXPECTED sid=%s account=%s line_ids=%s",
                    self.sid,
                    account_label,
                    failed_line_ids,
                )
                if normalized_account is None:
                    self._log(
                        "send_account_failed",
                        account_id=str(account_id),
                        error=str(exc),
                    )
                    continue
                self._log(
                    "send_account_failed",
                    account_id=f"{normalized_account:03d}",
                    error=str(exc),
                )
                account_summary = self._record_account_failure(
                    normalized_account,
                    resolved_pack,
                    pack_relative,
                    result_jsonl_path,
                    result_jsonl_relative,
                    result_json_path,
                    result_json_relative,
                    str(exc),
                )
                account_had_error = True
            results.append(account_summary)

            status = str(account_summary.get("status") or "").lower()
            if status == "skipped":
                skipped_existing += 1
            else:
                sent_count += 1

            if account_had_error:
                error_count += 1

            canonical_result_path = _canonical_result_path(result_jsonl_path)
            if canonical_result_path.is_file():
                results_written += 1
            else:
                tmp_path = canonical_result_path.with_name(
                    canonical_result_path.name + ".tmp"
                )
                label = (
                    f"{normalized_account:03d}"
                    if normalized_account is not None
                    else str(account_id)
                )
                log.error(
                    "VALIDATION_RESULTS_MISSING sid=%s account_id=%s pack=%s result=%s tmp_exists=%s",
                    self.sid,
                    label,
                    str(resolved_pack),
                    str(canonical_result_path),
                    tmp_path.exists(),
                )
                missing_results_accounts.append(label)
                if not account_had_error:
                    error_count += 1

        if missing_results_accounts:
            log.error(
                "VALIDATION_RESULTS_GAPS sid=%s accounts=%s",
                self.sid,
                ",".join(sorted(dict.fromkeys(missing_results_accounts))),
            )

        self._log_run_summary(
            packs=packs_total,
            sent=sent_count,
            results_written=results_written,
            skipped_existing=skipped_existing,
            errors=error_count,
        )
        completed_accounts = results_written + skipped_existing
        pending_accounts = packs_total - completed_accounts - error_count
        record_validation_results_summary(
            self.sid,
            results_total=packs_total,
            completed=completed_accounts,
            failed=error_count,
            pending=pending_accounts,
        )
        self._refresh_validation_progress()
        return results

    # ------------------------------------------------------------------
    # Account processing
    # ------------------------------------------------------------------
    def _collect_expected_line_info(
        self,
        account_id: int,
        pack_lines: Sequence[Mapping[str, Any]],
    ) -> list[tuple[str, str]]:
        expected: list[tuple[str, str]] = []
        for idx, pack_line in enumerate(pack_lines, start=1):
            field_name = self._coerce_field_name(pack_line, idx)
            if not self._is_allowed_field(field_name):
                continue
            identifier = self._coerce_identifier(account_id, idx, pack_line.get("id"))
            expected.append((identifier, field_name))
        return expected

    def _load_existing_result_lines(
        self, result_path: Path
    ) -> list[Mapping[str, Any]] | None:
        try:
            if not result_path.is_file():
                return None
        except OSError:
            return None

        try:
            existing_lines = read_jsonl(result_path)
        except (OSError, json.JSONDecodeError) as exc:
            log.warning(
                "VALIDATION_EXISTING_RESULTS_LOAD_FAILED sid=%s path=%s error=%s",
                self.sid,
                str(result_path),
                exc,
            )
            return None

        normalized: list[Mapping[str, Any]] = []
        for line in existing_lines:
            if not isinstance(line, Mapping):
                return None
            normalized.append(dict(line))
        return normalized

    def _existing_results_complete(
        self,
        result_jsonl_path: Path,
        result_summary_path: Path,
        expected_lines: Sequence[tuple[str, str]],
    ) -> list[Mapping[str, Any]] | None:
        tmp_path = result_jsonl_path.with_name(result_jsonl_path.name + ".tmp")
        if tmp_path.exists():
            return None

        if self._write_json_envelope:
            summary_tmp = result_summary_path.with_name(
                result_summary_path.name + ".tmp"
            )
            if summary_tmp.exists():
                return None

        existing = self._load_existing_result_lines(result_jsonl_path)
        if existing is None:
            return None

        if len(existing) != len(expected_lines):
            return None

        for (expected_id, _), payload in zip(expected_lines, existing):
            actual_id = str(payload.get("id") or "").strip()
            if actual_id != expected_id:
                return None

        if self._write_json_envelope:
            try:
                if not result_summary_path.is_file():
                    return None
            except OSError:
                return None

        return [dict(line) for line in existing]

    def _build_cached_summary(
        self,
        account_id: int,
        pack_path: Path,
        pack_display: str,
        result_jsonl_path: Path,
        result_jsonl_display: str,
        result_target_path: Path,
        result_target_display: str,
        pack_lines: Sequence[Mapping[str, Any]],
        expected_lines: Sequence[tuple[str, str]],
        cached_results: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        total_fields = len(pack_lines)
        fields_sent = len(cached_results)
        fields_skipped = max(total_fields - fields_sent, 0)
        conditional_sent = sum(
            1 for _, field in expected_lines if field in _CONDITIONAL_FIELDS
        )

        decision_metrics = _empty_decision_metrics()
        for (_, field_name), line in zip(expected_lines, cached_results):
            bucket = "conditional" if field_name in _CONDITIONAL_FIELDS else "non_conditional"
            decision = self._normalize_decision(line.get("decision"))
            if decision not in decision_metrics:
                decision_metrics[decision] = {"conditional": 0, "non_conditional": 0}
            decision_metrics[decision][bucket] += 1

        metrics_payload = {
            "total_fields": total_fields,
            "fields_sent": fields_sent,
            "fields_skipped": fields_skipped,
            "conditional_fields_sent": conditional_sent,
            "decision_counts": decision_metrics,
            "model_requests": 0,
        }

        cached_lines = [dict(line) for line in cached_results]

        summary_payload: dict[str, Any] = {
            "sid": self.sid,
            "account_id": account_id,
            "pack_path": str(pack_path),
            "pack_manifest_path": pack_display,
            "results_path": str(result_target_path),
            "results_manifest_path": result_target_display,
            "jsonl_path": str(result_jsonl_path),
            "jsonl_manifest_path": result_jsonl_display,
            "status": "skipped",
            "model": self.model,
            "request_lines": 0,
            "results": cached_lines,
            "completed_at": _utc_now(),
            "metrics": metrics_payload,
        }
        return summary_payload

    def _process_account(
        self,
        account_id: Any,
        normalized_account: int | None,
        pack_path: Path,
        pack_display: str,
        result_jsonl_path: Path,
        result_jsonl_display: str,
        result_summary_path: Path,
        result_summary_display: str,
    ) -> dict[str, Any]:
        account_int = normalized_account
        if account_int is None:
            raise ValidationPackError(f"Account id is not numeric: {account_id!r}")

        pack_identifier = f"acc_{account_int:03d}"
        error_filename = validation_result_error_filename_for_account(account_int)
        error_path = result_summary_path.with_name(error_filename)
        self._clear_error_sidecar(error_path, pack_id=pack_identifier)

        if self._write_json_envelope:
            result_target_path = result_summary_path
            result_target_display = result_summary_display
        else:
            result_target_path = result_jsonl_path
            result_target_display = result_jsonl_display

        if not result_target_display:
            fallback_display = (
                result_jsonl_display
                if result_target_path == result_jsonl_path
                else result_summary_display
            )
            result_target_display = fallback_display or self._display_path(
                result_target_path
            )

        try:
            pack_lines = list(
                self._iter_pack_lines(pack_path, display_path=pack_display)
            )
        except ValidationPackError:
            self._log(
                "send_account_start",
                account_id=f"{account_int:03d}",
                pack=pack_display,
                pack_absolute=str(pack_path),
                lines=0,
            )
            raise

        expected_lines = self._collect_expected_line_info(account_int, pack_lines)
        expected_count = len(expected_lines)
        cached_results = self._existing_results_complete(
            result_jsonl_path,
            result_target_path,
            expected_lines,
        )
        if cached_results is not None:
            summary_payload = self._build_cached_summary(
                account_int,
                pack_path,
                pack_display,
                result_jsonl_path,
                result_jsonl_display,
                result_target_path,
                result_target_display,
                pack_lines,
                expected_lines,
                cached_results,
            )
            log.info(
                "VALIDATION_SEND_ACCOUNT_SKIP sid=%s account_id=%03d pack=%s results=%s",
                self.sid,
                account_int,
                str(pack_path),
                len(cached_results),
            )
            self._log(
                "send_account_skipped",
                account_id=f"{account_int:03d}",
                pack=pack_display,
                pack_absolute=str(pack_path),
                results=len(cached_results),
                reason="existing_results",
            )
            # Update index to reflect that results exist, even though we skipped sending
            self._record_index_result(
                pack_path=pack_path,
                status="skipped",
                error=None,
                request_lines=0,
                result_path=result_jsonl_path,
                line_count=len(cached_results),
                completed_at=summary_payload.get("completed_at"),
            )
            return summary_payload

        log.info(
            "VALIDATION_SEND_ACCOUNT_START sid=%s account_id=%03d pack=%s lines=%s results=%s",
            self.sid,
            account_int,
            str(pack_path),
            len(pack_lines),
            str(result_target_path),
        )

        result_lines: list[dict[str, Any]] = []
        errors: list[str] = []
        total_fields = len(pack_lines)
        fields_sent = 0
        conditional_sent = 0
        decision_metrics = _empty_decision_metrics()
        failed_line_ids: list[str] = []
        current_line_id: str | None = None
        start_time = time.monotonic()
        model_requests = 0

        self._log(
            "send_account_start",
            account_id=f"{account_int:03d}",
            pack=pack_display,
            pack_absolute=str(pack_path),
            lines=len(pack_lines),
        )

        structured_retry_limit = _validation_max_retries()

        try:
            for idx, pack_line in enumerate(pack_lines, start=1):
                field_name = self._coerce_field_name(pack_line, idx)
                current_line_id = self._coerce_identifier(
                    account_int, idx, pack_line.get("id")
                )
                if not self._is_allowed_field(field_name):
                    self._log(
                        "send_line_skipped",
                        account_id=f"{account_int:03d}",
                        line_number=idx,
                        field=field_name,
                        reason="field_not_allowed",
                    )
                    continue
                raw_send_flag = pack_line.get("send_to_ai")
                if isinstance(raw_send_flag, bool):
                    send_flag = raw_send_flag
                elif raw_send_flag is None:
                    send_flag = None
                else:
                    send_flag = _coerce_bool_flag(raw_send_flag)

                prevalidated: tuple[dict[str, Any] | None, list[str], str] | None = None

                if send_flag is False:
                    response = self._build_deterministic_response(
                        pack_line,
                        account_id=account_int,
                        line_number=idx,
                        line_id=current_line_id,
                    )
                else:
                    validation_attempt = 0
                    try:
                        while True:
                            model_requests += 1
                            response = self._call_model(
                                pack_line,
                                account_id=account_int,
                                account_label=f"{account_int:03d}",
                                line_number=idx,
                                line_id=current_line_id,
                                pack_id=pack_identifier,
                                error_path=error_path,
                                result_path=result_target_path,
                                result_display=result_target_display,
                            )

                            prevalidated = self._validate_response_payload(
                                response, pack_line
                            )
                            normalized_response, schema_errors, schema_mode = prevalidated
                            if (
                                schema_mode == "structured"
                                and normalized_response is None
                                and schema_errors
                                and validation_attempt < structured_retry_limit
                            ):
                                validation_attempt += 1
                                log.warning(
                                    "VALIDATION_STRUCTURED_RESPONSE_RETRY sid=%s account_id=%03d line_id=%s attempt=%s errors=%s",
                                    self.sid,
                                    account_int,
                                    current_line_id,
                                    validation_attempt,
                                    schema_errors,
                                )
                                prevalidated = None
                                continue
                            break
                    except Exception as exc:
                        error_message = (
                            "AI request failed for acc "
                            f"{account_int:03d} pack={pack_display} -> "
                            f"{result_target_display}: {exc}"
                        )
                        errors.append(error_message)
                        log.error(
                            "VALIDATION_SEND_MODEL_ERROR sid=%s account_id=%03d line_id=%s "
                            "exc_type=%s message=%s",
                            self.sid,
                            account_int,
                            current_line_id,
                            type(exc).__name__,
                            exc,
                        )
                        response = self._fallback_response(
                            error_message,
                            pack_line,
                            account_id=account_int,
                            line_id=current_line_id,
                            line_number=idx,
                        )
                        prevalidated = None
                line_result, metadata = self._build_result_line(
                    account_int,
                    idx,
                    pack_line,
                    response,
                    prevalidated=prevalidated,
                )
                result_lines.append(line_result)
                if self._debug:
                    field_name = str(line_result.get("field", "")).strip() or self._coerce_field_name(
                        pack_line, idx
                    )
                    line_identifier = str(line_result.get("id", "")).strip()
                    decision_value = str(line_result.get("decision", ""))
                    rationale_text = str(line_result.get("rationale", ""))
                    log.info(
                        "VALIDATION_LINE_SENT account=%03d field=%s id=%s decision=%s rationale_chars=%s",
                        account_int,
                        field_name,
                        line_identifier or self._coerce_identifier(account_int, idx, pack_line.get("id")),
                        decision_value,
                        len(rationale_text),
                    )
                fields_sent += 1
                is_conditional = bool(metadata.get("conditional"))
                bucket = "conditional" if is_conditional else "non_conditional"
                if is_conditional:
                    conditional_sent += 1

                gate_info = metadata.get("gate_info") or None
                if gate_info:
                    decision_metrics["weak"][bucket] += 1
                    gate_log: dict[str, Any] = {
                        "account_id": f"{account_int:03d}",
                        "line_number": idx,
                        "field": metadata.get("field"),
                        "reason": gate_info.get("reason"),
                        "original_decision": metadata.get("original_decision"),
                        "final_decision": metadata.get("final_decision"),
                        "corroboration": gate_info.get("corroboration"),
                        "required_corroboration": gate_info.get("required_corroboration"),
                    }
                    unique_values = gate_info.get("unique_values")
                    if unique_values is not None:
                        gate_log["unique_values"] = unique_values
                    if "high_signal_keywords" in gate_info:
                        gate_log["high_signal_keywords"] = gate_info["high_signal_keywords"]
                    self._log("send_conditional_gate_downgrade", **gate_log)
                else:
                    final_decision = metadata.get("final_decision", "no_case")
                    decision_metrics.setdefault(
                        final_decision, {"conditional": 0, "non_conditional": 0}
                    )
                    decision_metrics[final_decision][bucket] += 1
                time.sleep(self._throttle)
        except ValidationPackError as exc:
            if current_line_id:
                failed_line_ids.append(current_line_id)
            if not getattr(exc, "line_ids", None) and failed_line_ids:
                setattr(exc, "line_ids", list(dict.fromkeys(failed_line_ids)))
            raise
        except Exception as exc:
            if current_line_id:
                failed_line_ids.append(current_line_id)
            if failed_line_ids and not getattr(exc, "line_ids", None):
                setattr(exc, "line_ids", list(dict.fromkeys(failed_line_ids)))
            raise

        if len(result_lines) < expected_count:
            missing = expected_count - len(result_lines)
            log.error(
                "VALIDATION_RESULT_COUNT_MISMATCH sid=%s account_id=%03d expected=%s actual=%s",
                self.sid,
                account_int,
                expected_count,
                len(result_lines),
            )
            self._log(
                "send_account_result_mismatch",
                account_id=f"{account_int:03d}",
                expected=expected_count,
                actual=len(result_lines),
                missing=missing,
            )
            errors.append("result_count_mismatch")

        status = "error" if errors else "done"
        error_message = "; ".join(errors) if errors else None
        metrics_payload = {
            "total_fields": total_fields,
            "fields_sent": fields_sent,
            "fields_skipped": max(total_fields - fields_sent, 0),
            "conditional_fields_sent": conditional_sent,
            "decision_counts": decision_metrics,
            "model_requests": model_requests,
        }
        completed_at = _utc_now()
        jsonl_path, summary_path = self._write_results(
            account_int,
            result_lines,
            status=status,
            error=error_message,
            jsonl_path=result_jsonl_path,
            jsonl_display=result_jsonl_display,
            summary_path=result_target_path,
            summary_display=result_target_display,
        )

        summary_payload: dict[str, Any] = {
            "sid": self.sid,
            "account_id": account_int,
            "pack_path": str(pack_path),
            "pack_manifest_path": pack_display,
            "results_path": str(summary_path),
            "results_manifest_path": result_target_display,
            "jsonl_path": str(jsonl_path),
            "jsonl_manifest_path": result_jsonl_display,
            "status": status,
            "model": self.model,
            "request_lines": model_requests,
            "results": result_lines,
            "completed_at": completed_at,
        }
        if error_message:
            summary_payload["error"] = error_message
        summary_payload["metrics"] = metrics_payload

        self._log(
            "send_account_done",
            account_id=f"{account_int:03d}",
            status=status,
            errors=len(errors),
            results=len(result_lines),
        )
        self._log(
            "send_account_metrics",
            account_id=f"{account_int:03d}",
            **metrics_payload,
        )
        duration = time.monotonic() - start_time
        log.info(
            "VALIDATION_SEND_ACCOUNT_END sid=%s account_id=%03d status=%s results=%s duration=%.3fs",
            self.sid,
            account_int,
            status,
            len(result_lines),
            duration,
        )
        self._record_index_result(
            pack_path=pack_path,
            status=status,
            error=error_message,
            request_lines=model_requests,
            result_path=summary_path,
            line_count=len(result_lines),
            completed_at=completed_at,
        )
        return summary_payload

    def _record_account_failure(
        self,
        account_id: int,
        pack_path: Path,
        pack_display: str,
        result_jsonl_path: Path,
        result_jsonl_display: str,
        result_summary_path: Path,
        result_summary_display: str,
        error: str,
        *,
        finalize: bool = False,
    ) -> dict[str, Any]:
        if self._write_json_envelope:
            result_target_path = result_summary_path
            result_target_display = result_summary_display
        else:
            result_target_path = result_jsonl_path
            result_target_display = result_jsonl_display

        if not result_target_display:
            fallback_display = (
                result_jsonl_display
                if result_target_path == result_jsonl_path
                else result_summary_display
            )
            result_target_display = fallback_display or self._display_path(
                result_target_path
            )

        jsonl_path = _canonical_result_path(result_jsonl_path)
        summary_path = _canonical_result_path(
            result_target_path, allow_json=self._write_json_envelope
        )

        summary_display_value: str
        if finalize:
            completed_at = _utc_now()
            jsonl_path, summary_path = self._write_results(
                account_id,
                [],
                status="error",
                error=error,
                jsonl_path=result_jsonl_path,
                jsonl_display=result_jsonl_display,
                summary_path=result_target_path,
                summary_display=result_target_display,
            )
            summary_display_value = result_target_display or summary_path.name
        else:
            completed_at = _utc_now()
            tmp_jsonl = self._ensure_incomplete_placeholder(jsonl_path)
            summary_display_value = result_target_display or jsonl_path.name
            summary_absolute: str
            if self._write_json_envelope:
                tmp_summary = self._ensure_incomplete_placeholder(summary_path)
                summary_absolute = str(tmp_summary.resolve())
            else:
                summary_absolute = str(tmp_jsonl.resolve())
            self._log(
                "send_account_results",
                account_id=f"{account_id:03d}",
                jsonl=result_jsonl_display or jsonl_path.name,
                jsonl_absolute=str(tmp_jsonl.resolve()),
                summary=summary_display_value,
                summary_absolute=summary_absolute,
                results=0,
                status="error",
                incomplete=True,
            )
            log.error(
                "VALIDATION_SEND_RESULTS_INCOMPLETE sid=%s account_id=%03d jsonl=%s tmp=%s",
                self.sid,
                account_id,
                str(jsonl_path),
                str(tmp_jsonl),
            )

        summary_payload = {
            "sid": self.sid,
            "account_id": account_id,
            "pack_path": str(pack_path),
            "pack_manifest_path": pack_display,
            "results_path": str(summary_path),
            "results_manifest_path": result_target_display,
            "jsonl_path": str(jsonl_path),
            "jsonl_manifest_path": result_jsonl_display,
            "status": "error",
            "model": self.model,
            "request_lines": 0,
            "results": [],
            "completed_at": completed_at,
            "error": error,
        }
        metrics_payload = {
            "total_fields": 0,
            "fields_sent": 0,
            "fields_skipped": 0,
            "conditional_fields_sent": 0,
            "decision_counts": _empty_decision_metrics(),
            "model_requests": 0,
        }
        summary_payload["metrics"] = metrics_payload
        self._log(
            "send_account_done",
            account_id=f"{account_id:03d}",
            status="error",
            errors=1,
            results=0,
        )
        self._log(
            "send_account_metrics",
            account_id=f"{account_id:03d}",
            **metrics_payload,
        )
        self._record_index_result(
            pack_path=pack_path,
            status="error",
            error=error,
            request_lines=0,
            result_path=None,
            line_count=0,
            completed_at=completed_at,
        )
        return summary_payload

    @staticmethod
    def _extract_default_decision(pack_line: Mapping[str, Any]) -> str | None:
        if not isinstance(pack_line, Mapping):
            return None

        candidates: list[Any] = []

        direct_value = pack_line.get("default_decision")
        if direct_value is not None:
            candidates.append(direct_value)

        finding = pack_line.get("finding")
        if isinstance(finding, Mapping):
            nested_value = finding.get("default_decision")
            if nested_value is not None:
                candidates.append(nested_value)

        for candidate in candidates:
            decision = _normalize_structured_decision(candidate)
            if decision in _VALID_DECISIONS:
                return decision

        return None

    def _build_deterministic_response(
        self,
        pack_line: Mapping[str, Any],
        *,
        account_id: int,
        line_number: int,
        line_id: str,
    ) -> Mapping[str, Any]:
        reason_code = pack_line.get("reason_code")
        if not isinstance(reason_code, str) or not reason_code.strip():
            raise ValidationPackError("Pack line missing reason_code")
        reason_code = reason_code.strip()

        raw_reason_label = pack_line.get("reason_label")
        if isinstance(raw_reason_label, str):
            reason_label = raw_reason_label.strip()
        elif raw_reason_label is None:
            reason_label = ""
        else:
            reason_label = str(raw_reason_label).strip()
        if not reason_label:
            reason_label = reason_code

        field = self._coerce_field_name(pack_line, line_number)
        finding = pack_line.get("finding")
        if not isinstance(finding, Mapping):
            finding = {}
        is_mismatch = _coerce_bool_flag(finding.get("is_mismatch"))

        (
            heuristic_decision,
            modifiers,
            has_long_history,
            has_semantic_majority,
        ) = self._deterministic_decision_logic(
            pack_line,
            reason_code=reason_code,
            is_mismatch=is_mismatch,
        )
        default_decision = self._extract_default_decision(pack_line)
        if default_decision:
            decision = default_decision
        else:
            decision = _normalize_structured_decision(heuristic_decision) or "no_case"

        rationale = self._compose_deterministic_rationale(
            decision,
            reason_code,
            reason_label,
            is_mismatch=is_mismatch,
            long_history=has_long_history,
            semantic_majority=has_semantic_majority,
        )
        citations = self._build_deterministic_citations(pack_line)
        confidence = self._deterministic_confidence(pack_line, decision)

        response = {
            "sid": str(pack_line.get("sid") or self.sid),
            "account_id": account_id,
            "id": line_id,
            "field": field,
            "decision": decision,
            "rationale": rationale,
            "citations": citations,
            "reason_code": reason_code,
            "reason_label": reason_label,
            "modifiers": modifiers,
            "confidence": confidence,
        }

        self._log(
            "send_line_deterministic",
            account_id=f"{account_id:03d}",
            line_number=line_number,
            field=field,
            reason_code=reason_code,
            decision=decision,
        )

        return response

    def _deterministic_decision_logic(
        self,
        pack_line: Mapping[str, Any],
        *,
        reason_code: str,
        is_mismatch: bool,
    ) -> tuple[str, dict[str, bool], bool, bool]:
        modifiers: dict[str, bool] = {
            "material_mismatch": bool(is_mismatch),
            "time_anchor": False,
            "doc_dependency": False,
        }

        long_history = self._has_long_consistent_history(pack_line)
        semantic_majority = self._has_semantic_majority(pack_line)

        decision = "neutral" if is_mismatch else "no_case"

        if reason_code == "C5_ALL_DIFF":
            modifiers["material_mismatch"] = True
            if long_history:
                decision = "supportive"
                modifiers["time_anchor"] = True
        elif reason_code == "C4_TWO_MATCH_ONE_DIFF":
            if semantic_majority:
                decision = "supportive"
                modifiers["material_mismatch"] = False
            else:
                modifiers["material_mismatch"] = bool(is_mismatch)
        else:
            modifiers["material_mismatch"] = bool(is_mismatch)

        return decision, modifiers, long_history, semantic_majority

    def _compose_deterministic_rationale(
        self,
        decision: str,
        reason_code: str,
        reason_label: str,
        *,
        is_mismatch: bool,
        long_history: bool,
        semantic_majority: bool,
    ) -> str:
        base_label = reason_label or reason_code or "This discrepancy"
        suffix = f" ({reason_code})" if reason_code else ""

        if decision == "strong_actionable":
            if long_history:
                return (
                    f"{base_label} is deterministically actionable with corroborating history"
                    f"{suffix}."
                )
            return f"{base_label} is deterministically actionable{suffix}."

        if decision == "supportive_needs_companion":
            if semantic_majority:
                return (
                    f"{base_label} aligns across bureaus and supports the dispute but needs corroboration"
                    f"{suffix}."
                )
            if long_history:
                return (
                    f"{base_label} is backed by consistent history but still needs corroboration"
                    f"{suffix}."
                )
            return f"{base_label} supports the dispute but needs corroboration{suffix}."

        if decision == "neutral_context_only":
            if is_mismatch:
                return f"{base_label} mismatch is recorded for context only{suffix}."
            return f"{base_label} is recorded for context only{suffix}."

        return f"{base_label} does not create an actionable dispute{suffix}."

    def _build_deterministic_citations(
        self, pack_line: Mapping[str, Any]
    ) -> list[str]:
        records = list(self._iter_bureau_records_for_citations(pack_line))
        citations: list[str] = []
        for bureau, values in records:
            normalized_value = values.get("normalized") if isinstance(values, Mapping) else None
            raw_value = values.get("raw") if isinstance(values, Mapping) else None
            text = self._stringify_citation_value(normalized_value)
            if text is None:
                text = self._stringify_citation_value(raw_value)
            if text is None:
                text = "None"
            citations.append(f"{bureau}: {text}")

        if not citations:
            citations.append("equifax: None")

        return citations

    def _iter_bureau_records_for_citations(
        self, pack_line: Mapping[str, Any]
    ) -> list[tuple[str, Mapping[str, Any]]]:
        records = _extract_bureau_records(pack_line)
        if not records:
            finding = pack_line.get("finding")
            if isinstance(finding, Mapping):
                bureau_values = finding.get("bureau_values")
                if isinstance(bureau_values, Mapping):
                    for bureau, value in bureau_values.items():
                        key = _normalize_bureau_key(bureau)
                        if not key or not isinstance(value, Mapping):
                            continue
                        records[key] = {
                                "raw": value.get("raw"),
                                "normalized": value.get("normalized"),
                            }
        return [(bureau, records[bureau]) for bureau in sorted(records)]

    @staticmethod
    def _stringify_citation_value(value: Any) -> str | None:
        if value is None:
            return None
        if isinstance(value, str):
            text = value.strip()
            return text or None
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, (int, float)):
            return str(value)
        try:
            return json.dumps(value, ensure_ascii=False, sort_keys=True)
        except (TypeError, ValueError):
            try:
                return str(value)
            except Exception:
                return None

    def _has_long_consistent_history(self, pack_line: Mapping[str, Any]) -> bool:
        context: Mapping[str, Any] | None = None
        raw_context = pack_line.get("context")
        if isinstance(raw_context, Mapping):
            context = raw_context
        if context is None:
            finding = pack_line.get("finding")
            if isinstance(finding, Mapping):
                finding_context = finding.get("context")
                if isinstance(finding_context, Mapping):
                    context = finding_context
        if context is None:
            return False

        history = context.get("history") if isinstance(context, Mapping) else None
        if not isinstance(history, Mapping):
            return False

        if not self._history_has_consistent_flag(history):
            return False

        span_months = self._extract_history_span_months(history)
        return span_months is not None and span_months >= 18

    def _history_has_consistent_flag(self, history: Mapping[str, Any]) -> bool:
        stack: list[Mapping[str, Any]] = [history]
        while stack:
            current = stack.pop()
            if _coerce_bool_flag(current.get("consistent")):
                return True
            for key, value in current.items():
                if isinstance(value, Mapping):
                    stack.append(value)
                else:
                    key_text = str(key).lower()
                    if _coerce_bool_flag(value) and "consistent" in key_text:
                        return True
                    if isinstance(value, str):
                        lowered = value.strip().lower()
                        if "consistent" in lowered and "inconsistent" not in lowered:
                            return True
        return False

    def _extract_history_span_months(self, history: Mapping[str, Any]) -> int | None:
        stack: list[Mapping[str, Any]] = [history]
        max_span: int | None = None
        while stack:
            current = stack.pop()
            for key, value in current.items():
                if isinstance(value, Mapping):
                    stack.append(value)
                key_text = str(key).lower()
                months = self._coerce_months_value(value)
                if months is not None and any(
                    token in key_text for token in ("month", "span", "duration", "range")
                ):
                    if max_span is None or months > max_span:
                        max_span = months
        return max_span

    @staticmethod
    def _coerce_months_value(value: Any) -> int | None:
        if isinstance(value, bool):
            return None
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return int(value)
        if isinstance(value, str):
            text = value.strip()
            if not text:
                return None
            match = re.search(r"\d+", text)
            if match:
                try:
                    return int(match.group())
                except ValueError:
                    return None
        return None

    def _has_semantic_majority(self, pack_line: Mapping[str, Any]) -> bool:
        records = self._iter_bureau_records_for_citations(pack_line)
        tokens: list[str] = []
        for _, values in records:
            normalized_value = None
            if isinstance(values, Mapping):
                normalized_value = values.get("normalized")
            token = self._normalize_semantic_value(normalized_value)
            if token is not None:
                tokens.append(token)
        if not tokens:
            return False
        counts = Counter(tokens)
        return any(count >= 2 for count in counts.values())

    @staticmethod
    def _normalize_semantic_value(value: Any) -> str | None:
        if value is None:
            return None
        if isinstance(value, str):
            text = value.strip().lower()
            return text or None
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, (int, float)):
            return str(value)
        try:
            return json.dumps(value, ensure_ascii=False, sort_keys=True)
        except (TypeError, ValueError):
            try:
                return str(value)
            except Exception:
                return None

    @staticmethod
    def _deterministic_confidence(
        pack_line: Mapping[str, Any], decision: str
    ) -> float:
        del pack_line, decision  # Deterministic heuristic placeholder
        return 0.65

    def _call_model(
        self,
        pack_line: Mapping[str, Any],
        *,
        account_id: int,
        account_label: str,
        line_number: int,
        line_id: str,
        pack_id: str,
        error_path: Path,
        result_path: Path,
        result_display: str,
    ) -> Mapping[str, Any]:
        prompt_payload = pack_line.get("prompt")
        if not isinstance(prompt_payload, Mapping):
            raise ValidationPackError("Pack line missing prompt payload")

        system_prompt = prompt_payload.get("system")
        user_prompt = prompt_payload.get("user")
        guidance_prompt = prompt_payload.get("guidance")

        if not isinstance(system_prompt, str) or not system_prompt:
            raise ValidationPackError("Pack prompt missing system message")
        if not isinstance(user_prompt, str) or not user_prompt:
            raise ValidationPackError("Pack prompt missing user message")
        if guidance_prompt is not None and not isinstance(guidance_prompt, str):
            raise ValidationPackError("Pack prompt guidance must be a string if provided")

        expected_output_schema = self._extract_expected_output_schema(pack_line)

        def _record_sidecar(status: int, body: str) -> None:
            try:
                normalized_status = int(status)
            except (TypeError, ValueError):
                normalized_status = 0
            payload = {
                "status": normalized_status,
                "body": body,
                "pack_id": pack_id,
            }
            self._write_error_sidecar(error_path, payload)

        base_user_prompt = user_prompt
        suffixes: list[str] = []
        max_retries = _validation_max_retries()
        attempt = 0

        while True:
            composed_user_prompt = base_user_prompt + "".join(suffixes)
            messages: list[dict[str, str]] = []
            messages.append({"role": "system", "content": system_prompt})
            if guidance_prompt is not None:
                messages.append({"role": "system", "content": guidance_prompt})
            messages.append({"role": "user", "content": composed_user_prompt})
            response_format = {
                "type": "json_schema",
                "json_schema": {
                    "name": "validation_expected_output",
                    "schema": expected_output_schema,
                },
            }
            request_payload = {
                "model": self.model,
                "messages": list(messages),
                "response_format": response_format,
            }

            try:
                create_params = inspect.signature(self._client.create).parameters
            except (TypeError, ValueError):
                create_params = {}

            if "payload" in create_params:
                response = self._client.create(
                    request_payload,
                    pack_id=pack_id,
                    on_error=_record_sidecar,
                )
            else:
                response = self._client.create(
                    **request_payload,
                    pack_id=pack_id,
                    on_error=_record_sidecar,
                )

            payload = request_payload
            if hasattr(response, "payload"):
                payload = response.payload  # type: ignore[assignment]
                status_code = getattr(response, "status_code", 0)
                latency = getattr(response, "latency", 0.0)
                retries = getattr(response, "retries", 0)
            elif isinstance(response, Mapping):
                payload = response
                status_code = int(getattr(response, "status_code", 0))
                latency_raw = getattr(response, "latency", 0.0)
                try:
                    latency = float(latency_raw)
                except (TypeError, ValueError):
                    latency = 0.0
                retries = int(getattr(response, "retries", 0))
            else:
                raise ValidationPackError(
                    f"Model client returned unsupported response type: {type(response)!r}"
                )
            log.info(
                "VALIDATION_SEND_MODEL_CALL sid=%s account_id=%s line_id=%s line_number=%s status=%s latency=%.3fs retries=%s attempt=%s",
                self.sid,
                account_label,
                line_id,
                line_number,
                status_code,
                latency,
                retries,
                attempt,
            )

            choices = payload.get("choices")
            if not isinstance(choices, Sequence) or not choices:
                log.error("VALIDATION_EMPTY_RESPONSE pack=%s", pack_id)
                serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True)
                _record_sidecar(int(status_code), serialized)
                raise ValidationPackError("Model response missing choices")
            message = choices[0].get("message") if isinstance(choices[0], Mapping) else None
            content = message.get("content") if isinstance(message, Mapping) else None
            if not isinstance(content, str) or not content.strip():
                log.error("VALIDATION_EMPTY_RESPONSE pack=%s", pack_id)
                serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True)
                _record_sidecar(int(status_code), serialized)
                raise ValidationPackError("Model response missing content")

            try:
                parsed = json.loads(content)
            except json.JSONDecodeError as exc:
                log.warning(
                    "VALIDATION_RESPONSE_PARSE_FAILED sid=%s account_id=%s line_id=%s reason=%s raw=%r",
                    self.sid,
                    account_label,
                    line_id,
                    exc,
                    content,
                )
                _record_sidecar(int(status_code), content)
                return make_fallback_decision(pack_line)
            if not isinstance(parsed, Mapping):
                log.warning(
                    "VALIDATION_RESPONSE_NOT_OBJECT sid=%s account_id=%s line_id=%s type=%s raw=%r",
                    self.sid,
                    account_label,
                    line_id,
                    type(parsed).__name__,
                    content,
                )
                _record_sidecar(int(status_code), content)
                return make_fallback_decision(pack_line)

            normalized_response, errors, _ = self._validate_response_payload(
                parsed, pack_line
            )
            if normalized_response is not None and not errors:
                result_target = result_display or str(result_path)
                log.info("VALIDATION_SENT_OK pack=%s -> %s", pack_id, result_target)
                return parsed

            if attempt >= max_retries:
                log.warning(
                    "VALIDATION_AI_RESPONSE_INVALID_MAX_RETRIES sid=%s account_id=%s line_id=%s errors=%s",
                    self.sid,
                    account_label,
                    line_id,
                    errors,
                )
                result_target = result_display or str(result_path)
                log.info("VALIDATION_SENT_OK pack=%s -> %s", pack_id, result_target)
                return parsed

            correction_errors = errors or [
                "response_did_not_match_expected_output"
            ]
            suffix = correction_suffix(correction_errors)
            suffixes.append(suffix)
            attempt += 1
            log.info(
                "VALIDATION_AI_RESPONSE_RETRY sid=%s account_id=%s line_id=%s attempt=%s errors=%s",
                self.sid,
                account_label,
                line_id,
                attempt,
                errors,
            )

    # ------------------------------------------------------------------
    # Guardrails
    # ------------------------------------------------------------------
    def _validate_response_payload(
        self,
        response: Mapping[str, Any],
        pack_line: Mapping[str, Any],
    ) -> tuple[dict[str, Any] | None, list[str], str]:
        if not isinstance(response, Mapping):
            return None, ["response_not_mapping"], "structured"

        try:
            expected_output = self._extract_expected_output_schema(pack_line)
        except ValidationPackError as exc:
            return None, [str(exc)], "structured"

        try:
            validator = Draft7Validator(expected_output)
        except Exception as exc:  # pragma: no cover - defensive
            error_message = f"invalid_expected_output_schema: {exc}"
            log.exception(
                "VALIDATION_EXPECTED_OUTPUT_SCHEMA_INVALID sid=%s error=%s",
                self.sid,
                exc,
            )
            return None, [error_message], "structured"

        schema_errors = [error.message for error in validator.iter_errors(response)]
        if schema_errors:
            return None, schema_errors, "structured"

        normalized, normalization_errors = validate_and_normalize(response, pack_line)
        if normalized is None:
            return None, normalization_errors, "structured"

        return normalized, [], "structured"

    @staticmethod
    def _extract_expected_output_schema(
        pack_line: Mapping[str, Any] | None,
    ) -> Mapping[str, Any]:
        if not isinstance(pack_line, Mapping):
            raise ValidationPackError("Pack line missing expected_output schema")

        expected_output = pack_line.get("expected_output")
        if not isinstance(expected_output, Mapping):
            raise ValidationPackError("Pack line missing expected_output schema")

        return expected_output

    # ------------------------------------------------------------------
    # Result construction & persistence
    # ------------------------------------------------------------------
    def _build_result_line(
        self,
        account_id: int,
        line_number: int,
        pack_line: Mapping[str, Any],
        response: Mapping[str, Any],
        *,
        prevalidated: tuple[dict[str, Any] | None, list[str], str] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        line_id = self._coerce_identifier(account_id, line_number, pack_line.get("id"))
        field = self._coerce_field_name(pack_line, line_number)
        if prevalidated is None:
            normalized_response, schema_errors, schema_mode = self._validate_response_payload(
                response, pack_line
            )
        else:
            normalized_response, schema_errors, schema_mode = prevalidated

        guardrail_info: dict[str, Any] | None = None
        labels: list[str] = []
        if normalized_response is None:
            emit_counter("validation.ai.response_invalid")
            raw_response: str
            try:
                raw_response = json.dumps(response, ensure_ascii=False, sort_keys=True)
            except Exception:
                raw_response = repr(response)
            log.warning(
                "VALIDATION_AI_RESPONSE_INVALID field=%s errors=%s raw=%s",
                field,
                schema_errors,
                raw_response,
            )
            original_decision = self._normalize_decision(response.get("decision"))
            if isinstance(pack_line, Mapping):
                fallback_context = dict(pack_line)
            else:
                fallback_context = {}
            fallback_context.setdefault("account_id", account_id)
            fallback_context.setdefault("id", line_id)
            fallback_context.setdefault("field", field)
            fallback_payload = make_fallback_decision(fallback_context)
            decision = str(fallback_payload.get("decision", "no_case"))
            rationale = str(
                fallback_payload.get(
                    "rationale",
                    "No valid model response; deterministic fallback.",
                )
            )
            citations = list(fallback_payload.get("citations", []))
            confidence = None
            rationale = _append_guardrail_note(rationale, "invalid_response")
            guardrail_info = {"reason": "invalid_response", "errors": schema_errors}
        else:
            decision = normalized_response["decision"]
            original_decision = decision
            rationale = normalized_response["rationale"]
            citations = list(normalized_response.get("citations", []))
            confidence = normalized_response.get("confidence")
            labels = list(normalized_response.get("labels", []))

            if (
                decision == "strong_actionable"
                and confidence is not None
                and confidence < self._confidence_threshold
            ):
                emit_counter("validation.ai.response_low_confidence")
                log.warning(
                    "VALIDATION_AI_LOW_CONFIDENCE field=%s confidence=%.6f threshold=%.6f",
                    field,
                    confidence,
                    self._confidence_threshold,
                )
                guardrail_info = {
                    "reason": "low_confidence",
                    "confidence": confidence,
                    "threshold": self._confidence_threshold,
                }
                decision = "no_case"
                rationale = _append_guardrail_note(rationale, "low_confidence")

        decision, rationale, gate_info = _enforce_conditional_gate(
            field, decision, rationale, pack_line
        )

        result = {
            "id": line_id,
            "account_id": account_id,
            "field": field,
            "decision": decision,
            "rationale": rationale,
            "citations": citations,
        }
        if confidence is not None:
            result["confidence"] = confidence
        if labels:
            result["labels"] = labels

        legacy_decision = _REVERSE_LEGACY_DECISION_MAP.get(decision, "no_case")
        result["legacy_decision"] = legacy_decision

        metadata: dict[str, Any] = {
            "field": field,
            "final_decision": decision,
            "original_decision": original_decision,
            "conditional": field in _CONDITIONAL_FIELDS,
            "gate_info": gate_info,
        }
        if labels:
            metadata["labels"] = labels
        if guardrail_info is not None:
            metadata["guardrail"] = guardrail_info
            log_payload: dict[str, Any] = {
                "account_id": f"{account_id:03d}",
                "line_number": line_number,
                "field": field,
                "reason": guardrail_info.get("reason"),
            }
            if "confidence" in guardrail_info:
                log_payload["confidence"] = guardrail_info["confidence"]
            if "threshold" in guardrail_info:
                log_payload["threshold"] = guardrail_info["threshold"]
            if "errors" in guardrail_info:
                log_payload["errors"] = guardrail_info["errors"]
            self._log("send_guardrail_triggered", **log_payload)
        return result, metadata

    def _write_results(
        self,
        account_id: int,
        result_lines: Sequence[Mapping[str, Any]],
        *,
        status: str = "done",
        error: str | None = None,
        jsonl_path: Path | None = None,
        jsonl_display: str | None = None,
        summary_path: Path | None = None,
        summary_display: str | None = None,
    ) -> tuple[Path, Path]:
        results_root = self._results_root or self._index.results_dir_path
        results_root.mkdir(parents=True, exist_ok=True)

        if jsonl_path is None:
            jsonl_path = (
                results_root
                / validation_result_jsonl_filename_for_account(account_id)
            )

        if summary_path is None:
            summary_path = (
                results_root
                / validation_result_summary_filename_for_account(account_id)
            )

        if self._write_json_envelope:
            summary_parent = summary_path.parent if summary_path else results_root
            summary_path = summary_parent / validation_result_json_filename_for_account(
                account_id
            )

        jsonl_path = _canonical_result_path(jsonl_path)
        summary_path = _canonical_result_path(
            summary_path, allow_json=self._write_json_envelope
        )
        jsonl_display = _canonical_result_display(jsonl_display)
        summary_display = _canonical_result_display(
            summary_display, allow_json=self._write_json_envelope
        )

        jsonl_path.parent.mkdir(parents=True, exist_ok=True)

        serialized_lines: list[dict[str, Any]] = []
        for line in result_lines:
            if not isinstance(line, Mapping):
                raise TypeError("Result line must be a mapping to serialize as JSONL")
            serialized_lines.append(dict(line))

        alias_name: str | None = None
        if jsonl_display:
            try:
                alias_name = PurePosixPath(jsonl_display).name
            except Exception:
                alias_name = None

        if serialized_lines:
            write_jsonl(jsonl_path, serialized_lines)
        else:
            try:
                jsonl_path.unlink()
            except FileNotFoundError:
                pass

        if alias_name:
            alias_path = jsonl_path.with_name(alias_name)
            if alias_path != jsonl_path:
                if serialized_lines:
                    write_jsonl(alias_path, serialized_lines)
                else:
                    try:
                        alias_path.unlink()
                    except FileNotFoundError:
                        pass
                    tmp_alias = alias_path.with_name(alias_path.name + ".tmp")
                    try:
                        tmp_alias.unlink()
                    except FileNotFoundError:
                        pass

        summary_target = summary_path
        summary_display_value: str
        if self._write_json_envelope:
            summary_target.parent.mkdir(parents=True, exist_ok=True)
            summary_payload: dict[str, Any] = {
                "sid": self.sid,
                "account_id": account_id,
                "status": status,
                "model": self.model,
                "request_lines": len(result_lines),
                "completed_at": _utc_now(),
                "results": list(serialized_lines),
            }
            if error:
                summary_payload["error"] = error
            write_json(summary_target, summary_payload)
            summary_display_value = summary_display or summary_target.name
        else:
            summary_target = jsonl_path
            summary_display_value = (
                summary_display or jsonl_display or jsonl_path.name
            )

        self._log(
            "send_account_results",
            account_id=f"{account_id:03d}",
            jsonl=jsonl_display or jsonl_path.name,
            jsonl_absolute=str(jsonl_path.resolve()),
            summary=summary_display_value,
            summary_absolute=str(summary_target.resolve()),
            results=len(result_lines),
            status=status,
        )
        log.info(
            "VALIDATION_SEND_RESULTS_WRITTEN sid=%s account_id=%03d jsonl=%s summary=%s decisions=%s status=%s",
            self.sid,
            account_id,
            str(jsonl_path),
            str(summary_target),
            len(result_lines),
            status,
        )
        return jsonl_path, summary_target

    def _ensure_incomplete_placeholder(self, target: Path) -> Path:
        """Ensure a ``.tmp`` sentinel exists for an incomplete results file."""

        target = _canonical_result_path(target, allow_json=self._write_json_envelope)
        try:
            target.unlink()
        except FileNotFoundError:
            pass
        except OSError as exc:  # pragma: no cover - defensive logging
            log.warning(
                "VALIDATION_INCOMPLETE_RESULT_CLEAN_FAILED sid=%s path=%s error=%s",
                self.sid,
                str(target),
                exc,
            )

        tmp_path = target.with_name(target.name + ".tmp")
        tmp_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with tmp_path.open("w", encoding="utf-8") as handle:
                handle.write("")
        except OSError as exc:  # pragma: no cover - defensive logging
            log.warning(
                "VALIDATION_INCOMPLETE_RESULT_WRITE_FAILED sid=%s path=%s error=%s",
                self.sid,
                str(tmp_path),
                exc,
            )
        return tmp_path

    def _write_error_sidecar(self, path: Path, payload: Mapping[str, Any]) -> None:
        data: dict[str, Any] = dict(payload)
        body_value = data.get("body", "")
        if not isinstance(body_value, str):
            body_value = "" if body_value is None else str(body_value)
        data["body"] = _truncate_response_body(body_value)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(data, ensure_ascii=False, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        except Exception as exc:  # pragma: no cover - defensive logging
            log.exception(
                "VALIDATION_ERROR_SIDECAR_WRITE_FAILED sid=%s pack=%s path=%s error=%s",
                self.sid,
                data.get("pack_id", "<unknown>"),
                str(path),
                exc,
            )

    def _clear_error_sidecar(self, path: Path, *, pack_id: str) -> None:
        try:
            path.unlink()
        except FileNotFoundError:
            return
        except OSError as exc:  # pragma: no cover - defensive logging
            log.warning(
                "VALIDATION_ERROR_SIDECAR_CLEANUP_FAILED sid=%s pack=%s path=%s error=%s",
                self.sid,
                pack_id,
                str(path),
                exc,
            )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _infer_runs_root(self) -> Path | None:
        try:
            resolved = self._index.index_path.resolve()
        except OSError:
            resolved = self._index.index_path

        sid = str(self.sid or "")
        run_dir: Path | None = None
        for parent in resolved.parents:
            if parent.name == sid:
                run_dir = parent
                break

        if run_dir is None:
            parents = list(resolved.parents)
            if len(parents) >= 3:
                run_dir = parents[2]

        if run_dir is None:
            return None

        return run_dir.parent

    def _get_index_writer(self) -> ValidationPackIndexWriter | None:
        if self._index_writer is not None:
            return self._index_writer

        try:
            writer = ValidationPackIndexWriter(
                sid=self.sid,
                index_path=self._index.index_path,
                packs_dir=self._index.packs_dir_path,
                results_dir=self._index.results_dir_path,
            )
        except Exception:  # pragma: no cover - defensive logging
            log.warning(
                "VALIDATION_INDEX_WRITER_INIT_FAILED sid=%s index=%s",
                self.sid,
                str(self._index.index_path),
                exc_info=True,
            )
            return None

        self._index_writer = writer
        return writer

    def _load_validation_stage_status(self) -> str | None:
        base_root = self._runs_root or Path("runs")
        run_dir = base_root / str(self.sid)
        runflow_path = run_dir / "runflow.json"
        try:
            raw = runflow_path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return None
        except OSError:  # pragma: no cover - defensive logging
            log.warning(
                "VALIDATION_STAGE_STATUS_READ_FAILED sid=%s path=%s",
                self.sid,
                str(runflow_path),
                exc_info=True,
            )
            return None

        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            return None

        stages = payload.get("stages")
        if not isinstance(stages, Mapping):
            return None

        validation_stage = stages.get("validation")
        if not isinstance(validation_stage, Mapping):
            return None

        status_value = validation_stage.get("status")
        if isinstance(status_value, str):
            normalized = status_value.strip().lower()
            return normalized or None
        return None

    def _refresh_validation_progress(self) -> None:
        try:
            runflow_barriers_refresh(self.sid)
        except Exception:  # pragma: no cover - defensive logging
            log.warning(
                "VALIDATION_BARRIERS_REFRESH_FAILED sid=%s",
                self.sid,
                exc_info=True,
            )

        try:
            refresh_validation_stage_from_index(
                self.sid, runs_root=self._runs_root
            )
        except Exception:  # pragma: no cover - defensive logging
            log.warning(
                "VALIDATION_STAGE_REFRESH_FAILED sid=%s",
                self.sid,
                exc_info=True,
            )
        else:
            if not self._stage_promotion_logged:
                status = self._load_validation_stage_status()
                if status == "success":
                    log.info(
                        "VALIDATION_STAGE_PROMOTED sid=%s status=%s",
                        self.sid,
                        status,
                    )
                    self._stage_promotion_logged = True

        try:
            reconcile_umbrella_barriers(self.sid, runs_root=self._runs_root)
        except Exception:  # pragma: no cover - defensive logging
            log.warning(
                "VALIDATION_BARRIERS_RECONCILE_FAILED sid=%s",
                self.sid,
                exc_info=True,
            )

    def _record_index_result(
        self,
        *,
        pack_path: Path,
        status: str,
        error: str | None,
        request_lines: int | None,
        result_path: Path | None,
        line_count: int,
        completed_at: str | None,
    ) -> None:
        normalized_status = str(status or "").strip().lower()
        
        # For "skipped" status with existing results, treat as "completed" to update index
        if normalized_status == "skipped" and result_path is not None:
            try:
                if result_path.exists() and result_path.is_file():
                    normalized_status = "completed"
                else:
                    # No result file, treat as truly skipped
                    return
            except OSError:
                # Can't verify result file, skip index update
                return
        
        if normalized_status in {"", "skipped"}:
            return

        if normalized_status not in {"done", "completed", "error", "failed"}:
            return

        writer = self._get_index_writer()
        if writer is None:
            return

        index_status = (
            "completed" if normalized_status in {"done", "completed"} else "failed"
        )

        try:
            writer.record_result(
                pack_path,
                status=index_status,
                error=error if index_status == "failed" else None,
                request_lines=request_lines,
                model=self.model,
                completed_at=completed_at or _utc_now(),
                result_path=result_path if index_status == "completed" else None,
                line_count=line_count,
            )
        except Exception:  # pragma: no cover - defensive logging
            log.warning(
                "VALIDATION_INDEX_RECORD_FAILED sid=%s pack=%s status=%s",
                self.sid,
                str(pack_path),
                index_status,
                exc_info=True,
            )
            return

        self._refresh_validation_progress()

    def _iter_pack_lines(
        self, pack_path: Path, *, display_path: str | None = None
    ) -> Iterable[Mapping[str, Any]]:
        display = display_path or str(pack_path)
        try:
            text = pack_path.read_text(encoding="utf-8")
        except FileNotFoundError as exc:
            raise ValidationPackError(f"Pack file missing: {display}") from exc
        except OSError as exc:
            raise ValidationPackError(f"Unable to read pack file: {display}") from exc

        lines: list[Mapping[str, Any]] = []
        for idx, raw in enumerate(text.splitlines(), start=1):
            if not raw.strip():
                continue
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValidationPackError(
                    f"Invalid JSON in pack line {idx} of {display}: {exc}"
                ) from exc
            if not isinstance(payload, Mapping):
                raise ValidationPackError(
                    f"Pack line {idx} of {display} is not an object"
                )
            lines.append(self._upgrade_pack_line(payload))
        return lines

    def _upgrade_pack_line(
        self, payload: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        upgraded = dict(payload)
        expected_output = payload.get("expected_output")
        if not isinstance(expected_output, Mapping):
            return upgraded

        expected_output_dict = dict(expected_output)
        properties = expected_output.get("properties")
        if isinstance(properties, Mapping):
            properties_dict = dict(properties)
            decision_schema = properties.get("decision")
            if isinstance(decision_schema, Mapping):
                decision_dict = dict(decision_schema)
                enum_values = decision_schema.get("enum")
                new_enum, changed = _remap_legacy_decision_enum(enum_values)
                if changed:
                    decision_dict["enum"] = new_enum
                    properties_dict["decision"] = decision_dict
                    expected_output_dict["properties"] = properties_dict
                    upgraded["expected_output"] = expected_output_dict
                    return upgraded
        return upgraded

    def _build_client(self) -> _ChatCompletionClient:
        api_key = (os.getenv("OPENAI_API_KEY") or "").strip()
        if not api_key:
            log.error(
                "VALIDATION_OPENAI_CREDENTIAL_ERROR model=%s detail=missing_api_key",
                getattr(self, "model", "<unknown>"),
            )
            raise ValidationPackError(
                "OPENAI_API_KEY is required to send validation packs"
            )
        base_url_raw = os.getenv("OPENAI_BASE_URL")
        base_url_clean = (base_url_raw or "").strip() or "https://api.openai.com/v1"
        timeout = _env_float("AI_REQUEST_TIMEOUT", _DEFAULT_TIMEOUT)
        project_header_enabled = os.getenv("OPENAI_SEND_PROJECT_HEADER", "0") == "1"
        global _AUTH_READY
        if not _AUTH_READY:
            with _AUTH_LOCK:
                if not _AUTH_READY:
                    auth_probe()
                    log.info(
                        "OPENAI_AUTH ready: key_prefix=%s project_header_enabled=%s base_url=%s",
                        _key_prefix(api_key),
                        project_header_enabled,
                        base_url_clean,
                    )
                    _AUTH_READY = True
        log.info(
            "VALIDATION_OPENAI_CLIENT_READY model=%s base_url=%s key_present=yes timeout=%s",
            getattr(self, "model", "<unknown>"),
            base_url_clean,
            timeout,
        )
        return _ChatCompletionClient(
            base_url=base_url_clean,
            api_key=api_key,
            timeout=timeout,
            project_id=None,
        )

    @staticmethod
    def _normalize_account_id(account_id: Any) -> int:
        if account_id is None:
            raise ValueError("account_id is required")
        try:
            return int(str(account_id))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"account_id must be numeric: {account_id!r}") from exc

    def _fallback_response(
        self,
        message: str,
        pack_line: Mapping[str, Any] | None = None,
        *,
        account_id: int | None = None,
        line_id: str | None = None,
        line_number: int | None = None,
    ) -> Mapping[str, Any]:
        field = "unknown"
        if pack_line is not None:
            try:
                field = self._coerce_field_name(pack_line, line_number or 1)
            except Exception:
                field = str(pack_line.get("field") or "unknown")

        reason_code = "FALLBACK_ERROR"
        reason_label = "AI request failed"
        if isinstance(pack_line, Mapping):
            raw_reason = pack_line.get("reason_code")
            if isinstance(raw_reason, str) and raw_reason.strip():
                reason_code = raw_reason.strip()
            raw_label = pack_line.get("reason_label")
            if isinstance(raw_label, str) and raw_label.strip():
                reason_label = raw_label.strip()

        citations: list[str]
        if isinstance(pack_line, Mapping):
            citations = self._build_deterministic_citations(pack_line)
        else:
            citations = ["equifax: None"]

        modifiers = {
            "material_mismatch": False,
            "time_anchor": False,
            "doc_dependency": False,
        }

        fallback_account = account_id if isinstance(account_id, int) else 0
        fallback_id = line_id or f"fallback_{int(time.time())}"

        rationale = f"AI error: {message} ({reason_code}); deterministic fallback"

        return {
            "sid": self.sid,
            "account_id": fallback_account,
            "id": fallback_id,
            "field": field,
            "decision": "no_case",
            "rationale": rationale,
            "citations": citations,
            "reason_code": reason_code,
            "reason_label": reason_label,
            "modifiers": modifiers,
            "confidence": 0.0,
            "checks": {
                "materiality": False,
                "supports_consumer": False,
                "doc_requirements_met": False,
                "mismatch_code": reason_code or "unknown",
            },
        }

    def _normalize_decision(self, value: Any) -> str:
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in _VALID_DECISIONS:
                return lowered
            mapped = _LEGACY_DECISION_MAP.get(lowered)
            if mapped in _VALID_DECISIONS:
                return mapped
        return "no_case"

    def _coerce_identifier(self, account_id: int, line_number: int, candidate: Any) -> str:
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()
        return f"acc_{account_id:03d}__line_{line_number:03d}"

    def _coerce_field_name(
        self, pack_line: Mapping[str, Any], line_number: int
    ) -> str:
        field_key = pack_line.get("field_key")
        if isinstance(field_key, str) and field_key.strip():
            return field_key.strip()
        field = pack_line.get("field")
        if isinstance(field, str) and field.strip():
            return field.strip()
        return f"line_{line_number:03d}"

    def _is_allowed_field(self, field: str) -> bool:
        if field in _ALLOWED_FIELDS:
            return True

        canonical = field.strip().lower().replace(" ", "_")
        if canonical in _ALLOWED_FIELDS:
            return True

        log.debug("ALLOWING_UNKNOWN_FIELD field=%s canonical=%s", field, canonical)
        return True

    def _load_index(self) -> ValidationIndex:
        return self._index

    def _log(self, event: str, **payload: Any) -> None:
        log_path = self._log_path
        log_path.parent.mkdir(parents=True, exist_ok=True)
        entry: MutableMapping[str, Any] = {
            "timestamp": _utc_now(),
            "sid": self.sid,
            "event": event,
        }
        entry.update(sanitize_validation_log_payload(payload))
        line = json.dumps(entry, ensure_ascii=False, sort_keys=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")


def send_validation_packs(
    manifest: Mapping[str, Any] | ValidationIndex | Path | str,
    *,
    stage: str | None = None,
) -> list[dict[str, Any]]:
    """Send all validation packs referenced by ``manifest``."""

    resolved_stage = stage or _MANIFEST_STAGE

    wait_info = _infer_manifest_index_wait_info(manifest, stage=resolved_stage)
    sid_hint = "<unknown>"
    if wait_info is not None:
        index_path, sid_hint = wait_info
        if not _wait_for_index_file(index_path, sid_hint):
            return []

    preparation = _prepare_validation_index(manifest, stage=resolved_stage)
    if preparation.skip:
        if preparation.index is not None and not preparation.index.packs:
            log.info(
                "VALIDATION_NO_PACKS sid=%s",
                preparation.index.sid or sid_hint,
            )
        return []

    sender = ValidationPackSender(
        manifest,
        stage=resolved_stage,
        preloaded_view=preparation.view,
    )
    results = sender.send()
    _update_stage_status_after_send(preparation.index, resolved_stage)
    return results


__all__ = ["send_validation_packs", "ValidationPackSender", "ValidationPackError"]
