"""Helpers for building validation AI adjudication packs."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
from contextlib import suppress
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
import textwrap
from typing import Any, Iterable, Mapping, Sequence

import yaml

from backend.ai.validation_index import (
    ValidationIndexEntry,
    ValidationPackIndexWriter,
)
from backend.core.ai.paths import (
    ValidationAccountPaths,
    ensure_validation_account_paths,
    ensure_validation_paths,
)
from backend.pipeline.runs import RunManifest
from backend.core.logic.utils.json_utils import parse_json

log = logging.getLogger(__name__)

_DEFAULT_MODEL = "gpt-4o-mini"
_CONFIG_PATH = Path(__file__).with_name("ai_packs_config.yml")
_DEFAULT_RETRY_BACKOFF = (1.0, 3.0, 10.0)


def _maybe_slice(iterable: Iterable[int]) -> Iterable[int]:
    """Return ``iterable`` unchanged so every account index is handled."""

    debug_first_n = os.getenv("DEBUG_FIRST_N", "").strip()
    if debug_first_n:
        log.debug(
            "DEBUG_FIRST_N=%s ignored for validation AI packs", debug_first_n
        )
    return iterable


@dataclass(frozen=True)
class ValidationPacksConfig:
    """Configuration for validation AI pack generation."""

    enable_write: bool = True
    enable_infer: bool = True
    model: str = _DEFAULT_MODEL
    weak_limit: int = 0
    max_attempts: int = 3
    backoff_seconds: tuple[float, ...] = _DEFAULT_RETRY_BACKOFF


def _load_yaml_mapping(path: Path) -> Mapping[str, Any]:
    try:
        raw_text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return {}
    except OSError:
        log.warning("VALIDATION_AI_CONFIG_READ_FAILED path=%s", path, exc_info=True)
        return {}

    try:
        loaded = yaml.safe_load(raw_text) or {}
    except Exception:
        log.warning("VALIDATION_AI_CONFIG_PARSE_FAILED path=%s", path, exc_info=True)
        return {}

    if isinstance(loaded, Mapping):
        return loaded

    log.warning(
        "VALIDATION_AI_CONFIG_TYPE_INVALID path=%s type=%s",
        path,
        type(loaded).__name__,
    )
    return {}


@lru_cache(maxsize=1)
def _load_global_config_section() -> Mapping[str, Any]:
    data = _load_yaml_mapping(_CONFIG_PATH)
    section = data.get("validation_packs") if isinstance(data, Mapping) else None
    if isinstance(section, Mapping):
        return dict(section)
    return dict(data) if isinstance(data, Mapping) else {}


def _load_local_config_section(base_dir: Path) -> Mapping[str, Any]:
    config_path = Path(base_dir) / "ai_packs_config.yml"
    data = _load_yaml_mapping(config_path)
    section = data.get("validation_packs") if isinstance(data, Mapping) else None
    if isinstance(section, Mapping):
        return dict(section)
    return dict(data) if isinstance(data, Mapping) else {}


def _coerce_bool(raw: Any, default: bool) -> bool:
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, str):
        lowered = raw.strip().lower()
        if lowered in {"1", "true", "yes", "y", "on"}:
            return True
        if lowered in {"0", "false", "no", "n", "off"}:
            return False
    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        return bool(raw)
    return default


def _coerce_int(raw: Any, default: int, *, minimum: int | None = None) -> int:
    try:
        value = int(raw)
    except (TypeError, ValueError):
        value = int(default)
    if minimum is not None and value < minimum:
        return minimum
    return value


def _coerce_str(raw: Any, default: str) -> str:
    if isinstance(raw, str):
        text = raw.strip()
        return text or default
    if raw is None:
        return default
    return str(raw)


def _coerce_backoff(raw: Any) -> tuple[int, tuple[float, ...]]:
    attempts = 3
    schedule: tuple[float, ...] = _DEFAULT_RETRY_BACKOFF

    if isinstance(raw, Mapping):
        raw_attempts = raw.get("max_attempts") or raw.get("attempts")
        if raw_attempts is not None:
            attempts = _coerce_int(raw_attempts, attempts, minimum=1)

        backoff_value = (
            raw.get("backoff_seconds")
            or raw.get("backoff")
            or raw.get("delays")
            or raw.get("schedule")
        )
        if isinstance(backoff_value, Sequence) and not isinstance(
            backoff_value, (str, bytes, bytearray)
        ):
            parsed = _coerce_float_sequence(backoff_value)
            if parsed:
                schedule = parsed
        elif backoff_value is not None:
            try:
                single = float(backoff_value)
            except (TypeError, ValueError):
                single = None
            if single is not None:
                schedule = (max(0.0, single),)

    elif isinstance(raw, Sequence) and not isinstance(raw, (str, bytes, bytearray)):
        parsed = _coerce_float_sequence(raw)
        if parsed:
            schedule = parsed
            attempts = max(attempts, len(schedule) + 1)
    elif raw is not None:
        try:
            single_val = float(raw)
        except (TypeError, ValueError):
            single_val = None
        if single_val is not None:
            schedule = (max(0.0, single_val),)
            attempts = max(attempts, 2)

    if attempts < 1:
        attempts = 1

    if not schedule:
        schedule = _DEFAULT_RETRY_BACKOFF

    return attempts, schedule


def _coerce_float_sequence(raw: Sequence[Any]) -> tuple[float, ...]:
    values: list[float] = []
    for entry in raw:
        if entry is None:
            continue
        try:
            val = float(entry)
        except (TypeError, ValueError):
            continue
        values.append(max(0.0, val))
    return tuple(values)


def _coerce_validation_config(raw: Mapping[str, Any]) -> ValidationPacksConfig:
    enable_write = _coerce_bool(raw.get("enable_write"), True)
    enable_infer = _coerce_bool(raw.get("enable_infer"), True)
    model = _coerce_str(raw.get("model"), _DEFAULT_MODEL)
    weak_limit = _coerce_int(raw.get("weak_limit"), 0, minimum=0)

    attempts, backoff_schedule = _coerce_backoff(raw.get("retry"))

    return ValidationPacksConfig(
        enable_write=enable_write,
        enable_infer=enable_infer,
        model=model,
        weak_limit=weak_limit,
        max_attempts=attempts,
        backoff_seconds=backoff_schedule,
    )


def load_validation_packs_config(
    base_dir: Path | str | None = None,
) -> ValidationPacksConfig:
    """Return the effective validation packs configuration."""

    base_path = Path(base_dir) if base_dir is not None else None

    merged: dict[str, Any] = {}
    merged.update(_load_global_config_section())
    if base_path is not None:
        merged.update(_load_local_config_section(base_path))

    return _coerce_validation_config(merged)


def load_validation_packs_config_for_run(
    sid: str,
    *,
    runs_root: Path | str | None = None,
) -> ValidationPacksConfig:
    """Convenience wrapper to read config for ``sid`` without touching disk."""

    root_path = Path(runs_root) if runs_root is not None else Path("runs")
    base_dir = root_path / sid / "ai_packs" / "validation"
    return load_validation_packs_config(base_dir)


def _normalize_indices(indices: Iterable[int | str]) -> list[int]:
    normalized: set[int] = set()
    for idx in indices:
        try:
            normalized.add(int(str(idx)))
        except Exception:
            continue
    return sorted(normalized)


def build_validation_ai_packs_for_accounts(
    sid: str,
    *,
    account_indices: Sequence[int | str],
    runs_root: Path | str | None = None,
    ai_client: Any | None = None,
) -> dict[str, Any]:
    """Trigger validation AI pack building for the provided account indices.

    The builder currently ensures the filesystem scaffold for validation AI
    packs exists so subsequent stages can populate payloads and prompts.

    Returns a statistics mapping describing how many accounts were processed,
    skipped, or failed so that callers can surface operational issues without
    inspecting logs.
    """

    normalized_indices = _normalize_indices(account_indices)
    stats = {
        "sid": sid,
        "total_accounts": len(normalized_indices),
        "written_accounts": 0,
        "skipped_accounts": 0,
        "errors": 0,
        "inference_errors": 0,
    }
    if not normalized_indices:
        log.info(
            "VALIDATION_AI_PACKS_SUMMARY sid=%s total=%d written=%d skipped=%d errors=%d inference_errors=%d",
            sid,
            stats["total_accounts"],
            stats["written_accounts"],
            stats["skipped_accounts"],
            stats["errors"],
            stats["inference_errors"],
        )
        return stats

    runs_root_path = Path(runs_root) if runs_root is not None else Path("runs")
    base_dir = runs_root_path / sid / "ai_packs" / "validation"
    packs_config = load_validation_packs_config(base_dir)

    if not packs_config.enable_write:
        log.info(
            "VALIDATION_AI_PACKS_DISABLED sid=%s reason=write_disabled base=%s",
            sid,
            base_dir,
        )
        log.info(
            "VALIDATION_AI_PACKS_SUMMARY sid=%s total=%d written=%d skipped=%d errors=%d inference_errors=%d",
            sid,
            stats["total_accounts"],
            stats["written_accounts"],
            stats["skipped_accounts"],
            stats["errors"],
            stats["inference_errors"],
        )
        return stats

    validation_paths = ensure_validation_paths(runs_root_path, sid, create=True)
    log_path = validation_paths.log_file

    model_name = packs_config.model

    if ai_client is None and packs_config.enable_infer:
        ai_client = _build_ai_client()

    if not packs_config.enable_infer:
        ai_client = None

    processed_accounts: list[int] = []
    index_entries: list[ValidationIndexEntry] = []
    accounts_root = runs_root_path / sid / "cases" / "accounts"
    index_writer = ValidationPackIndexWriter(
        sid=sid,
        index_path=validation_paths.index_file,
        packs_dir=validation_paths.packs_dir,
        results_dir=validation_paths.results_dir,
    )
    existing_index = index_writer.load_accounts()

    for idx in _maybe_slice(normalized_indices):
        account_id: int | None = None
        try:
            account_paths = ensure_validation_account_paths(
                validation_paths, idx, create=True
            )
            account_id = account_paths.account_id

            summary = _load_summary(accounts_root, idx)
            weak_items = _collect_weak_items(summary)
            if packs_config.weak_limit > 0:
                weak_items = weak_items[: packs_config.weak_limit]
            weak_count = len(weak_items)
            pack_lines, weak_fields = _build_pack_lines(
                account_paths.account_id, sid, weak_items
            )
            line_count = len(pack_lines)
            source_hash = _compute_source_hash(summary, weak_items, pack_lines)
            existing_entry = existing_index.get(account_paths.account_id)
            existing_hash = (
                str(existing_entry.get("source_hash") or "") if existing_entry else ""
            )

            pack_exists = account_paths.pack_file.exists()
            result_exists = account_paths.result_summary_file.exists()
            prompt_exists = account_paths.prompt_file.exists()

            skip_build = (
                bool(existing_entry)
                and existing_hash
                and existing_hash == source_hash
                and pack_exists
                and result_exists
                and prompt_exists
            )

            statuses: list[str]
            if skip_build:
                statuses = ["up_to_date"]
                if line_count == 0:
                    statuses.append("no_weak_items")
                result_payload = _load_model_results(account_paths.result_summary_file)
                if result_payload is None:
                    result_payload = {
                        "status": str(existing_entry.get("status") or "unknown"),
                        "timestamp": existing_entry.get("built_at") or _utc_now(),
                        "model": existing_entry.get("model") or model_name,
                    }
                inference_status = str(result_payload.get("status") or "unknown")
                model_used = (
                    result_payload.get("model")
                    or existing_entry.get("model")
                    or model_name
                )
                built_at = str(
                    existing_entry.get("built_at")
                    or result_payload.get("timestamp")
                    or _utc_now()
                )
                request_lines = existing_entry.get("request_lines")
                sent_at = existing_entry.get("sent_at")
                completed_at = existing_entry.get("completed_at")
                error_msg = existing_entry.get("error")
                stats["skipped_accounts"] += 1
            else:
                statuses = ["pack_written"]
                _persist_pack_lines(account_paths.pack_file, pack_lines)

                if weak_items:
                    prompt_text = _render_prompt(sid, idx, weak_items)
                    _write_prompt(account_paths.prompt_file, prompt_text)
                else:
                    prompt_text = ""
                    _write_prompt(account_paths.prompt_file, prompt_text)
                    statuses.append("no_weak_items")

                result_payload = _run_model_inference(
                    ai_client,
                    prompt_text,
                    model_name,
                    sid=sid,
                    account_idx=idx,
                    has_weak_items=bool(weak_items),
                    config=packs_config,
                )
                _write_model_results(account_paths.result_summary_file, result_payload)

                inference_status = str(result_payload.get("status") or "unknown")
                if inference_status == "ok":
                    statuses.append("infer_done")
                elif inference_status == "error":
                    statuses.append("errors")

                model_used = result_payload.get("model") or model_name
                built_at = str(result_payload.get("timestamp") or _utc_now())
                request_lines = line_count or None
                sent_at = None
                completed_at = None
                error_msg = None

                stats["written_accounts"] += 1

                if inference_status == "skipped":
                    stats["skipped_accounts"] += 1
                elif inference_status == "error":
                    stats["inference_errors"] += 1

            log_entry: dict[str, Any] = {
                "timestamp": result_payload.get("timestamp") or _utc_now(),
                "account_index": int(idx),
                "weak_count": weak_count,
                "statuses": statuses,
                "inference_status": inference_status,
            }
            reason = result_payload.get("reason")
            if reason:
                log_entry["inference_reason"] = str(reason)
            if model_used:
                log_entry["model"] = str(model_used)

            _append_validation_log_entry(log_path, log_entry)

            index_entries.append(
                ValidationIndexEntry(
                    account_id=int(idx),
                    pack_path=account_paths.pack_file,
                    result_jsonl_path=account_paths.result_jsonl_file,
                    result_json_path=account_paths.result_summary_file,
                    weak_fields=weak_fields,
                    line_count=line_count,
                    status=inference_status,
                    built_at=built_at,
                    request_lines=request_lines if request_lines is not None else None,
                    model=str(model_used) if model_used else None,
                    sent_at=str(sent_at) if sent_at else None,
                    completed_at=str(completed_at) if completed_at else None,
                    error=str(error_msg) if error_msg else None,
                    source_hash=source_hash,
                )
            )

            processed_accounts.append(account_paths.account_id)
        except Exception:  # pragma: no cover - defensive logging
            stats["errors"] += 1
            log.exception(
                "VALIDATION_AI_PACK_ACCOUNT_FAILED sid=%s account=%s",
                sid,
                account_id if account_id is not None else idx,
            )
            continue

    manifest_path = runs_root_path / sid / "manifest.json"
    manifest = RunManifest.load_or_create(manifest_path, sid)
    index_path = validation_paths.index_file
    if index_entries:
        index_writer.bulk_upsert(index_entries)
    manifest.upsert_validation_packs_dir(
        validation_paths.base,
        packs_dir=validation_paths.packs_dir,
        results_dir=validation_paths.results_dir,
        index_file=validation_paths.index_file,
        log_file=log_path,
    )

    log.info(
        "VALIDATION_AI_PACKS_INITIALIZED sid=%s base=%s accounts=%s",
        sid,
        validation_paths.base,
        ",".join(f"{account_id:03d}" for account_id in processed_accounts),
    )

    log.info(
        "VALIDATION_AI_PACKS_SUMMARY sid=%s total=%d written=%d skipped=%d errors=%d inference_errors=%d",
        sid,
        stats["total_accounts"],
        stats["written_accounts"],
        stats["skipped_accounts"],
        stats["errors"],
        stats["inference_errors"],
    )

    return stats


def _ensure_placeholder_files(paths: ValidationAccountPaths) -> None:
    """Create empty scaffold files for a validation pack if they are missing."""

    _ensure_file(paths.pack_file)
    _ensure_file(paths.prompt_file)
    _ensure_file(paths.result_summary_file, "{}\n")


def _ensure_file(path: Path, default_contents: str = "") -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(default_contents, encoding="utf-8")


def _load_summary(accounts_root: Path, account_idx: int) -> Mapping[str, Any] | None:
    """Return the parsed summary.json payload for ``account_idx`` if present."""

    summary_path = accounts_root / str(account_idx) / "summary.json"
    try:
        raw_text = summary_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    except OSError:
        log.warning(
            "VALIDATION_SUMMARY_READ_FAILED account=%s path=%s",
            account_idx,
            summary_path,
            exc_info=True,
        )
        return None

    try:
        payload = json.loads(raw_text)
    except json.JSONDecodeError:
        log.warning(
            "VALIDATION_SUMMARY_INVALID_JSON account=%s path=%s",
            account_idx,
            summary_path,
            exc_info=True,
        )
        return None

    if not isinstance(payload, Mapping):
        log.warning(
            "VALIDATION_SUMMARY_INVALID_TYPE account=%s path=%s type=%s",
            account_idx,
            summary_path,
            type(payload).__name__,
        )
        return None

    return payload


def _collect_weak_items(summary: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    """Extract validation requirements that require AI adjudication."""

    if not isinstance(summary, Mapping):
        return []

    validation = summary.get("validation_requirements")
    if not isinstance(validation, Mapping):
        return []

    findings = validation.get("findings")
    if not isinstance(findings, Sequence):
        return []

    field_consistency = validation.get("field_consistency")
    if isinstance(field_consistency, Mapping):
        consistency_map: Mapping[str, Any] = field_consistency
    else:
        consistency_map = {}

    weak_items: list[dict[str, Any]] = []

    for entry in findings:
        if not isinstance(entry, Mapping):
            continue

        if not entry.get("ai_needed"):
            continue

        raw_field = entry.get("field")
        if raw_field is None:
            continue

        field = str(raw_field)

        documents = entry.get("documents")
        if isinstance(documents, Sequence) and not isinstance(
            documents, (str, bytes, bytearray)
        ):
            documents_list = [str(doc) for doc in documents]
        elif documents is None:
            documents_list = []
        else:
            documents_list = [str(documents)]

        item: dict[str, Any] = {
            "field": field,
            "category": entry.get("category"),
            "min_days": entry.get("min_days"),
            "duration_unit": entry.get("duration_unit"),
            "documents": documents_list,
        }

        if "min_days_business" in entry:
            item["min_days_business"] = entry.get("min_days_business")

        consistency_details = consistency_map.get(field)
        if isinstance(consistency_details, Mapping):
            item["consensus"] = consistency_details.get("consensus")

            disagreeing = consistency_details.get("disagreeing_bureaus")
            if isinstance(disagreeing, Sequence) and not isinstance(
                disagreeing, (str, bytes, bytearray)
            ):
                item["disagreeing_bureaus"] = sorted(str(b) for b in disagreeing)
            else:
                item["disagreeing_bureaus"] = []

            missing = consistency_details.get("missing_bureaus")
            if isinstance(missing, Sequence) and not isinstance(
                missing, (str, bytes, bytearray)
            ):
                item["missing_bureaus"] = sorted(str(b) for b in missing)
            else:
                item["missing_bureaus"] = []

            raw_values = consistency_details.get("raw")
            raw_map = raw_values if isinstance(raw_values, Mapping) else {}

            normalized_values = consistency_details.get("normalized")
            normalized_map = (
                normalized_values if isinstance(normalized_values, Mapping) else {}
            )

            values: dict[str, dict[str, Any]] = {}
            for bureau in ("transunion", "experian", "equifax"):
                values[bureau] = {
                    "raw": raw_map.get(bureau),
                    "normalized": normalized_map.get(bureau),
                }

            item["values"] = values
        else:
            item["consensus"] = None
            item["disagreeing_bureaus"] = []
            item["missing_bureaus"] = []
            item["values"] = {
                bureau: {"raw": None, "normalized": None}
                for bureau in ("transunion", "experian", "equifax")
            }

        weak_items.append(item)

    return weak_items


def _write_pack(
    path: Path,
    *,
    account_id: int,
    sid: str,
    weak_items: Sequence[Mapping[str, Any]],
) -> tuple[int, list[str]]:
    """Write ``weak_items`` as JSONL entries to ``path``.

    Returns a tuple of ``(line_count, weak_fields)`` where ``weak_fields`` is the
    ordered list of field identifiers written to the pack.
    """

    pack_lines, weak_fields = _build_pack_lines(account_id, sid, weak_items)
    _persist_pack_lines(path, pack_lines)
    return len(pack_lines), weak_fields


def _build_pack_lines(
    account_id: int, sid: str, weak_items: Sequence[Mapping[str, Any]]
) -> tuple[list[str], list[str]]:
    normalized_lines: list[str] = []
    weak_fields: list[str] = []

    for idx, entry in enumerate(weak_items, start=1):
        if not isinstance(entry, Mapping):
            continue

        line_payload = {"account_id": int(account_id), "field_index": idx, "sid": sid}
        line_payload.update({key: value for key, value in entry.items()})

        try:
            serialized = json.dumps(line_payload, sort_keys=True, ensure_ascii=False)
        except TypeError:
            log.exception("VALIDATION_PACK_SERIALIZE_FAILED account=%s", account_id)
            continue

        normalized_lines.append(serialized)
        field_name = entry.get("field")
        if isinstance(field_name, str) and field_name.strip():
            weak_fields.append(field_name.strip())

    return normalized_lines, weak_fields


def _persist_pack_lines(path: Path, lines: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    contents = "\n".join(str(line) for line in lines)
    if contents:
        contents += "\n"
    path.write_text(contents, encoding="utf-8")


def _load_model_results(path: Path) -> Mapping[str, Any] | None:
    try:
        raw_text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return _load_legacy_jsonl_results(path)
    except OSError:
        log.warning("VALIDATION_RESULTS_READ_FAILED path=%s", path, exc_info=True)
        return _load_legacy_jsonl_results(path)

    try:
        payload = json.loads(raw_text)
    except json.JSONDecodeError:
        log.warning("VALIDATION_RESULTS_INVALID_JSON path=%s", path, exc_info=True)
        return _load_legacy_jsonl_results(path)

    if isinstance(payload, Mapping):
        return payload

    log.warning(
        "VALIDATION_RESULTS_UNEXPECTED_TYPE path=%s type=%s",
        path,
        type(payload).__name__,
    )
    return _load_legacy_jsonl_results(path)


_LEGACY_RESULT_NAME_RE = re.compile(r"acc_(?P<account>\d+)\.result", re.IGNORECASE)


def _load_legacy_jsonl_results(path: Path) -> Mapping[str, Any] | None:
    legacy_path = path.with_suffix(".jsonl")
    try:
        raw_text = legacy_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    except OSError:
        log.warning(
            "VALIDATION_RESULTS_LEGACY_READ_FAILED path=%s",
            legacy_path,
            exc_info=True,
        )
        return None

    results: list[dict[str, Any]] = []
    for line_number, line in enumerate(raw_text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            log.warning(
                "VALIDATION_RESULTS_LEGACY_PARSE_FAILED path=%s line=%d",
                legacy_path,
                line_number,
                exc_info=True,
            )
            continue

        if isinstance(payload, Mapping):
            results.append(dict(payload))
        else:
            log.debug(
                "VALIDATION_RESULTS_LEGACY_NON_MAPPING path=%s line=%d type=%s",
                legacy_path,
                line_number,
                type(payload).__name__,
            )

    account_id: int | str | None = None
    match = _LEGACY_RESULT_NAME_RE.match(legacy_path.stem)
    if match:
        account_raw = match.group("account")
        try:
            account_id = int(account_raw)
        except (TypeError, ValueError):
            account_id = account_raw

    summary: dict[str, Any] = {
        "status": "done",
        "request_lines": len(results),
        "results": results,
        "completed_at": _utc_now(),
    }
    if account_id is not None:
        summary["account_id"] = account_id

    log.info(
        "VALIDATION_RESULTS_LEGACY_LOADED path=%s results=%d",
        legacy_path,
        len(results),
    )
    return summary


def _compute_source_hash(
    summary: Mapping[str, Any] | None,
    weak_items: Sequence[Mapping[str, Any]],
    pack_lines: Sequence[str],
) -> str:
    segment = _extract_weak_source_segment(summary, weak_items)
    normalized_payload = {
        "segment": segment,
        "pack_lines": list(pack_lines),
    }
    serialized = json.dumps(normalized_payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _extract_weak_source_segment(
    summary: Mapping[str, Any] | None,
    weak_items: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any]:
    findings: list[Any] = []
    fields: list[str] = []

    if isinstance(summary, Mapping):
        validation = summary.get("validation_requirements")
        if isinstance(validation, Mapping):
            raw_findings = validation.get("findings")
            if isinstance(raw_findings, Sequence):
                for entry in raw_findings:
                    if not isinstance(entry, Mapping):
                        continue
                    if not entry.get("ai_needed"):
                        continue
                    findings.append(_json_clone(entry))
                    field_name = entry.get("field")
                    if isinstance(field_name, str) and field_name.strip():
                        fields.append(field_name.strip())

            field_consistency_raw = validation.get("field_consistency")
            if isinstance(field_consistency_raw, Mapping):
                if not fields:
                    for item in weak_items:
                        if not isinstance(item, Mapping):
                            continue
                        field_val = item.get("field")
                        if isinstance(field_val, str) and field_val.strip():
                            fields.append(field_val.strip())

                consistency: dict[str, Any] = {}
                for field in sorted({field for field in fields if field}):
                    value = field_consistency_raw.get(field)
                    consistency[field] = _json_clone(value) if value is not None else None
            else:
                consistency = {}
        else:
            consistency = {}
    else:
        consistency = {}

    return {
        "findings": findings,
        "field_consistency": consistency,
    }


def _json_clone(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_clone(val) for key, val in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_json_clone(entry) for entry in value]
    return value


def _append_validation_log_entry(path: Path, entry: Mapping[str, Any]) -> None:
    try:
        serialized = json.dumps(entry, sort_keys=True, ensure_ascii=False)
    except TypeError:
        log.exception("VALIDATION_LOG_SERIALIZE_FAILED path=%s", path)
        return

    try:
        existing = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        existing = ""
    except OSError:
        log.warning("VALIDATION_LOG_READ_FAILED path=%s", path, exc_info=True)
        existing = ""

    if existing and not existing.endswith("\n"):
        existing += "\n"

    new_contents = (existing + serialized + "\n") if existing else (serialized + "\n")

    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(path.name + ".tmp")

    try:
        temp_path.write_text(new_contents, encoding="utf-8")
        temp_path.replace(path)
    except OSError:
        log.warning("VALIDATION_LOG_WRITE_FAILED path=%s", path, exc_info=True)
        with suppress(FileNotFoundError, OSError):
            temp_path.unlink(missing_ok=True)


def _render_prompt(sid: str, account_idx: int, weak_items: Sequence[Mapping[str, Any]]) -> str:
    schema_block = textwrap.dedent(
        """{
  \"sid\": \"<copy the SID from the user section>\",
  \"account_index\": <copy the account index as an integer>,
  \"decisions\": [
    {
      \"field\": \"<field name>\",
      \"decision\": \"STRONG\" | \"NO_CLAIM\"
    }
  ]
}"""
    )

    system_lines = textwrap.dedent(
        """SYSTEM:
You are an adjudication assistant reviewing credit report inconsistencies.
Evaluate each weak field independently and decide whether the consumer has a strong claim.
Follow these rules:
1. Base decisions only on the provided data for each field.
2. Return the decisions in the same order the fields are provided.
3. Use decision value \"STRONG\" when the evidence indicates a strong inconsistency; otherwise use \"NO_CLAIM\".
Return a STRICT JSON object matching this schema (no extra keys, commentary, or trailing characters):
"""
    )

    system_lines += textwrap.indent(schema_block, "  ")
    system_lines += textwrap.dedent(
        """

Set \"sid\" to the provided SID and \"account_index\" to the provided account index.
Do not include any explanation outside the JSON response.
"""
    )

    weak_fields_json = json.dumps(
        list(weak_items), indent=2, sort_keys=True, ensure_ascii=False
    )

    prompt = (
        f"{system_lines}\n\n"
        "USER:\n"
        f"SID: {sid}\n"
        f"ACCOUNT_INDEX: {account_idx}\n"
        "WEAK_FIELDS:\n"
        f"{weak_fields_json}\n"
    )

    return prompt


def _write_prompt(path: Path, prompt: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(prompt, encoding="utf-8")


def _write_model_results(path: Path, payload: Mapping[str, Any]) -> None:
    try:
        serialized = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    except TypeError:
        log.exception("VALIDATION_AI_RESULTS_SERIALIZE_FAILED path=%s", path)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(serialized + "\n", encoding="utf-8")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )


def _build_ai_client() -> Any | None:
    try:
        from backend.core.services.ai_client import get_ai_client

        return get_ai_client()
    except Exception:
        log.warning("VALIDATION_AI_CLIENT_UNAVAILABLE", exc_info=True)
        return None


def _extract_response_text(response: Any) -> str | None:
    if response is None:
        return None

    for attr in ("output_text", "text"):
        value = getattr(response, attr, None)
        if isinstance(value, str) and value.strip():
            return value

    output = getattr(response, "output", None)
    if isinstance(output, list) and output:
        first = output[0]
        content = getattr(first, "content", None)
        if isinstance(content, list) and content:
            first_content = content[0]
            text_val = getattr(first_content, "text", None)
            if isinstance(text_val, str) and text_val.strip():
                return text_val
            if isinstance(first_content, Mapping):
                text_val = first_content.get("text")
                if isinstance(text_val, str) and text_val.strip():
                    return text_val

    if isinstance(response, Mapping):
        text_val = response.get("output_text") or response.get("text")
        if isinstance(text_val, str) and text_val.strip():
            return text_val

    return None


def _run_model_inference(
    ai_client: Any | None,
    prompt: str,
    model: str,
    *,
    sid: str,
    account_idx: int | str,
    has_weak_items: bool,
    config: ValidationPacksConfig,
) -> dict[str, Any]:
    timestamp = _utc_now()

    if not has_weak_items:
        return {
            "status": "skipped",
            "reason": "no_weak_items",
            "model": model,
            "timestamp": timestamp,
            "duration_ms": 0,
            "attempts": 0,
        }

    if not config.enable_infer:
        return {
            "status": "skipped",
            "reason": "inference_disabled",
            "model": model,
            "timestamp": timestamp,
            "duration_ms": 0,
            "attempts": 0,
        }

    if ai_client is None:
        log.warning(
            "VALIDATION_AI_CLIENT_MISSING sid=%s account=%s", sid, account_idx
        )
        return {
            "status": "skipped",
            "reason": "no_client",
            "model": model,
            "timestamp": timestamp,
            "duration_ms": 0,
            "attempts": 0,
        }

    attempts = 0
    total_duration_ms = 0
    response: Any | None = None
    last_error: Exception | None = None

    max_attempts = max(1, config.max_attempts)

    while attempts < max_attempts:
        attempts += 1
        started = time.perf_counter()
        try:
            response = ai_client.response_json(
                prompt=prompt,
                model=model,
                response_format={"type": "json_object"},
            )
            total_duration_ms += int((time.perf_counter() - started) * 1000)
            last_error = None
            break
        except Exception as exc:  # pragma: no cover - defensive logging
            total_duration_ms += int((time.perf_counter() - started) * 1000)
            last_error = exc
            log.warning(
                "VALIDATION_AI_CALL_FAILED sid=%s account=%s attempt=%s error=%s",
                sid,
                account_idx,
                attempts,
                exc,
            )
            if attempts >= max_attempts:
                break
            backoff_idx = min(attempts - 1, len(config.backoff_seconds) - 1)
            delay = config.backoff_seconds[backoff_idx]
            if delay > 0:
                time.sleep(delay)

    if last_error is not None or response is None:
        reason = "unknown"
        if last_error is not None:
            reason = last_error.__class__.__name__
        return {
            "status": "error",
            "reason": reason,
            "model": model,
            "timestamp": timestamp,
            "duration_ms": total_duration_ms,
            "attempts": attempts,
        }

    raw_text = _extract_response_text(response)
    result: dict[str, Any] = {
        "status": "ok",
        "model": model,
        "timestamp": timestamp,
        "duration_ms": total_duration_ms,
        "attempts": attempts,
    }

    if raw_text is None:
        result.update({"status": "error", "reason": "empty_response"})
        return result

    result["raw"] = raw_text
    parsed, error_reason = parse_json(raw_text)
    result["response"] = parsed
    if error_reason:
        result["status"] = "error"
        result["reason"] = error_reason

    return result

