"""Helpers to maintain the validation AI pack index."""

from __future__ import annotations

import json
import logging
import os
import tempfile
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Iterable, Mapping, Sequence

from backend.validation.index_schema import (
    ValidationIndex,
    ValidationPackRecord,
    load_validation_index,
)

log = logging.getLogger(__name__)

_SCHEMA_VERSION = 2


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )


@dataclass(frozen=True)
class ValidationIndexEntry:
    """Single entry describing a validation pack/result pair."""

    account_id: int
    pack_path: Path
    result_jsonl_path: Path | None
    result_json_path: Path | None
    weak_fields: Sequence[str]
    line_count: int
    status: str
    built_at: str | None = None
    request_lines: int | None = None
    model: str | None = None
    sent_at: str | None = None
    completed_at: str | None = None
    error: str | None = None
    source_hash: str | None = None
    extra: Mapping[str, object] = field(default_factory=dict)

    def to_json_payload(self, index_dir: Path) -> dict[str, object]:
        weak_fields = [str(field) for field in self.weak_fields if str(field).strip()]
        payload: dict[str, object] = {
            "account_id": int(self.account_id),
            "pack": _relativize_path(self.pack_path, index_dir),
            "weak_fields": weak_fields,
            "lines": int(self.line_count),
            "built_at": str(self.built_at or _utc_now()),
            "status": str(self.status or "built"),
        }

        if self.result_json_path is not None:
            payload["result_json"] = _relativize_path(
                self.result_json_path, index_dir
            )

        if self.result_jsonl_path is not None:
            payload["result_jsonl"] = _relativize_path(
                self.result_jsonl_path, index_dir
            )

        if self.request_lines is not None:
            normalized_request = _normalize_optional_int(self.request_lines)
            if normalized_request is not None:
                payload["request_lines"] = normalized_request
        if self.model is not None:
            normalized_model = _normalize_optional_str(self.model)
            if normalized_model is not None:
                payload["model"] = normalized_model
        if self.sent_at:
            normalized_sent = _normalize_optional_str(self.sent_at)
            if normalized_sent is not None:
                payload["sent_at"] = normalized_sent
        if self.completed_at:
            normalized_completed = _normalize_optional_str(self.completed_at)
            if normalized_completed is not None:
                payload["completed_at"] = normalized_completed
        if self.error:
            normalized_error = _normalize_optional_str(self.error)
            if normalized_error is not None:
                payload["error"] = normalized_error
        if self.source_hash:
            payload["source_hash"] = str(self.source_hash)

        for key, value in (self.extra or {}).items():
            if key in payload:
                continue
            payload[key] = value

        return payload


def _atomic_write_json(path: Path, document: Mapping[str, object]) -> None:
    """
    Atomically write JSON document to disk with retry logic for Windows file locking.
    
    On Windows, os.replace() can fail with PermissionError if the target file is
    temporarily locked by another process (antivirus, search indexer, etc.).
    We retry a few times with short backoff to handle transient locks.
    """
    try:
        serialized = json.dumps(document, ensure_ascii=False, indent=2)
    except TypeError:
        log.exception("VALIDATION_INDEX_SERIALIZE_FAILED path=%s", path)
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    
    # Write to temp file first
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as handle:
            handle.write(serialized)
            handle.write("\n")
    except OSError:
        log.warning("VALIDATION_INDEX_TMP_WRITE_FAILED path=%s tmp=%s", path, tmp_name, exc_info=True)
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        return
    
    # Attempt atomic replace with retry for Windows PermissionError
    MAX_RETRIES = 5
    replace_succeeded = False
    
    for attempt in range(MAX_RETRIES):
        try:
            os.replace(tmp_name, path)
            replace_succeeded = True
            break
        except PermissionError as exc:
            if attempt == MAX_RETRIES - 1:
                # Final attempt failed
                log.warning(
                    "VALIDATION_INDEX_WRITE_FAILED path=%s attempts=%s reason=permission_error",
                    path,
                    MAX_RETRIES,
                    exc_info=True,
                )
                break
            else:
                # Retry with exponential backoff
                backoff_seconds = 0.05 * (2 ** attempt)  # 0.05, 0.1, 0.2, 0.4 seconds
                log.warning(
                    "VALIDATION_INDEX_WRITE_RETRY path=%s attempt=%s/%s backoff=%.3fs",
                    path,
                    attempt + 1,
                    MAX_RETRIES,
                    backoff_seconds,
                )
                import time
                time.sleep(backoff_seconds)
        except OSError as exc:
            # Other OS errors (not permission) - fail immediately
            log.warning("VALIDATION_INDEX_WRITE_FAILED path=%s reason=os_error", path, exc_info=True)
            break
    
    # Clean up temp file if replace failed
    if not replace_succeeded:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass


def _ensure_posix(relative: Path | PurePosixPath | str) -> str:
    return str(PurePosixPath(str(relative)))


def _relativize_path(path: Path, base_dir: Path) -> str:
    resolved_path = Path(path).resolve()
    resolved_base = Path(base_dir).resolve()
    try:
        relative = resolved_path.relative_to(resolved_base)
    except ValueError:
        relative = Path(os.path.relpath(resolved_path, resolved_base))
    return _ensure_posix(relative)


def _normalize_optional_str(value: object) -> str | None:
    if isinstance(value, str):
        text = value.strip()
        return text or None
    if value is None:
        return None
    return str(value)


def _normalize_optional_int(value: object) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def write_validation_manifest_v2(
    sid: str,
    packs_dir: Path,
    results_dir: Path,
    entries: Iterable[ValidationIndexEntry],
    *,
    index_path: Path,
) -> None:
    index_dir = index_path.parent.resolve()
    normalized_entries = [
        entry.to_json_payload(index_dir) for entry in entries
    ]

    normalized_entries.sort(
        key=lambda item: (
            _safe_int(item.get("account_id")),
            str(item.get("pack") or ""),
        )
    )

    document = {
        "schema_version": _SCHEMA_VERSION,
        "sid": sid,
        "root": ".",
        "packs_dir": _relativize_path(packs_dir, index_dir),
        "results_dir": _relativize_path(results_dir, index_dir),
        "packs": normalized_entries,
    }

    _atomic_write_json(index_path, document)


class ValidationPackIndexWriter:
    """Maintain the consolidated validation pack index file."""

    def __init__(
        self,
        *,
        sid: str,
        index_path: Path,
        packs_dir: Path,
        results_dir: Path,
    ) -> None:
        self.sid = str(sid)
        self._index_path = Path(index_path)
        self._packs_dir = Path(packs_dir)
        self._results_dir = Path(results_dir)
        self._index_path.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def load_accounts(self) -> dict[int, dict[str, object]]:
        """Return a mapping of account id to existing index entries."""

        entries = self._load_entries()
        index_dir = self._index_path.parent.resolve()
        result: dict[int, dict[str, object]] = {}
        for entry in entries.values():
            payload = entry.to_json_payload(index_dir)
            result[int(entry.account_id)] = payload
        return result

    def upsert(self, entry: ValidationIndexEntry) -> None:
        self.bulk_upsert([entry])

    def bulk_upsert(self, entries: Iterable[ValidationIndexEntry]) -> None:
        new_entries = [entry for entry in entries]
        if not new_entries:
            return

        validated_entries: list[ValidationIndexEntry] = []
        for entry in new_entries:
            if entry.line_count < 0:
                log.warning(
                    "VALIDATION_INDEX_SKIP_EMPTY_LINES sid=%s account_id=%s pack=%s",
                    self.sid,
                    entry.account_id,
                    entry.pack_path,
                )
                continue

            try:
                exists = entry.pack_path.exists()
            except OSError:
                exists = False

            if not exists or not entry.pack_path.is_file():
                log.warning(
                    "VALIDATION_INDEX_SKIP_MISSING_PACK sid=%s account_id=%s path=%s",
                    self.sid,
                    entry.account_id,
                    entry.pack_path,
                )
                continue

            validated_entries.append(entry)

        if not validated_entries:
            return

        existing = self._load_entries()
        for entry in validated_entries:
            existing[self._entry_key(entry)] = entry

        write_validation_manifest_v2(
            self.sid,
            self._packs_dir,
            self._results_dir,
            existing.values(),
            index_path=self._index_path,
        )

    def mark_sent(
        self,
        pack_path: Path | str,
        *,
        request_lines: int | None = None,
        model: str | None = None,
    ) -> dict[str, object] | None:
        """Update the index entry for ``pack_path`` to ``sent`` status."""

        set_values: dict[str, object] = {
            "status": "sent",
            "sent_at": _utc_now(),
        }
        if request_lines is not None:
            normalized_request = _normalize_optional_int(request_lines)
            if normalized_request is not None:
                set_values["request_lines"] = normalized_request
        if model is not None:
            normalized_model = _normalize_optional_str(model)
            if normalized_model is not None:
                set_values["model"] = normalized_model

        return self._update_entry_fields(
            Path(pack_path),
            set_values=set_values,
            remove_keys=("completed_at", "error"),
        )

    def record_result(
        self,
        pack_path: Path | str,
        *,
        status: str,
        error: str | None = None,
        request_lines: int | None = None,
        model: str | None = None,
        completed_at: str | None = None,
        result_path: Path | str | None = None,
        line_count: int | None = None,
    ) -> dict[str, object] | None:
        """Persist the final status for ``pack_path`` in the index."""

        normalized_status = str(status).strip().lower()
        if normalized_status in {"done", "completed"}:
            normalized_status = "completed"
        elif normalized_status in {"error", "failed"}:
            normalized_status = "failed"
        else:
            raise ValueError("status must be 'done'/'completed' or 'error'/'failed'")

        set_values: dict[str, object] = {
            "status": normalized_status,
            "completed_at": completed_at or _utc_now(),
        }

        if request_lines is not None:
            normalized_request = _normalize_optional_int(request_lines)
            if normalized_request is not None:
                set_values["request_lines"] = normalized_request

        if model is not None:
            normalized_model = _normalize_optional_str(model)
            if normalized_model is not None:
                set_values["model"] = normalized_model

        if line_count is not None:
            normalized_lines = _normalize_optional_int(line_count)
            if normalized_lines is not None:
                set_values["lines"] = normalized_lines

        if normalized_status == "failed":
            normalized_error = _normalize_optional_str(error) or "unknown"
            set_values["error"] = normalized_error
            if result_path is None:
                remove_keys = ("result_json", "result_jsonl", "results_path")
            else:
                remove_keys = ("result_jsonl",)
        else:
            remove_keys = ("error",)
            if result_path is not None:
                set_values["result_json"] = Path(result_path)
            else:
                remove_keys = remove_keys + ("result_json", "result_jsonl", "results_path")

        if result_path is None and normalized_status != "failed":
            remove_keys = remove_keys + ("result_json", "result_jsonl", "results_path")

        return self._update_entry_fields(
            Path(pack_path),
            set_values=set_values,
            remove_keys=remove_keys,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _entry_key(self, entry: ValidationIndexEntry) -> str:
        return _relativize_path(entry.pack_path, self._index_path.parent)

    def _load_entries(self) -> dict[str, ValidationIndexEntry]:
        try:
            index = load_validation_index(self._index_path)
        except FileNotFoundError:
            return {}
        except Exception:  # pragma: no cover - defensive logging
            log.warning(
                "VALIDATION_INDEX_READ_FAILED path=%s", self._index_path, exc_info=True
            )
            return {}

        entries: dict[str, ValidationIndexEntry] = {}
        for record in index.packs:
            entry = self._entry_from_record(index, record)
            try:
                exists = entry.pack_path.exists()
            except OSError:
                exists = False

            if not exists or not entry.pack_path.is_file():
                log.warning(
                    "VALIDATION_INDEX_PACK_MISSING sid=%s account_id=%s path=%s index=%s",
                    self.sid,
                    entry.account_id,
                    entry.pack_path,
                    self._index_path,
                )
                continue

            try:
                size = entry.pack_path.stat().st_size
            except OSError:
                size = 0

            if size <= 0:
                log.warning(
                    "VALIDATION_INDEX_PACK_EMPTY sid=%s account_id=%s path=%s index=%s",
                    self.sid,
                    entry.account_id,
                    entry.pack_path,
                    self._index_path,
                )
                continue

            entries[self._entry_key(entry)] = entry
        return entries

    def _entry_from_record(
        self, index: ValidationIndex, record: ValidationPackRecord
    ) -> ValidationIndexEntry:
        pack_path = index.resolve_pack_path(record)
        if record.result_jsonl:
            result_jsonl_path = index.resolve_result_jsonl_path(record)
        else:
            result_jsonl_path = None

        result_json_path: Path | None
        if record.result_json:
            try:
                result_json_path = index.resolve_result_json_path(record)
            except ValueError:
                result_json_path = None
            except FileNotFoundError:  # pragma: no cover - defensive
                result_json_path = None
        else:
            result_json_path = None

        extra: dict[str, object] = dict(record.extra)
        request_lines = _normalize_optional_int(extra.pop("request_lines", None))
        model = _normalize_optional_str(extra.pop("model", None))
        sent_at = _normalize_optional_str(extra.pop("sent_at", None))
        completed_at = _normalize_optional_str(extra.pop("completed_at", None))
        error = _normalize_optional_str(extra.pop("error", None))

        status = _normalize_optional_str(record.status) or "built"
        built_at = _normalize_optional_str(record.built_at) or _utc_now()
        source_hash = _normalize_optional_str(record.source_hash)
        weak_fields = tuple(
            str(field).strip()
            for field in record.weak_fields
            if str(field).strip()
        )

        return ValidationIndexEntry(
            account_id=record.account_id,
            pack_path=pack_path,
            result_jsonl_path=result_jsonl_path,
            result_json_path=result_json_path,
            weak_fields=weak_fields,
            line_count=record.lines,
            status=status,
            built_at=built_at,
            request_lines=request_lines,
            model=model,
            sent_at=sent_at,
            completed_at=completed_at,
            error=error,
            source_hash=source_hash,
            extra=extra,
        )

    def _mutate_entry(
        self,
        entry: ValidationIndexEntry,
        *,
        set_values: Mapping[str, object],
        remove_keys: Iterable[str] = (),
    ) -> ValidationIndexEntry:
        extra = dict(entry.extra)
        updates: dict[str, object] = {}

        for key in remove_keys:
            normalized_key = str(key)
            if normalized_key in {"request_lines", "model", "sent_at", "completed_at", "error"}:
                updates[normalized_key] = None
            elif normalized_key in {"result_json", "result_jsonl", "results_path"}:
                if normalized_key == "result_jsonl":
                    updates["result_jsonl_path"] = None
                else:
                    updates["result_json_path"] = None
            else:
                extra.pop(normalized_key, None)

        for key, value in set_values.items():
            normalized_key = str(key)
            if normalized_key == "status":
                updates["status"] = _normalize_optional_str(value) or entry.status
            elif normalized_key == "built_at":
                updates["built_at"] = _normalize_optional_str(value) or entry.built_at or _utc_now()
            elif normalized_key == "request_lines":
                updates["request_lines"] = _normalize_optional_int(value)
            elif normalized_key == "model":
                updates["model"] = _normalize_optional_str(value)
            elif normalized_key == "sent_at":
                updates["sent_at"] = _normalize_optional_str(value)
            elif normalized_key == "completed_at":
                updates["completed_at"] = _normalize_optional_str(value)
            elif normalized_key == "error":
                updates["error"] = _normalize_optional_str(value)
            elif normalized_key in {"result_json", "result_json_path", "results_path"}:
                if value in {None, ""}:
                    updates["result_json_path"] = None
                else:
                    updates["result_json_path"] = Path(str(value))
            elif normalized_key in {"result_jsonl", "result_jsonl_path"}:
                if value in {None, ""}:
                    updates["result_jsonl_path"] = None
                else:
                    updates["result_jsonl_path"] = Path(str(value))
            elif normalized_key in {"lines", "line_count"}:
                normalized_lines = _normalize_optional_int(value)
                if normalized_lines is not None:
                    updates["line_count"] = normalized_lines
            else:
                extra[normalized_key] = value

        updates["extra"] = extra
        return replace(entry, **updates)

    def _update_entry_fields(
        self,
        pack_path: Path,
        *,
        set_values: Mapping[str, object],
        remove_keys: Iterable[str] = (),
    ) -> dict[str, object] | None:
        entries = self._load_entries()
        key = _relativize_path(Path(pack_path), self._index_path.parent)
        entry = entries.get(key)
        if entry is None:
            return None

        updated_entry = self._mutate_entry(
            entry,
            set_values=set_values,
            remove_keys=remove_keys,
        )
        entries[key] = updated_entry

        write_validation_manifest_v2(
            self.sid,
            self._packs_dir,
            self._results_dir,
            entries.values(),
            index_path=self._index_path,
        )

        log.info(
            "VALIDATION_INDEX_STATUS_TRANSITION sid=%s pack=%s from=%s to=%s",
            self.sid,
            key,
            entry.status,
            updated_entry.status,
        )

        return updated_entry.to_json_payload(self._index_path.parent.resolve())


def _safe_int(value: object) -> int:
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0
