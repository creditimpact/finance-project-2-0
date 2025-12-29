"""Utilities for working with validation manifest index files."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence

from backend.core.ai.paths import (
    validation_result_jsonl_filename_for_account,
    validation_result_summary_filename_for_account,
    validation_write_json_enabled,
)

SCHEMA_VERSION = 2


log = logging.getLogger(__name__)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )


def _ensure_posix(path: Path) -> str:
    text = path.as_posix()
    return text or "."


def _relativize(target: Path, base: Path) -> str:
    target_resolved = target.resolve()
    base_resolved = base.resolve()

    try:
        relative = target_resolved.relative_to(base_resolved)
    except ValueError:
        try:
            relative = Path(os.path.relpath(target_resolved, base_resolved))
        except ValueError:
            # ``relpath`` can raise ValueError on Windows when the drive differs.
            # Fall back to the absolute POSIX representation – callers should
            # ensure paths share a common root when generating schema v2 files.
            return _ensure_posix(target_resolved)

    return _ensure_posix(relative)


def _normalize_string(value: Any, *, default: str = "") -> str:
    if isinstance(value, str):
        text = value.strip()
        return text or default
    if value is None:
        return default
    return str(value)


def _normalize_path_text(value: str) -> str:
    """Normalize schema path fields to use forward slashes."""

    if not value:
        return value
    return value.replace("\\", "/")


def _canonicalize_result_json_path(value: str) -> str:
    """Ensure legacy ``.result.json`` paths use the canonical ``.jsonl`` suffix."""

    if not value:
        return value

    candidate = PurePosixPath(value)
    if candidate.suffix.lower() != ".json":
        return value

    name = candidate.name
    if not name.endswith(".result.json"):
        return value

    if validation_write_json_enabled():
        return value

    return candidate.with_suffix(".jsonl").as_posix()


def _looks_like_windows_absolute(path: str) -> bool:
    if not path:
        return False
    if path.startswith("//"):
        return True
    if len(path) >= 3 and path[1] == ":" and path[2] == "/":
        return True
    return False


def _normalize_int(value: Any, *, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _normalize_string_list(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, Sequence) or isinstance(value, (bytes, bytearray)):
        return ()
    result: list[str] = []
    for item in value:
        if item is None:
            continue
        text = str(item).strip()
        if text:
            result.append(text)
    return tuple(result)


def _collect_unknown_fields(
    data: Mapping[str, Any], known_keys: Iterable[str]
) -> dict[str, Any]:
    known = set(known_keys)
    extras: dict[str, Any] = {}
    for key, value in data.items():
        if key in known:
            continue
        extras[key] = value
    return extras


@dataclass(frozen=True)
class ValidationPackRecord:
    """Single entry within the validation manifest index."""

    account_id: int
    pack: str
    result_jsonl: str | None
    result_json: str
    lines: int
    status: str
    built_at: str
    weak_fields: tuple[str, ...] = ()
    source_hash: str | None = None
    extra: Mapping[str, Any] = field(default_factory=dict)

    @staticmethod
    def from_mapping(data: Mapping[str, Any]) -> "ValidationPackRecord":
        account_id = _normalize_int(data.get("account_id"))
        pack_path = _normalize_string(
            data.get("pack")
            or data.get("pack_path")
            or data.get("pack_file")
            or data.get("pack_filename")
        )
        pack_path = _normalize_path_text(pack_path)
        result_jsonl_raw = _normalize_string(
            data.get("result_jsonl")
            or data.get("result_jsonl_path")
            or data.get("result_jsonl_file")
        )
        result_jsonl_raw = _normalize_path_text(result_jsonl_raw)
        result_jsonl_normalized = _canonicalize_result_json_path(result_jsonl_raw)
        result_jsonl = result_jsonl_normalized or None
        result_json = _normalize_string(
            data.get("result_json")
            or data.get("result_json_path")
            or data.get("result_summary_path")
            or data.get("result_path")
        )
        result_json = _canonicalize_result_json_path(
            _normalize_path_text(result_json)
        )

        lines = _normalize_int(data.get("lines") or data.get("line_count"))
        status = _normalize_string(data.get("status"), default="built")
        built_at = _normalize_string(data.get("built_at"), default=_utc_now())
        weak_fields = _normalize_string_list(data.get("weak_fields"))
        source_hash_value = data.get("source_hash")
        if isinstance(source_hash_value, str) and source_hash_value.strip():
            source_hash = source_hash_value.strip()
        else:
            source_hash = None

        extras = _collect_unknown_fields(
            data,
            (
                "account_id",
                "pack",
                "pack_path",
                "pack_file",
                "pack_filename",
                "result_jsonl",
                "result_jsonl_path",
                "result_jsonl_file",
                "result_json",
                "result_json_path",
                "result_summary_path",
                "result_path",
                "lines",
                "line_count",
                "status",
                "built_at",
                "weak_fields",
                "source_hash",
            ),
        )

        return ValidationPackRecord(
            account_id=account_id,
            pack=pack_path or "",
            result_jsonl=result_jsonl,
            result_json=result_json or "",
            lines=lines,
            status=status or "built",
            built_at=built_at,
            weak_fields=weak_fields,
            source_hash=source_hash,
            extra=extras,
        )

    def to_json_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "account_id": self.account_id,
            "pack": self.pack,
            "result_json": self.result_json,
            "lines": self.lines,
            "status": self.status,
            "built_at": self.built_at,
        }
        if self.result_jsonl:
            payload["result_jsonl"] = self.result_jsonl
        if self.weak_fields:
            payload["weak_fields"] = list(self.weak_fields)
        if self.source_hash:
            payload["source_hash"] = self.source_hash
        if self.extra:
            payload.update(self.extra)
        return payload


@dataclass(frozen=True)
class ValidationIndex:
    """Represents the validation manifest index."""

    index_path: Path
    sid: str
    packs_dir: str
    results_dir: str
    packs: Sequence[ValidationPackRecord]
    schema_version: int = SCHEMA_VERSION
    root: str = "."

    @property
    def index_dir(self) -> Path:
        return self.index_path.parent.resolve()

    @property
    def root_dir(self) -> Path:
        root_text = _normalize_path_text(self.root) if self.root else "."

        if _looks_like_windows_absolute(root_text):
            return Path(root_text)

        posix_path = PurePosixPath(root_text)
        if posix_path.is_absolute():
            return Path(posix_path).resolve()

        return (self.index_dir / posix_path).resolve()

    @property
    def packs_dir_path(self) -> Path:
        return self._resolve_from_root(self.packs_dir)

    @property
    def results_dir_path(self) -> Path:
        return self._resolve_from_root(self.results_dir)

    def _resolve_from_root(self, path_text: str) -> Path:
        normalized = _normalize_path_text(path_text) if path_text else "."

        if _looks_like_windows_absolute(normalized):
            return Path(normalized)

        posix_path = PurePosixPath(normalized)
        if posix_path.is_absolute():
            return Path(posix_path).resolve()

        return (self.root_dir / posix_path).resolve()

    def resolve_path(self, relative: str) -> Path:
        return self._resolve_from_root(relative)

    def resolve_pack_path(self, record: ValidationPackRecord) -> Path:
        return self.resolve_path(record.pack)

    def resolve_result_jsonl_path(self, record: ValidationPackRecord) -> Path:
        """Compute the absolute path for an account's ``.result.jsonl`` file."""

        account_value: Any = record.account_id
        try:
            account_number = int(account_value)
        except (TypeError, ValueError):
            raise ValueError(
                "Validation pack record missing valid 'account_id' for result path"
            ) from None

        results_dir_relative = Path(_normalize_path_text(self.results_dir) or ".")
        result_dir = (self.root_dir / results_dir_relative).resolve()
        filename = validation_result_jsonl_filename_for_account(account_number)
        return result_dir / filename

    def resolve_result_json_path(self, record: ValidationPackRecord) -> Path:
        return self.resolve_path(record.result_json)

    def to_json_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "sid": self.sid,
            "root": self.root,
            "packs_dir": self.packs_dir,
            "results_dir": self.results_dir,
            "packs": [record.to_json_payload() for record in self.packs],
        }

    def write(self) -> None:
        document = self.to_json_payload()
        serialized = json.dumps(document, ensure_ascii=False, indent=2)
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        self.index_path.write_text(serialized + "\n", encoding="utf-8")


def _normalize_root(packs_dir: Path, results_dir: Path, index_dir: Path) -> str:
    try:
        common_base = Path(os.path.commonpath([packs_dir, results_dir]))
    except ValueError:
        common_base = index_dir
    return _relativize(common_base, index_dir)


def build_validation_index(
    *,
    index_path: Path,
    sid: str,
    packs_dir: Path,
    results_dir: Path,
    records: Sequence[ValidationPackRecord],
) -> ValidationIndex:
    index_dir = index_path.parent.resolve()
    packs_dir_resolved = packs_dir.resolve()
    results_dir_resolved = results_dir.resolve()

    root = _normalize_root(packs_dir_resolved, results_dir_resolved, index_dir)
    root_path = (index_dir / PurePosixPath(root)).resolve()

    packs_dir_rel = _relativize(packs_dir_resolved, root_path)
    results_dir_rel = _relativize(results_dir_resolved, root_path)

    index = ValidationIndex(
        index_path=index_path,
        sid=sid,
        root=root or ".",
        packs_dir=packs_dir_rel,
        results_dir=results_dir_rel,
        packs=list(records),
    )

    validated_records: list[ValidationPackRecord] = []
    for record in index.packs:
        if record.lines <= 0:
            log.warning(
                "VALIDATION_INDEX_SKIP_EMPTY_LINES sid=%s account_id=%s pack=%s",
                sid,
                record.account_id,
                record.pack,
            )
            continue

        try:
            pack_path = index.resolve_pack_path(record)
        except Exception:  # pragma: no cover - defensive
            log.warning(
                "VALIDATION_INDEX_RESOLVE_FAILED sid=%s account_id=%s pack=%s",
                sid,
                record.account_id,
                record.pack,
                exc_info=True,
            )
            continue

        try:
            exists = pack_path.exists()
        except OSError:
            exists = False

        if not exists or not pack_path.is_file():
            log.warning(
                "VALIDATION_INDEX_PACK_MISSING sid=%s account_id=%s path=%s",
                sid,
                record.account_id,
                pack_path,
            )
            continue

        try:
            size = pack_path.stat().st_size
        except OSError:
            size = 0

        if size <= 0:
            log.warning(
                "VALIDATION_INDEX_PACK_EMPTY sid=%s account_id=%s path=%s",
                sid,
                record.account_id,
                pack_path,
            )
            continue

        validated_records.append(record)

    if len(validated_records) != len(index.packs):
        index = ValidationIndex(
            index_path=index.index_path,
            sid=index.sid,
            root=index.root,
            packs_dir=index.packs_dir,
            results_dir=index.results_dir,
            packs=tuple(validated_records),
            schema_version=index.schema_version,
        )

    return index


def load_validation_index(path: Path | str) -> ValidationIndex:
    index_path = Path(path)
    text = index_path.read_text(encoding="utf-8")
    document = json.loads(text)
    if not isinstance(document, Mapping):
        raise TypeError("Validation index root must be a mapping")
    return _index_from_document(document, index_path)


def _index_from_document(document: Mapping[str, Any], index_path: Path) -> ValidationIndex:
    schema_version = _normalize_int(document.get("schema_version"), default=1)

    if schema_version >= 2:
        sid = _normalize_string(document.get("sid"))
        root = _normalize_path_text(
            _normalize_string(document.get("root"), default=".")
        )
        packs_dir = _normalize_path_text(
            _normalize_string(document.get("packs_dir"), default="packs")
        )
        results_dir = _normalize_path_text(
            _normalize_string(document.get("results_dir"), default="results")
        )

        raw_packs = document.get("packs")
        entries: list[ValidationPackRecord] = []
        if isinstance(raw_packs, Sequence):
            for pack in raw_packs:
                if isinstance(pack, Mapping):
                    entries.append(ValidationPackRecord.from_mapping(pack))

        return ValidationIndex(
            index_path=index_path,
            sid=sid,
            root=root or ".",
            packs_dir=packs_dir or "packs",
            results_dir=results_dir or "results",
            packs=entries,
            schema_version=schema_version,
        )

    # legacy schema (v1)
    sid = _normalize_string(document.get("sid"))
    packs_dir_raw = document.get("packs_dir")
    results_dir_raw = document.get("results_dir")
    index_dir = index_path.parent.resolve()

    packs_dir = Path(_normalize_string(packs_dir_raw)).resolve()
    results_dir = Path(_normalize_string(results_dir_raw)).resolve()

    raw_items = document.get("items")
    records: list[ValidationPackRecord] = []
    if isinstance(raw_items, Sequence):
        for item in raw_items:
            if not isinstance(item, Mapping):
                continue
            account_id = _normalize_int(item.get("account_id"))
            pack_raw = _normalize_string(item.get("pack"))
            pack_path = Path(pack_raw).resolve() if pack_raw else packs_dir

            pack_rel = _relativize(pack_path, index_dir)
            result_jsonl_path = (
                results_dir / validation_result_jsonl_filename_for_account(account_id)
            )
            result_summary_path = (
                results_dir / validation_result_summary_filename_for_account(account_id)
            )
            record = ValidationPackRecord(
                account_id=account_id,
                pack=pack_rel,
                result_jsonl=_relativize(result_jsonl_path, index_dir),
                result_json=_relativize(result_summary_path, index_dir),
                lines=_normalize_int(item.get("lines")),
                status=_normalize_string(item.get("status"), default="built"),
                built_at=_normalize_string(item.get("built_at"), default=_utc_now()),
                weak_fields=_normalize_string_list(item.get("weak_fields")),
                source_hash=_normalize_string(item.get("source_hash")) or None,
                extra=_collect_unknown_fields(
                    item,
                    (
                        "account_id",
                        "account_key",
                        "pack",
                        "lines",
                        "status",
                        "built_at",
                        "weak_fields",
                        "source_hash",
                    ),
                ),
            )
            records.append(record)

    return build_validation_index(
        index_path=index_path,
        sid=sid,
        packs_dir=packs_dir,
        results_dir=results_dir,
        records=records,
    )


__all__ = [
    "SCHEMA_VERSION",
    "ValidationIndex",
    "ValidationPackRecord",
    "build_validation_index",
    "load_validation_index",
]

