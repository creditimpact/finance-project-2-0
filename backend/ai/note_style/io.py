"""Filesystem helpers for inspecting note_style stage artifacts."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

from backend.core.ai.paths import (
    NoteStylePaths,
    ensure_note_style_paths,
    note_style_pack_filename,
    normalize_note_style_account_id,
)


log = logging.getLogger(__name__)


@dataclass(frozen=True)
class NoteStyleSnapshot:
    """Represents the current on-disk state of the note_style stage."""

    packs_expected: set[str]
    packs_built: set[str]
    packs_completed: set[str]
    packs_failed: set[str]


@dataclass(frozen=True)
class NoteStyleStageView:
    """Derived lifecycle information for the note_style stage."""

    packs_expected: frozenset[str]
    packs_built: frozenset[str]
    packs_completed: frozenset[str]
    packs_failed: frozenset[str]
    state: str
    built_complete: bool

    @property
    def total_expected(self) -> int:
        return len(self.packs_expected)

    @property
    def built_total(self) -> int:
        return len(self.packs_built & self.packs_expected)

    @property
    def completed_total(self) -> int:
        return len(self.packs_expected & self.packs_completed)

    @property
    def failed_total(self) -> int:
        return len(self.packs_expected & self.packs_failed)

    @property
    def terminal_accounts(self) -> frozenset[str]:
        return frozenset((self.packs_completed | self.packs_failed) & self.packs_expected)

    @property
    def terminal_total(self) -> int:
        return len(self.terminal_accounts)

    @property
    def missing_builds(self) -> frozenset[str]:
        return frozenset(self.packs_expected - self.packs_built)

    @property
    def pending_results(self) -> frozenset[str]:
        return frozenset(self.packs_expected - self.terminal_accounts)

    @property
    def ready_to_send(self) -> frozenset[str]:
        return frozenset((self.packs_built - self.terminal_accounts) & self.packs_expected)

    @property
    def has_expected(self) -> bool:
        return bool(self.packs_expected)

    @property
    def is_terminal(self) -> bool:
        return self.state in {"success", "empty", "failed"}


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


def _extract_note_text(payload: object) -> str:
    if payload is None:
        return ""

    if isinstance(payload, str):
        return payload.strip()

    if isinstance(payload, Mapping):
        for path in _NOTE_VALUE_PATHS:
            current: object = payload
            for key in path:
                if not isinstance(current, Mapping):
                    break
                current = current.get(key)
            else:
                text = _extract_note_text(current)
                if text:
                    return text

        for value in payload.values():
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


def _discover_response_accounts(responses_dir: Path) -> set[str]:
    discovered: set[str] = set()

    try:
        entries = list(responses_dir.glob("*.result.json"))
    except FileNotFoundError:
        return discovered
    except NotADirectoryError:
        return discovered

    for entry in entries:
        if not entry.is_file():
            continue

        payload = _load_json_mapping(entry)
        if not isinstance(payload, Mapping):
            continue

        note_text = _extract_note_text(payload)
        if not note_text.strip():
            continue

        account_id = entry.stem.replace(".result", "")
        normalized = normalize_note_style_account_id(account_id)
        if normalized:
            discovered.add(normalized)

    return discovered


def _resolve_runs_root(runs_root: Path | str | None) -> Path:
    if runs_root is None:
        env_value = os.getenv("RUNS_ROOT")
        return Path(env_value) if env_value else Path("runs")
    return Path(runs_root)


def _coerce_stage_paths(sid: str, runs_root: Path) -> NoteStylePaths | None:
    try:
        return ensure_note_style_paths(runs_root, sid, create=False)
    except Exception:  # pragma: no cover - defensive logging
        log.debug(
            "NOTE_STYLE_SNAPSHOT_FALLBACK sid=%s runs_root=%s", sid, runs_root, exc_info=True
        )
    return None


def _load_json_mapping(path: Path) -> Mapping[str, object] | None:
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    except OSError:  # pragma: no cover - defensive logging
        log.debug("NOTE_STYLE_SNAPSHOT_READ_FAILED path=%s", path, exc_info=True)
        return None

    text = raw.strip()
    if not text:
        return None

    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        for line in raw.splitlines():
            line_text = line.strip()
            if not line_text:
                continue
            try:
                payload = json.loads(line_text)
            except json.JSONDecodeError:
                continue
            else:
                break
        else:
            return None

    if isinstance(payload, Mapping):
        return payload
    return None


def _normalize_pack_name(name: str) -> str | None:
    if not name.startswith("acc_"):
        return None
    remainder = name[len("acc_") :]
    if not remainder:
        return None
    account_piece = remainder.split(".", 1)[0]
    normalized = normalize_note_style_account_id(account_piece)
    return normalized or None


def _resolve_index_entries(
    *,
    index_payload: Mapping[str, object] | None,
    stage_base: Path,
) -> dict[str, Path | None]:
    if not isinstance(index_payload, Mapping):
        return {}

    packs_payload = index_payload.get("packs")
    if not isinstance(packs_payload, list):
        packs_payload = index_payload.get("items") if isinstance(index_payload, Mapping) else None
    if not isinstance(packs_payload, list):
        return {}

    resolved: dict[str, Path | None] = {}
    for entry in packs_payload:
        if not isinstance(entry, Mapping):
            continue
        account_value = entry.get("account_id") or entry.get("account")
        normalized = normalize_note_style_account_id(account_value)
        if not normalized:
            continue
        pack_value = entry.get("pack_path") or entry.get("pack")
        pack_path: Path | None = None
        if isinstance(pack_value, str) and pack_value.strip():
            candidate = Path(pack_value)
            pack_path = (stage_base / candidate).resolve() if not candidate.is_absolute() else candidate.resolve()
        resolved.setdefault(normalized, pack_path)
    return resolved


def _gather_pack_files(packs_dir: Path) -> dict[str, Path]:
    results: dict[str, Path] = {}
    try:
        entries = list(packs_dir.iterdir())
    except FileNotFoundError:
        return results
    except NotADirectoryError:
        return results

    for entry in entries:
        if not entry.is_file():
            continue
        normalized = _normalize_pack_name(entry.name)
        if not normalized:
            continue
        results.setdefault(normalized, entry.resolve())
    return results


def _payload_indicates_failure(payload: Mapping[str, object]) -> bool:
    status_value = payload.get("status")
    if isinstance(status_value, str) and status_value.strip().lower() in {"failed", "error"}:
        return True

    error_value = payload.get("error")
    if isinstance(error_value, Mapping):
        return bool(error_value)
    if isinstance(error_value, str):
        return bool(error_value.strip())
    if error_value not in (None, "", {}):
        return True
    return False


def _collect_result_sets(results_dir: Path) -> tuple[set[str], set[str]]:
    completed: set[str] = set()
    failed: set[str] = set()

    try:
        entries = list(results_dir.iterdir())
    except FileNotFoundError:
        return completed, failed
    except NotADirectoryError:
        return completed, failed

    for entry in entries:
        if not entry.is_file():
            continue
        if ".tmp" in entry.name:
            continue
        normalized = _normalize_pack_name(entry.name)
        if not normalized:
            continue
        payload = _load_json_mapping(entry)
        if not isinstance(payload, Mapping):
            continue
        if _payload_indicates_failure(payload):
            failed.add(normalized)
        else:
            completed.add(normalized)
    return completed, failed


def note_style_snapshot(
    sid: str,
    *,
    runs_root: Path | str | None = None,
) -> NoteStyleSnapshot:
    """Return a consistent snapshot of note_style packs and results for ``sid``."""

    sid_text = str(sid or "").strip()
    if not sid_text:
        return NoteStyleSnapshot(set(), set(), set(), set())

    runs_root_path = _resolve_runs_root(runs_root)
    paths = _coerce_stage_paths(sid_text, runs_root_path)

    if paths is None:
        stage_base = (runs_root_path / sid_text / "ai_packs" / "note_style").resolve()
        packs_dir = stage_base / "packs"
        results_dir = stage_base / "results"
        index_file = stage_base / "index.json"
    else:
        stage_base = paths.base
        packs_dir = paths.packs_dir
        results_dir = paths.results_dir
        index_file = paths.index_file

    index_payload = _load_json_mapping(index_file)
    expected_map: dict[str, Path | None] = _resolve_index_entries(
        index_payload=index_payload,
        stage_base=stage_base,
    )

    # Derive completion/failure from index statuses if provided
    index_completed: set[str] = set()
    index_failed: set[str] = set()
    if isinstance(index_payload, Mapping):
        packs_payload = index_payload.get("packs")
        if not isinstance(packs_payload, list):
            packs_payload = index_payload.get("items") if isinstance(index_payload, Mapping) else None
        if isinstance(packs_payload, list):
            for entry in packs_payload:
                if not isinstance(entry, Mapping):
                    continue
                account_value = entry.get("account_id") or entry.get("account")
                normalized = normalize_note_style_account_id(account_value)
                if not normalized:
                    continue
                status_value = entry.get("status")
                if isinstance(status_value, str):
                    status_norm = status_value.strip().lower()
                    if status_norm == "completed":
                        index_completed.add(normalized)
                    elif status_norm in {"failed", "error"}:
                        index_failed.add(normalized)

    pack_files = _gather_pack_files(packs_dir)
    packs_completed, packs_failed = _collect_result_sets(results_dir)
    # Include index-declared terminal statuses
    packs_completed |= index_completed
    packs_failed |= index_failed

    responses_dir = runs_root_path / sid_text / "frontend" / "review" / "responses"
    response_accounts = _discover_response_accounts(responses_dir)

    if not expected_map:
        if pack_files:
            expected_map = {account: path for account, path in pack_files.items()}
        elif packs_completed or packs_failed:
            expected_map = {account: None for account in packs_completed | packs_failed}

    packs_expected = set(expected_map.keys())
    if response_accounts:
        packs_expected.update(response_accounts)
    packs_built: set[str] = set()

    for account, pack_path in expected_map.items():
        candidate = pack_path
        if candidate is None:
            candidate = packs_dir / note_style_pack_filename(account)
        try:
            exists = candidate.is_file()
        except OSError:  # pragma: no cover - defensive logging
            exists = False
        if exists:
            packs_built.add(account)

    packs_built.update(pack_files.keys())

    return NoteStyleSnapshot(
        packs_expected=packs_expected,
        packs_built=packs_built,
        packs_completed=packs_completed,
        packs_failed=packs_failed,
    )


def _determine_stage_state(
    *,
    packs_expected: set[str],
    packs_built: set[str],
    packs_completed: set[str],
    packs_failed: set[str],
) -> tuple[str, bool]:
    expected_total = len(packs_expected)
    built_total = len(packs_expected & packs_built)
    completed_total = len(packs_expected & packs_completed)
    failed_total = len(packs_expected & packs_failed)
    terminal_total = completed_total + failed_total

    built_complete = bool(expected_total) and built_total == expected_total

    if expected_total == 0:
        return ("empty", False)
    if not built_complete:
        return ("pending", False)
    if terminal_total == 0:
        return ("built", True)
    if terminal_total < expected_total:
        return ("in_progress", True)
    if expected_total and completed_total == 0 and failed_total >= expected_total:
        return ("failed", True)
    return ("success", True)


def note_style_stage_view(
    sid: str,
    *,
    runs_root: Path | str | None = None,
    snapshot: NoteStyleSnapshot | None = None,
) -> NoteStyleStageView:
    """Return the derived lifecycle view for ``sid``."""

    sid_text = str(sid or "").strip()
    if not sid_text:
        return NoteStyleStageView(
            packs_expected=frozenset(),
            packs_built=frozenset(),
            packs_completed=frozenset(),
            packs_failed=frozenset(),
            state="empty",
            built_complete=False,
        )

    snapshot_value = snapshot or note_style_snapshot(sid_text, runs_root=runs_root)

    packs_expected = set(snapshot_value.packs_expected)
    packs_built = set(snapshot_value.packs_built) & packs_expected
    packs_completed = set(snapshot_value.packs_completed) & packs_expected
    packs_failed = set(snapshot_value.packs_failed) & packs_expected

    state, built_complete = _determine_stage_state(
        packs_expected=packs_expected,
        packs_built=packs_built,
        packs_completed=packs_completed,
        packs_failed=packs_failed,
    )

    return NoteStyleStageView(
        packs_expected=frozenset(packs_expected),
        packs_built=frozenset(packs_built),
        packs_completed=frozenset(packs_completed),
        packs_failed=frozenset(packs_failed),
        state=state,
        built_complete=built_complete,
    )


__all__ = [
    "NoteStyleSnapshot",
    "NoteStyleStageView",
    "note_style_snapshot",
    "note_style_stage_view",
]
