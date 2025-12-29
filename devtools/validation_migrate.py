"""Utility to migrate validation AI packs into the consolidated layout.

Usage::

    python devtools/validation_migrate.py <SID> [--runs-root /path/to/runs]

The script moves legacy per-account validation folders into the new structure::

    ai_packs/validation/
      packs/val_acc_<ACC>.jsonl
      packs/val_acc_<ACC>.jsonl.prompt.txt
      results/acc_<ACC>.result.jsonl
      results/acc_<ACC>.result.json
      index.json

It is safe to run multiple times and skips accounts that are already migrated.

PowerShell snippet to remove empty directories after migration::

    Get-ChildItem -Path "<validation_base>" -Directory -Recurse \
      | Where-Object { ($_.GetFileSystemInfos().Count -eq 0) } \
      | Remove-Item -Force
"""

from __future__ import annotations

import argparse
import json
import sys
from contextlib import suppress
from pathlib import Path
from typing import Iterable, Mapping

from backend.ai.validation_index import (
    ValidationIndexEntry,
    ValidationPackIndexWriter,
    write_validation_manifest_v2,
)
from backend.core.ai.paths import (
    ensure_validation_paths,
    validation_pack_filename_for_account,
    validation_result_jsonl_filename_for_account,
    validation_result_filename_for_account,
)
from backend.core.logic.validation_ai_packs import _write_pack as write_pack_lines


def _parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sid", help="SID to migrate")
    parser.add_argument(
        "--runs-root",
        default=Path("runs"),
        type=Path,
        help="Root directory containing run folders (default: runs)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show actions without mutating the filesystem",
    )
    return parser.parse_args(list(argv))


def _load_legacy_weak_items(path: Path) -> list[Mapping[str, object]]:
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return []
    except OSError:
        return []

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return []

    if isinstance(payload, Mapping):
        weak_items = payload.get("weak_items")
        if isinstance(weak_items, list):
            return [item for item in weak_items if isinstance(item, Mapping)]
    return []


def _load_existing_pack_items(path: Path) -> list[Mapping[str, object]]:
    if not path.exists():
        return []
    items: list[Mapping[str, object]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, Mapping):
            continue
        normalized = {key: value for key, value in payload.items() if key not in {"sid", "account_id", "field_index"}}
        items.append(normalized)
    return items


def _merge_items(existing: list[Mapping[str, object]], new: list[Mapping[str, object]]) -> list[Mapping[str, object]]:
    merged: list[Mapping[str, object]] = []
    seen: set[str] = set()
    for item in list(existing) + list(new):
        if not isinstance(item, Mapping):
            continue
        serialized = json.dumps(item, sort_keys=True, ensure_ascii=False)
        if serialized in seen:
            continue
        seen.add(serialized)
        merged.append(dict(item))
    return merged


def _choose_result_file(results_dir: Path) -> Path | None:
    if not results_dir.is_dir():
        return None
    preferred = results_dir / "model.json"
    if preferred.is_file():
        return preferred
    for candidate in sorted(results_dir.glob("*.json")):
        if candidate.is_file():
            return candidate
    return None


def _load_result_payload(path: Path) -> Mapping[str, object] | None:
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

    return payload if isinstance(payload, Mapping) else None


def migrate_sid(sid: str, runs_root: Path, *, dry_run: bool = False) -> None:
    sid = str(sid)
    runs_root = runs_root.resolve()
    validation_paths = ensure_validation_paths(runs_root, sid, create=True)
    base_dir = validation_paths.base

    legacy_dirs = [
        child
        for child in sorted(base_dir.iterdir())
        if child.is_dir() and child.name not in {"packs", "results"}
    ]

    index_entries: list[ValidationIndexEntry] = []

    for legacy_dir in legacy_dirs:
        try:
            account_id = int(legacy_dir.name)
        except ValueError:
            continue

        pack_filename = validation_pack_filename_for_account(account_id)
        pack_path = validation_paths.packs_dir / pack_filename
        prompt_path = pack_path.with_name(pack_path.name + ".prompt.txt")

        legacy_pack = legacy_dir / "pack.json"
        legacy_prompt = legacy_dir / "prompt.txt"
        legacy_results_dir = legacy_dir / "results"

        existing_items = _load_existing_pack_items(pack_path)
        new_items = _load_legacy_weak_items(legacy_pack)
        merged_items = _merge_items(existing_items, new_items)

        if dry_run:
            line_count = len(merged_items)
            weak_fields = [str(item.get("field") or "") for item in merged_items]
        else:
            line_count, weak_fields = write_pack_lines(
                pack_path,
                account_id=account_id,
                sid=sid,
                weak_items=merged_items,
            )

        if legacy_prompt.is_file() and not dry_run:
            prompt_path.parent.mkdir(parents=True, exist_ok=True)
            prompt_path.write_text(legacy_prompt.read_text(encoding="utf-8"), encoding="utf-8")
            legacy_prompt.unlink(missing_ok=True)

        summary_filename = validation_result_filename_for_account(account_id)
        jsonl_filename = validation_result_jsonl_filename_for_account(account_id)
        result_path = validation_paths.results_dir / summary_filename
        jsonl_path = validation_paths.results_dir / jsonl_filename
        result_payload: Mapping[str, object] | None = None

        source_result = _choose_result_file(legacy_results_dir)
        if source_result is not None:
            result_payload = _load_result_payload(source_result)
            if not dry_run:
                result_path.parent.mkdir(parents=True, exist_ok=True)
                result_path.write_text(source_result.read_text(encoding="utf-8"), encoding="utf-8")
                jsonl_path.parent.mkdir(parents=True, exist_ok=True)
                if jsonl_path.exists():
                    jsonl_path.unlink()
                jsonl_path.write_text("", encoding="utf-8")
                source_result.unlink(missing_ok=True)
        elif result_path.exists():
            result_payload = _load_result_payload(result_path)
        elif jsonl_path.exists():
            jsonl_path.unlink(missing_ok=True)
            jsonl_path.write_text("", encoding="utf-8")
        elif not dry_run:
            jsonl_path.parent.mkdir(parents=True, exist_ok=True)
            jsonl_path.write_text("", encoding="utf-8")

        status = "built"
        built_at = None
        model = None
        request_lines = None
        completed_at = None

        if isinstance(result_payload, Mapping):
            status = str(result_payload.get("status") or status)
            model = result_payload.get("model")
            request_lines = result_payload.get("request_lines")
            built_at = result_payload.get("timestamp") or result_payload.get("built_at")
            completed_at = result_payload.get("completed_at")

        entry = ValidationIndexEntry(
            account_id=account_id,
            pack_path=pack_path,
            result_jsonl_path=jsonl_path,
            result_json_path=result_path,
            weak_fields=tuple(field for field in weak_fields if field),
            line_count=line_count,
            status=status,
            built_at=str(built_at) if built_at else None,
            request_lines=int(request_lines) if isinstance(request_lines, (int, float)) else None,
            model=str(model) if model else None,
            completed_at=str(completed_at) if completed_at else None,
        )
        index_entries.append(entry)

        if not dry_run:
            legacy_pack.unlink(missing_ok=True)
            if legacy_results_dir.is_dir():
                for leftover in legacy_results_dir.iterdir():
                    leftover.unlink(missing_ok=True)
                legacy_results_dir.rmdir()
            with suppress(OSError):
                if not any(legacy_dir.iterdir()):
                    legacy_dir.rmdir()

    if dry_run:
        for entry in index_entries:
            print(f"Would migrate account {entry.account_id:03d} -> {entry.pack_path}")
        return

    index_path = validation_paths.index_file
    index_path.unlink(missing_ok=True)
    writer = ValidationPackIndexWriter(
        sid=sid,
        index_path=index_path,
        packs_dir=validation_paths.packs_dir,
        results_dir=validation_paths.results_dir,
    )
    if index_entries:
        writer.bulk_upsert(index_entries)
    else:
        write_validation_manifest_v2(
            sid=sid,
            packs_dir=validation_paths.packs_dir,
            results_dir=validation_paths.results_dir,
            entries=[],
            index_path=index_path,
        )

    print(f"Migrated {len(index_entries)} validation accounts for SID {sid}.")
    print("Validation packs directory:", validation_paths.packs_dir)
    print("Validation results directory:", validation_paths.results_dir)
    print("Validation index path:", validation_paths.index_file)


def main(argv: Iterable[str] | None = None) -> int:
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    migrate_sid(args.sid, args.runs_root, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
