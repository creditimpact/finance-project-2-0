"""Command-line helpers for working with validation manifest indexes."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import replace
from itertools import count
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence, TextIO

from backend.core.ai.paths import (
    ensure_validation_paths,
    validation_index_path,
    validation_pack_filename_for_account,
    validation_result_jsonl_filename_for_account,
    validation_result_summary_filename_for_account,
)

from .index_schema import (
    ValidationIndex,
    build_validation_index,
    load_validation_index,
)


def _single_result_file_enabled() -> bool:
    raw = os.getenv("VALIDATION_SINGLE_RESULT_FILE")
    if raw is None:
        return True

    lowered = raw.strip().lower()
    if lowered in {"1", "true", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "no", "n", "off"}:
        return False
    return True


def load_index_for_sid(sid: str, *, runs_root: Path | str | None = None) -> ValidationIndex:
    """Return the :class:`ValidationIndex` for ``sid``."""

    index_path = validation_index_path(sid, runs_root=runs_root, create=False)
    return load_validation_index(index_path)


def _display_relative(path: Path, base: Path) -> str:
    """Return ``path`` relative to ``base`` using POSIX separators."""

    path_resolved = path.resolve()
    base_resolved = base.resolve()
    try:
        relative = path_resolved.relative_to(base_resolved)
        return PurePosixPath(relative).as_posix() or "."
    except ValueError:
        try:
            relpath = Path(os_path_relpath(path_resolved, base_resolved))
        except (OSError, ValueError):  # pragma: no cover - defensive
            return path_resolved.as_posix()
        return PurePosixPath(relpath).as_posix() or "."


def os_path_relpath(target: Path, base: Path) -> str:
    """``os.path.relpath`` wrapper that accepts :class:`Path` arguments."""

    from os import path as os_path

    return os_path.relpath(str(target), str(base))


def check_index(
    index: ValidationIndex,
    *,
    stream: TextIO = sys.stdout,
) -> bool:
    """Verify that every pack referenced by ``index`` exists on disk."""

    total = len(index.packs)
    rows: list[list[str]] = []
    missing = 0

    index_dir = index.index_dir

    for record in index.packs:
        pack_path = index.resolve_pack_path(record)
        pack_exists = pack_path.is_file()
        if not pack_exists:
            missing += 1

        pack_display = record.pack or _display_relative(pack_path, index_dir)
        result_jsonl_display = record.result_jsonl or _display_relative(
            index.resolve_result_jsonl_path(record), index_dir
        )
        result_json_display = record.result_json or _display_relative(
            index.resolve_result_json_path(record), index_dir
        )

        rows.append(
            [
                f"{record.account_id:03d}",
                pack_display,
                "OK" if pack_exists else "MISSING",
                str(record.lines or 0),
                result_jsonl_display,
                result_json_display,
            ]
        )

    headers = ["ACCOUNT", "PACK", "STATUS", "LINES", "RESULT_JSONL", "RESULT_JSON"]

    def _column_width(idx: int) -> int:
        column_values = [headers[idx], *[row[idx] for row in rows]]
        return max(len(value) for value in column_values) if column_values else len(headers[idx])

    widths = [_column_width(i) for i in range(len(headers))]
    align_right = {0, 3}

    def _format_row(cells: Sequence[str]) -> str:
        formatted: list[str] = []
        for idx, cell in enumerate(cells):
            width = widths[idx]
            if idx in align_right:
                formatted.append(cell.rjust(width))
            else:
                formatted.append(cell.ljust(width))
        return "  ".join(formatted)

    stream.write(f"Validation packs for SID {index.sid}:\n")
    stream.write(_format_row(headers) + "\n")
    separator_width = sum(widths) + 2 * (len(headers) - 1) if widths else 0
    if separator_width:
        stream.write("-" * separator_width + "\n")

    if rows:
        for row in rows:
            stream.write(_format_row(row) + "\n")
    else:
        stream.write("(no packs)\n")

    stream.write("\n")
    stream.write(
        f"Manifest: {_display_relative(index.index_path, index_dir)}\n"
    )
    stream.write(
        f"Packs dir: {_display_relative(index.packs_dir_path, index_dir)}\n"
    )
    stream.write(
        f"Results dir: {_display_relative(index.results_dir_path, index_dir)}\n"
    )

    if missing:
        stream.write(f"Missing packs detected: {missing} of {total}.\n")
    else:
        stream.write(f"All {total} packs present for SID {index.sid}.\n")

    return missing == 0


def _schema_version_from_document(document: Mapping[str, Any]) -> int:
    value = document.get("schema_version")
    try:
        return int(value)
    except (TypeError, ValueError):
        return 1


def _resolve_backup_path(index_path: Path) -> Path:
    base = index_path.with_name(f"{index_path.stem}.v1.json")
    if not base.exists():
        return base
    for suffix in count(1):
        candidate = index_path.with_name(f"{index_path.stem}.v1.{suffix}.json")
        if not candidate.exists():
            return candidate
    raise RuntimeError("Unable to determine backup path for manifest")


def rewrite_index_to_v2(
    index_path: Path,
    *,
    document: Mapping[str, Any],
    original_text: str,
    stream: TextIO = sys.stdout,
) -> tuple[ValidationIndex, bool]:
    """Convert a legacy schema v1 manifest to v2 and persist the result."""

    original_version = _schema_version_from_document(document)
    if original_version >= 2:
        stream.write("Validation manifest already uses schema version 2.\n")
        return load_validation_index(index_path), False

    backup_path = _resolve_backup_path(index_path)
    backup_path.write_text(original_text, encoding="utf-8")
    stream.write(f"Backed up legacy manifest to {backup_path}.\n")

    index = load_validation_index(index_path)
    index.write()
    stream.write(f"Rewrote validation manifest to schema v2 at {index_path}.\n")
    return index, True


def rewrite_index_to_canonical_layout(
    index_path: Path,
    *,
    runs_root: Path | str | None = None,
    stream: TextIO = sys.stdout,
) -> tuple[ValidationIndex, bool]:
    """Align ``index.json`` paths with the ai_packs/validation layout."""

    index = load_validation_index(index_path)
    base_dir = index_path.parent.resolve()

    sid_hint: str | None = None
    try:
        sid_hint = base_dir.parents[1].name
    except IndexError:
        sid_hint = None

    sid = index.sid or sid_hint
    if not sid:
        raise ValueError(f"Unable to determine SID for manifest at {index_path}")

    if runs_root is None:
        try:
            runs_root_path = base_dir.parents[2]
        except IndexError as exc:  # pragma: no cover - defensive
            raise ValueError(
                f"Unable to determine runs root for manifest at {index_path}"
            ) from exc
        runs_root_path = runs_root_path.resolve()
    else:
        runs_root_path = Path(runs_root).resolve()

    validation_paths = ensure_validation_paths(runs_root_path, sid, create=True)

    expected_base = validation_paths.base.resolve()
    expected_packs_dir = validation_paths.packs_dir.resolve()
    expected_results_dir = validation_paths.results_dir.resolve()

    changed = False

    if index.root_dir.resolve() != expected_base:
        changed = True
    if index.packs_dir_path.resolve() != expected_packs_dir:
        changed = True
    if index.results_dir_path.resolve() != expected_results_dir:
        changed = True

    canonical_records = []
    single_result = _single_result_file_enabled()
    for record in index.packs:
        pack_filename = validation_pack_filename_for_account(record.account_id)
        json_filename = validation_result_summary_filename_for_account(record.account_id)
        jsonl_filename = (
            validation_result_jsonl_filename_for_account(record.account_id)
            if not single_result
            else None
        )

        expected_pack_path = validation_paths.packs_dir / pack_filename
        expected_json_path = validation_paths.results_dir / json_filename

        if index.resolve_pack_path(record).resolve() != expected_pack_path.resolve():
            changed = True

        if jsonl_filename:
            expected_jsonl_path = validation_paths.results_dir / jsonl_filename
            try:
                actual_jsonl_path = index.resolve_result_jsonl_path(record).resolve()
            except ValueError:
                actual_jsonl_path = None
            if actual_jsonl_path != expected_jsonl_path.resolve():
                changed = True
        elif record.result_jsonl:
            changed = True

        if index.resolve_result_json_path(record).resolve() != expected_json_path.resolve():
            changed = True

        pack_rel = (PurePosixPath("packs") / pack_filename).as_posix()
        json_rel = (PurePosixPath("results") / json_filename).as_posix()
        if jsonl_filename:
            jsonl_rel: str | None = (PurePosixPath("results") / jsonl_filename).as_posix()
        else:
            jsonl_rel = None

        canonical_records.append(
            replace(
                record,
                pack=pack_rel,
                result_jsonl=jsonl_rel,
                result_json=json_rel,
            )
        )

    if not changed:
        return index, False

    canonical_index = build_validation_index(
        index_path=index_path,
        sid=sid,
        packs_dir=validation_paths.packs_dir,
        results_dir=validation_paths.results_dir,
        records=tuple(canonical_records),
    )
    canonical_index = replace(
        canonical_index,
        schema_version=index.schema_version or canonical_index.schema_version,
    )
    
    # Write the index and mark as canonical
    document = canonical_index.to_json_payload()
    document["canonical_layout"] = True
    serialized = json.dumps(document, ensure_ascii=False, indent=2)
    canonical_index.index_path.parent.mkdir(parents=True, exist_ok=True)
    canonical_index.index_path.write_text(serialized + "\n", encoding="utf-8")
    
    stream.write(
        f"Rewrote validation manifest to canonical layout at {index_path}.\n"
    )
    return canonical_index, True


def _parse_argv(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validation manifest utilities")
    parser.add_argument("--sid", required=True, help="Run SID to inspect")
    parser.add_argument(
        "--runs-root",
        help="Base runs/ directory (defaults to ./runs or RUNS_ROOT env)",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Verify that every pack referenced by the index exists",
    )
    parser.add_argument(
        "--rewrite-v2",
        action="store_true",
        help="Rewrite legacy schema v1 manifests to v2 (creates index.v1.json backup)",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_argv(argv)
    runs_root = Path(args.runs_root) if args.runs_root else None

    index_path = validation_index_path(args.sid, runs_root=runs_root, create=False)

    try:
        original_text = index_path.read_text(encoding="utf-8")
        document = json.loads(original_text)
        if not isinstance(document, Mapping):
            raise TypeError("Validation index root must be an object")
    except FileNotFoundError:
        print(
            f"Validation index not found for SID {args.sid!r}.",
            file=sys.stderr,
        )
        return 2
    except (OSError, json.JSONDecodeError, TypeError) as exc:
        print(
            f"Unable to load validation index for SID {args.sid!r}: {exc}",
            file=sys.stderr,
        )
        return 2

    index: ValidationIndex
    if args.rewrite_v2:
        index, _ = rewrite_index_to_v2(
            index_path,
            document=document,
            original_text=original_text,
            stream=sys.stdout,
        )
    else:
        index = load_validation_index(index_path)

    if args.check:
        ok = check_index(index)
        return 0 if ok else 1

    print(index.index_path)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    raise SystemExit(main())

