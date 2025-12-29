"""Display consolidated validation AI pack index information."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.ai.validation_index import ValidationPackIndexWriter
from backend.core.ai.paths import ensure_validation_paths
from backend.pipeline.runs import RUNS_ROOT_ENV

__all__ = ["main"]


def _resolve_runs_root(explicit: str | None) -> Path:
    if explicit:
        return Path(explicit)
    env_value = os.getenv(RUNS_ROOT_ENV)
    if env_value:
        return Path(env_value)
    return Path("runs")


def _coerce_int(value: object) -> int:
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0


def _weak_field_count(entry: dict[str, object]) -> int:
    weak_fields = entry.get("weak_fields")
    if isinstance(weak_fields, (list, tuple, set)):
        return sum(1 for field in weak_fields if str(field).strip())

    lines = _coerce_int(entry.get("lines"))
    if lines:
        return lines

    return _coerce_int(entry.get("request_lines"))


def _line_count(entry: dict[str, object]) -> int:
    lines = _coerce_int(entry.get("lines"))
    if lines:
        return lines
    return _coerce_int(entry.get("request_lines"))


def _format_table(rows: list[list[str]]) -> str:
    if not rows:
        return ""

    headers = ["Account", "Weak", "Lines", "Status", "Model"]
    align_right = {0, 1, 2}

    widths = [len(header) for header in headers]
    for row in rows:
        for idx, cell in enumerate(row):
            widths[idx] = max(widths[idx], len(cell))

    header_line = "  ".join(
        header.rjust(widths[idx]) if idx in align_right else header.ljust(widths[idx])
        for idx, header in enumerate(headers)
    )

    separator = "  ".join("-" * width for width in widths)

    body = [
        "  ".join(
            cell.rjust(widths[idx]) if idx in align_right else cell.ljust(widths[idx])
            for idx, cell in enumerate(row)
        )
        for row in rows
    ]

    return "\n".join([header_line, separator, *body])


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Show consolidated validation AI pack index details for a SID",
    )
    parser.add_argument("sid", help="Run identifier (SID)")
    parser.add_argument(
        "--runs-root",
        dest="runs_root",
        default=None,
        help="Override runs root directory (defaults to $RUNS_ROOT or ./runs)",
    )

    args = parser.parse_args(list(argv) if argv is not None else None)

    runs_root = _resolve_runs_root(args.runs_root)
    validation_paths = ensure_validation_paths(runs_root, args.sid, create=False)
    index_path = validation_paths.index_file
    writer = ValidationPackIndexWriter(
        sid=args.sid,
        index_path=index_path,
        packs_dir=validation_paths.packs_dir,
        results_dir=validation_paths.results_dir,
    )
    accounts = writer.load_accounts()

    print(f"SID: {args.sid}")
    print(f"Index: {index_path}")

    if not accounts:
        print("No validation packs recorded in the index.")
        return 0

    rows: list[list[str]] = []
    total_weak = 0
    status_counts: dict[str, int] = {}

    for account_id in sorted(accounts):
        entry = accounts[account_id]
        weak_count = _weak_field_count(entry)
        total_weak += weak_count

        line_count = _line_count(entry)

        status = str(entry.get("status") or "unknown")
        status_counts[status] = status_counts.get(status, 0) + 1

        model = str(entry.get("model") or "-")

        rows.append(
            [
                f"{account_id:03d}",
                str(weak_count),
                str(line_count),
                status,
                model,
            ]
        )

    table = _format_table(rows)
    if table:
        print(table)

    print()
    print(f"Accounts: {len(rows)}  Weak fields: {total_weak}")
    if status_counts:
        print("Status counts:")
        for status in sorted(status_counts):
            print(f"  {status}: {status_counts[status]}")

    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
