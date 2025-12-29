"""CLI entry point for running the pre-validation date detector."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from backend.prevalidation.tasks import detect_and_persist_date_convention


def detect_dates_for_sid(
    sid: str,
    *,
    runs_root: Path | str | None = None,
) -> dict[str, Any] | None:
    """Run the date convention detector for ``sid`` and return the block."""

    normalized_sid = str(sid or "").strip()
    if not normalized_sid:
        raise ValueError("sid is required")

    root_arg: Path | str | None
    if runs_root is None:
        root_arg = None
    else:
        root_arg = Path(runs_root)

    block = detect_and_persist_date_convention(normalized_sid, runs_root=root_arg)
    return block if isinstance(block, dict) else None


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m backend.validation.preflight.detect_dates",
        description="Run the pre-validation date convention detector for a run.",
    )
    parser.add_argument("--sid", required=True, help="Run SID to scan")
    parser.add_argument(
        "--runs-root",
        help="Root directory containing run folders (default: RUNS_ROOT env or ./runs)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        block = detect_dates_for_sid(args.sid, runs_root=args.runs_root)
    except FileNotFoundError as exc:
        parser.error(str(exc))
    except OSError as exc:
        parser.error(str(exc))
    except ValueError as exc:
        parser.error(str(exc))

    payload = {"sid": args.sid, "date_convention": block}
    json.dump(payload, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry
    sys.exit(main())
