"""Command line helpers for pre-validation utilities."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from .date_convention_detector import detect_month_language_for_run


def _resolve_runs_root(value: str | None) -> Path:
    if value:
        return Path(value)

    env_value = os.getenv("RUNS_ROOT")
    if env_value:
        return Path(env_value)

    return Path("runs")


def _print_json(payload: dict[str, Any]) -> None:
    json.dump(payload, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")


def _cmd_detect(args: argparse.Namespace) -> int:
    sid = args.sid
    if not sid:
        raise ValueError("sid is required")

    runs_root = _resolve_runs_root(args.runs_root)
    run_dir = runs_root / sid
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    payload = detect_month_language_for_run(str(run_dir))
    if not isinstance(payload, dict):
        raise RuntimeError("Detector returned unexpected payload")

    _print_json(payload)
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m backend.prevalidation.cli",
        description="Helpers for running pre-validation utilities.",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    detect_parser = subparsers.add_parser(
        "detect", help="Run the date convention detector for a given SID."
    )
    detect_parser.add_argument("--sid", required=True, help="Run SID to scan")
    detect_parser.add_argument(
        "--runs-root",
        help="Root directory containing run folders (default: RUNS_ROOT env or ./runs)",
    )
    detect_parser.set_defaults(func=_cmd_detect)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    handler = getattr(args, "func", None)
    if handler is None:
        parser.print_help()
        return 1

    try:
        return handler(args)
    except FileNotFoundError as exc:
        parser.error(str(exc))
    except OSError as exc:
        parser.error(str(exc))
    except RuntimeError as exc:
        parser.error(str(exc))
    except ValueError as exc:
        parser.error(str(exc))

    return 1


if __name__ == "__main__":
    sys.exit(main())
