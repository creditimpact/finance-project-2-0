"""CLI wrapper around :func:`backend.validation.send_packs.send_validation_packs`."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from backend.core.ai.paths import validation_index_path

from .send_packs import ValidationPackError, send_validation_packs


def _parse_argv(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Send validation packs referenced by the manifest index"
    )
    parser.add_argument("--sid", required=True, help="Run SID to send")
    parser.add_argument(
        "--runs-root",
        help="Base runs/ directory (defaults to ./runs or RUNS_ROOT env)",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_argv(argv)
    runs_root = Path(args.runs_root) if args.runs_root else None

    index_path = validation_index_path(args.sid, runs_root=runs_root, create=False)
    if not index_path.is_file():
        print(
            f"Validation index not found for SID {args.sid!r} at {index_path}",
            file=sys.stderr,
        )
        return 2

    try:
        send_validation_packs(index_path)
    except ValidationPackError as exc:
        print(f"Validation send failed: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    raise SystemExit(main())
