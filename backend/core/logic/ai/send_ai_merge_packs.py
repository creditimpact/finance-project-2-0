"""Programmatic helpers for sending merge AI packs."""

from __future__ import annotations

import os
from pathlib import Path

from scripts.send_ai_merge_packs import main as _send_main

_DEFAULT_RUNS_ROOT = Path(os.environ.get("RUNS_ROOT", "runs"))


def run_send_for_sid(
    sid: str, *, runs_root: Path | str | None = None, packs_dir: Path | str | None = None
) -> None:
    """Send adjudication packs for ``sid`` using the CLI implementation."""

    argv: list[str] = ["--sid", str(sid)]
    if packs_dir is not None:
        argv.extend(["--packs-dir", str(packs_dir)])

    root = Path(runs_root) if runs_root is not None else _DEFAULT_RUNS_ROOT
    argv.extend(["--runs-root", str(root)])

    _send_main(argv)


__all__ = ["run_send_for_sid"]
