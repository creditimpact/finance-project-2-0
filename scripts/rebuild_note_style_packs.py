"""Re-build note_style AI packs from existing frontend response results."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Iterable, Sequence

try:  # pragma: no cover - runtime bootstrap
    import scripts._bootstrap  # type: ignore  # noqa: F401
except Exception:  # pragma: no cover - fallback for direct execution
    import sys

    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from backend.ai.manifest import ensure_note_style_section
from backend.ai.note_style_stage import (
    build_note_style_pack_for_account,
    discover_note_style_response_accounts,
)
from backend.runflow.decider import record_stage


log = logging.getLogger(__name__)


def _iter_sids(runs_root: Path, explicit: Sequence[str] | None) -> Iterable[str]:
    if explicit:
        for sid in sorted({str(candidate).strip() for candidate in explicit if candidate}):
            if sid:
                yield sid
        return

    if not runs_root.is_dir():
        return

    for entry in sorted(runs_root.iterdir()):
        if not entry.is_dir():
            continue
        yield entry.name


def _mark_empty_success(sid: str, runs_root: Path) -> None:
    try:
        record_stage(
            sid,
            "note_style",
            status="success",
            counts={"packs_total": 0},
            empty_ok=True,
            metrics={"packs_total": 0},
            results={"results_total": 0, "completed": 0, "failed": 0},
            runs_root=runs_root,
        )
    except Exception:  # pragma: no cover - defensive logging
        log.exception("NOTE_STYLE_BACKFILL_STAGE_RECORD_FAILED sid=%s", sid)


def _process_sid(sid: str, runs_root: Path) -> None:
    ensure_note_style_section(sid, runs_root=runs_root)
    accounts = discover_note_style_response_accounts(sid, runs_root=runs_root)
    if not accounts:
        log.info("NOTE_STYLE_BACKFILL_NO_RESPONSES sid=%s", sid)
        _mark_empty_success(sid, runs_root)
        return

    built = 0
    skipped = 0

    for account in accounts:
        result = build_note_style_pack_for_account(
            sid, account.account_id, runs_root=runs_root
        )
        status = str(result.get("status") or "").lower()
        if status == "completed":
            built += 1
        elif status.startswith("skipped"):
            skipped += 1

    log.info(
        "NOTE_STYLE_BACKFILL_DONE sid=%s discovered=%s built=%s skipped=%s",
        sid,
        len(accounts),
        built,
        skipped,
    )


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs-root",
        default="runs",
        help="Root directory containing runs/<SID> outputs",
    )
    parser.add_argument(
        "--sid",
        action="append",
        dest="sids",
        help="Specific SID to rebuild (may be provided multiple times)",
    )

    args = parser.parse_args(argv)

    runs_root = Path(args.runs_root).resolve()

    logging.basicConfig(level=logging.INFO)

    processed_any = False
    for sid in _iter_sids(runs_root, args.sids):
        processed_any = True
        _process_sid(sid, runs_root)

    if not processed_any:
        log.info("NOTE_STYLE_BACKFILL_NO_RUNS root=%s", runs_root)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
