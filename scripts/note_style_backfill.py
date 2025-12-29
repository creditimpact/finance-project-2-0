"""Backfill utility for correcting note_style runflow statuses."""

from __future__ import annotations

import argparse
import json
import logging
import os
from collections.abc import Iterable
from pathlib import Path
from typing import Mapping

from backend.ai.note_style.io import note_style_snapshot
from backend.core.io.json_io import _atomic_write_json
from backend.runflow.decider import _now_iso


log = logging.getLogger(__name__)


def _normalize_status(value: object) -> str:
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized:
            return normalized
    return ""


def _determine_next_status(*, terminal_total: int, expected_total: int) -> str:
    if terminal_total > 0 and terminal_total < expected_total:
        return "in_progress"
    return "built"


def _update_stage_payload(
    stage_payload: Mapping[str, object] | None,
    *,
    status: str,
    timestamp: str,
    expected_total: int,
    completed_total: int,
    failed_total: int,
) -> dict[str, object]:
    metrics_payload = {"packs_total": expected_total}
    results_payload = {
        "results_total": expected_total,
        "completed": completed_total,
        "failed": failed_total,
    }
    summary_payload = {
        "packs_total": expected_total,
        "results_total": expected_total,
        "completed": completed_total,
        "failed": failed_total,
        "empty_ok": False,
        "metrics": dict(metrics_payload),
        "results": dict(results_payload),
    }

    updated = dict(stage_payload or {})
    updated.update(
        {
            "status": status,
            "last_at": timestamp,
            "empty_ok": False,
            "metrics": metrics_payload,
            "results": results_payload,
            "summary": summary_payload,
            "sent": False,
            "completed_at": None,
        }
    )
    return updated


def backfill_note_style_runflow(
    runs_root: Path | str,
    *,
    dry_run: bool = False,
) -> list[str]:
    """Downgrade incorrect terminal note_style statuses under ``runs_root``.

    Returns a list of run identifiers that were modified (or would be modified
    when ``dry_run`` is true).
    """

    runs_root_path = Path(runs_root)
    fixed: list[str] = []

    for run_dir in sorted(
        entry for entry in runs_root_path.iterdir() if entry.is_dir()
    ):
        sid = run_dir.name
        runflow_path = run_dir / "runflow.json"
        if not runflow_path.is_file():
            continue
        try:
            raw = runflow_path.read_text(encoding="utf-8")
        except OSError:
            log.warning("backfill_runflow_read_failed sid=%s", sid, exc_info=True)
            continue
        try:
            runflow_payload = json.loads(raw)
        except json.JSONDecodeError:
            log.warning("backfill_runflow_parse_failed sid=%s", sid, exc_info=True)
            continue

        stages_obj = runflow_payload.get("stages")
        if not isinstance(stages_obj, Mapping):
            continue
        stage_payload = stages_obj.get("note_style")
        if stage_payload is None:
            continue

        status_value = _normalize_status(
            stage_payload.get("status") if isinstance(stage_payload, Mapping) else None
        )
        if status_value not in {"success", "completed"}:
            continue

        snapshot = note_style_snapshot(sid, runs_root=runs_root_path)
        expected_accounts = snapshot.packs_expected
        if not expected_accounts:
            continue

        expected_total = len(expected_accounts)
        completed_total = len(snapshot.packs_completed & expected_accounts)
        failed_total = len(snapshot.packs_failed & expected_accounts)
        terminal_total = completed_total + failed_total

        if terminal_total >= expected_total:
            continue

        timestamp = _now_iso()
        next_status = _determine_next_status(
            terminal_total=terminal_total, expected_total=expected_total
        )

        updated_stage = _update_stage_payload(
            stage_payload if isinstance(stage_payload, Mapping) else {},
            status=next_status,
            timestamp=timestamp,
            expected_total=expected_total,
            completed_total=completed_total,
            failed_total=failed_total,
        )

        stages_obj = dict(stages_obj)
        stages_obj["note_style"] = updated_stage
        runflow_payload = dict(runflow_payload)
        runflow_payload["stages"] = stages_obj
        runflow_payload["last_writer"] = "note_style_backfill"
        runflow_payload["updated_at"] = timestamp

        fixed.append(sid)
        log.info(
            "backfill_fixed_runflow sid=%s status=%s->%s expected=%s terminal=%s",  # noqa: G004
            sid,
            status_value,
            next_status,
            expected_total,
            terminal_total,
        )

        if dry_run:
            continue

        _atomic_write_json(runflow_path, runflow_payload)

    return fixed


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=None,
        help="Base runs directory (defaults to $RUNS_ROOT or ./runs)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only report affected runs without writing changes.",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Enable verbose logging output."
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = _parse_args(argv)
    runs_root = args.runs_root
    if runs_root is None:
        env_value = Path(os.getenv("RUNS_ROOT")) if os.getenv("RUNS_ROOT") else None
        runs_root = env_value or Path("runs")

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    fixed = backfill_note_style_runflow(runs_root, dry_run=args.dry_run)
    log.info("backfill_complete runs=%s", len(fixed))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    raise SystemExit(main())
