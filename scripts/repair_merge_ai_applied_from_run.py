"""
Backfill merge_ai_applied flag for completed merge stages.

This script repairs existing runs where merge completed successfully but the
merge_ai_applied flag wasn't set (because it didn't exist yet). This prevents
validation from being blocked by the new merge_ready barrier check.

Usage:
    # All runs
    python scripts/repair_merge_ai_applied_from_run.py --runs-root runs

    # Single SID
    python scripts/repair_merge_ai_applied_from_run.py --runs-root runs --sid 83830ae4-6406-4a7e-ad80-a3f721a3787b

    # Dry run (no changes)
    python scripts/repair_merge_ai_applied_from_run.py --runs-root runs --dry-run
"""
import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from backend.runflow.decider import reconcile_umbrella_barriers

log = logging.getLogger("repair_merge_ai_applied")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
)


def _load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _needs_repair(runflow: dict[str, Any]) -> bool:
    """Return True if merge stage is successful but merge_ai_applied is not set."""
    stages = runflow.get("stages") if isinstance(runflow, dict) else {}
    merge_stage = stages.get("merge") if isinstance(stages, dict) else {}

    # Check if merge succeeded
    status_text = (merge_stage.get("status") or "").strip().lower()
    if status_text != "success":
        return False

    # Check if already has flag
    merge_ai_applied = merge_stage.get("merge_ai_applied", False)
    if merge_ai_applied:
        return False

    # Check if it's NOT a zero-packs case (zero-packs don't need the flag)
    empty_ok = merge_stage.get("empty_ok", False)
    if empty_ok:
        # Zero-packs case - doesn't need repair (fast path unaffected)
        return False

    # Check if there are actual result files (non-zero-packs)
    results = merge_stage.get("results") if isinstance(merge_stage, dict) else {}
    result_files = 0
    if isinstance(results, dict):
        rf = results.get("result_files")
        try:
            result_files = int(rf) if rf is not None else 0
        except Exception:
            result_files = 0

    metrics = merge_stage.get("metrics") if isinstance(merge_stage, dict) else {}
    if result_files == 0 and isinstance(metrics, dict):
        rf = metrics.get("result_files")
        try:
            result_files = int(rf) if rf is not None else 0
        except Exception:
            result_files = 0

    # If there are result files, this is a non-zero-packs case that needs repair
    return result_files > 0


def _extract_merge_snapshot(runflow: dict[str, Any]) -> dict[str, Any]:
    """Extract merge stage key fields for before/after logging."""
    stages = runflow.get("stages") if isinstance(runflow, dict) else {}
    merge_stage = stages.get("merge") if isinstance(stages, dict) else {}
    results = merge_stage.get("results") if isinstance(merge_stage, dict) else {}
    metrics = merge_stage.get("metrics") if isinstance(merge_stage, dict) else {}

    return {
        "status": (merge_stage.get("status") or ""),
        "empty_ok": bool(merge_stage.get("empty_ok")),
        "result_files": (results.get("result_files") if isinstance(results, dict) else None),
        "created_packs": (metrics.get("created_packs") if isinstance(metrics, dict) else None),
        "merge_ai_applied": bool(merge_stage.get("merge_ai_applied")),
        "merge_ai_applied_at": (merge_stage.get("merge_ai_applied_at") or None),
    }


def repair_runs(runs_root: Path, only_sid: str | None = None, dry_run: bool = False) -> int:
    """Backfill merge_ai_applied for all completed non-zero-packs merge stages."""
    root = Path(runs_root)
    if not root.exists():
        print(f"Runs root not found: {root}", file=sys.stderr)
        return 2

    repaired = 0
    skipped = 0
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        sid = child.name
        if only_sid and sid != only_sid:
            continue

        runflow_path = child / "runflow.json"
        if not runflow_path.exists():
            continue

        runflow = _load_json(runflow_path)

        if not _needs_repair(runflow):
            skipped += 1
            continue

        before = _extract_merge_snapshot(runflow)

        if dry_run:
            log.info(
                "DRY_RUN sid=%s would_repair=True before=%s",
                sid,
                json.dumps(before, ensure_ascii=False),
            )
            repaired += 1
            continue

        try:
            # Apply the repair: set merge_ai_applied=True
            stages = runflow.get("stages")
            if not isinstance(stages, dict):
                stages = {}
                runflow["stages"] = stages

            merge_stage = stages.get("merge")
            if not isinstance(merge_stage, dict):
                merge_stage = {}
                stages["merge"] = merge_stage

            merge_stage["merge_ai_applied"] = True
            merge_stage["merge_ai_applied_at"] = datetime.utcnow().isoformat() + "Z"

            # Save runflow
            _save_json(runflow_path, runflow)

            # Reconcile umbrella barriers to update merge_ready flag
            reconcile_umbrella_barriers(sid, runs_root=root)

            repaired += 1

            # After snapshot
            after_runflow = _load_json(runflow_path)
            after = _extract_merge_snapshot(after_runflow)

            log.info(
                "MERGE_AI_APPLIED_REPAIRED sid=%s before=%s after=%s",
                sid,
                json.dumps(before, ensure_ascii=False),
                json.dumps(after, ensure_ascii=False),
            )
        except Exception:
            log.exception("REPAIR_FAILED sid=%s", sid)
            continue

    log.info(
        "REPAIR_COMPLETE repaired=%d skipped=%d dry_run=%s",
        repaired,
        skipped,
        dry_run,
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Backfill merge_ai_applied flag for completed merge stages"
    )
    parser.add_argument(
        "--runs-root",
        type=str,
        required=True,
        help="Path to runs root directory",
    )
    parser.add_argument(
        "--sid",
        type=str,
        help="Optional: repair only this SID",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Log what would be repaired without making changes",
    )
    args = parser.parse_args()

    runs_root = Path(args.runs_root)
    return repair_runs(runs_root, only_sid=args.sid, dry_run=args.dry_run)


if __name__ == "__main__":
    sys.exit(main())
