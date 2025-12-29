"""
Debug helper to send Validation AI packs for a single SID, writing results
and updating manifest status safely without clobbering validation natives.

Usage:
  python -m backend.debug.send_validation_for_sid --sid <SID> [--runs-root <path>]

After success, runflow validation should reflect:
- packs_count = N (3 for the target case)
- results_received = N
- missing_results = 0
- no validation_results_missing error
"""
from __future__ import annotations

import argparse
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from backend.pipeline.runs import (
    RunManifest,
    load_manifest_from_disk,
    save_manifest_to_disk,
)
from backend.validation.send_packs import send_validation_packs
from backend.validation.index_schema import load_validation_index
from backend.runflow.decider import refresh_validation_stage_from_index

log = logging.getLogger(__name__)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _get_validation_paths(manifest: RunManifest) -> dict[str, str]:
    data: Mapping[str, Any] = manifest.data if isinstance(manifest.data, Mapping) else {}
    ai = data.get("ai") or {}
    packs = ai.get("packs") or {}
    validation = packs.get("validation") or {}

    # Use exactly what is in the manifest; do not rewrite/resolve
    packs_dir = validation.get("packs_dir") or validation.get("packs")
    results_dir = validation.get("results_dir") or validation.get("results")
    index_path = validation.get("index")

    return {
        "packs_dir": str(packs_dir) if packs_dir else "",
        "results_dir": str(results_dir) if results_dir else "",
        "index": str(index_path) if index_path else "",
    }


def debug_send_validation_for_sid(sid: str, runs_root: Path) -> dict[str, Any]:
    """
    Send existing validation packs for a single SID and update manifest status safely.

    - Loads manifest from disk (disk-first API)
    - Discovers packs in manifest-declared packs_dir (val_acc_*.jsonl)
    - Calls the existing synchronous sender (send_validation_packs)
    - Verifies results exist in results_dir via the index
    - Updates only ai.status.validation fields using save_manifest_to_disk
    - Triggers runflow reconciliation of the validation stage
    """
    runs_root = runs_root if isinstance(runs_root, Path) else Path(runs_root)
    manifest = load_manifest_from_disk(runs_root, sid)

    paths = _get_validation_paths(manifest)
    packs_dir = paths.get("packs_dir") or ""
    results_dir = paths.get("results_dir") or ""
    index_path = paths.get("index") or ""

    if not packs_dir or not results_dir or not index_path:
        log.error(
            "DEBUG_VALIDATION_PATHS_MISSING sid=%s packs_dir=%s results_dir=%s index=%s",
            sid,
            packs_dir,
            results_dir,
            index_path,
        )
        raise RuntimeError("Manifest is missing validation stage paths")

    packs_dir_path = Path(packs_dir)
    pack_files = [p for p in packs_dir_path.glob("val_acc_*.jsonl") if p.is_file()]
    if not pack_files:
        log.error(
            "DEBUG_VALIDATION_NO_PACKS sid=%s packs_dir=%s", sid, str(packs_dir_path)
        )
        return {
            "sid": sid,
            "packs_found": 0,
            "sent": False,
            "results_written": 0,
            "error": "no_packs_found",
        }

    # Use existing infra — pass the manifest.json path so it reads stage paths.
    manifest_path = manifest.path
    log.info(
        "DEBUG_VALIDATION_SENDING sid=%s manifest=%s packs_dir=%s results_dir=%s",
        sid,
        str(manifest_path),
        packs_dir,
        results_dir,
    )
    # Respect the sender's env conventions (e.g., VALIDATION_MODEL, timeouts). Users can set envs externally.
    results = send_validation_packs(str(manifest_path), stage="validation")

    # Verify results by loading the index and checking files exist per record
    index = load_validation_index(Path(index_path))
    expected = len(index.packs)
    received = 0
    for record in index.packs:
        try:
            result_path = index.resolve_result_jsonl_path(record)
        except Exception:
            # Fall back: if resolution fails, consider missing
            continue
        try:
            if result_path.is_file():
                received += 1
        except OSError:
            continue

    completed = expected > 0 and received == expected

    # Update only ai.status.validation fields safely using disk-first save
    def _mutate_status(fm: RunManifest) -> None:
        status = fm.ensure_ai_stage_status("validation")
        status["built"] = True
        status["sent"] = True
        # completed_at only when all expected results are present
        status["completed_at"] = _utc_now() if completed else None
        status["failed"] = bool(expected > received)
        # Optional state hint for operators
        status["state"] = "completed" if completed else ("error" if status["failed"] else "in_progress")

    save_manifest_to_disk(runs_root, sid, _mutate_status, caller="backend.debug.send_validation_for_sid")

    # Trigger runflow reconciliation for this SID to recalc validation stage
    try:
        refresh_validation_stage_from_index(sid, runs_root=runs_root)
    except Exception as exc:  # best-effort
        log.warning(
            "DEBUG_VALIDATION_RECONCILE_FAILED sid=%s error=%s", sid, str(exc), exc_info=True
        )

    summary = {
        "sid": sid,
        "packs_found": len(pack_files),
        "expected_results": expected,
        "results_written": received,
        "completed": completed,
        "send_results_count": len(results),
    }
    log.info(
        "DEBUG_VALIDATION_SUMMARY sid=%s packs_found=%s expected=%s received=%s completed=%s",
        sid,
        summary["packs_found"],
        summary["expected_results"],
        summary["results_written"],
        summary["completed"],
    )
    return summary


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Send validation AI packs for a single SID")
    parser.add_argument("--sid", required=True, help="Run SID to process")
    parser.add_argument(
        "--runs-root",
        default=str(Path.cwd() / "runs"),
        help="Runs root directory (defaults to ./runs)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    sid = args.sid
    runs_root = Path(args.runs_root)

    logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))

    try:
        summary = debug_send_validation_for_sid(sid, runs_root)
    except Exception as exc:
        log.error("DEBUG_VALIDATION_SEND_FAILED sid=%s error=%s", sid, str(exc), exc_info=True)
        return 2

    print(
        f"SID={summary['sid']} packs_found={summary['packs_found']} expected={summary.get('expected_results', 0)} received={summary.get('results_written', 0)} completed={summary.get('completed', False)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
