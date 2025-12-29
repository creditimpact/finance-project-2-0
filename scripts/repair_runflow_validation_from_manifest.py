import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

from backend.runflow.decider import refresh_validation_stage_from_index, reconcile_umbrella_barriers
from backend.validation.apply_results_v2 import apply_validation_results_for_sid

log = logging.getLogger("repair_v2_runflow")


def _load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _needs_repair(manifest: dict[str, Any], runflow: dict[str, Any]) -> bool:
    ai_status = (manifest.get("ai") or {}).get("status") or {}
    vstat = ai_status.get("validation") or {}
    applied = bool(vstat.get("applied") or vstat.get("results_apply_done") or vstat.get("validation_ai_applied"))

    if not applied:
        return False

    stages = runflow.get("stages") if isinstance(runflow, dict) else {}
    vstage = stages.get("validation") if isinstance(stages, dict) else {}

    status_text = (vstage.get("status") or "").strip().lower()
    if status_text != "success":
        return True

    results_total = 0
    results = vstage.get("results") if isinstance(vstage, dict) else {}
    if isinstance(results, dict):
        rt = results.get("results_total")
        try:
            results_total = int(rt)
        except Exception:
            results_total = 0

    manifest_total = 0
    try:
        mt = vstat.get("results_total")
        if isinstance(mt, (int, float)) and not isinstance(mt, bool):
            manifest_total = int(mt)
    except Exception:
        manifest_total = 0

    if results_total <= 0 and manifest_total > 0:
        return True

    metrics = vstage.get("metrics") if isinstance(vstage, dict) else {}
    completed_flag = bool(isinstance(metrics, dict) and metrics.get("validation_ai_completed"))
    if not completed_flag:
        return True

    return False


def _extract_runflow_validation_snapshot(runflow: dict[str, Any]) -> dict[str, Any]:
    stages = runflow.get("stages") if isinstance(runflow, dict) else {}
    vstage = stages.get("validation") if isinstance(stages, dict) else {}
    metrics = vstage.get("metrics") if isinstance(vstage, dict) else {}
    results = vstage.get("results") if isinstance(vstage, dict) else {}
    return {
        "status": (vstage.get("status") or ""),
        "results_total": (results.get("results_total") if isinstance(results, dict) else None),
        "completed": (results.get("completed") if isinstance(results, dict) else None),
        "failed": (results.get("failed") if isinstance(results, dict) else None),
        "validation_ai_completed": bool(metrics.get("validation_ai_completed")) if isinstance(metrics, dict) else False,
        "validation_ai_applied": bool(metrics.get("validation_ai_applied")) if isinstance(metrics, dict) else False,
    }


def repair_runs(runs_root: Path, only_sid: str | None = None) -> int:
    root = Path(runs_root)
    if not root.exists():
        print(f"Runs root not found: {root}", file=sys.stderr)
        return 2

    repaired = 0
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        sid = child.name
        if only_sid and sid != only_sid:
            continue

        manifest_path = child / "manifest.json"
        runflow_path = child / "runflow.json"
        if not manifest_path.exists():
            continue

        manifest = _load_json(manifest_path)
        runflow = _load_json(runflow_path) if runflow_path.exists() else {}

        if not _needs_repair(manifest, runflow):
            continue

        before = _extract_runflow_validation_snapshot(runflow)

        try:
            # 1) Normalize/apply results (idempotent) to ensure index packs marked completed
            apply_validation_results_for_sid(sid, runs_root=root)
            # 2) Refresh runflow validation stage from index + manifest
            refresh_validation_stage_from_index(sid, runs_root=root)
            # 3) Reconcile umbrella barriers for readiness flags
            reconcile_umbrella_barriers(sid, runs_root=root)
            repaired += 1
            # After snapshot
            after_path = child / "runflow.json"
            after = _extract_runflow_validation_snapshot(_load_json(after_path) if after_path.exists() else {})
            log.info(
                "RUNFLOW_REPAIRED sid=%s before=%s after=%s",
                sid,
                json.dumps(before, ensure_ascii=False),
                json.dumps(after, ensure_ascii=False),
            )
        except Exception as exc:  # pragma: no cover - field repair errors are logged
            log.error("RUNFLOW_REPAIR_FAILED sid=%s error=%s", sid, exc, exc_info=True)

    return repaired


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Repair runflow validation stage from manifest+index (V2 authoritative)")
    parser.add_argument("--runs-root", default="runs", help="Path to runs root (default: runs)")
    parser.add_argument("--sid", default=None, help="Optional single SID to repair")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    count = repair_runs(Path(args.runs_root), args.sid)
    print(f"Repaired {count} run(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
