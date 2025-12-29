import os
import sys
import json
from pathlib import Path

# Ensure project root on path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.pipeline.validation_orchestrator import ValidationOrchestrator
from backend.runflow.decider import _compute_umbrella_barriers


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: python devtools/run_validation_orchestrator.py <sid> [runs_root]", file=sys.stderr)
        return 2
    sid = sys.argv[1]
    runs_root = sys.argv[2] if len(sys.argv) > 2 else os.getenv("RUNS_ROOT")

    # Force orchestrator mode on for this invocation
    os.environ.setdefault("VALIDATION_ORCHESTRATOR_MODE", "1")

    # ── DEFENSIVE WARNING: Check merge_ready barrier ──────────────────────
    runs_root_path = Path(runs_root) if runs_root else Path(os.getcwd()) / "runs"
    run_dir = runs_root_path / sid
    if run_dir.exists():
        barriers = _compute_umbrella_barriers(run_dir)
        merge_ready = barriers.get("merge_ready", False)
        if not merge_ready:
            print(
                f"⚠️  WARNING: merge_ready=False for sid={sid}. "
                f"Validation may fail or produce incomplete results. "
                f"Barriers: {barriers}",
                file=sys.stderr,
            )
    # ──────────────────────────────────────────────────────────────────────

    orch = ValidationOrchestrator(runs_root=runs_root)
    result = orch.run_for_sid(sid)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
