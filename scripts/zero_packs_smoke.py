from pathlib import Path
import os, json, shutil
from importlib import reload

# Import after env setup so modules read flags on import
def main():
    sid = "SMOKE_ZERO_PACKS"
    runs_root = Path("tmp_zero_smoke")
    shutil.rmtree(runs_root, ignore_errors=True)
    runs_root.mkdir(parents=True, exist_ok=True)

    # Make sure the app sees the runs root + fastpath flags
    os.environ.setdefault("RUNS_ROOT", str(runs_root))
    os.environ.setdefault("MERGE_SKIP_COUNTS_ENABLED", "1")
    os.environ.setdefault("MERGE_ZERO_PACKS_SIGNAL", "1")
    os.environ.setdefault("RUNFLOW_MERGE_ZERO_PACKS_FASTPATH", "1")
    os.environ.setdefault("RUNFLOW_EMIT_ZERO_PACKS_STEP", "1")
    os.environ.setdefault("UMBRELLA_INCLUDE_MERGE_ZERO_PACKS_FLAG", "1")
    os.environ.setdefault("VALIDATION_ZERO_PACKS_FASTPATH", "1")

    # Import project modules
    from backend.core import runflow as runflow_module
    from backend.runflow import decider
    reload(runflow_module); reload(decider)

    # Simulate a merge with 0 packs but scored pairs > 0
    merge_dir = runs_root / sid / "ai_packs" / "merge"
    (merge_dir / "results").mkdir(parents=True, exist_ok=True)
    (merge_dir / "packs").mkdir(parents=True, exist_ok=True)

    index_payload = {
        "totals": {
            "scored_pairs": 3,
            "created_packs": 0,
            "merge_zero_packs": True,
            "skip_counts": {"missing_original_creditor": 3},
            "skip_reason_top": "missing_original_creditor",
        },
        "pairs": [],
    }
    (merge_dir / "pairs_index.json").write_text(json.dumps(index_payload), encoding="utf-8")
    (merge_dir / "logs.txt").write_text("PACK_SKIPPED 9-10 reason=missing_original_creditor\n", encoding="utf-8")

    # Finalize merge → should persist metrics + barriers
    result = decider.finalize_merge_stage(sid, runs_root=runs_root)
    print("FINALIZE.METRICS:", result.get("metrics"))

    # Read runflow snapshot to assert wiring
    snapshot = json.loads((runs_root / sid / "runflow.json").read_text(encoding="utf-8"))
    m = snapshot["stages"]["merge"]
    print("SNAP.MERGE.SUMMARY:", m.get("summary"))
    print("SNAP.MERGE.METRICS:", m.get("metrics"))
    umb = snapshot.get("umbrella_barriers", {})
    print("UMBRELLA:", umb)

    # Hard checks (fail fast)
    summary = m.get("summary") or {}
    assert summary.get("merge_zero_packs") is True, "merge_zero_packs not persisted"
    assert summary.get("skip_reason_top") == "missing_original_creditor", "top skip reason missing"
    assert umb.get("merge_zero_packs") is True, "umbrella flag missing/false"
    print("OK: zero-pack metrics persisted and umbrella flag set.")

if __name__ == "__main__":
    main()
