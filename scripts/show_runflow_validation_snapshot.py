import argparse
import json
from pathlib import Path
from typing import Any


def load(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(description="Print key validation fields from runflow.json")
    p.add_argument("sid")
    p.add_argument("--runs-root", default="runs")
    args = p.parse_args(argv)

    run_dir = Path(args.runs_root) / args.sid
    runflow = load(run_dir / "runflow.json")
    manifest = load(run_dir / "manifest.json")

    stages = runflow.get("stages", {}) if isinstance(runflow, dict) else {}
    validation = stages.get("validation", {}) if isinstance(stages, dict) else {}

    print("=== manifest.ai.status.validation ===")
    vstat = (manifest.get("ai") or {}).get("status", {}).get("validation", {})
    print(json.dumps(vstat, indent=2, ensure_ascii=False))

    print("\n=== runflow.stages.validation ===")
    out = {
        "status": validation.get("status"),
        "sent": validation.get("sent"),
        "results": (validation.get("results") or {}),
        "metrics": (validation.get("metrics") or {}),
        "summary": (validation.get("summary") or {}),
        "last_writer": runflow.get("last_writer"),
        "updated_at": runflow.get("updated_at"),
    }
    print(json.dumps(out, indent=2, ensure_ascii=False))

    umbrella = runflow.get("umbrella_barriers", {}) if isinstance(runflow, dict) else {}
    print("\n=== runflow.umbrella_barriers (selected) ===")
    print(json.dumps({
        "validation_ready": umbrella.get("validation_ready"),
        "all_ready": umbrella.get("all_ready"),
        "checked_at": umbrella.get("checked_at"),
    }, indent=2, ensure_ascii=False))

    return 0


if __name__ == "__main__":
    import sys
    raise SystemExit(main(sys.argv[1:]))
