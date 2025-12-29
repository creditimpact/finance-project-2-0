"""Verify that validation natives are properly wired into manifest and runflow."""

import json
import sys
from pathlib import Path

# Ensure project root on path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def verify_sid(sid: str, runs_root: Path | None = None) -> dict:
    """Verify validation natives for a given SID."""
    import os
    
    if runs_root is None:
        runs_root = Path(os.getenv("RUNS_ROOT", "runs"))
    else:
        runs_root = Path(runs_root)
    
    run_dir = runs_root / sid
    manifest_path = run_dir / "manifest.json"
    runflow_path = run_dir / "runflow.json"
    
    results = {
        "sid": sid,
        "manifest_exists": manifest_path.exists(),
        "runflow_exists": runflow_path.exists(),
        "validation_natives_ok": False,
        "runflow_marker_ok": False,
        "errors": [],
    }
    
    # Check manifest
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            
            # Check ai.packs.validation
            ai_packs_val = manifest.get("ai", {}).get("packs", {}).get("validation", {})
            required_keys = ["base", "dir", "packs", "packs_dir", "results", "results_dir", "index", "logs"]
            missing_keys = [k for k in required_keys if not ai_packs_val.get(k)]
            
            if missing_keys:
                results["errors"].append(f"manifest.ai.packs.validation missing keys: {missing_keys}")
            
            # Check artifacts.ai.packs.validation
            artifacts_val = manifest.get("artifacts", {}).get("ai", {}).get("packs", {}).get("validation", {})
            artifacts_missing = [k for k in required_keys if not artifacts_val.get(k)]
            
            if artifacts_missing:
                results["errors"].append(f"manifest.artifacts.ai.packs.validation missing keys: {artifacts_missing}")
            
            # Check ai.validation
            ai_val = manifest.get("ai", {}).get("validation", {})
            if not ai_val.get("base") or not ai_val.get("dir"):
                results["errors"].append("manifest.ai.validation missing base/dir")
            
            # Ensure NO status changes
            status_section = manifest.get("ai", {}).get("status", {}).get("validation", {})
            if status_section.get("sent") or status_section.get("completed_at"):
                results["errors"].append("UNEXPECTED: validation status flags changed (sent/completed_at)")
            
            if not missing_keys and not artifacts_missing and ai_val.get("base"):
                results["validation_natives_ok"] = True
                results["manifest_paths"] = {
                    "base": ai_packs_val.get("base"),
                    "packs": ai_packs_val.get("packs"),
                    "results": ai_packs_val.get("results"),
                    "index": ai_packs_val.get("index"),
                    "logs": ai_packs_val.get("logs"),
                }
        except Exception as e:
            results["errors"].append(f"manifest read error: {e}")
    else:
        results["errors"].append("manifest.json not found")
    
    # Check runflow
    if runflow_path.exists():
        try:
            runflow = json.loads(runflow_path.read_text(encoding="utf-8"))
            
            meta = runflow.get("meta", {})
            if meta.get("validation_paths_initialized"):
                results["runflow_marker_ok"] = True
                results["runflow_marker"] = {
                    "validation_paths_initialized": True,
                    "validation_paths_initialized_at": meta.get("validation_paths_initialized_at"),
                }
            else:
                results["errors"].append("runflow.meta.validation_paths_initialized not set")
            
            # Ensure validation stage status NOT changed
            validation_stage = runflow.get("stages", {}).get("validation", {})
            if validation_stage.get("status") in ("success", "sent", "completed"):
                results["errors"].append("UNEXPECTED: validation stage status changed")
        except Exception as e:
            results["errors"].append(f"runflow read error: {e}")
    else:
        results["errors"].append("runflow.json not found")
    
    return results


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: python devtools/verify_validation_natives.py <sid> [runs_root]", file=sys.stderr)
        return 2
    
    sid = sys.argv[1]
    runs_root = Path(sys.argv[2]) if len(sys.argv) > 2 else None
    
    results = verify_sid(sid, runs_root)
    
    print(json.dumps(results, ensure_ascii=False, indent=2))
    
    if results["validation_natives_ok"] and results["runflow_marker_ok"] and not results["errors"]:
        print("\n✓ All checks passed", file=sys.stderr)
        return 0
    else:
        print(f"\n✗ Checks failed: {len(results['errors'])} errors", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
