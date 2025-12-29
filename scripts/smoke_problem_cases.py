try:  # pragma: no cover - import shim
    import scripts._bootstrap  # KEEP FIRST
except ModuleNotFoundError:  # pragma: no cover - fallback for direct execution
    import sys as _sys
    from pathlib import Path as _Path

    _repo_root = _Path(__file__).resolve().parent.parent
    if str(_repo_root) not in _sys.path:
        _sys.path.insert(0, str(_repo_root))
    import scripts._bootstrap  # type: ignore  # KEEP FIRST

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from backend.core.logic.report_analysis.problem_case_builder import build_problem_cases
from backend.core.logic.report_analysis.problem_extractor import detect_problem_accounts
from backend.pipeline.runs import RunManifest


def _resolve_manifest(manifest_arg: str | None, sid_arg: str | None) -> RunManifest:
    if manifest_arg:
        return RunManifest(Path(manifest_arg)).load()
    if sid_arg:
        return RunManifest.for_sid(sid_arg)
    # fallback to runs/current.txt
    cur = Path("runs") / "current.txt"
    if cur.exists():
        sid = cur.read_text(encoding="utf-8").strip()
        if sid:
            return RunManifest.for_sid(sid)
    return RunManifest.from_env_or_latest()


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Smoke test helper for problem account extraction",
    )
    ap.add_argument("--sid", help="Run SID (defaults to runs/current.txt)")
    ap.add_argument("--manifest", help="Path to run manifest")
    args = ap.parse_args()

    m = _resolve_manifest(args.manifest, args.sid)
    sid = m.sid

    # Analyzer: read strictly via manifest, no legacy inputs
    candidates: List[Dict[str, Any]] = detect_problem_accounts(sid)
    summary = build_problem_cases(sid, candidates)

    # Resolve canonical outputs via manifest for reporting
    try:
        accounts_index = Path(m.get("cases", "accounts_index"))
    except Exception:
        # Fallback derive from standard layout if not registered (older runs)
        cases_dir = Path(summary.get("out") or (Path("runs") / sid / "cases"))
        accounts_index = cases_dir / "accounts" / "index.json"

    out_obj: Dict[str, Any] = {
        "sid": sid,
        "problematic": int(summary.get("problematic", 0) or 0),
        "out": str(accounts_index.resolve()),
    }
    # Optional: list first few account folders
    try:
        acc_index_obj = json.loads(accounts_index.read_text(encoding="utf-8"))
        items = list(acc_index_obj.get("items") or [])
        out_obj["sample"] = items[: min(5, len(items))]
    except Exception:
        pass

    print(json.dumps(out_obj, ensure_ascii=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

