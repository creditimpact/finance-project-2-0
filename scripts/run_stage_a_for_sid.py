from __future__ import annotations

try:  # import shim for direct execution
    import scripts._bootstrap  # KEEP FIRST
except ModuleNotFoundError:  # fallback path setup
    import sys as _sys
    from pathlib import Path as _Path

    _repo_root = _Path(__file__).resolve().parent.parent
    if str(_repo_root) not in _sys.path:
        _sys.path.insert(0, str(_repo_root))
    import scripts._bootstrap  # type: ignore  # KEEP FIRST

import json
import os
import sys
from pathlib import Path

from backend.pipeline.runs import RunManifest, require_pdf_for_sid
from backend.core.logic.report_analysis.text_provider import (
    extract_and_cache_text,
    load_cached_text,
)
from backend.core.logic.report_analysis.block_exporter import run_stage_a as _run_stage_a


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print("Usage: python scripts/run_stage_a_for_sid.py <SID>")
        return 2

    sid = argv[1]

    # Ensure PDF exists via manifest or uploads
    try:
        pdf_path = require_pdf_for_sid(sid)
    except FileNotFoundError as exc:
        print(json.dumps({"sid": sid, "ok": False, "error": str(exc)}))
        return 2

    # Ensure cached text is available (build if missing)
    try:
        cached = load_cached_text(sid)
        have = bool(cached and cached.get("pages"))
    except Exception:
        have = False
    if not have:
        ocr_on = os.getenv("OCR_ENABLED", "0") == "1"
        extract_and_cache_text(session_id=sid, pdf_path=str(pdf_path), ocr_enabled=ocr_on)

    # Prepare canonical output directory under runs/<SID>/traces/accounts_table
    m = RunManifest.for_sid(sid, allow_create=False)
    traces_dir = m.ensure_run_subdir("traces_dir", "traces")
    accounts_out_dir = (traces_dir / "accounts_table").resolve()
    accounts_out_dir.mkdir(parents=True, exist_ok=True)

    # Execute Stage A
    result = _run_stage_a(sid=sid, accounts_out_dir=accounts_out_dir)

    # Surface key artifact paths
    artifacts = (result or {}).get("artifacts") or {}
    payload = {
        "sid": sid,
        "ok": bool(result and result.get("ok", True)),
        "accounts_table_dir": str(accounts_out_dir),
        "artifacts": {
            "full_tsv": str(artifacts.get("full_tsv", "")),
            "accounts_json": str(artifacts.get("accounts_json", "")),
            "general_info_json": str(artifacts.get("general_info_json", "")),
        },
    }
    print(json.dumps(payload, ensure_ascii=False))
    return 0 if payload["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
