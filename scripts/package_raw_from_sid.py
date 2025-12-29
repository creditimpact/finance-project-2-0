"""Compatibility wrapper to the new RAW builder.

This script historically repackaged RAW from block_XX.json + debug windows.
It now delegates to the Stage-B builder that consumes layout_snapshot.json
and block_windows.json to produce accounts_raw/* and _raw_index.json.
"""

import os
import sys
from pathlib import Path as _Path

from backend.core.logic.report_analysis.raw_builder import build_raw_from_windows


def main(sid: str) -> None:  # preserved signature
    build_raw_from_windows(sid, _Path.cwd())


if __name__ == "__main__":
    sid = None
    # Prefer CLI arg
    if len(sys.argv) > 1:
        sid = sys.argv[1]
    # Fallback to env var
    if not sid:
        sid = os.environ.get("SID")
    # Fallback to latest session under traces/blocks
    if not sid:
        base = _Path("traces") / "blocks"
        try:
            sid = sorted([p.name for p in base.iterdir() if p.is_dir()])[-1]
        except Exception:
            sid = None
    if not sid:
        print("Usage: python scripts/package_raw_from_sid.py <SID>  (or set SID env)")
        raise SystemExit(2)
    main(sid)
