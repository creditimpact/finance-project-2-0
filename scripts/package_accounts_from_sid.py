from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple


def _latest_sid(base: Path) -> str | None:
    if not base.exists():
        return None
    dirs = [p for p in base.iterdir() if p.is_dir()]
    if not dirs:
        return None
    dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return dirs[0].name


def main(argv: List[str]) -> int:
    from backend.core.text.env_guard import ensure_env_and_paths
    from backend.core.logic.report_analysis.account_packager import (
        package_account_block,
        write_account_package,
    )

    ensure_env_and_paths()

    sid = os.getenv("SID") or (argv[1] if len(argv) > 1 else None)
    base = Path("traces") / "blocks"
    if not sid:
        sid = _latest_sid(base)
    if not sid:
        print("No SID provided and no traces/blocks found.")
        return 2

    run_dir = base / sid
    if not run_dir.exists():
        print(f"SID not found: {run_dir}")
        return 2

    # Load index once; support multiple possible key names for headline
    idx_path = run_dir / "_index.json"
    idx_rows: List[Dict[str, Any]] = []
    if idx_path.exists():
        try:
            idx_rows = json.loads(idx_path.read_text(encoding="utf-8")) or []
        except Exception:
            idx_rows = []
    headline_by_id: Dict[int, str] = {}
    for row in idx_rows:
        try:
            i = int(row.get("i"))
        except Exception:
            continue
        # Prefer hierarchical headline if present
        hl = row.get("index_headline") or row.get("headline") or row.get("heading")
        if isinstance(hl, str) and hl:
            headline_by_id[i] = hl

    # Iterate block_*.json files
    blocks = sorted(run_dir.glob("block_*.json"))
    scanned = 0
    written = 0
    out_dir = run_dir / "accounts"
    out_dir.mkdir(parents=True, exist_ok=True)

    for path in blocks:
        scanned += 1
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"Skip unreadable {path.name}: {e}")
            continue

        # Parse block id from filename `block_XX.json`
        try:
            stem = path.stem  # block_XX
            blk_id = int(stem.split("_")[1])
        except Exception:
            # Fallback to metadata or enumerate order
            blk_id = scanned

        index_headline = headline_by_id.get(blk_id)
        try:
            pkg = package_account_block(data, index_headline)
            write_account_package(sid, blk_id, pkg, index_slug=None)
            written += 1
        except Exception as e:
            print(f"Package error for {path.name}: {e}")
            continue

    print(json.dumps({
        "sid": sid,
        "blocks_scanned": scanned,
        "packages_written": written,
        "output_dir": str(out_dir),
    }, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

