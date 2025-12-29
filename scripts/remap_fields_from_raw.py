from __future__ import annotations

import json
import os
import os.path as _p
from typing import Any, Dict

from backend.core.logic.report_analysis.mapper_from_raw import map_raw_to_fields


def main(sid: str) -> int:
    base = _p.join("traces", "blocks", sid, "accounts_raw")
    idx_path = _p.join(base, "_raw_index.json")
    if not _p.exists(idx_path):
        print("RAW index not found:", idx_path)
        return 2
    try:
        idx = json.loads(open(idx_path, "r", encoding="utf-8").read())
    except Exception as e:
        print("Cannot read RAW index:", e)
        return 2
    blocks = idx.get("blocks") or []
    out_dir = _p.join("traces", "blocks", sid, "accounts")
    os.makedirs(out_dir, exist_ok=True)
    total_rows = 0
    filled_counts = {"transunion": 0, "experian": 0, "equifax": 0}
    for b in blocks:
        raw_path = b.get("raw_coords_path")
        if not raw_path or not _p.exists(raw_path):
            continue
        try:
            raw_pkg = json.loads(open(raw_path, "r", encoding="utf-8").read())
        except Exception:
            continue
        rows = raw_pkg.get("rows") or []
        total_rows += len(rows)
        fields = map_raw_to_fields(raw_pkg)
        # Count filled keys per bureau
        for bureau in ("transunion", "experian", "equifax"):
            filled_counts[bureau] += sum(1 for v in fields.get(bureau, {}).values() if str(v or "").strip())
        # Write compatibility accounts file
        block_id = b.get("block_id") or 0
        out_path = _p.join(out_dir, f"account_{int(block_id):02d}.json")
        open(out_path, "w", encoding="utf-8").write(json.dumps({
            "block_id": block_id,
            "block_heading": b.get("heading"),
            "fields": fields,
        }, ensure_ascii=False, indent=2))
    print(json.dumps({
        "sid": sid,
        "rows_total": total_rows,
        "filled_counts": filled_counts,
        "out_dir": out_dir,
    }, indent=2))
    return 0


if __name__ == "__main__":
    sid = os.environ.get("SID")
    if not sid:
        base = _p.join("traces", "blocks")
        try:
            sid = sorted(os.listdir(base))[-1]
        except Exception:
            sid = None
    if not sid:
        print("No SID provided and no traces/blocks found")
    else:
        raise SystemExit(main(sid))

