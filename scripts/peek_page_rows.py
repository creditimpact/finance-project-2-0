import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def _load_json(p: Path):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def _first_src_loc(src_cell: List[dict]) -> str:
    if not isinstance(src_cell, list) or not src_cell:
        return ""
    t = src_cell[0] if isinstance(src_cell[0], dict) else None
    if not t:
        return ""
    try:
        pg = int(t.get("page", 0) or 0)
        ln = t.get("line", "-")
        x0 = float(t.get("x0") or 0.0)
        y0 = float(t.get("y0") or 0.0)
        return f"p={pg} ln={ln} x0={x0:.1f} y0={y0:.1f}"
    except Exception:
        return ""


def _truncate(s: str, n: int = 40) -> str:
    s = (s or "").replace("\n", " ").strip()
    return s if len(s) <= n else (s[: n - 1] + "…")


def main() -> None:
    ap = argparse.ArgumentParser(description="Peek rows per page for a block (QA)")
    ap.add_argument("--sid", required=True, help="Session ID under traces/blocks")
    ap.add_argument("--block", required=True, type=int, help="Block ID to inspect")
    ap.add_argument("--root", default=str(Path.cwd()), help="Project root (defaults to CWD)")
    ap.add_argument("--show-src", action="store_true", help="Print first source token bbox per cell")
    args = ap.parse_args()

    root = Path(args.root)
    base = root / "traces" / "blocks" / args.sid
    idx_path = base / "accounts_table" / "_table_index.json"
    idx = _load_json(idx_path) or {}
    blocks = list(idx.get("blocks") or [])
    entry = next((b for b in blocks if int(b.get("block_id", 0) or 0) == args.block), None)
    if not entry:
        print(f"Index missing block_id={args.block} at {idx_path}")
        return
    tbl_path = Path(entry.get("table_path") or "")
    if not tbl_path.exists():
        print(f"Table JSON not found: {tbl_path}")
        return
    tbl = _load_json(tbl_path) or {}
    rows = list(tbl.get("rows") or [])
    rows.sort(key=lambda r: (int(r.get("page", 0) or 0), float(r.get("y", 0.0) or 0.0)))

    # Header
    print("page | y        | label                              | TU                                 | EX                                 | EQ")
    print("-----+----------+------------------------------------+-------------------------------------+-------------------------------------+-------------------------------------")
    for r in rows:
        page = int(r.get("page", 0) or 0)
        y = float(r.get("y", 0.0) or 0.0)
        lab = _truncate(str(r.get("label", "")))
        tu = _truncate(str(r.get("tu", "")))
        ex = _truncate(str(r.get("ex", "")))
        eq = _truncate(str(r.get("eq", "")))
        print(f"{page:>4d} | {y:>8.1f} | {lab:<36} | {tu:<37} | {ex:<37} | {eq:<37}")
        if args.show_src:
            src = r.get("src") or {}
            if isinstance(src, dict):
                l0 = _first_src_loc(src.get("label") or [])
                t0 = _first_src_loc(src.get("tu") or [])
                e0 = _first_src_loc(src.get("ex") or [])
                q0 = _first_src_loc(src.get("eq") or [])
                print(f"      src: label[{len(src.get('label') or [])}] {l0} | tu[{len(src.get('tu') or [])}] {t0} | ex[{len(src.get('ex') or [])}] {e0} | eq[{len(src.get('eq') or [])}] {q0}")


if __name__ == "__main__":
    main()

