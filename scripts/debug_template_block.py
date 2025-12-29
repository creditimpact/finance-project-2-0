import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def _load_json(p: Path) -> Dict[str, Any] | List[Any] | None:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def _fmt_range(v: List[float] | tuple) -> str:
    try:
        a, b = float(v[0]), float(v[1])
        return f"[{a:.1f}, {b:.1f}]"
    except Exception:
        return str(v)


def main() -> None:
    ap = argparse.ArgumentParser(description="Debug a single SmartCredit template block")
    ap.add_argument("--sid", required=True, help="Session ID under traces/blocks")
    ap.add_argument("--block", required=True, type=int, help="Block ID to inspect")
    ap.add_argument("--root", default=str(Path.cwd()), help="Project root (defaults to CWD)")
    args = ap.parse_args()

    root = Path(args.root)
    base = root / "traces" / "blocks" / args.sid

    idx_path = base / "accounts_table" / "_table_index.json"
    win_path = base / "block_windows.json"
    layout_path = base / "layout_snapshot.json"

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

    table = _load_json(tbl_path) or {}
    rows = list(table.get("rows") or [])
    meta = dict(table.get("meta") or {})

    # Window info from Stage-A
    win = None
    page_no = None
    wins = _load_json(win_path) or {}
    for r in list(wins.get("blocks") or []):
        if int(r.get("block_id", 0) or 0) == args.block:
            win = r.get("window")
            try:
                page_no = int((win or {}).get("page", 0) or 0)
            except Exception:
                page_no = 0
            break

    print("=== BLOCK ===")
    print(f"SID       : {args.sid}")
    print(f"BLOCK     : {args.block}")
    print(f"HEADING   : {table.get('block_heading')}")
    print(f"ROWS      : {len(rows)}")

    print("\n=== WIN ===")
    if win:
        print(
            "page=%s x=[%.1f, %.1f] y=[%.1f, %.1f]"
            % (
                (win.get("page")),
                float(win.get("x_min", 0.0)),
                float(win.get("x_max", 0.0)),
                float(win.get("y_top", 0.0)),
                float(win.get("y_bottom", 0.0)),
            )
        )
    else:
        print("no window found in block_windows.json")

    print("\n=== EFF ===")
    exmin = meta.get("eff_x_min")
    exmax = meta.get("eff_x_max")
    print(f"eff_x_min: {exmin}")
    print(f"eff_x_max: {exmax}")
    print(f"label_max_x: {meta.get('label_max_x')}")

    print("\n=== BANDS ===")
    bands = dict(meta.get("bands") or {})
    if bands:
        for k, v in bands.items():
            print(f"{k:10s}: {_fmt_range(v)}")
    else:
        print("(none)")

    # Optional: token counts in y-band and effective window (if available)
    try:
        layout = _load_json(layout_path) or {}
        pages = list(layout.get("pages") or [])
        page = pages[page_no - 1] if page_no and 1 <= page_no <= len(pages) else None
        tokens = list((page or {}).get("tokens") or [])
        if win and tokens:
            y_top = float(win.get("y_top", 0.0)); y_bottom = float(win.get("y_bottom", 0.0))
            yband = [t for t in tokens if float(t.get("y0", 0.0)) <= y_bottom and float(t.get("y1", 0.0)) >= y_top]
            def midx(t):
                return (float(t.get("x0", 0.0)) + float(t.get("x1", 0.0))) / 2.0
            if exmin is not None and exmax is not None:
                inwin = [t for t in yband if float(exmin) <= midx(t) <= float(exmax)]
            else:
                inwin = [t for t in yband if float(win.get("x_min", 0.0)) <= midx(t) <= float(win.get("x_max", 0.0))]
            print("\n=== TOK ===")
            print(f"page_tokens={len(tokens)} yband={len(yband)} inwin={len(inwin)}")
        else:
            print("\n=== TOK ===\n(no tokens/page context)")
    except Exception:
        print("\n=== TOK ===\n(error loading layout tokens)")

    # Simple column distribution
    non_empty = {"label": 0, "tu": 0, "ex": 0, "eq": 0}
    for r in rows:
        for c in non_empty:
            if str(r.get(c, "")).strip():
                non_empty[c] += 1
    print("\n=== DIST ===")
    print(
        "label=%d tu=%d ex=%d eq=%d"
        % (non_empty["label"], non_empty["tu"], non_empty["ex"], non_empty["eq"])
    )


if __name__ == "__main__":
    main()

