import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple


def _load_json(p: Path) -> Dict[str, Any] | List[Any] | None:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def main() -> None:
    ap = argparse.ArgumentParser(description="List detected left-column labels for a block")
    ap.add_argument("--sid", required=True, help="Session ID under traces/blocks")
    ap.add_argument("--block", required=True, type=int, help="Block ID to inspect")
    ap.add_argument("--root", default=str(Path.cwd()), help="Project root (defaults to CWD)")
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
    rows.sort(key=lambda r: float(r.get("y", 0.0)))

    # Distinct non-empty label values in reading order
    seen = set()
    ordered: List[Tuple[float, str]] = []
    for r in rows:
        lab = str(r.get("label", "")).strip()
        if not lab:
            continue
        key = lab
        if key in seen:
            continue
        seen.add(key)
        ordered.append((float(r.get("y", 0.0)), lab))

    print("=== LABELS (distinct, in order) ===")
    for y, lab in ordered:
        print(f"y={y:.1f}  {lab}")

    # Optional: anchors from meta.debug.anchors
    meta = dict(tbl.get("meta") or {})
    dbg = dict(meta.get("debug") or {})
    anchors = list(dbg.get("anchors") or [])
    if anchors:
        try:
            anchors = sorted(anchors, key=lambda a: float(a.get("y", 0.0)))
        except Exception:
            pass
        print("\n=== ANCHORS (collected) ===")
        for a in anchors:
            try:
                yv = float(a.get("y", 0.0))
            except Exception:
                yv = 0.0
            print(f"y={yv:.1f}  {str(a.get('label') or '')}")


if __name__ == "__main__":
    main()

