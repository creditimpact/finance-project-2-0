import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from backend.core.logic.report_analysis.canonical_labels import (
    detect_value_type,
    LABEL_SCHEMA,
)


def _load_json(p: Path):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def _row_val_text(r: Dict[str, Any]) -> str:
    return " ".join([(r.get("tu") or ""), (r.get("ex") or ""), (r.get("eq") or "")]).strip()


def main() -> None:
    ap = argparse.ArgumentParser(description="Check value-type mismatches per block")
    ap.add_argument("--sid", required=True, help="Session ID under traces/blocks")
    ap.add_argument("--block", type=int, default=None, help="Optional single block ID")
    ap.add_argument("--root", default=str(Path.cwd()), help="Project root (defaults to CWD)")
    args = ap.parse_args()

    root = Path(args.root)
    base = root / "traces" / "blocks" / args.sid
    idx_path = base / "accounts_table" / "_table_index.json"
    idx = _load_json(idx_path) or {}
    blocks = list(idx.get("blocks") or [])
    if args.block is not None:
        blocks = [b for b in blocks if int(b.get("block_id", 0) or 0) == int(args.block)]

    def _safe(s: str) -> str:
        try:
            return "".join(ch if ord(ch) < 128 else "?" for ch in (s or ""))
        except Exception:
            return str(s)

    total_mismatches = 0
    for b in blocks:
        bid = int(b.get("block_id", 0) or 0)
        tbl_path = Path(b.get("table_path") or "")
        if not tbl_path.exists():
            print(f"Block {bid}: table not found: {tbl_path}")
            continue
        tbl = _load_json(tbl_path) or {}
        mode = str(tbl.get("mode") or "grid_table")
        rows: List[Dict[str, Any]] = list(tbl.get("rows") or [])
        if mode != "grid_table":
            print(f"Block {bid}: mode={mode} (skip)")
            continue
        mismatches: List[Dict[str, Any]] = []
        for r in rows:
            lab = str(r.get("label", "")).strip()
            if not lab:
                continue
            val_txt = _row_val_text(r)
            det = detect_value_type(val_txt)
            exp = LABEL_SCHEMA.get(lab, "text")
            if det != exp:
                mismatches.append({
                    "y": float(r.get("y", 0.0)),
                    "label": lab,
                    "detected": det,
                    "expected": exp,
                    "tu": r.get("tu", ""),
                    "ex": r.get("ex", ""),
                    "eq": r.get("eq", ""),
                })
        print(f"Block {bid}: mismatches={len(mismatches)}")
        total_mismatches += len(mismatches)
        for m in sorted(mismatches, key=lambda x: x["y"]):
            snippet = (" | ".join(s for s in [m.get("tu") or "", m.get("ex") or "", m.get("eq") or ""] if s))[:120]
            # Page-aware + provenance
            page = int(m.get("page", 0)) if isinstance(m.get("page"), (int, float)) else 0
            src = m.get("src") or {}
            counts = {
                "tu": len(src.get("tu") or []),
                "ex": len(src.get("ex") or []),
                "eq": len(src.get("eq") or []),
            } if isinstance(src, dict) else {"tu": 0, "ex": 0, "eq": 0}
            first_tok = None
            if isinstance(src, dict):
                for key in ("tu", "ex", "eq"):
                    arr = src.get(key) or []
                    if arr:
                        first_tok = arr[0]
                        break
            loc = ""
            if isinstance(first_tok, dict):
                try:
                    loc = f" p={int(first_tok.get('page', page) or 0)} ln={first_tok.get('line','-')} x0={float(first_tok.get('x0') or 0):.1f} y0={float(first_tok.get('y0') or 0):.1f}"
                except Exception:
                    loc = ""
            print(
                f"  page={page} y={m['y']:.1f} label='{_safe(m['label'])}' det={m['detected']} exp={m['expected']} :: "
                f"{_safe(snippet)}  src_counts(tu/ex/eq)={counts['tu']}/{counts['ex']}/{counts['eq']}{loc}"
            )

    if total_mismatches > 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
