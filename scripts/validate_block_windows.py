import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _load_json(p: Path):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def _as_int(v: Any, default: int = 0) -> int:
    try:
        return int(v)
    except Exception:
        return default


def _as_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default


def _fmt_span(sp: Dict[str, Any]) -> str:
    p = _as_int(sp.get("page"))
    xm = _as_float(sp.get("x_min"))
    xM = _as_float(sp.get("x_max"))
    y0 = _as_float(sp.get("y_min"))
    y1 = _as_float(sp.get("y_max"))
    ln0 = sp.get("line_min")
    ln1 = sp.get("line_max")
    t = _as_int(sp.get("token_count"))
    tin = _as_int(sp.get("token_count_in_span")) if "token_count_in_span" in sp else t
    tass = _as_int(sp.get("token_count_assigned")) if "token_count_assigned" in sp else 0
    assigned = bool(sp.get("assigned")) if "assigned" in sp else False
    ln_part = f" lines={ln0}-{ln1}" if (ln0 is not None or ln1 is not None) else ""
    return (
        f"page={p} x=[{xm:.1f},{xM:.1f}] y=[{y0:.1f},{y1:.1f}] tokens={t} in_span={tin} assigned={tass} ({'Y' if assigned else 'N'}){ln_part}"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Validate Stage-A spans for a session")
    ap.add_argument("--sid", required=True, help="Session ID under traces/blocks")
    ap.add_argument("--root", default=str(Path.cwd()), help="Project root (defaults to CWD)")
    args = ap.parse_args()

    base = Path(args.root) / "traces" / "blocks" / args.sid
    bw_path = base / "block_windows.json"
    if not bw_path.exists():
        print(f"block_windows.json not found at {bw_path}")
        raise SystemExit(2)
    bw = _load_json(bw_path) or {}
    blocks = list(bw.get("blocks") or [])
    if not blocks:
        print("No blocks in block_windows.json")
        raise SystemExit(2)

    TOL = 0.5
    has_issues = False

    # Sort by block_id
    blocks_sorted = sorted(blocks, key=lambda b: _as_int(b.get("block_id", 0)))

    for idx, blk in enumerate(blocks_sorted):
        bid = _as_int(blk.get("block_id", 0))
        print(f"\n== Block {bid} :: {blk.get('heading')} ==")
        # Anchors
        ha = blk.get("head_anchor") or {}
        ta = blk.get("tail_anchor") or {}
        if ha:
            print(
                f"head_anchor: page={_as_int(ha.get('page'))} y={_as_float(ha.get('y')):.1f} line={ha.get('line')}"
            )
        else:
            print("head_anchor: -")
        if ta:
            print(
                f"tail_anchor: page={_as_int(ta.get('page'))} y={_as_float(ta.get('y')):.1f} line={ta.get('line')}"
            )
        else:
            print("tail_anchor: -")

        spans = list(blk.get("spans") or [])
        if not spans:
            print("(no spans)")
        else:
            print("spans:")
            for sp in sorted(spans, key=lambda s: (_as_int(s.get("page")), _as_float(s.get("y_min")))):
                print("  - " + _fmt_span(sp))

        # Summary
        pages = sorted({int(s.get("page")) for s in spans if s and s.get("page") is not None}) if spans else []
        tokens_in_spans = sum(_as_int(s.get("token_count_in_span", s.get("token_count", 0))) for s in spans)
        print(f"summary: pages={len(pages)} tokens_in_spans={tokens_in_spans}")

        # Compare with next block for gaps/overlaps on shared pages
        if idx + 1 < len(blocks_sorted):
            nxt = blocks_sorted[idx + 1]
            spans_n = list(nxt.get("spans") or [])
            if spans and spans_n:
                byp_cur: Dict[int, Tuple[float, float]] = {}
                byp_nxt: Dict[int, Tuple[float, float]] = {}
                for s in spans:
                    p = _as_int(s.get("page"))
                    ymx = _as_float(s.get("y_max"))
                    ymin = _as_float(s.get("y_min"))
                    prev = byp_cur.get(p)
                    if prev is None:
                        byp_cur[p] = (ymin, ymx)
                    else:
                        byp_cur[p] = (min(prev[0], ymin), max(prev[1], ymx))
                for s in spans_n:
                    p = _as_int(s.get("page"))
                    ymx = _as_float(s.get("y_max"))
                    ymin = _as_float(s.get("y_min"))
                    prev = byp_nxt.get(p)
                    if prev is None:
                        byp_nxt[p] = (ymin, ymx)
                    else:
                        byp_nxt[p] = (min(prev[0], ymin), max(prev[1], ymx))
                shared = sorted(set(byp_cur.keys()) & set(byp_nxt.keys()))
                for p in shared:
                    c_ymax = byp_cur[p][1]
                    n_ymin = byp_nxt[p][0]
                    delta = n_ymin - c_ymax
                    if delta < -TOL:
                        print(f"OVERLAP: next block { _as_int(nxt.get('block_id',0)) } on page {p}: current_ymax={c_ymax:.1f} > next_ymin={n_ymin:.1f}")
                        has_issues = True
                    elif delta > TOL:
                        print(f"GAP: next block { _as_int(nxt.get('block_id',0)) } on page {p}: next_ymin={n_ymin:.1f} - current_ymax={c_ymax:.1f} = {delta:.1f}")
                        has_issues = True

    if has_issues:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

