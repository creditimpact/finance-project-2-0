from __future__ import annotations

from dataclasses import dataclass  # noqa: F401  # reserved for potential future use
from typing import List, Dict, Any, Tuple
from statistics import median
import json
import os
import re


def _mid_y(t: Dict[str, Any]) -> float:
    y0 = float(t.get("y0", 0))
    y1 = float(t.get("y1", y0))
    return (y0 + y1) / 2.0


def _mid_x(t: Dict[str, Any]) -> float:
    x0 = float(t.get("x0", 0))
    x1 = float(t.get("x1", x0))
    return (x0 + x1) / 2.0


def cluster_rows(tokens: List[Dict[str, Any]], dy: float = 4.0) -> List[List[Dict[str, Any]]]:
    """Group tokens into visual rows by Y midpoints; keep order within each row."""
    toks = sorted(tokens or [], key=lambda t: (_mid_y(t), _mid_x(t)))
    rows: List[List[Dict[str, Any]]] = []
    cur: List[Dict[str, Any]] = []
    cur_y: float | None = None
    for t in toks:
        y = _mid_y(t)
        if cur_y is None or abs(y - cur_y) <= dy:
            cur.append(t)
            # update running median of the row
            cur_y = median(_mid_y(x) for x in cur)
        else:
            rows.append(cur)
            cur = [t]
            cur_y = y
    if cur:
        rows.append(cur)
    # Ensure deterministic row order by median Y
    try:
        rows.sort(
            key=lambda r: float(
                median(
                    [
                        (float(t.get("y0", 0.0)) + float(t.get("y1", 0.0))) / 2.0
                        for t in r
                    ]
                )
            ) if r else 0.0
        )
    except Exception:
        pass
    return rows


def assign_by_bands(row: List[Dict[str, Any]], bands: Dict[str, Tuple[float, float]]) -> Dict[str, List[Dict[str, Any]]]:
    out: Dict[str, List[Dict[str, Any]]] = {"transunion": [], "experian": [], "equifax": []}
    for t in sorted(row or [], key=_mid_x):
        x = _mid_x(t)
        for k, (xL, xR) in bands.items():
            if xL <= x <= xR:
                out.setdefault(k, []).append(t)
                break
    return out


def package_block_raw_coords(block: Dict[str, Any], bureau_cols: Dict[str, Tuple[float, float]]) -> Dict[str, Any]:
    """Produce a raw, coordinate-based 3-column grid for a block.

    - Requires that block["layout_tokens"] already exist (L1 windowing provided).
    - Does not attempt label detection or field mapping.
    - Does not rely on page numbers; passes through existing layout_window if present.
    """
    ltoks = list(block.get("layout_tokens") or [])
    rows = cluster_rows(ltoks, dy=4.0)

    # If no bands were detected, still emit a RAW package with empty per-bureau cells
    if not bureau_cols:
        out_rows: List[Dict[str, Any]] = []
        for r in rows:
            _vals = [_mid_y(t) for t in r]
            yc = float(median(_vals)) if _vals else 0.0
            out_rows.append(
                {
                    "row_y": yc,
                    "transunion": [],
                    "experian": [],
                    "equifax": [],
                    "transunion_text": "",
                    "experian_text": "",
                    "equifax_text": "",
                }
            )
        return {
            "block_id": block.get("id") or block.get("block_id") or (block.get("meta", {}) or {}).get("block_id"),
            "block_heading": block.get("heading"),
            "index_headline": block.get("index_headline"),
            "mode": "raw_coords",
            "layout_window": (block.get("meta", {}) or {}).get("debug", {}) .get("layout_window"),
            "bureau_cols": {},
            "rows": out_rows,
            # token_count reflects all tokens in the window when bands are absent
            "stats": {"row_count": len(out_rows), "token_count": len(ltoks)},
        }
    grid: List[Dict[str, Any]] = []
    
    def _jx0(t: Dict[str, Any]) -> float:
        try:
            return (float(t.get("x0", 0.0)) + float(t.get("x1", 0.0))) / 2.0
        except Exception:
            return 0.0
    
    _SYM_ONLY = re.compile(r"^[^\w]+$")

    def _join_text(tokens_list: List[Dict[str, Any]]) -> str:
        toks = sorted(tokens_list or [], key=_jx0)
        out: List[str] = []
        prev_x1: float | None = None
        for tt in toks:
            txt = str(tt.get("text", "")).strip()
            if not txt or _SYM_ONLY.match(txt):
                continue
            try:
                x0 = float(tt.get("x0", 0.0))
                x1 = float(tt.get("x1", 0.0))
            except Exception:
                x0, x1 = 0.0, 0.0
            if prev_x1 is not None:
                gap = x0 - prev_x1
                avg_char = (x1 - x0) / max(1, len(txt))
                if gap > 1.5 * avg_char:
                    out.append(" ")
            out.append(txt)
            prev_x1 = x1
        return "".join(out).strip()
    for r in rows:
        bands = assign_by_bands(r, bureau_cols or {}) if bureau_cols else {"transunion": [], "experian": [], "equifax": []}
        # Sort tokens within each band left->right
        tu_sorted = sorted(bands.get("transunion", []), key=_jx0)
        ex_sorted = sorted(bands.get("experian", []), key=_jx0)
        eq_sorted = sorted(bands.get("equifax", []), key=_jx0)
        grid.append(
            {
                "row_y": (lambda _v: float(median(_v)) if _v else 0.0)([_mid_y(t) for t in r]),
                "transunion": [
                    {"text": t.get("text", ""), "x0": t.get("x0"), "y0": t.get("y0"), "x1": t.get("x1"), "y1": t.get("y1")}
                    for t in tu_sorted
                ],
                "experian": [
                    {"text": t.get("text", ""), "x0": t.get("x0"), "y0": t.get("y0"), "x1": t.get("x1"), "y1": t.get("y1")}
                    for t in ex_sorted
                ],
                "equifax": [
                    {"text": t.get("text", ""), "x0": t.get("x0"), "y0": t.get("y0"), "x1": t.get("x1"), "y1": t.get("y1")}
                    for t in eq_sorted
                ],
            }
        )

    # Concatenated text per cell using gap-aware LTR join
    for row in grid:
        row["transunion_text"] = _join_text(row.get("transunion", []))
        row["experian_text"] = _join_text(row.get("experian", []))
        row["equifax_text"] = _join_text(row.get("equifax", []))

    # Stats
    row_count = len(grid)
    token_count = 0
    for row in grid:
        token_count += sum(len(row.get(k, [])) for k in ("transunion", "experian", "equifax"))

    return {
        "block_id": block.get("id") or block.get("block_id") or (block.get("meta", {}) or {}).get("block_id"),
        "block_heading": block.get("heading"),
        "index_headline": block.get("index_headline"),
        "mode": "raw_coords",
        "layout_window": (block.get("meta", {}) or {}).get("debug", {}) .get("layout_window"),
        "bureau_cols": bureau_cols or {},
        "rows": grid,
        "stats": {"row_count": row_count, "token_count": token_count},
    }


__all__ = [
    "cluster_rows",
    "assign_by_bands",
    "package_block_raw_coords",
    "write_block_raw_coords",
]


def _slug(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"[^a-z0-9]+", "-", s).strip("-")
    return s or "blk"


def write_block_raw_coords(session_id: str, block_id: int, pkg: dict, index_headline: str | None) -> str:
    root = os.path.join("traces", "blocks", session_id, "accounts_raw")
    os.makedirs(root, exist_ok=True)
    name = f"account_raw_{block_id:02d}"
    slug_src = index_headline or pkg.get("block_heading") or f"blk-{block_id:02d}"
    if slug_src:
        name += "__" + _slug(slug_src)
    path = os.path.join(root, name + ".json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(pkg, f, ensure_ascii=False, indent=2)
    return path
