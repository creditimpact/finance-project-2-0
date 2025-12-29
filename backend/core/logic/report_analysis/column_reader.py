from __future__ import annotations

from typing import Any, Dict, List, Tuple
import re
import os
import logging


from .normalize_fields import (
    LABELS,
    LABEL_TO_KEY as _LABEL_TO_KEY,
    CANONICAL_KEYS as _ALL_KEYS,
    clean_value as _nf_clean,
    join_parts as _nf_join,
    ensure_all_keys as _nf_ensure_keys,
)

BUREAUS = {0: "transunion", 1: "experian", 2: "equifax"}
BLOCK_DEBUG = os.getenv("BLOCK_DEBUG", "0") == "1"
logger = logging.getLogger(__name__)


def _norm(s: str) -> str:
    s = (s or "").replace("\u00A0", " ")
    s = re.sub(r"\s+", " ", s.strip())
    s = s.rstrip(":")
    return s.upper()


def _is_label(text: str) -> str | None:
    # Backward-compat helper used only by older flow; keep lightweight
    t = _norm(text)
    # Accept uppercase forms if provided
    legacy = {
        "ACCOUNT #",
        "HIGH BALANCE",
        "LAST VERIFIED",
        "DATE OF LAST ACTIVITY",
        "DATE REPORTED",
        "DATE OPENED",
        "BALANCE OWED",
        "CLOSED DATE",
        "ACCOUNT RATING",
        "ACCOUNT DESCRIPTION",
        "DISPUTE STATUS",
        "CREDITOR TYPE",
        "ORIGINAL CREDITOR",
        "ACCOUNT STATUS",
        "PAYMENT STATUS",
        "CREDITOR REMARKS",
        "PAYMENT AMOUNT",
        "LAST PAYMENT",
        "TERM LENGTH",
        "PAST DUE AMOUNT",
        "ACCOUNT TYPE",
        "PAYMENT FREQUENCY",
        "CREDIT LIMIT",
    }
    return t if t in legacy else None


def _collect_value_tokens(
    tokens: List[dict],
    y_min: float,
    y_max: float,
    x_min: float,
) -> Dict[int, List[dict]]:
    by_col: Dict[int, List[dict]] = {0: [], 1: [], 2: []}
    for t in tokens:
        col = t.get("col")
        if col not in (0, 1, 2):
            continue
        y0 = float(t.get("y0", 0))
        x0 = float(t.get("x0", 0))
        if y0 < y_min or y0 > y_max:
            continue
        if x0 <= x_min:  # only take tokens to the right of label
            continue
        if _is_label(str(t.get("text", ""))):  # skip labels in case they are column-assigned
            continue
        by_col[col].append(t)
    # Stable sort: by line then x
    for k in by_col:
        by_col[k].sort(key=lambda z: (int(z.get("line", 0)), float(z.get("x0", 0))))
    return by_col


def _norm_simple(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _line_center_y(line_tokens: List[dict]) -> float:
    ys: List[float] = []
    for t in line_tokens:
        try:
            y0 = float(t.get("y0", 0.0))
            y1 = float(t.get("y1", y0))
        except Exception:
            continue
        ys.append((y0 + y1) / 2.0)
    return sorted(ys)[len(ys) // 2] if ys else 0.0


def _token_mid_x(t: dict) -> float:
    try:
        x0 = float(t.get("x0", 0.0))
        x1 = float(t.get("x1", x0))
        return (x0 + x1) / 2.0
    except Exception:
        return 0.0


def extract_bureau_table(block: dict) -> Dict[str, Dict[str, str]]:
    """
    Build a 3-bureau table for all 22 keys from layout tokens.

    Input: block with 'layout_tokens' and optional 'meta.debug.layout_window'.
    Steps:
      1) For each LABEL:
         a) Find label line by matching normalized left text.
         b) Determine row-Y as median(midY of tokens on that line).
         c) Collect non-label tokens within Y-band [rowY-3, rowY+3].
         d) For each bureau, keep tokens whose midX is within that bureau’s (xL,xR).
         e) Sort by X; join with spaces; cleanup: normalize dash placeholders to "--" and leave blanks empty; merge hard-wraps for next Y band within ΔY ≤ 6.
      2) Build dict for each bureau with all 22 keys present (empty string if missing).
    """

    from .column_reader import detect_bureau_columns  # local import to avoid cycles

    tokens: List[dict] = list((block or {}).get("layout_tokens") or [])

    # Detect bureau bands from header tokens on the same block
    bureau_cols = detect_bureau_columns(tokens)
    if not bureau_cols:
        # Fallback: infer from assigned 'col' if available
        cols: Dict[int, List[float]] = {0: [], 1: [], 2: []}
        for t in tokens:
            c = t.get("col")
            if c in (0, 1, 2):
                cols[c].append(_token_mid_x(t))
        temp: Dict[str, Tuple[float, float]] = {}
        for c, name in BUREAUS.items():
            xs = sorted(cols[c])
            if xs:
                temp[name] = (xs[0] - 1.0, xs[-1] + 1.0)
        bureau_cols = temp

    # Prepare result with all keys empty
    result: Dict[str, Dict[str, str]] = {b: {k: "" for k in _ALL_KEYS} for b in ("transunion", "experian", "equifax")}
    if not tokens or not bureau_cols:
        return result

    # Normalize labels for matching
    norm_label_map = {lbl: _norm_simple(lbl) for lbl in LABELS}
    norm_to_lbl = {v: k for k, v in norm_label_map.items()}
    known_norms = set(norm_to_lbl.keys())

    # Group tokens by line id (preferred) or y bucket
    lines: Dict[int, List[dict]] = {}
    if any("line" in t for t in tokens):
        for t in tokens:
            try:
                ln = int(t.get("line", 0))
            except Exception:
                ln = 0
            lines.setdefault(ln, []).append(t)
    else:
        for t in tokens:
            yb = int(round(float(t.get("y0", 0.0)) / 6.0))
            lines.setdefault(yb, []).append(t)

    # Build ordered lines with metadata
    line_infos: List[Tuple[int, float, str, List[dict]]] = []  # (id, y, norm_text, toks)
    for lid, toks in lines.items():
        toks_sorted = sorted(toks, key=lambda z: float(z.get("x0", 0.0)))
        joined = " ".join(str(z.get("text", "")) for z in toks_sorted)
        norm = _norm_simple(joined)
        yc = _line_center_y(toks_sorted)
        line_infos.append((lid, yc, norm, toks_sorted))
    line_infos.sort(key=lambda x: x[1])

    # Compute left boundary beyond which values start (min of bands)
    try:
        min_xL = min(v[0] for v in bureau_cols.values())
    except Exception:
        min_xL = -1e9

    # Helper: collect tokens within a y-band and bureau band
    def collect_for_band(yc: float) -> Dict[str, List[str]]:
        band_min, band_max = yc - 3.0, yc + 3.0
        parts: Dict[str, List[str]] = {"transunion": [], "experian": [], "equifax": []}
        for _, yline, _normtxt, toks in line_infos:
            if yline < band_min or yline > band_max:
                continue
            for t in toks:
                txt = str(t.get("text", "")).strip()
                if not txt:
                    continue
                mx = _token_mid_x(t)
                if mx < min_xL:
                    continue
                for bname, (xL, xR) in bureau_cols.items():
                    if xL <= mx <= xR:
                        parts[bname].append((mx, txt))
                        break
        # sort within each bureau by x
        out_parts: Dict[str, List[str]] = {}
        for b in parts:
            out_parts[b] = [t for _, t in sorted(parts[b], key=lambda p: p[0])]
        return out_parts

    # Build values per label, with continuation within ΔY ≤ 6
    for lbl in LABELS:
        expected_norm = norm_label_map[lbl]
        field_key = _LABEL_TO_KEY[lbl]
        # find matching line
        rowY = None
        for lid, yc, normtxt, toks in line_infos:
            if normtxt == expected_norm:
                rowY = yc
                break
        if rowY is None:
            # label not present; leave empty
            continue

        # Collect initial band
        parts0 = collect_for_band(rowY)
        # Try continuation: next band within +6 that is not a label line
        acc: Dict[str, List[str]] = {"transunion": [], "experian": [], "equifax": []}
        for b in ("transunion", "experian", "equifax"):
            acc[b].extend(parts0.get(b, []))

        # search for next nearby line that is not a label
        next_candidates = [y for _, y, normtxt, _ in line_infos if (y > rowY + 3.0 and y <= rowY + 9.0 and normtxt not in known_norms)]
        # We can chain a couple if very close; iterate sorted
        for ny in sorted(next_candidates):
            parts_n = collect_for_band(ny)
            for b in ("transunion", "experian", "equifax"):
                if parts_n.get(b):
                    # Add a separator marker, then the next parts
                    acc[b].append(" ")
                    acc[b].extend(parts_n[b])

        # Assign cleaned values (join with single spaces, collapse excess)
        for b in ("transunion", "experian", "equifax"):
            val = _nf_clean(_nf_join(acc[b]))
            result[b][field_key] = val

    # Ensure all 22 keys exist
    for b in ("transunion", "experian", "equifax"):
        result[b] = _nf_ensure_keys(result[b])

    return result


import math

from .header_utils import normalize_bureau_header


def detect_bureau_columns(header_tokens: list[dict]) -> dict[str, tuple[float, float]]:
    """Detect stable, non-overlapping X-intervals per bureau from header tokens.

    Algorithm
    - Filter tokens whose normalized text ∈ {transunion, experian, equifax}.
    - For each bureau, choose a representative x-center as the median of its
      occurrences (tolerates duplicates and minor drift).
    - Sort by x-center; for each bureau i compute a symmetric interval around
      its center using a half-distance to neighbors with a 0.95 margin:
        w_i = min((x[i+1]-x[i])/2, (x[i]-x[i-1])/2) * 0.95
      For edges, use the single available neighbor. Clamp: 60 ≤ w ≤ 140.
    - Finally, ensure no overlaps by clipping adjacent intervals at the
      midpoint between neighboring centers with a small 0.5px gap.

    Returns a mapping like {"transunion": (xL, xR), "experian": (xL, xR), "equifax": (xL, xR)}
    for all bureaus detected in the provided tokens.
    """
    # Collect mid-x values per bureau
    per_bureau: dict[str, list[float]] = {"transunion": [], "experian": [], "equifax": []}
    for t in header_tokens or []:
        name = normalize_bureau_header(str(t.get("text", "")))
        if name not in {"transunion", "experian", "equifax"}:
            continue
        try:
            x0 = float(t.get("x0", 0.0))
            x1 = float(t.get("x1", x0))
        except Exception:
            continue
        mid = (x0 + x1) / 2.0
        per_bureau[name].append(mid)

    # Choose a stable representative mid for each present bureau (median)
    centers: list[tuple[str, float]] = []
    for name, xs in per_bureau.items():
        if not xs:
            continue
        xs_sorted = sorted(xs)
        m = xs_sorted[len(xs_sorted) // 2]
        centers.append((name, m))

    # Need at least two to derive widths; ideally three
    if len(centers) < 2:
        return {}

    centers.sort(key=lambda p: p[1])

    # Compute base symmetric intervals using neighbor distances
    intervals: list[tuple[str, float, float, float]] = []  # (name, mid, L, R)
    n = len(centers)
    for i, (name, mid) in enumerate(centers):
        left_gap = centers[i][1] - centers[i - 1][1] if i - 1 >= 0 else math.inf
        right_gap = centers[i + 1][1] - centers[i][1] if i + 1 < n else math.inf
        base_gap = right_gap if left_gap is math.inf else left_gap if right_gap is math.inf else min(left_gap, right_gap)
        w = (base_gap / 2.0) * 0.95 if math.isfinite(base_gap) else 100.0
        # Clamp width to a sane range
        w = max(60.0, min(140.0, w))
        L, R = mid - w, mid + w
        intervals.append((name, mid, L, R))

    # De-overlap by clipping at midpoints between neighboring centers
    adjusted: list[tuple[str, float, float]] = []  # (name, L, R)
    for i, (name, mid, L, R) in enumerate(intervals):
        # Clip right edge by next boundary
        if i + 1 < n:
            next_mid = intervals[i + 1][1]
            boundary = (mid + next_mid) / 2.0
            if R > boundary - 0.5:
                R = boundary - 0.5
        # Clip left edge by previous boundary
        if i - 1 >= 0:
            prev_mid = intervals[i - 1][1]
            boundary = (prev_mid + mid) / 2.0
            if L < boundary + 0.5:
                L = boundary + 0.5
        # Ensure valid ordering
        if L > R:
            # Collapse to a tiny non-overlapping span around mid
            L, R = mid - 1.0, mid + 1.0
        adjusted.append((name, L, R))

    # Build output mapping
    out: dict[str, tuple[float, float]] = {}
    for name, L, R in adjusted:
        out[name] = (float(L), float(R))
    return out


__all__ = ["extract_bureau_table", "detect_bureau_columns"]


def build_debug_rows(tokens: List[dict], bureau_cols: Dict[str, Tuple[float, float]], max_labels: int = 8) -> List[Dict[str, Any]]:
    """Build debug rows for the first N labels with y and TU/EX/EQ previews.

    Returns a list of dicts: {label, y, tu, ex, eq}.
    Safe when tokens or columns are missing (returns []).
    """
    try:
        if not tokens or not bureau_cols:
            return []

        # Reuse lightweight helpers
        norm_label_map = {lbl: _norm_simple(lbl) for lbl in LABELS}
        known_norms = set(norm_label_map.values())

        # Build lines
        lines: Dict[int, List[dict]] = {}
        if any("line" in t for t in tokens):
            for t in tokens:
                try:
                    ln = int(t.get("line", 0))
                except Exception:
                    ln = 0
                lines.setdefault(ln, []).append(t)
        else:
            for t in tokens:
                yb = int(round(float(t.get("y0", 0.0)) / 6.0))
                lines.setdefault(yb, []).append(t)

        line_infos: List[Tuple[int, float, str, List[dict]]] = []
        for lid, toks in lines.items():
            toks_sorted = sorted(toks, key=lambda z: float(z.get("x0", 0.0)))
            joined = " ".join(str(z.get("text", "")) for z in toks_sorted)
            norm = _norm_simple(joined)
            yc = _line_center_y(toks_sorted)
            line_infos.append((lid, yc, norm, toks_sorted))
        line_infos.sort(key=lambda x: x[1])

        # Collect within a y band
        def collect_for_band(yc: float) -> Dict[str, List[str]]:
            band_min, band_max = yc - 3.0, yc + 3.0
            parts: Dict[str, List[Tuple[float, str]]] = {"transunion": [], "experian": [], "equifax": []}
            min_xL = min(v[0] for v in bureau_cols.values())
            for _, yline, _normtxt, toks in line_infos:
                if yline < band_min or yline > band_max:
                    continue
                for t in toks:
                    txt = str(t.get("text", "")).strip()
                    if not txt:
                        continue
                    mx = _token_mid_x(t)
                    if mx < min_xL:
                        continue
                    for bname, (xL, xR) in bureau_cols.items():
                        if xL <= mx <= xR:
                            parts[bname].append((mx, txt))
                            break
            out: Dict[str, List[str]] = {}
            for b in parts:
                out[b] = [t for _, t in sorted(parts[b], key=lambda p: p[0])]
            return out

        rows: List[Dict[str, Any]] = []
        for lbl in LABELS[:max_labels]:
            expected = norm_label_map[lbl]
            rowY = None
            for _, yc, normtxt, _ in line_infos:
                if normtxt == expected:
                    rowY = yc
                    break
            if rowY is None:
                continue
            parts = collect_for_band(rowY)
            rows.append(
                {
                    "label": lbl,
                    "y": rowY,
                    "tu": _nf_clean(_nf_join(parts.get("transunion", []))),
                    "ex": _nf_clean(_nf_join(parts.get("experian", []))),
                    "eq": _nf_clean(_nf_join(parts.get("equifax", []))),
                }
            )
        return rows
    except Exception:
        return []


__all__.append("build_debug_rows")
