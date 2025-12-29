from __future__ import annotations

from typing import Any, Dict, List, Tuple
from pathlib import Path
import json
import re

from .normalize_fields import (
    LABELS,
    LABEL_TO_KEY,
    CANONICAL_KEYS,
    ensure_all_keys,
    clean_value,
    is_effectively_blank,
)


def _mid_x(t: dict) -> float:
    try:
        x0 = float(t.get("x0", 0.0)); x1 = float(t.get("x1", x0))
        return (x0 + x1) / 2.0
    except Exception:
        return 0.0


def _mid_y(t: dict) -> float:
    try:
        y0 = float(t.get("y0", 0.0)); y1 = float(t.get("y1", y0))
        return (y0 + y1) / 2.0
    except Exception:
        return 0.0


def _norm_simple(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _group_lines(tokens: List[dict]) -> List[tuple[int, float, str, List[dict]]]:
    # Prefer provided 'line' ids; fallback to y-buckets
    lines: Dict[int, List[dict]] = {}
    if any("line" in t for t in tokens):
        for t in tokens:
            try:
                lid = int(t.get("line", 0))
            except Exception:
                lid = 0
            lines.setdefault(lid, []).append(t)
    else:
        for t in tokens:
            yb = int(round(float(t.get("y0", 0.0)) / 6.0))
            lines.setdefault(yb, []).append(t)

    infos: List[tuple[int, float, str, List[dict]]] = []
    for lid, toks in lines.items():
        toks_sorted = sorted(toks, key=lambda z: float(z.get("x0", 0.0)))
        yc = 0.0
        if toks_sorted:
            ys = sorted((_mid_y(t) for t in toks_sorted))
            yc = ys[len(ys) // 2]
        joined = " ".join(str(z.get("text", "")) for z in toks_sorted)
        norm = _norm_simple(joined)
        infos.append((lid, yc, norm, toks_sorted))
    infos.sort(key=lambda x: x[1])
    return infos


def _collect_row_values(
    tokens: List[dict],
    line_infos: List[tuple[int, float, str, List[dict]]],
    bureau_cols: Dict[str, Tuple[float, float]],
    rowY: float,
) -> Dict[str, Dict[str, Any]]:
    # Collect tokens for initial band and continuation within +6px (non-label lines)
    known_norms = {_norm_simple(lbl) for lbl in LABELS}
    min_xL = min(v[0] for v in bureau_cols.values()) if bureau_cols else -1e9

    def collect_band(yc: float) -> Dict[str, List[dict]]:
        band_min, band_max = yc - 3.0, yc + 3.0
        acc: Dict[str, List[dict]] = {"transunion": [], "experian": [], "equifax": []}
        for _, yline, _normtxt, toks in line_infos:
            if yline < band_min or yline > band_max:
                continue
            for t in toks:
                mx = _mid_x(t)
                my = _mid_y(t)
                if mx < min_xL or my < band_min or my > band_max:
                    continue
                for bname, (xL, xR) in bureau_cols.items():
                    if xL <= mx <= xR:
                        acc[bname].append(t)
                        break
        # sort within each bureau
        for b in acc:
            acc[b].sort(key=lambda z: float(z.get("x0", 0.0)))
        return acc

    parts0 = collect_band(rowY)
    parts = {b: list(v) for b, v in parts0.items()}
    # Continuations: next lines within (rowY+3, rowY+9] that are not a label line
    next_y = [y for _, y, normtxt, _ in line_infos if (y > rowY + 3.0 and y <= rowY + 9.0 and normtxt not in known_norms)]
    for ny in sorted(next_y):
        add = collect_band(ny)
        for b in parts:
            if add.get(b):
                # Separate bands by conceptual space (value joining will insert spaces)
                parts[b].append({"text": " ", "x0": 0, "y0": 0, "x1": 0, "y1": 0})
                parts[b].extend(add[b])

    # Build value + tokens map per bureau
    out: Dict[str, Dict[str, Any]] = {}
    for b in ("transunion", "experian", "equifax"):
        toks_b = [t for t in parts.get(b, []) if str(t.get("text", "")).strip()]
        val = clean_value(" ".join(str(t.get("text", "")).strip() for t in toks_b))
        out[b] = {
            "value": val,
            "tokens": [
                {"text": t.get("text"), "x0": t.get("x0"), "y0": t.get("y0"), "x1": t.get("x1"), "y1": t.get("y1")}
                for t in toks_b
            ],
        }
    return out


def package_account_block(block: dict, index_headline: str | None) -> dict:
    """Build an account package JSON purely from a block JSON and optional index headline.

    Does not rely on page numbers.
    """
    blk = dict(block or {})
    heading = blk.get("heading")
    fields = blk.get("fields") or {}
    # Ensure 22 keys per bureau
    out_fields: Dict[str, Dict[str, str]] = {}
    for b in ("transunion", "experian", "equifax"):
        out_fields[b] = ensure_all_keys(fields.get(b) or {})

    # Presence flags: any non-empty value
    presence = {
        b: any(
            not is_effectively_blank(v)
            for v in out_fields.get(b, {}).values()
        )
        for b in ("transunion", "experian", "equifax")
    }

    # Layout section (optional)
    layout_tokens: List[dict] = list(blk.get("layout_tokens") or [])
    debug_meta = (blk.get("meta") or {}).get("debug") or {}
    bureau_cols = debug_meta.get("bureau_cols") or {}

    layout_section: Dict[str, Any] | None = None
    if layout_tokens and bureau_cols:
        # Build window
        win = debug_meta.get("layout_window") or {}
        if not win:
            # derive coarse window from tokens if debug window missing
            xs0 = [float(t.get("x0", 0.0)) for t in layout_tokens]
            xs1 = [float(t.get("x1", 0.0)) for t in layout_tokens]
            ys = [(_mid_y(t)) for t in layout_tokens]
            win = {
                "x_min": min(xs0) if xs0 else 0.0,
                "x_max": max(xs1) if xs1 else 0.0,
                "y_top": min(ys) if ys else 0.0,
                "y_bottom": max(ys) if ys else 0.0,
            }

        # Build per-label rows with values+tokens
        line_infos = _group_lines(layout_tokens)
        rows: List[Dict[str, Any]] = []
        norm_label_map = {lbl: _norm_simple(lbl) for lbl in LABELS}
        for lbl in LABELS:
            expected = norm_label_map[lbl]
            rowY = None
            for _, yc, normtxt, _ in line_infos:
                if normtxt == expected:
                    rowY = yc
                    break
            if rowY is None:
                rows.append({
                    "label": lbl,
                    "row_y": None,
                    "transunion": {"value": "", "tokens": []},
                    "experian": {"value": "", "tokens": []},
                    "equifax": {"value": "", "tokens": []},
                })
                continue
            values = _collect_row_values(layout_tokens, line_infos, bureau_cols, rowY)
            rows.append({
                "label": lbl,
                "row_y": rowY,
                "transunion": values.get("transunion") or {"value": "", "tokens": []},
                "experian": values.get("experian") or {"value": "", "tokens": []},
                "equifax": values.get("equifax") or {"value": "", "tokens": []},
            })

        layout_section = {
            "window": {k: win.get(k) for k in ("y_top", "y_bottom", "x_min", "x_max")},
            "bureau_cols": bureau_cols,
            "labels": rows,
        }

    # block id and filename may be inferred outside; here keep placeholders if missing
    block_id = blk.get("meta", {}).get("block_id") or blk.get("block_id")
    block_filename = blk.get("meta", {}).get("block_filename") or blk.get("block_filename")

    pkg: Dict[str, Any] = {
        "block_id": block_id,
        "block_filename": block_filename,
        "block_heading": heading,
        "index_headline": index_headline,
        "presence": presence,
        "fields": out_fields,
    }
    if layout_section is not None:
        pkg["layout"] = layout_section

    return pkg


def _slugify(s: str | None) -> str | None:
    if not s:
        return None
    t = s.lower()
    t = re.sub(r"[^a-z0-9]+", "-", t)
    t = re.sub(r"-+", "-", t).strip("-")
    return t or None


def write_account_package(session_id: str, block_id: int, pkg: dict, index_slug: str | None) -> str:
    base = Path("traces") / "blocks" / session_id / "accounts"
    base.mkdir(parents=True, exist_ok=True)
    slug = index_slug or _slugify(pkg.get("index_headline")) or None
    fname = f"account_{block_id:02d}{('__' + slug) if slug else ''}.json"
    path = base / fname
    path.write_text(json.dumps(pkg, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(path)


__all__ = ["package_account_block", "write_account_package"]

