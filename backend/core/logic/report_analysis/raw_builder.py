from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from backend.pipeline.runs import RunManifest
from typing import Any, Dict, List

from backend.core.logic.report_analysis.account_packager_coords import (
    package_block_raw_coords,
    write_block_raw_coords,
    _slug,
)
from backend.core.logic.report_analysis.column_reader import (
    detect_bureau_columns,
)
from backend.core.logic.report_analysis.normalize_fields import LABELS as _SC_LABELS


logger = logging.getLogger(__name__)
RAW_DEBUG = os.getenv("RAW_DEBUG", "0") == "1"

# Export mode flags for RAW builder
RAW_OUTPUT_MODE = os.getenv("RAW_OUTPUT_MODE", "raw_coords")  # values: raw_coords | grid_table
GRID_DY = float(os.getenv("GRID_DY", "3.5"))  # row merge sensitivity by Y
GRID_DX = float(os.getenv("GRID_DX", "10.0"))  # token gap threshold by X for spacing
GRID_DEBUG = os.getenv("GRID_DEBUG", "0") == "1"
GRID_MODE = os.getenv("GRID_MODE", "generic")  # values: generic | smartcredit_4col
GRID_LABEL_MARGIN = float(os.getenv("GRID_LABEL_MARGIN", "8.0"))  # left margin before TU band


def _mid(a: Any, b: Any) -> float:
    try:
        return (float(a) + float(b)) / 2.0
    except Exception:
        return 0.0


def build_raw_from_windows(sid: str, root: Path) -> None:
    """
    Read layout_snapshot.json + block_windows.json under runs/<sid>/traces/blocks/,
    build one RAW JSON per block into accounts_raw/, and write _raw_index.json.
    """
    m = RunManifest.for_sid(sid)
    base = m.ensure_run_subdir("traces_blocks_dir", "traces/blocks")
    # Guardrail: ensure canonical runs path
    low = str(base.resolve()).lower()
    assert ("/runs/" in low) or ("\\runs\\" in low), "RAW builder base must live under runs/<SID>"
    layout_path = base / "layout_snapshot.json"
    windows_path = base / "block_windows.json"

    if not layout_path.exists() or not windows_path.exists():
        raise FileNotFoundError("layout_snapshot.json or block_windows.json missing")

    try:
        layout = json.loads(layout_path.read_text(encoding="utf-8"))
    except Exception as e:
        raise RuntimeError(f"failed to read layout_snapshot.json: {e}")
    try:
        windows = json.loads(windows_path.read_text(encoding="utf-8"))
    except Exception as e:
        raise RuntimeError(f"failed to read block_windows.json: {e}")

    pages_list: List[dict] = list(layout.get("pages") or [])
    pages: Dict[int, dict] = {}
    for idx, p in enumerate(pages_list, start=1):
        try:
            num = int(p.get("number", idx) or idx)
        except Exception:
            num = idx
        pages[num] = p

    raw_entries: List[Dict[str, Any]] = []

    # Log start summary
    try:
        logger.info(
            "RAW_BUILDER: start sid=%s pages=%d blocks=%d",
            sid,
            len(pages),
            len(windows.get("blocks") or []),
        )
    except Exception:
        pass

    for row in list(windows.get("blocks") or []):
        try:
            block_id = int(row.get("block_id", 0) or 0)
        except Exception:
            block_id = 0
        heading = row.get("heading")
        index_headline = row.get("index_headline")
        window = row.get("window") or None

        if not block_id or not window:
            logger.info("RAW_BUILDER: block_id=%s no window/page; skipping", block_id)
            continue

        try:
            page_no = int(window.get("page", 0) or 0)
        except Exception:
            page_no = 0
        page = pages.get(page_no)
        if not page:
            logger.info("RAW_BUILDER: block_id=%s no window/page; skipping", block_id)
            continue

        toks_all = list(page.get("tokens") or [])
        try:
            x_min = float(window.get("x_min", 0.0) or 0.0)
            x_max = float(window.get("x_max", 0.0) or 0.0)
            y_top = float(window.get("y_top", 0.0) or 0.0)
            y_bottom = float(window.get("y_bottom", 0.0) or 0.0)
        except Exception:
            logger.info("RAW_BUILDER: block_id=%s window parse failed; skipping", block_id)
            continue

        # Filter tokens by midpoint within the window (no Y flip)
        ltoks: List[dict] = []
        for t in toks_all:
            try:
                mx = _mid(t.get("x0", 0.0), t.get("x1", 0.0))
                my = _mid(t.get("y0", 0.0), t.get("y1", 0.0))
            except Exception:
                continue
            if (x_min <= mx <= x_max) and (y_top <= my <= y_bottom):
                ltoks.append(t)

        # Stable reading-order sort: top->bottom, then left->right
        try:
            ltoks.sort(
                key=lambda tt: (
                    _mid(tt.get("y0", 0.0), tt.get("y1", 0.0)),
                    _mid(tt.get("x0", 0.0), tt.get("x1", 0.0)),
                )
            )
        except Exception:
            pass

        # Optional: short sample of first 3 token texts when RAW_DEBUG=1
        try:
            if os.getenv("RAW_DEBUG") == "1" and ltoks:
                logger.debug(
                    "RAW_BUILDER: sample id=%s tokens=%s",
                    block_id,
                    [t.get("text") for t in ltoks[:3]],
                )
        except Exception:
            pass

        if not ltoks:
            logger.info("RAW_BUILDER: block_id=%s no tokens in window; skipping", block_id)
            continue

        # Detect bureau columns from header tokens found inside the window
        header_tokens = [
            t for t in ltoks if str(t.get("text", "")).strip().lower() in ("transunion", "experian", "equifax")
        ]
        bureau_cols = detect_bureau_columns(header_tokens) if header_tokens else {}

        blk_for_pkg = {
            "block_id": block_id,
            "heading": heading,
            "index_headline": index_headline,
            "layout_tokens": ltoks,
            "meta": {"debug": {"layout_window": window}},
        }

        if RAW_OUTPUT_MODE == "grid_table":
            # Write geometric table to accounts_table/ and skip raw_coords path
            if GRID_MODE == "smartcredit_4col":
                _write_grid_smartcredit(
                    sid,
                    block_id,
                    heading,
                    index_headline,
                    window,
                    ltoks,
                    base,
                )
            else:
                _write_grid_table_generic(
                    sid,
                    block_id,
                    heading,
                    index_headline,
                    window,
                    ltoks,
                    base,
                )
            continue
        else:
            pkg = package_block_raw_coords(blk_for_pkg, bureau_cols if bureau_cols else {})
            out_path = write_block_raw_coords(sid, block_id, pkg, index_headline)

            stats = pkg.get("stats") or {}
            raw_entries.append(
                {
                    "block_id": block_id,
                    "heading": heading,
                    "index_headline": index_headline,
                    "raw_coords_path": out_path,
                    "row_count": int(stats.get("row_count", 0) or 0),
                    "token_count": int(stats.get("token_count", 0) or 0),
                }
            )

    # Write session-level RAW index (only in raw_coords mode)
    if RAW_OUTPUT_MODE != "grid_table":
        raw_dir = base / "accounts_raw"
        raw_dir.mkdir(parents=True, exist_ok=True)
        raw_index = {"session_id": sid, "blocks": raw_entries}
        (raw_dir / "_raw_index.json").write_text(
            json.dumps(raw_index, ensure_ascii=False, indent=2), encoding="utf-8"
        )


def build_raw_from_snapshot_and_windows(session_id: str) -> str:
    """Compatibility wrapper that builds RAW from snapshot+windows using CWD root.

    Returns the path to the written _raw_index.json as a string.
    """
    # Canonical path via manifest
    m = RunManifest.for_sid(session_id)
    base = m.ensure_run_subdir("traces_blocks_dir", "traces/blocks")
    # Reuse the main implementation
    build_raw_from_windows(session_id, Path("."))
    return str(base / "accounts_raw" / "_raw_index.json")


__all__ = ["build_raw_from_windows", "build_raw_from_snapshot_and_windows"]


def _row_cluster(tokens: List[dict], dy: float) -> List[List[dict]]:
    """Group tokens to visual rows by mid-Y distance (reading order Y->X)."""
    if not tokens:
        return []
    tokens = sorted(
        tokens,
        key=lambda t: (
            _mid(t.get("y0", 0.0), t.get("y1", 0.0)),
            _mid(t.get("x0", 0.0), t.get("x1", 0.0)),
        ),
    )
    rows: List[List[dict]] = []
    cur: List[dict] = []
    cur_y: float | None = None
    for t in tokens:
        my = _mid(t.get("y0", 0.0), t.get("y1", 0.0))
        if cur_y is None or abs(my - cur_y) <= dy:
            cur.append(t)
            ys = [
                _mid(x.get("y0", 0.0), x.get("y1", 0.0))
                for x in cur
            ]
            ys.sort()
            cur_y = ys[len(ys) // 2]
        else:
            rows.append(cur)
            cur = [t]
            cur_y = my
    if cur:
        rows.append(cur)
    return rows


def _infer_columns(all_tokens: List[dict], dx: float) -> List[float]:
    """Infer approximate column center X from gaps across all tokens (block-level)."""
    xs = sorted([
        _mid(t.get("x0", 0.0), t.get("x1", 0.0))
        for t in (all_tokens or [])
    ])
    if not xs:
        return []
    centers: List[float] = []
    run: List[float] = [xs[0]]
    for x in xs[1:]:
        if abs(x - run[-1]) > dx:
            centers.append(sum(run) / len(run))
            run = [x]
        else:
            run.append(x)
    if run:
        centers.append(sum(run) / len(run))
    return centers


def _closest_column(x: float, centers: List[float]) -> int:
    if not centers:
        return 0
    best_i, best_d = 0, float("inf")
    for i, c in enumerate(centers):
        d = abs(x - c)
        if d < best_d:
            best_i, best_d = i, d
    return best_i


_SYM_ONLY = re.compile(r"^[^\w]+$", re.UNICODE)


def _join_text_ltr(tokens: List[dict]) -> str:
    """Join tokens left-to-right with light gap spacing; drop symbol-only tokens."""
    toks = sorted(tokens or [], key=lambda t: _mid(t.get("x0", 0.0), t.get("x1", 0.0)))
    out: List[str] = []
    prev_x1: float | None = None
    prev_txt: str | None = None
    avg_char = 5.0
    for t in toks:
        txt = str(t.get("text", "")).strip()
        if not txt:
            continue
        # Keep hyphen tokens; skip other symbol-only tokens
        if txt != '-' and _SYM_ONLY.match(txt):
            continue
        try:
            x0 = float(t.get("x0", 0.0)); x1 = float(t.get("x1", x0))
        except Exception:
            x0, x1 = 0.0, 0.0
        # Special handling for standalone hyphen to ensure natural spacing: "Bank - Mortgage"
        if txt == '-':
            if out and not out[-1].endswith(" "):
                out.append(" ")
            out.append("-")
            out.append(" ")
            prev_x1 = x1
            prev_txt = txt
            continue
        # Add a space when the gap between adjacent tokens is sufficiently large.
        # Slightly increase the threshold to reduce accidental over-spacing.
        if prev_x1 is not None:
            gap = x0 - prev_x1
            if gap > 1.2 * avg_char:
                out.append(" ")
        out.append(txt)
        prev_x1 = x1
        prev_txt = txt
    return "".join(out).strip()


def _write_grid_table_generic(
    sid: str,
    block_id: int,
    heading: str | None,
    index_headline: str | None,
    window: Dict[str, Any],
    ltoks: List[dict],
    base_dir: Path,
) -> None:
    rows = _row_cluster(ltoks, GRID_DY)
    col_centers = _infer_columns(ltoks, GRID_DX)
    table_rows: List[Dict[str, Any]] = []
    for r in rows:
        cells_tok: Dict[int, List[dict]] = {}
        for t in r:
            mx = _mid(t.get("x0", 0.0), t.get("x1", 0.0))
            ci = _closest_column(mx, col_centers) if col_centers else 0
            cells_tok.setdefault(ci, []).append(t)
        max_ci = max(cells_tok.keys(), default=-1)
        cells: List[str] = []
        for ci in range(max_ci + 1):
            cells.append(_join_text_ltr(cells_tok.get(ci, [])))
        row_y = float(sum(_mid(t.get("y0", 0.0), t.get("y1", 0.0)) for t in r) / len(r)) if r else 0.0
        entry: Dict[str, Any] = {"y": row_y, "cells": cells}
        if GRID_DEBUG:
            entry["debug_tokens"] = r
        table_rows.append(entry)

    out_dir = base_dir / "accounts_table"
    out_dir.mkdir(parents=True, exist_ok=True)
    slug = _slug(heading or f"block-{block_id}")
    out_path = out_dir / f"account_table_{block_id:02d}__{slug}.json"
    obj: Dict[str, Any] = {
        "session_id": sid,
        "block_id": block_id,
        "block_heading": heading,
        "index_headline": index_headline,
        "mode": "grid_table",
        "layout_window": window,
        "rows": table_rows,
        "meta": {"columns": col_centers},
    }
    if GRID_DEBUG:
        obj["debug"] = {"token_count": len(ltoks)}
    out_path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")

    # Update per-session table index
    idx_path = out_dir / "_table_index.json"
    idx: Dict[str, Any] = {"session_id": sid, "blocks": []}
    if idx_path.exists():
        try:
            idx = json.loads(idx_path.read_text(encoding="utf-8"))
        except Exception:
            pass
    idx["blocks"] = [b for b in (idx.get("blocks") or []) if int((b or {}).get("block_id", -1)) != block_id]
    idx["blocks"].append(
        {
            "block_id": block_id,
            "heading": heading,
            "index_headline": index_headline,
            "table_path": str(out_path),
            "row_count": len(table_rows),
            "column_count": len(col_centers) if col_centers else 1,
        }
    )
    idx_path.write_text(json.dumps(idx, ensure_ascii=False, indent=2), encoding="utf-8")
    try:
        logger.info("GRID_TABLE(generic): sid=%s blocks=%d", sid, len(idx.get("blocks") or []))
    except Exception:
        pass


def _write_grid_smartcredit(
    sid: str,
    block_id: int,
    heading: str | None,
    index_headline: str | None,
    window: Dict[str, Any],
    ltoks: List[dict],
    base_dir: Path,
 ) -> None:
    """Write a 4-column SmartCredit table with fields: label, tu, ex, eq.

    - Detect TU/EX/EQ bands; if missing, fall back to generic grid writer.
    - Label column = tokens left of min(tu, ex, eq).xL - GRID_LABEL_MARGIN.
    - Rows clustered by Y using GRID_DY; cell text joined LTR with spacing.
    - Attempts to normalize label to a known SmartCredit label from catalog.
    """
    # Detect bands from header tokens in the window
    header_tokens = [
        t for t in ltoks if str(t.get("text", "")).strip().lower() in ("transunion", "experian", "equifax")
    ]
    bands_raw = detect_bureau_columns(header_tokens) if header_tokens else {}
    if not bands_raw:
        _write_grid_table_generic(sid, block_id, heading, index_headline, window, ltoks, base_dir)
        return

    def _pair(v: tuple[float, float] | list[float] | None) -> tuple[float, float] | None:
        if not v:
            return None
        return (float(v[0]), float(v[1]))

    tu_band = _pair(bands_raw.get("transunion"))
    ex_band = _pair(bands_raw.get("experian"))
    eq_band = _pair(bands_raw.get("equifax"))
    if not (tu_band and ex_band and eq_band):
        _write_grid_table_generic(sid, block_id, heading, index_headline, window, ltoks, base_dir)
        return

    label_max_x = min(tu_band[0], ex_band[0], eq_band[0]) - GRID_LABEL_MARGIN

    # Simple label normalization against catalog
    import re as _re

    def _norm_label(s: str) -> str:
        s = (s or "").lower()
        s = _re.sub(r"[^a-z0-9]+", " ", s)
        return _re.sub(r"\s+", " ", s).strip()

    def _lev(a: str, b: str) -> int:
        la, lb = len(a), len(b)
        if la == 0:
            return lb
        if lb == 0:
            return la
        dp = list(range(lb + 1))
        for i in range(1, la + 1):
            prev = dp[0]
            dp[0] = i
            ca = a[i - 1]
            for j in range(1, lb + 1):
                tmp = dp[j]
                cb = b[j - 1]
                cost = 0 if ca == cb else 1
                dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + cost)
                prev = tmp
        return dp[lb]

    cat_norm = [(lbl, _norm_label(lbl)) for lbl in _SC_LABELS]
    _SYM_ONLY_STR = _re.compile(r"^[^A-Za-z0-9]+$")

    def _best_label_match(label: str) -> str | None:
        n = _norm_label(label)
        if not n:
            return None
        for lbl, nn in cat_norm:
            if n == nn or n.startswith(nn) or nn.startswith(n):
                return lbl
        best = (None, 1_000)
        for lbl, nn in cat_norm:
            d = _lev(n, nn)
            if d < best[1]:
                best = (lbl, d)
        return best[0] if best[1] <= 3 else None

    rows = _row_cluster(ltoks, GRID_DY)

    def _assign_band_token(t: dict) -> str | None:
        mx = _mid(t.get("x0", 0.0), t.get("x1", 0.0))
        if tu_band[0] <= mx <= tu_band[1]:
            return "tu"
        if ex_band[0] <= mx <= ex_band[1]:
            return "ex"
        if eq_band[0] <= mx <= eq_band[1]:
            return "eq"
        return None

    out_rows: List[Dict[str, Any]] = []
    for r in rows:
        label_tokens: List[dict] = []
        tu_tokens: List[dict] = []
        ex_tokens: List[dict] = []
        eq_tokens: List[dict] = []

        for t in r:
            mx = _mid(t.get("x0", 0.0), t.get("x1", 0.0))
            if mx <= label_max_x:
                label_tokens.append(t)
            else:
                band = _assign_band_token(t)
                if band == "tu":
                    tu_tokens.append(t)
                elif band == "ex":
                    ex_tokens.append(t)
                elif band == "eq":
                    eq_tokens.append(t)

        label_text = _join_text_ltr(label_tokens)
        tu_text = _join_text_ltr(tu_tokens)
        ex_text = _join_text_ltr(ex_tokens)
        eq_text = _join_text_ltr(eq_tokens)

        # Skip header row (only bureau names, no label)
        bureau_set = {tu_text.lower(), ex_text.lower(), eq_text.lower()}
        if not label_text and bureau_set == {"transunion", "experian", "equifax"}:
            continue

        label_final = _best_label_match(label_text) or label_text

        def _empty_or_symbol(s: str) -> bool:
            s2 = (s or "").strip()
            return (not s2) or (_SYM_ONLY_STR.match(s2) is not None)

        # Skip fully-empty or symbol-only rows
        if all(_empty_or_symbol(x) for x in (label_final, tu_text, ex_text, eq_text)):
            continue

        row_y = float(sum(_mid(t.get("y0", 0.0), t.get("y1", 0.0)) for t in r) / len(r)) if r else 0.0
        entry: Dict[str, Any] = {
            "y": row_y,
            "label": label_final,
            "tu": tu_text,
            "ex": ex_text,
            "eq": eq_text,
        }
        if GRID_DEBUG:
            entry["debug_tokens"] = r
        out_rows.append(entry)

    out_dir = base_dir / "accounts_table"
    out_dir.mkdir(parents=True, exist_ok=True)
    slug = _slug(heading or f"block-{block_id}")
    out_path = out_dir / f"account_table_{block_id:02d}__{slug}.json"
    meta_bands = {
        "tu": [tu_band[0], tu_band[1]],
        "ex": [ex_band[0], ex_band[1]],
        "eq": [eq_band[0], eq_band[1]],
    }
    # Minimal, clean schema for SmartCredit 4-col output (no cell arrays, no per-token coords)
    obj: Dict[str, Any] = {
        "session_id": sid,
        "block_id": block_id,
        "block_heading": heading,
        "mode": "grid_table",
        "rows": out_rows,
        "meta": {"bands": meta_bands, "label_max_x": float(label_max_x)},
    }
    if GRID_DEBUG:
        obj["debug"] = {"token_count": len(ltoks)}
    out_path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")

    idx_path = out_dir / "_table_index.json"
    idx: Dict[str, Any] = {"session_id": sid, "blocks": []}
    if idx_path.exists():
        try:
            idx = json.loads(idx_path.read_text(encoding="utf-8"))
        except Exception:
            pass
    idx["blocks"] = [b for b in (idx.get("blocks") or []) if int((b or {}).get("block_id", -1)) != block_id]
    idx["blocks"].append(
        {
            "block_id": block_id,
            "heading": heading,
            "index_headline": index_headline,
            "table_path": str(out_path),
            "row_count": len(out_rows),
            "column_count": 4,
        }
    )
    idx_path.write_text(json.dumps(idx, ensure_ascii=False, indent=2), encoding="utf-8")
    try:
        logger.info("GRID_TABLE(smartcredit_4col): sid=%s blocks=%d", sid, len(idx.get("blocks") or []))
    except Exception:
        pass
        row_y = float(sum(_mid(t.get("y0", 0.0), t.get("y1", 0.0)) for t in r) / len(r)) if r else 0.0
        entry: Dict[str, Any] = {"y": row_y, "cells": cells}
        if GRID_DEBUG:
            entry["debug_tokens"] = r
        table_rows.append(entry)

    out_dir = base_dir / "accounts_table"
    out_dir.mkdir(parents=True, exist_ok=True)
    slug = (heading or f"block-{block_id}").lower().replace(" ", "-")
    out_path = out_dir / f"account_table_{block_id:02d}__{slug}.json"
    obj: Dict[str, Any] = {
        "session_id": sid,
        "block_id": block_id,
        "block_heading": heading,
        "index_headline": index_headline,
        "mode": "grid_table",
        "layout_window": window,
        "rows": table_rows,
        "meta": {"columns": ["label", "transunion", "experian", "equifax"]},
    }
    if GRID_DEBUG:
        obj["debug"] = {"token_count": len(ltoks)}
    out_path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")

    # Update per-session table index
    idx_path = out_dir / "_table_index.json"
    idx: Dict[str, Any] = {"session_id": sid, "blocks": []}
    if idx_path.exists():
        try:
            idx = json.loads(idx_path.read_text(encoding="utf-8"))
        except Exception:
            pass
    idx["blocks"] = [b for b in (idx.get("blocks") or []) if int((b or {}).get("block_id", -1)) != block_id]
    idx["blocks"].append(
        {
            "block_id": block_id,
            "heading": heading,
            "index_headline": index_headline,
            "table_path": str(out_path),
            "row_count": len(table_rows),
            "column_count": 4,
        }
    )
    idx_path.write_text(json.dumps(idx, ensure_ascii=False, indent=2), encoding="utf-8")
    try:
        logger.info("GRID_TABLE(smartcredit_4col): sid=%s blocks=%d", sid, len(idx.get("blocks") or []))
    except Exception:
        pass
