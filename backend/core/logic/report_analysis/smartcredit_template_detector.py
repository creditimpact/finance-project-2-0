from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

from .normalize_fields import LABELS as _CANON_LABELS
from .column_reader import detect_bureau_columns as _detect_bureau_columns


logger = logging.getLogger(__name__)


def _mid(a: Any, b: Any) -> float:
    try:
        return (float(a) + float(b)) / 2.0
    except Exception:
        return 0.0


def _norm_simple(s: str) -> str:
    import re

    s = (s or "").lower()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _load_stageA(session_id: str, root: Path) -> Tuple[dict, dict] | None:
    base = root / "traces" / "blocks" / session_id
    layout_path = base / "layout_snapshot.json"
    windows_path = base / "block_windows.json"
    try:
        layout = json.loads(layout_path.read_text(encoding="utf-8"))
        windows = json.loads(windows_path.read_text(encoding="utf-8"))
        return layout, windows
    except FileNotFoundError:
        logger.warning(
            "TEMPLATE: detector missing Stage-A artifacts (sid=%s)",
            session_id,
        )
        return None
    except Exception as exc:
        logger.warning(
            "TEMPLATE: detector failed to read Stage-A (sid=%s): %s",
            session_id,
            str(exc),
        )
        return None


def _slice_tokens_for_window(page: dict, window: dict) -> List[dict]:
    toks_all = list(page.get("tokens") or [])
    try:
        x_min = float(window.get("x_min", 0.0) or 0.0)
        x_max = float(window.get("x_max", 0.0) or 0.0)
        y_top = float(window.get("y_top", 0.0) or 0.0)
        y_bottom = float(window.get("y_bottom", 0.0) or 0.0)
    except Exception:
        return []

    ltoks: List[dict] = []
    for t in toks_all:
        try:
            mx = _mid(t.get("x0", 0.0), t.get("x1", 0.0))
            my = _mid(t.get("y0", 0.0), t.get("y1", 0.0))
        except Exception:
            continue
        if (x_min <= mx <= x_max) and (y_top <= my <= y_bottom):
            ltoks.append(t)
    try:
        ltoks.sort(
            key=lambda tt: (
                _mid(tt.get("y0", 0.0), tt.get("y1", 0.0)),
                _mid(tt.get("x0", 0.0), tt.get("x1", 0.0)),
            )
        )
    except Exception:
        pass
    return ltoks


def _group_lines(tokens: List[dict]) -> List[Tuple[int, float, str]]:
    """Return list of (line_id, centerY, norm_text)."""
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
            try:
                yb = int(round(float(t.get("y0", 0.0)) / 6.0))
            except Exception:
                yb = 0
            lines.setdefault(yb, []).append(t)

    out: List[Tuple[int, float, str]] = []
    for lid, toks in lines.items():
        toks_sorted = sorted(toks, key=lambda z: float(z.get("x0", 0.0)))
        joined = " ".join(str(z.get("text", "")) for z in toks_sorted)
        # center Y: median of y midpoints
        ys: List[float] = []
        for z in toks_sorted:
            try:
                y0 = float(z.get("y0", 0.0))
                y1 = float(z.get("y1", y0))
            except Exception:
                continue
            ys.append((y0 + y1) / 2.0)
        yc = sorted(ys)[len(ys) // 2] if ys else 0.0
        out.append((lid, yc, _norm_simple(joined)))
    out.sort(key=lambda p: p[1])
    return out


def detect(session_id: str, root: Path | None = None) -> dict:
    """
    Returns: {"confidence": float, "blocks_checked": int, "blocks_with_bands": int, "label_coverage_mean": float}
    """

    try:
        logger.info("Template using Stage-A artifacts only (sid=%s)", session_id)
        base_root = root or Path.cwd()
        loaded = _load_stageA(session_id, base_root)
        if not loaded:
            return {
                "confidence": 0.0,
                "blocks_checked": 0,
                "blocks_with_bands": 0,
                "label_coverage_mean": 0.0,
                "reason": "missing_artifacts",
            }
        layout, windows = loaded

        # page number -> page dict
        pages: Dict[int, dict] = {}
        for idx, p in enumerate(list(layout.get("pages") or []), start=1):
            try:
                num = int(p.get("number", idx) or idx)
            except Exception:
                num = idx
            pages[num] = p

        # prepare normalized label set
        label_norms = {_norm_simple(lbl) for lbl in _CANON_LABELS}

        blocks_checked = 0
        blocks_with_bands = 0
        coverage_values: List[float] = []

        for row in list(windows.get("blocks") or []):
            window = row.get("window") or None
            if not window:
                continue
            try:
                page_no = int(window.get("page", 0) or 0)
            except Exception:
                page_no = 0
            page = pages.get(page_no)
            if not page:
                continue

            tokens = _slice_tokens_for_window(page, window)
            if not tokens:
                continue

            blocks_checked += 1

            # Header bands detection (TU/EX/EQ)
            header_tokens = [
                t
                for t in tokens
                if str(t.get("text", "")).strip().lower() in ("transunion", "experian", "equifax")
            ]
            bands = _detect_bureau_columns(header_tokens) if header_tokens else {}
            if len(bands) >= 2:
                blocks_with_bands += 1

            # Label coverage per block
            matched: set[str] = set()
            for _lid, _yc, normtxt in _group_lines(tokens):
                if normtxt in label_norms:
                    matched.add(normtxt)
            coverage = (len(matched) / max(1, len(_CANON_LABELS)))
            coverage_values.append(coverage)

        label_cov_mean = (sum(coverage_values) / len(coverage_values)) if coverage_values else 0.0
        band_ratio = (blocks_with_bands / blocks_checked) if blocks_checked else 0.0
        confidence = 0.5 * band_ratio + 0.5 * label_cov_mean

        logger.info(
            "TEMPLATE: detector blocks=%d bands_blocks=%d label_cov_mean=%.3f conf=%.3f",
            blocks_checked,
            blocks_with_bands,
            label_cov_mean,
            confidence,
        )

        return {
            "confidence": float(confidence),
            "blocks_checked": int(blocks_checked),
            "blocks_with_bands": int(blocks_with_bands),
            "label_coverage_mean": float(label_cov_mean),
        }
    except Exception as exc:
        logger.warning("TEMPLATE: detector unexpected failure sid=%s: %s", session_id, str(exc), exc_info=True)
        return {
            "confidence": 0.0,
            "blocks_checked": 0,
            "blocks_with_bands": 0,
            "label_coverage_mean": 0.0,
            "reason": "unexpected_error",
        }


__all__ = ["detect"]
