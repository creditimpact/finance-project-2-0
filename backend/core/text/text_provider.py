from __future__ import annotations

import io
import logging
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .layout_models import Page, Token


logger = logging.getLogger(__name__)
BLOCK_DEBUG = os.getenv("BLOCK_DEBUG", "0") == "1"


def _norm_header(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"[®\u00AE]", "", s)
    s = re.sub(r"\s+", " ", s)
    return s.upper()


HEADER_TOKENS = {"TRANSUNION", "EXPERIAN", "EQUIFAX"}


def _detect_columns_from_headers(tokens: List[Token], page_width: float) -> List[Tuple[float, float]]:
    heads: List[Tuple[float, float]] = []  # (x_center, raw_x0)
    for t in tokens:
        txt = _norm_header(t["text"]).replace("AR", "")  # tolerate stray suffix
        if txt in HEADER_TOKENS:
            x_center = (float(t["x0"]) + float(t["x1"])) / 2.0
            heads.append((x_center, float(t["x0"])) )
    heads.sort(key=lambda x: x[0])
    if len(heads) < 2:
        return []
    # Build 3 ranges by midpoints between header centers; if only 2 found, make 2 ranges
    xcs = [h[0] for h in heads]
    ranges: List[Tuple[float, float]] = []
    # For 3, compute boundaries: [0, mid(0-1)], [mid(0-1), mid(1-2)], [mid(1-2), width]
    if len(xcs) >= 3:
        m01 = (xcs[0] + xcs[1]) / 2.0
        m12 = (xcs[1] + xcs[2]) / 2.0
        ranges = [(0.0, m01), (m01, m12), (m12, float(page_width))]
    else:
        m01 = (xcs[0] + xcs[1]) / 2.0
        ranges = [(0.0, m01), (m01, float(page_width))]
    return ranges


def _assign_columns(tokens: List[Token], col_ranges: List[Tuple[float, float]]) -> None:
    for t in tokens:
        x_center = (float(t["x0"]) + float(t["x1"])) / 2.0
        assigned: Optional[int] = None
        for idx, (x0, x1) in enumerate(col_ranges):
            if x0 <= x_center < x1:
                assigned = idx
                break
        t["col"] = assigned


def _group_lines(tokens: List[Token]) -> None:
    # Sort by y, then x, then assign incremental line numbers when y jumps
    tokens.sort(key=lambda t: (float(t["y0"]), float(t["x0"])) )
    line_no = 0
    prev_y = None
    y_thresh = 4.0
    for t in tokens:
        y = float(t["y0"])  # top
        if prev_y is None or abs(y - prev_y) > y_thresh:
            line_no += 1
            prev_y = y
        t["line"] = line_no


def _extract_pdf_tokens(pdf_path: str) -> Tuple[List[Page], str]:
    import fitz  # type: ignore

    pages: List[Page] = []
    full_lines: List[str] = []
    with fitz.open(pdf_path) as doc:
        for i, page in enumerate(doc, start=1):
            width = int(page.rect.width)
            height = int(page.rect.height)
            tokens: List[Token] = []
            try:
                words = page.get_text("words")  # list of (x0,y0,x1,y1,word,block,line,word_no)
            except Exception:
                words = []
            for w in words or []:
                # Handle both tuple and dict forms
                if isinstance(w, (list, tuple)) and len(w) >= 5:
                    x0, y0, x1, y1, text = w[:5]
                elif isinstance(w, dict):
                    x0, y0, x1, y1, text = w.get("x0", 0), w.get("y0", 0), w.get("x1", 0), w.get("y1", 0), w.get("text", "")
                else:
                    continue
                if not str(text).strip():
                    continue
                tokens.append(Token(text=str(text), x0=float(x0), y0=float(y0), x1=float(x1), y1=float(y1), line=0, col=None))
            # Fallback to lines if words empty
            if not tokens:
                try:
                    text_dict = page.get_text("dict")
                    for b in text_dict.get("blocks", []):
                        for l in b.get("lines", []):
                            for s in l.get("spans", []):
                                text = s.get("text", "").strip()
                                if not text:
                                    continue
                                x0, y0 = s.get("bbox", [0, 0, 0, 0])[0:2]
                                x1, y1 = s.get("bbox", [0, 0, 0, 0])[2:4]
                                tokens.append(Token(text=text, x0=float(x0), y0=float(y0), x1=float(x1), y1=float(y1), line=0, col=None))
                except Exception:
                    pass

            _group_lines(tokens)
            # Detect columns using headers on this page
            col_ranges = _detect_columns_from_headers(tokens, width)
            if col_ranges:
                if BLOCK_DEBUG:
                    logger.info("TP1: page=%d columns detected x=%s", i, col_ranges)
                _assign_columns(tokens, col_ranges)
            else:
                for t in tokens:
                    t["col"] = None

            # Build full_text in reading order
            lines: Dict[int, List[str]] = {}
            for t in tokens:
                lines.setdefault(t["line"], []).append(t["text"])
            ordered_lines = [" ".join(lines[k]) for k in sorted(lines.keys())]
            full_lines.extend(ordered_lines)

            pages.append(Page(number=i, width=width, height=height, tokens=tokens))
    return pages, "\n".join(full_lines)


def _extract_ocr_tokens(pdf_path: str) -> Tuple[List[Page], str]:
    # Best-effort OCR path using tesseract TSV if available; otherwise returns empty
    try:
        import fitz  # type: ignore
    except Exception:
        return [], ""

    if not shutil.which("tesseract"):
        logger.warning("TP1: tesseract not found; OCR layout unavailable")
        return [], ""

    pages: List[Page] = []
    full_lines: List[str] = []
    with fitz.open(pdf_path) as doc:
        for i, page in enumerate(doc, start=1):
            width = int(page.rect.width)
            height = int(page.rect.height)
            pix = page.get_pixmap(dpi=300)
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as imgf:
                imgf.write(pix.tobytes("png"))
                img_path = imgf.name
            try:
                # tesseract <img> stdout tsv
                cmd = [
                    "tesseract",
                    img_path,
                    "stdout",
                    "--psm",
                    "6",
                    "--oem",
                    "1",
                    "-c",
                    "preserve_interword_spaces=1",
                    "tsv",
                ]
                proc = subprocess.run(cmd, capture_output=True, check=False, text=True)
                tsv = proc.stdout
                tokens: List[Token] = []
                for j, line in enumerate(tsv.splitlines()):
                    if j == 0 or not line.strip():
                        continue
                    parts = line.split("\t")
                    if len(parts) < 12:
                        continue
                    text = parts[11]
                    try:
                        x, y, w, h = map(int, (parts[6], parts[7], parts[8], parts[9]))
                    except Exception:
                        continue
                    if not text.strip():
                        continue
                    tokens.append(Token(text=text, x0=x, y0=y, x1=x + w, y1=y + h, line=0, col=None))
                _group_lines(tokens)
                col_ranges = _detect_columns_from_headers(tokens, width)
                if col_ranges:
                    logger.info("TP1: page=%d columns detected x=%s", i, col_ranges)
                    _assign_columns(tokens, col_ranges)
                else:
                    for t in tokens:
                        t["col"] = None

                lines: Dict[int, List[str]] = {}
                for t in tokens:
                    lines.setdefault(t["line"], []).append(t["text"])
                ordered_lines = [" ".join(lines[k]) for k in sorted(lines.keys())]
                full_lines.extend(ordered_lines)
                pages.append(Page(number=i, width=width, height=height, tokens=tokens))
            finally:
                try:
                    os.unlink(img_path)
                except Exception:
                    pass
    return pages, "\n".join(full_lines)


def load_text_with_layout(pdf_path: str) -> Dict[str, Any]:
    """
    Extract tokens with coordinates from ``pdf_path``.

    Returns a mapping of the form:
      {
        "pages": [
          {
            "number": int,
            "width": int,
            "height": int,
            "tokens": [
              {"text": str, "x0": float, "y0": float, "x1": float, "y1": float, "line": int, "col": 0|1|2|None},
              ...
            ],
          },
          ...
        ],
        "full_text": str,
      }
    """
    pdf_path = str(pdf_path)
    if BLOCK_DEBUG:
        logger.info("TP1: load layout start file=%s", pdf_path)

    pages, full_text = _extract_pdf_tokens(pdf_path)

    if not any(p.get("tokens") for p in pages):
        # Probably image-based; try OCR TSV path
        ocr_pages, ocr_full = _extract_ocr_tokens(pdf_path)
        if ocr_pages:
            pages, full_text = ocr_pages, ocr_full or full_text

    # Post-pass: if tokens have no column assignment (e.g., custom extractors), try detect + assign
    try:
        for p in pages:
            toks = p.get("tokens", [])
            needs_cols = toks and all(t.get("col") is None for t in toks)
            if needs_cols:
                _group_lines(toks)
                col_ranges = _detect_columns_from_headers(toks, p.get("width", 0))
                if col_ranges:
                    logger.info("TP1: page=%d columns detected x=%s", p.get("number", 0), col_ranges)
                    _assign_columns(toks, col_ranges)
    except Exception:
        pass

    # Log per-page token counts
    if BLOCK_DEBUG:
        try:
            total = sum(len(p["tokens"]) for p in pages)
            logger.info("TP1: tokens count=%d", total)
        except Exception:
            pass

    return {"pages": pages, "full_text": full_text}


# Expose helpers for tests
__all__ = [
    "load_text_with_layout",
]
