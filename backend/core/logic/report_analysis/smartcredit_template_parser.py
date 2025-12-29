from __future__ import annotations

import json
import logging
from pathlib import Path
from backend.pipeline.runs import RunManifest
from typing import Any, Dict, List, Tuple

import backend.config as config
from .column_reader import (
    detect_bureau_columns as _detect_bureau_columns,
    extract_bureau_table as _extract_bureau_table,
)
from .normalize_fields import CANONICAL_KEYS as _CANON_KEYS


logger = logging.getLogger(__name__)


def _mid(a: Any, b: Any) -> float:
    try:
        return (float(a) + float(b)) / 2.0
    except Exception:
        return 0.0


def _read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _load_stageA(session_id: str, root: Path) -> Tuple[dict, dict] | None:
    try:
        m = RunManifest.for_sid(session_id)
        base = m.ensure_run_subdir("traces_blocks_dir", "traces/blocks")
    except Exception:
        base = root / "traces" / "blocks" / session_id
    layout_path = base / "layout_snapshot.json"
    windows_path = base / "block_windows.json"
    layout = _read_json(layout_path)
    windows = _read_json(windows_path)
    if not layout or not windows:
        return None
    return layout, windows


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


def _bands_to_codes(bands: Dict[str, Tuple[float, float]]) -> List[str]:
    order = ["transunion", "experian", "equifax"]
    code_map = {"transunion": "TU", "experian": "EX", "equifax": "EQ"}
    out: List[str] = []
    for b in order:
        if b in bands:
            out.append(code_map[b])
    return out


def parse(session_id: str, root: Path | None = None) -> dict:
    """
    Returns: {"ok": bool, "accounts_count": int, "accounts_path": str, "confidence": float}
    Writes: runs/<SID>/traces/blocks/accounts_template/accounts_template.json
    """

    try:
        logger.info("Template using Stage-A artifacts only (sid=%s)", session_id)
        base_root = root or Path.cwd()
        loaded = _load_stageA(session_id, base_root)
        if not loaded:
            logger.warning("TEMPLATE: parser missing Stage-A artifacts (sid=%s)", session_id)
            return {
                "ok": False,
                "accounts_count": 0,
                "accounts_path": "",
                "confidence": 0.0,
                "reason": "missing_artifacts",
            }
        layout, windows = loaded

        # Build pages index
        pages: Dict[int, dict] = {}
        for idx, p in enumerate(list(layout.get("pages") or []), start=1):
            try:
                num = int(p.get("number", idx) or idx)
            except Exception:
                num = idx
            pages[num] = p

        accounts: List[dict] = []
        good_accounts = 0
        checked_accounts = 0

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

            checked_accounts += 1
            header_tokens = [
                t
                for t in tokens
                if str(t.get("text", "")).strip().lower() in ("transunion", "experian", "equifax")
            ]
            bands = _detect_bureau_columns(header_tokens) if header_tokens else {}
            bands_codes = _bands_to_codes(bands)

            # Prepare block for extractor
            block_for_extract = {
                "layout_tokens": tokens,
                "meta": {"debug": {"layout_window": window}},
            }
            fields_per_bureau = _extract_bureau_table(block_for_extract)

            # Normalize into fields dict: per canonical key collect 3-bureau values
            fields: Dict[str, Dict[str, str]] = {}
            for key in _CANON_KEYS:
                fields[key] = {
                    "transunion": str((fields_per_bureau.get("transunion", {}) or {}).get(key, "")),
                    "experian": str((fields_per_bureau.get("experian", {}) or {}).get(key, "")),
                    "equifax": str((fields_per_bureau.get("equifax", {}) or {}).get(key, "")),
                }

            # Count mapped labels for confidence heuristic (any bureau non-empty)
            mapped = 0
            for key in _CANON_KEYS:
                vals = fields.get(key) or {}
                if any(bool((vals.get(b) or "").strip()) for b in ("transunion", "experian", "equifax")):
                    mapped += 1

            min_labels = int(getattr(config, "TEMPLATE_LABEL_MIN_PER_BLOCK", 8))
            min_bureaus = int(getattr(config, "TEMPLATE_MIN_BUREAUS", 2))
            if mapped >= min_labels and len(bands_codes) >= min_bureaus:
                good_accounts += 1

            accounts.append(
                {
                    "block_id": int(row.get("block_id", 0) or 0),
                    "heading": row.get("heading"),
                    "fields": fields,
                    "bands_detected": bands_codes,
                }
            )

        # Overall confidence: share of accounts meeting minimum quality
        confidence = (good_accounts / checked_accounts) if checked_accounts else 0.0

        try:
            m = RunManifest.for_sid(session_id)
            out_blocks = m.ensure_run_subdir("traces_blocks_dir", "traces/blocks")
        except Exception:
            out_blocks = base_root / "traces" / "blocks" / session_id
        low = str(out_blocks.resolve()).lower()
        assert ("/runs/" in low) or ("\\runs\\" in low), "Template out_dir must live under runs/<SID>"
        out_dir = out_blocks / "accounts_template"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "accounts_template.json"
        output = {
            "session_id": session_id,
            "template": "smartcredit_v1",
            "confidence": float(confidence),
            "accounts": accounts,
        }
        out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")

        logger.info(
            "TEMPLATE: parsed accounts=%d conf=%.3f -> accounts_template.json",
            len(accounts),
            confidence,
        )
        return {
            "ok": True,
            "accounts_count": len(accounts),
            "accounts_path": str(out_path),
            "confidence": float(confidence),
        }
    except Exception:
        logger.warning("TEMPLATE: parser unexpected failure sid=%s", session_id, exc_info=True)
        return {
            "ok": False,
            "accounts_count": 0,
            "accounts_path": "",
            "confidence": 0.0,
            "reason": "unexpected_error",
        }


__all__ = ["parse"]
