from __future__ import annotations

import logging
import re
from typing import Dict, List, Any, Tuple, Optional

logger = logging.getLogger(__name__)

# Simple normalizer for matching headers
_WS_RE = re.compile(r"\s+")


def _norm(s: str | None) -> str:
    if not isinstance(s, str):
        return ""
    s2 = _WS_RE.sub(" ", s.strip()).lower()
    return s2


def _looks_bureau(text: str) -> Optional[str]:
    t = _norm(text)
    if t.startswith("transunion"):
        return "transunion"
    if t.startswith("experian"):
        return "experian"
    if t.startswith("equifax"):
        return "equifax"
    return None


def _is_two_year_header(joined: str) -> bool:
    s = _norm(joined)
    return ("two" in s and "year" in s and "payment" in s and "history" in s)


def _is_seven_year_header(joined: str) -> bool:
    s = _norm(joined)
    return ("seven" in s and "year" in s and ("history" in s or "days" in s))


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(float(x))
    except Exception:
        return default


def extract_2y_months_from_tsv_rows(
    *,
    session_id: str,
    heading: str,
    tsv_rows: List[Dict[str, str]],
) -> Dict[str, List[str]] | None:
    """
    Phase-1 TSV-based month extractor: returns month label tokens per bureau.

    - Finds the 2Y region by locating a line containing "Two-Year Payment History"
    - Locates a triad header line with all three bureau names to determine x-column anchors
    - Scans subsequent lines in the 2Y region, classifies short, non-numeric tokens by bureau column
    - Returns {"transunion": [...], "experian": [...], "equifax": [...]} preserving order

    Notes: This is intentionally conservative and only passes through tokens that
    look like month labels; statuses are not parsed here.
    """
    logger.info("TSV_2Y: EXTRACTOR_CALLED sid=%s heading=%s rows=%d", session_id, heading, len(tsv_rows))
    try:
        # Group tokens by (page, line)
        by_line: Dict[Tuple[int, int], List[Dict[str, Any]]] = {}
        for r in tsv_rows:
            p = _safe_int(r.get("page"))
            l = _safe_int(r.get("line"))
            by_line.setdefault((p, l), []).append(r)
        # Sort tokens within each line by x0
        for key in by_line.keys():
            by_line[key].sort(key=lambda t: _safe_float(t.get("x0")))
        # Determine reading order of lines (by page, then line)
        line_keys = sorted(by_line.keys())

        # 1) Find 2Y header line
        two_year_start_idx: Optional[int] = None
        for i, key in enumerate(line_keys):
            toks = by_line[key]
            joined = " ".join([str(t.get("text", "")) for t in toks])
            if _is_two_year_header(joined):
                two_year_start_idx = i
                break
        if two_year_start_idx is None:
            logger.info(
                "TSV_2Y: no header found sid=%s heading=%s rows=%d",
                session_id,
                heading,
                len(tsv_rows),
            )
            return {}

        # 1b) Find 7Y header line to set an upper bound
        seven_year_start_idx: Optional[int] = None
        for i in range(two_year_start_idx + 1, len(line_keys)):
            toks = by_line[line_keys[i]]
            joined = " ".join([str(t.get("text", "")) for t in toks])
            if _is_seven_year_header(joined):
                seven_year_start_idx = i
                break
        # If not found, treat end of file as upper bound
        if seven_year_start_idx is None:
            seven_year_start_idx = len(line_keys)

        # Log detected bounds
        two_key = line_keys[two_year_start_idx]
        sev_key = line_keys[seven_year_start_idx - 1] if seven_year_start_idx > two_year_start_idx else line_keys[two_year_start_idx]
        logger.info(
            "TSV_2Y: bounds sid=%s heading=%s twoY_start=(p=%s,l=%s,i=%d) sevenY_start_i=%d",
            session_id,
            heading,
            two_key[0],
            two_key[1],
            two_year_start_idx,
            seven_year_start_idx,
        )

        # 2) Find triad header to get x anchors (tu/xp/eq)
        tu_x: Optional[float] = None
        xp_x: Optional[float] = None
        eq_x: Optional[float] = None
        header_line_idx: Optional[int] = None
        for i in range(two_year_start_idx + 1, seven_year_start_idx):
            toks = by_line[line_keys[i]]
            names: Dict[str, float] = {}
            for t in toks:
                b = _looks_bureau(str(t.get("text", "")))
                if b:
                    names[b] = _safe_float(t.get("x0"))
            # prefer a line with all three, otherwise keep searching
            if len(names) == 3:
                tu_x, xp_x, eq_x = names.get("transunion"), names.get("experian"), names.get("equifax")
                header_line_idx = i
                break
        if tu_x is None or xp_x is None or eq_x is None:
            # Fallback path: per-bureau header + month lines inside 2Y bounds
            logger.info(
                "TSV_2Y: triad header not found; using per-bureau fallback sid=%s heading=%s",
                session_id,
                heading,
            )

            def _line_contains_bureau(i: int, bureau: str) -> bool:
                toks = by_line[line_keys[i]]
                for t in toks:
                    if _looks_bureau(str(t.get("text", ""))) == bureau:
                        return True
                return False

            def _select_month_line_after(header_i: int) -> Optional[int]:
                start = header_i + 1
                end = min(header_i + 7, seven_year_start_idx)
                best: Optional[Tuple[int, int]] = None  # (idx, score)
                for j in range(start, end):
                    toks = by_line[line_keys[j]]
                    joined = " ".join([str(t.get("text", "")) for t in toks])
                    if _is_seven_year_header(joined):
                        break
                    # stop if this line announces another bureau header
                    if any(_line_contains_bureau(j, b) for b in ("transunion", "experian", "equifax")):
                        break
                    invalid_line = any(
                        (txt := str(t.get("text", ""))).strip().endswith(":") or any(pat.search(txt) for pat in INVALID_PATTERNS)
                        for t in toks
                    )
                    if invalid_line:
                        continue
                    score = sum(1 for t in toks if _is_month_like(str(t.get("text", ""))))
                    if score > 0 and (best is None or score > best[1]):
                        best = (j, score)
                return best[0] if best else None

            out_fb: Dict[str, List[str]] = {"transunion": [], "experian": [], "equifax": []}
            bureau_headers: Dict[str, int] = {}
            for i in range(two_year_start_idx + 1, seven_year_start_idx):
                for b in ("transunion", "experian", "equifax"):
                    if b not in bureau_headers and _line_contains_bureau(i, b):
                        bureau_headers[b] = i
            # For each bureau, pick month line after its header and collect tokens
            for b in ("transunion", "experian", "equifax"):
                hi = bureau_headers.get(b)
                if hi is None:
                    continue
                mi = _select_month_line_after(hi)
                if mi is None:
                    continue
                key = line_keys[mi]
                toks = by_line[key]
                vals: List[str] = []
                for t in toks:
                    txt = str(t.get("text", ""))
                    if _is_month_like(txt):
                        vals.append(txt)
                if vals:
                    out_fb[b].extend(vals)
                    logger.info(
                        "TSV_2Y: selected month line (fb) sid=%s heading=%s bureau=%s line=(p=%s,l=%s,i=%d) tokens=%s",
                        session_id,
                        heading,
                        b,
                        key[0],
                        key[1],
                        mi,
                        vals[:30],
                    )

            # Final validation
            flat_fb = [t.lower() for arr in out_fb.values() for t in arr]
            if any(t.endswith(":") or re.fullmatch(r"\d+:", t) or t in {"30:", "60:", "90:", "30 days", "60 days"} for t in flat_fb):
                logger.info("TSV_2Y: fallback invalid tokens detected sid=%s heading=%s sample=%s", session_id, heading, flat_fb[:10])
                return {}
            if any(len(v) > 0 for v in out_fb.values()):
                logger.info(
                    "TSV_2Y: extracted months (fb) sid=%s heading=%s counts tu=%d xp=%d eq=%d samples tu=%s xp=%s eq=%s",
                    session_id,
                    heading,
                    len(out_fb["transunion"]),
                    len(out_fb["experian"]),
                    len(out_fb["equifax"]),
                    out_fb["transunion"][:5],
                    out_fb["experian"][:5],
                    out_fb["equifax"][:5],
                )
                return out_fb
            logger.info("TSV_2Y: fallback found no month rows sid=%s heading=%s", session_id, heading)
            return {}

        # Compute column split thresholds between anchors
        xs = sorted([tu_x, xp_x, eq_x])  # type: ignore
        b1 = (xs[0] + xs[1]) / 2.0
        b2 = (xs[1] + xs[2]) / 2.0

        header_key = line_keys[header_line_idx]
        logger.info(
            "TSV_2Y: header sid=%s heading=%s line=(p=%s,l=%s,i=%d) anchors tu=%.1f xp=%.1f eq=%.1f splits b1=%.1f b2=%.1f",
            session_id,
            heading,
            header_key[0],
            header_key[1],
            header_line_idx,
            tu_x,
            xp_x,
            eq_x,
            b1,
            b2,
        )

        # 3) Scan lines until 7Y header or end; classify month-like tokens by x-column
        out: Dict[str, List[str]] = {"transunion": [], "experian": [], "equifax": []}

        def _classify_bureau_by_thresholds(x0: float) -> str:
            if x0 < b1:
                return "transunion"
            if x0 < b2:
                return "experian"
            return "equifax"

        # Month token filters
        STATUS_LITERALS = {"ok", "co", "kd", "kd:", "co:", "ok:", "charge", "off"}
        STATUS_NUMBERS = {"0", "30", "60", "90", "120", "150", "180"}
        INVALID_PATTERNS = [
            re.compile(r"^\d+:$", re.IGNORECASE),  # 30:, 60:, 90:
            re.compile(r"^\d+\s*(day|days)$", re.IGNORECASE),  # 30 days
        ]

        def _has_hebrew(s: str) -> bool:
            return any("\u0590" <= ch <= "\u05FF" for ch in s)

        def _is_month_like(txt: str) -> bool:
            s = txt.strip()
            if not s:
                return False
            low = s.lower()
            if low in STATUS_LITERALS or low in STATUS_NUMBERS:
                return False
            if any(pat.search(s) for pat in INVALID_PATTERNS):
                return False
            if s.endswith(":"):
                return False
            # Skip bureau names
            if low in {"transunion", "experian", "equifax"}:
                return False
            # Strongly prefer Hebrew tokens (month abbreviations in this PDF are Hebrew)
            if _has_hebrew(s):
                return True
            # For non-Hebrew, reject common English words that appear in credit reports
            common_words = {
                "account", "status", "payment", "date", "amount", "balance",
                "limit", "credit", "reported", "remarks", "creditor", "type",
                "description", "rating", "days", "months", "months", "duration",
                "frequency", "month", "year", "open", "closed", "active",
                "charged", "off", "installment", "accounts", "comprised", "of",
                "fixed", "terms", "with", "regular", "payments", "cycle",
                "high", "credit", "line", "revolving", "installment", "note"
            }
            if low in common_words:
                return False
            # Only accept short, all-alpha tokens (2-4 chars) for non-Hebrew
            # This covers month abbreviations like "Feb", "Jan", "Jul", etc.
            if re.fullmatch(r"[A-Za-z]{2,4}", s):
                return True
            return False

        # Search within a small window of lines after header, bounded by 7Y start
        start_i = header_line_idx + 1
        end_i = min(header_line_idx + 7, seven_year_start_idx)

        candidates: List[Tuple[int, Dict[str, List[Tuple[float, str]]], int]] = []
        for i in range(start_i, end_i):
            key = line_keys[i]
            toks = by_line[key]
            # If this line itself announces 7Y, stop
            joined = " ".join([str(t.get("text", "")) for t in toks])
            if _is_seven_year_header(joined):
                break
            per_bureau: Dict[str, List[Tuple[float, str]]] = {"transunion": [], "experian": [], "equifax": []}
            invalid_line = False
            for t in toks:
                txt = str(t.get("text", ""))
                if any(pat.search(txt) for pat in INVALID_PATTERNS) or txt.strip().endswith(":"):
                    invalid_line = True
                    # Continue scanning tokens but mark as invalid candidate
                if not _is_month_like(txt):
                    continue
                bx = _classify_bureau_by_thresholds(_safe_float(t.get("x0")))
                per_bureau[bx].append((_safe_float(t.get("x0")), txt))
            total = sum(len(v) for v in per_bureau.values())
            if total > 0:
                candidates.append((i, per_bureau, total if not invalid_line else max(1, total - 5)))

        # Sort candidates by score (descending), prefer more tokens
        candidates.sort(key=lambda c: c[2], reverse=True)

        selected_idxes: List[int] = []
        for i, per_bureau, _score in candidates:
            # Validate against forbidden lists like ["30:", "60:", "90:"]
            flat = [txt for arr in per_bureau.values() for (_x, txt) in arr]
            if any(any(pat.search(txt) for pat in INVALID_PATTERNS) or txt.endswith(":") for txt in flat):
                continue
            # Looks acceptable; keep this line
            selected_idxes.append(i)
            # For phase-1, a single good line is usually enough
            break

        if not selected_idxes:
            logger.info(
                "TSV_2Y: months not present in 2Y window sid=%s heading=%s header_i=%d bounds=(%d,%d)",
                session_id,
                heading,
                header_line_idx,
                start_i,
                end_i,
            )
            return {}

        # Build final out from selected line(s)
        preview_tokens: List[str] = []
        for i in selected_idxes:
            key = line_keys[i]
            per_bureau_pairs = candidates[[idx for idx, *_ in candidates].index(i)][1]
            for b in ("transunion", "experian", "equifax"):
                # sort by x to preserve left->right order on that line
                per_bureau_pairs[b].sort(key=lambda it: it[0])
                out[b].extend([txt for (_x, txt) in per_bureau_pairs[b]])
                preview_tokens.extend([txt for (_x, txt) in per_bureau_pairs[b]][:10])
            logger.info(
                "TSV_2Y: selected month line sid=%s heading=%s line=(p=%s,l=%s,i=%d) tokens=%s",
                session_id,
                heading,
                key[0],
                key[1],
                i,
                preview_tokens[:30],
            )

        # Final validation: ensure we didn't accidentally pick 7Y headers like 30:,60:,90:
        forbidden = {"30:", "60:", "90:", "30 days", "60 days"}
        flat_out = [t.lower() for b in out.values() for t in b]
        if any(t in forbidden or re.fullmatch(r"\d+:", t) for t in flat_out):
            logger.info(
                "TSV_2Y: invalid tokens detected after selection sid=%s heading=%s tokens_sample=%s",
                session_id,
                heading,
                flat_out[:10],
            )
            return {}

        # Log final counts
        logger.info(
            "TSV_2Y: extracted months sid=%s heading=%s counts tu=%d xp=%d eq=%d samples tu=%s xp=%s eq=%s",
            session_id,
            heading,
            len(out["transunion"]),
            len(out["experian"]),
            len(out["equifax"]),
            out["transunion"][:5],
            out["experian"][:5],
            out["equifax"][:5],
        )

        return out
    except Exception:
        logger.exception(
            "TSV_2Y: extractor failed sid=%s heading=%s rows=%d",
            session_id,
            heading,
            len(tsv_rows) if isinstance(tsv_rows, list) else -1,
        )
        return None
