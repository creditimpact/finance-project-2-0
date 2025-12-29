from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional


from backend.config import (
    HISTORY_X_MATCH_ENABLED,
    HISTORY_Y_CLUSTER_DY,
    HISTORY_Y_PAIR_MAX_DY,
    HISTORY_X_SEAM_GUARD,
    HISTORY_ROW_PAIR_MAX_DY,
    HISTORY_DEBUG,
)

logger = logging.getLogger(__name__)


def _mid(a: Any, b: Any) -> float:
    try:
        return (float(a) + float(b)) / 2.0
    except Exception:
        return 0.0


def _join_text(tokens: List[dict]) -> str:
    def midx(t: dict) -> float:
        return _mid(t.get("x0", 0.0), t.get("x1", 0.0))

    toks = sorted(tokens or [], key=midx)
    out: List[str] = []
    prev: dict | None = None
    for t in toks:
        txt = str(t.get("text", "")).strip()
        if not txt:
            continue
        if prev is None:
            out.append(txt)
            prev = t
            continue
        try:
            gap = float(t.get("x0", 0.0)) - float(prev.get("x1", 0.0))
        except Exception:
            gap = 0.0
        if gap >= 1.0 or (out[-1] and txt and out[-1][-1].isalnum() and txt[0].isalnum()):
            out.append(" ")
        out.append(txt)
        prev = t
    return "".join(out).strip()


def _norm(s: str | None) -> str:
    import re
    return re.sub(r"\W+", "", (s or "").lower())


_HIST_LABEL_NORM = "twoyearpaymenthistory"
_BUREAUS = ("transunion", "experian", "equifax")

_HEB_TO_EN_MONTH = {
    "ינו׳": "Jan",
    "פבר׳": "Feb",
    "מרץ": "Mar",
    "אפר׳": "Apr",
    "מאי": "May",
    "יוני": "Jun",
    "יולי": "Jul",
    "אוג׳": "Aug",
    "ספט׳": "Sep",
    "אוק׳": "Oct",
    "נוב׳": "Nov",
    "דצמ׳": "Dec",
}

_EN_MONTHS = {
    "jan": "Jan",
    "feb": "Feb",
    "mar": "Mar",
    "apr": "Apr",
    "may": "May",
    "jun": "Jun",
    "jul": "Jul",
    "aug": "Aug",
    "sep": "Sep",
    "oct": "Oct",
    "nov": "Nov",
    "dec": "Dec",
}

_STATUS = {"ok", "30", "60", "90", "120", "150", "180"}


def _is_month_token(txt: str) -> str | None:
    z = (txt or "").strip()
    if not z:
        return None
    if z in _HEB_TO_EN_MONTH:
        return _HEB_TO_EN_MONTH[z]
    nz = _norm(z)
    if nz[:3] in _EN_MONTHS:
        return _EN_MONTHS[nz[:3]]
    return None


def _is_status_token(txt: str) -> Optional[str]:
    z = (txt or "").strip().upper()
    if not z:
        return None
    # Harden: exclude colon-bearing tokens (e.g., '30:' in 7-year summaries)
    if ":" in z:
        return None
    if z == "OK":
        return "OK"
    # numeric codes like 30/60/90
    if z.isdigit() and z in _STATUS:
        return z
    return None


def _cluster_lines_y(tokens: List[dict], dy: float) -> List[List[dict]]:
    toks = sorted(tokens or [], key=lambda t: (_mid(t.get("y0", 0.0), t.get("y1", 0.0)), _mid(t.get("x0", 0.0), t.get("x1", 0.0))))
    groups: List[List[dict]] = []
    cur: List[dict] = []
    cur_y: float | None = None
    for t in toks:
        y = _mid(t.get("y0", 0.0), t.get("y1", 0.0))
        if cur_y is None or abs(y - cur_y) <= dy:
            cur.append(t)
            try:
                cur_y = sum(_mid(x.get("y0", 0.0), x.get("y1", 0.0)) for x in cur) / len(cur)
            except Exception:
                cur_y = y
        else:
            groups.append(cur)
            cur = [t]
            cur_y = y
    if cur:
        groups.append(cur)
    return groups


def _line_center_y(tokens: List[dict]) -> float:
    ys: List[float] = []
    for t in tokens:
        try:
            y0 = float(t.get("y0", 0.0))
            y1 = float(t.get("y1", y0))
            ys.append(_mid(y0, y1))
        except Exception:
            continue
    if not ys:
        return 0.0
    ys.sort()
    return ys[len(ys) // 2]


def _token_mid_x(t: dict) -> float:
    try:
        return _mid(t.get("x0", 0.0), t.get("x1", 0.0))
    except Exception:
        return 0.0


def extract_two_year_payment_history(
    session_id: str,
    block_id: int,
    heading: str | None,
    page_tokens: List[dict],
    window: Dict[str, Any],
    bands: Dict[str, Tuple[float, float]] | None,
    out_dir: Path,
) -> Optional[str]:
    """Extract two-year payment history with row-aware NO-SKIP pairing and X-range overlap matching."""
    # 1) Filter tokens to window Y range
    try:
        y_top = float(window.get("y_top", 0.0) or 0.0)
        y_bottom = float(window.get("y_bottom", 0.0) or 0.0)
    except Exception:
        return None
    yband = [t for t in (page_tokens or []) if float(t.get("y0", 0.0)) <= y_bottom and float(t.get("y1", 0.0)) >= y_top]

    # 2) Find heading line for Two-Year Payment History
    lines = _cluster_lines_y(yband, dy=1.5)
    hist_y_start: Optional[float] = None
    for line in lines:
        text = _join_text(line)
        if _norm(text) == _HIST_LABEL_NORM:
            try:
                hist_y_start = sum(_mid(x.get("y0", 0.0), x.get("y1", 0.0)) for x in line) / max(1, len(line))
            except Exception:
                hist_y_start = _mid(line[0].get("y0", 0.0), line[0].get("y1", 0.0))
            break
    if hist_y_start is None:
        return None

    # 3) Per-bureau collect (month,status) pairs
    values: Dict[str, List[str]] = {}
    months_axis: List[str] = []
    monthly_output: Dict[str, List[Dict[str, str]]] = {}
    # Determine bands or fallback to whole window thirds
    # NOTE: For Two-Year Payment History, bureaus are STACKED vertically.
    # We intentionally do NOT use X-bands for bureau ownership.
    # Tokens are scoped by Y only between bureau headers (and EQ stop-loss).

    # Prepare tokens by Y after historical header
    yband_after = [t for t in yband if _mid(t.get("y0", 0.0), t.get("y1", 0.0)) >= hist_y_start]

    # Find bureau header Y positions for Y-window scoping (STACKED)
    bureau_headers: Dict[str, float] = {}
    for bureau in _BUREAUS:
        for t in yband_after:
            if _norm(t.get("text")) == bureau:
                bureau_headers[bureau] = _mid(t.get("y0", 0.0), t.get("y1", 0.0))
                break

    # Detect 7-year stop-loss header (Equifax boundary). We are permissive:
    # any token whose normalized text contains both 'dayslate' and 'history' and a '7'.
    seven_year_y: Optional[float] = None
    for t in yband_after:
        txt = _norm(t.get("text"))
        if "dayslate" in txt and "history" in txt and "7" in txt:
            seven_year_y = _mid(t.get("y0", 0.0), t.get("y1", 0.0))
            break

    def tokens_in_bureau_stacked(name: str) -> List[dict]:
        """Filter tokens by Y-window per bureau (STACKED layout). No X filtering."""
        # Determine Y-window for this bureau
        y_start = bureau_headers.get(name, hist_y_start)
        y_end = y_bottom
        bureau_list = list(_BUREAUS)
        try:
            idx = bureau_list.index(name)
            for next_b in bureau_list[idx + 1:]:
                if next_b in bureau_headers:
                    y_end = bureau_headers[next_b]
                    break
        except ValueError:
            pass

        # Equifax stop-loss at 7-year header
        if name == "equifax" and seven_year_y and (y_start or 0.0) < seven_year_y < y_end:
            if HISTORY_DEBUG:
                logger.info("HIST_STOP_LOSS_EQ y_end=%.1f reason=7year_header", seven_year_y)
            y_end = seven_year_y

        out: List[dict] = []
        for t in yband_after:
            ty = _mid(t.get("y0", 0.0), t.get("y1", 0.0))
            if (y_start or 0.0) <= ty < y_end:
                out.append(t)

        # reading-order sort
        out.sort(key=lambda t: (_mid(t.get("y0", 0.0), t.get("y1", 0.0)), _mid(t.get("x0", 0.0), t.get("x1", 0.0))))

        if HISTORY_DEBUG:
            m_cnt = 0; s_cnt = 0; o_cnt = 0
            for t in out:
                txt = str(t.get("text", "")).strip()
                if _is_month_token(txt):
                    m_cnt += 1
                elif _is_status_token(txt):
                    s_cnt += 1
                else:
                    o_cnt += 1
            logger.info(
                "HIST_SCOPE_STACKED bureau=%s y_start=%.1f y_end=%.1f months=%d statuses=%d other=%d",
                name, (y_start or 0.0), y_end, m_cnt, s_cnt, o_cnt
            )

        return out

    def _pair_rows_no_skip(month_rows: List[List[dict]], status_rows: List[List[dict]], bureau: str) -> List[Tuple[List[dict], Optional[List[dict]]]]:
        """NO-SKIP pairing: each status row pairs with FIRST month row below it only."""
        pairs: List[Tuple[List[dict], Optional[List[dict]]]] = []
        
        # Compute Y-centers and sort
        m_infos = [(row, _line_center_y(row)) for row in month_rows]
        m_infos.sort(key=lambda x: x[1])  # Sort by Y ascending (top to bottom)
        
        s_infos = [(row, _line_center_y(row)) for row in status_rows]
        s_infos.sort(key=lambda x: x[1])  # Sort by Y ascending
        
        # Track which status rows have been used
        used_status: set = set()
        
        # For each status row (top to bottom), find FIRST month row below it
        status_to_month: Dict[int, int] = {}  # status_idx -> month_idx
        
        for s_idx, (srow, sy) in enumerate(s_infos):
            if s_idx in used_status:
                continue
            
            # Find FIRST month row below this status row
            best_m_idx: Optional[int] = None
            best_dy: Optional[float] = None
            
            for m_idx, (mrow, my) in enumerate(m_infos):
                # Month must be BELOW status (my > sy)
                if my <= sy:
                    continue
                
                dy = my - sy
                
                # Within tolerance?
                if dy > HISTORY_ROW_PAIR_MAX_DY:
                    continue
                
                # This is the FIRST month row below (smallest Y > sy)
                # NO-SKIP: we take it immediately
                best_m_idx = m_idx
                best_dy = dy
                break  # Take FIRST, don't look further
            
            if best_m_idx is not None:
                status_to_month[s_idx] = best_m_idx
                used_status.add(s_idx)
                if HISTORY_DEBUG:
                    logger.info(
                        "HIST_PAIR bureau=%s status_y=%.1f month_y=%.1f dy=%.1f paired=yes",
                        bureau, sy, m_infos[best_m_idx][1], best_dy
                    )
            else:
                if HISTORY_DEBUG:
                    logger.info(
                        "HIST_PAIR bureau=%s status_y=%.1f paired=no reason=no_month_below_or_too_far",
                        bureau, sy
                    )
        
        # Build final pairs: for each month row, find its paired status (if any)
        month_to_status: Dict[int, int] = {m_idx: s_idx for s_idx, m_idx in status_to_month.items()}
        
        for m_idx, (mrow, my) in enumerate(m_infos):
            if m_idx in month_to_status:
                s_idx = month_to_status[m_idx]
                srow = s_infos[s_idx][0]
                pairs.append((mrow, srow))
            else:
                pairs.append((mrow, None))
                if HISTORY_DEBUG:
                    logger.info(
                        "HIST_PAIR bureau=%s month_y=%.1f paired=no reason=no_status_above",
                        bureau, my
                    )
        
        return pairs

    def _assign_statuses_by_overlap(month_row: List[dict], status_row: Optional[List[dict]], bureau: str) -> List[Tuple[str, str]]:
        """Match months to statuses using X-range overlap (not midpoint)."""
        # Extract month tokens
        month_tokens = []
        for t in month_row:
            txt = str(t.get("text", "")).strip()
            if _is_month_token(txt):
                month_tokens.append(t)
        
        if not month_tokens:
            return []
        
        # Sort months by X (left to right)
        month_tokens.sort(key=lambda t: float(t.get("x0", 0.0)))
        
        # Extract status tokens
        status_tokens = []
        if status_row:
            for t in status_row:
                txt = str(t.get("text", "")).strip()
                val = _is_status_token(txt)
                if val:
                    # normalize 30+ -> 30
                    if val.endswith("+") and val[:-1].isdigit():
                        val = val[:-1]
                    status_tokens.append((t, val))
        
        # Match each month to status with max overlap
        result: List[Tuple[str, str]] = []
        used_status: set = set()  # For 1:1 enforcement
        
        for m_tok in month_tokens:
            month_text = str(m_tok.get("text", "")).strip()
            try:
                m_x0 = float(m_tok.get("x0", 0.0))
                m_x1 = float(m_tok.get("x1", 0.0))
            except Exception:
                result.append((month_text, "--"))
                continue
            
            best_status = "--"
            best_overlap = 0.0
            best_s_idx = None
            
            for s_idx, (s_tok, s_val) in enumerate(status_tokens):
                if s_idx in used_status:  # 1:1 enforcement
                    continue
                
                try:
                    s_x0 = float(s_tok.get("x0", 0.0))
                    s_x1 = float(s_tok.get("x1", 0.0))
                except Exception:
                    continue
                
                # Calculate X-range overlap
                overlap = max(0.0, min(m_x1, s_x1) - max(m_x0, s_x0))
                
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_status = s_val
                    best_s_idx = s_idx
                
                if HISTORY_DEBUG and overlap > 0:
                    logger.info(
                        "HIST_MATCH bureau=%s month=%s m=[%.1f,%.1f] status=%s s=[%.1f,%.1f] overlap=%.1f",
                        bureau, month_text, m_x0, m_x1, s_val, s_x0, s_x1, overlap
                    )
            
            if best_s_idx is not None and best_overlap > 0:
                used_status.add(best_s_idx)
            
            result.append((month_text, best_status))
        
        return result

    for bureau in _BUREAUS:
        toks = tokens_in_bureau_stacked(bureau)
        if not toks:
            continue

        if HISTORY_X_MATCH_ENABLED:
            # Split into month tokens and status tokens
            month_toks = [t for t in toks if _is_month_token(str(t.get("text", "")).strip())]
            status_toks = [t for t in toks if _is_status_token(str(t.get("text", "")).strip())]

            # Cluster rows by Y with configurable dy
            m_rows = _cluster_lines_y(month_toks, dy=HISTORY_Y_CLUSTER_DY)
            s_rows = _cluster_lines_y(status_toks, dy=HISTORY_Y_CLUSTER_DY)

            if HISTORY_DEBUG:
                logger.info("HIST_MONTH_ROWS bureau=%s count=%d", bureau, len(m_rows))
                logger.info("HIST_STATUS_ROWS bureau=%s count=%d", bureau, len(s_rows))

            # Use NO-SKIP pairing
            pairs = _pair_rows_no_skip(m_rows, s_rows, bureau)
            if HISTORY_DEBUG:
                logger.info("HIST_PAIR_ROWS bureau=%s pairs=%d", bureau, len(pairs))

            # Process pairs top->bottom using overlap matching
            per_row: List[List[Tuple[str, str]]] = []
            for mrow, srow in pairs:
                per_row.append(_assign_statuses_by_overlap(mrow, srow, bureau))

            # Flatten rows: within each row, months left->right by anchor X
            monthly: List[Dict[str, str]] = []
            for row in per_row:
                for month_text, val in row:
                    monthly.append({"month": month_text, "value": val})

            # Limit to last 24 entries (tail) if longer
            if len(monthly) > 24:
                monthly = monthly[-24:]

            monthly_output[bureau] = monthly
            # Derive values[] and candidate months axis from this bureau
            values[bureau] = [it.get("value", "") for it in monthly]
            if len(monthly) > len(months_axis):
                months_axis = [it.get("month", "") for it in monthly]

        else:
            # Legacy sequential logic (unchanged for backward compatibility)
            # find bureau heading within these tokens to start after
            start_y = None
            for t in toks:
                if _norm(t.get("text")) == bureau:
                    start_y = _mid(t.get("y0", 0.0), t.get("y1", 0.0))
                    break
            if start_y is None:
                start_y = hist_y_start
                
            seq_status: List[str] = []
            seq_months: List[str] = []
            last_status: Optional[str] = None
            for t in toks:
                if _mid(t.get("y0", 0.0), t.get("y1", 0.0)) < start_y:
                    continue
                txt = str(t.get("text", "")).strip()
                st = _is_status_token(txt)
                if st:
                    last_status = st
                    continue
                m = _is_month_token(txt)
                if m:
                    seq_months.append(m)
                    seq_status.append(last_status or "")
                    last_status = None
            # limit to last 24 months if longer
            if len(seq_months) > 24:
                seq_months = seq_months[-24:]
                seq_status = seq_status[-24:]
            values[bureau] = seq_status
            if len(seq_months) > len(months_axis):
                months_axis = seq_months

    if not months_axis or not any(values.get(b) for b in _BUREAUS):
        return None

    # Normalize values length to months length
    for b in _BUREAUS:
        seq = values.get(b) or []
        if len(seq) < len(months_axis):
            seq = seq + [""] * (len(months_axis) - len(seq))
        elif len(seq) > len(months_axis):
            seq = seq[: len(months_axis)]
        values[b] = seq

    # Write artifact
    out_obj = {
        "session_id": session_id,
        "block_id": int(block_id),
        "heading": heading,
        "type": "two_year_payment_history",
        "bureaus": list(_BUREAUS),
        "months": months_axis,
        "values": values,
    }
    if HISTORY_X_MATCH_ENABLED:
        out_obj["monthly"] = monthly_output
    out_path = out_dir / f"account_table_{block_id:02d}__{(heading or f'block-{block_id}').lower().replace(' ', '-').replace('/', '-')}.history.json"
    out_path.write_text(json.dumps(out_obj, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(out_path)


def extract_seven_year_delinquency(
    session_id: str,
    block_id: int,
    heading: str | None,
    page_tokens: List[dict],
    window: Dict[str, Any],
    bands: Dict[str, Tuple[float, float]] | None,
    out_dir: Path,
) -> Optional[str]:
    """Extracts the "Days Late - 7 Year History" summary per bureau.
    Returns the output filepath if extracted, else None.
    """
    try:
        y_top = float(window.get("y_top", 0.0) or 0.0)
        y_bottom = float(window.get("y_bottom", 0.0) or 0.0)
    except Exception:
        return None
    yband = [t for t in (page_tokens or []) if float(t.get("y0", 0.0)) <= y_bottom and float(t.get("y1", 0.0)) >= y_top]

    # Find heading line tolerant of dash styles
    target_norm = "dayslate7yearhistory"
    lines = _cluster_lines_y(yband, dy=1.5)
    start_y = None
    for line in lines:
        text = _join_text(line)
        if _norm(text) == target_norm:
            try:
                start_y = sum(_mid(x.get("y0", 0.0), x.get("y1", 0.0)) for x in line) / max(1, len(line))
            except Exception:
                start_y = _mid(line[0].get("y0", 0.0), line[0].get("y1", 0.0))
            break
    if start_y is None:
        return None

    # Choose bands
    bands_use: Dict[str, Tuple[float, float]] = {}
    if bands:
        for name, rng in bands.items():
            bands_use[name.lower()] = (float(rng[0]), float(rng[1]))
    else:
        try:
            x_min = float(window.get("x_min", 0.0)); x_max = float(window.get("x_max", 0.0))
        except Exception:
            return None
        w = (x_max - x_min) / 3.0 if (x_max - x_min) > 0 else 0.0
        bands_use = {
            "transunion": (x_min, x_min + w),
            "experian": (x_min + w, x_min + 2 * w),
            "equifax": (x_min + 2 * w, x_max),
        }

    # Tokens after header only
    yband_after = [t for t in yband if _mid(t.get("y0", 0.0), t.get("y1", 0.0)) >= start_y]

    def tokens_in_band(name: str) -> List[dict]:
        xL, xR = bands_use.get(name, (None, None))
        if xL is None:
            return []
        out: List[dict] = []
        for t in yband_after:
            mx = _mid(t.get("x0", 0.0), t.get("x1", 0.0))
            if xL <= mx <= xR:
                out.append(t)
        return out

    import re
    pat = re.compile(r"\b(30|60|90)\s*:\s*(\d+)\b")

    values: Dict[str, Dict[str, int]] = {b: {"30": 0, "60": 0, "90": 0} for b in _BUREAUS}
    found_any = False
    for bureau in _BUREAUS:
        toks = tokens_in_band(bureau)
        if not toks:
            continue
        # Join by lines to get text like "30: 0"
        lines_b = _cluster_lines_y(toks, dy=1.0)
        for line in lines_b:
            text = _join_text(line)
            for m in pat.finditer(text):
                key, val = m.group(1), m.group(2)
                try:
                    values[bureau][key] = int(val)
                    found_any = True
                except Exception:
                    pass

    if not found_any:
        return None

    out_obj = {
        "session_id": session_id,
        "block_id": int(block_id),
        "type": "seven_year_delinquency",
        "values": values,
    }
    out_path = out_dir / f"account_table_{block_id:02d}__{(heading or f'block-{block_id}').lower().replace(' ', '-').replace('/', '-')}.delinquency.json"
    out_path.write_text(json.dumps(out_obj, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(out_path)

