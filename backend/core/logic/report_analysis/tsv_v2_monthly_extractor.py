"""
TSV v2: Monthly status pairing for 2Y section.

Rules (from PDF):
- STATUS is ABOVE month rows
- Same bounded 2Y slice as months extraction (TU→EX→EQ)
- No truncation or invented output
- Token expansion: split packed tokens (e.g., "OK OK OK" → 3 tokens)
- Y-clustering: ~2.5 pt tolerance
- Row pairing: status row → first month row below (NO-SKIP greedy)
- X-overlap: assign month→status by X position
- Output: exactly one pair per month token
"""

import hashlib
import logging
import re
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)

STATUS_TOKENS = {"ok", "co", "30", "60", "90", "120", "150", "180"}
HEBREW_MONTHS = {
    "ינו׳", "פבר׳", "מרץ", "אפר׳", "מאי", "יוני",
    "יולי", "אוג׳", "ספט׳", "אוק׳", "נוב׳", "דצמ׳",
}
ENGLISH_MONTHS = {
    "jan", "feb", "mar", "apr", "may", "jun",
    "jul", "aug", "sep", "sept", "oct", "nov", "dec",
}

# Explicit month→number mappings (normalized lowercase for English; raw for Hebrew)
EN_MONTH_TO_NUM = {
    "jan": 1,
    "feb": 2,
    "mar": 3,
    "apr": 4,
    "may": 5,
    "jun": 6,
    "jul": 7,
    "aug": 8,
    "sep": 9,
    "sept": 9,
    "oct": 10,
    "nov": 11,
    "dec": 12,
}

HEB_MONTH_TO_NUM = {
    "ינו׳": 1,
    "פבר׳": 2,
    "מרץ": 3,
    "אפר׳": 4,
    "מאי": 5,
    "יוני": 6,
    "יולי": 7,
    "אוג׳": 8,
    "ספט׳": 9,
    "אוק׳": 10,
    "נוב׳": 11,
    "דצמ׳": 12,
}

YEAR_MARKER_RE = re.compile(r"^[""'’](\d{2})$")


def _normalize_month_token(token: str) -> str:
    normalized = token.strip().lower()
    normalized = normalized.rstrip(".")
    normalized = re.sub(r"\s*\(?\s*['’]?\d{2,4}\)?$", "", normalized)
    return normalized


def _is_month_token(text: str) -> bool:
    normalized = _normalize_month_token(text)
    if normalized in ENGLISH_MONTHS:
        return True
    if YEAR_MARKER_RE.match(text.strip()):
        return True
    return text.strip() in HEBREW_MONTHS


def _month_num_from_text(text: str) -> Optional[int]:
    """Map a month token to its numeric month (1-12) if possible."""
    raw = text.strip()
    if not raw:
        return None
    # Year markers handled separately as anchors
    normalized = _normalize_month_token(raw)
    if normalized in EN_MONTH_TO_NUM:
        return EN_MONTH_TO_NUM[normalized]
    if raw in HEB_MONTH_TO_NUM:
        return HEB_MONTH_TO_NUM[raw]
    return None


def _determine_direction(month_nums: List[Optional[int]]) -> str:
    """Infer dominant traversal direction (forward vs backward in time)."""
    fwd = 0
    back = 0
    prev: Optional[int] = None
    for m in month_nums:
        if m is None:
            continue
        if prev is None:
            prev = m
            continue
        if prev == 12 and m == 1:
            fwd += 1
        elif prev == 1 and m == 12:
            back += 1
        elif m > prev:
            fwd += 1
        elif m < prev:
            back += 1
        prev = m
    return "forward" if fwd >= back else "backward"


def _assign_years_and_keys(pairs: List[Dict[str, Any]]) -> None:
    """Assign derived_year and month_year_key for all entries in-place."""

    # Ensure every entry has derived_month_num when possible
    month_nums: List[Optional[int]] = []
    for entry in pairs:
        mnum = entry.get("derived_month_num")
        if not isinstance(mnum, int):
            mnum = _month_num_from_text(str(entry.get("month", "")))
            if mnum is not None:
                entry["derived_month_num"] = mnum
        month_nums.append(mnum)

    direction = _determine_direction(month_nums)

    # Identify anchors (year tokens)
    anchor_indices = [i for i, e in enumerate(pairs) if e.get("year_token_raw") and isinstance(e.get("derived_year"), int)]
    if not anchor_indices:
        logger.warning("TSV_V2_MONTHLY: no year anchors found; derived_year cannot be assigned")
        # Still attempt month_year_key only where year already present
        for entry in pairs:
            mnum = entry.get("derived_month_num")
            year = entry.get("derived_year")
            if isinstance(mnum, int) and isinstance(year, int):
                entry["month_year_key"] = f"{year:04d}-{mnum:02d}"
        return

    # Forward pass (document order)
    current_year: Optional[int] = None
    prev_month: Optional[int] = None
    for i, entry in enumerate(pairs):
        mnum = month_nums[i]
        if i in anchor_indices:
            current_year = entry.get("derived_year")
            prev_month = mnum if mnum is not None else prev_month
            continue
        if current_year is None or mnum is None:
            prev_month = mnum if mnum is not None else prev_month
            continue

        if direction == "forward":
            if prev_month == 12 and mnum == 1:
                current_year += 1
        else:
            if prev_month == 1 and mnum == 12:
                current_year -= 1

        entry["derived_year"] = current_year
        prev_month = mnum

    # Backward pass (reverse order) to fill gaps before first anchor
    current_year = None
    next_month: Optional[int] = None
    for i in range(len(pairs) - 1, -1, -1):
        entry = pairs[i]
        mnum = month_nums[i]
        if isinstance(entry.get("derived_year"), int):
            current_year = entry["derived_year"]
            next_month = mnum if mnum is not None else next_month
            continue
        if current_year is None or mnum is None:
            next_month = mnum if mnum is not None else next_month
            continue

        year = current_year
        if direction == "forward":
            if mnum == 12 and next_month == 1:
                year = current_year - 1
        else:  # backward
            if mnum == 1 and next_month == 12:
                year = current_year + 1

        entry["derived_year"] = year
        current_year = year
        next_month = mnum

    # Final month_year_key population
    for entry in pairs:
        mnum = entry.get("derived_month_num")
        year = entry.get("derived_year")
        if isinstance(mnum, int) and isinstance(year, int):
            entry["month_year_key"] = f"{year:04d}-{mnum:02d}"


@dataclass
class Token:
    """Normalized token with position and text."""
    text: str
    page: int
    line: int
    x0: float
    x1: float
    y0: float
    y1: float
    
    @property
    def x_mid(self) -> float:
        return (self.x0 + self.x1) / 2.0
    
    @property
    def y_mid(self) -> float:
        return (self.y0 + self.y1) / 2.0


def expand_packed_tokens(tokens: List[Token]) -> List[Token]:
    """
    Split tokens containing multiple space-separated values.
    
    E.g., "OK OK OK 90 120" → ["OK", "OK", "OK", "90", "120"]
    Distribute x-range across split tokens proportionally.
    """
    expanded = []
    for tok in tokens:
        parts = tok.text.split()
        if len(parts) <= 1:
            expanded.append(tok)
            continue
        # Distribute x-range across parts
        x_span = tok.x1 - tok.x0
        part_width = x_span / len(parts)
        for i, part in enumerate(parts):
            part_x0 = tok.x0 + i * part_width
            part_x1 = part_x0 + part_width
            expanded.append(
                Token(
                    text=part,
                    page=tok.page,
                    x0=part_x0,
                    x1=part_x1,
                    y0=tok.y0,
                    y1=tok.y1,
                )
            )
    return expanded


def cluster_by_y(tokens: List[Token], dy_tolerance: float = 2.5) -> List[List[Token]]:
    """Cluster tokens into rows by Y coordinate."""
    if not tokens:
        return []
    sorted_toks = sorted(tokens, key=lambda t: (t.page, t.y_mid))
    rows = []
    current_row = [sorted_toks[0]]
    for tok in sorted_toks[1:]:
        if tok.page == current_row[0].page and abs(tok.y_mid - current_row[0].y_mid) <= dy_tolerance:
            current_row.append(tok)
        else:
            rows.append(current_row)
            current_row = [tok]
    rows.append(current_row)
    return rows


def x_overlap(tok: Token, row: List[Token]) -> Optional[Token]:
    """Find token in row with overlapping X range (for pairing)."""
    best = None
    best_overlap = 0
    for r_tok in row:
        overlap_start = max(tok.x0, r_tok.x0)
        overlap_end = min(tok.x1, r_tok.x1)
        if overlap_end > overlap_start:
            overlap = overlap_end - overlap_start
            if overlap > best_overlap:
                best_overlap = overlap
                best = r_tok
    return best


def extract_tsv_v2_monthly(
    session_id: str,
    heading: str,
    idx: int,
    tokens_by_line: Dict[Tuple[int, int], List[dict]],
    lines: List[dict],
    tsv_v2_months: Optional[Dict[str, List[str]]] = None,
) -> Optional[Dict[str, List[Dict[str, str]]]]:
    """
    Extract monthly status pairings for 2Y section.
    
    Uses exact same bounds and slice as tsv_v2_months.
    
    Returns:
        Dict with keys 'transunion', 'experian', 'equifax'
        Each containing list of {"month": ..., "status": ...} pairs
    """
    if not tokens_by_line or not lines or tsv_v2_months is None:
        logger.warning(
            "TSV_V2_MONTHLY: missing input sid=%s heading=%s has_tsv_v2_months=%s",
            session_id, heading, tsv_v2_months is not None
        )
        return None

    # Account bounds from per-account lines
    account_pages = [ln.get("page") for ln in lines if "page" in ln]
    account_page_start = min(account_pages) if account_pages else None
    account_page_end = max(account_pages) if account_pages else None

    def _fingerprint(tokens: List[Token]) -> str:
        # Use first+last 50 tokens with page/line/text/x0/y0 to form a stable hash
        if not tokens:
            return ""
        sample = tokens[:50] + tokens[-50:] if len(tokens) > 100 else tokens
        parts = [
            f"p{t.page}-l{t.line}:{t.text}:{t.x0:.1f}:{t.y0:.1f}"
            for t in sample
        ]
        digest = hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()
        return digest
    
    # Step 1: Reconstruct full token stream from tokens_by_line
    sorted_keys = sorted(tokens_by_line.keys())
    all_tokens = []
    for (page, line) in sorted_keys:
        tok_list = tokens_by_line[(page, line)]
        for tok in tok_list:
            all_tokens.append(
                Token(
                    text=str(tok.get("text", "")),
                    page=page,
                    line=line,
                    x0=float(tok.get("x0", 0.0)),
                    x1=float(tok.get("x1", 0.0)),
                    y0=float(tok.get("y0", 0.0)),
                    y1=float(tok.get("y1", 0.0)),
                )
            )
    
    if not all_tokens:
        logger.warning("TSV_V2_MONTHLY: no tokens sid=%s heading=%s", session_id, heading)
        return None
    
    # Step 2: Find 2Y region bounds (same as months extraction)
    twoY_start_idx = None
    for i, tok in enumerate(all_tokens):
        text = tok.text.lower().strip()
        if "two" in text and ("year" in text or "payment" in text):
            twoY_start_idx = i
            logger.info("TSV_V2_MONTHLY: 2Y start idx=%d text='%s'", i, tok.text)
            break
    if twoY_start_idx is None:
        for i, tok in enumerate(all_tokens):
            if "history" in tok.text.lower().strip() and len(tok.text.strip()) > 3:
                twoY_start_idx = i
                logger.info("TSV_V2_MONTHLY: 2Y start (by history) idx=%d text='%s'", i, tok.text)
                break
    if twoY_start_idx is None:
        logger.warning("TSV_V2_MONTHLY: no 2Y start found sid=%s heading=%s", session_id, heading)
        return None
    
    # Find 7Y end
    sevenY_start_idx = len(all_tokens)
    for i in range(twoY_start_idx + 1, len(all_tokens)):
        text = all_tokens[i].text.lower().strip()
        if ("seven" in text or "7" in text) and ("year" in text or "history" in text or len(text) <= 2):
            sevenY_start_idx = i
            logger.info("TSV_V2_MONTHLY: 7Y start idx=%d text='%s'", i, all_tokens[i].text)
            break
    
    two_year_tokens = all_tokens[twoY_start_idx:sevenY_start_idx]
    slice_pages = [t.page for t in two_year_tokens]
    slice_page_min = min(slice_pages) if slice_pages else None
    slice_page_max = max(slice_pages) if slice_pages else None

    logger.warning(
        "TSV_V2_MONTHLY: sid=%s account_idx=%d heading=%s account_pages=[%s,%s]"
        " slice_pages=[%s,%s] token_count=%d fingerprint=%s",
        session_id,
        idx + 1,
        heading,
        account_page_start,
        account_page_end,
        slice_page_min,
        slice_page_max,
        len(two_year_tokens),
        _fingerprint(two_year_tokens),
    )

    if account_page_start is not None and account_page_end is not None and slice_page_min is not None and slice_page_max is not None:
        assert slice_page_min >= account_page_start and slice_page_max <= account_page_end, (
            f"2Y slice not within account pages sid={session_id} idx={idx+1} heading={heading} "
            f"slice_pages=[{slice_page_min},{slice_page_max}] account_pages=[{account_page_start},{account_page_end}]"
        )
    
    # Step 3: Find bureau markers
    bureau_indices = {}
    for i, tok in enumerate(two_year_tokens):
        text = tok.text.lower().strip()
        if "transunion" in text and "transunion" not in bureau_indices:
            bureau_indices["transunion"] = i
        elif "experian" in text and "experian" not in bureau_indices:
            bureau_indices["experian"] = i
        elif "equifax" in text and "equifax" not in bureau_indices:
            bureau_indices["equifax"] = i
    
    if not bureau_indices:
        logger.warning("TSV_V2_MONTHLY: no bureaus found sid=%s heading=%s", session_id, heading)
        return None
    
    # Step 4: Define slices (same as months)
    sorted_bureaus = sorted(bureau_indices.items(), key=lambda x: x[1])
    slices = {}
    for i, (bureau, idx) in enumerate(sorted_bureaus):
        next_idx = sorted_bureaus[i + 1][1] if i + 1 < len(sorted_bureaus) else len(two_year_tokens)
        slices[bureau] = (idx, next_idx)
    
    # Step 5: For each bureau, extract and pair months with statuses
    result = {"transunion": [], "experian": [], "equifax": []}
    
    for bureau in ("transunion", "experian", "equifax"):
        start, end = slices[bureau]
        b_tokens = two_year_tokens[start:end]
        
        # Extract months and statuses using shared month detection
        month_tokens = [t for t in b_tokens if _is_month_token(t.text)]
        status_toks = [t for t in b_tokens if t.text.strip().lower() in STATUS_TOKENS]
        
        # Log before expansion
        logger.info(
            "TSV_V2_MONTHLY: bureau=%s before_expansion months=%d statuses=%d",
            bureau, len(month_tokens), len(status_toks)
        )
        
        # Expand packed tokens
        month_tokens_expanded = expand_packed_tokens(month_tokens)
        status_toks_expanded = expand_packed_tokens(status_toks)
        
        logger.info(
            "TSV_V2_MONTHLY: bureau=%s after_expansion months=%d statuses=%d",
            bureau, len(month_tokens_expanded), len(status_toks_expanded)
        )
        
        # Determine page composition
        month_pages = sorted(set(t.page for t in month_tokens_expanded)) if month_tokens_expanded else []
        status_pages = sorted(set(t.page for t in status_toks_expanded)) if status_toks_expanded else []
        is_cross_page = month_pages and status_pages and month_pages != status_pages
        
        logger.warning(
            "TSV_V2_MONTHLY: %s BUREAU_ANALYSIS month_pages=%s status_pages=%s cross_page=%s start_idx=%d end_idx=%d",
            bureau.upper(), month_pages, status_pages, is_cross_page, start, end
        )
        
        # Cluster into rows (per-page)
        month_rows = cluster_by_y(month_tokens_expanded, dy_tolerance=2.5)
        status_rows = cluster_by_y(status_toks_expanded, dy_tolerance=2.5)
        
        logger.info(
            "TSV_V2_MONTHLY: bureau=%s clustered month_rows=%d status_rows=%d",
            bureau, len(month_rows), len(status_rows)
        )
        
        # Row-based pairing with optional cross-page support
        row_pairings: Dict[int, Optional[int]] = {}
        used_status_rows = set()
        
        for mi, m_row in enumerate(month_rows):
            m_row_page = m_row[0].page
            m_row_y = m_row[0].y_mid
            best_si = None
            best_dy = None
            
            # Find closest status row that is above this month row
            for si, s_row in enumerate(status_rows):
                if si in used_status_rows:
                    continue
                s_row_page = s_row[0].page
                s_row_y = s_row[0].y_mid
                
                # Check if status row is logically above
                # In PDF coords (Y down): status is above if status_y < month_y (same page)
                # or status_page < month_page (cross-page)
                is_above = False
                if is_cross_page:
                    # Cross-page: allow status on earlier page
                    is_above = (s_row_page < m_row_page) or (s_row_page == m_row_page and s_row_y < m_row_y)
                else:
                    # Same-page: require status_y < month_y (standard "above" in PDF coords)
                    is_above = (s_row_page == m_row_page and s_row_y < m_row_y)
                
                if not is_above:
                    continue
                
                # Compute distance
                if s_row_page == m_row_page:
                    dy = m_row_y - s_row_y
                else:
                    # Cross-page: page difference weighted
                    dy = (m_row_page - s_row_page) * 800.0 + (m_row_y - s_row_y)
                
                if best_dy is None or dy < best_dy:
                    best_dy = dy
                    best_si = si
            
            if best_si is not None:
                row_pairings[mi] = best_si
                used_status_rows.add(best_si)
            else:
                row_pairings[mi] = None
        
        logger.info(
            "TSV_V2_MONTHLY: bureau=%s row_pairings=%s",
            bureau, {k: v for k, v in row_pairings.items()}
        )
        
        # Token-level pairing within rows, with bureau slice enforcement
        pairs = []
        matched_count = 0
        example_matches = []
        
        for mi in range(len(month_rows)):
            m_row = month_rows[mi]
            si = row_pairings.get(mi)
            s_row = status_rows[si] if si is not None else []
            
            for m_tok in sorted(m_row, key=lambda t: t.x_mid):
                status_text = "--"
                matched_status = None
            
                if s_row:
                    # Find status token with X-overlap
                    matched_status = x_overlap(m_tok, s_row)
                    if matched_status:
                        # ENFORCE: matched status must be in bureau slice [start, end)
                        # Check by verifying it's in the b_tokens list (bureau-local)
                        if matched_status in b_tokens:
                            status_text = matched_status.text.strip().lower()
                            matched_count += 1
                            if len(example_matches) < 5:
                                overlap = min(m_tok.x1, matched_status.x1) - max(m_tok.x0, matched_status.x0)
                                example_matches.append({
                                    'month': m_tok.text,
                                    'status': status_text,
                                    'overlap': overlap,
                                    'status_page': matched_status.page,
                                    'status_y': matched_status.y_mid,
                                    'status_x0': matched_status.x0,
                                    'status_x1': matched_status.x1,
                                    'month_page': m_tok.page,
                                    'month_y': m_tok.y_mid,
                                    'month_x0': m_tok.x0,
                                    'month_x1': m_tok.x1,
                                })
                        else:
                            # This would be a bug - status came from outside slice
                            logger.error(
                                "TSV_V2_MONTHLY: BUREAU_SLICE_VIOLATION bureau=%s status_text=%s not in bureau slice",
                                bureau, matched_status.text
                            )
            
                month_entry = {
                    "month": m_tok.text.strip(),
                    "status": status_text,
                }

                # Month normalization (numeric month for all tokens where possible)
                month_num = _month_num_from_text(m_tok.text)
                if month_num is not None:
                    month_entry["derived_month_num"] = month_num

                year_match = YEAR_MARKER_RE.match(m_tok.text.strip())
                if year_match:
                    yy = int(year_match.group(1))
                    derived_year = 2000 + yy
                    month_entry.update({
                        "month_label_normalized": "Jan",
                        "derived_month_num": 1,
                        "derived_year": derived_year,
                        "year_token_raw": m_tok.text.strip(),
                    })
                pairs.append(month_entry)
        
        logger.warning(
            "TSV_V2_MONTHLY: %s RESULT month_pages=%s status_pages=%s matched=%d/%d examples=%s",
            bureau.upper(), month_pages, status_pages, matched_count, len(pairs), example_matches
        )

        # Enrich with derived_year and month_year_key after month-number normalization
        _assign_years_and_keys(pairs)
        
        result[bureau] = pairs
        logger.info(
            "TSV_V2_MONTHLY: bureau=%s output_pairs=%d first5=%s last5=%s",
            bureau, len(pairs),
            pairs[:5] if pairs else [],
            pairs[-5:] if pairs else []
        )
    
    # Validate invariant: len(monthly) == len(months)
    for bureau in ("transunion", "experian", "equifax"):
        monthly_count = len(result[bureau])
        months_count = len(tsv_v2_months.get(bureau, []))
        if monthly_count != months_count:
            logger.warning(
                "TSV_V2_MONTHLY: INVARIANT VIOLATION bureau=%s monthly=%d months=%d",
                bureau, monthly_count, months_count
            )
        else:
            logger.info(
                "TSV_V2_MONTHLY: invariant OK bureau=%s count=%d",
                bureau, monthly_count
            )
    
    logger.info(
        "TSV_V2_MONTHLY: extraction complete sid=%s heading=%s tu=%d ex=%d eq=%d",
        session_id, heading,
        len(result.get("transunion", [])),
        len(result.get("experian", [])),
        len(result.get("equifax", []))
    )
    
    return result
