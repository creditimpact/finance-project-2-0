"""
TSV v2: Simple Y-segmentation based 2Y months extraction.

Extracts Hebrew month abbreviations by bureau from the 2Y region,
using simple Y-coordinate segmentation (TU/EX/EQ bureau markers).

No status mapping, no 7Y interference; just months by bureau.
"""

import logging
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)

# Hebrew month abbreviations (as they appear in the document)
HEBREW_MONTHS = {
    "ינו׳", "פבר׳", "מרץ", "אפר׳", "מאי", "יוני",
    "יולי", "אוג׳", "ספט׳", "אוק׳", "נוב׳", "דצמ׳",
}


def extract_tsv_v2_months_by_bureau(
    session_id: str,
    heading: str,
    tokens_by_line: Dict[tuple, List[dict]],
    lines: List[dict],
) -> Optional[Dict[str, List[str]]]:
    """
    Extract 2Y months by bureau using Y-segmentation.
    
    Process:
    1. Find 2Y region: start = "Two-Year Payment History", end = "Seven Year" header
    2. Locate bureau markers: "Transunion", "Experian", "Equifax" (first occurrence in region)
    3. Slice by bureau (TU to EX, EX to EQ, EQ to 7Y)
    4. Extract Hebrew month tokens only from each slice
    
    Returns:
        Dict with keys 'transunion', 'experian', 'equifax' containing month lists.
        None if 2Y region not found or no months extracted.
    """
    if not tokens_by_line or not lines:
        return None
    
    # Step 1: Collect all tokens with page, line indices for easy lookup
    # Sort by page, line to maintain reading order
    all_tokens = []
    sorted_keys = sorted(tokens_by_line.keys())
    for (page, line) in sorted_keys:
        tok_list = tokens_by_line[(page, line)]
        for tok in tok_list:
            # Preserve location info
            tok_copy = dict(tok)
            tok_copy["_page"] = page
            tok_copy["_line"] = line
            all_tokens.append(tok_copy)
    
    if not all_tokens:
        logger.warning("TSV_V2: no tokens found sid=%s heading=%s", session_id, heading)
        return None
    
    # Step 2: Find 2Y region start by looking for "payment history" or similar
    # The text may be split across lines, so check line-by-line for key terms
    twoY_start_idx = None
    twoY_keywords = ["payment history", "history", "paymenthistory"]
    
    # First, look for obvious "Two-Year" or "two year" markers
    for i, tok in enumerate(all_tokens):
        text = tok.get("text", "").lower().strip()
        if "two" in text and ("year" in text or "payment" in text):
            twoY_start_idx = i
            logger.info("TSV_V2: 2Y region start found at idx=%d text='%s'", i, tok.get("text", ""))
            break
    
    # If not found by "two year", look for "history" token (common in payment history line)
    if twoY_start_idx is None:
        for i, tok in enumerate(all_tokens):
            text = tok.get("text", "").lower().strip()
            if "history" in text and len(text) > 3:
                twoY_start_idx = i
                logger.info("TSV_V2: 2Y region (by history) found at idx=%d text='%s'", i, tok.get("text", ""))
                break
    
    if twoY_start_idx is None:
        logger.warning("TSV_V2: 2Y region start not found sid=%s heading=%s", session_id, heading)
        return None
    
    # Step 3: Find 2Y region end (7Y header)
    sevenY_start_idx = None
    sevenY_keywords = ["seven", "year", "7y"]
    
    # Look for "seven" or "7" year markers
    for i in range(twoY_start_idx + 1, len(all_tokens)):
        text = all_tokens[i].get("text", "").lower().strip()
        if ("seven" in text or "7" in text) and ("year" in text or "history" in text or len(text) <= 2):
            sevenY_start_idx = i
            logger.info("TSV_V2: 7Y region start found at idx=%d text='%s'", i, all_tokens[i].get("text", ""))
            break
    
    if sevenY_start_idx is None:
        logger.warning("TSV_V2: 7Y region not found, using end of tokens sid=%s heading=%s", session_id, heading)
        sevenY_start_idx = len(all_tokens)
    
    # Extract 2Y region tokens
    two_year_tokens = all_tokens[twoY_start_idx:sevenY_start_idx]
    logger.info("TSV_V2: 2Y region range idx_start=%d idx_end=%d token_count=%d",
                twoY_start_idx, sevenY_start_idx, len(two_year_tokens))
    
    # Step 4: Find bureau markers in 2Y region
    bureau_indices = {}
    bureau_keywords = {
        "transunion": ["transunion", "tu"],
        "experian": ["experian", "ex"],
        "equifax": ["equifax", "eq"],
    }
    
    for bureau, keywords in bureau_keywords.items():
        for i, tok in enumerate(two_year_tokens):
            text = tok.get("text", "").lower().strip()
            if any(kw in text for kw in keywords):
                bureau_indices[bureau] = i
                logger.info("TSV_V2: bureau '%s' found at 2Y_idx=%d text='%s'",
                           bureau, i, tok.get("text", ""))
                break
    
    if not bureau_indices:
        logger.warning("TSV_V2: no bureau markers found in 2Y region sid=%s heading=%s", session_id, heading)
        return None
    
    # Step 5: Define slices by bureau order
    # Sort by index to determine order
    sorted_bureaus = sorted(bureau_indices.items(), key=lambda x: x[1])
    
    slices = {}
    for i, (bureau, idx) in enumerate(sorted_bureaus):
        if i == 0:
            # TU: from TU to next bureau (or end)
            next_idx = sorted_bureaus[1][1] if len(sorted_bureaus) > 1 else len(two_year_tokens)
            slices[bureau] = (idx, next_idx)
        elif i == 1:
            # EX: from EX to next bureau (or end)
            next_idx = sorted_bureaus[2][1] if len(sorted_bureaus) > 2 else len(two_year_tokens)
            slices[bureau] = (idx, next_idx)
        else:
            # EQ: from EQ to end
            slices[bureau] = (idx, len(two_year_tokens))
    
    # Step 6: Extract Hebrew months from each slice
    result = {"transunion": [], "experian": [], "equifax": []}
    
    for bureau, (start_idx, end_idx) in slices.items():
        slice_tokens = two_year_tokens[start_idx:end_idx]
        months = []
        
        for tok in slice_tokens:
            text = tok.get("text", "").strip()
            if text in HEBREW_MONTHS:
                months.append(text)
        
        result[bureau] = months
        logger.info("TSV_V2: slice '%s' idx_range=(%d, %d) extracted_months=%d samples=%s",
                   bureau, start_idx, end_idx, len(months), months[:3] if months else [])
    
    total_months = sum(len(m) for m in result.values())
    if total_months == 0:
        logger.warning("TSV_V2: no months extracted sid=%s heading=%s", session_id, heading)
        return None
    
    logger.info("TSV_V2: extraction complete sid=%s heading=%s tu=%d ex=%d eq=%d",
               session_id, heading, len(result.get("transunion", [])),
               len(result.get("experian", [])), len(result.get("equifax", [])))
    
    return result
