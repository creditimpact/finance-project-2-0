from __future__ import annotations

import logging
import re
from typing import List, Dict, Tuple

logger = logging.getLogger(__name__)


SUMMARY_TITLES = {
    "TOTAL ACCOUNTS",
    "CLOSED OR PAID ACCOUNT/ZERO",
    "INQUIRIES",
    "PUBLIC INFORMATION",
    "COLLECTIONS",
    "PERSONAL INFORMATION",
    "SCORE FACTORS",
    "CREDIT SUMMARY",
    "ALERTS",
    "EMPLOYMENT DATA",
}

HEADING_RE = re.compile(r"^[A-Z0-9/&\- ]{3,}$")
# Accept: ACCOUNT # | ACCT # | ACCOUNT NO/NO./NUMBER | ACCT Nº/N° etc.
ACCT_LABEL_RE = re.compile(r"\b(?:ACCOUNT|ACCT)\s*#", re.I)

ADDRESS_TOKENS = [
    " ST",
    " STREET",
    " AVE",
    " AV",
    " AVENUE",
    " RD",
    " ROAD",
    " BLVD",
    " LANE",
    " LN",
    " DR",
    " DRIVE",
    " CT",
    " COURT",
    " PKWY",
    " WAY",
    " HWY",
    " HIGHWAY",
    " UNIT",
    " APT",
    " SUITE",
    " STE",
    " PO BOX",
]

# Explicit non-account headings to ban as issuer headlines
NON_ACCOUNT_HEADINGS = {
    "INDIVIDUAL",
    "JOINT",
    "AUTHORIZED USER",
    "BANK - MORTGAGE LOANS",
    "ACCOUNT NOT DISPUTED",
    "CURRENT",
    "CLOSED",
    "PAID",
    "PAST DUE",
    "OPEN",
    "CREDIT CARD",
    "INSTALLMENT",
    "REAL ESTATE",
    "AUTO LOAN",
    "®",
    "--",
    "-",
    "",
}


def _norm(s: str) -> str:
    """Normalize for section/trigger matching.

    - Replace NBSP/registration
    - Collapse spaces
    - Keep only A-Z0-9/&- and space
    - Uppercase
    """
    s = (s or "").replace("\u00A0", " ").replace("®", " ")
    s = re.sub(r"\s+", " ", s.strip())
    s = re.sub(r"[^A-Za-z0-9/&\- ]+", "", s)
    return s.upper()


def _norm_ws(s: str) -> str:
    """Normalize whitespace and common marks but keep degree symbols for label regex."""
    s = (s or "").replace("\u00A0", " ").replace("®", " ")
    s = re.sub(r"\s+", " ", s.strip())
    return s.upper()


def _is_summary_title(s: str) -> bool:
    return _norm(s) in SUMMARY_TITLES


def _looks_like_address(raw: str) -> bool:
    s = (raw or "").upper()
    has_digit = any(c.isdigit() for c in s)
    if not has_digit:
        return False
    return any(tok in s for tok in ADDRESS_TOKENS)


def _find_account_heading(lines: List[str], acct_line_idx: int, *, window: int = 3) -> Tuple[int | None, str | None]:
    """Return (heading_idx, heading_text) within 3 lines above acct_line_idx.

    Rejects address-like headings and certain non-account headings.
    """
    start = max(0, acct_line_idx - window)
    for j in range(acct_line_idx - 1, start - 1, -1):
        raw = lines[j].strip()
        if not raw:
            continue
        nrm = _norm(raw)
        # Skip bureau tri-header lines between heading and account label
        if ("TRANSUNION" in nrm and "EXPERIAN" in nrm and "EQUIFAX" in nrm):
            continue
        if nrm in {"NONE REPORTED", "CLOSED OR PAID ACCOUNT/ZERO"}:
            logger.info("BLOCK: reject heading as non-account heading=%r", raw)
            continue
        if nrm in NON_ACCOUNT_HEADINGS:
            logger.info("BLOCK: reject heading as non-account heading=%r reason=banlist", raw)
            continue
        if _looks_like_address(raw):
            logger.info("BLOCK: reject heading as address heading=%r", raw)
            continue
        # Accept only issuer-like headings per HEADING_RE after normalization
        if HEADING_RE.match(nrm):
            return j, raw
    return None, None


def _find_bureau_header_idxs(lines: List[str]) -> List[int]:
    """Return indices of lines that look like a bureau header (contain all three names)."""
    idxs: List[int] = []
    for i, ln in enumerate(lines):
        nrm = _norm(ln)
        if "TRANSUNION" in nrm and "EXPERIAN" in nrm and "EQUIFAX" in nrm:
            idxs.append(i)
    return idxs


def segment_account_blocks(full_text: str) -> List[Dict]:
    """Split full_text into deterministic account/summary blocks.

    Returns list of dicts { heading, lines, meta:{block_type, via, start_line, end_line} }.
    """
    if not full_text:
        return []
    lines = [ln.rstrip("\r\n") for ln in str(full_text).splitlines()]
    n = len(lines)

    # Locate explicit section markers
    idx_public = next((i for i, ln in enumerate(lines) if _norm(ln) == "PUBLIC INFORMATION"), None)
    idx_inq = next((i for i, ln in enumerate(lines) if _norm(ln) == "INQUIRIES"), None)
    idx_personal = next((i for i, ln in enumerate(lines) if _norm(ln) == "PERSONAL INFORMATION"), None)

    # Scan for account headings via Account # lines only (strict)
    candidates: List[Tuple[int, str, str, str]] = []  # (heading_idx, heading_text, via, validation)
    i = 0
    while i < n:
        ln = lines[i]
        if ACCT_LABEL_RE.search(_norm_ws(ln)):
            h_idx, h_text = _find_account_heading(lines, i, window=5)
            if h_idx is not None and h_text:
                # Flexible forward validation within 60 lines
                WINDOW_FWD = 60
                end_probe = min(n, h_idx + 1 + WINDOW_FWD)
                has_bureau_token = False
                has_core_label = False
                for s in lines[h_idx + 1 : end_probe]:
                    p = _norm(s)
                    if any(tok in p for tok in ("TRANSUNION", "EXPERIAN", "EQUIFAX")):
                        has_bureau_token = True
                        break
                    if re.match(r"^(DATE OPENED|ACCOUNT STATUS|PAYMENT STATUS|DATE REPORTED)", p):
                        has_core_label = True
                        break
                validation = "strong" if (has_bureau_token or has_core_label) else "weak"
                if validation == "weak":
                    logger.info(
                        "BLOCK: weak-validated account heading=%r reason=no_bureau_or_core_labels_within_60",
                        h_text,
                    )
                # Accept heading
                logger.info(
                    "BLOCK: accept account heading=%r via=prev_of_account_hash",
                    h_text,
                )
                candidates.append((h_idx, h_text, "prev_of_account_hash", validation))
                # Skip ahead a bit to avoid duplicate detection in the same block
                i = i + 1
                continue
        i += 1

    # No fallback: tri-header alone does not open an account block

    # Deduplicate by heading index and sort; prefer prev_of_account_hash when duplicated
    seen_indices: set[int] = set()
    starts: List[int] = []
    headings: Dict[int, str] = {}
    via_by_idx: Dict[int, str] = {}
    for h_idx, h_text, via, validation in sorted(candidates, key=lambda x: x[0]):
        if h_idx in seen_indices:
            # Upgrade via if we had bureau_header and now have prev_of_account_hash
            if via_by_idx.get(h_idx) != "prev_of_account_hash" and via == "prev_of_account_hash":
                via_by_idx[h_idx] = via
            continue
        seen_indices.add(h_idx)
        starts.append(h_idx)
        headings[h_idx] = h_text
        via_by_idx[h_idx] = via
        # Store validation in a side map
        # We piggyback via_by_idx for simplicity; meta will be enriched below
        # A separate dict keeps clarity
    validation_map: Dict[int, str] = {}
    for h_idx, h_text, via, validation in sorted(candidates, key=lambda x: x[0]):
        if h_idx in seen_indices:
            validation_map[h_idx] = validation

    blocks: List[Dict] = []

    # Optional Personal Information block from start to first account
    # Personal Information block from start to first account/public/inquiries
    p_start = 0
    personal_end_candidates = [n]
    if starts:
        personal_end_candidates.append(starts[0])
    if idx_public is not None:
        personal_end_candidates.append(idx_public)
    if idx_inq is not None:
        personal_end_candidates.append(idx_inq)
    p_end = min(personal_end_candidates)
    if p_end > p_start:
        blocks.append(
            {
                "heading": "Personal Information",
                "lines": lines[p_start:p_end],
                "meta": {
                    "block_type": "summary",
                    "via": "section_header",
                    "start_line": p_start + 1,
                    "end_line": p_end,
                },
            }
        )
        logger.info("BLOCK: open summary heading='Personal Information'")

    # Build account blocks with boundaries
    for idx, start in enumerate(starts):
        # End at next account start, or Public Information, or Inquiries, whichever comes first after start
        candidates_end = [n]
        if idx + 1 < len(starts):
            candidates_end.append(starts[idx + 1])
        if idx_public is not None and idx_public > start:
            candidates_end.append(idx_public)
        if idx_inq is not None and idx_inq > start:
            candidates_end.append(idx_inq)
        end = min(candidates_end)
        # Slice and optionally drop pure noise lines (only ® or --)
        raw_block_lines = lines[start:end]
        filtered_lines: List[str] = []
        for ln in raw_block_lines:
            t = (ln or "").strip()
            if t in {"®", "--", "-"}:
                # debug only; no-op otherwise
                continue
            filtered_lines.append(ln)
        if idx > 0:
            logger.info(
                "BLOCK: split account at new Account # heading=%r prev_heading=%r",
                headings.get(start, lines[start].strip()),
                headings.get(starts[idx - 1], lines[starts[idx - 1]].strip()),
            )
        blocks.append(
            {
                "heading": headings.get(start, lines[start].strip()),
                "lines": filtered_lines,
                "meta": {
                    "block_type": "account",
                    "via": via_by_idx.get(start, "prev_of_account_hash"),
                    "start_line": start + 1,
                    "end_line": end,
                    "validation": validation_map.get(start, "strong"),
                },
            }
        )

    # Public Information block
    if idx_public is not None:
        end_pub = idx_inq if idx_inq is not None and idx_inq > idx_public else n
        blocks.append(
            {
                "heading": "Public Information",
                "lines": lines[idx_public:end_pub],
                "meta": {
                    "block_type": "summary",
                    "via": "section_header",
                    "start_line": idx_public + 1,
                    "end_line": end_pub,
                },
            }
        )
        logger.info("BLOCK: open summary heading='Public Information'")

    # Inquiries block
    if idx_inq is not None:
        blocks.append(
            {
                "heading": "Inquiries",
                "lines": lines[idx_inq:n],
                "meta": {
                    "block_type": "summary",
                    "via": "section_header",
                    "start_line": idx_inq + 1,
                    "end_line": n,
                },
            }
        )
        logger.info("BLOCK: open summary heading='Inquiries'")

    return blocks


