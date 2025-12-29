"""Utilities for parsing credit report PDFs into text and sections."""

import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence, cast
from .text_provider import load_cached_text

logger = logging.getLogger(__name__)


def _save_parsed_triples(
    sid: str | None, account_id: str | None, heading: str | None, payload: dict
) -> None:
    if not sid or not account_id or not payload:
        return
    out_dir = os.path.join("traces", sid, "debug", "parsed_triples")
    os.makedirs(out_dir, exist_ok=True)
    safe_id = account_id.replace("/", "_")
    fn = os.path.join(out_dir, f"{safe_id}.json")
    try:
        with open(fn, "w", encoding="utf-8") as f:
            json.dump({"heading": heading, **payload}, f, ensure_ascii=False, indent=2)
        logger.info(
            "PARSEDBG: saved parsed_triples for account=%s file=%s", account_id, fn
        )
    except Exception as e:  # pragma: no cover - debug helper
        logger.warning("PARSEDBG: failed to save for account=%s err=%s", account_id, e)


def _clean_line(s: str) -> str:
    """Remove registration symbols and collapse excess whitespace."""
    s = s.replace("Â", "").replace("®", "").replace("™", "")
    s = re.sub(r"\s+", " ", s.strip())
    return s


BUREAU_NAME_PATTERN = r"(Trans\s*Union|Experian|Equifax)"
BUREAU_PATTERNS = {
    "Transunion": re.compile(r"trans\s*union", re.I),
    "Experian": re.compile(r"experian", re.I),
    "Equifax": re.compile(r"equifax", re.I),
}


def detect_bureau_order(lines: list[str]) -> list[str] | None:
    """Return bureau order from ``lines`` if all three are found.

    Works on cleaned, case-insensitive lines and can handle headers that span
    up to two consecutive lines. Any occurrence of ``TransUnion``/``Trans Union``,
    ``Experian`` and ``Equifax`` is considered regardless of additional
    characters. Returns the canonical lowercase bureau names in the detected
    order, or ``None`` if no header is found.
    """

    # Clean the lines to remove stray characters such as ``Â`` or ``®``
    cleaned = [_clean_line(ln) for ln in lines]

    pats = {name.lower(): pat for name, pat in BUREAU_PATTERNS.items()}

    for i in range(len(cleaned)):
        candidates = [cleaned[i]]
        if i + 1 < len(cleaned):
            candidates.append(f"{cleaned[i]} {cleaned[i + 1]}")
        for cand in candidates:
            positions: dict[str, int] = {}
            for name, pat in pats.items():
                m = pat.search(cand)
                if m:
                    positions[name] = m.start()
            if len(positions) == 3:
                ordered = [
                    name for name, _ in sorted(positions.items(), key=lambda x: x[1])
                ]
                return ordered
    return None


# --- Standard 25-field set per bureau ---
ACCOUNT_FIELD_SET: tuple[str, ...] = (
    "account_number_display",
    "account_number_last4",
    "high_balance",
    "last_verified",
    "date_of_last_activity",
    "date_reported",
    "date_opened",
    "balance_owed",
    "closed_date",
    "account_rating",
    "account_description",
    "dispute_status",
    "creditor_type",
    "original_creditor",
    "account_status",
    "payment_status",
    "creditor_remarks",
    "payment_amount",
    "last_payment",
    "term_length",
    "past_due_amount",
    "account_type",
    "payment_frequency",
    "credit_limit",
    "two_year_payment_history",
    "seven_year_days_late",
)


# --- Alias mapping (OCR/headers -> standard keys) ---
ALIAS_TO_STD: dict[str, str] = {
    # Numbers / amounts
    "balance": "balance_owed",
    "balance owed": "balance_owed",
    "current_balance": "balance_owed",
    "current balance": "balance_owed",
    "amount": "balance_owed",
    "amt": "balance_owed",
    "high balance": "high_balance",
    "original amount": "high_balance",
    "past_due": "past_due_amount",
    "past due": "past_due_amount",
    "past due amount": "past_due_amount",
    "amount past due": "past_due_amount",
    "pmt_amount": "payment_amount",
    "payment amount": "payment_amount",
    "monthly payment": "payment_amount",
    "scheduled payment": "payment_amount",
    "last_pmt": "last_payment",
    "last payment": "last_payment",
    "last payment date": "last_payment",
    "limit": "credit_limit",
    "credit limit": "credit_limit",
    "h/c": "credit_limit",
    "hc": "credit_limit",
    "high credit": "credit_limit",
    # Dates
    "last_reported": "date_reported",
    "reported_date": "date_reported",
    "date reported": "date_reported",
    "reported": "date_reported",
    "last reported": "date_reported",
    "opened": "date_opened",
    "date opened": "date_opened",
    "open date": "date_opened",
    "closed": "closed_date",
    "closed date": "closed_date",
    "date closed": "closed_date",
    "dla": "date_of_last_activity",
    "dola": "date_of_last_activity",
    "date of last activity": "date_of_last_activity",
    "last activity": "date_of_last_activity",
    "verified": "last_verified",
    "last verified": "last_verified",
    "verified on": "last_verified",
    # Account number / last4
    "account #": "account_number_display",
    "account#": "account_number_display",
    "acct #": "account_number_display",
    "acct#": "account_number_display",
    "account_no": "account_number_display",
    "account no": "account_number_display",
    "account no.": "account_number_display",
    "acct no": "account_number_display",
    "acct no.": "account_number_display",
    "account number": "account_number_display",
    "acct number": "account_number_display",
    "account_last4": "account_number_last4",
    "acct_last4": "account_number_last4",
    "last4": "account_number_last4",
    # Status & descriptions
    "account status": "account_status",
    "status": "account_status",
    "current status": "account_status",
    "rating": "account_status",
    "payment status": "payment_status",
    "pay status": "payment_status",
    "status detail": "payment_status",
    "current": "payment_status",
    "account description": "account_description",
    "description": "account_description",
    "desc": "account_description",
    "dispute status": "dispute_status",
    "dispute": "dispute_status",
    "dispute flag": "dispute_status",
    "account dispute status": "dispute_status",
    "original creditor": "original_creditor",
    "original creditor 01": "original_creditor",
    "original creditor 02": "original_creditor",
    "originalcreditor": "original_creditor",
    "orig creditor": "original_creditor",
    "orig. creditor": "original_creditor",
    # Remarks
    "creditor remarks": "creditor_remarks",
    "remarks": "creditor_remarks",
    "comment": "creditor_remarks",
    "comments": "creditor_remarks",
    "notes": "creditor_remarks",
    # Types
    "creditor type": "creditor_type",
    "creditor category": "creditor_type",
    "creditor": "creditor_type",
    "account type": "account_type",
    "type": "account_type",
    "term length": "term_length",
    "term": "term_length",
    "terms": "term_length",
    "loan term": "term_length",
    "contract length": "term_length",
    "payment frequency": "payment_frequency",
    "frequency": "payment_frequency",
    "two-year payment history": "two_year_payment_history",
    "two year payment history": "two_year_payment_history",
    "2-year payment history": "two_year_payment_history",
    "two-year history": "two_year_payment_history",
    "2 year hist": "two_year_payment_history",
    "days late - 7 year history": "seven_year_days_late",
    "7-year days late": "seven_year_days_late",
    "7 year late summary": "seven_year_days_late",
    "late - 7 yrs": "seven_year_days_late",
    # Other shorthand
    "freq": "payment_frequency",
}

ACCOUNT_NUMBER_ALIASES = {
    k for k, v in ALIAS_TO_STD.items() if v == "account_number_display"
}

ACCOUNT_NUMERIC_KEY_RE = re.compile(
    r"^(account|acct)\s*(?:number|no|#)?\s*[0-9xX\*-]+$",
    re.I,
)


NUMERIC_FIELDS = {
    "high_balance",
    "balance_owed",
    "credit_limit",
    "past_due_amount",
    "payment_amount",
}

DATE_FIELDS = {
    "date_opened",
    "date_reported",
    "closed_date",
    "last_verified",
    "last_payment",
    "date_of_last_activity",
}


def _std_field_name(raw_key: str) -> str:
    k = (raw_key or "").strip().lower()
    k = re.sub(r"\s+", " ", k)
    if ACCOUNT_NUMERIC_KEY_RE.match(k):
        logger.info("alias_norm: key='account #' matched from raw='%s'", raw_key)
        k = "account #"
    return ALIAS_TO_STD.get(k, k)


PREFIX_STRIP_RE = re.compile(
    r"^(?:payment\s*status|account\s*status|creditor\s*remarks)\s*:\s*",
    re.I,
)


def _strip_leaked_prefix(key_norm: str, val: str | None) -> str | None:
    if not val:
        return val
    fam = None
    if key_norm in ("payment_status", "account_type"):
        fam = "payment"
    elif key_norm in ("account_status",):
        fam = "account"
    elif key_norm in ("creditor_remarks",):
        fam = "creditor"
    else:
        return val

    s = val.strip()
    if fam == "payment" and s.lower().startswith("payment"):
        s = PREFIX_STRIP_RE.sub("", s).strip()
    elif fam == "account" and s.lower().startswith("account"):
        s = PREFIX_STRIP_RE.sub("", s).strip()
    elif fam == "creditor" and s.lower().startswith("creditor"):
        s = PREFIX_STRIP_RE.sub("", s).strip()
    return s


def _assign_std(
    dst: dict[str, Any],
    key: str,
    val: Any,
    *,
    raw_val: Any | None = None,
    provenance: (
        Literal["aligned", "fallback", "footer", "collection", "unknown"] | None
    ) = None,
    bureau: str | None = None,
) -> None:
    """Assign *val* to ``dst`` under the canonical field name for ``key``.

    Values are stored as ``{"raw", "normalized", "provenance"}`` so the
    original string is preserved even when normalisation fails.
    """

    std = _std_field_name(key)
    if std not in ACCOUNT_FIELD_SET:
        return

    prov = provenance or "unknown"

    # Accept pre-structured values (already carrying provenance)
    if isinstance(val, Mapping) and {"raw", "normalized"} <= set(val.keys()):
        raw = val.get("raw")
        norm = val.get("normalized")
        prov = val.get("provenance", prov)
        abbr = {"transunion": "tu", "experian": "ex", "equifax": "eq"}
        b = abbr.get(bureau or "", bureau)
        logger.info(
            "CELL: key=%s prov=%s raw=%r norm=%r bureau=%s",
            std,
            prov,
            raw,
            norm,
            b,
        )
        dst[std] = {"raw": raw, "normalized": norm, "provenance": prov}
        return

    raw = raw_val if raw_val is not None else val
    normalized: Any | None = None

    if val is not None:
        if std == "account_number_display":
            val = re.sub(r"^[\-\s]+", "", str(val or ""))
        if std in NUMERIC_FIELDS:
            num = to_number(str(val))
            normalized = num if isinstance(num, (int, float)) else None
        elif std in DATE_FIELDS:
            iso = to_iso_date(str(val))
            normalized = iso if re.match(r"\d{4}-\d{2}-\d{2}", str(iso)) else None
        else:
            normalized = str(val).strip()

    if normalized is None and raw not in (None, "") and val is not None:
        logger.info("NORM: failed key=%s raw=%r prov=%s", std, raw, prov)

    abbr = {"transunion": "tu", "experian": "ex", "equifax": "eq"}
    b = abbr.get(bureau or "", bureau)
    logger.info(
        "CELL: key=%s prov=%s raw=%r norm=%r bureau=%s",
        std,
        prov,
        raw,
        normalized,
        b,
    )

    dst[std] = {"raw": raw, "normalized": normalized, "provenance": prov}




_PAYMENT_MARK_RE = re.compile(r"payment\s*status", re.I)
_CREDITOR_MARK_RE = re.compile(r"creditor\s*remarks?", re.I)
_ACCOUNT_MARK_RE = re.compile(r"account\s*status|account\s*description", re.I)


def scan_page_markers(page_texts: Sequence[str]) -> dict[str, Any]:
    """Scan ``page_texts`` for marker strings and return a summary dict."""
    pages_payment_status: list[int] = []
    pages_creditor_remarks: list[int] = []
    pages_account_status: list[int] = []
    for idx, text in enumerate(page_texts, start=1):
        if _PAYMENT_MARK_RE.search(text):
            pages_payment_status.append(idx)
        if _CREDITOR_MARK_RE.search(text):
            pages_creditor_remarks.append(idx)
        if _ACCOUNT_MARK_RE.search(text):
            pages_account_status.append(idx)
    return {
        "has_payment_status": bool(pages_payment_status),
        "has_creditor_remarks": bool(pages_creditor_remarks),
        "has_account_status": bool(pages_account_status),
        "pages_payment_status": pages_payment_status,
        "pages_creditor_remarks": pages_creditor_remarks,
        "pages_account_status": pages_account_status,
    }


from backend.core.logic.utils.names_normalization import (  # noqa: E402
    normalize_bureau_name,
)
from backend.core.logic.utils.norm import normalize_heading  # noqa: E402
from backend.core.logic.utils.text_parsing import extract_account_blocks  # noqa: E402
from backend.core.models.bureau import BureauAccount  # noqa: E402

from .constants import BUREAUS, INQUIRY_FIELDS, PUBLIC_INFO_FIELDS  # noqa: E402
from .normalize import to_iso_date, to_number  # noqa: E402

# Mapping of account detail labels to canonical keys. Each tuple contains the
# canonical key and a regex that matches variations of the label in the PDF
# table. The parser is case/space tolerant.
_DETAIL_LABELS: list[tuple[str, re.Pattern[str]]] = [
    ("account_number", re.compile(r"(?:account|acct)\s*(?:#|number|no\.?)", re.I)),
    ("high_balance", re.compile(r"high\s*balance", re.I)),
    ("last_verified", re.compile(r"last\s*verified", re.I)),
    ("date_of_last_activity", re.compile(r"date\s*of\s*last\s*activity", re.I)),
    ("date_reported", re.compile(r"date\s*reported", re.I)),
    ("date_opened", re.compile(r"date\s*opened", re.I)),
    ("balance_owed", re.compile(r"balance\s*owed", re.I)),
    ("closed_date", re.compile(r"closed\s*date|date\s*closed", re.I)),
    ("account_rating", re.compile(r"account\s*rating", re.I)),
    ("account_description", re.compile(r"account\s*description", re.I)),
    ("dispute_status", re.compile(r"dispute\s*status", re.I)),
    ("creditor_type", re.compile(r"creditor\s*type", re.I)),
    ("original_creditor", re.compile(r"original\s*creditor", re.I)),
    ("account_status", re.compile(r"account\s*status", re.I)),
    ("payment_status", re.compile(r"payment\s*status", re.I)),
    ("creditor_remarks", re.compile(r"creditor\s*remarks?", re.I)),
    ("payment_amount", re.compile(r"payment\s*amount", re.I)),
    ("last_payment", re.compile(r"last\s*payment", re.I)),
    ("term_length", re.compile(r"term\s*length", re.I)),
    ("past_due_amount", re.compile(r"past\s*due\s*amount", re.I)),
    ("account_type", re.compile(r"account\s*type", re.I)),
    ("payment_frequency", re.compile(r"payment\s*frequency", re.I)),
    ("credit_limit", re.compile(r"credit\s*limit", re.I)),
]

_MONEY_FIELDS = {
    "high_balance",
    "balance_owed",
    "credit_limit",
    "past_due_amount",
    "payment_amount",
}

_DATE_FIELDS = {
    "date_opened",
    "closed_date",
    "date_reported",
    "last_payment",
    "last_verified",
    "date_of_last_activity",
}


def _normalize_date(value: str) -> str | None:
    """Normalize various date formats to ``YYYY-MM`` or ``YYYY-MM-DD``."""

    value = value.strip()
    if not value:
        return None
    from datetime import datetime

    fmts_day = ["%m/%d/%Y", "%m/%d/%y", "%Y-%m-%d", "%b %d %Y", "%B %d %Y"]
    fmts_month = ["%m/%Y", "%Y-%m", "%b %Y", "%B %Y"]
    for fmt in fmts_day:
        try:
            dt = datetime.strptime(value, fmt)
            return dt.strftime("%Y-%m-%d")
        except ValueError:
            pass
    for fmt in fmts_month:
        try:
            dt = datetime.strptime(value, fmt)
            return dt.strftime("%Y-%m")
        except ValueError:
            pass
    return None


def _normalize_detail_value(key: str, value: str) -> tuple[Any | None, str | None]:
    """Return normalized value and raw string for ``key``."""

    raw = value.strip()
    if not raw:
        return None, None
    if key == "account_number":
        cleaned = re.sub(r"\s+", "", raw)
        if not re.search(r"\d", cleaned):
            return None, None
        return cleaned, raw
    if key in _MONEY_FIELDS:
        digits = re.sub(r"[^0-9]", "", raw)
        if not digits:
            return None, raw
        return int(digits), raw
    if key in _DATE_FIELDS:
        norm = _normalize_date(raw)
        return norm, raw
    return raw, raw




def extract_three_column_fields(
    pdf_path: str | Path,
    session_id: str | None = None,
) -> tuple[
    dict[str, dict[str, str]],
    dict[str, dict[str, str]],
    dict[str, dict[str, str]],
    dict[str, str],
    dict[str, str],
    dict[str, str],
    dict[str, dict[str, dict[str, Any]]],
]:
    """Return empty structures; PDF reopening removed.

    A ``session_id`` may be supplied for future extensions that operate on
    cached text, but is currently unused.
    """

    if session_id:
        try:
            _ = load_cached_text(session_id)
        except Exception:
            pass

    return {}, {}, {}, {}, {}, {}, {}

def bureau_data_from_dict(
    data: Mapping[str, list[dict[str, Any]]],
) -> Mapping[str, list[BureauAccount]]:
    """Convert raw bureau ``data`` to typed ``BureauAccount`` objects.

    Parameters
    ----------
    data:
        Mapping of section name to list of account dictionaries.

    Returns
    -------
    dict[str, list[BureauAccount]]
        Mapping with the same keys but ``BureauAccount`` instances as values.
    """
    result: dict[str, list[BureauAccount]] = {}
    for section, items in data.items():
        if isinstance(items, list):
            result[section] = [BureauAccount.from_dict(it) for it in items]
    return result


PAYMENT_STATUS_RE = re.compile(r"payment status:\s*(.+)", re.I)
CREDITOR_REMARKS_RE = re.compile(r"creditor remarks:\s*(.+)", re.I)

# ---------------------------------------------------------------------------
# Payment status parsing
# ---------------------------------------------------------------------------

# Account number extraction
ACCOUNT_NUMBER_ROW_RE = re.compile(
    r"(?:Account\s*(?:#|Number|No\.?)|Acct\s*(?:#|No\.?))\s*:?\s*(?P<tu>.+?)\s{2,}(?P<ex>.+?)\s{2,}(?P<eq>.+?)(?:\n|$)",
    re.I | re.S,
)
ACCOUNT_NUMBER_LINE_RE = re.compile(
    r"(?:Account\s*(?:#|Number|No\.?)|Acct\s*(?:#|No\.?))\s*:?\s*(.+)",
    re.I,
)


def _normalize_account_number(value: str) -> str | None:
    """Return a cleaned ``value`` if it contains at least one digit.

    The normalization removes whitespace and dashes while preserving any mask
    characters such as ``*``. If no digits are present the function returns
    ``None`` so callers can skip storing meaningless placeholders like
    ``"t disputed"``.
    """

    value = value.strip()
    if not re.search(r"\d", value):
        return None
    return re.sub(r"[\s-]", "", value)


def extract_payment_statuses(
    text: str,
) -> tuple[dict[str, dict[str, str]], dict[str, str]]:
    """Extract ``Payment Status`` lines for each bureau section.

    The function detects the bureau column boundaries from the header row in
    each account block and then slices the single ``Payment Status`` line into
    individual bureau values. The extracted values are normalized to lowercase
    with collapsed internal whitespace. A raw fallback of the right-hand side
    of the ``Payment Status`` line is also returned.

    Parameters
    ----------
    text:
        Raw text from the SmartCredit report.

    Returns
    -------
    tuple[dict[str, dict[str, str]], dict[str, str]]
        Two mappings: ``payment_statuses_by_heading`` and
        ``payment_status_raw_by_heading``.
    """

    def _normalize_val(value: str) -> str:
        return re.sub(r"\s+", " ", value.strip()).lower()

    statuses: dict[str, dict[str, str]] = {}
    raw_map: dict[str, str] = {}
    for block in extract_account_blocks(text):
        if not block:
            continue
        heading = block[0].strip()
        acc_norm = normalize_heading(heading)

        bureau_order = detect_bureau_order(block[1:])
        logger.info(
            "parse_account_block bureau_header_detected=%s order=%s",
            bool(bureau_order),
            bureau_order,
        )
        ps_line: str | None = None
        for raw_line in block[1:]:
            clean_line = _clean_line(raw_line)
            if re.search(r"payment\s*status", clean_line, re.I):
                ps_line = raw_line
                break
        if ps_line:
            raw_match = re.search(r"Payment\s*Status\s*:?(.*)", ps_line, re.I)
            if raw_match:
                rhs_raw = raw_match.group(1)
                rhs_clean = rhs_raw.replace("Â", "").replace("®", "")
                raw_map[acc_norm] = rhs_clean.strip()
                if bureau_order:
                    rhs = rhs_clean.strip()
                    parts = re.split(r"\s{2,}", rhs)
                    parts += ["", "", ""]
                    vals = {}
                    for bureau, part in zip(bureau_order, parts):
                        norm_val = _normalize_val(part)
                        if norm_val:
                            vals[bureau] = norm_val
                    if vals:
                        statuses[acc_norm] = vals

        # Additional per-bureau lines may specify payment status individually
        current_bureau: str | None = None
        for line in block[1:]:
            clean = _clean_line(line)
            if sum(1 for pat in BUREAU_PATTERNS.values() if pat.search(clean)) > 1:
                current_bureau = None
                continue
            bureau_match = re.match(rf"{BUREAU_NAME_PATTERN}\b", clean, re.I)
            if bureau_match:
                current_bureau = normalize_bureau_name(bureau_match.group(1)).title()
                ps_inline = PAYMENT_STATUS_RE.search(clean)
                if ps_inline:
                    statuses.setdefault(acc_norm, {})[current_bureau] = _normalize_val(
                        ps_inline.group(1)
                    )
                continue

            if current_bureau and not re.search(r"payment\s*status", clean, re.I):
                ps = PAYMENT_STATUS_RE.match(clean)
                if ps:
                    statuses.setdefault(acc_norm, {})[current_bureau] = _normalize_val(
                        ps.group(1)
                    )

    return statuses, raw_map


def extract_account_numbers(text: str) -> dict[str, dict[str, str]]:
    """Extract account numbers for each bureau section.

    Parameters
    ----------
    text:
        Raw text from the SmartCredit report.

    Returns
    -------
    dict[str, dict[str, str]]
        Mapping of normalized account names to ``bureau -> account_number``.
    """

    numbers: dict[str, dict[str, str]] = {}
    for block in extract_account_blocks(text):
        if not block:
            continue
        heading = block[0].strip()
        acc_norm = normalize_heading(heading)

        block_text = "\n".join(block[1:])
        row = ACCOUNT_NUMBER_ROW_RE.search(block_text)
        if row:
            tu = _normalize_account_number(row.group("tu"))
            ex = _normalize_account_number(row.group("ex"))
            eq = _normalize_account_number(row.group("eq"))
            if tu:
                numbers.setdefault(acc_norm, {})[
                    normalize_bureau_name("TransUnion")
                ] = tu
            if ex:
                numbers.setdefault(acc_norm, {})[normalize_bureau_name("Experian")] = ex
            if eq:
                numbers.setdefault(acc_norm, {})[normalize_bureau_name("Equifax")] = eq

        current_bureau: str | None = None
        for line in block[1:]:
            clean = _clean_line(line)
            bureau_match = re.match(rf"{BUREAU_NAME_PATTERN}\b", clean, re.I)
            if bureau_match:
                current_bureau = normalize_bureau_name(bureau_match.group(1))
                # Bureau line itself might contain the account number
                inline = ACCOUNT_NUMBER_LINE_RE.search(clean)
                if inline:
                    value = _normalize_account_number(inline.group(1))
                    if value:
                        numbers.setdefault(acc_norm, {})[current_bureau] = value
                continue

            if current_bureau:
                m = ACCOUNT_NUMBER_LINE_RE.match(clean)
                if m:
                    value = _normalize_account_number(m.group(1))
                    if value:
                        numbers.setdefault(acc_norm, {})[current_bureau] = value

    return numbers


def extract_creditor_remarks(text: str) -> dict[str, dict[str, str]]:
    """Extract ``Creditor Remarks`` lines for each bureau section.

    Parameters
    ----------
    text:
        Raw text from the SmartCredit report.

    Returns
    -------
    dict[str, dict[str, str]]
        Mapping of normalized account names to a mapping of
        ``bureau -> remarks`` strings.
    """

    remarks: dict[str, dict[str, str]] = {}
    for block in extract_account_blocks(text):
        if not block:
            continue
        heading = block[0].strip()
        acc_norm = normalize_heading(heading)
        current_bureau: str | None = None
        for line in block[1:]:
            clean = _clean_line(line)
            bureau_match = re.match(rf"{BUREAU_NAME_PATTERN}\b", clean, re.I)
            if bureau_match:
                current_bureau = normalize_bureau_name(bureau_match.group(1)).title()
                # If the bureau line itself contains remarks
                rem_inline = CREDITOR_REMARKS_RE.search(clean)
                if rem_inline:
                    remarks.setdefault(acc_norm, {})[current_bureau] = rem_inline.group(
                        1
                    ).strip()
                continue

            if current_bureau:
                rem = CREDITOR_REMARKS_RE.match(clean)
                if rem:
                    remarks.setdefault(acc_norm, {})[current_bureau] = rem.group(
                        1
                    ).strip()

    return remarks


# ---------------------------------------------------------------------------
# Bureau meta tables for accounts (raw.account_history.by_bureau)
# ---------------------------------------------------------------------------


def _ensure_paths(obj: dict, *path: str) -> dict:
    """Ensure nested dictionaries exist for ``path`` and return the final node."""

    cur: dict = obj
    for key in path:
        nxt = cur.get(key)
        if not isinstance(nxt, dict):
            nxt = {}
            cur[key] = nxt
        cur = nxt
    return cur


def _empty_bureau_map() -> dict[str, Any]:
    """Return a single-bureau map with the 25-field set all ``None``."""

    return {field: None for field in ACCOUNT_FIELD_SET}


def _to_num(s: str) -> Any | None:
    s = (s or "").strip()
    if s in {"--", "-", ""}:
        return None
    return to_number(s)


def _to_iso(s: str) -> Any | None:
    s = (s or "").strip()
    if s in {"--", "-", ""}:
        return None
    return to_iso_date(s)


SUMMARY_HEADING_TOKENS = (
    "total accounts",
    "open accounts",
    "closed accounts",
    "delinquent",
    "derogatory",
    "balances",
)

SUMMARY_REQUIRED_WORDS = (
    "account",
    "acct",
    "high balance",
    "date",
    "balance",
    "remarks",
)


def is_summary_block(block_lines: list[str]) -> bool:
    """Heuristics to exclude dashboard/summary blocks (e.g., 'Total Accounts 10 9 7', 'Open Accounts: ...').
    Return True if the block is a summary, False otherwise.
    """

    if not block_lines:
        return False

    heading = _clean_line(block_lines[0] or "").lower()

    if any(tok in heading for tok in SUMMARY_HEADING_TOKENS):
        return True

    if not any(word in heading for word in SUMMARY_REQUIRED_WORDS):
        cleaned = [_clean_line(ln) for ln in block_lines]
        has_bureau = any(any(b in ln.lower() for b in BUREAUS) for ln in cleaned)
        if not has_bureau:
            return True

    return False


def build_block_fuzzy(blocks: list[Mapping[str, Any]]) -> dict[str, list[str]]:
    """Return mapping of normalized headings to their OCR lines.

    Parameters
    ----------
    blocks:
        List of blocks as produced by ``extract_account_blocks`` with
        ``{"heading": <raw heading>, "lines": [...]}`` entries.

    Returns
    -------
    dict[str, list[str]]
        Mapping of multiple fuzzy-normalized heading variants to the
        corresponding list of lines for quick lookup.
    """

    try:  # local import to avoid heavy dependencies at module import time
        from backend.core.logic.utils.names_normalization import normalize_creditor_name
    except Exception:  # pragma: no cover - should not happen

        def normalize_creditor_name(s: str) -> str:  # type: ignore
            return re.sub(r"\s+", " ", (s or "").lower()).strip()

    def _alnum(s: str) -> str:
        return re.sub(r"[^a-z0-9]", "", s)

    mapping: dict[str, list[str]] = {}
    for blk in blocks or []:
        heading = blk.get("heading") or ""
        lines = list(blk.get("lines") or [])
        base = normalize_creditor_name(heading)
        if not base:
            continue
        words = base.split()
        keys = {base, _alnum(base)}
        if words:
            keys.add(words[0])
        if len(words) > 1:
            keys.add(" ".join(words[:2]))
        for k in filter(None, keys):
            mapping.setdefault(k, lines)
    return mapping


def _find_block_lines_for_account(
    sections: Mapping[str, Any], account: Mapping[str, Any] | str
) -> list[str]:
    """Return OCR lines for the block matching ``account``.

    Parameters
    ----------
    sections:
        Mapping returned by :func:`analyze_report` which must include a
        ``"blocks_by_account_fuzzy"`` entry.
    account:
        Account mapping containing ``normalized_name`` or a raw/normalized
        creditor name string.

    Returns
    -------
    list[str]
        List of lines for the matched block.  Returns an empty list when no
        suitable block is found or on error.
    """

    try:
        if isinstance(account, Mapping):
            raw_name = account.get("normalized_name") or account.get("name") or ""
        else:
            raw_name = str(account or "")

        try:  # normalize in a way consistent with account detection
            from backend.core.logic.utils.names_normalization import (
                normalize_creditor_name,
            )
        except Exception:  # pragma: no cover - fallback minimal normalizer

            def normalize_creditor_name(s: str) -> str:  # type: ignore
                return re.sub(r"\s+", " ", (s or "").lower()).strip()

        norm = normalize_creditor_name(raw_name)

        mapping = sections.get("blocks_by_account_fuzzy") or {}
        if not isinstance(mapping, Mapping) or not mapping:
            logger.warning("blocks_by_account_fuzzy_missing")
            return []

        lines = mapping.get(norm)
        if lines:
            block_lines = list(lines)
            if is_summary_block(block_lines):
                logger.info(
                    "parse_skip summary_block heading=%r",
                    block_lines[0] if block_lines else "",
                )
                return []
            logger.debug(
                "block_lines_found name=%s method=exact lines=%d",
                norm,
                len(block_lines),
            )
            return block_lines

        try:
            from .analyze_report import _fuzzy_match  # type: ignore

            choices = set(mapping.keys())
            match = _fuzzy_match(norm, choices)
            if match:
                from rapidfuzz import fuzz as _fuzz

                score = _fuzz.WRatio(norm, match) / 100.0
                if score >= 0.9 and mapping.get(match):
                    lines = mapping[match]
                    block_lines = list(lines)
                    if is_summary_block(block_lines):
                        logger.info(
                            "parse_skip summary_block heading=%r",
                            block_lines[0] if block_lines else "",
                        )
                        return []
                    logger.debug(
                        "block_lines_found name=%s method=fuzzy match=%s lines=%d",
                        norm,
                        match,
                        len(block_lines),
                    )
                    return block_lines
        except Exception:
            pass

        logger.info("no_block_lines_for_account name=%s", norm)
        return []
    except Exception:  # pragma: no cover - defensive
        logger.exception("find_block_lines_failed")
        return []


BUREAU_LINE_RE = re.compile(
    r"^(Trans\s*Union|Experian|Equifax)\s+([0-9\*]+)\s+([\d,]+|0|--)\s+([-\d/]+|--)\s+([-\d/]+|--)\s+([-\d/]+|--)\s+([-\d/]+|--)\s+([\d,]+|0|--)\s+--?\s+(\w+)\s+(.*?)\s+(Bank|All Banks|National.*|.*?)\s+(.*?)\s+(Current|Late|.*?)(?:\s+--)?\s+(\d+|0|--)\s+([-\d/]+|--)\s+--?\s+(\d+|0|--)",
    re.I,
)

TRIPLE_LINE_RE = re.compile(
    r"^(?P<key>[^:]{2,}):\s*"
    r"(?P<v1>--|.+?)"
    r"(?:\s{2,}(?P<v2>--|.+?))?"
    r"(?:\s{2,}(?P<v3>--|.+))?$"
)


def _split_triple_fallback(
    value_part: str, bureau_order: Sequence[str] | None = None
) -> tuple[str | None, str | None, str | None]:
    """Heuristically split *value_part* into three columns when alignment is missing.

    The function attempts a best-effort split of ``value_part`` into three
    bureau-specific values.  It first tries strong separators (two or more
    spaces or ``|``).  If those fail, it applies a set of heuristics to bucket a
    stream of tokens without breaking multi-word values.  ``bureau_order`` is
    only used to ensure a deterministic default order and to emphasise that the
    returned tuple corresponds to that order.
    """

    _order = list(bureau_order or ["transunion", "experian", "equifax"])

    s = re.sub(r"\s+", " ", value_part.strip())

    # --- 1) strong separator split -------------------------------------------------
    parts = [p.strip() for p in re.split(r"\s{2,}|\s+\|\s+", s) if p is not None]
    used_heuristic = False
    if len(parts) < 2:
        used_heuristic = True

        date_re = re.compile(r"^\d{1,2}[./-]\d{1,2}[./-]\d{2,4}$")
        money_re = re.compile(r"^\$?\d[\d,]*(?:\.\d+)?$")
        status_phrases = {
            "open",
            "closed",
            "current",
            "collection",
            "charge off",
            "chargeoff",
            "paid",
            "paid in full",
            "settled",
            "repossession",
            "foreclosure",
        }

        tokens = s.split()

        def bucket_tokens() -> list[str | None]:
            segs: list[str | None] = []
            start = 0
            i = 0
            while i < len(tokens) and len(segs) < 3:
                tok = tokens[i]
                if tok == "--":
                    if i > start:
                        segs.append(" ".join(tokens[start:i]))
                    segs.append(None)
                    i += 1
                    start = i
                    continue

                matched_phrase: str | None = None
                matched_len = 0
                for length in (3, 2, 1):
                    if i + length <= len(tokens):
                        phrase = " ".join(tokens[i : i + length]).lower()
                        if phrase in status_phrases:
                            matched_phrase = " ".join(tokens[i : i + length])
                            matched_len = length
                            break

                if matched_phrase:
                    if i > start:
                        segs.append(" ".join(tokens[start:i]))
                    segs.append(matched_phrase)
                    i += matched_len
                    start = i
                    continue

                if date_re.fullmatch(tok) or money_re.fullmatch(tok):
                    if i > start:
                        segs.append(" ".join(tokens[start:i]))
                    segs.append(tok)
                    i += 1
                    start = i
                    continue

                i += 1

            if len(segs) < 3 and start < len(tokens):
                segs.append(" ".join(tokens[start:]))
            return segs

        parts = bucket_tokens()

    # --- 2) normalise parts --------------------------------------------------------
    parts = [None if (p is None or p == "--" or p == "") else p for p in parts]
    parts = (parts + [None, None, None])[:3]

    if used_heuristic and parts[1] is None and parts[2] is None:
        logger.info("triple_parse fallback_partial columns=1/3 reason=weak_separators")

    return parts[0], parts[1], parts[2]


def _is_page_footer(line: str) -> bool:
    """Return ``True`` if *line* looks like a page footer/header.

    Heuristics cover SmartCredit artifacts, generic URLs and timestamp lines
    that frequently appear between account segments.
    """

    low = line.lower()
    if "smartcredit" in low or "credit report & scores" in low:
        return True
    if re.match(r"https?://", line):
        return True
    if re.match(r"\d{1,2}/\d{1,2}/\d{2,4},\s*\d{1,2}:\d{2}\s*(?:am|pm)", low):
        return True
    return False


def _join_wrapped_field_lines(lines: list[str]) -> list[str]:
    """Merge a line with the following one when value text wraps."""

    joined: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if i + 1 < len(lines):
            nxt = lines[i + 1]
            key_part: str | None = None
            rest: str = ""
            for sep in (":", "-", "—"):
                if sep in line:
                    key_part, _, rest = line.partition(sep)
                    break
            if key_part:
                key_norm = _std_field_name(key_part.strip())
                if (
                    key_norm in ACCOUNT_FIELD_SET
                    or key_norm != key_part.strip().lower()
                ):
                    vals = [v for v in re.split(r"\s{2,}", rest.strip()) if v]
                    if len(vals) <= 1:
                        nxt_low = nxt.strip().lower()
                        next_has_key = any(
                            nxt_low.startswith(alias)
                            and len(nxt_low) > len(alias)
                            and nxt_low[len(alias)] in " :-—"
                            for alias in ALIAS_TO_STD
                        )
                        if not next_has_key and not re.match(
                            rf"^{BUREAU_NAME_PATTERN}", nxt, re.I
                        ):
                            if re.match(r"^[a-z]{1,10}\b", nxt):
                                line = f"{line.rstrip()} {nxt.strip()}".strip()
                                logger.info(
                                    "JOIN: merged line %d+%d key=%s", i, i + 1, key_norm
                                )
                                i += 1
        joined.append(line)
        i += 1
    return joined


def _is_section_header(line: str) -> bool:
    """Heuristically detect account or category headers."""

    low = line.lower().strip()
    if not low:
        return False
    if low.startswith(
        (
            "collections",
            "public records",
            "inquiries",
            "revolving accounts",
        )
    ):
        return True
    if ":" in line:
        return False
    return bool(line == line.upper() and re.search(r"[A-Z]", line))


def _maybe_resume_after_history(line: str) -> bool:
    """Return ``True`` if line appears to resume key/value details."""

    if TRIPLE_LINE_RE.match(line):
        return True
    return ":" in line


def _begin_account_session(lines: list[str], index: int) -> dict[str, Any]:
    """Initialize a new account session starting at ``lines[index]``."""

    heading = _clean_line(lines[index])
    order = detect_bureau_order(lines[index : index + 5]) or [
        "transunion",
        "experian",
        "equifax",
    ]
    session = {
        "heading": heading,
        "bureau_order": order,
        "collected_rows": [heading],
        "in_history": False,
        "section": "details",
        "start_index": index,
    }
    logger.info("STITCH: start account=%s bureau_order=%s", heading, order)
    return session


def parse_three_footer_lines(lines: list[str]) -> dict[str, dict[str, Any | None]]:
    """Parse the trailing footer lines mapping to bureaus without shifting.

    The footer typically consists of three lines – one per bureau in the
    TransUnion → Experian → Equifax order – containing ``Account Type``,
    ``Payment Frequency`` and ``Credit Limit`` segments.  A bureau line may be
    missing or contain only a subset of these fields.  ``Credit Limit`` is
    considered numeric only when it clearly represents a number; otherwise the
    normalised value is left as ``None``.
    """

    out: dict[str, dict[str, Any | None]] = {
        b: {"account_type": None, "payment_frequency": None, "credit_limit": None}
        for b in BUREAUS
    }

    # Identify window before the two-year history where the footer lives
    try:
        hist_idx = next(
            i for i, ln in enumerate(lines) if "two-year payment history" in ln.lower()
        )
    except StopIteration:
        hist_idx = len(lines)

    window = lines[max(0, hist_idx - 15) : hist_idx]

    # Collect up to three candidate lines from the bottom that look like footer
    footer_pat = re.compile(
        r"(?i)(account\s*type|payment\s*(?:frequency|freq)|credit\s*limit)"
    )
    candidates: list[str] = []
    for raw in reversed(window):
        clean = _clean_line(raw)
        if not clean:
            continue
        if footer_pat.search(clean):
            candidates.insert(0, clean)
            if len(candidates) == 3:
                break

    order = detect_bureau_order(window) or list(BUREAUS)
    logger.info("FOOTER: order=%s lines=%d", order, len(candidates))

    pats = {b.lower(): pat for b, pat in BUREAU_PATTERNS.items()}
    bureau_lines: dict[str, str] = {}
    unassigned: list[str] = []

    for line in candidates:
        assigned = False
        for b, pat in pats.items():
            if pat.search(line):
                bureau_lines[b] = line
                assigned = True
                break
        if not assigned:
            unassigned.append(line)

    remaining = [b for b in order if b not in bureau_lines]
    if not bureau_lines and len(candidates) == 2:
        # Heuristic: when only two unlabeled lines are present assume TU and EQ
        bureau_lines[order[0]] = candidates[0]
        bureau_lines[order[2]] = candidates[1]
    else:
        for line, b in zip(unassigned, remaining):
            bureau_lines[b] = line

    for b in order:
        line = bureau_lines.get(b)
        if not line:
            logger.info("FOOTER: missing bureau=%s (inserted None)", b)
            continue

        atype, freq, limit_raw, limit_norm = _parse_footer_fields(line)

        out[b]["account_type"] = atype
        out[b]["payment_frequency"] = freq
        out[b]["credit_limit"] = limit_raw
        if limit_raw and limit_norm is None:
            logger.info(
                "FOOTER: non-numeric credit_limit for bureau=%s raw=%r", b, limit_raw
            )

        logger.info(
            "FOOTER: bureau=%s atype=%r freq=%r limit_raw=%r limit_norm=%r",
            b,
            atype,
            freq,
            limit_raw,
            limit_norm,
        )

    return out


def _parse_footer_fields(
    line: str,
) -> tuple[Any | None, Any | None, str | None, Any | None]:
    """Extract account type, payment frequency and credit limit from ``line``."""

    account_re = re.compile(
        r"(?i)account\s*type\s*[:\-]?\s*(?P<val>.+?)(?=(payment|credit|$))"
    )
    freq_re = re.compile(
        r"(?i)payment\s*(?:frequency|freq)\s*[:\-]?\s*(?P<val>.+?)(?=(account|credit|limit|$))"
    )
    limit_re = re.compile(
        r"(?i)credit\s*limit\s*[:\-]?\s*(?P<val>.+?)(?=(account|payment|freq|$))"
    )

    atype = None
    freq = None
    limit_raw: str | None = None

    m = account_re.search(line)
    if m:
        atype = _clean_line(m.group("val")).lower() or None

    m = freq_re.search(line)
    if m:
        freq = _clean_line(m.group("val")).lower() or None

    m = limit_re.search(line)
    if m:
        limit_raw = _clean_line(m.group("val"))

    if atype is None and freq is None and limit_raw is None:
        atype = _clean_line(line).lower() or None

    limit_norm = None
    if limit_raw:
        parsed = to_number(limit_raw)
        if isinstance(parsed, (int, float)):
            limit_norm = parsed
        else:
            limit_norm = None

    return atype, freq, limit_raw, limit_norm


def parse_two_year_history(lines: list[str]) -> dict[str, Any | None]:
    """Extract per-bureau two-year payment history strings."""

    out = {b: None for b in BUREAUS}

    try:
        start = next(
            i for i, ln in enumerate(lines) if "two-year payment history" in ln.lower()
        )
        end = next(
            i
            for i, ln in enumerate(lines[start + 1 :], start + 1)
            if "days late -7 year history" in ln.lower()
        )
    except StopIteration:
        return out

    seg_lines = lines[start + 1 : end]
    current: str | None = None
    buffer: list[str] = []
    for ln in seg_lines:
        ln = _clean_line(ln)
        m = re.match(rf"{BUREAU_NAME_PATTERN}\s*(.*)", ln, re.I)
        if m:
            if current and buffer:
                out[current] = " ".join(" ".join(buffer).split())
            current = re.sub(r"\s+", "", m.group(1)).lower()
            buffer = [m.group(2)]
        else:
            if current:
                buffer.append(ln)
    if current and buffer:
        out[current] = " ".join(" ".join(buffer).split())
    return out


def parse_seven_year_days_late(lines: list[str]) -> dict[str, Any | None]:
    """Sum 30/60/90 day late counts over seven years per bureau."""

    out = {b: None for b in BUREAUS}

    try:
        start = next(
            i for i, ln in enumerate(lines) if "days late -7 year history" in ln.lower()
        )
    except StopIteration:
        return out

    text = "\n".join(lines[start + 1 :])
    for b in BUREAUS:
        m = re.search(rf"{b}.*?30:(\d+)\s+60:(\d+)\s+90:(\d+)", text, re.I)
        if m:
            d30, d60, d90 = map(int, m.groups())
            out[b] = d30 + d60 + d90
    return out


def parse_account_block(
    block_lines: list[str],
    heading: str | None = None,
    *,
    sid: str | None = None,
    account_id: str | None = None,
) -> dict[str, dict[str, Any | None]]:
    lines_raw = block_lines or []
    lines_clean = [_clean_line(x) for x in lines_raw]
    lines = _join_wrapped_field_lines(lines_clean)
    logger.info("parse_account_block start lines=%d", len(lines))
    logger.info("parse_account_block lines[0..3]=%r", lines[:4])

    parsed_triples: dict[str, Any] = {
        "heading": heading,
        "bureau_order": [],
        "rows": [],
    }

    def _init_maps():
        return {b: _empty_bureau_map() for b in BUREAUS}

    bureau_maps = _init_maps()

    order = detect_bureau_order(lines)
    if order:
        parsed_triples["bureau_order"] = list(order)

    vt_idx: int | None = None
    for i, line in enumerate(lines):
        if re.match(r"field\s*:", line, re.I):
            vt_idx = i
            break

    logger.info(
        "parse_account_block bureau_header_detected=%s order=%s",
        bool(order),
        order,
    )

    parsed: set[str] = set()

    if order is not None and vt_idx is not None:
        count = 0
        for idx, line in enumerate(lines[vt_idx + 1 :], start=vt_idx + 1):
            raw_line = lines_raw[idx]
            row_dbg: dict[str, Any] = {"raw": raw_line}
            m = TRIPLE_LINE_RE.match(line)
            raw_vals: list[str | None]
            if m:
                logger.info(
                    "triple_parse layout=aligned key=%s", m.group("key").strip()
                )
                source = "aligned"
                key = m.group("key").strip()
                raw_vals = [m.group("v1"), m.group("v2"), m.group("v3")]
                if raw_vals[1] is None and raw_vals[2] is None:
                    _, _, rest = line.partition(":")
                    v1, v2, v3 = _split_triple_fallback(rest.strip(), order)
                    raw_vals = [v1, v2, v3]
            else:
                key, sep, rest = line.partition(":")
                if not sep or not rest.strip():
                    row_dbg["drop_reason"] = "malformed"
                    parsed_triples["rows"].append(row_dbg)
                    continue
                source = "fallback"
                v1, v2, v3 = _split_triple_fallback(rest.strip(), order)
                raw_vals = [v1, v2, v3]
                std_key = _std_field_name(key)
                abbr = {"transunion": "tu", "experian": "ex", "equifax": "eq"}
                order_str = ",".join(abbr.get(b, b) for b in order)
                logger.info(
                    "triple_parse layout=fallback key=%s v1=%s v2=%s v3=%s order=%s source=fallback",
                    std_key,
                    v1,
                    v2,
                    v3,
                    order_str,
                )
            std_key = _std_field_name(key)
            if (
                std_key == "account_number_display"
                and raw_vals[1] is None
                and raw_vals[2] is None
                and raw_vals[0]
            ):
                toks = str(raw_vals[0]).split()
                if len(toks) == 3:
                    raw_vals = toks
            if std_key not in ACCOUNT_FIELD_SET:
                low = key.lower()
                if "status" in low:
                    std_key = "payment_status" if "pay" in low else "account_status"
                else:
                    row_dbg["drop_reason"] = (
                        "alias_not_found" if std_key == low else "not_in_field_set"
                    )
                    parsed_triples["rows"].append(row_dbg)
                    if row_dbg["drop_reason"] == "alias_not_found":
                        logger.info(
                            "alias_not_found: raw_key=%r after_normalize=%r",
                            key,
                            std_key,
                        )
                    logger.info(
                        "triple_parsed key=%s v1=%s v2=%s v3=%s dropped=unknown_field",
                        key,
                        raw_vals[0],
                        raw_vals[1],
                        raw_vals[2],
                    )
                    continue
            values: list[Any | None] = []
            for idx2, rv in enumerate(raw_vals):
                if rv is None or str(rv).strip() in {"", "--"}:
                    values.append(None)
                else:
                    val = str(rv).strip()
                    if idx2 == 2:
                        val = re.sub(r"\s+", " ", val)
                        val = re.sub(r"[^\w/* ]", "", val).strip()
                        if " " in val:
                            val = val.split()[0]
                    values.append(val)
            val_map = {b: v for b, v in zip(order, values)}
            values_dict = {
                "tu": val_map.get("transunion"),
                "ex": val_map.get("experian"),
                "eq": val_map.get("equifax"),
            }
            orig = dict(values_dict)
            values_dict["tu"] = _strip_leaked_prefix(std_key, values_dict.get("tu"))
            values_dict["ex"] = _strip_leaked_prefix(std_key, values_dict.get("ex"))
            values_dict["eq"] = _strip_leaked_prefix(std_key, values_dict.get("eq"))
            if values_dict != orig:
                logger.info(
                    "VALPREFIX: stripped prefix for key=%s values_before=%s values_after=%s",
                    std_key,
                    orig,
                    values_dict,
                )
            val_map = {
                "transunion": values_dict["tu"],
                "experian": values_dict["ex"],
                "equifax": values_dict["eq"],
            }
            values = [val_map.get(b) for b in order]
            if all(v is None for v in values):
                row_dbg["drop_reason"] = "empty"
                parsed_triples["rows"].append(row_dbg)
                continue
            row_dbg.update(
                {
                    "key_norm": std_key,
                    "values": values_dict,
                    "source": source,
                }
            )
            parsed_triples["rows"].append(row_dbg)
            for b, v in zip(order, values):
                if v is None:
                    continue
                bm = bureau_maps[b]
                if bm.get(std_key) in (None, ""):
                    _assign_std(
                        bm,
                        std_key,
                        v,
                        raw_val=v,
                        provenance=source,
                        bureau=b,
                    )
                    if std_key == "account_number_display":
                        digits = re.sub(r"\D", "", str(v))
                        last4 = digits[-4:] if len(digits) >= 4 else None
                        _assign_std(
                            bm,
                            "account_number_last4",
                            last4,
                            raw_val=last4,
                            provenance=source,
                            bureau=b,
                        )
            logger.info(
                "triple_parsed key=%s v1=%s v2=%s v3=%s dropped=None",
                std_key,
                values[0],
                values[1],
                values[2],
            )
            count += 1
        logger.info(
            "parse_account_block layout=vertical_triples fields_parsed=%d",
            count,
        )
        parsed.update(order)

    elif order is None:
        start = vt_idx + 1 if vt_idx is not None else 0
        count = 0
        default_order = ["transunion", "experian", "equifax"]
        parsed_triples["bureau_order"] = list(default_order)
        for idx, line in enumerate(lines[start:], start=start):
            raw_line = lines_raw[idx]
            row_dbg: dict[str, Any] = {"raw": raw_line}
            m = TRIPLE_LINE_RE.match(line)
            raw_vals: list[str | None]
            if m:
                logger.info(
                    "triple_parse layout=aligned key=%s", m.group("key").strip()
                )
                source = "aligned"
                key = m.group("key").strip()
                raw_vals = [m.group("v1"), m.group("v2"), m.group("v3")]
                if raw_vals[1] is None and raw_vals[2] is None:
                    _, _, rest = line.partition(":")
                    v1, v2, v3 = _split_triple_fallback(rest.strip(), default_order)
                    raw_vals = [v1, v2, v3]
            else:
                key, sep, rest = line.partition(":")
                if not sep or not rest.strip():
                    row_dbg["drop_reason"] = "malformed"
                    parsed_triples["rows"].append(row_dbg)
                    continue
                source = "fallback"
                v1, v2, v3 = _split_triple_fallback(rest.strip(), default_order)
                raw_vals = [v1, v2, v3]
                std_key = _std_field_name(key)
                abbr = {"transunion": "tu", "experian": "ex", "equifax": "eq"}
                order_str = ",".join(abbr.get(b, b) for b in default_order)
                logger.info(
                    "triple_parse layout=fallback key=%s v1=%s v2=%s v3=%s order=%s source=fallback",
                    std_key,
                    v1,
                    v2,
                    v3,
                    order_str,
                )
            std_key = _std_field_name(key)
            if (
                std_key == "account_number_display"
                and raw_vals[1] is None
                and raw_vals[2] is None
                and raw_vals[0]
            ):
                toks = str(raw_vals[0]).split()
                if len(toks) == 3:
                    raw_vals = toks
            if std_key not in ACCOUNT_FIELD_SET:
                low = key.lower()
                if "status" in low:
                    std_key = "payment_status" if "pay" in low else "account_status"
                else:
                    row_dbg["drop_reason"] = (
                        "alias_not_found" if std_key == low else "not_in_field_set"
                    )
                    parsed_triples["rows"].append(row_dbg)
                    if row_dbg["drop_reason"] == "alias_not_found":
                        logger.info(
                            "alias_not_found: raw_key=%r after_normalize=%r",
                            key,
                            std_key,
                        )
                    logger.info(
                        "triple_parsed key=%s v1=%s v2=%s v3=%s dropped=unknown_field",
                        key,
                        raw_vals[0],
                        raw_vals[1],
                        raw_vals[2],
                    )
                    continue
            values: list[Any | None] = []
            for idx2, rv in enumerate(raw_vals):
                if rv is None or str(rv).strip() in {"", "--"}:
                    values.append(None)
                else:
                    val = str(rv).strip()
                    if idx2 == 2:
                        val = re.sub(r"\s+", " ", val)
                        val = re.sub(r"[^\w/* ]", "", val).strip()
                        if " " in val:
                            val = val.split()[0]
                    values.append(val)
            val_map = {b: v for b, v in zip(default_order, values)}
            values_dict = {
                "tu": val_map.get("transunion"),
                "ex": val_map.get("experian"),
                "eq": val_map.get("equifax"),
            }
            orig = dict(values_dict)
            values_dict["tu"] = _strip_leaked_prefix(std_key, values_dict.get("tu"))
            values_dict["ex"] = _strip_leaked_prefix(std_key, values_dict.get("ex"))
            values_dict["eq"] = _strip_leaked_prefix(std_key, values_dict.get("eq"))
            if values_dict != orig:
                logger.info(
                    "VALPREFIX: stripped prefix for key=%s values_before=%s values_after=%s",
                    std_key,
                    orig,
                    values_dict,
                )
            val_map = {
                "transunion": values_dict["tu"],
                "experian": values_dict["ex"],
                "equifax": values_dict["eq"],
            }
            values = [val_map.get(b) for b in default_order]
            if all(v is None for v in values):
                row_dbg["drop_reason"] = "empty"
                parsed_triples["rows"].append(row_dbg)
                continue
            row_dbg.update(
                {
                    "key_norm": std_key,
                    "values": values_dict,
                    "source": source,
                }
            )
            parsed_triples["rows"].append(row_dbg)
            for b, v in zip(default_order, values):
                if v is None:
                    continue
                bm = bureau_maps[b]
                if bm.get(std_key) in (None, ""):
                    _assign_std(
                        bm,
                        std_key,
                        v,
                        raw_val=v,
                        provenance=source,
                        bureau=b,
                    )
                    if std_key == "account_number_display":
                        digits = re.sub(r"\D", "", str(v))
                        last4 = digits[-4:] if len(digits) >= 4 else None
                        _assign_std(
                            bm,
                            "account_number_last4",
                            last4,
                            raw_val=last4,
                            provenance=source,
                            bureau=b,
                        )
            logger.info(
                "triple_parsed key=%s v1=%s v2=%s v3=%s dropped=None",
                std_key,
                values[0],
                values[1],
                values[2],
            )
            count += 1
        if count:
            logger.info(
                "parse_account_block layout=vertical_triples (fallback) default_order=TEQ"
            )
            logger.info(
                "parse_account_block layout=vertical_triples fields_parsed=%d", count
            )
            parsed.update(default_order)

    if not parsed:
        # --- Header-based column spans -----------------------------------
        header_idx: int | None = None
        header_line: str | None = None
        for i, line in enumerate(lines):
            low = line.lower()
            if "account #" in low and "high balance" in low:
                header_idx = i
                header_line = line
                break

        spans: list[tuple[str, int, int]] = []
        if header_line is not None:
            col_aliases: dict[str, list[str]] = {
                "account_number_display": ["account #"],
                "high_balance": ["high balance"],
                "last_verified": ["last verified"],
                "date_of_last_activity": ["date of last activity"],
                "date_reported": ["date reported", "last reported"],
                "date_opened": ["date opened"],
                "balance_owed": [
                    "balance owed",
                    "current balance",
                    "balance",
                    "amount",
                ],
                "closed_date": ["closed date"],
                "account_rating": ["account rating"],
                "account_description": ["account description"],
                "dispute_status": ["dispute status"],
                "creditor_type": ["creditor type"],
                "account_status": ["account status"],
                "payment_status": ["payment status"],
                "creditor_remarks": ["creditor remarks", "remarks", "comment"],
                "payment_amount": ["payment amount"],
                "last_payment": ["last payment"],
                "term_length": ["term length"],
                "past_due_amount": ["past due amount"],
            }
            hlow = header_line.lower()
            pos_list: list[tuple[int, str]] = []
            for key, labels in col_aliases.items():
                positions = [hlow.find(lbl) for lbl in labels]
                positions = [p for p in positions if p >= 0]
                if positions:
                    pos_list.append((min(positions), key))
            pos_list.sort()
            for idx, (start, key) in enumerate(pos_list):
                end = (
                    pos_list[idx + 1][0]
                    if idx + 1 < len(pos_list)
                    else len(header_line)
                )
                spans.append((key, start, end))

        column_spans = spans
        logger.info(
            "parse_account_block header_detected=%s columns=%s",
            bool(header_line),
            column_spans,
        )
        if spans and header_idx is not None:
            for line in lines[header_idx + 1 :]:
                m = re.match(rf"{BUREAU_NAME_PATTERN}\s+(.*)", line, re.I)
                if not m:
                    continue
                bureau = re.sub(r"\s+", "", m.group(1)).lower()
                body = m.group(2)
                body = body.ljust(len(header_line))
                bm = bureau_maps[bureau]
                for key, start, end in spans:
                    seg = body[start:end].strip()
                    if seg in {"", "--"}:
                        _assign_std(
                            bm,
                            key,
                            None,
                            raw_val=seg,
                            provenance="aligned",
                            bureau=bureau,
                        )
                    else:
                        clean = _strip_leaked_prefix(key, seg)
                        _assign_std(
                            bm,
                            key,
                            clean,
                            raw_val=clean,
                            provenance="aligned",
                            bureau=bureau,
                        )
                        if key == "account_number_display":
                            digits = re.sub(r"\D", "", clean)
                            last4 = digits[-4:] if len(digits) >= 4 else None
                            _assign_std(
                                bm,
                                "account_number_last4",
                                last4,
                                raw_val=last4,
                                provenance="aligned",
                                bureau=bureau,
                            )
                parsed.add(bureau)

    # Fallback to legacy regex parsing if header spans failed
    if not parsed:
        joined = lines
        bureau_rows = [
            ln for ln in joined if re.match(rf"^{BUREAU_NAME_PATTERN}\s", ln, re.I)
        ]
        for row in bureau_rows:
            m = BUREAU_LINE_RE.search(row)
            if not m:
                continue
            bname = re.sub(r"\s+", "", m.group(1)).lower()
            b = (
                bname
                if bname in BUREAUS
                else (
                    "transunion"
                    if "trans" in bname
                    else ("experian" if "exp" in bname else "equifax")
                )
            )
            masked = m.group(2)
            bm = bureau_maps[b]
            _assign_std(
                bm,
                "account_number_display",
                masked,
                raw_val=masked,
                provenance="aligned",
                bureau=b,
            )
            digits = re.sub(r"\D", "", masked) if re.search(r"\d", masked) else ""
            last4 = digits[-4:] if len(digits) >= 4 else None
            _assign_std(
                bm,
                "account_number_last4",
                last4,
                raw_val=last4,
                provenance="aligned",
                bureau=b,
            )
            _assign_std(
                bm,
                "high_balance",
                _strip_leaked_prefix("high_balance", m.group(3)),
                raw_val=_strip_leaked_prefix("high_balance", m.group(3)),
                provenance="aligned",
                bureau=b,
            )
            _assign_std(
                bm,
                "last_verified",
                _strip_leaked_prefix("last_verified", m.group(4)),
                raw_val=_strip_leaked_prefix("last_verified", m.group(4)),
                provenance="aligned",
                bureau=b,
            )
            _assign_std(
                bm,
                "date_of_last_activity",
                _strip_leaked_prefix("date_of_last_activity", m.group(5)),
                raw_val=_strip_leaked_prefix("date_of_last_activity", m.group(5)),
                provenance="aligned",
                bureau=b,
            )
            _assign_std(
                bm,
                "date_reported",
                _strip_leaked_prefix("date_reported", m.group(6)),
                raw_val=_strip_leaked_prefix("date_reported", m.group(6)),
                provenance="aligned",
                bureau=b,
            )
            _assign_std(
                bm,
                "date_opened",
                _strip_leaked_prefix("date_opened", m.group(7)),
                raw_val=_strip_leaked_prefix("date_opened", m.group(7)),
                provenance="aligned",
                bureau=b,
            )
            _assign_std(
                bm,
                "balance_owed",
                _strip_leaked_prefix("balance_owed", m.group(8)),
                raw_val=_strip_leaked_prefix("balance_owed", m.group(8)),
                provenance="aligned",
                bureau=b,
            )
            _assign_std(
                bm,
                "account_rating",
                _strip_leaked_prefix("account_rating", m.group(9)),
                raw_val=_strip_leaked_prefix("account_rating", m.group(9)),
                provenance="aligned",
                bureau=b,
            )
            _assign_std(
                bm,
                "account_description",
                _strip_leaked_prefix("account_description", m.group(10)),
                raw_val=_strip_leaked_prefix("account_description", m.group(10)),
                provenance="aligned",
                bureau=b,
            )
            _assign_std(
                bm,
                "creditor_type",
                _strip_leaked_prefix("creditor_type", m.group(11)),
                raw_val=_strip_leaked_prefix("creditor_type", m.group(11)),
                provenance="aligned",
                bureau=b,
            )
            _assign_std(
                bm,
                "account_status",
                _strip_leaked_prefix("account_status", m.group(12)),
                raw_val=_strip_leaked_prefix("account_status", m.group(12)),
                provenance="aligned",
                bureau=b,
            )
            _assign_std(
                bm,
                "payment_status",
                _strip_leaked_prefix("payment_status", m.group(13)),
                raw_val=_strip_leaked_prefix("payment_status", m.group(13)),
                provenance="aligned",
                bureau=b,
            )
            _assign_std(
                bm,
                "payment_amount",
                _strip_leaked_prefix("payment_amount", m.group(14)),
                raw_val=_strip_leaked_prefix("payment_amount", m.group(14)),
                provenance="aligned",
                bureau=b,
            )
            _assign_std(
                bm,
                "last_payment",
                _strip_leaked_prefix("last_payment", m.group(15)),
                raw_val=_strip_leaked_prefix("last_payment", m.group(15)),
                provenance="aligned",
                bureau=b,
            )
            _assign_std(
                bm,
                "past_due_amount",
                _strip_leaked_prefix("past_due_amount", m.group(16)),
                raw_val=_strip_leaked_prefix("past_due_amount", m.group(16)),
                provenance="aligned",
                bureau=b,
            )

    # Footer triplet lines (account type / payment frequency / credit limit)
    pre_scan = 15
    slice_from = max(0, len(lines) - pre_scan)
    last_lines = lines[slice_from:]
    if last_lines:
        logger.info(
            "FOOTER: pre-scan last_lines=%d sample_first=%r sample_last=%r",
            len(last_lines),
            last_lines[0],
            last_lines[-1],
        )
    else:
        logger.info(
            "FOOTER: pre-scan empty (block_len=%d, slice_from=%d, slice_to=%d)",
            len(lines),
            slice_from,
            len(lines),
        )
    footer = parse_three_footer_lines(last_lines)
    for b in BUREAUS:
        for k in ("account_type", "payment_frequency", "credit_limit"):
            val = footer.get(b, {}).get(k)
            if val is None:
                continue
            clean = val
            if isinstance(val, str):
                clean = _strip_leaked_prefix(k, val)
                if clean != val:
                    abbr = {"transunion": "tu", "experian": "ex", "equifax": "eq"}
                    logger.info(
                        "VALPREFIX: stripped prefix for key=%s values_before=%s values_after=%s",
                        k,
                        {abbr[b]: val},
                        {abbr[b]: clean},
                    )
            _assign_std(
                bureau_maps[b],
                k,
                clean,
                raw_val=val,
                provenance="footer",
                bureau=b,
            )

    hist2y = parse_two_year_history(lines)
    sev7 = parse_seven_year_days_late(lines)
    for b in BUREAUS:
        bm = bureau_maps[b]
        _assign_std(
            bm,
            "two_year_payment_history",
            hist2y[b],
            raw_val=hist2y[b],
            provenance="aligned",
            bureau=b,
        )
        _assign_std(
            bm,
            "seven_year_days_late",
            sev7[b],
            raw_val=sev7[b],
            provenance="aligned",
            bureau=b,
        )
    result = bureau_maps
    for b in ("transunion", "experian", "equifax"):
        bm = result.get(b, _empty_bureau_map())
        filled = sum(1 for k in ACCOUNT_FIELD_SET if bm[k] is not None)
        logger.info("parse_account_block result bureau=%s filled=%d/25", b, filled)
    if sid and account_id:
        _save_parsed_triples(sid, account_id, heading, parsed_triples)
    return result


def parse_collection_block(
    block_lines: list[str],
    bureau_order: Sequence[str] | None = None,
    *,
    sid: str | None = None,
    account_id: str | None = None,
    heading: str | None = None,
) -> dict[str, dict[str, Any | None]]:
    """Parse simplified collection/charge-off blocks."""

    logger.info("parse_collection_block start lines=%d", len(block_lines))
    parsed_triples: dict[str, Any] = {
        "heading": heading,
        "bureau_order": list(bureau_order) if bureau_order else [],
        "rows": [],
    }

    def _init():
        return {b: _empty_bureau_map() for b in BUREAUS}

    maps = _init()
    lines_clean = [_clean_line(ln) for ln in block_lines]
    lines = _join_wrapped_field_lines(lines_clean)
    order = detect_bureau_order(lines)
    if not order:
        if bureau_order:
            order = list(bureau_order)
            logger.info("COLL: using bureau_order from stitch=%s", bureau_order)
        else:
            order = ["transunion", "experian", "equifax"]
    parsed_triples["bureau_order"] = list(order)

    neg_pat = re.compile(r"(collection|charge[-\s]?off|repossession)", re.I)

    for idx, line in enumerate(lines):
        raw_line = block_lines[idx] if idx < len(block_lines) else line
        row_dbg: dict[str, Any] = {"raw": raw_line}
        key: str | None = None
        raw_vals: list[str | None] = []
        source = "fallback"
        m = TRIPLE_LINE_RE.match(line)
        if m:
            source = "aligned"
            key = m.group("key").strip()
            raw_vals = [m.group("v1"), m.group("v2"), m.group("v3")]
        else:
            parts = re.split(r"\s*[:\-—]\s*", line, 1)
            if len(parts) == 2 and parts[1].strip():
                key = parts[0].strip()
                v1, v2, v3 = _split_triple_fallback(parts[1].strip(), order)
                raw_vals = [v1, v2, v3]
            else:
                segs = re.split(r"\s{2,}", line)
                if len(segs) >= 4:
                    source = "no_colon"
                    key = segs[0].strip()
                    raw_vals = segs[1:4]
                else:
                    low = line.lower()
                    for alias in sorted(ALIAS_TO_STD.keys(), key=len, reverse=True):
                        if low.startswith(alias):
                            after = low[len(alias) :]
                            if (
                                not after
                                or after[0] in " :-—"
                                or (
                                    alias in ACCOUNT_NUMBER_ALIASES
                                    and re.match(r"[0-9xX*]", after[0])
                                )
                            ):
                                rest = line[len(alias) :]
                                if after and after[0] in " :-—":
                                    rest = rest.lstrip(" :-—")
                                key = alias
                                if alias in ACCOUNT_NUMBER_ALIASES:
                                    logger.info(
                                        "alias_norm: key='account #' matched from raw='%s'",
                                        raw_line,
                                    )
                                v1, v2, v3 = _split_triple_fallback(rest, order)
                                raw_vals = [v1, v2, v3]
                                if (
                                    alias in ACCOUNT_NUMBER_ALIASES
                                    and raw_vals[1] is None
                                    and raw_vals[2] is None
                                    and raw_vals[0]
                                ):
                                    toks = str(raw_vals[0]).split()
                                    if len(toks) == 3:
                                        raw_vals = toks
                                break
                    if key is None:
                        v1, v2, v3 = _split_triple_fallback(line, order)
                        if any(v and neg_pat.search(v) for v in (v1, v2, v3)):
                            key = "payment status"
                            raw_vals = [v1, v2, v3]
                        else:
                            row_dbg["drop_reason"] = "malformed"
                            parsed_triples["rows"].append(row_dbg)
                            continue

        std_key = _std_field_name(key)
        if source == "fallback":
            abbr = {"transunion": "tu", "experian": "ex", "equifax": "eq"}
            order_str = ",".join(abbr.get(b, b) for b in order)
            logger.info(
                "triple_parse layout=fallback key=%s v1=%s v2=%s v3=%s order=%s source=fallback",
                std_key,
                raw_vals[0],
                raw_vals[1],
                raw_vals[2],
                order_str,
            )
        if std_key not in ACCOUNT_FIELD_SET:
            row_dbg["drop_reason"] = (
                "alias_not_found"
                if std_key == (key or "").lower()
                else "not_in_field_set"
            )
            parsed_triples["rows"].append(row_dbg)
            if row_dbg["drop_reason"] == "alias_not_found":
                logger.info(
                    "alias_not_found: raw_key=%r after_normalize=%r",
                    key,
                    std_key,
                )
            continue
        values: list[Any | None] = []
        for rv in raw_vals:
            if rv is None or rv.strip() in {"", "--"}:
                values.append(None)
            else:
                values.append(rv.strip())
        val_map = {b: v for b, v in zip(order, values)}
        values_dict = {
            "tu": val_map.get("transunion"),
            "ex": val_map.get("experian"),
            "eq": val_map.get("equifax"),
        }
        orig = dict(values_dict)
        values_dict["tu"] = _strip_leaked_prefix(std_key, values_dict.get("tu"))
        values_dict["ex"] = _strip_leaked_prefix(std_key, values_dict.get("ex"))
        values_dict["eq"] = _strip_leaked_prefix(std_key, values_dict.get("eq"))
        if values_dict != orig:
            logger.info(
                "VALPREFIX: stripped prefix for key=%s values_before=%s values_after=%s",
                std_key,
                orig,
                values_dict,
            )
        val_map = {
            "transunion": values_dict["tu"],
            "experian": values_dict["ex"],
            "equifax": values_dict["eq"],
        }
        values = [val_map.get(b) for b in order]
        if all(v is None for v in values):
            row_dbg["drop_reason"] = "empty"
            parsed_triples["rows"].append(row_dbg)
            continue
        row_dbg.update(
            {
                "key_norm": std_key,
                "values": values_dict,
                "source": source,
            }
        )
        parsed_triples["rows"].append(row_dbg)
        for b, v in zip(order, values):
            if v is None:
                continue
            _assign_std(
                maps[b],
                std_key,
                v,
                raw_val=v,
                provenance="collection",
                bureau=b,
            )
        logger.info(
            "COLL: parsed key=%s tu=%r ex=%r eq=%r",
            std_key,
            values[0],
            values[1],
            values[2],
        )
    # Footer triplet lines (account type / payment frequency / credit limit)
    pre_scan = 15
    slice_from = max(0, len(lines) - pre_scan)
    last_lines = lines[slice_from:]
    if last_lines:
        logger.info(
            "FOOTER: pre-scan last_lines=%d sample_first=%r sample_last=%r",
            len(last_lines),
            last_lines[0],
            last_lines[-1],
        )
    else:
        logger.info(
            "FOOTER: pre-scan empty (block_len=%d, slice_from=%d, slice_to=%d)",
            len(lines),
            slice_from,
            len(lines),
        )
    footer = parse_three_footer_lines(last_lines)
    for b in BUREAUS:
        for k in ("account_type", "payment_frequency", "credit_limit"):
            val = footer.get(b, {}).get(k)
            if val is None:
                continue
            clean = val
            if isinstance(val, str):
                clean = _strip_leaked_prefix(k, val)
                if clean != val:
                    abbr = {"transunion": "tu", "experian": "ex", "equifax": "eq"}
                    logger.info(
                        "VALPREFIX: stripped prefix for key=%s values_before=%s values_after=%s",
                        k,
                        {abbr[b]: val},
                        {abbr[b]: clean},
                    )
            _assign_std(
                maps[b],
                k,
                clean,
                raw_val=val,
                provenance="footer",
                bureau=b,
            )

    result = maps
    for b in ("transunion", "experian", "equifax"):
        m = result.get(b) or {}
        non_null = sum(1 for f in ACCOUNT_FIELD_SET if m.get(f) is not None)
        logger.info("COLL: result bureau=%s filled=%d/25", b, non_null)
    if sid and account_id:
        _save_parsed_triples(sid, account_id, heading, parsed_triples)
    return result


def _flush_account_session(
    session: Mapping[str, Any],
) -> dict[str, dict[str, Any | None]]:
    """Flush collected rows of an account session via :func:`parse_account_block`."""

    rows = [ln for ln in session.get("collected_rows", []) if not _is_page_footer(ln)]
    block_maps = parse_account_block(rows)
    fields = sum(
        1 for b in BUREAUS for v in block_maps.get(b, {}).values() if v is not None
    )
    logger.info(
        "STITCH: end account=%s rows_collected=%d fields_before_merge=%d",
        session.get("heading"),
        len(rows),
        fields,
    )
    return block_maps


def stitch_account_blocks(lines: list[str]) -> list[dict[str, dict[str, Any | None]]]:
    """Stitch scattered account segments and parse them sequentially."""

    results: list[dict[str, dict[str, Any | None]]] = []
    session: dict[str, Any] | None = None

    for idx, raw in enumerate(lines):
        line = _clean_line(raw)
        if not session:
            if _is_section_header(line):
                session = _begin_account_session(lines, idx)
            continue

        if _is_section_header(line) and idx != session.get("start_index"):
            results.append(_flush_account_session(session))
            session = _begin_account_session(lines, idx)
            continue

        if _is_page_footer(line):
            logger.info("STITCH: skip footer line='%s'", line[:50])
            continue

        session["collected_rows"].append(line)

        low = line.lower()
        if (
            "two-year payment history" in low
            or "days late -7 year history" in low
            or "days late – 7 year history" in low
        ):
            if not session.get("in_history"):
                session["in_history"] = True
                logger.info("STITCH: enter history account=%s", session["heading"])
            continue

        if session.get("in_history") and _maybe_resume_after_history(line):
            session["in_history"] = False
            logger.info(
                "STITCH: resume after history with order=%s",
                session.get("bureau_order"),
            )

    if session:
        results.append(_flush_account_session(session))

    return results


def _find_bureau_entry(acc: Mapping[str, Any], bureau: str) -> Mapping[str, Any] | None:
    """Find matching entry in acc['bureaus'] for a bureau (accepts title/lower)."""
    items = acc.get("bureaus") or []
    if not isinstance(items, list):
        return None
    targets = {bureau, bureau.lower(), bureau.title()}
    for it in items:
        if not isinstance(it, Mapping):
            continue
        bname = it.get("bureau") or it.get("name")
        if isinstance(bname, str) and bname in targets:
            return it
    return None


def _fill_bureau_map_from_sources(
    acc: Mapping[str, Any],
    bureau: str,
    dst: dict[str, Any],
    account_block_lines: list[str] | None = None,
    *,
    sid: str | None = None,
) -> None:
    """Fill a single bureau's 25-field map for an account.

    Source priority (highest to lowest):
    - acc.bureaus[]
    - acc.bureau_details[bureau]
    - acc.raw.account_history.by_bureau[bureau] (existing)

    Gentle normalization for numeric/date fields only when unambiguous.
    Also backfills account_number_display/last4 from top-level when missing.
    """

    heading = acc.get("normalized_name") or acc.get("name")
    acc_id = acc.get("account_id") or heading

    # 1) bureaus[] entry
    src = _find_bureau_entry(acc, bureau)
    if isinstance(src, Mapping):
        for s_key, val in src.items():
            std = _std_field_name(s_key)
            if std not in ACCOUNT_FIELD_SET:
                continue
            if dst.get(std) is None and val not in (None, "", {}, []):
                _assign_std(
                    dst,
                    s_key,
                    val,
                    raw_val=val,
                    provenance="aligned",
                    bureau=bureau,
                )

    # 2) bureau_details[bureau]
    details = (acc.get("bureau_details") or {}).get(bureau)
    if not isinstance(details, Mapping):
        details = (acc.get("bureau_details") or {}).get(bureau.lower()) or (
            acc.get("bureau_details") or {}
        ).get(bureau.title())
    if isinstance(details, Mapping):
        for key, val in details.items():
            std = _std_field_name(key)
            if std not in ACCOUNT_FIELD_SET:
                continue
            if dst.get(std) is None and val not in (None, "", {}, []):
                _assign_std(
                    dst,
                    key,
                    val,
                    raw_val=val,
                    provenance="aligned",
                    bureau=bureau,
                )

    # 3) Existing raw.by_bureau values as last resort
    try:
        existing = (
            acc.get("raw", {})
            .get("account_history", {})
            .get("by_bureau", {})
            .get(bureau, {})
        )
        if isinstance(existing, Mapping):
            for key, val in existing.items():
                std = _std_field_name(key)
                if std not in ACCOUNT_FIELD_SET:
                    continue
                if dst.get(std) is None and val not in (None, "", {}, []):
                    _assign_std(
                        dst,
                        key,
                        val,
                        raw_val=val,
                        provenance="aligned",
                        bureau=bureau,
                    )
    except Exception:
        pass

    # Backfill account number fields (top-level fallbacks)
    if dst.get("account_number_last4") in (None, ""):
        last4 = acc.get("account_number_last4") or acc.get("account_number")
        if isinstance(last4, str):
            digits = re.sub(r"\D", "", last4)
            _assign_std(
                dst,
                "account_number_last4",
                digits[-4:] if len(digits) >= 4 else None,
                raw_val=digits[-4:] if len(digits) >= 4 else None,
                provenance="aligned",
                bureau=bureau,
            )
        elif isinstance(last4, (int, float)):
            s = str(int(last4))
            _assign_std(
                dst,
                "account_number_last4",
                s[-4:] if len(s) >= 4 else None,
                raw_val=s[-4:] if len(s) >= 4 else None,
                provenance="aligned",
                bureau=bureau,
            )

    if dst.get("account_number_display") in (None, ""):
        disp = (
            acc.get("account_number_raw")
            or acc.get("account_number_display")
            or acc.get("account_number")
        )
        if disp not in (None, "", {}, []):
            _assign_std(
                dst,
                "account_number_display",
                disp,
                raw_val=disp,
                provenance="aligned",
                bureau=bureau,
            )

    # --- BEFORE return, after merging from known sources ---
    missing_before = [k for k, v in dst.items() if v is None]
    logger.debug(
        "pre-parse gap: account=%s bureau=%s missing=%d top=%s",
        acc.get("normalized_name") or acc.get("name"),
        bureau,
        len(missing_before),
        ",".join(missing_before[:5]),
    )

    # If still missing keys and block lines were provided, backfill via parser
    if missing_before and account_block_lines:
        try:
            block_maps = parse_account_block(
                account_block_lines,
                heading=heading,
                sid=sid,
                account_id=acc_id,
            )
            bm = block_maps.get(bureau, {})
            for k in dst.keys():
                if dst[k] is None and k in bm:
                    _assign_std(
                        dst,
                        k,
                        bm[k],
                        raw_val=bm[k],
                        provenance="aligned",
                        bureau=bureau,
                    )
        except Exception:
            logger.exception(
                "parse_account_block_failed account=%s bureau=%s",
                acc.get("normalized_name") or acc.get("name"),
                bureau,
            )

        missing_after_acc = [k for k, v in dst.items() if v is None]
        if missing_after_acc:
            try:
                coll_maps = parse_collection_block(
                    account_block_lines,
                    sid=sid,
                    account_id=acc_id,
                    heading=heading,
                )
                bm2 = coll_maps.get(bureau, {})
                for k in dst.keys():
                    if dst[k] is None and k in bm2:
                        _assign_std(
                            dst,
                            k,
                            bm2[k],
                            raw_val=bm2[k],
                            provenance="collection",
                            bureau=bureau,
                        )
            except Exception:
                logger.exception(
                    "parse_collection_block_failed account=%s bureau=%s",
                    acc.get("normalized_name") or acc.get("name"),
                    bureau,
                )

    filled = sum(1 for v in dst.values() if v is not None)
    logger.info(
        "parser_bureau_fill account=%s bureau=%s filled=%d/25",
        acc.get("normalized_name") or acc.get("name"),
        bureau,
        filled,
    )


def attach_bureau_meta_tables(sections: Mapping[str, Any]) -> None:
    """Attach per-bureau meta tables and supplemental RAW blocks."""

    accounts = sections.get("all_accounts") or []
    if not isinstance(accounts, list):
        return

    session_id = sections.get("session_id") or ""

    # Build fuzzy block lookup if raw blocks were provided
    if sections.get("fbk_blocks") and not sections.get("blocks_by_account_fuzzy"):
        try:
            sections["blocks_by_account_fuzzy"] = build_block_fuzzy(
                sections.get("fbk_blocks") or []
            )
        except Exception:
            logger.exception(
                "build_block_fuzzy_in_attach_failed session=%s", session_id
            )

    # Normalize report-level inquiries/public info once
    inq_src = sections.get("inquiries") or []
    norm_inqs: list[dict[str, Any]] = []
    if isinstance(inq_src, list):
        for inq in inq_src:
            if not isinstance(inq, Mapping):
                continue
            bureau = (
                normalize_bureau_name(inq.get("bureau")) if inq.get("bureau") else None
            )
            item = {
                "bureau": bureau.lower() if bureau else None,
                "subscriber": inq.get("subscriber")
                or inq.get("creditor_name")
                or inq.get("name"),
                "date": to_iso_date(inq.get("date")) if inq.get("date") else None,
                "type": inq.get("type"),
                "permissible_purpose": inq.get("permissible_purpose"),
                "remarks": inq.get("remarks"),
                "_provenance": inq.get("_provenance", {}),
            }
            for k in INQUIRY_FIELDS:
                item.setdefault(k, None)
            norm_inqs.append(item)

    pub_src = sections.get("public_information") or []
    norm_pub: list[dict[str, Any]] = []
    if isinstance(pub_src, list):
        for item in pub_src:
            if not isinstance(item, Mapping):
                continue
            bureau = (
                normalize_bureau_name(item.get("bureau"))
                if item.get("bureau")
                else None
            )
            date_val = item.get("date_filed") or item.get("date")
            pi = {
                "bureau": bureau.lower() if bureau else None,
                "item_type": item.get("item_type") or item.get("type"),
                "status": item.get("status"),
                "date_filed": to_iso_date(date_val) if date_val else None,
                "amount": to_number(item.get("amount")) if item.get("amount") else None,
                "remarks": item.get("remarks"),
                "_provenance": item.get("_provenance", {}),
            }
            for k in PUBLIC_INFO_FIELDS:
                pi.setdefault(k, None)
            norm_pub.append(pi)

    for acc in accounts:
        if not isinstance(acc, dict):
            continue
        by = _ensure_paths(acc, "raw", "account_history", "by_bureau")

        raw = acc.setdefault("raw", {})
        raw.setdefault("inquiries", {"items": []})
        raw.setdefault("public_information", {"items": []})

        if norm_inqs:
            if not raw.get("inquiries", {}).get("items"):
                raw["inquiries"]["items"] = norm_inqs
        elif inq_src:
            slug = (
                acc.get("account_id")
                or acc.get("normalized_name")
                or acc.get("name")
                or ""
            )
            logger.warning(
                "inquiries_detected_but_not_written session=%s account=%s",
                session_id,
                slug,
            )

        if norm_pub:
            if not raw.get("public_information", {}).get("items"):
                raw["public_information"]["items"] = norm_pub
        elif pub_src:
            slug = (
                acc.get("account_id")
                or acc.get("normalized_name")
                or acc.get("name")
                or ""
            )
            logger.warning(
                "public_info_detected_but_not_written session=%s account=%s",
                session_id,
                slug,
            )

        account_block_lines = _find_block_lines_for_account(sections, acc)

        for b in BUREAUS:
            dst = by.get(b)
            if not isinstance(dst, Mapping):
                dst = _empty_bureau_map()
            else:
                for field in ACCOUNT_FIELD_SET:
                    dst.setdefault(field, None)
            _fill_bureau_map_from_sources(
                acc,
                b,
                dst,
                account_block_lines,
                sid=session_id,
            )
            by[b] = dst

        tu = sum(1 for v in by.get("transunion", {}).values() if v is not None)
        ex = sum(1 for v in by.get("experian", {}).values() if v is not None)
        eq = sum(1 for v in by.get("equifax", {}).values() if v is not None)
        try:
            slug = (
                acc.get("account_id")
                or acc.get("normalized_name")
                or acc.get("name")
                or ""
            )
            logger.info(
                "parser_bureau_fill session=%s account=%s tu=%d/25 ex=%d/25 eq=%d/25",
                session_id,
                slug,
                tu,
                ex,
                eq,
            )
        except Exception:
            pass
        logger.info(
            "bureau_meta_coverage name=%s tu_missing=%d ex_missing=%d eq_missing=%d",
            acc.get("normalized_name") or acc.get("name"),
            25 - tu,
            25 - ex,
            25 - eq,
        )
