from __future__ import annotations

import re
from typing import List

# Exported allow-list so other modules can reuse
NON_COLON_LABELS = {
    "account#",  # appears as "Account #"
    "account",   # normalized variant when non-alphanumerics removed
    "highbalance",
    "lastverified",
    "dateoflastactivity",
    "datereported",
    "dateopened",
    "balanceowed",
    "closeddate",
    "accountrating",
    "accountdescription",
    "disputestatus",
    "creditortype",
    "originalcreditor",
    "accountstatus",
    "paymentstatus",
    "creditorremarks",
    "paymentamount",
    "lastpayment",
    "termlength",
    "pastdueamount",
    # Personal information common labels
    "name",
    "fullname",
    "address",
    "currentaddress",
    "formeraddress",
    "previousaddress",
    "city",
    "state",
    "zip",
    "zipcode",
    "phone",
    "email",
    "dob",
    "dateofbirth",
    "creditreportdate",
    "alsoknownas",
    "ssn",
    "socialsecurity",
    "employer",
}

_BUREAU_HDRS = {"transunion", "experian", "equifax"}

# Canonical label -> normalized field name
LABEL_MAP = {
    "Account #": "account_number_display",  # treat as field label even though it ends with '#'
    "High Balance": "high_balance",
    "Last Verified": "last_verified",
    "Date of Last Activity": "date_of_last_activity",
    "Date Reported": "date_reported",
    "Date Opened": "date_opened",
    "Balance Owed": "balance_owed",
    "Closed Date": "closed_date",
    "Account Rating": "account_rating",
    "Account Description": "account_description",
    "Dispute Status": "dispute_status",
    "Creditor Type": "creditor_type",
    "Original Creditor": "original_creditor",
    "Original Creditor 01": "original_creditor",
    "Original Creditor 02": "original_creditor",
    "Orig. Creditor": "original_creditor",
    "Orig Creditor": "original_creditor",
    "Account Status": "account_status",
    "Payment Status": "payment_status",
    "Creditor Remarks": "creditor_remarks",
    "Payment Amount": "payment_amount",
    "Last Payment": "last_payment",
    "Term Length": "term_length",
    "Past Due Amount": "past_due_amount",
    "Account Type": "account_type",
    "Payment Frequency": "payment_frequency",
    "Credit Limit": "credit_limit",
    # "Two-Year Payment History": intentionally NOT mapped now
}

__all__ = ["LABEL_MAP"]

# Canonical label -> expected value type
LABEL_SCHEMA = {
    "Account #": "id",
    "High Balance:": "money",
    "Last Verified:": "date",
    "Date of Last Activity:": "date",
    "Date Reported:": "date",
    "Date Opened:": "date",
    "Balance Owed:": "money",
    "Closed Date:": "date",
    "Account Rating:": "enum",
    "Account Description:": "text",
    "Dispute Status:": "enum",
    "Creditor Type:": "enum",
    "Original Creditor:": "text",
    "Original Creditor 01:": "text",
    "Original Creditor 02:": "text",
    "Orig. Creditor:": "text",
    "Orig Creditor:": "text",
    "Account Status:": "enum",
    "Payment Status:": "enum",
    "Creditor Remarks:": "text",
    "Payment Amount:": "money",
    "Last Payment:": "date",
    "Term Length:": "term",
    "Past Due Amount:": "money",
    "Amount:": "money",
    "Remarks:": "text",
}

# Section headings that mark the end of the overview portion
SECTION_HEADINGS = {
    "twoyearpaymenthistory",
    "dayslate7yearhistory",
}


def _norm(s: str | None) -> str:
    """Normalize by lowercasing and removing non-alphanumerics."""
    return re.sub(r"\W+", "", (s or "").lower())


# --- Value-type detectors ---
MONEY_RE = re.compile(r"\$\s*\d")
DATE_RE = re.compile(r"\b\d{1,2}[./]\d{1,2}[./]\d{2,4}\b")
TERM_RE = re.compile(r"\b\d{2,4}\s*month\(s\)\b", re.IGNORECASE)

ENUM_WORDS = {"closed", "current", "paid", "individual", "open", "delinquent"}
DESC_HINTS = {"mortgage", "loan", "card", "finance", "bank", "company"}


# Enumerations (expanded)
ACCOUNT_STATUS_ENUM = {"open", "closed", "paid", "current", "delinquent", "chargeoff", "collection"}
PAYMENT_STATUS_ENUM = {
    "current",
    "late",
    "ok",
    "paid",
    "120",
    "90",
    "60",
    "30",
    "co",
    "cof",
    "collection/chargeoff",
}
ACCOUNT_RATING_ENUM = {"individual", "joint", "authorized user", "co-signer", "authorized", "authorizeduser"}
CREDITOR_TYPE_ENUM = {
    "bank - mortgage loans",
    "mortgage companies - finance",
    "bank",
    "credit card",
    "auto loan",
}

# History tokens (months and OK/30/60/90...)
_HEB_MONTHS = {"ינו׳", "פבר׳", "מרץ", "אפר׳", "מאי", "יוני", "יולי", "אוג׳", "ספט׳", "אוק׳", "נוב׳", "דצמ׳"}
_EN_MONTHS3 = {"jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"}


def looks_like_history_token(text: str) -> bool:
    z = (text or "").strip()
    if not z:
        return False
    zl = z.lower()
    if zl == "ok" or zl in {"30", "60", "90", "120", "150", "180"}:
        return True
    if z in _HEB_MONTHS:
        return True
    if len(zl) >= 3 and zl[:3] in _EN_MONTHS3:
        return True
    return False


def detect_value_type(text: str) -> str:
    """Heuristic detector for a value cell's semantic type.
    Returns one of: money, date, term, enum, text, id, empty
    """
    z = (text or "").strip()
    if not z:
        return "empty"
    zl = z.lower()
    if MONEY_RE.search(z):
        return "money"
    if DATE_RE.search(z):
        return "date"
    if TERM_RE.search(z):
        return "term"
    # Expanded enum detection
    if zl in ACCOUNT_STATUS_ENUM or zl in PAYMENT_STATUS_ENUM or zl in ACCOUNT_RATING_ENUM or zl in CREDITOR_TYPE_ENUM:
        return "enum"
    if "collection/chargeoff" in zl or "chargeoff" in zl or "collection" in zl:
        return "enum"
    if any(w in zl for w in ENUM_WORDS):
        return "enum"
    if any(w in zl for w in DESC_HINTS):
        return "text"
    # crude ID match (account numbers, masked digits)
    if re.search(r"\b[\d\*]{4,}\b", zl):
        return "id"
    return "text"


def extract_canonical_labels(block_lines: List[str]) -> List[str]:
    """
    Given the raw block['lines'] (flat list of strings) for a single account block,
    return an ordered list of canonical labels (left column) that describe the
    account overview fields (before the first bureau header).
    """
    lines = [str(x or "").strip() for x in (block_lines or [])]
    # 1) find first bureau header index
    first_bureau_idx = None
    for i, line in enumerate(lines):
        if _norm(line) in _BUREAU_HDRS:
            first_bureau_idx = i
            break
    label_zone = lines[: first_bureau_idx if first_bureau_idx is not None else len(lines)]

    # 2) collect labels by rule
    out: List[str] = []
    seen_norm: set[str] = set()
    for raw in label_zone:
        if not raw:
            continue
        keep = False
        if raw.endswith(":"):
            keep = True
        else:
            z = _norm(raw)
            if z in NON_COLON_LABELS:
                keep = True
        if not keep:
            continue
        # Remove duplicates by normalized form (without trailing colon)
        key = _norm(raw[:-1] if raw.endswith(":") else raw)
        if key in seen_norm:
            continue
        seen_norm.add(key)
        out.append(raw)

    return out


def find_section_cut_index(block_lines: List[str]) -> int:
    """
    Return the index in block_lines where the overview ends (before history).
    Find the first occurrence of any SECTION_HEADINGS (normalized).
    If not found, return len(block_lines).
    """
    lines = [str(x or "").strip() for x in (block_lines or [])]
    for i, line in enumerate(lines):
        if _norm(line) in SECTION_HEADINGS:
            return i
    return len(lines)
