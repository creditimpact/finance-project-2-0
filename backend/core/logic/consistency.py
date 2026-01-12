"""Normalization and field consistency helpers for bureau data."""

from __future__ import annotations

import os
import re
from datetime import date, datetime
from typing import Any, Dict, Mapping, MutableMapping, Optional, Sequence, Tuple

from backend.core.logic.context import get_validation_context
from backend.core.merge.acctnum import acctnum_level, acctnum_match_level

__all__ = ["compute_field_consistency", "compute_inconsistent_fields"]


_BUREAU_KEYS: Tuple[str, ...] = ("transunion", "experian", "equifax")
_AMOUNT_FIELD_HINTS = {
    "amount",
    "balance",
    "limit",
    "payment",
    "value",
    "due",
    "credit",
    "loan",
    "debt",
}

_ENUM_DOMAINS: Dict[str, Dict[str, str]] = {
    "account_status": {
        "open": "open",
        "opened": "open",
        "active": "open",
        "current": "open",
        "close": "closed",
        "closed": "closed",
        "paidandclosed": "closed",
        "paidclosed": "closed",
        "closedpaid": "closed",
        "paid": "paid",
        "pif": "paid",
        "chargeoff": "chargeoff",
        "chargedoff": "chargeoff",
        "chargeoffsold": "chargeoff",
        "collection": "collection",
        "collections": "collection",
        "repossession": "repossession",
        "transferred": "transferred",
        "sold": "sold",
        "foreclosure": "foreclosure",
    },
    "payment_status": {
        "current": "ok",
        "paysasagreed": "ok",
        "payingasagreed": "ok",
        "paidasagreed": "ok",
        "ok": "ok",
        "neverlate": "ok",
        "chargeoff": "chargeoff",
        "chargedoff": "chargeoff",
        "collection": "collection",
        "collections": "collection",
        "repossession": "repossession",
        "late30": "late30",
        "late60": "late60",
        "late90": "late90",
        "late120": "late120",
        "late150": "late150",
        "late180": "late180",
        "120": "late120",
        "150": "late150",
        "180": "late180",
        "30": "late30",
        "60": "late60",
        "90": "late90",
    },
    "account_type": {
        "creditcard": "credit_card",
        "creditcards": "credit_card",
        "bankcreditcard": "bank_credit_card",
        "bankcreditcards": "bank_credit_card",
        "autoloan": "auto_loan",
        "autoloans": "auto_loan",
        "studentloan": "student_loan",
        "studentloans": "student_loan",
        "personalloan": "personal_loan",
        "personalloans": "personal_loan",
        "mortgage": "mortgage",
        "revolving": "revolving",
        "installment": "installment",
        "collection": "collection",
        "chargeaccount": "charge_account",
    },
    "creditor_type": {
        "bank": "bank",
        "banks": "bank",
        "banking": "bank",
        "bankinstitution": "bank",
        "bankinstitutions": "bank",
        "allbanks": "bank",
        "allbank": "bank",
        "thebank": "bank",
        "bankcreditcards": "bank_credit_cards",
        "bankcard": "bank_credit_cards",
        "bankcards": "bank_credit_cards",
        "bankcreditcard": "bank_credit_cards",
        "creditcard": "bank_credit_cards",
        "creditcards": "bank_credit_cards",
        "creditcardcompany": "bank_credit_cards",
        "creditcardcompanies": "bank_credit_cards",
        "nationalcreditcardcompanies": "bank_credit_cards",
        "collectionagency": "collection_agency",
        "collectionagencies": "collection_agency",
        "debtcollectionagency": "collection_agency",
        "debtcollector": "collection_agency",
        "debtcollectors": "collection_agency",
        "creditunion": "credit_union",
        "creditunions": "credit_union",
        "mortgagelender": "mortgage_lender",
        "mortgagelenders": "mortgage_lender",
        "mortgagecompany": "mortgage_lender",
        "mortgagecompanies": "mortgage_lender",
        "mortgagefinance": "mortgage_lender",
        "mortgagefinancing": "mortgage_lender",
        "mortgageloan": "mortgage_lender",
        "mortgageloans": "mortgage_lender",
        "mortgage": "mortgage_lender",
        "mortgagecompaniesfinance": "mortgage_lender",
        "bankmortgageloans": "mortgage_lender",
        "savingsandloansmortgage": "mortgage_lender",
    },
}


def _normalize_alias_key(text: str) -> str:
    cleaned = re.sub(r"[^a-z0-9]+", " ", text.lower())
    return " ".join(cleaned.split())


_ACCOUNT_RATING_ALIASES: Dict[str, str] = {}

for alias in (
    "paid as agreed",
    "pays as agreed",
    "in good standing",
    "up to date",
    "paid on time",
):
    _ACCOUNT_RATING_ALIASES[_normalize_alias_key(alias)] = "current"

for alias in (
    "charged off",
    "chg off",
    "chargeoff",
):
    _ACCOUNT_RATING_ALIASES[_normalize_alias_key(alias)] = "charge off"

for alias in (
    "in collections",
    "sent to collections",
    "collection",
):
    _ACCOUNT_RATING_ALIASES[_normalize_alias_key(alias)] = "collections"

for days, canonical in (("30", "30 days late"), ("60", "60 days late"), ("90", "90 days late")):
    for alias in (
        f"{days} day late",
        f"{days}d late",
        f"late {days}",
    ):
        _ACCOUNT_RATING_ALIASES[_normalize_alias_key(alias)] = canonical

_HISTORY_FIELDS = {"two_year_payment_history", "seven_year_history"}

_SEVEN_YEAR_KEY_ALIASES = {
    "late30": "late30",
    "30 days late": "late30",
    "30 day late": "late30",
    "30": "late30",
    "past due 30": "late30",
    "30 days past due": "late30",
    "late60": "late60",
    "60 days late": "late60",
    "60 day late": "late60",
    "60": "late60",
    "past due 60": "late60",
    "late90": "late90",
    "90 days late": "late90",
    "90 day late": "late90",
    "90": "late90",
    "past due 90": "late90",
    "charge off": "late90",
    "charge offs": "late90",
    "charge-off count": "late90",
    "charge off count": "late90",
    "chargeoffs": "late90",
    "co_count": "late90",
    "co count": "late90",
    "co": "late90",
}


def _coerce_history_count(value: Any) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, (int, float)):
        return int(value)
    if value is None:
        return 0
    text = str(value).strip()
    if not text or text in {"--"}:
        return 0
    match = re.search(r"-?\d+", text)
    if not match:
        return 0
    try:
        return int(match.group())
    except ValueError:
        return 0
_ACCOUNT_NUMBER_FIELDS = {"account_number_display"}

DATE_PATS = [
    (re.compile(r"^\s*(\d{4})-(\d{1,2})-(\d{1,2})\s*$"), "%Y-%m-%d"),  # 2024-01-08
    (re.compile(r"^\s*(\d{1,2})\.(\d{1,2})\.(\d{4})\s*$"), "%d.%m.%Y"),  # 16.8.2022
    (re.compile(r"^\s*(\d{1,2})/(\d{1,2})/(\d{4})\s*$"), "%m/%d/%Y"),  # 1/8/2024
]

_EN_MONTH_MAP = {
    "jan": 1,
    "january": 1,
    "feb": 2,
    "february": 2,
    "mar": 3,
    "march": 3,
    "apr": 4,
    "april": 4,
    "may": 5,
    "jun": 6,
    "june": 6,
    "jul": 7,
    "july": 7,
    "aug": 8,
    "august": 8,
    "sep": 9,
    "sept": 9,
    "september": 9,
    "oct": 10,
    "october": 10,
    "nov": 11,
    "november": 11,
    "dec": 12,
    "december": 12,
}

_HE_MONTH_MAP = {
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

_MONTH_MAPS: Dict[str, Mapping[str, int]] = {
    "en": _EN_MONTH_MAP,
    "he": _HE_MONTH_MAP,
}

def _normalize_year(value: int) -> int:
    if value < 100:
        return 2000 + value if value < 50 else 1900 + value
    return value


def _extract_int_token(token: str) -> Optional[int]:
    match = re.search(r"\d+", token)
    if not match:
        return None
    try:
        return int(match.group())
    except ValueError:
        return None


def _normalize_month_token(token: str, language: str) -> Optional[int]:
    if language == "he":
        cleaned = token.strip().strip('.,"')
        return _HE_MONTH_MAP.get(cleaned)

    cleaned = token.strip().lower().strip(".,")
    return _EN_MONTH_MAP.get(cleaned)


def _select_year_from_tokens(
    tokens: Sequence[str], month_index: int
) -> Tuple[Optional[int], Optional[int]]:
    numeric_tokens = [
        (idx, _extract_int_token(token)) for idx, token in enumerate(tokens)
    ]
    numeric_tokens = [
        (idx, value) for idx, value in numeric_tokens if value is not None
    ]
    if not numeric_tokens:
        return None, None

    for idx, value in numeric_tokens:
        if value >= 1000:
            return _normalize_year(value), idx

    after = [(idx, value) for idx, value in numeric_tokens if idx > month_index]
    before = [(idx, value) for idx, value in numeric_tokens if idx < month_index]

    candidate_pool = after or before or numeric_tokens
    idx, value = candidate_pool[-1]
    return _normalize_year(value), idx


def _select_day_from_tokens(
    tokens: Sequence[str],
    month_index: int,
    year_index: Optional[int],
    prefer_dmy: bool,
) -> Optional[int]:
    candidate_indices = []
    if prefer_dmy:
        candidate_indices.extend([month_index - 1, month_index + 1])
    else:
        candidate_indices.extend([month_index + 1, month_index - 1])

    def _value_at(index: int) -> Optional[int]:
        if index < 0 or index >= len(tokens):
            return None
        if year_index is not None and index == year_index:
            return None
        value = _extract_int_token(tokens[index])
        if value is None:
            return None
        if 1 <= value <= 31:
            return value
        return None

    for index in candidate_indices:
        candidate = _value_at(index)
        if candidate is not None:
            return candidate

    for index, token in enumerate(tokens):
        if index in {month_index, year_index}:
            continue
        value = _extract_int_token(token)
        if value is not None and 1 <= value <= 31:
            return value

    return None


def _parse_month_tokens(
    value: str, month_language: str, prefer_dmy: bool
) -> Optional[str]:
    tokens = [token for token in re.split(r"\s+", value) if token]
    if not tokens:
        return None

    month_index: Optional[int] = None
    month_value: Optional[int] = None
    language = month_language if month_language in _MONTH_MAPS else "en"

    for idx, token in enumerate(tokens):
        candidate = _normalize_month_token(token, language)
        if candidate is None and language != "en":
            candidate = _normalize_month_token(token, "en")
        if candidate is not None:
            month_index = idx
            month_value = candidate
            break

    if month_index is None or month_value is None:
        return None

    year, year_index = _select_year_from_tokens(tokens, month_index)
    if year is None:
        return None

    day = _select_day_from_tokens(tokens, month_index, year_index, prefer_dmy)
    if day is None:
        return None

    try:
        return datetime(year, month_value, day).strftime("%Y-%m-%d")
    except ValueError:
        return None


_NON_ALNUM_RE = re.compile(r"[^A-Za-z0-9]")

_CREDITOR_TYPE_NOISE_WORDS = {
    "the",
    "all",
    "and",
    "of",
    "for",
    "to",
    "by",
    "at",
    "on",
    "company",
    "companies",
    "co",
    "corp",
    "corporation",
    "inc",
    "incorporated",
    "llc",
    "ltd",
    "na",
}


def _get_env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _get_env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


_AMOUNT_TOL_ABS = _get_env_float("AMOUNT_TOL_ABS", 50.0)
_AMOUNT_TOL_RATIO = _get_env_float("AMOUNT_TOL_RATIO", 0.01)
_TOLERANT_AMOUNT_FIELDS = {
    "high_balance",
    "credit_limit",
    "payment_amount",
    "balance_owed",
    "past_due_amount",
}

_DATE_TOLERANCE_DAYS = max(0, _get_env_int("LAST_PAYMENT_DAY_TOL", 5))
_DATE_TOLERANT_FIELDS = {
    "date_opened",
    "closed_date",
    "date_of_last_activity",
    "date_reported",
    "last_payment",
    "last_verified",
}


def _is_missing(v: Any) -> bool:
    return v is None or (isinstance(v, str) and v.strip() in ("", "--"))


def _get_bureau_value(bureaus_json: Mapping[str, Any], field: str, bureau: str) -> Any:
    """Return the raw bureau value for a field, handling history blocks."""

    if field not in ("two_year_payment_history", "seven_year_history"):
        branch = bureaus_json.get(bureau, {})
        if isinstance(branch, Mapping):
            return branch.get(field)
        return None

    if field == "two_year_payment_history":
        # Prefer monthly TSV v2 (list of {month,status})
        new_block = bureaus_json.get("two_year_payment_history_monthly_tsv_v2", {})
        if isinstance(new_block, Mapping):
            value = new_block.get(bureau)
            if value is not None:
                return value

        # Fallback to legacy list[str]
        block = bureaus_json.get("two_year_payment_history", {})
        if isinstance(block, Mapping):
            value = block.get(bureau)
            if value is not None:
                return value
        branch = bureaus_json.get(bureau, {})
        if isinstance(branch, Mapping):
            return branch.get(field)
        return None

    if field == "seven_year_history":
        block = bureaus_json.get("seven_year_history", {})
        if isinstance(block, Mapping):
            value = block.get(bureau)
            if value is not None:
                return value
        branch = bureaus_json.get(bureau, {})
        if isinstance(branch, Mapping):
            return branch.get(field)
        return None

    return None


def normalize_amount(raw: Optional[str]) -> Optional[float]:
    """Parse $, commas, parentheses and sentinel markers into a float."""

    if _is_missing(raw):
        return None
    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        return float(raw)

    text = str(raw).strip()
    if not text:
        return None

    negative = False
    if text.startswith("(") and text.endswith(")"):
        negative = True
        text = text[1:-1]

    cleaned = re.sub(r"[,$\s]", "", text)
    if not cleaned:
        return None

    match = re.search(r"-?\d+(?:\.\d+)?", cleaned)
    if not match:
        return None

    try:
        value = float(match.group())
    except ValueError:
        return None

    if negative and value >= 0:
        value = -value
    return value


def normalize_date(raw: Optional[str]) -> Optional[str]:
    """Normalize several common date formats to YYYY-MM-DD."""

    if raw is None:
        return None

    s = str(raw).strip()
    if s in ("", "--"):
        return None

    context = get_validation_context()
    date_convention = getattr(context, "date_convention", {})
    convention = str(date_convention.get("convention", "MDY")).upper()
    prefer_dmy = convention == "DMY"
    month_language = str(date_convention.get("month_language", "en")).lower()

    for rx, fmt in DATE_PATS:
        if rx.match(s):
            fmt_to_use = fmt
            if fmt == "%m/%d/%Y" and prefer_dmy:
                fmt_to_use = "%d/%m/%Y"
            try:
                dt = datetime.strptime(s, fmt_to_use)
                return dt.strftime("%Y-%m-%d")
            except Exception:
                pass

    try:
        parts = [part for part in re.split(r"[.\-/]", s) if part]
        if len(parts) == 3 and all(part.isdigit() for part in parts):
            first_len = len(parts[0])
            third_len = len(parts[2])
            first, second, third = (int(part) for part in parts)

            if first_len == 4 or first > 31:
                year_source = first
                month_value, day_value = second, third
            else:
                year_source = third
                if third_len == 4 or third > 31:
                    year_source = third
                if prefer_dmy:
                    day_value, month_value = first, second
                else:
                    month_value, day_value = first, second

            year_value = _normalize_year(year_source)
            return datetime(year_value, month_value, day_value).strftime("%Y-%m-%d")
    except Exception:
        pass

    sanitized = re.sub(r"[,\-]", " ", s)
    sanitized = re.sub(r"\s+", " ", sanitized).strip()
    if sanitized:
        parsed = _parse_month_tokens(sanitized, month_language, prefer_dmy)
        if parsed is not None:
            return parsed

    return None


def normalize_term_length(raw: Any) -> Optional[int]:
    """Convert textual term length descriptors into months."""

    if _is_missing(raw):
        return None

    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        return int(round(float(raw)))

    text = str(raw).strip().lower()
    if not text:
        return None

    text = text.replace(",", " ")
    text = text.replace("(", " ").replace(")", " ")

    match = re.search(r"-?\d+(?:\.\d+)?", text)
    if not match:
        return None

    try:
        value = float(match.group())
    except ValueError:
        return None

    if re.search(r"\b(years?|yrs?|yr\b|y\b)", text):
        months = value * 12
    else:
        months = value

    return int(round(months))


def normalize_payment_frequency(raw: Any) -> Optional[str]:
    """Map payment frequency text into a canonical label."""

    if _is_missing(raw):
        return None

    text = str(raw).strip().lower()
    if not text:
        return None

    cleaned = re.sub(r"[^a-z0-9]+", " ", text)
    collapsed = " ".join(cleaned.split())

    direct_map = {
        "monthly": "monthly",
        "per month": "monthly",
        "every month": "monthly",
        "once a month": "monthly",
        "monthly every month": "monthly",
        "month": "monthly",
        "bi weekly": "biweekly",
        "biweekly": "biweekly",
        "every other week": "biweekly",
        "fortnightly": "biweekly",
        "once every two weeks": "biweekly",
        "twice a month": "semimonthly",
        "twice monthly": "semimonthly",
        "two per month": "semimonthly",
        "semi monthly": "semimonthly",
        "semimonthly": "semimonthly",
        "semi monthly pay": "semimonthly",
        "weekly": "weekly",
        "per week": "weekly",
        "every week": "weekly",
        "once a week": "weekly",
        "quarterly": "quarterly",
        "per quarter": "quarterly",
        "every quarter": "quarterly",
        "annually": "annually",
        "annual": "annually",
        "per year": "annually",
        "every year": "annually",
        "once a year": "annually",
        "yearly": "annually",
    }

    if collapsed in direct_map:
        return direct_map[collapsed]

    # Handle descriptive phrasing that includes multiple tokens.
    if "month" in collapsed:
        if "twice" in collapsed or "two" in collapsed:
            return "semimonthly"
        return "monthly"
    if "week" in collapsed:
        if "other" in collapsed or "two" in collapsed:
            return "biweekly"
        return "weekly"
    if "quarter" in collapsed:
        return "quarterly"
    if "year" in collapsed or "annual" in collapsed:
        return "annually"

    return _normalize_text(raw)


def _creditor_type_lookup_keys(text: str) -> Sequence[str]:
    lowered = text.lower()
    cleaned = re.sub(r"[^a-z0-9]+", " ", lowered)
    tokens = [
        token
        for token in cleaned.split()
        if token and token not in _CREDITOR_TYPE_NOISE_WORDS
    ]
    if not tokens:
        return ()

    collapsed = "".join(tokens)
    spaced = " ".join(tokens)

    if collapsed == spaced.replace(" ", ""):
        return (collapsed, spaced)

    return (collapsed, spaced, *tokens)


def _normalize_text(raw: Any) -> Optional[str]:
    if _is_missing(raw):
        return None
    text = str(raw).strip().lower()
    if not text:
        return None
    cleaned = re.sub(r"[^a-z0-9]+", " ", text)
    collapsed = " ".join(cleaned.split())
    return collapsed or None


def _apply_account_rating_alias(value: Any) -> Any:
    if _is_missing(value):
        return None
    key = _normalize_alias_key(str(value))
    canonical = _ACCOUNT_RATING_ALIASES.get(key)
    if canonical:
        return canonical
    return value


def normalize_enum(field: str, raw: Optional[str]) -> Optional[str]:
    """Normalize enums using domain-specific alias tables."""

    if _is_missing(raw):
        return None

    text = str(raw).strip().lower()
    if not text:
        return None

    domain = _ENUM_DOMAINS.get(field, {})
    compact = _NON_ALNUM_RE.sub("", text)
    normalized_space = " ".join(text.split())

    candidates: list[str] = []
    seen: set[str] = set()

    def add_candidate(value: Optional[str]) -> None:
        if not value:
            return
        if value in seen:
            return
        candidates.append(value)
        seen.add(value)

    add_candidate(compact)
    add_candidate(text)
    add_candidate(normalized_space)

    if field == "creditor_type":
        for key in _creditor_type_lookup_keys(text):
            add_candidate(key)

    for key in candidates:
        if key in domain:
            return domain[key]

    return normalized_space


_ACCOUNT_NUMBER_MISSING_WORDS = {
    "na",
    "none",
    "notavailable",
    "notprovided",
    "unknown",
}
_ACCOUNT_NUMBER_MASK_ONLY_RE = re.compile(r"^[\s\-#*xX]+$")


def _account_number_mask_class(text: str, digits: Sequence[str]) -> str:
    stripped = text.strip()
    if not stripped:
        return "missing"

    upper = stripped.upper()

    if not digits:
        letters = re.sub(r"[^A-Z]+", "", upper)
        if not letters or letters.lower() in _ACCOUNT_NUMBER_MISSING_WORDS:
            return "missing"
        if _ACCOUNT_NUMBER_MASK_ONLY_RE.match(upper):
            return "missing"
        return "nondigit"

    if _ACCOUNT_NUMBER_MASK_ONLY_RE.match(upper):
        return "missing"

    digit_count = len(digits)
    has_mask_char = bool(re.search(r"[#*X]", upper))
    has_alpha = bool(re.search(r"[A-Z]", upper))

    if has_mask_char:
        if digit_count <= 4:
            return "masked_last4"
        return "masked"

    if has_alpha:
        if digit_count <= 4:
            return "text_last4"
        return "alphanumeric"

    if digit_count <= 4:
        return "last4_only"

    return "unmasked"


def normalize_account_number_display(
    raw: Optional[str],
) -> Optional[Dict[str, Optional[str]]]:
    """Return a display/last4 structure for account numbers."""

    if _is_missing(raw):
        return None

    text = str(raw).strip()
    digits = re.findall(r"\d", text)
    mask_class = _account_number_mask_class(text, digits)

    if mask_class == "missing":
        return None

    last4 = "".join(digits[-4:]) if digits else None
    if last4 and len(last4) != 4:
        last4 = None

    return {"display": text, "last4": last4, "mask_class": mask_class}


def normalize_two_year_history(raw: Any) -> Dict[str, Any] | None:
    """Normalize two-year payment history tokens and summary counts."""

    raw_tokens: list[str] = []
    if isinstance(raw, str):
        raw_tokens = [part.strip() for part in re.split(r"[\s,]+", raw) if part.strip()]
    elif isinstance(raw, Sequence) and not isinstance(raw, (bytes, bytearray, str)):
        for entry in raw:
            if isinstance(entry, Mapping):
                status = entry.get("status") or entry.get("value")
                if status:
                    raw_tokens.append(str(status).strip())
            elif entry is not None:
                raw_tokens.append(str(entry).strip())

    tokens: list[str] = []
    counts = {"CO": 0, "late30": 0, "late60": 0, "late90": 0}

    for token in raw_tokens:
        s = token.strip().upper()
        if not s:
            continue
        tokens.append(s)
        if s == "CO":
            counts["CO"] += 1
        elif s in ("30", "LATE30"):
            counts["late30"] += 1
        elif s in ("60", "LATE60"):
            counts["late60"] += 1
        elif s in ("90", "120", "150", "180", "LATE90"):
            counts["late90"] += 1

    if not tokens and all(count == 0 for count in counts.values()):
        return None

    return {"tokens": tokens, "counts": counts}


def normalize_seven_year_history(raw: Any) -> Dict[str, int] | None:
    """Normalize seven-year history counters to late30/late60/late90."""

    counts = {"late30": 0, "late60": 0, "late90": 0}

    if isinstance(raw, Mapping):
        items = raw.items()
    else:
        items = []

    for key, value in items:
        key_text = str(key).strip().lower()
        key_text = re.sub(r"[\s_-]+", " ", key_text)
        canonical = _SEVEN_YEAR_KEY_ALIASES.get(key_text)
        if not canonical:
            continue
        counts[canonical] += _coerce_history_count(value)

    if not isinstance(raw, Mapping):
        tokens: list[str] = []
        if isinstance(raw, str):
            tokens = [part.strip() for part in re.split(r"[\s,]+", raw) if part.strip()]
        elif isinstance(raw, Sequence) and not isinstance(raw, (bytes, bytearray, str)):
            tokens = [str(part).strip() for part in raw if part]
        for token in tokens:
            canonical = _SEVEN_YEAR_KEY_ALIASES.get(token.lower())
            if canonical:
                counts[canonical] += 1

    if all(value == 0 for value in counts.values()):
        return None

    return counts


def _looks_like_amount(field: str) -> bool:
    lowered = field.lower()
    if any(lowered.endswith(suffix) for suffix in _AMOUNT_FIELD_HINTS):
        return True
    return any(token in lowered for token in ("amount", "balance", "limit", "payment"))


def _normalize_field(field: str, value: Any) -> Any:
    if field in _HISTORY_FIELDS:
        if field == "two_year_payment_history":
            return normalize_two_year_history(value)
        return normalize_seven_year_history(value)
    if field in _ACCOUNT_NUMBER_FIELDS:
        return normalize_account_number_display(value)
    if field in _ENUM_DOMAINS:
        return normalize_enum(field, value)
    if field == "account_rating":
        aliased = _apply_account_rating_alias(value)
        return _normalize_text(aliased)
    if field == "term_length":
        return normalize_term_length(value)
    if field == "payment_frequency":
        return normalize_payment_frequency(value)
    if field in {"last_payment", "last_verified"}:
        return normalize_date(value)
    if "date" in field.lower():
        return normalize_date(value)
    if _looks_like_amount(field):
        return normalize_amount(value)
    return _normalize_text(value)


def _freeze_value(field: str, value: Any) -> Any:
    if value is None:
        return None
    if field == "two_year_payment_history":
        if isinstance(value, Mapping):
            counts = value.get("counts", {})
            tokens = tuple(value.get("tokens", []))
            return (
                counts.get("CO", 0),
                counts.get("late30", 0),
                counts.get("late60", 0),
                counts.get("late90", 0),
                tokens,
            )
    if field == "seven_year_history":
        if isinstance(value, Mapping):
            return (
                value.get("late30", 0),
                value.get("late60", 0),
                value.get("late90", 0),
            )
    if isinstance(value, Mapping):
        return tuple(sorted((key, _freeze_value(field, item)) for key, item in value.items()))
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(_freeze_value(field, item) for item in value)
    return value


def _determine_consensus(field: str, groups: Mapping[Any, Sequence[str]]) -> Tuple[str, list[str]]:
    items = [(key, list(bureaus)) for key, bureaus in groups.items() if bureaus]
    if not items:
        return "unanimous", []

    if len(items) == 1:
        return "unanimous", []

    total = sum(len(bureaus) for _, bureaus in items)
    sorted_items = sorted(items, key=lambda item: (-len(item[1]), str(item[0])))
    top_key, top_bureaus = sorted_items[0]
    top_count = len(top_bureaus)

    if top_count > total / 2:
        disagreeing = [bureau for key, bureaus in items if key != top_key for bureau in bureaus]
        disagreeing.sort()
        return "majority", disagreeing

    disagreeing = [bureau for _, bureaus in items for bureau in bureaus]
    disagreeing.sort()
    return "split", disagreeing


def _amounts_equal_with_tolerance(a: float, b: float) -> bool:
    diff = abs(a - b)
    scale = max(abs(a), abs(b))
    tolerance = max(_AMOUNT_TOL_ABS, _AMOUNT_TOL_RATIO * scale)
    return diff <= tolerance


def _select_amount_group_key(groups: Mapping[Any, Sequence[str]], value: float) -> float:
    for key in list(groups.keys()):
        if isinstance(key, (int, float)) and _amounts_equal_with_tolerance(value, float(key)):
            return float(key)
    return value


def _parse_normalized_date(value: Any) -> Optional[date]:
    if isinstance(value, str):
        try:
            return datetime.strptime(value, "%Y-%m-%d").date()
        except ValueError:
            return None
    return None


def _dates_equal_with_tolerance(a: date, b: date) -> bool:
    if _DATE_TOLERANCE_DAYS <= 0:
        return a == b
    return abs((a - b).days) <= _DATE_TOLERANCE_DAYS


def _select_date_group_key(groups: Mapping[Any, Sequence[str]], value: date) -> date:
    for key in list(groups.keys()):
        if isinstance(key, date) and _dates_equal_with_tolerance(value, key):
            return key
    return value


def _account_number_letters(value: Any) -> str:
    if value is None:
        return ""
    return "".join(ch for ch in str(value) if ch.isalpha()).upper()


def _account_number_relation(a: Any, b: Any) -> str:
    level, debug = acctnum_match_level(str(a or ""), str(b or ""))
    if level == "exact_or_known_match":
        return "match"

    reason = (debug or {}).get("why", "")
    if reason == "visible_digits_conflict":
        return "conflict"

    strict_level, strict_debug = acctnum_level(a, b)
    if strict_level == "exact_or_known_match":
        return "match"

    strict_reason = (strict_debug or {}).get("why", "")
    if strict_reason in {"digit_conflict", "alnum_conflict"}:
        return "conflict"

    if strict_reason == "empty":
        letters_a = _account_number_letters(a)
        letters_b = _account_number_letters(b)
        if letters_a and letters_b and letters_a != letters_b:
            return "conflict"

    return "insufficient"


def _populate_account_number_groups(
    groups: MutableMapping[Any, list[str]],
    bureaus: Sequence[str],
    values: Mapping[str, Any],
) -> None:
    account_groups: list[Dict[str, Any]] = []

    for bureau in bureaus:
        value = values.get(bureau)
        best_group: Optional[Dict[str, Any]] = None
        best_priority = -1

        for group in account_groups:
            members: list[str] = group["members"]
            statuses = [
                _account_number_relation(value, values.get(other))
                for other in members
            ]

            if any(status == "conflict" for status in statuses):
                continue

            priority = 1 if any(status == "match" for status in statuses) else 0
            if priority > best_priority:
                best_priority = priority
                best_group = group

        if best_group is None:
            best_group = {"key": object(), "members": []}
            account_groups.append(best_group)

        best_group["members"].append(bureau)

    for group in account_groups:
        groups.setdefault(group["key"], []).extend(group["members"])


def compute_field_consistency(bureaus_json: Dict[str, Any]) -> Dict[str, Any]:
    """Compute normalized values and consensus for every field across bureaus."""

    fields = set(_HISTORY_FIELDS)

    for bureau in _BUREAU_KEYS:
        branch = bureaus_json.get(bureau)
        if isinstance(branch, Mapping):
            fields.update(branch.keys())

    results: Dict[str, Any] = {}

    for field in sorted(fields):
        normalized: MutableMapping[str, Any] = {}
        raw: MutableMapping[str, Any] = {}
        groups: Dict[Any, list[str]] = {}
        missing_bureaus: list[str] = []
        present_bureaus: list[str] = []
        account_number_present: list[str] = []
        account_number_values: Dict[str, Any] = {}

        for bureau in _BUREAU_KEYS:
            value = _get_bureau_value(bureaus_json, field, bureau)
            raw[bureau] = value
            norm_value = _normalize_field(field, value)
            normalized[bureau] = norm_value

            is_missing = _is_missing(value)
            if not is_missing and field in _HISTORY_FIELDS and norm_value is None:
                is_missing = True
            if not is_missing and field == "account_number_display" and norm_value is None:
                is_missing = True

            if is_missing:
                missing_bureaus.append(bureau)
                key = ("__missing__",)
                groups.setdefault(key, []).append(bureau)
            else:
                present_bureaus.append(bureau)
                if field == "account_number_display":
                    account_number_present.append(bureau)
                    account_number_values[bureau] = value
                    continue
                frozen = _freeze_value(field, norm_value)
                date_key = None
                if field in _DATE_TOLERANT_FIELDS:
                    date_key = _parse_normalized_date(norm_value)
                if date_key is not None:
                    key = _select_date_group_key(groups, date_key)
                elif (
                    field in _TOLERANT_AMOUNT_FIELDS
                    and isinstance(frozen, (int, float))
                ):
                    key = _select_amount_group_key(groups, float(frozen))
                else:
                    key = frozen
                groups.setdefault(key, []).append(bureau)

        if field == "account_number_display" and account_number_present:
            _populate_account_number_groups(groups, account_number_present, account_number_values)

        if all(norm is None for norm in normalized.values()):
            continue

        consensus, disagreeing = _determine_consensus(field, groups)
        results[field] = {
            "consensus": consensus,
            "normalized": dict(normalized),
            "raw": dict(raw),
            "disagreeing_bureaus": disagreeing,
            "missing_bureaus": sorted({bureau for bureau in missing_bureaus}) if present_bureaus else [],
        }

    return results


def compute_inconsistent_fields(bureaus_json: Mapping[str, Any]) -> Dict[str, Any]:
    """Return fields whose consensus is not unanimous."""

    consistency = compute_field_consistency(dict(bureaus_json))
    return {
        field: info
        for field, info in consistency.items()
        if info.get("consensus") != "unanimous"
    }

