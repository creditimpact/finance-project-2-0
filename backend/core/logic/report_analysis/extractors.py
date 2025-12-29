import re
from backend.core.logic.utils.names_normalization import normalize_creditor_name


def _parse_account_sections(text: str) -> dict:
    """Split ``text`` into account-like sections and extract fields.

    Returns a mapping of normalized account names to dictionaries containing
    ``account_number``, ``status`` and ``dofd`` when present. Sections that do
    not contain any of these fields are ignored.
    """
    sections = re.split(r"\n{2,}", text)
    results: dict[str, dict] = {}
    for sec in sections:
        lines = [line.strip() for line in sec.splitlines() if line.strip()]
        if not lines:
            continue
        name_norm = normalize_creditor_name(lines[0])
        account_number = status = dofd = None
        for line in lines[1:]:
            lower = line.lower()
            if lower.startswith("account number") or lower.startswith("acct #"):
                account_number = line.split(":", 1)[1].strip()
            elif lower.startswith("status"):
                status = line.split(":", 1)[1].strip()
            elif "date of first delinquency" in lower or lower.startswith("dofd"):
                dofd = line.split(":", 1)[1].strip()
        if account_number or status or dofd:
            results[name_norm] = {
                "account_number": account_number,
                "status": status,
                "dofd": dofd,
            }
    return results


def extract_account_number_masks(text: str) -> dict[str, str]:
    """Return mapping of account names to account number masks."""
    parsed = _parse_account_sections(text)
    return {
        name: vals["account_number"]
        for name, vals in parsed.items()
        if vals.get("account_number")
    }


def extract_account_statuses(text: str) -> dict[str, str]:
    """Return mapping of account names to statuses."""
    parsed = _parse_account_sections(text)
    return {
        name: vals["status"]
        for name, vals in parsed.items()
        if vals.get("status")
    }


def extract_dofd(text: str) -> dict[str, str]:
    """Return mapping of account names to Date of First Delinquency."""
    parsed = _parse_account_sections(text)
    return {
        name: vals["dofd"]
        for name, vals in parsed.items()
        if vals.get("dofd")
    }


def extract_inquiry_dates(text: str) -> dict[str, str]:
    """Return mapping of inquiry creditor names to dates."""
    pattern = re.compile(
        r"(?P<name>[^\n]+)\nInquiry Date:\s*(?P<date>\d{1,2}/\d{1,2}/\d{4})",
        re.IGNORECASE,
    )
    results: dict[str, str] = {}
    for match in pattern.finditer(text):
        name = normalize_creditor_name(match.group("name").strip())
        results[name] = match.group("date").strip()
    return results
