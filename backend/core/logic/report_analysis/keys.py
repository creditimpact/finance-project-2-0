import hashlib
from typing import Optional


def normalize_issuer(name: str) -> str:
    """Normalize issuer names by removing basic punctuation and collapsing spaces."""
    if not name:
        return ""
    cleaned = name.replace(".", " ").replace("-", " ").replace("/", " ")
    tokens = cleaned.split()
    collapsed: list[str] = []
    i = 0
    while i < len(tokens):
        if len(tokens[i]) == 1:
            letters: list[str] = []
            while i < len(tokens) and len(tokens[i]) == 1:
                letters.append(tokens[i])
                i += 1
            collapsed.append("".join(letters))
        else:
            collapsed.append(tokens[i])
            i += 1
    return " ".join(collapsed).upper().strip()


def compute_logical_account_key(
    issuer: Optional[str],
    last4: Optional[str],
    account_type: Optional[str],
    opened_date: Optional[str],
) -> Optional[str]:
    """Compute a stable logical key for accounts.

    Returns None when there are no anchor fields (issuer, last4, opened_date).
    """
    issuer_norm = normalize_issuer(issuer or "")
    acct = (account_type or "").strip().upper()
    opened = (opened_date or "").strip()
    last4_clean = (last4 or "").strip()
    if not issuer_norm and not last4_clean and not opened:
        return None
    if last4_clean:
        basis = f"L4:{last4_clean}|I:{issuer_norm}|T:{acct}|D:{opened}"
    else:
        basis = f"NO4|I:{issuer_norm}|T:{acct}|D:{opened}"
    return hashlib.sha1(basis.encode("utf-8")).hexdigest()[:16]
