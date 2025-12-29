"""Text parsing helpers for account histories and flags."""

from __future__ import annotations

import re
from collections import deque

LATE_PATTERN = re.compile(
    r"(\d+\s*x\s*30|\d+\s*x\s*60|30[-\s]*day[s]?\s*late|60[-\s]*day[s]?\s*late|90[-\s]*day[s]?\s*late|late payment|past due)",
    re.I,
)

NO_LATE_PATTERN = re.compile(
    r"(no late payments|never late|never been late|no history of late)",
    re.I,
)

GENERIC_NAME_RE = re.compile(r"days?\s+late|payment\s+history|year\s+history", re.I)

# Relaxed heading candidate and bureau header patterns (exact definitions)
HEADING_CANDIDATE_RE = re.compile(
    r"^(?!transunion$|experian$|equifax$)"
    r"(?!.*(days\s+late|payment\s+history|year\s+history))"
    r"(?=[A-Za-z0-9])[A-Za-z0-9/&\-',\. ]{3,60}$",
    re.IGNORECASE,
)
BUREAU_HDR_RE = re.compile(r"^(TransUnion|Experian|Equifax)\b", re.IGNORECASE)


# Internal helpers ---------------------------------------------------------


def _has_late_flag(text: str) -> bool:
    """Return True when the text indicates late payments."""
    clean = str(text or "").lower()
    if NO_LATE_PATTERN.search(clean):
        return False
    if re.search(r"0\s*x\s*(30|60|90)", clean):
        return False
    return bool(LATE_PATTERN.search(clean))


_MONTHS = {
    "jan",
    "feb",
    "mar",
    "apr",
    "may",
    "jun",
    "jul",
    "aug",
    "sep",
    "sept",
    "oct",
    "nov",
    "dec",
}


def _is_calendar_line(line: str) -> bool:
    tokens = re.findall(r"[A-Za-z]+|\d+", line.lower())
    has_month = False
    for t in tokens:
        if t.isdigit():
            continue
        if t in _MONTHS:
            has_month = True
            continue
        return False
    return has_month


def _potential_account_name(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    lower = stripped.lower()
    if lower == "none reported" or _is_calendar_line(stripped):
        return False
    if re.fullmatch(r"(ok\b\s*){2,}", lower):
        return False
    tokens = lower.split()
    if len(tokens) >= 2 and len(set(tokens)) == 1 and len(tokens[0]) <= 3:
        return False
    # Relaxed candidate: allow mixed case, punctuation and digits
    if not HEADING_CANDIDATE_RE.match(stripped):
        return False
    return (
        len(stripped) >= 3
        and not re.match(r"(TransUnion|Experian|Equifax)\b", stripped, re.I)
        and re.search(r"[A-Z]", stripped)
        and not GENERIC_NAME_RE.search(stripped)
    )


def _parse_late_counts(segment: str) -> dict[str, int]:
    """Return late payment counts avoiding account number artifacts."""
    counts: dict[str, int] = {}
    pattern = re.compile(r"(?<!\d)(30|60|90)[\s-]*:?\s*(\d+)(?!\d)")
    for part in pattern.finditer(segment):
        num = int(part.group(2))
        if num > 12:
            print(f"[~] Ignoring unrealistic late count {num} for {part.group(1)}")
            continue
        counts[part.group(1)] = num
    return counts


ACCOUNT_FIELD_RE = re.compile(
    r"(account\s*(?:no\.|number|#))|balance|opened|closed",
    re.I,
)


def _has_account_fields(block: list[str]) -> bool:
    for line in block[1:4]:
        if ACCOUNT_FIELD_RE.search(line):
            return True
    return False


def extract_account_blocks(text: str, debug: bool = False) -> list[list[str]]:
    """Return blocks of lines corresponding to individual accounts."""

    raw_lines = text.splitlines()
    lines: list[str] = []
    i = 0
    while i < len(raw_lines):
        current = raw_lines[i].strip()
        if i + 1 < len(raw_lines):
            nxt = raw_lines[i + 1].strip()
            if (
                current.isupper()
                and nxt.isupper()
                and len(current.split()) <= 2
                and len(nxt.split()) <= 2
            ):
                merged = f"{current} {nxt}"
                if debug:
                    print(
                        f"[~] Merged split heading '{current}' + '{nxt}' -> '{merged}'"
                    )
                lines.append(merged)
                i += 2
                continue
        lines.append(current)
        i += 1

    blocks: list[list[str]] = []
    current_block: list[str] = []
    capturing = False
    await_equifax_counts = False

    recent: deque[str] = deque(maxlen=8)
    for idx, line in enumerate(lines):
        # Secondary heuristic: if a bureau header appears before a heading, pick
        # the nearest previous relaxed heading candidate.
        if not capturing and BUREAU_HDR_RE.match(line):
            candidate: str | None = None
            for prev in reversed(recent):
                prev_s = prev.strip()
                if HEADING_CANDIDATE_RE.match(prev_s):
                    candidate = prev_s
                    break
            if candidate:
                current_block = [candidate]
                capturing = True
                await_equifax_counts = False
                if debug:
                    print(f"[+] Start block (heuristic) '{candidate}'")

        if _potential_account_name(line):
            if capturing:
                if _has_account_fields(current_block):
                    blocks.append(current_block)
                elif debug:
                    print(
                        f"[~] Discarded block '{current_block[0]}' (no account fields)"
                    )
            current_block = [line]
            capturing = True
            await_equifax_counts = False
            if debug:
                print(f"[+] Start block '{line}'")
            continue

        if not capturing:
            recent.append(line)
            continue

        current_block.append(line)
        recent.append(line)

        if re.match(r"Equifax\b", line, re.I):
            await_equifax_counts = True
            if all(k in line for k in ("30:", "60:", "90:")):
                if _has_account_fields(current_block):
                    blocks.append(current_block)
                elif debug:
                    print(
                        f"[~] Discarded block '{current_block[0]}' (no account fields)"
                    )
                if debug:
                    print(
                        f"[INFO] End block '{current_block[0]}' after Equifax counts line"
                    )
                current_block = []
                capturing = False
                await_equifax_counts = False
            continue

        if await_equifax_counts and all(k in line for k in ("30:", "60:", "90:")):
            if _has_account_fields(current_block):
                blocks.append(current_block)
            elif debug:
                print(f"[~] Discarded block '{current_block[0]}' (no account fields)")
            if debug:
                print(
                    f"[INFO] End block '{current_block[0]}' after Equifax counts line"
                )
            current_block = []
            capturing = False
            await_equifax_counts = False
            continue

    if capturing and current_block:
        if _has_account_fields(current_block):
            blocks.append(current_block)
        elif debug:
            print(f"[~] Discarded block '{current_block[0]}' (no account fields)")
        if debug:
            print(f"[INFO] End block '{current_block[0]}' (EOF)")

    return blocks


def extract_account_headings(text: str) -> list[tuple[str, str]]:
    """Return unique normalized account headings found in ``text``.

    The function leverages :func:`extract_account_blocks` to locate candidate
    account sections and then normalizes their headings using
    :func:`norm.normalize_heading`.  Only the first
    occurrence of each normalized name is retained in the returned list.

    Parameters
    ----------
    text:
        Raw text extracted from a credit report PDF.

    Returns
    -------
    list[tuple[str, str]]
        Tuples of ``(normalized_name, raw_heading)``.
    """

    from .norm import normalize_heading

    headings: list[tuple[str, str]] = []
    seen: set[str] = set()
    for block in extract_account_blocks(text):
        if not block:
            continue
        raw = block[0].strip()
        norm = normalize_heading(raw)
        if norm and norm not in seen:
            headings.append((norm, raw))
            seen.add(norm)
    return headings


def parse_late_history_from_block(
    block: list[str], debug: bool = False
) -> tuple[dict[str, dict[str, int]], dict[str, str]]:
    """Parse bureau late-payment counts and raw history strings from a block."""

    details: dict[str, dict[str, int]] = {}
    grids: dict[str, str] = {}
    pending_bureau: str | None = None
    found_bureau = False

    for line in block:
        clean = line.strip()
        bureau_match = re.match(r"(TransUnion|Experian|Equifax)\s*:?(.*)", clean, re.I)
        if bureau_match:
            bureau = bureau_match.group(1).title()
            rest = bureau_match.group(2)
            counts = _parse_late_counts(rest)
            grids[bureau] = rest.strip()
            found_bureau = True
            if counts:
                details[bureau] = counts
                pending_bureau = None
            else:
                pending_bureau = bureau
            continue

        if pending_bureau:
            counts = _parse_late_counts(clean)
            if counts:
                details[pending_bureau] = counts
            if pending_bureau in grids:
                grids[pending_bureau] = f"{grids[pending_bureau]} {clean}".strip()
            else:
                grids[pending_bureau] = clean
            if not counts and debug:
                print(
                    f"[~] Missing counts for {pending_bureau} in block starting '{block[0]}'"
                )
            pending_bureau = None

    if not found_bureau and debug:
        print(f"[~] No bureau lines found in block starting '{block[0]}'")

    return details, grids


def extract_late_history_blocks(
    text: str,
    known_accounts: set[str] | None = None,
    return_raw_map: bool = False,
    debug: bool = False,
    timeout: int = 4,
) -> dict:
    """Parse late payment history blocks and link them to accounts."""

    account_map: dict[str, dict[str, dict[str, int]]] = {}
    raw_map: dict[str, str] = {}
    grid_map: dict[str, dict[str, str]] = {}

    def norm(name: str) -> str:
        from .norm import normalize_heading

        return normalize_heading(name)

    normalized_accounts = {norm(n): n for n in known_accounts or []}

    for block in extract_account_blocks(text, debug=debug):
        if not block:
            continue
        heading_raw = block[0].strip()
        acc_norm = norm(heading_raw)

        if known_accounts and acc_norm not in normalized_accounts:
            from difflib import get_close_matches

            match = get_close_matches(
                acc_norm, normalized_accounts.keys(), n=1, cutoff=0.8
            )
            if not match:
                if debug:
                    print(f"[~] Skipping unrecognized account '{heading_raw}'")
                continue
            if debug:
                print(f"[~] Fuzzy matched '{acc_norm}' -> '{match[0]}'")
            acc_norm = match[0]

        details, grids = parse_late_history_from_block(block, debug=debug)
        if not details:
            if debug:
                print(f"[~] Dropped candidate '{acc_norm}' (no details)")
            continue

        if not GENERIC_NAME_RE.search(acc_norm):
            account_map[acc_norm] = details
            raw_map.setdefault(acc_norm, heading_raw)
            if grids:
                grid_map[acc_norm] = grids
            if debug:
                found = sorted(details.keys())
                missing = [
                    b for b in {"Transunion", "Experian", "Equifax"} if b not in details
                ]
                print(
                    f"[INFO] End block '{heading_raw}' found={found or []} missing={missing or []}"
                )
                print(f"[INFO] Parsed block '{heading_raw}' -> {details}")

    for norm_name, bureaus in account_map.items():
        raw_name = raw_map.get(norm_name, norm_name)
        print(f"[INFO] Parsed block '{raw_name}' -> {bureaus}")

    if return_raw_map:
        return account_map, raw_map, grid_map
    return account_map


def _total_lates(info) -> int:
    """Return the sum of all late payment counts across bureaus."""
    total = 0
    if isinstance(info, dict):
        for bureau_vals in info.values():
            if isinstance(bureau_vals, dict):
                for v in bureau_vals.values():
                    try:
                        total += int(v)
                    except (TypeError, ValueError):
                        continue
    return total


def has_late_indicator(acc: dict) -> bool:
    """Return True if account has explicit late payment info or matching text."""
    late = acc.get("late_payments")
    total_lates = _total_lates(late)
    if total_lates > 0:
        return True
    if "Late Payments" in acc.get("flags", []):
        # Some bureaus mark accounts with this flag even when no counts are
        # reported; ignore the flag if all counts are zero
        return False
    text = " ".join(
        str(acc.get(f, "")) for f in ["status", "remarks", "advisor_comment", "flags"]
    )
    if NO_LATE_PATTERN.search(text.lower()):
        return False
    return _has_late_flag(text)


CHARGEOFF_RE = re.compile(r"charge[- ]?off", re.I)
COLLECTION_RE = re.compile(r"collection", re.I)


def enforce_collection_status(acc: dict) -> None:
    """Ensure accounts mentioning both charge-off and collection are tagged as a collection.

    The original status string from the credit report is preserved in
    ``reported_status`` so downstream logic (e.g., letter generation) can display
    the exact wording. Only classification fields like ``account_type`` and
    ``flags`` are modified.
    """

    text = " ".join(
        str(acc.get(field, ""))
        for field in [
            "status",
            "remarks",
            "account_type",
            "account_status",
            "advisor_comment",
            "flags",
            "tags",
        ]
    ).lower()

    if CHARGEOFF_RE.search(text) and COLLECTION_RE.search(text):
        if acc.get("status") and "reported_status" not in acc:
            acc["reported_status"] = acc["status"]
        # Preserve the status field so the full text can be shown in letters.
        if (
            acc.get("account_type")
            and "collection" not in str(acc["account_type"]).lower()
        ):
            acc["account_type"] = "Collection"
        else:
            acc.setdefault("account_type", "Collection")
        for field in ("flags", "tags"):
            val = acc.get(field)
            if val is None:
                acc[field] = ["Collection"]
            elif isinstance(val, list):
                if not any("collection" in str(v).lower() for v in val):
                    val.append("Collection")
            else:
                if "collection" not in str(val).lower():
                    acc[field] = [val, "Collection"]
