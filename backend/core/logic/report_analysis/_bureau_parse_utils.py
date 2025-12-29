from __future__ import annotations

import re
from typing import Iterable, Literal, Tuple, List, Dict


def norm(s: str) -> str:
    """Normalize OCR text for robust matching.

    - Replace NBSP/®/™/tabs with spaces
    - Collapse whitespace
    - Strip leading/trailing spaces
    """
    s = (s or "").replace("\u00A0", " ").replace("®", " ").replace("™", " ")
    s = s.replace("\t", " ")
    s = re.sub(r"\s+", " ", s.strip())
    return s


def _upper_compact(s: str) -> str:
    return re.sub(r"\s+", "", norm(s).upper())


def is_bureau(s: str) -> Literal["transunion", "experian", "equifax", None]:
    u = _upper_compact(s)
    if "TRANSUNION" in u or re.search(r"TRANS\s*UNION", u):
        return "transunion"
    if "EXPERIAN" in u:
        return "experian"
    if "EQUIFAX" in u:
        return "equifax"
    return None


LABELS_TOP: List[str] = [
    "Account #",
    "High Balance",
    "Last Verified",
    "Date of Last Activity",
    "Date Reported",
    "Date Opened",
    "Balance Owed",
    "Closed Date",
    "Account Rating",
    "Account Description",
    "Dispute Status",
    "Creditor Type",
    "Original Creditor",
]

_LABEL_VARIANTS_TOP = {
    "Original Creditor 01": "Original Creditor",
    "Original Creditor 02": "Original Creditor",
    "Orig. Creditor": "Original Creditor",
}

LABELS_BOTTOM: List[str] = [
    "Account Status",
    "Payment Status",
    "Creditor Remarks",
    "Payment Amount",
    "Last Payment",
    "Term Length",
    "Past Due Amount",
    "Account Type",
    "Payment Frequency",
    "Credit Limit",
]


def _label_to_pattern(label: str) -> re.Pattern[str]:
    """Return a tolerant regex for a canonical label (accept trailing ':')."""
    # Replace spaces with \s+; allow hyphen or en dash equivalence
    esc = (
        label.replace("-", "[-–]")
        .replace("#", r"\s*#")
        .replace(" ", r"\s+")
    )
    return re.compile(rf"^{esc}\s*:?$", re.IGNORECASE)


_TOP_PATTERNS: List[Tuple[re.Pattern[str], str]] = [
    (_label_to_pattern(lab), lab) for lab in LABELS_TOP
]
_TOP_PATTERNS.extend(
    (_label_to_pattern(alias), canonical)
    for alias, canonical in _LABEL_VARIANTS_TOP.items()
)
_BOTTOM_PATTERNS: List[Tuple[re.Pattern[str], str]] = [
    (_label_to_pattern(lab), lab) for lab in LABELS_BOTTOM
]


def is_label(s: str) -> Tuple[Literal["top", "bottom", None], str | None]:
    """Return (kind, canonical_label) if ``s`` is a recognized label, else (None, None)."""
    t = norm(s)
    for pat, lab in _TOP_PATTERNS:
        if pat.match(t):
            return "top", lab
    for pat, lab in _BOTTOM_PATTERNS:
        if pat.match(t):
            return "bottom", lab
    return None, None


def find_label_groups(lines: Iterable[str]) -> List[Dict[str, object]]:
    """Detect contiguous groups of top/bottom labels within ``lines``.

    Returns a list of dicts: {"start": int, "end": int, "type": "top"|"bottom", "labels": [..]}
    where end is inclusive index of the last label in the group.
    """
    out: List[Dict[str, object]] = []
    cur: Dict[str, object] | None = None
    cur_type: Literal["top", "bottom", None] = None
    cur_labels: List[str] = []
    start_idx: int | None = None

    ls = list(lines)
    for idx, raw in enumerate(ls):
        kind, canon = is_label(raw)
        if kind is None:
            if cur is not None:
                out.append({
                    "start": start_idx if start_idx is not None else idx,
                    "end": idx - 1,
                    "type": cur_type,
                    "labels": list(cur_labels),
                })
                cur = None
                cur_type = None
                cur_labels = []
                start_idx = None
            continue
        if cur is None:
            # start new group
            cur = {}
            cur_type = kind
            start_idx = idx
            cur_labels = [canon] if canon else []
        else:
            if kind != cur_type:
                # close previous and start new
                out.append({
                    "start": start_idx if start_idx is not None else idx,
                    "end": idx - 1,
                    "type": cur_type,
                    "labels": list(cur_labels),
                })
                cur_type = kind
                start_idx = idx
                cur_labels = [canon] if canon else []
            else:
                if canon:
                    cur_labels.append(canon)

    if cur is not None:
        out.append({
            "start": start_idx if start_idx is not None else 0,
            "end": len(ls) - 1,
            "type": cur_type,
            "labels": list(cur_labels),
        })
    return out

