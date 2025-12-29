"""Inquiry parsing utilities."""

from __future__ import annotations

import re

from .names_normalization import normalize_bureau_name

INQUIRY_RE = re.compile(
    r"(?P<creditor>[A-Za-z0-9 .,'&/-]{3,})\s+(?P<date>\d{1,2}/\d{2,4})\s+(?P<bureau>TransUnion|Experian|Equifax)",
    re.I,
)

# Additional patterns for structured inquiry section parsing
INQ_HEADER_RE = re.compile(
    r"Creditor Name\s+Date of Inquiry\s+Credit Bureau",
    re.I,
)
INQ_LINE_RE = re.compile(
    r"(?P<creditor>[A-Za-z0-9 .,'&/-]+?)\s+(?P<date>\d{1,2}/\d{1,2}/\d{2,4})\s+(?P<bureau>(?:TransUnion|Trans Union|TU|Experian|EX|Equifax|EQ))",
    re.I,
)


def extract_inquiries(text: str) -> list[dict]:
    """Return inquiry tuples parsed from the text."""

    lines = [ln.strip() for ln in text.splitlines()]
    start = None
    for i, line in enumerate(lines):
        if line == "Inquiries":
            for j in (i + 1, i + 2):
                if j < len(lines) and INQ_HEADER_RE.search(lines[j]):
                    start = j + 1
                    break
            if start is not None:
                break

    found: list[dict] = []
    if start is not None:
        for line in lines[start:]:
            if line.startswith("Creditor Contacts"):
                break
            m = INQ_LINE_RE.search(line)
            if m:
                found.append(
                    {
                        "creditor_name": m.group("creditor").strip(),
                        "date": m.group("date"),
                        "bureau": normalize_bureau_name(m.group("bureau")),
                    }
                )

    if not found:
        compact = re.sub(r"\s+", " ", text)
        for m in INQUIRY_RE.finditer(compact):
            found.append(
                {
                    "creditor_name": m.group("creditor").strip(),
                    "date": m.group("date"),
                    "bureau": normalize_bureau_name(m.group("bureau")),
                }
            )

    return found
