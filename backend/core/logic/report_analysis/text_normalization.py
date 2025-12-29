"""Utilities for normalizing extracted report text.

The normalization process aims to make downstream parsing deterministic by
reducing variability in whitespace, unicode representations, dates and monetary
amounts. The function returns both the normalized string and counters describing
how many transformations were applied so that PII-safe telemetry can be
recorded.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from backend.config import (
    TEXT_NORMALIZE_AMOUNT_CANONICAL,
    TEXT_NORMALIZE_BIDI_STRIP,
    TEXT_NORMALIZE_COLLAPSE_SPACES,
    TEXT_NORMALIZE_DATE_ISO,
)


# Bidirectional/RTL markers which should be stripped.
BIDI_CHARS = {
    "\u200e",  # LRM
    "\u200f",  # RLM
    "\u202a",  # LRE
    "\u202b",  # RLE
    "\u202c",  # PDF
    "\u202d",  # LRO
    "\u202e",  # RLO
}


@dataclass
class NormalizationStats:
    """Counts of normalization operations for telemetry."""

    dates_converted: int = 0
    amounts_converted: int = 0
    bidi_stripped: int = 0
    space_reduced_chars: int = 0


def _strip_control_chars(text: str) -> str:
    """Remove non-printable control characters except whitespace."""

    buf: list[str] = []
    for ch in text:
        if ch in "\n\r\t ":
            buf.append(ch)
        elif unicodedata.category(ch).startswith("C"):
            continue
        else:
            buf.append(ch)
    return "".join(buf)


def normalize_page(text: str) -> tuple[str, NormalizationStats]:
    """Normalize a single page of report text.

    Returns a tuple of the normalized text and :class:`NormalizationStats`.
    """

    stats = NormalizationStats()

    # Unicode normalization
    txt = unicodedata.normalize("NFKC", text)

    # Strip bidi markers first so we can count them
    if TEXT_NORMALIZE_BIDI_STRIP:
        count = sum(txt.count(ch) for ch in BIDI_CHARS)
        if count:
            stats.bidi_stripped += count
            txt = "".join(ch for ch in txt if ch not in BIDI_CHARS)

    # Remove remaining control characters (but preserve whitespace for now)
    txt = _strip_control_chars(txt)

    # Collapse whitespace/newlines
    if TEXT_NORMALIZE_COLLAPSE_SPACES:
        before_len = len(txt)
        txt = re.sub(r"\s+", " ", txt)
        stats.space_reduced_chars += before_len - len(txt)

    # Normalize dates to ISO format
    if TEXT_NORMALIZE_DATE_ISO:
        date_pattern = re.compile(
            r"\b(?P<a>\d{1,4})(?P<sep>[/-])(?P<b>\d{1,2})(?P=sep)(?P<c>\d{1,4})\b"
        )

        def _replace_date(match: re.Match[str]) -> str:
            a = match.group("a")
            b = match.group("b")
            c = match.group("c")
            sep = match.group("sep")
            try:
                if len(a) == 4:  # YYYY/MM/DD
                    year, month, day = int(a), int(b), int(c)
                elif len(c) == 4:  # **/**/YYYY
                    year = int(c)
                    if sep == "/":  # MM/DD/YYYY
                        month, day = int(a), int(b)
                    else:  # DD-MM-YYYY
                        day, month = int(a), int(b)
                else:
                    return match.group(0)
                from datetime import date

                iso = date(year, month, day).isoformat()
                stats.dates_converted += 1
                return iso
            except Exception:
                return match.group(0)

        txt = date_pattern.sub(_replace_date, txt)

    # Normalize amounts
    if TEXT_NORMALIZE_AMOUNT_CANONICAL:
        amount_pattern = re.compile(
            r"(?<!\w)[\$€£₪]?\s*[+-]?\d[\d,]*(?:\.\d+)?"
        )

        def _replace_amount(match: re.Match[str]) -> str:
            s = match.group(0)
            clean = re.sub(r"[^0-9.-]", "", s)
            if clean != s:
                stats.amounts_converted += 1
            return clean

        txt = amount_pattern.sub(_replace_amount, txt)

    return txt.strip(), stats


__all__ = ["normalize_page", "NormalizationStats"]

