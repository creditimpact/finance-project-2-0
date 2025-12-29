"""Utilities for normalising numeric and date values.

These helpers are shared across the report analysis logic to ensure that
numbers and dates parsed from the various reports follow a consistent
representation.
"""

from __future__ import annotations

import re
from datetime import datetime

__all__ = ["to_number", "to_iso_date"]


def to_number(value: str) -> float | str:
    """Convert ``value`` to a ``float`` when possible.

    The function removes currency symbols (``$``), comma thousands separators
    and optional trailing ``CR``/``DR`` markers (case-insensitive).  If the
    cleaned string cannot be converted to ``float`` the original value is
    returned unchanged.

    Parameters
    ----------
    value:
        Textual representation of the number.

    Returns
    -------
    float | str
        ``float`` when ``value`` represents a number, otherwise ``value``.
    """

    cleaned = value.strip()
    # Strip currency symbols and thousands separators
    cleaned = re.sub(r"[,$]", "", cleaned)
    # Remove optional "CR"/"DR" tokens
    cleaned = re.sub(r"\b(?:CR|DR)\b", "", cleaned, flags=re.IGNORECASE)
    cleaned = cleaned.strip()

    try:
        return float(cleaned)
    except ValueError:
        return value


def to_iso_date(value: str) -> str:
    """Normalise ``value`` to the ISO ``YYYY-MM-DD`` format.

    Supported input formats are ``MM/DD/YYYY``, ``YYYY-MM-DD`` and ``MM/YYYY``
    (which defaults to the first day of the month).  Unrecognised values are
    returned unchanged.

    Parameters
    ----------
    value:
        Textual representation of the date.

    Returns
    -------
    str
        The normalised date or the original ``value`` when parsing fails.
    """

    s = value.strip()
    for fmt in ("%m/%d/%Y", "%Y-%m-%d", "%m/%Y"):
        try:
            dt = datetime.strptime(s, fmt)
            return dt.strftime("%Y-%m-%d")
        except ValueError:
            continue
    return value

