"""Heading normalization utilities."""

from __future__ import annotations

import re

from .names_normalization import COMMON_CREDITOR_ALIASES


def _canonicalize_base(s: str) -> str:
    """Canonicalize *s* by stripping punctuation and normalizing case."""

    name = s.upper().strip()
    name = re.sub(r"[/-]+", " ", name)
    name = re.sub(r"[^A-Z0-9 ]", "", name)
    name = re.sub(r"\s+", " ", name)
    return name.strip()


_ALIASES = {_canonicalize_base(a): c for a, c in COMMON_CREDITOR_ALIASES.items()}


def normalize_heading(s: str) -> str:
    """Return a normalized account heading.

    The normalization uppercases and strips punctuation/slashes, collapses
    whitespace and applies :data:`COMMON_CREDITOR_ALIASES`.
    """

    if not s:
        return ""
    name = _canonicalize_base(s)
    alias = _ALIASES.get(name)
    if alias:
        return alias
    name = re.sub(
        r"\b(BANK|USA|NA|N\.A\.|LLC|INC|CORP|CO|COMPANY)\b",
        "",
        name,
    )
    name = re.sub(r"\s+", " ", name)
    return name.strip().lower()
