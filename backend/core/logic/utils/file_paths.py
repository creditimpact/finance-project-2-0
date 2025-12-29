"""Helpers for working with file system paths."""

from __future__ import annotations

import re


def safe_filename(name: str) -> str:
    """Return a filename-safe version of ``name``."""
    cleaned = name.strip().replace(" ", "_")
    return re.sub(r"[\\/:*?\"<>|]", "_", cleaned)
