"""Backwards-compatible wrapper around the tag minimization utility."""

from __future__ import annotations

from os import PathLike
from pathlib import Path
from typing import Union

from backend.core.logic.tags.compact import compact_account_tags


def compact_tags_for_account(account_dir: Union[str, Path, PathLike[str]]) -> None:
    """Reduce ``tags.json`` to minimal tags and persist verbose context."""

    account_path = Path(account_dir)
    compact_account_tags(account_path)


__all__ = ["compact_tags_for_account"]

