"""Utilities for reading validation summary artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

__all__ = ["load_findings_from_summary"]


def load_findings_from_summary(
    runs_root: Path | str, sid: str, acc_index: int | str
) -> list[Any]:
    """Return cached validation findings for ``acc_index``.

    This helper is intentionally defensive and will swallow any I/O or parsing
    errors, returning an empty list when the summary cannot be loaded.
    """

    summary_path = (
        Path(runs_root) / str(sid) / "cases" / "accounts" / str(acc_index) / "summary.json"
    )

    if not summary_path.exists():
        return []

    try:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception:
        return []

    if not isinstance(payload, dict):
        return []

    validation_block = payload.get("validation_requirements")
    if not isinstance(validation_block, dict):
        return []

    findings = validation_block.get("findings")
    if isinstance(findings, list):
        return list(findings)

    if isinstance(findings, Iterable) and not isinstance(findings, (str, bytes, bytearray)):
        return list(findings)

    return []
