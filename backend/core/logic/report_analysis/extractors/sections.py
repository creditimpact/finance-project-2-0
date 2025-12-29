"""Section detection helpers."""
from __future__ import annotations

from typing import Dict, List

BUREAU_HEADERS = {
    "experian accounts": "Experian",
    "equifax accounts": "Equifax",
    "transunion accounts": "TransUnion",
}


def detect(pages: List[str]) -> Dict[str, object]:
    """Locate coarse sections within ``pages``.

    Returns a mapping with keys:
        - ``bureaus``: mapping of bureau name -> list of lines
        - ``report_meta``: list of lines for report meta section
        - ``summary``: list of lines for summary section
    """

    text = "\n".join(pages)
    lines = [ln.strip() for ln in text.splitlines()]
    sections: Dict[str, object] = {"bureaus": {}}
    current_key: tuple[str, str | None] | None = None
    buffer: List[str] = []

    def _flush() -> None:
        nonlocal buffer, current_key
        if not current_key:
            buffer = []
            return
        kind, name = current_key
        if kind == "bureau" and name:
            sections.setdefault("bureaus", {})[name] = buffer
        else:
            sections[kind] = buffer
        buffer = []

    for line in lines:
        low = line.lower()
        if low in BUREAU_HEADERS:
            _flush()
            current_key = ("bureau", BUREAU_HEADERS[low])
            continue
        if low in {"report meta", "personal information"}:
            _flush()
            current_key = ("report_meta", None)
            continue
        if low == "summary":
            _flush()
            current_key = ("summary", None)
            continue
        buffer.append(line)

    _flush()
    return sections
