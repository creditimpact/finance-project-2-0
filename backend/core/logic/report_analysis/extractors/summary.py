"""Summary counts extraction."""
from __future__ import annotations

from typing import Dict, List

from backend.core.case_store.api import load_session_case, save_session_case
from .tokens import SUMMARY_FIELD_MAP, parse_amount


def extract(lines: List[str], *, session_id: str) -> Dict[str, object]:
    fields: Dict[str, object] = {}
    for line in lines:
        if ":" not in line:
            continue
        label, value = [p.strip() for p in line.split(":", 1)]
        key = SUMMARY_FIELD_MAP.get(label.lower())
        if not key:
            continue
        amt = parse_amount(value)
        fields[key] = amt if amt is not None else value.strip()
    case = load_session_case(session_id)
    if fields:
        case.summary = case.summary.model_copy(update=fields)
        save_session_case(case)
    return fields
