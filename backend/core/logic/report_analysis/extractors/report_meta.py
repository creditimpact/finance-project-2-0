"""Report-level metadata extraction."""
from __future__ import annotations

from typing import Dict, List

from backend.core.case_store.api import load_session_case, save_session_case
from .tokens import META_FIELD_MAP, parse_date


def extract(lines: List[str], *, session_id: str) -> Dict[str, object]:
    meta: Dict[str, object] = {}
    personal: Dict[str, object] = {}
    for line in lines:
        if ":" not in line:
            continue
        label, value = [p.strip() for p in line.split(":", 1)]
        key = META_FIELD_MAP.get(label.lower())
        if not key:
            continue
        if key == "credit_report_date":
            meta["credit_report_date"] = parse_date(value) or value.strip()
        elif key == "dob":
            personal["dob"] = parse_date(value) or value.strip()
        elif key in {"name", "also_known_as", "current_address", "previous_address", "employer"}:
            personal[key] = value.strip()
        else:
            meta[key] = value.strip()
    case = load_session_case(session_id)
    if personal:
        pi = case.report_meta.personal_information.model_copy(update=personal)
        meta["personal_information"] = pi
        case.report_meta.personal_information = pi
    if "credit_report_date" in meta:
        case.report_meta.credit_report_date = meta["credit_report_date"]
    save_session_case(case)
    return meta
