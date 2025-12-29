from __future__ import annotations

import logging
from typing import Dict, List

from backend.telemetry.metrics import emit_counter
from backend.core.case_store.api import get_account_case, list_accounts
from backend.core.case_store.merge import safe_deep_merge
from backend.core.case_store.models import AccountCase, Bureau
from backend.core.orchestrators import compute_logical_account_key

logger = logging.getLogger(__name__)

_BUREAU_CODES = {
    Bureau.Experian: "EX",
    Bureau.Equifax: "EQ",
    Bureau.TransUnion: "TU",
}


def get_logical_key_for_case(case: AccountCase) -> str | None:
    """Best-effort logical key resolution for an ``AccountCase``."""
    summary = getattr(case, "summary", None)
    key = getattr(summary, "logical_key", None)
    if key:
        return str(key)

    last4 = (case.fields.account_number or "")[-4:]
    creditor = (case.fields.creditor_type or "") or (case.fields.account_type or "")
    opened = case.fields.date_opened or ""
    if not any([last4, creditor, opened]):
        return None
    try:
        return compute_logical_account_key(case)
    except Exception:  # pragma: no cover - defensive
        return None


def gather_cases_by_logical_key(session_id: str, logical_key: str) -> List[AccountCase]:
    """Return all cases in ``session_id`` matching ``logical_key``."""
    matches: List[AccountCase] = []
    for acc_id in list_accounts(session_id):  # type: ignore[operator]
        case = get_account_case(session_id, acc_id)  # type: ignore[operator]
        key = get_logical_key_for_case(case)
        if key == logical_key:
            matches.append(case)
    return matches


def build_by_bureau_shim(session_id: str, account_id: str) -> Dict[str, Dict]:
    """Synthesize a transient ``by_bureau`` map for legacy sessions."""
    base_case = get_account_case(session_id, account_id)
    logical_key = get_logical_key_for_case(base_case)
    if not logical_key:
        emit_counter("stage1.legacy_shim.missing_key")
        return {}

    cases = gather_cases_by_logical_key(session_id, logical_key)
    by_bureau: Dict[str, Dict] = {}
    for case in cases:
        code = _BUREAU_CODES.get(case.bureau, case.bureau.value[:2].upper())
        fields = case.fields.model_dump()
        existing = by_bureau.get(code)
        by_bureau[code] = safe_deep_merge(existing or {}, fields)

    emit_counter("stage1.legacy_shim.applied")
    try:  # pragma: no cover - best effort logging
        logger.debug(
            "legacy_shim.applied",
            extra={
                "session_id": session_id,
                "account_id": account_id,
                "logical_key": logical_key,
                "present_bureaus": sorted(by_bureau.keys()),
            },
        )
    except Exception:
        logger.exception("legacy_shim_logging_failed")
    return by_bureau
