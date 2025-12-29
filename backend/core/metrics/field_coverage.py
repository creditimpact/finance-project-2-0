import logging
from typing import Dict, List

from backend.core.case_store import api as case_store

logger = logging.getLogger(__name__)


# Simple metrics interface with gauge and count methods.
class _Metrics:
    def gauge(
        self, name: str, value: float, tags: Dict[str, str] | None = None
    ) -> None:
        logger.info("metric %s %s %s", name, value, tags)

    def count(self, name: str, value: int, tags: Dict[str, str] | None = None) -> None:
        logger.info("metric %s %s %s", name, value, tags)


metrics = _Metrics()

EXPECTED_FIELDS: Dict[str, List[str]] = {
    "Experian": [
        "balance_owed",
        "credit_limit",
        "high_balance",
        "date_opened",
        "account_status",
        "past_due_amount",
        "payment_status",
        "two_year_payment_history",
    ],
    "Equifax": [
        "balance_owed",
        "credit_limit",
        "high_balance",
        "date_opened",
        "account_status",
        "past_due_amount",
        "payment_status",
        "two_year_payment_history",
    ],
    "TransUnion": [
        "balance_owed",
        "credit_limit",
        "high_balance",
        "date_opened",
        "account_status",
        "past_due_amount",
        "payment_status",
        "two_year_payment_history",
    ],
}

_MAX_LOG_FIELDS = 20
_TOP_K = 10


def _is_filled(value: object) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return value.strip() != ""
    if isinstance(value, (list, tuple, set, dict)):
        return len(value) > 0
    return True


def emit_account_field_coverage(
    *, session_id: str, account_id: str, bureau: str, fields: Dict[str, object]
) -> None:
    try:
        expected = EXPECTED_FIELDS.get(bureau, [])
        missing = [f for f in expected if not _is_filled(fields.get(f))]
        filled_count = len(expected) - len(missing)
        expected_count = len(expected)
        coverage_pct = round(100 * filled_count / max(1, expected_count))
        metrics.gauge(
            "stage1.field_coverage.account",
            coverage_pct,
            tags={"session_id": session_id, "account_id": account_id, "bureau": bureau},
        )
        if missing:
            logger.info(
                "field_coverage.missing %s",
                {
                    "session_id": session_id,
                    "account_id": account_id,
                    "bureau": bureau,
                    "missing": missing[:_MAX_LOG_FIELDS],
                },
            )
    except Exception:
        logger.exception("field_coverage_account_failed")


def emit_session_field_coverage_summary(*, session_id: str) -> None:
    try:
        missing_counts: Dict[str, int] = {}
        for account_id in case_store.list_accounts(session_id):
            case = case_store.get_account_case(session_id, account_id)
            bureau = getattr(case.bureau, "value", str(case.bureau))
            fields = case.fields.model_dump()
            expected = EXPECTED_FIELDS.get(bureau, [])
            for field in expected:
                if not _is_filled(fields.get(field)):
                    missing_counts[field] = missing_counts.get(field, 0) + 1
        top_items = sorted(missing_counts.items(), key=lambda kv: kv[1], reverse=True)[
            :_TOP_K
        ]
        for name, count in top_items:
            metrics.count(
                "stage1.field_coverage.missing.count",
                count,
                tags={"session_id": session_id, "field": name},
            )
        logger.info(
            "field_coverage.session_summary %s",
            {
                "session_id": session_id,
                "top_missing": [{"field": n, "count": c} for n, c in top_items],
            },
        )
    except Exception:
        logger.exception("field_coverage_session_failed")
