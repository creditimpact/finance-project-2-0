"""Canonical field lists for validation workflows."""

from __future__ import annotations

# Fields that always warrant investigation when mismatched or missing.
ALWAYS_INVESTIGATABLE_FIELDS: tuple[str, ...] = (
    # Open / Identification
    "date_opened",
    "closed_date",
    "account_type",
    "creditor_type",
    # Terms
    "high_balance",
    "credit_limit",
    "term_length",
    "payment_amount",
    "payment_frequency",
    # Activity
    "balance_owed",
    "last_payment",
    "past_due_amount",
    "date_of_last_activity",
    # Status / Reporting
    "account_status",
    "payment_status",
    "date_reported",
    # Histories
    "two_year_payment_history",
)

# Fields that require corroboration before escalating to a strong dispute.
CONDITIONAL_FIELDS: tuple[str, ...] = (
    "account_rating",
)

ALL_VALIDATION_FIELDS: tuple[str, ...] = (
    *ALWAYS_INVESTIGATABLE_FIELDS,
    *CONDITIONAL_FIELDS,
)

ALWAYS_INVESTIGATABLE_FIELD_SET = frozenset(ALWAYS_INVESTIGATABLE_FIELDS)
CONDITIONAL_FIELD_SET = frozenset(CONDITIONAL_FIELDS)
ALL_VALIDATION_FIELD_SET = frozenset(ALL_VALIDATION_FIELDS)


__all__ = [
    "ALWAYS_INVESTIGATABLE_FIELDS",
    "CONDITIONAL_FIELDS",
    "ALL_VALIDATION_FIELDS",
    "ALWAYS_INVESTIGATABLE_FIELD_SET",
    "CONDITIONAL_FIELD_SET",
    "ALL_VALIDATION_FIELD_SET",
]

