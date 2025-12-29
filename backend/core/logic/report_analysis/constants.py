"""Shared constants for report analysis parsing/materialization."""

BUREAUS: tuple[str, str, str] = ("transunion", "experian", "equifax")

ACCOUNT_FIELD_SET: tuple[str, ...] = (
    "account_number_display",
    "account_number_last4",
    "high_balance",
    "last_verified",
    "date_of_last_activity",
    "date_reported",
    "date_opened",
    "balance_owed",
    "closed_date",
    "account_rating",
    "account_description",
    "dispute_status",
    "creditor_type",
    "original_creditor",
    "account_status",
    "payment_status",
    "creditor_remarks",
    "payment_amount",
    "last_payment",
    "term_length",
    "past_due_amount",
    "account_type",
    "payment_frequency",
    "credit_limit",
    "two_year_payment_history",
    "seven_year_days_late",
)

INQUIRY_FIELDS: tuple[str, ...] = (
    "bureau",
    "subscriber",
    "date",
    "type",
    "permissible_purpose",
    "remarks",
    "_provenance",
)

PUBLIC_INFO_FIELDS: tuple[str, ...] = (
    "bureau",
    "item_type",
    "status",
    "date_filed",
    "amount",
    "remarks",
    "_provenance",
)
