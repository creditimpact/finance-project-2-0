"""Claim metadata shared across frontend review features."""

from __future__ import annotations

from typing import Dict, List

CLAIM_FIELD_LINK_MAP: Dict[str, List[str]] = {
    "not_mine": [
        "account_number_display",
        "account_description",
        "creditor_type",
        "creditor_remarks",
        "last_verified",
        "date_reported",
    ],
    "paid_in_full": [
        "balance_owed",
        "past_due_amount",
        "payment_status",
        "last_payment",
        "account_status",
    ],
    "settled": [
        "balance_owed",
        "payment_status",
        "account_status",
        "creditor_remarks",
        "last_payment",
    ],
    "closed_but_open": ["account_status", "closed_date"],
    "authorized_user": [
        "account_description",
        "two_year_payment_history",
        "seven_year_history",
    ],
    "bankruptcy": [
        "payment_status",
        "account_status",
        "creditor_remarks",
        "balance_owed",
        "past_due_amount",
        "date_of_last_activity",
    ],
    "insurance_paid": [
        "creditor_type",
        "balance_owed",
        "past_due_amount",
        "payment_status",
        "creditor_remarks",
    ],
    "repo_cured": [
        "account_type",
        "account_status",
        "payment_status",
        "two_year_payment_history",
        "closed_date",
        "balance_owed",
    ],
    "judgment_satisfied": ["payment_status", "creditor_remarks"],
    "student_loan_rehab": [
        "creditor_type",
        "payment_status",
        "account_status",
        "date_opened",
        "two_year_payment_history",
    ],
    "wrong_dofd": [
        "date_of_last_activity",
        "last_payment",
        "two_year_payment_history",
        "seven_year_history",
    ],
    "paid_by_third_party": [
        "payment_status",
        "balance_owed",
        "past_due_amount",
        "last_payment",
        "creditor_remarks",
    ],
    "mixed_file": [
        "account_number_display",
        "account_type",
        "creditor_type",
        "payment_amount",
        "payment_frequency",
        "term_length",
        "account_status",
        "creditor_remarks",
        "date_opened",
        "date_reported",
        "last_verified",
    ],
    "late_history": [
        "payment_status",
        "two_year_payment_history",
        "seven_year_history",
        "past_due_amount",
        "last_payment",
        "date_of_last_activity",
    ],
}

CLAIM_DOC_KEY_ALIASES: Dict[str, Dict[str, str]] = {
    "paid_in_full": {
        "pay_proof": "proof_of_payment",
        "payoff_letter": "paid_in_full_letter",
    },
    "judgment_satisfied": {
        "satisfaction_or_vacate_order": "judgment_vacated",
    },
    "student_loan_rehab": {
        "rehab_completion_or_consolidation_payoff": "student_loan_rehab",
    },
    "wrong_dofd": {
        "original_chargeoff_letter_or_old_statements": "billing_statement",
    },
    "authorized_user": {
        "statement_showing_AU_or_issuer_letter": "statement_showing_AU_or_issuer_letter",
    },
}

DOC_KEY_ALIAS_TO_CANONICAL: Dict[str, str] = {
    alias: canonical
    for mapping in CLAIM_DOC_KEY_ALIASES.values()
    for alias, canonical in mapping.items()
}

__all__ = [
    "CLAIM_FIELD_LINK_MAP",
    "CLAIM_DOC_KEY_ALIASES",
    "DOC_KEY_ALIAS_TO_CANONICAL",
]
