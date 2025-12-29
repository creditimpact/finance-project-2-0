from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True, slots=True)
class ClaimDefinition:
    key: str
    label: str
    requires_docs: bool
    required_docs: List[str]
    hint: str | None = None


_CLAIM_DEFINITIONS: Dict[str, ClaimDefinition] = {
    "not_mine_fraud": ClaimDefinition(
        key="not_mine_fraud",
        label="Not mine / Identity theft",
        requires_docs=True,
        required_docs=["id_theft_report_or_police"],
        hint="FTC Identity Theft Report or police report",
    ),
    "paid_in_full": ClaimDefinition(
        key="paid_in_full",
        label="Paid in full",
        requires_docs=True,
        required_docs=["pay_proof", "payoff_letter"],
    ),
    "settlement_done": ClaimDefinition(
        key="settlement_done",
        label="Settled",
        requires_docs=True,
        required_docs=["settlement_letter", "settlement_payment_proof"],
    ),
    "closed_but_reported_open": ClaimDefinition(
        key="closed_but_reported_open",
        label="Closed but reported open",
        requires_docs=True,
        required_docs=["closure_letter_or_official_screenshot"],
    ),
    "authorized_user": ClaimDefinition(
        key="authorized_user",
        label="Authorized user only",
        requires_docs=True,
        required_docs=["statement_showing_AU_or_issuer_letter"],
    ),
    "bankruptcy_discharge": ClaimDefinition(
        key="bankruptcy_discharge",
        label="Included in bankruptcy",
        requires_docs=True,
        required_docs=["bk_discharge_order", "bk_schedule_with_account"],
    ),
    "medical_ins_paid": ClaimDefinition(
        key="medical_ins_paid",
        label="Insurance paid (medical)",
        requires_docs=True,
        required_docs=["insurance_EOB", "payment_proof_if_any"],
    ),
    "repo_foreclosure_cured": ClaimDefinition(
        key="repo_foreclosure_cured",
        label="Repo/Foreclosure cured",
        requires_docs=True,
        required_docs=["release_or_reinstatement_letter", "final_payment_proofs"],
    ),
    "judgment_satisfied": ClaimDefinition(
        key="judgment_satisfied",
        label="Judgment satisfied/vacated",
        requires_docs=True,
        required_docs=["satisfaction_or_vacate_order"],
    ),
    "student_loan_rehab": ClaimDefinition(
        key="student_loan_rehab",
        label="Student loan rehab/consol.",
        requires_docs=True,
        required_docs=["rehab_completion_or_consolidation_payoff"],
    ),
    "wrong_dofd": ClaimDefinition(
        key="wrong_dofd",
        label="Wrong DOFD / re-aging",
        requires_docs=True,
        required_docs=["original_chargeoff_letter_or_old_statements"],
    ),
    "third_party_paid": ClaimDefinition(
        key="third_party_paid",
        label="Paid by employer/3rd party",
        requires_docs=True,
        required_docs=["third_party_payment_letter", "payment_proof"],
    ),
    "mixed_file": ClaimDefinition(
        key="mixed_file",
        label="Mixed file / info mismatch",
        requires_docs=False,
        required_docs=[],
        hint="We’ll use ID + Proof of Address already on file",
    ),
}


CLAIMS: Dict[str, ClaimDefinition] = _CLAIM_DEFINITIONS
CLAIM_KEYS = set(CLAIMS.keys())
BASIC_ALWAYS_INCLUDED = ["gov_id", "proof_of_address"]


def get_claim_definition(claim_key: str) -> ClaimDefinition | None:
    return CLAIMS.get(claim_key)
