from __future__ import annotations

from datetime import date, datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, ConfigDict, confloat, conint


class Bureau(str, Enum):
    Equifax = "Equifax"
    Experian = "Experian"
    TransUnion = "TransUnion"


class Artifact(BaseModel):
    primary_issue: Optional[str] = None
    issue_types: Optional[List[str]] = None
    problem_reasons: Optional[List[str]] = None
    confidence: Optional[confloat(ge=0.0, le=1.0)] = None
    tier: Optional[str] = None
    decision_source: Optional[str] = None
    debug: Optional[Dict[str, Any]] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    model_config = ConfigDict(extra="allow")


class AccountFields(BaseModel):
    account_number: Optional[str] = None
    high_balance: Optional[confloat(gt=-1e18, lt=1e18)] = None
    last_verified: Optional[date | str] = None
    date_of_last_activity: Optional[date | str] = None
    date_reported: Optional[date | str] = None
    date_opened: Optional[date | str] = None
    balance_owed: Optional[confloat(gt=-1e18, lt=1e18)] = None
    closed_date: Optional[date | str] = None
    account_rating: Optional[str] = None
    account_description: Optional[str] = None
    dispute_status: Optional[str] = None
    creditor_type: Optional[str] = None
    account_status: Optional[str] = None
    payment_status: Optional[str] = None
    creditor_remarks: Optional[str] = None
    payment_amount: Optional[confloat(gt=-1e18, lt=1e18)] = None
    last_payment: Optional[date | str] = None
    term_length: Optional[str | int] = None
    past_due_amount: Optional[confloat(gt=-1e18, lt=1e18)] = None
    account_type: Optional[str] = None
    payment_frequency: Optional[str] = None
    credit_limit: Optional[confloat(gt=-1e18, lt=1e18)] = None
    two_year_payment_history: Optional[str | List[str]] = None
    days_late_7y: Optional[str | List[str]] = None

    model_config = ConfigDict(extra="allow")


class AccountCase(BaseModel):
    bureau: Bureau
    fields: AccountFields = Field(default_factory=AccountFields)
    artifacts: Dict[str, Artifact | None] = Field(default_factory=dict)
    tags: Dict[str, Any] = Field(default_factory=dict)
    version: int = 0


class PersonalInformation(BaseModel):
    name: Optional[str] = None
    also_known_as: Optional[str | List[str]] = None
    dob: Optional[date | str] = None
    current_address: Optional[str] = None
    previous_address: Optional[str] = None
    employer: Optional[str] = None


class ReportMeta(BaseModel):
    credit_report_date: Optional[date | str] = None
    personal_information: Optional[PersonalInformation] = Field(
        default_factory=PersonalInformation
    )
    public_information: List[Dict[str, Any]] = Field(default_factory=list)
    inquiries: List[Dict[str, Any]] = Field(default_factory=list)
    raw_source: Dict[str, Any] = Field(default_factory=dict)


class Summary(BaseModel):
    total_accounts: conint(ge=0) = 0
    open_accounts: conint(ge=0) = 0
    closed_accounts: conint(ge=0) = 0
    delinquent: conint(ge=0) = 0
    derogatory: conint(ge=0) = 0
    balances: Optional[confloat(gt=-1e18, lt=1e18)] = None
    payments: Optional[confloat(gt=-1e18, lt=1e18)] = None
    public_records: conint(ge=0) = 0
    inquiries_2y: conint(ge=0) = 0
    logical_index: Dict[str, str] = Field(default_factory=dict)


class SessionCase(BaseModel):
    session_id: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    report_meta: ReportMeta = Field(default_factory=ReportMeta)
    summary: Summary = Field(default_factory=Summary)
    accounts: Dict[str, AccountCase]
    version: int = 0


__all__ = [
    "Bureau",
    "Artifact",
    "AccountFields",
    "AccountCase",
    "PersonalInformation",
    "ReportMeta",
    "Summary",
    "SessionCase",
]
