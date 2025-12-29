# Data Models

This document summarizes dataclasses under `models/` and their relationships.

## account.py

### `LateHistory`
- `date: str`
- `status: str`

### `Inquiry`
- `creditor_name: str`
- `date: str`
- `bureau: Optional[str]`

### `Account`
- `account_id: Optional[str]`
- `name: str`
- `account_number: Optional[str]`
- `reported_status: Optional[str]`
- `status: Optional[str]`
- `flags: List[str]`
- `extras: Dict[str, object]`

## problem_account.py

### `ProblemAccount`
- `name: str`
- `account_number_last4: Optional[str]`
- `account_fingerprint: Optional[str]`
- `primary_issue: str`
- `issue_types: List[str]`
- `late_payments: Dict[str, Any]`
- `payment_statuses: Dict[str, Any]`
- `bureau_statuses: Dict[str, Any]`
- `original_creditor: Optional[str]`
- `source_stage: str`
- `extras: Dict[str, Any]`

## bureau.py

### `BureauAccount`
- extends `ProblemAccount`
- `bureau: Optional[str]`
- `section: Optional[str]`

### `BureauSection`
- `name: str`
- `accounts: List[BureauAccount]`

### `BureauPayload`
- `disputes: List[ProblemAccount]`
- `goodwill: List[ProblemAccount]`
- `inquiries: List[Inquiry]`
- `high_utilization: List[ProblemAccount]`
- Returned by `extract_problematic_accounts_from_report` instead of a raw `dict`.

## client.py

### `ClientInfo`
- `name: Optional[str]`
- `legal_name: Optional[str]`
- `address: Optional[str]`
- `email: Optional[str]`
- `state: Optional[str]`
- `goal: Optional[str]`
- `session_id: str`
- `structured_summaries: Any`
- `account_inquiry_matches: Optional[List[Dict[str, Any]]]`
- `extras: Dict[str, Any]`

### `ProofDocuments`
- `smartcredit_report: str`
- `extras: Dict[str, Any]`

## letter.py

### `LetterAccount`
- `name: str`
- `account_number: str`
- `status: str`
- `paragraph: Optional[str]`
- `requested_action: Optional[str]`
- `personal_note: Optional[str]`

### `LetterContext`
- `client_name: str`
- `client_address_lines: List[str]`
- `bureau_name: str`
- `bureau_address: str`
- `date: str`
- `opening_paragraph: str`
- `accounts: List[LetterAccount]`
- `inquiries: List[Inquiry]`
- `closing_paragraph: str`
- `is_identity_theft: bool`

### `LetterArtifact`
- `html: str`
- `pdf_path: Optional[str]`

## strategy.py

### `Recommendation`
- `action_tag: Optional[str]`
- `recommended_action: Optional[str]`
- `advisor_comment: Optional[str]`
- `flags: List[str]`

### `StrategyItem`
- `account_id: str`
- `name: str`
- `account_number: Optional[str]`
- `recommendation: Recommendation | None`

### `StrategyPlan`
- `accounts: List[StrategyItem]`

All models expose `from_dict()` and `to_dict()` helpers for conversion to and from plain dictionaries.
