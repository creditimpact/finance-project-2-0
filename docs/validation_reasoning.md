# Deterministic Field Escalation & Reason Metadata

This document summarizes how the deterministic escalation pipeline classifies fields, assigns reason metadata, and coordinates rollout behind a feature flag.

## Eligibility Policy (21 Fields)

Eighteen fields are **always eligible** when either missing or mismatched because they directly impact tradeline interpretation and dispute strength:

- `date_opened` – foundational for determining tradeline age and statute clocks.
- `closed_date` – clarifies whether the account remains active or should be closed.
- `account_type` – ensures bureaus agree on the product being reported.
- `creditor_type` – distinguishes original creditors from debt buyers/collectors.
- `high_balance` – validates the peak obligation that bureaus report.
- `credit_limit` – anchors revolving utilization and balance accuracy.
- `term_length` – verifies installment contracts and amortization assumptions.
- `payment_amount` – confirms expected installment or minimum payment values.
- `payment_frequency` – aligns repayment cadence across bureaus.
- `balance_owed` – core balance accuracy check impacting disputes and utilization.
- `last_payment` – confirms most recent customer payment activity.
- `past_due_amount` – highlights delinquency severity and consistency.
- `date_of_last_activity` – critical for limitation periods and activity tracking.
- `account_status` – ensures bureaus agree on open/closed/collection status.
- `payment_status` – captures delinquency buckets and charge-off states.
- `date_reported` – establishes data freshness and bureau sync timing.
- `two_year_payment_history` – protects against missing or inconsistent monthly grids.
- `seven_year_history` – checks long-term derogatory reporting windows.

Three fields are **conditional** and only escalate on mismatches (not pure missing) because silence is acceptable but disagreement signals risk:

- `creditor_remarks` – remarks are optional, but contradictory narratives must surface.
- `account_rating` – bureau-specific rating systems vary; only conflicts warrant escalation.
- `account_number_display` – masked numbers may be withheld, but conflicting displays imply data leakage.

All inputs normalize `"--"`, empty strings, and `null` to “missing” before eligibility checks.

## Reporting Patterns

Every escalated field is labeled with one of six patterns describing bureau coverage and agreement. Examples assume three bureaus (Experian, Equifax, TransUnion).

1. **AllMissing** – Every bureau omits the field (e.g., all three return `null`).
2. **SingleReported** – Exactly one bureau reports a value while the others are missing (e.g., Experian has `5000`, others `null`).
3. **MajorityMissing** – Two bureaus are missing and one reports data inconsistent with policy (e.g., two `null`, one `4000`).
4. **AllReportedAgree** – All bureaus report the field and values match, but escalation occurs because the pipeline required downstream LLM enrichment (e.g., identical payment histories flagged for semantic audit).
5. **AllReportedMismatch** – All bureaus report the field but at least two values conflict (e.g., `4500`, `5000`, `4800`).
6. **PartialMismatch** – Some bureaus report the field, others are missing, and at least two reported values conflict (e.g., Experian `Open`, Equifax `Closed`, TransUnion missing).

These patterns enable deterministic metadata and reason templates without relying on LLM interpretation.

## Flag Semantics

Each escalated field includes four boolean flags:

- `missing` – `true` when at least one normalized bureau value is missing.
- `mismatch` – `true` when two reported bureau values conflict after normalization.
- `both` – `true` when the field is simultaneously missing (for any bureau) **and** mismatched among the reported values.
- `eligible` – `true` when the field passes the policy rules above and should continue through downstream enrichment (LLM invoked only for the free-text fields).

These flags are mutually informative: `both` implies `missing` and `mismatch`, whereas `eligible` reflects policy gating rather than raw data state.

## Feature Flag & Rollout

Deterministic escalation is gated by `ENABLE_DETERMINISTIC_FIELD_ESCALATION`.

- **Default:** disabled (`0`) in production until coverage dashboards confirm parity with legacy behavior.
- **Canary rollout:** enable for 5% of accounts using `DETERMINISTIC_ESCALATION_CANARY_PERCENT` to monitor metadata completeness.
- **Ramp-up:** increase canary percentage incrementally (25% → 50% → 100%) after verifying audit logs and partner feedback.
- **Fallback:** set `ENABLE_DETERMINISTIC_FIELD_ESCALATION=0` to revert instantly to AI-driven escalation.

Audit dashboards should track pattern distribution, flag coverage, and escalation volume to validate determinations before full rollout.

