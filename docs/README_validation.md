# Validation Workflow Overview

The validation pipeline evaluates bureau discrepancies in two passes: deterministic
screening to detect "missing" reports versus "mismatched" values, followed by
optional AI adjudication for semantic disagreements. All logic shares the same
normalisation rules so that bureau data is compared on an even footing before
any escalation decisions are made.

## Findings without AI

* **Missing** – Only the 18 deterministic fields participate. When at least one
  bureau reports a value and another bureau omits it entirely we create a
  *Missing* finding. Packs are never created for these cases and the AI path is
  skipped entirely.
* **Deterministic mismatches** – Amounts, dates, enumerations and history blocks
  are reconciled with pure code. Normalisation includes tolerance windows (see
  below) so that equivalent reporting formats do not escalate.

### Date conventions & tolerance

- The tolerance layer resolves each run's date formatting rules by loading
  `trace/date_convention.json` through
  `backend.validation.utils_dates.load_date_convention_for_sid`. The
  manifest is treated purely as a locator for that trace file when run
  directories move between machines; the tolerance check never copies date
  settings from the manifest itself.
- All reporting dates (`date_opened`, `closed_date`, `date_reported`,
  `date_of_last_activity`, `last_payment`, and `last_verified`) share the same
  ±5 day tolerance. Differences inside that window are considered matches and
  are suppressed before they can raise C4/C5 mismatches.

### Account rating alias map

`account_rating` normalisation applies a curated alias map (see
`backend/core/logic/consistency.py`) so that high-signal synonyms coalesce to a
stable set of canonical ratings. Keep the mapping intentionally small to avoid
unexpected merges; new entries should come with concrete report examples.

## AI escalation

Only three semantic fields may route to AI: `account_type`, `creditor_type` and
`account_rating`. After normalisation, a mismatch on any of these fields sets
`send_to_ai=true` and generates a validation pack when the validation packs
feature flag is enabled. The associated reason codes are limited to the C3/C4/C5
series and packs are never generated for missing data.

## Environment configuration

The behaviour of the validation pipeline can be tuned through the following
environment variables. Each reader falls back to the documented default when no
value is supplied.

| Variable | Default | Description |
| --- | --- | --- |
| `AMOUNT_TOL_ABS` | `50` | Absolute USD tolerance for amount mismatches. |
| `AMOUNT_TOL_RATIO` | `0.01` | Relative ratio tolerance for amount mismatches. |
| `LAST_PAYMENT_DAY_TOL` | `5` | Day window applied when comparing payment dates. |
| `VALIDATION_ROLLBACK` | `0` | Master kill-switch that restores the legacy pipeline by disabling validation requirements and AI routes. |
| `VALIDATION_PACKS_ENABLED` | `1` | Toggle to build validation packs. |
| `VALIDATION_REASON_ENABLED` | `0` | Enables reason capture and observability logging. |
| `VALIDATION_INCLUDE_CREDITOR_REMARKS` | `0` | Optional toggle to include `creditor_remarks` validation (disabled by default). |
| `VALIDATION_DRY_RUN` | `0` | When `1`, writes shadow findings without updating legacy outputs. |
| `VALIDATION_CANARY_PERCENT` | `100` | Percentage of accounts evaluated by the new validator (0–100). |

Boolean toggles accept the standard set of truthy values (`1`, `true`, `yes`,
`on`, `y`) and falsy values (`0`, `false`, `no`, `off`, `n`). Unrecognised inputs
fall back to their defaults so a misconfigured deployment will not disable
critical observability.

These flags are consumed by both the deterministic merge layer and the AI pack
builders. For example, tolerance defaults flow into the merge configuration used
by `account_merge.get_merge_cfg`, while `_reasons_enabled()` and
`_packs_enabled()` in the AI path read their respective toggles when deciding
whether to attach packs, send them automatically, and emit reason codes. See the
[Validation Requirements Runbook](./VALIDATION_RUNBOOK.md) for operational
procedures, smoke tests, and rollback guidance.

## History normalisation

Payment history blocks are compared after removing empty dictionaries so that
missing data is not incorrectly treated as a mismatch. This keeps the focus on
substantive discrepancies and aligns with our policy to exclude `creditor_remarks`
from the validation scope entirely.

For deeper implementation details explore the modules under
`backend/core/logic/report_analysis/` and `backend/ai/` which contain the
configuration readers and pack builders.
