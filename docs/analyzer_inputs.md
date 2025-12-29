# Analyzer Inputs and Triad Adapter

This document explains what the problem detector expects as input and how we
adapt triad-shaped Stage‑A account data to the flat fields used by the rules
engine.

## What the detector consumes

- Entry point: `evaluate_account_problem(fields: dict) -> dict | None`
- Input shape: a flat mapping of normalized fields describing an account.
- In `detect_problem_accounts(sid)`, we use `account["fields"]` if present.
  Otherwise, we derive a flat mapping from the triad data with
  `build_rule_fields_from_triad(account)`.

## Fields produced by the triad adapter

The adapter returns the following keys:

- `past_due_amount` (float | null)
- `balance_owed` (float | null)
- `credit_limit` (float | null)
- `payment_status` (str | null)
- `account_status` (str | null)
- `days_late_7y` (int)
- `has_derog_2y` (bool)
- `account_type` (str | null)
- `creditor_remarks` (str | null)

These are passed unchanged to `evaluate_account_problem`.

## How values are derived

- Bureau precedence: If `account.triad.order` is present, use that
  order; otherwise fallback to `transunion → experian → equifax`.
- Status picking: For each textual field (e.g., `payment_status`,
  `account_status`, `creditor_remarks`, `account_type`), pick the first
  non-empty, non-"--" value following the bureau precedence.
- Currency parsing: Numbers like `"$12,091"` are normalized by removing
  any character except digits, decimal point, and minus sign, then parsed
  as float. Blank or invalid values produce `null`.
- 2‑year derogatory flag (`has_derog_2y`): True if any token in any
  bureau’s `two_year_payment_history` is not equal to `OK` (case-insensitive).
- 7‑year late days (`days_late_7y`): For each bureau, compute
  `late30 + late60 + late90` from `seven_year_history` and take the
  maximum across bureaus (conservative aggregation).

## Example

Given this triad snippet:

```json
{
  "triad": {"order": ["experian", "equifax", "transunion"]},
  "triad_fields": {
    "transunion": {"payment_status": "Current", "past_due_amount": "$0", "credit_limit": "$2,500"},
    "experian":   {"payment_status": "Late",    "past_due_amount": "$12,091", "credit_limit": "$2,600"},
    "equifax":    {"payment_status": "OK",      "past_due_amount": "$0", "credit_limit": "$2,700"}
  },
  "two_year_payment_history": {"experian": ["OK", "30", "OK"]},
  "seven_year_history": {
    "transunion": {"late30": 1, "late60": 0, "late90": 0},
    "experian":   {"late30": 0, "late60": 2, "late90": 0},
    "equifax":    {"late30": 0, "late60": 0, "late90": 3}
  }
}
```

The adapter yields:

```json
{
  "past_due_amount": 12091.0,
  "credit_limit": 2600.0,
  "payment_status": "Late",
  "days_late_7y": 3,
  "has_derog_2y": true
}
```

## Implementation notes

- Loader: accounts are read from the run manifest via
  `traces.accounts_table.accounts_json` (no PDF or legacy `traces/blocks` paths).
- Adapter and detector live in
  `backend/core/logic/report_analysis/problem_extractor.py`.
- The rules in `problem_detection` are unchanged; only inputs are adapted when
  `fields` are missing in the Stage‑A account object.

## Decision rules & thresholds

The analyzer applies simple, semantic rules to the flat `fields` mapping in
`backend/core/logic/report_analysis/problem_extractor.py`:

- Numeric thresholds:
  - `past_due_amount > 0` → add reason `past_due_amount:<value>` and consider delinquent.
  - `days_late_7y >= 1` → add reason `late_history: days_late_7y=<n>`.

- Status tokens (case-insensitive, substring match):
  - BAD_PAYMENT = {`late`, `delinquent`, `past due`, `charge-off`, `collection`, `derog`, `120`, `150`, `co`}
    - If `payment_status` contains any, add `bad_payment_status:<value>`.
  - BAD_ACCOUNT = {`collections`, `charge-off`, `charged off`, `repossession`, `foreclosure`}
    - If `account_status` contains any, add `bad_account_status:<value>`.

- Optional consistency check:
  - If `balance_owed > 0` and `account_status == "Closed"`, add `positive_balance_on_closed`.

- Primary issue precedence (first match wins):
  1) `charge_off` / `collection` (from status tokens)
  2) `delinquency` (from `past_due_amount > 0`)
  3) `late_history` (from `days_late_7y >= 1`)
  4) `status` (other BAD status tokens)

- Emission gate:
  - A candidate is emitted only if `problem_reasons` is non-empty.

### Provenance

The adapter returns `(fields, prov)` where `prov` maps each derived field to the
source bureau, e.g., `{"payment_status":"experian", "past_due_amount":"transunion"}`.
The detector augments `reason.debug.signals` with bureau-tagged strings like:

- `past_due_amount:12091.00 (bureau=experian)`
- `payment_status:120 (bureau=experian)`
- `account_status:collections (bureau=transunion)`

This helps explain which bureau supplied each triggering value.

## Problematic account merge stage

After problematic accounts are detected, we run a deterministic merge stage
to collapse near-duplicate accounts (e.g., an original charge-off and its
collection tradeline) before building case folders. The scorer compares every
pair of candidates from the same run/SID and assigns a weighted similarity
score in the range 0–1.

### Scoring features and weights

The scorer computes five partial scores and combines them with configurable
weights. Missing or malformed values simply contribute `0.0` for that part, so
The legacy five-part merge scorer has been fully retired. The replacement
0–100 pair scorer compares bureau values directly from
`runs/<sid>/cases/accounts/<idx>/bureaus.json` and exposes the
`score_pair_0_100` / `score_all_pairs_0_100` APIs for downstream
consumers.【F:backend/core/logic/report_analysis/account_merge.py†L272-L352】【F:backend/core/logic/report_analysis/account_merge.py†L600-L711】

### Observability and logs

- Pairwise scoring emits `MERGE_V2_SCORE ...` / `MERGE_V2_DECISION ...`
  logs for every comparison. Use ripgrep to inspect them, e.g. `rg
  "MERGE_V2_DECISION" runs/<sid>/ -g"*.log"`.【F:backend/core/logic/report_analysis/account_merge.py†L855-L895】
- `MERGE_V2_TRIGGER` entries capture every trigger the scorer evaluates so
  dashboards can efficiently trace merge traffic alongside the per-pair
  scores.【F:backend/core/logic/report_analysis/account_merge.py†L855-L895】

### Where merge conclusions are stored

`persist_merge_tags` returns the traditional best-partner payload for
callers, but it now persists conclusions exclusively in
`runs/<sid>/cases/accounts/<idx>/tags.json`. Each run rewrites the merge
tags for AI/auto pairs (`merge_pair`) and the selected best partner
(`merge_best`) so downstream systems can read a single source of truth
without touching `summary.json`.【F:backend/core/logic/report_analysis/account_merge.py†L1213-L1263】 The lean case builder consumes the
same helpers, ensuring analyzer issue tags and merge tags live together in
each account's tags file.【F:backend/core/logic/report_analysis/problem_case_builder.py†L557-L612】
