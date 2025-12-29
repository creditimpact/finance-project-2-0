# Case Store

## Overview
The Case Store is the system's source of truth for per-account data. Cases are
created during extraction, one per real account, and later consumed by Stage-A
and the UI. Stage-A reads cases to adjudicate issues per bureau while the UI
queries the store for account details and decisions.

## Case identity & session index
Each session maintains a logical index mapping bureau-specific account keys to
canonical account identifiers. The index is stored at `summary.logical_index`
and has the form `logical_key → account_id` for the session. The function
`compute_logical_account_key` groups bureau reports for the same real account,
ensuring that all matching bureau entries share a single `account_id`.

## Case shape (per real account)
Cases store all bureau and normalized fields for a single account. The minimal
JSON layout is:

```json
{
  "account_id": "UUID",
  "fields": {
    "by_bureau": {
      "EX": { "account_number_last4": "1234", "balance_owed": 100.0, "credit_limit": 1000.0, "date_opened": "2019-08", "payment_status": "OK", "two_year_payment_history": [ { "date": "2024-01", "status": "OK" } ] },
      "EQ": { "...": "..." },
      "TU": { "...": "..." }
    },
    "normalized": {
      "current_balance": { "value": 100.0, "sources": { "EX": 100.0, "EQ": 100.0 }, "status": "agreed" }
    }
  },
  "artifacts": {
    "stageA_detection.EX": { "primary_issue": "late_payment", "tier": "medium", "problem_reasons": ["30d_late"], "decision_source": "rules", "confidence": 0.82 },
    "stageA_detection.EQ": { "...": "..." },
    "stageA_detection.TU": { "...": "..." }
  }
}
```

`by_bureau` keys are limited to `EX`, `EQ`, and `TU`; missing bureaus are
omitted. Keyed lists such as `two_year_payment_history` are merge-safe and are
never truncated.

## Normalized overlay (feature‑flagged)
Normalized fields live in `fields.normalized`. Each entry follows the structure
`{ field: { value, sources: {EX|EQ|TU}, status: agreed|conflict|derived|missing } }`.
Coverage metrics derived from the registry (Task 7) determine when this overlay
is present. When the `NORMALIZED_OVERLAY_ENABLED` flag is set, normalization runs
and populates this section.

## Stage‑A artifacts
When `ONE_CASE_PER_ACCOUNT_ENABLED=1`, Stage‑A writes per‑bureau artifacts as
`artifacts["stageA_detection.EX|EQ|TU"]`. For compatibility during transition, a
legacy winner may also be stored at `artifacts["stageA_detection"]`. Each
artifact payload includes `primary_issue`, `tier`, `problem_reasons`,
`decision_source`, `confidence`, and optional debugging metadata.

## Collectors output (UI contract)
The UI consumes problem accounts as a flat list:

```json
[
  {
    "account_id": "UUID",
    "bureau": "EX|EQ|TU",
    "primary_issue": "...",
    "tier": "...",
    "problem_reasons": ["..."],
    "decision_source": "rules|rules+ai",
    "confidence": 0.0
  }
]
```

Tier determines which confidence value is surfaced when multiple bureaus report
an issue on the same account.

## API
`GET /api/account/<session_id>/<account_id>` returns the case:
- `fields.by_bureau`
- `fields.normalized` when present
- `artifacts.stageA_detection.<BUREAU>` for available bureaus
- `meta.flags` and `meta.present_bureaus`

Missing accounts yield `404`. The endpoint is read‑only and has no OCR
dependency.

## Feature flags
- `ONE_CASE_PER_ACCOUNT_ENABLED` – enables per‑account mode (read in extractor
  and Stage‑A).
- `SAFE_MERGE_ENABLED` – controls deep‑merge semantics in the case store merge.
- `NORMALIZED_OVERLAY_ENABLED` – toggles normalized overlay population during
  normalization.

## Idempotency & concurrency guarantees
All writes use deep‑merge semantics with optimistic concurrency (CAS). Re‑runs do
not lose data, and concurrent updates retain existing information.

## Metrics (rollout observability)
- `stage1.per_account_mode.enabled`
- `stage1.by_bureau.present{bureau}`
- `stage1.logical_index.collisions`
- Field coverage metrics from Task 6
- Normalized registry coverage metrics from Task 7

## Legacy compatibility
Sessions created before per‑account mode are served through a read‑time shim,
which avoids writes and maps legacy structures into the new shape. The legacy
materializer is removed; consumers should call `/api/account` for case views.

## Examples & FAQs
- **Partial‑bureau case:** a case may contain only `EX` and `TU` keys if `EQ`
  data is absent.
- **Adding a bureau later:** new bureau data merges into `fields.by_bureau` while
  preserving existing data.
- **Payment history consumption:** treat `two_year_payment_history` as an
  append‑only keyed list; clients should merge by key and never assume fixed
  length.
