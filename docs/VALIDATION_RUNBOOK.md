# Validation Requirements Runbook

## Overview

The validation requirements pipeline normalizes bureau data, separates **Missing**
versus **Mismatch** findings, and only escalates semantic discrepancies in
`account_type`, `creditor_type`, or `account_rating` to AI after deterministic
normalization succeeds. Missing findings are never sent to AI and never generate
packs. The pipeline enforces tolerances for amount/date comparisons, converts
term lengths to months, normalizes payment frequency enums, and excludes
`creditor_remarks` entirely.

## Feature flags and toggles

| Flag | Default | Purpose |
| ---- | ------- | ------- |
| `VALIDATION_ROLLBACK` | `0` | Hard rollback switch. When set to `1`, the legacy pipeline is restored by disabling validation requirements and AI packs entirely. |
| `ENABLE_VALIDATION_REQUIREMENTS` | `1` | Enables validation findings to be written to `summary.json`. Ignored when `VALIDATION_ROLLBACK=1`. |
| `ENABLE_VALIDATION_AI` | `0` | Enables AI escalation for semantic mismatches (C3/C4/C5). Ignored when `VALIDATION_ROLLBACK=1`. |
| `VALIDATION_DRY_RUN` | `0` | When `1`, run the validator without writing legacy outputs. |
| `VALIDATION_CANARY_PERCENT` | `100` | Percentage of accounts evaluated by the validator. |
| `VALIDATION_DEBUG` | `0` | Keeps verbose debug blocks in `summary.json`. |

### Rollback procedure

1. Export `VALIDATION_ROLLBACK=1` (in orchestrator, `.env`, or runtime shell).
2. Redeploy or restart the worker processes. No other toggles are required.
3. Confirm the pipeline is disabled by verifying logs contain
   `"validation_requirements"..."skipped": true` and that new runs omit the
   `validation_requirements` block in `summary.json`.

To re-enable the validator, unset the flag (or set it back to `0`) and redeploy.

## Deployment checklist

1. **Config diff** – Validate that env files (`staging.env`, `production.env`)
   have the desired flag values.
2. **Unit tests** – Run
   `pytest tests/backend/core/logic/test_validation_requirements.py` and
   `pytest tests/backend/test_validation_rollback.py`.
3. **Integration smoke** – Execute one orchestrated run with
   `VALIDATION_DRY_RUN=1` to ensure findings are generated without impacting
   legacy flows.

## Monitoring & alerting

* **Batch metrics** – Track `validation_findings.count` and
  `validation_findings.missing` in Grafana (panel: *Validation Pipeline*).
* **AI escalations** – Monitor `validation_ai.calls` / `validation_ai.errors` to
  ensure semantic-only packs are being produced.
* **Error logs** – Search for `VALIDATION_REQUIREMENTS_PIPELINE_FAILED` in
  log aggregation for pipeline regressions.

When rolling back (`VALIDATION_ROLLBACK=1`), the metrics above should flatline
and log volume for validation should drop to zero within one ingestion cycle.
