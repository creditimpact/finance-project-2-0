# Stage 3 Hardening Guide

Stage 3 consolidates bureau segments into a hardened summary and records
versioned metadata for reproducibility.

## Flow
1. Split the raw credit report into bureau‑specific segments.
2. For each segment, fetch a cached analysis or call the model
   (`ANALYSIS_MODEL_VERSION`).
3. Validate the response against `analysis_schema.json` and populate
   defaults.
4. Merge bureau data, compute summary metrics, and flag
   `needs_human_review` for low‑confidence runs.
5. Emit a `report_segment` event with latency, cost, and token metrics.

## Schema and Prompt Versions
- **Prompt version:** `ANALYSIS_PROMPT_VERSION = 2`
- **Schema version:** `ANALYSIS_SCHEMA_VERSION = 1`
- **Schema path:** `backend/core/logic/report_analysis/analysis_schema.json`

### Summary Metrics Fields
The `summary_metrics` object includes:
- `num_collections`
- `num_late_payments`
- `high_utilization`
- `recent_inquiries`
- `total_inquiries`
- `num_negative_accounts`
- `num_accounts_over_90_util`
- `account_types_in_problem`

## Logged Metrics
`report_segment` events record:
- `stage3_tokens_in`
- `stage3_tokens_out`
- `latency_ms`
- `cost_est`
- `cache_hit`
- `error_code`
- `confidence`
- `needs_human_review`
- `missing_bureaus`
- `repair_count`
- `remediation_applied`

## Troubleshooting
- **Non‑zero `error_code`:** parsing failure. Inspect bureau headings and
  raw text.
- **Low `confidence`:** values below `0.7` set `needs_human_review`.
- **Missing bureaus:** non‑empty `missing_bureaus` indicates unparsed
  sections in the report.
- **Unexpected `repair_count` or `remediation_applied`:** extractors
  disagreed with model output; verify account numbers, statuses, and
  dates.
