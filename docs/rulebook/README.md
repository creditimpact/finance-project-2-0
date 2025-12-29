# Rulebook

The rulebook converts tri-merge account diffs into semantic tags that drive downstream routing.

## Mismatch-to-tag mapping

Tri-merge diffs surface discrepancies across credit bureaus. Each rule inspects a mismatch and emits a tag describing the issue.
Examples:

- Name or address discrepancies → `identity_mismatch`
- Balance or status differences → `account_mismatch`
- Missing or extra accounts → `presence_mismatch`

## Precedence

Rules execute in priority order. When multiple rules match the same mismatch, the first rule wins and its tag suppresses lower-priority rules.

## Metrics

The engine tracks:

- Count of diffs evaluated
- Tag emission counts
- Rule evaluation latency

These metrics surface in dashboards and help tune rule quality.

See [flow.mmd](flow.mmd) for a visual overview of the evaluation pipeline.
