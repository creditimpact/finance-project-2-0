# Evaluation Methodology

This document describes a lightweight procedure for validating changes to the
credit report pipeline. The goal is to run the same sample of reports against
multiple code revisions and compare high‑level metrics.

## 1. Sample reports

1. Collect **5–10** representative credit reports in JSON format.
2. Store them in a folder such as `sample_reports/`.
3. The files should contain the following optional fields used by the scripts:
   - `accounts` – list of parsed accounts
   - `inquiries` – list of inquiry records
   - `duplicates_removed` – list of entries removed as duplicates
   - `cost` – numeric processing cost for the report

## 2. Collect metrics

Run the evaluation script on the sampled reports and persist the aggregated
metrics. Execute this once on the current baseline (`pre`) and once after your
changes (`post`).

```bash
# Pre‑change metrics
python scripts/run_eval.py sample_reports/*.json --output pre_metrics.json

# Post‑change metrics
python scripts/run_eval.py sample_reports/*.json --output post_metrics.json
```

Each run reports the average of these metrics across all sampled reports:

| Metric | Description |
| ------ | ----------- |
| `accounts_found` | Number of accounts extracted |
| `inquiries_found` | Number of inquiry records |
| `dup_removed` | Count of duplicates removed |
| `latency` | Processing time in seconds |
| `cost` | Dollar cost of processing |

## 3. Compare results

Use the comparison helper to view the delta between the two runs:

```bash
python scripts/compare_eval.py pre_metrics.json post_metrics.json
```

Investigate any unexpected changes before finalizing your pull request.

## 4. Reporting

Record the metric outputs and any observations in the pull request description
or linked issue so reviewers can see the impact of the change.
