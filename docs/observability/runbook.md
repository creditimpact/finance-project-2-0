# Observability Runbook

This runbook outlines service level objectives (SLOs) and alert thresholds for key pipelines.

## SLOs

### Planner
- **SLA violations**: `planner.sla_violations_total` should remain at 0.
  - Alert when any violation occurs over a 5m window.

### Outcome
- **Ingest latency**: 95th percentile latency below 5 minutes.
  - Alert when p95 `outcome.ingest_latency_ms` exceeds 5m for 15m.

### Tri-Merge
- **Mismatch rate**: stay within normal baseline.
  - Alert when `tri_merge.mismatches_total` rate is more than 2× the 1h baseline.

## Alert thresholds

| Metric | Threshold | Window |
| --- | --- | --- |
| `planner.sla_violations_total` | > 0 | 5m |
| `outcome.ingest_latency_ms` p95 | > 5m | 15m |
| `tri_merge.mismatches_total` | > 2× 1h baseline | 5m |
