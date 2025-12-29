# Analytics

## Purpose
Capture lightweight metrics and snapshots about each credit repair run for internal diagnostics.

## Pipeline position
Invoked after report analysis and strategy generation to persist summary statistics and failure reasons.

## Files
- `__init__.py`: package marker.
- `analytics_tracker.py`: write JSON analytics snapshots to disk.
  - Key function: `save_analytics_snapshot()`.
  - Exposes cache counters: `log_cache_hit`, `log_cache_miss`, `log_cache_eviction`.
  - Writes cache metrics JSON every 100 events or on shutdown.
  - Internal deps: `backend.api.config`.
- `analytics/` – helper subpackage (e.g., `strategist_failures.py`) providing counters; no separate README.

### Monitoring counters

Common counters emitted for dashboards:

- `letters_without_strategy_context` – attempts to generate letters without required strategy data.
- `guardrail_fix_count.{letter_type}` – number of guardrail remediation passes by letter type.
- `policy_override_reason.{reason}` – policy-based overrides grouped by reason.
- `rulebook.tag_selected.{tag}` – counts how often a rulebook action tag is chosen.
- `rulebook.suppressed_rules.{rule_name}` – rules skipped due to precedence or exclusion.
- `planner.cycle_progress{cycle,step}` – tracker for step advancement per cycle (cycle/step labels each <10).
- `planner.time_to_next_step_ms` – milliseconds until next eligible planner action (single gauge; alert on sustained spikes).
- `planner.resolution_rate` – fraction of accounts completed in a run (single gauge; alert if <0.8).
- `planner.avg_cycles_per_resolution` – average cycles required for completed accounts (single gauge; watch >p95).
- `planner.sla_violations_total` – cumulative SLA violations when sends lag (single counter).
- `planner.error_count` – planner exceptions caught (single counter).
- `outcome.verified`, `outcome.updated`, `outcome.deleted`, `outcome.nochange` – counts of bureau classification results (four counters).

## Entry points
- `analytics_tracker.save_analytics_snapshot`

## Guardrails / constraints
- Intended for internal use only; avoid storing sensitive client data in snapshots.

### Cache metrics example

```
{
  "timestamp": "2024-05-06T12:00:00",
  "cache": {"hits": 10, "misses": 2, "evictions": 1}
}
```

## Batch runner rollout

Enable the batch analytics job endpoint by setting `ENABLE_BATCH_RUNNER=1`.
Start with a 5% account canary and ramp gradually while monitoring cost and
result cardinality.

### Rollback

Set `ENABLE_BATCH_RUNNER=0` to stop accepting new batch jobs and revert to the
baseline state.
