# Report analysis metrics

The `report_segment` audit event records detailed metrics for each bureau
analysis run. Logged fields include:

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

## Grafana / Kibana panels

- **Confidence trend** – average `confidence` by bureau. Trigger an alert when
  the 5‑minute moving average drops below `0.7`.
- **Cost anomalies** – sum of `cost_est` per hour. Alert when cost exceeds the
  95th percentile of the previous day.

## SLO suggestions

- **Low confidence spike** – alert if more than 5 % of segments report
  `confidence < 0.7` over a 10‑minute window.
- **Cost regression** – alert when `cost_est` per report doubles relative to
  the trailing weekly median.

These panels and alerts can be added in Grafana or Kibana by querying the
`report_segment` logs and aggregating on the fields above.
