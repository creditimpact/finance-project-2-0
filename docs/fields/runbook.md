# Field population runbook

This runbook explains how missing field data is escalated when automatic fillers
cannot populate a value.

## Error emission

Filler failures emit an audit event:

```
fields.populate_errors{tag, field, reason}
```

The event payload identifies the action tag, the missing field, and the reason
for failure. The field is recorded on the account context under
`missing_fields` for downstream consumers.

## Metrics

Three counters track filler performance:

* `fields.populated_total` – incremented when a filler successfully populates a
  field. Labeled by `tag` and `field`; aggregate by `field` for dashboards to
  avoid high cardinality.
* `fields.populate_errors` – emitted when a filler fails. Labels include
  `tag`, `field`, and `reason`; aggregate by `field`.
* `finalize.missing_fields_after_population` – counts tags that still lack
  required fields after the population step. Labeled by `tag` only.

### Alert thresholds

* **Populate errors** – alert if
  `sum(rate(fields_populate_errors[5m])) by (field)` exceeds `0` for any field.
* **Missing after population** – page when
  `sum(rate(finalize_missing_fields_after_population[5m]))` is greater than
  `1`.
* **Populated total** – investigate if
  `sum(rate(fields_populated_total[5m])) by (field)` drops to `0` for any field
  for five minutes.

### Example queries

```promql
sum by(field) (rate(fields_populated_total[5m]))
sum by(field) (rate(fields_populate_errors[5m]))
sum by(tag) (rate(finalize_missing_fields_after_population[5m]))
```

## Escalation

* **Critical fields** – `name`, `address`, `date_of_birth`, `ssn_masked`,
  `creditor_name`, `account_number_masked`, `inquiry_creditor_name`, and
  `inquiry_date`. When any of these are missing the planner defers the
  action tag and the user is prompted to supply the information.
* **Optional fields** – `days_since_cra_result`, `amount`, and `medical_status`.
  These fall back to a safe default template when missing and do not block
  processing.

The planner or letter router can reference `critical_missing_fields` on the
context to determine which pathway was taken.

## Rollout

Enable the feature by setting `ENABLE_FIELD_POPULATION=1`. Ramp traffic by
increasing `FIELD_POPULATION_CANARY_PERCENT` from `0` to `100` while monitoring
the metrics above for regressions.

## Rollback

Set `ENABLE_FIELD_POPULATION=0` to bypass fillers. Alternatively, lower
`FIELD_POPULATION_CANARY_PERCENT` to limit rollout.
