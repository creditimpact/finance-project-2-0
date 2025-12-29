# Strategy Planner Stage

The strategy planner stage runs immediately after validation writes `summary.json` for an
account. It reads normalized findings (`validation_requirements.findings`) and emits a
per-account planning bundle containing a master plan and seven weekday variants.

## Modules

- `types.py` – dataclasses shared across strategy components.
- `io.py` – environment-aware helpers for loading findings and resolving account paths.
- `order_rules.py` – business rules that score findings and produce the canonical order.
- `calendar.py` – utilities that translate business-day wait times to calendar spans.
- `planner.py` – orchestration that generates canonical and weekday-specific plans.
- `writer.py` – atomic writers for `plan.json` and `plan_wd*.json` artifacts.
- `runner.py` – CLI entry point for manual planner execution.
- `exceptions.py` – custom exception hierarchy for strategy failures.

## Environment

The planner is driven by the `PLANNER_*` variables declared in `.env`. Key settings:

- `ENABLE_STRATEGY_PLANNER` toggles the stage on/off.
- `PLANNER_ACCOUNT_SUBDIR` controls the per-account output directory name.
- `PLANNER_MASTER_FILENAME` and `PLANNER_WEEKDAY_FILENAME_PREFIX` customize filenames.
- `PLANNER_WEEKEND` sets the weekend days (comma separated integers, 0 = Monday).

## Outputs

For each account (e.g. `runs/<SID>/cases/accounts/<accountId>`), the planner writes:

```
strategy/
  plan.json
  plan_wd0.json
  plan_wd1.json
  ...
  plan_wd6.json
```

The master file includes an overview (`canonical_order`, `by_weekday`, `best_overall`,
`notes`). Weekday files contain the plan slice relevant to a specific start weekday.
