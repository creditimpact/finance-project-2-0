"""CLI entry point for executing the strategy planner manually."""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Iterator, List, Optional

from .config import PlannerEnv, load_planner_env
from .exceptions import (
    PlannerConfigurationError,
    StrategyPlannerError,
    UnsupportedDurationUnitError,
)
from .io import (
    append_strategy_logs,
    load_findings_from_summary,
    resolve_strategy_dir_for_account,
    write_plan_files_atomically,
)
from .planner import (
    compute_optimal_plan,
    build_per_bureau_inventories,
    BUREAUS,
)

LOG = logging.getLogger(__name__)


def _default_runs_root() -> Path:
    """Return the default root directory for run artifacts."""

    override = os.getenv("RUNS_ROOT")
    if override:
        return Path(override)
    return Path.cwd() / "runs"


def _iter_summary_paths_for_sid(sid: str, runs_root: Path) -> Iterator[Path]:
    """Yield `summary.json` paths for all accounts in the given `sid`."""

    accounts_dir = runs_root / sid / "cases" / "accounts"
    if not accounts_dir.exists():
        return iter(())

    summaries: List[Path] = []
    for child in accounts_dir.iterdir():
        if not child.is_dir():
            continue
        summary_path = child / "summary.json"
        if summary_path.exists():
            summaries.append(summary_path)
    return iter(sorted(summaries))


def _resolve_forced_start(raw: Optional[str], allow_override: bool) -> Optional[int]:
    """Parse a CLI-provided forced start weekday when overrides are allowed."""

    if not allow_override or raw is None:
        return None
    value = raw.strip()
    if value == "":
        return None
    try:
        forced = int(value)
    except ValueError as exc:
        raise PlannerConfigurationError("Forced start weekday must be an integer") from exc
    if forced < 0 or forced > 6:
        raise PlannerConfigurationError("Forced start weekday must be between 0 and 6")
    return forced


def run_for_summary(
    summary_path: Path,
    *,
    mode: str,
    forced_start: Optional[int],
    env: PlannerEnv,
) -> None:
    """
    New behaviour (per-bureau only):

    - Load findings from summary.json.
    - Build per-bureau inventories from the findings, using the existing
      build_per_bureau_inventories + bureau_dispute_state logic.
    - For each bureau that has at least one finding in its inventory,
      run compute_optimal_plan on that filtered list and write
      plan.json + plan_wd*.json into strategy/<bureau>/.
    - DO NOT write a global plan at strategy/plan.json at all.
    """
    try:
        findings = load_findings_from_summary(summary_path)
    except UnsupportedDurationUnitError as exc:
        raise PlannerConfigurationError(str(exc)) from exc
    except StrategyPlannerError as exc:
        LOG.error("PLANNER_LOAD_FAILED path=%s", summary_path, exc_info=exc)
        raise

    weekend = env.weekend or {5, 6}
    if not weekend:
        weekend = {5, 6}

    account_id = summary_path.parent.name

    append_strategy_logs(
        summary_path,
        (
            {
                "event": "planner_enter",
                "account": account_id,
                "mode": mode,
            },
            {
                "event": "planner_env_effective",
                "account": account_id,
                "mode": mode,
                "timezone": env.timezone,
                "max_calendar_span": env.max_calendar_span,
                "last_submit_window": list(env.last_submit_window),
                "no_weekend_submit": env.no_weekend_submit,
                "include_supporters": env.include_supporters,
                "strength_metric": env.strength_metric,
                "handoff_range": [env.handoff_min_business_days, env.handoff_max_business_days],
                "enforce_45d_cap": env.enforce_45d_cap,
                "enable_boosters": env.enable_boosters,
                # Skeleton #2 flags for verification
                "PLANNER_ENABLE_SKELETON2": env.skeleton2_enabled,
                "PLANNER_SKELETON2_MAX_ITEMS_PER_HANDOFF": env.skeleton2_max_items_per_handoff,
                "PLANNER_SKELETON2_MIN_SLA_DAYS": env.skeleton2_min_sla_days,
                "PLANNER_SKELETON2_ENFORCE_CADENCE": env.skeleton2_enforce_cadence,
                "PLANNER_SKELETON2_PLACEMENT_MODE": env.skeleton2_placement_mode,
                "PLANNER_SKELETON2_ENABLE_DAY40_STRONGEST": env.skeleton2_enable_day40_strongest,
            },
        ),
        account_subdir=env.account_subdir,
        account_id=account_id,
    )

    # Build bureau-specific inventories from the findings
    per_bureau_inventories = build_per_bureau_inventories(findings)
    strategy_dir = resolve_strategy_dir_for_account(summary_path.parent, env.account_subdir)

    # Log the inventory sizes so we can debug later
    LOG.info(
        "Per-bureau inventories sizes account=%s eq=%d ex=%d tu=%d",
        account_id,
        len(per_bureau_inventories.get("equifax", [])),
        len(per_bureau_inventories.get("experian", [])),
        len(per_bureau_inventories.get("transunion", [])),
    )

    any_inventory = any(per_bureau_inventories.get(bureau) for bureau in BUREAUS)

    if not any_inventory:
        # IMPORTANT: we do NOT fall back to a global joint plan anymore.
        # If there is no per-bureau ammo, we just log and exit.
        LOG.info(
            "Per-bureau inventories are empty for account=%s; no plans written.",
            account_id,
        )
        append_strategy_logs(
            summary_path,
            (
                {
                    "event": "planner_exit",
                    "account": account_id,
                    "mode": mode,
                    "status": "no_per_bureau_inventory",
                },
            ),
            account_subdir=env.account_subdir,
            account_id=account_id,
        )
        return

    # Main per-bureau path: one plan per bureau
    for bureau in BUREAUS:
        bureau_findings = per_bureau_inventories.get(bureau, [])
        if not bureau_findings:
            continue

        try:
            bureau_plan = compute_optimal_plan(
                bureau_findings,
                mode="per_bureau_joint_optimize",
                weekend=weekend,
                holidays=env.holidays,
                forced_start=forced_start,
                account_id=account_id,
                timezone_name=env.timezone,
                max_calendar_span=env.max_calendar_span,
                last_submit_window=env.last_submit_window,
                no_weekend_submit=env.no_weekend_submit,
                include_supporters=env.include_supporters,
                exclude_natural_text=env.exclude_natural_text,
                strength_metric=env.strength_metric,
                handoff_min_business_days=env.handoff_min_business_days,
                handoff_max_business_days=env.handoff_max_business_days,
                enforce_span_cap=env.enforce_45d_cap,
                target_effective_days=env.target_effective_days,
                min_increment_days=env.min_increment_days,
                dedup_by=env.dedup_by,
                output_mode=env.output_mode,
                include_notes=env.include_notes,
                enable_boosters=env.enable_boosters,
                bureau=bureau,
            )
        except PlannerConfigurationError as exc:
            LOG.info(
                "Per-bureau planner skipped path=%s bureau=%s reason=%s",
                summary_path,
                bureau,
                exc,
            )
            continue

        bureau_dir = strategy_dir / bureau
        bureau_dir.mkdir(parents=True, exist_ok=True)

        best_overall = bureau_plan.get("master", {}).get("best_overall", {})
        log_lines = [
            {
                "event": "planner_written",
                "account": account_id,
                "mode": "per_bureau_joint_optimize",
                "bureau": bureau,
                "best_start": bureau_plan.get("best_weekday"),
                "total_span_days": best_overall.get("calendar_span_days"),
            },
            {
                "event": "planner_exit",
                "account": account_id,
                "mode": "per_bureau_joint_optimize",
                "bureau": bureau,
                "status": "written",
            },
        ]

        write_plan_files_atomically(
            plan=bureau_plan,
            out_dir=bureau_dir,
            master_name=env.master_name,
            weekday_prefix=env.weekday_prefix,
            log_lines=log_lines,
            account_id=account_id,
            omit_summary_and_constraints=env.output_omit_summary_and_constraints,
        )

        LOG.info("Per-bureau planner ready path=%s bureau=%s", summary_path, bureau)


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the strategy planner stage")
    parser.add_argument("--sid", help="Run identifier (required for --sid mode)")
    parser.add_argument(
        "--account-summary",
        help="Explicit path to an account summary.json (overrides --sid traversal)",
    )
    parser.add_argument("--mode", help="Planner mode override (defaults to env)")
    parser.add_argument(
        "--forced-start",
        help="Override start weekday (0=Mon..6=Sun) when overrides are enabled",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    logging.basicConfig(level=logging.INFO)
    args = _parse_args(argv)

    env = load_planner_env()

    allow_override = env.allow_override
    forced_start = _resolve_forced_start(args.forced_start, allow_override)

    mode = args.mode or env.mode

    if args.account_summary:
        summary_path = Path(args.account_summary)
        if not summary_path.exists():
            raise PlannerConfigurationError(f"summary.json not found at {summary_path}")
        run_for_summary(summary_path, mode=mode, forced_start=forced_start, env=env)
        return

    if not args.sid:
        raise PlannerConfigurationError("Either --sid or --account-summary must be provided")

    runs_root = _default_runs_root()
    for summary_path in _iter_summary_paths_for_sid(args.sid, runs_root):
        run_for_summary(summary_path, mode=mode, forced_start=forced_start, env=env)


if __name__ == "__main__":
    main()

