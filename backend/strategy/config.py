"""Configuration helpers for the strategy planner stage."""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date
from typing import Optional, Set, Tuple

from .calendar import holidays_from_env, weekend_from_env


@dataclass(frozen=True)
class PlannerEnv:
    enabled: bool
    mode: str
    allow_override: bool
    forced_weekday: Optional[int]
    account_subdir: str
    master_name: str
    weekday_prefix: str
    weekend: Set[int]
    holidays: Optional[Set[date]]
    debug: bool
    timezone: str
    max_calendar_span: int
    last_submit_window: Tuple[int, int]
    no_weekend_submit: bool
    include_supporters: bool
    strength_metric: str
    enforce_45d_cap: bool
    exclude_natural_text: bool
    handoff_min_business_days: int
    handoff_max_business_days: int
    target_effective_days: int
    min_increment_days: int
    dedup_by: str
    output_mode: str
    include_notes: bool
    enable_boosters: bool
    # Skeleton #2 enrichment layer
    skeleton2_enabled: bool
    skeleton2_max_items_per_handoff: int
    skeleton2_min_sla_days: int
    skeleton2_enforce_cadence: bool
    skeleton2_placement_mode: str
    skeleton2_enable_day40_strongest: bool
    # Output filtering
    output_omit_summary_and_constraints: bool

    @property
    def forced_start(self) -> Optional[int]:
        """Return the effective forced weekday override when enabled."""

        return self.forced_weekday if self.allow_override else None

    @classmethod
    def from_env(cls) -> "PlannerEnv":
        """Load planner configuration from environment variables."""

        return load_planner_env()


def env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    lowered = value.strip().lower()
    if lowered in {"1", "true", "yes", "on"}:
        return True
    if lowered in {"0", "false", "no", "off"}:
        return False
    return default


def env_int_or_none(name: str) -> Optional[int]:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return None
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"Environment variable {name} must be an integer") from exc


def env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"Environment variable {name} must be an integer") from exc


def _parse_last_window(raw: Optional[str]) -> Tuple[int, int]:
    default = (0, 40)
    if raw is None or not raw.strip():
        return default
    text = raw.strip()
    if "-" not in text:
        try:
            value = int(text)
        except ValueError as exc:
            raise ValueError("PLANNER_LAST_SUBMIT_WINDOW must be in the form 'start-end'") from exc
        return (value, value)
    start_text, end_text = text.split("-", 1)
    try:
        start = int(start_text)
        end = int(end_text)
    except ValueError as exc:
        raise ValueError("PLANNER_LAST_SUBMIT_WINDOW must be numeric") from exc
    if start > end:
        start, end = end, start
    if start < 0:
        start = 0
    return (start, end)


def load_planner_env() -> PlannerEnv:
    mode = os.getenv("PLANNER_MODE", "joint_optimize") or "joint_optimize"
    account_subdir = os.getenv("PLANNER_ACCOUNT_SUBDIR", "strategy") or "strategy"
    master_name = os.getenv("PLANNER_MASTER_FILENAME", "plan.json") or "plan.json"
    weekday_prefix = os.getenv("PLANNER_WEEKDAY_FILENAME_PREFIX", "plan_wd") or "plan_wd"
    weekend_raw = os.getenv("PLANNER_WEEKEND", "5,6")
    holidays_source = os.getenv("PLANNER_HOLIDAYS_SOURCE", "none")
    holidays_static = os.getenv("PLANNER_HOLIDAYS_STATIC_LIST", "")
    holidays_region = os.getenv("PLANNER_HOLIDAYS_REGION", "")
    timezone = os.getenv("PLANNER_TIMEZONE", "America/New_York") or "America/New_York"
    max_calendar_span = env_int("PLANNER_MAX_CALENDAR_SPAN", 45)
    last_submit_window = _parse_last_window(os.getenv("PLANNER_LAST_SUBMIT_WINDOW"))
    no_weekend_submit = env_bool("PLANNER_NO_WEEKEND_SUBMIT", True)
    include_supporters = env_bool("PLANNER_INCLUDE_SUPPORTERS", True)
    strength_metric = (os.getenv("PLANNER_STRENGTH_METRIC", "score") or "score").strip().lower()
    enforce_45d_cap = env_bool("PLANNER_ENFORCE_45D_CAP", False)
    exclude_natural_text = env_bool("PLANNER_EXCLUDE_NATURAL_TEXT", True)
    handoff_min = max(env_int("PLANNER_HANDOFF_MIN_BUSINESS_DAYS", 1), 1)
    handoff_max = max(env_int("PLANNER_HANDOFF_MAX_BUSINESS_DAYS", handoff_min), handoff_min)
    if handoff_max < handoff_min:  # defensive, though env_int already guards
        handoff_min, handoff_max = handoff_max, handoff_min
    if strength_metric not in {"score", "min_days"}:
        strength_metric = "score"
    target_effective_days = max(env_int("PLANNER_TARGET_EFFECTIVE_DAYS", 45), 1)
    min_increment_days = max(env_int("PLANNER_MIN_INCREMENT_DAYS", 1), 0)
    dedup_by = (os.getenv("PLANNER_DEDUP_BY", "decision") or "decision").strip().lower()
    if dedup_by not in {"decision", "field", "category"}:
        dedup_by = "decision"
    output_mode = (os.getenv("PLANNER_OUTPUT_MODE", "compact") or "compact").strip().lower()
    if output_mode not in {"compact", "verbose"}:
        output_mode = "compact"
    include_notes = env_bool("PLANNER_INCLUDE_NOTES", False)
    enable_boosters = env_bool("ENABLE_STRATEGY_BOOSTERS", False)

    # Skeleton #2 enrichment layer
    skeleton2_enabled = env_bool("PLANNER_ENABLE_SKELETON2", False)
    skeleton2_max_items = max(env_int("PLANNER_SKELETON2_MAX_ITEMS_PER_HANDOFF", 1), 1)
    skeleton2_min_sla = max(env_int("PLANNER_SKELETON2_MIN_SLA_DAYS", 5), 1)
    skeleton2_enforce_cadence = env_bool("PLANNER_SKELETON2_ENFORCE_CADENCE", False)
    skeleton2_placement = (os.getenv("PLANNER_SKELETON2_PLACEMENT_MODE", "half_sla_centered") or "half_sla_centered").strip().lower()
    if skeleton2_placement not in {"half_sla_centered"}:
        skeleton2_placement = "half_sla_centered"
    skeleton2_enable_day40 = env_bool("PLANNER_SKELETON2_ENABLE_DAY40_STRONGEST", False)
    
    # Output filtering flags
    output_omit = env_bool("PLANNER_OUTPUT_OMIT_SUMMARY_AND_CONSTRAINTS", False)

    return PlannerEnv(
        enabled=env_bool("ENABLE_STRATEGY_PLANNER", True),
        mode=mode.strip() or "joint_optimize",
        allow_override=env_bool("PLANNER_ALLOW_START_OVERRIDE", True),
        forced_weekday=env_int_or_none("PLANNER_FORCED_START_WEEKDAY"),
        account_subdir=account_subdir,
        master_name=master_name,
        weekday_prefix=weekday_prefix,
        weekend=weekend_from_env(weekend_raw),
        holidays=holidays_from_env(holidays_source, holidays_static, holidays_region),
        debug=env_bool("PLANNER_DEBUG", False),
        timezone=timezone.strip() or "America/New_York",
        max_calendar_span=max(1, max_calendar_span),
        last_submit_window=last_submit_window,
        no_weekend_submit=no_weekend_submit,
        include_supporters=include_supporters,
        strength_metric=strength_metric,
        enforce_45d_cap=enforce_45d_cap,
        exclude_natural_text=exclude_natural_text,
        handoff_min_business_days=handoff_min,
        handoff_max_business_days=handoff_max,
        target_effective_days=target_effective_days,
        min_increment_days=min_increment_days,
        dedup_by=dedup_by,
        output_mode=output_mode,
        include_notes=include_notes,
        enable_boosters=enable_boosters,
        skeleton2_enabled=skeleton2_enabled,
        skeleton2_max_items_per_handoff=skeleton2_max_items,
        skeleton2_min_sla_days=skeleton2_min_sla,
        skeleton2_enforce_cadence=skeleton2_enforce_cadence,
        skeleton2_placement_mode=skeleton2_placement,
        skeleton2_enable_day40_strongest=skeleton2_enable_day40,
        output_omit_summary_and_constraints=output_omit,
    )
