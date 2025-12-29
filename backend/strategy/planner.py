"""Planner orchestration for computing strategy dispute plans.

DEADLINE SEMANTICS (Updated):
-------------------------------
The planner enforces a single deadline rule:
    last_submit <= 40 (relative to anchor) AND not on weekend

Legacy behavior used a window [37, 40] where plans had to "hit" a specific target
day within that range. This has been replaced with a simpler upper-bound rule.

The `last_submit_window` parameter is maintained for backward compatibility but
only its upper bound (typically 40) is enforced. The lower bound is ignored.

All acceptance/scoring logic (pack_sequence_to_target_window, _enrich_sequence_with_contributions,
master plan selection) now uses this unified deadline check, not legacy window-hit semantics.

CONSISTENCY GUARANTEE:
----------------------
inventory_selected, sequence_debug, and sequence_compact are guaranteed to be consistent:
- planned_submit_index matches calendar_day_index
- planned_submit_date matches submit.date
- effective_contribution_days match across all views
All are derived from the final enriched sequence after forcing and enrichment.
"""

from __future__ import annotations

import os
from copy import deepcopy
from datetime import date, datetime, timedelta, timezone
from typing import Dict, List, Optional, Sequence, Set, Tuple, TypedDict, Literal

from zoneinfo import ZoneInfo

from .calendar import (
    advance_business_days_date,
    business_days_between,
    find_business_day_in_window,
    next_occurrence_of_weekday,
    roll_if_weekend,
    subtract_business_days_date,
)
from .exceptions import PlannerConfigurationError
from .order_rules import build_strategy_orders, rank_findings, _normalized_decision
from .types import Finding


CAP_REFERENCE_DAY = 45
_TRUE_FLAG_VALUES = {"1", "true", "yes", "on"}
_FALSE_FLAG_VALUES = {"0", "false", "no", "off"}
_BOOSTER_ALLOWED_DECISIONS = {"strong_actionable", "supportive_needs_companion"}
_WEEKDAY_ABBREVIATIONS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
BUREAUS: Tuple[str, ...] = ("equifax", "experian", "transunion")

ALL_VALIDATION_FIELDS: Set[str] = {
    "account_number_display",
    "date_opened",
    "closed_date",
    "account_type",
    "creditor_type",
    "high_balance",
    "credit_limit",
    "term_length",
    "payment_amount",
    "payment_frequency",
    "balance_owed",
    "last_payment",
    "past_due_amount",
    "date_of_last_activity",
    "account_status",
    "payment_status",
    "account_rating",
    "last_verified",
    "date_reported",
    "two_year_payment_history",
    "seven_year_history",
}

REQUIRED_MISSING_FIELDS: Set[str] = {
    "payment_status",
    "account_status",
    "past_due_amount",
    "balance_owed",
    "high_balance",
    "credit_limit",
    "date_of_last_activity",
    "date_opened",
    "date_reported",
    "last_payment",
    "seven_year_history",
    "two_year_payment_history",
}

REQUIRED_MISMATCH_FIELDS: Set[str] = ALL_VALIDATION_FIELDS


def _coerce_truthy_flag(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in _TRUE_FLAG_VALUES:
            return True
        if lowered in _FALSE_FLAG_VALUES:
            return False
    if isinstance(value, (int, float)):
        return bool(value)
    return bool(value)


def _get_finding_attr(finding: object, name: str) -> object:
    if isinstance(finding, Finding):
        return getattr(finding, name, None)
    if isinstance(finding, dict):  # pragma: no cover - defensive fallback
        return finding.get(name)
    return None


def _extract_bureau_state_map(finding: object) -> Optional[Dict[str, str]]:
    candidate = getattr(finding, "bureau_dispute_state", None)
    if isinstance(candidate, dict):
        return candidate

    if isinstance(finding, dict):  # pragma: no cover - defensive fallback
        raw_map = finding.get("bureau_dispute_state")
        if isinstance(raw_map, dict):
            return raw_map

    alternate = getattr(finding, "bureau_dispute_state_raw", None)
    if isinstance(alternate, dict):
        return alternate

    if isinstance(finding, dict):  # pragma: no cover - defensive fallback
        raw_alt = finding.get("bureau_dispute_state_raw")
        if isinstance(raw_alt, dict):
            return raw_alt

    return None


def _resolve_bureau_dispute_metadata(
    finding: object,
    bureau_norm: Optional[str],
) -> Tuple[Optional[str], bool, bool, bool]:
    if not bureau_norm:
        return None, False, False, False

    field_value = _get_finding_attr(finding, "field")
    field_norm = str(field_value or "").strip().lower()
    if not field_norm:
        return None, False, False, False

    state_map = _extract_bureau_state_map(finding)
    if not isinstance(state_map, dict):
        return None, False, False, False

    state_value: Optional[str] = state_map.get(bureau_norm)
    if state_value is None:
        for key, value in state_map.items():  # pragma: no cover - defensive guard
            if str(key).strip().lower() == bureau_norm:
                state_value = value
                break

    if state_value is None:
        return None, False, False, False

    state_norm = str(state_value).strip().lower()

    is_missing_flag = _coerce_truthy_flag(_get_finding_attr(finding, "is_missing"))
    is_mismatch_flag = _coerce_truthy_flag(_get_finding_attr(finding, "is_mismatch"))

    bureau_is_missing = (
        state_norm == "missing"
        and is_missing_flag
        and field_norm in REQUIRED_MISSING_FIELDS
    )
    bureau_is_mismatch = is_mismatch_flag and field_norm in REQUIRED_MISMATCH_FIELDS

    return state_norm, bureau_is_missing, bureau_is_mismatch, True


class DedupNote(TypedDict):
    kept: str
    dropped: List[str]
    reason: str


class RoleSelectionMeta(TypedDict):
    closer_field: str
    opener_field: str
    domain_tiebreak_applied: bool
    openers_eligible: int
    closers_eligible: int
    reason: str


class SequenceExplainer(TypedDict, total=False):
    placement: str
    base_placement: str
    why_here: str
    handoff_rule: str
    score: Dict[str, object]
    strength_metric: str
    strength_value: int
    adjustments: List[str]


class SequenceItem(TypedDict, total=False):
    idx: int
    field: str
    role: str
    min_days: int
    submit_on: Dict[str, object]
    submit: Dict[str, object]
    sla_window: Dict[str, Dict[str, object]]
    calendar_day_index: int
    delta_from_prev_days: int
    handoff_days_before_prev_sla_end: int
    remaining_to_45_cap: int
    decision: str
    category: str
    explainer: SequenceExplainer
    effective_contribution_days: int
    effective_contribution_days_unbounded: int
    unused_sla_days: int
    running_total_days: int
    running_total_days_after: int
    running_total_days_unbounded_after: int
    raw_business_sla_days: int
    raw_calendar_sla_days: int
    notes: List[str]
    timeline: Dict[str, object]


class SequenceCompactWindow(TypedDict):
    start_date: str
    end_date: str


class SequenceCompactTimeline(TypedDict):
    from_day: int
    to_day: int


class SequenceCompactDays(TypedDict):
    effective: int
    effective_unbounded: int
    cumulative: int
    cumulative_unbounded: int


class SequenceCompactEntry(TypedDict, total=False):
    idx: int
    field: str
    role: str
    submit_date: str
    submit_weekday: str
    window: SequenceCompactWindow
    timeline: SequenceCompactTimeline
    days: SequenceCompactDays
    is_closer: bool
    why_here: str


class BoosterHeader(TypedDict, total=False):
    field: str
    role: Literal["booster"]
    order_idx: int
    paired_with_field: Optional[str]
    paired_with_idx: Optional[int]
    planned_submit_index: Optional[int]
    planned_submit_date: Optional[str]
    reason_code: Optional[str]
    category: Optional[str]


class BoosterStepWindow(TypedDict, total=False):
    start_date: str
    end_date: str


class BoosterStepTimeline(TypedDict, total=False):
    from_day: int
    to_day: int


class BoosterStepDays(TypedDict, total=False):
    effective: int
    effective_unbounded: int
    cumulative: int
    cumulative_unbounded: int


class BoosterStep(TypedDict, total=False):
    idx: int
    field: str
    role: Literal["booster"]
    submit_date: str
    submit_weekday: str
    window: BoosterStepWindow
    timeline: BoosterStepTimeline
    days: BoosterStepDays
    is_booster: bool
    anchor_idx: Optional[int]
    anchor_field: Optional[str]
    anchor_reason_code: Optional[str]
    bundle_key: Optional[str]
    why_here: Optional[str]


class EnrichmentItem(TypedDict, total=False):
    idx: int
    field: str
    enrichment_type: Literal["skeleton2"]
    placement_reason: str
    calendar_day_index: int
    planned_submit_index: int
    submit_on: Dict[str, object]
    sla_window: Dict[str, Dict[str, object]]
    min_days: int
    business_sla_days: int
    between_skeleton1_indices: List[int]
    handoff_reference_day: int
    half_sla_offset: int
    strength_value: int
    role: str
    decision: str
    category: str
    # Day-40 specific fields
    day40_rule_applied: bool
    day40_target_index: int
    day40_adjustment_reason: Optional[str]
    pre_closer_field: str
    pre_closer_unbounded_end: int


class InventoryAllEntry(TypedDict):
    field: str
    default_decision: str
    business_sla_days: int
    role_guess: str


class InventorySelectedEntry(TypedDict, total=False):
    field: str
    default_decision: str
    business_sla_days: int
    role: str
    order_idx: int
    planned_submit_index: int
    planned_submit_date: str
    effective_contribution_days: int
    effective_contribution_days_unbounded: int
    running_total_after: int
    running_total_unbounded_after: int
    is_closer: bool


class InventoryHeader(TypedDict, total=False):
    inventory_all: List[InventoryAllEntry]
    inventory_selected: List[InventorySelectedEntry]
    dedup_notes: List[DedupNote]


class WeekdayPlan(TypedDict, total=False):
    schema_version: int
    anchor: Dict[str, object]
    timezone: str
    sequence_debug: List[SequenceItem]
    sequence_compact: List[SequenceCompactEntry]
    sequence_boosters: List[BoosterStep]
    enrichment_sequence: List[EnrichmentItem]
    calendar_span_days: int
    last_calendar_day_index: int
    summary: Dict[str, object]
    constraints: Dict[str, object]
    skipped: List[Dict[str, object]]
    reason: str
    inventory_header: InventoryHeader
    inventory_boosters: List[BoosterHeader]


class BestOverall(TypedDict):
    start_weekday: int
    calendar_span_days: int
    order: List[str]
    last_calendar_day_index: int


class MasterPlan(TypedDict, total=False):
    schema_version: int
    generated_at: str
    timezone: str
    mode_used: str
    bureau: str
    weekend: List[int]
    holidays: List[str]
    best_overall: BestOverall
    by_weekday: Dict[str, WeekdayPlan]
    meta: Dict[str, object]
    reason: str
    calendar_span_days: int
    last_calendar_day_index: int
    summary: Dict[str, object]
    constraints: Dict[str, object]
    skipped: List[Dict[str, object]]
    inventory_header: InventoryHeader
    inventory_boosters: List[BoosterHeader]
    sequence_boosters: List[BoosterStep]


class PlannerOutputs(TypedDict):
    master: MasterPlan
    weekday_plans: Dict[int, WeekdayPlan]
    schedule_logs: List[Dict[str, object]]
    best_weekday: int
    inventory_boosters: List[BoosterHeader]
    sequence_boosters: List[BoosterStep]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _resolve_timezone(timezone_name: str) -> ZoneInfo:
    candidate = (timezone_name or "UTC").strip() or "UTC"
    try:
        return ZoneInfo(candidate)
    except Exception:
        return ZoneInfo("UTC")


def _advance_to_business_day(anchor: date, weekend: Set[int], holidays: Set[date]) -> date:
    current = anchor
    weekend = {day % 7 for day in weekend}
    while current.weekday() in weekend or current in holidays:
        current += timedelta(days=1)
    return current


def _serialize_day(day: date) -> Dict[str, object]:
    weekday = day.weekday()
    weekday_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    return {
        "date": day.isoformat(),
        "weekday": weekday,
        "weekday_name": weekday_names[weekday],
    }


def _validate_findings(findings: Sequence[Finding]) -> None:
    if not isinstance(findings, Sequence):
        raise ValueError("findings must be a sequence of Finding objects")

    for idx, finding in enumerate(findings):
        if isinstance(finding, Finding):
            field = getattr(finding, "field", None)
            min_days = getattr(finding, "min_days", None)
            duration_unit = getattr(finding, "duration_unit", None)
        elif isinstance(finding, dict):
            field = finding.get("field")
            min_days = finding.get("min_days")
            duration_unit = finding.get("duration_unit")
        else:
            raise ValueError(f"finding[{idx}] must be a Finding or mapping")

        if not field:
            raise ValueError(f"finding[{idx}] missing field")
        if min_days is None:
            raise ValueError(f"finding[{idx}] missing min_days")
        if not duration_unit:
            raise ValueError(f"finding[{idx}] missing duration_unit")


def _placement_for(role: str, index: int, total: int) -> str:
    role_norm = (role or "supporter").strip().lower()
    if role_norm == "opener":
        return "second_strongest_first"
    if role_norm == "closer":
        return "closer_anchor"
    if index == 0:
        return "sequence_anchor"
    if index == total - 1:
        return "sequence_tail"
    return "support_chain"


def _prepare_items(
    sequence: Sequence[Finding],
    metadata: Dict[str, Dict[str, object]],
    *,
    strength_metric: str,
) -> List[Dict[str, object]]:
    items: List[Dict[str, object]] = []
    for order_index, finding in enumerate(sequence):
        if isinstance(finding, Finding):
            field = finding.field
            category = finding.category or ""
            min_days = max(int(finding.min_days or 0), 0)
            default_decision = finding.default_decision or ""
            reason_code = finding.reason_code
            documents = list(finding.documents or []) if finding.documents else None
        else:
            field = str(finding.get("field"))
            category = str(finding.get("category", ""))
            min_days = max(int(finding.get("min_days", 0)), 0)
            default_decision = str(finding.get("default_decision", ""))
            reason_code = finding.get("reason_code") if isinstance(finding, dict) else None
            docs_raw = finding.get("documents") if isinstance(finding, dict) else None
            documents = list(docs_raw) if isinstance(docs_raw, list) else None

        meta_entry = metadata.get(field, {})
        role = str(meta_entry.get("role", meta_entry.get("placement", "supporter"))) or "supporter"
        score_dict = meta_entry.get("score") or {
            "base": min_days,
            "bonuses": {},
            "total": min_days,
        }
        if strength_metric == "min_days":
            strength_value = min_days
        else:
            strength_value = int(score_dict.get("total", min_days))

        reason_payload = meta_entry.get("reason_code")
        if reason_payload is None:
            reason_payload = reason_code

        documents_payload = meta_entry.get("documents")
        if documents_payload is None and documents is not None:
            documents_payload = documents

        normalized_decision = _normalized_decision(str(default_decision or meta_entry.get("default_decision", "")))

        item: Dict[str, object] = {
            "field": field,
            "role": role,
            "category": meta_entry.get("category", category),
            "min_days": min_days,
            "default_decision": default_decision,
            "decision": default_decision or meta_entry.get("default_decision", ""),
            "score": score_dict,
            "strength_metric": strength_metric,
            "strength_value": strength_value,
            "order_index": order_index,
            "why_here": meta_entry.get("why_here", ""),
            "bureaus_present": meta_entry.get("bureaus_present", 0),
            "placement": _placement_for(role, order_index, len(sequence)),
            "reason_code": reason_payload,
            "documents": documents_payload,
            "normalized_decision": normalized_decision,
        }
        items.append(item)
    return items


def _assign_base_placements(items: Sequence[Dict[str, object]]) -> None:
    total = len(items)
    for index, item in enumerate(items):
        item.setdefault("order_index", index)
        placement = item.get("placement")
        if not placement:
            placement = _placement_for(str(item.get("role", "supporter")), index, total)
        item["placement"] = placement


def should_include_in_bureau_inventory(finding: Finding, bureau: str) -> bool:
    bureau_norm = (bureau or "").strip().lower()
    if not bureau_norm:
        return False

    state_norm, bureau_is_missing, bureau_is_mismatch, state_present = _resolve_bureau_dispute_metadata(
        finding,
        bureau_norm,
    )
    if not state_present:
        return False

    if bureau_is_missing:
        return True

    if bureau_is_mismatch:
        return True

    return False


def build_per_bureau_inventories(findings: Sequence[Finding]) -> Dict[str, List[Finding]]:
    per_bureau: Dict[str, List[Finding]] = {bureau: [] for bureau in BUREAUS}
    for finding in findings:
        for bureau in BUREAUS:
            if should_include_in_bureau_inventory(finding, bureau):
                per_bureau[bureau].append(finding)
    return per_bureau


def _select_findings_varlen(
    items: List[Dict[str, object]],
    *,
    run_dt: datetime,
    tz: ZoneInfo,
    weekend: Set[int],
    holidays: Set[date],
    target_window: Tuple[int, int],
    max_span: int,
    enforce_span_cap: bool,
    handoff_min: int,
    handoff_max: int,
    target_effective_days: int,
    min_increment_days: int,
    dedup_by: str,
    include_supporters: bool,
    include_notes: bool,
) -> Tuple[
    List[Dict[str, object]],
    List[Dict[str, object]],
    List[DedupNote],
    int,
    RoleSelectionMeta,
]:
    selection_logs: List[Dict[str, object]] = []
    dedup_map: Dict[str, Dict[str, object]] = {}
    dedup_records: Dict[str, Dict[str, object]] = {}
    dedup_dropped = 0

    for item in items:
        if not include_supporters and str(item.get("role")) == "supporter":
            selection_logs.append(
                {
                    "event": "candidate_rejected",
                    "decision": item.get("field"),
                    "reason": "supporters_disabled",
                    "ts": _utc_now_iso(),
                }
            )
            continue

        key = _dedup_key_for(item, dedup_by)
        incumbent = dedup_map.get(key)
        if incumbent is None:
            dedup_map[key] = item
            continue

        if _is_stronger(item, incumbent):
            selection_logs.append(
                {
                    "event": "candidate_rejected",
                    "decision": incumbent.get("field"),
                    "reason": "duplicate",
                    "ts": _utc_now_iso(),
                }
            )
            dedup_map[key] = item
            record = dedup_records.setdefault(
                key,
                {
                    "kept": str(item.get("field")),
                    "dropped": set(),
                    "reason": f"dedup: {dedup_by}",
                },
            )
            record["kept"] = str(item.get("field"))
            record["dropped"].add(str(incumbent.get("field")))
            dedup_dropped += 1
        else:
            selection_logs.append(
                {
                    "event": "candidate_rejected",
                    "decision": item.get("field"),
                    "reason": "duplicate",
                    "ts": _utc_now_iso(),
                }
            )
            record = dedup_records.setdefault(
                key,
                {
                    "kept": str(incumbent.get("field")),
                    "dropped": set(),
                    "reason": f"dedup: {dedup_by}",
                },
            )
            record["dropped"].add(str(item.get("field")))
            dedup_dropped += 1

    pool = list(dedup_map.values())
    if len(pool) < 2:
        raise PlannerConfigurationError(
            "Variable-length planner requires at least two primary findings after deduplication"
        )

    sorted_pool = sorted(pool, key=_selection_sort_key)
    if not sorted_pool:
        raise PlannerConfigurationError("No eligible findings available after deduplication")

    strong_pool = [
        item
        for item in sorted_pool
        if _normalized_decision(str(item.get("default_decision", item.get("decision", ""))))
        == "strong_actionable"
    ]

    # NEW SELECTION LOGIC: Choose closer first (max business_sla_days), then opener (best score with days<=closer preference)
    
    # Step 1: Define closer candidates (strong_actionable + supportive_needs_companion)
    closer_candidates = [
        item for item in sorted_pool
        if _normalized_decision(str(item.get("default_decision", item.get("decision", ""))))
        in {"strong_actionable", "supportive_needs_companion"}
    ]
    if not closer_candidates:
        closer_candidates = sorted_pool  # Fallback if no strong/supportive items
    
    # Step 2: Choose closer = item with maximum business_sla_days, break ties by score
    max_sla = max(max(int(c.get("min_days", 0)), 0) for c in closer_candidates)
    closer_pool = [c for c in closer_candidates if max(int(c.get("min_days", 0)), 0) == max_sla]
    closer_candidate = max(closer_pool, key=lambda item: (
        int(item.get("strength_value", 0)),
        int(item.get("min_days", 0)),
        -int(item.get("order_index", 0))
    ))
    closer_field_name = str(closer_candidate.get("field"))
    closer_sla = max(int(closer_candidate.get("min_days", 0)), 0)
    closers_eligible_count = len(closer_candidates)
    
    # Step 3: Define opener candidates (strong_actionable only)
    opener_candidates = [
        item for item in sorted_pool
        if _normalized_decision(str(item.get("default_decision", item.get("decision", ""))))
        == "strong_actionable"
    ]
    if not opener_candidates:
        opener_candidates = sorted_pool  # Fallback
    openers_eligible_count = len(opener_candidates)
    
    # Step 4: Choose opener = best score strong_actionable, preferring items with days <= closer_sla
    opener_filtered = [o for o in opener_candidates if max(int(o.get("min_days", 0)), 0) <= closer_sla]
    if opener_filtered:
        # Prefer items with days <= closer
        opener_candidate = max(opener_filtered, key=lambda item: (
            int(item.get("strength_value", 0)),
            int(item.get("min_days", 0)),
            -int(item.get("order_index", 0))
        ))
    else:
        # Fallback: choose best score from all opener_candidates
        opener_candidate = max(opener_candidates, key=lambda item: (
            int(item.get("strength_value", 0)),
            int(item.get("min_days", 0)),
            -int(item.get("order_index", 0))
        ))
    opener_field_name = str(opener_candidate.get("field"))
    
    # Step 5: Ensure opener != closer when possible
    if opener_field_name == closer_field_name and len(sorted_pool) > 1:
        # Keep closer, recompute opener from remaining candidates
        remaining_opener_candidates = [o for o in opener_candidates if str(o.get("field")) != closer_field_name]
        if remaining_opener_candidates:
            opener_filtered_remaining = [o for o in remaining_opener_candidates if max(int(o.get("min_days", 0)), 0) <= closer_sla]
            if opener_filtered_remaining:
                opener_candidate = max(opener_filtered_remaining, key=lambda item: (
                    int(item.get("strength_value", 0)),
                    int(item.get("min_days", 0)),
                    -int(item.get("order_index", 0))
                ))
            else:
                opener_candidate = max(remaining_opener_candidates, key=lambda item: (
                    int(item.get("strength_value", 0)),
                    int(item.get("min_days", 0)),
                    -int(item.get("order_index", 0))
                ))
            opener_field_name = str(opener_candidate.get("field"))
    
    # Validate closer has max SLA among all items
    all_slas = [max(int(item.get("min_days", 0)), 0) for item in sorted_pool]
    if closer_sla < max(all_slas):
        raise PlannerConfigurationError(
            f"Closer {closer_field_name} (SLA={closer_sla}) is not the max-SLA candidate (max={max(all_slas)})"
        )
    
    domain_tiebreak_applied = False  # No longer using domain tiebreak in new logic
    selection_reason = "closer=max_business_sla_days; opener=best_score_strong_with_days_le_closer_preference"

    chosen_sequence: List[Dict[str, object]] = []
    chosen_fields: Set[str] = set()

    for base_candidate in (opener_candidate, closer_candidate):
        field_name = str(base_candidate.get("field"))
        if field_name in chosen_fields:
            continue
        chosen_sequence.append(deepcopy(base_candidate))
        chosen_fields.add(field_name)

    remaining: List[Dict[str, object]] = [
        deepcopy(item) for item in sorted_pool if str(item.get("field")) not in chosen_fields
    ]

    base_plan, _ = _evaluate_sequence_for_selection(
        chosen_sequence,
        weekday=0,
        run_dt=run_dt,
        tz=tz,
        weekend=weekend,
        holidays=holidays,
        target_window=target_window,
        max_span=max_span,
        enforce_span_cap=enforce_span_cap,
        handoff_min=handoff_min,
        handoff_max=handoff_max,
        include_notes=include_notes,
        opener_field=opener_field_name,
        closer_field=closer_field_name,
    )
    current_total = base_plan["summary"].get("total_effective_days", 0)
    current_window_hit = bool(base_plan["summary"].get("last_submit_in_window", False))
    current_last_index = int(base_plan["summary"].get("last_submit", 0))

    while remaining:
        eligible: List[Dict[str, object]] = []
        rejected_batch: List[Dict[str, object]] = []

        for candidate in remaining:
            tentative_sequence = chosen_sequence[:-1] + [deepcopy(candidate)] + [chosen_sequence[-1]]
            plan, success = _evaluate_sequence_for_selection(
                tentative_sequence,
                weekday=0,
                run_dt=run_dt,
                tz=tz,
                weekend=weekend,
                holidays=holidays,
                target_window=target_window,
                max_span=max_span,
                enforce_span_cap=enforce_span_cap,
                handoff_min=handoff_min,
                handoff_max=handoff_max,
                include_notes=include_notes,
                opener_field=opener_field_name,
                closer_field=closer_field_name,
            )

            new_total = plan["summary"].get("total_effective_days", 0)
            delta = new_total - current_total
            hits_window = bool(plan["summary"].get("last_submit_in_window", False))
            last_index = int(plan["summary"].get("last_submit", 0))
            improves_window = hits_window and not current_window_hit

            if not success:
                rejected_batch.append(
                    {"decision": candidate.get("field"), "reason": "breaks_last_window"}
                )
                continue

            if delta < 0:
                rejected_batch.append(
                    {
                        "decision": candidate.get("field"),
                        "reason": "delta_negative",
                        "delta": delta,
                    }
                )
                continue

            if delta < min_increment_days and not improves_window:
                rejected_batch.append(
                    {"decision": candidate.get("field"), "reason": "delta_below_min", "delta": delta}
                )
                continue

            within_target = new_total <= max(target_effective_days, 0)
            eligible.append(
                {
                    "candidate": candidate,
                    "plan": plan,
                    "delta": delta,
                    "total": new_total,
                    "within_target": within_target,
                    "hits_window": hits_window,
                    "improves_window": improves_window,
                    "last_index": last_index,
                }
            )

        timestamp = _utc_now_iso()
        if not eligible:
            for entry in rejected_batch:
                payload = {
                    "event": "candidate_rejected",
                    "decision": entry.get("decision"),
                    "reason": entry.get("reason"),
                    "ts": timestamp,
                }
                if "delta" in entry:
                    payload["delta_days"] = entry["delta"]
                selection_logs.append(payload)
            break

        for entry in rejected_batch:
            payload = {
                "event": "candidate_rejected",
                "decision": entry.get("decision"),
                "reason": entry.get("reason"),
                "ts": timestamp,
            }
            if "delta" in entry:
                payload["delta_days"] = entry["delta"]
            selection_logs.append(payload)

        eligible.sort(
            key=lambda record: (
                0 if record["improves_window"] else 1,
                0 if record["hits_window"] else 1,
                0 if record["within_target"] else 1,
                -record["delta"],
                abs(target_window[1] - record["last_index"]),
                -int(record["candidate"].get("strength_value", 0)),
                int(record["candidate"].get("order_index", 0)),
            )
        )
        chosen_record = eligible[0]
        candidate = chosen_record["candidate"]

        chosen_sequence = chosen_sequence[:-1] + [deepcopy(candidate)] + [chosen_sequence[-1]]
        current_total = chosen_record["total"]
        current_window_hit = bool(chosen_record.get("hits_window", False))
        current_last_index = int(chosen_record.get("last_index", current_last_index))
        remaining = [item for item in remaining if item["field"] != candidate["field"]]

        selection_logs.append(
            {
                "event": "candidate_accepted",
                "decision": candidate.get("field"),
                "delta_days": chosen_record["delta"],
                "running_total": current_total,
                "hits_window": chosen_record.get("hits_window", False),
                "improves_window": chosen_record.get("improves_window", False),
                "last_index": chosen_record.get("last_index", current_last_index),
                "ts": _utc_now_iso(),
            }
        )

    dedup_notes: List[DedupNote] = []
    for record in dedup_records.values():
        dropped = sorted({entry for entry in record.get("dropped", set()) if entry})
        if not dropped:
            continue
        dedup_notes.append(
            {
                "kept": str(record.get("kept")),
                "dropped": dropped,
                "reason": str(record.get("reason", f"dedup: {dedup_by}")),
            }
        )

    dedup_notes.sort(key=lambda note: (note["kept"], note["reason"]))

    role_meta: RoleSelectionMeta = {
        "closer_field": closer_field_name,
        "opener_field": opener_field_name,
        "domain_tiebreak_applied": domain_tiebreak_applied,
        "openers_eligible": openers_eligible_count,
        "closers_eligible": closers_eligible_count,
        "reason": selection_reason,
    }

    return (
        chosen_sequence,
        selection_logs,
        dedup_notes,
        dedup_dropped,
        role_meta,
    )


def _enrich_sequence_with_contributions(
    sequence: List[SequenceItem],
    *,
    submit_history: Sequence[date],
    sla_history: Sequence[date],
    anchor_date: date,
    last_window: Tuple[int, int],
    weekend: Set[int],
    enforce_span_cap: bool,
    include_notes: bool,
) -> Tuple[int, Optional[int], Dict[str, object]]:
    if not sequence:
        summary: Dict[str, object] = {
            "first_submit": 0,
            "last_submit": 0,
            "last_submit_in_window": False,
            "total_items": 0,
            "total_effective_days": 0,
            "final_submit_date": None,
            "final_sla_end_date": None,
            "distance_to_45": CAP_REFERENCE_DAY,
            "over_45_by_days": 0,
        }
        if not enforce_span_cap:
            summary["total_effective_days_unbounded"] = 0
            summary["final_effective_days_unbounded"] = 0
        return 0, 0 if not enforce_span_cap else None, summary

    total_effective = 0
    total_unbounded = 0
    total_overlap_unbounded_days = 0
    running_total = 0
    running_total_unbounded = 0
    for idx, entry in enumerate(sequence):
        submit_date = submit_history[idx]
        sla_end = sla_history[idx]
        raw_business = int(entry.get("min_days", 0))
        raw_calendar = max((sla_end - submit_date).days, 0)
        entry["raw_business_sla_days"] = raw_business
        entry["raw_calendar_sla_days"] = raw_calendar

        d_i = entry.get("calendar_day_index", 0)
        notes: List[str] = []
        next_index = sequence[idx + 1]["calendar_day_index"] if idx + 1 < len(sequence) else None

        # ALWAYS compute unbounded from full SLA window (submit to SLA end)
        sla_end_index = max((sla_end - anchor_date).days, 0)
        unbounded = max(sla_end_index - d_i, 0)

        # Compute bounded effective contribution for sequencing/visualization
        if next_index is not None:
            window_to_next = max(next_index - d_i, 0)
            effective = min(window_to_next, raw_calendar)
        else:
            cap45 = max(CAP_REFERENCE_DAY - d_i, 0)
            effective = min(unbounded, cap45)

        entry["effective_contribution_days"] = effective
        entry["unused_sla_days"] = max(raw_calendar - effective, 0)

        running_total += effective
        running_total_unbounded += unbounded
        entry["running_total_days"] = running_total
        entry["running_total_days_after"] = running_total
        if not enforce_span_cap:
            entry["effective_contribution_days_unbounded"] = unbounded
            entry["running_total_days_unbounded_after"] = running_total_unbounded
        entry["delta_days"] = effective
        entry["contrib"] = {
            "effective_days": effective,
            "unused_days": entry["unused_sla_days"],
        }
        entry["timeline"] = {
            "from_day": d_i,
            "to_day": d_i + effective,
            "running_total_days_after": running_total,
        }

        adjustments = entry.get("explainer", {}).get("adjustments") or []
        if adjustments:
            adjustments = entry["explainer"].get("adjustments", [])
            notes.extend(f"placement_adjustment:{adj}" for adj in adjustments)
        if include_notes and notes:
            entry["notes"] = notes
        elif not include_notes and "notes" in entry:
            entry.pop("notes", None)
        elif include_notes:
            entry.setdefault("notes", [])

        total_effective += effective
        total_unbounded += unbounded

        # Add running_unbounded_at_submit: total inbound BEFORE this item's contribution
        if not enforce_span_cap:
            running_unbounded_before_this_item = running_total_unbounded - unbounded
            entry["running_unbounded_at_submit"] = running_unbounded_before_this_item

        # Compute explicit overlap metrics with previous item when available (UNBOUNDED)
        if not enforce_span_cap and idx > 0:
            # Unbounded overlap = max(prev_SLA_end_index - curr_submit_index, 0)
            prev_sla_end_index = max((sla_history[idx - 1] - anchor_date).days, 0)
            curr_submit_index = int(entry.get("calendar_day_index", 0))
            overlap_unbounded_days = max(prev_sla_end_index - curr_submit_index, 0)

            entry["overlap_raw_days_with_prev"] = int(overlap_unbounded_days)
            entry["overlap_unbounded_days_with_prev"] = int(overlap_unbounded_days)
            entry["overlap_effective_unbounded_with_prev"] = int(overlap_unbounded_days)  # Backward compat
            total_overlap_unbounded_days += int(overlap_unbounded_days)

    last_entry = sequence[-1]
    first_submit = sequence[0]["calendar_day_index"]
    last_submit = last_entry["calendar_day_index"]
    last_weekday = last_entry["submit_on"]["weekday"]
    # Changed: Only enforce upper bound of 40, no lower bound required
    last_in_window = last_submit <= 40 and last_weekday not in weekend

    summary: Dict[str, object] = {
        "first_submit": first_submit,
        "last_submit": last_submit,
        "last_submit_in_window": last_in_window,
        "total_items": len(sequence),
        "total_effective_days": total_effective,
        "final_submit_date": last_entry["submit_on"]["date"],
        "final_sla_end_date": last_entry["sla_window"]["end"]["date"],
        "distance_to_45": max(CAP_REFERENCE_DAY - last_submit, 0),
    }

    over_45_by_days = max(total_unbounded - total_effective, 0)
    summary["over_45_by_days"] = over_45_by_days

    if not enforce_span_cap:
        # Apply identity formula: total_unbounded = sum(items) - overlap
        sum_unbounded = sum(int(e.get("effective_contribution_days_unbounded", 0)) for e in sequence)
        calculated_total_unbounded = sum_unbounded - int(total_overlap_unbounded_days)
        
        summary["total_effective_days_unbounded"] = calculated_total_unbounded
        summary["final_effective_days_unbounded"] = calculated_total_unbounded
        summary["total_overlap_unbounded_days"] = int(total_overlap_unbounded_days)

        # Expose components for debugging and validation
        summary["_debug_sum_items_unbounded"] = sum_unbounded
        summary["_debug_calculated_inbound"] = calculated_total_unbounded
        summary["_debug_identity_valid"] = True  # Always true by construction now
        
        # Return the calculated value (identity-based)
        total_unbounded = calculated_total_unbounded

    return total_effective, (total_unbounded if not enforce_span_cap else None), summary


def _build_inventory_header_from_sequence(
    sequence: List[Dict[str, object]],
    inventory_base_map: Dict[str, InventoryAllEntry],
    inventory_all: List[InventoryAllEntry],
    closer_field: str,
    bureau_value: Optional[str],
    dedup_notes: Optional[List[str]],
) -> InventoryHeader:
    """Build inventory_header from a specific sequence_debug.
    
    This ensures inventory_selected is consistent with the sequence it represents.
    """
    inventory_selected: List[InventorySelectedEntry] = []
    
    # Anchor date for unbounded index calculations
    anchor_date: Optional[datetime.date] = None
    try:
        # Caller wraps this later; attempt to locate via a submit date of first item
        # Sequence entries contain sla_window start which equals submit date
        if sequence:
            first = sequence[0]
            submit_payload = first.get("submit") or first.get("submit_on") or {}
            date_str = str(submit_payload.get("date", ""))
            if date_str:
                anchor_date = date.fromisoformat(date_str)
    except Exception:
        anchor_date = None

    for order_idx, sequence_entry in enumerate(sequence, start=1):
        field_name = str(sequence_entry.get("field"))
        base_entry = inventory_base_map.get(field_name, {})

        submit_payload = sequence_entry.get("submit") or {}
        submit_date = str(submit_payload.get("date", sequence_entry.get("submit_on", {}).get("date", "")))

        effective_unbounded = int(
            sequence_entry.get(
                "effective_contribution_days_unbounded",
                sequence_entry.get("effective_contribution_days", 0),
            )
        )
        running_unbounded_after = int(
            sequence_entry.get(
                "running_total_days_unbounded_after",
                sequence_entry.get(
                    "running_total_days_after",
                    sequence_entry.get("running_total_days", 0),
                ),
            )
        )

        # Compute unbounded timeline indices
        calendar_day_index = int(sequence_entry.get("calendar_day_index", 0))
        sla_window_payload = sequence_entry.get("sla_window", {})
        sla_end_index = calendar_day_index
        sla_start_index = calendar_day_index
        if anchor_date and isinstance(sla_window_payload, dict):
            try:
                end_date_str = str((sla_window_payload.get("end") or {}).get("date", ""))
                if end_date_str:
                    end_date = date.fromisoformat(end_date_str)
                    sla_end_index = (end_date - anchor_date).days
            except Exception:
                pass
            try:
                start_date_str = str((sla_window_payload.get("start") or {}).get("date", ""))
                if start_date_str:
                    start_date = date.fromisoformat(start_date_str)
                    sla_start_index = (start_date - anchor_date).days
            except Exception:
                sla_start_index = calendar_day_index
        # Fallback: if SLA end still equals submit index yet unbounded span is longer
        # Extend unbounded timeline using the full unbounded effective span (never truncate to raw_calendar_sla_days)
        if sla_end_index == calendar_day_index and effective_unbounded > int(sequence_entry.get("effective_contribution_days", 0)):
            if effective_unbounded > 0:
                sla_end_index = calendar_day_index + effective_unbounded

        selected_entry: InventorySelectedEntry = {
            "field": field_name,
            "default_decision": str(base_entry.get("default_decision", sequence_entry.get("decision", field_name))),
            "business_sla_days": int(base_entry.get("business_sla_days", sequence_entry.get("min_days", 0))),
            "role": str(sequence_entry.get("role", "")),
            "order_idx": order_idx,
            "planned_submit_index": int(sequence_entry.get("calendar_day_index", 0)),
            "planned_submit_date": submit_date,
            "effective_contribution_days": int(sequence_entry.get("effective_contribution_days", 0)),
            "effective_contribution_days_unbounded": effective_unbounded,
            "running_total_after": int(sequence_entry.get("running_total_days_after", sequence_entry.get("running_total_days", 0))),
            "running_total_unbounded_after": running_unbounded_after,
            "is_closer": field_name == closer_field,
            # Canonical unbounded timeline
            "timeline_unbounded": {
                "from_day_unbounded": calendar_day_index,
                "to_day_unbounded": sla_end_index,
                "sla_start_index": sla_start_index,
                "sla_end_index": sla_end_index,
            },
        }
        # Mirror overlap days into inventorySelected for i>1
        try:
            overlap_days = int(
                sequence_entry.get(
                    "overlap_unbounded_days_with_prev",
                    sequence_entry.get("overlap_effective_unbounded_with_prev", 0),
                )
            )
        except Exception:
            overlap_days = 0
        if order_idx > 1:
            selected_entry["overlap_days_with_prev"] = overlap_days
        if bureau_value:
            selected_entry["bureau"] = bureau_value
            state_value = base_entry.get("bureau_dispute_state") if isinstance(base_entry, dict) else None
            if state_value:
                selected_entry["bureau_dispute_state"] = state_value
            selected_entry["bureau_is_missing"] = bool(base_entry.get("bureau_is_missing", False)) if isinstance(base_entry, dict) else False
            selected_entry["bureau_is_mismatch"] = bool(base_entry.get("bureau_is_mismatch", False)) if isinstance(base_entry, dict) else False
        inventory_selected.append(selected_entry)

    inventory_header: InventoryHeader = {
        "inventory_all": deepcopy(inventory_all),
        "inventory_selected": inventory_selected,
    }
    if dedup_notes:
        inventory_header["dedup_notes"] = dedup_notes
    
    return inventory_header


def _with_inventory_header(plan: WeekdayPlan, header: InventoryHeader) -> WeekdayPlan:
    ordered: Dict[str, object] = {}
    for key in ("schema_version", "anchor", "timezone"):
        if key in plan:
            ordered[key] = plan[key]
    ordered["inventory_header"] = deepcopy(header)
    if "inventory_boosters" in plan:
        ordered["inventory_boosters"] = deepcopy(plan.get("inventory_boosters", []))
    ordered["sequence_compact"] = plan.get("sequence_compact", [])
    ordered["sequence_debug"] = plan.get("sequence_debug", [])
    if "sequence_boosters" in plan:
        ordered["sequence_boosters"] = plan.get("sequence_boosters", [])
    for key in (
        "calendar_span_days",
        "last_calendar_day_index",
        "summary",
        "constraints",
        "reason",
        "skipped",
    ):
        if key in plan:
            ordered[key] = plan[key]
    for key, value in plan.items():
        if key not in ordered:
            ordered[key] = value
    return ordered  # type: ignore[return-value]


def _dedup_key_for(item: Dict[str, object], dedup_by: str) -> str:
    if dedup_by == "field":
        return str(item.get("field"))
    if dedup_by == "category":
        value = item.get("category")
    else:
        decision_raw = str(item.get("default_decision", item.get("decision", "")))
        decision_norm = _normalized_decision(decision_raw)
        if decision_norm in {"strong_actionable", "dispute_primary"}:
            return f"{decision_norm}:{item.get('field')}"
        value = item.get("decision")
    return str(value) if value else str(item.get("field"))


def _is_stronger(candidate: Dict[str, object], incumbent: Dict[str, object]) -> bool:
    cand_tuple = (
        int(candidate.get("strength_value", 0)),
        int(candidate.get("min_days", 0)),
        -int(candidate.get("order_index", 0)),
    )
    inc_tuple = (
        int(incumbent.get("strength_value", 0)),
        int(incumbent.get("min_days", 0)),
        -int(incumbent.get("order_index", 0)),
    )
    return cand_tuple > inc_tuple


def _selection_sort_key(item: Dict[str, object]) -> Tuple[int, int, int, str]:
    return (
        -int(item.get("strength_value", 0)),
        -int(item.get("min_days", 0)),
        int(item.get("order_index", 0)),
        str(item.get("field")),
    )


def pack_sequence_to_target_window(
    items: Sequence[Dict[str, object]],
    opener_field: Optional[str],
    closer_field: str,
    anchor_payload: Dict[str, object],
    handoff_range: Tuple[int, int],
    *,
    window: Tuple[int, int],
    enforce_span_cap: bool,
    max_span: int,
    include_notes: bool,
) -> Tuple[WeekdayPlan, List[Dict[str, object]], Dict[str, object]]:
    weekday = int(anchor_payload.get("weekday", 0))

    anchor_date_value = anchor_payload.get("date")
    if isinstance(anchor_date_value, date):
        anchor_date = anchor_date_value
    elif anchor_date_value:
        anchor_date = datetime.fromisoformat(str(anchor_date_value)).date()
    else:
        anchor_date = date.today()

    anchor_reason = str(anchor_payload.get("reason", ""))

    tz_value = anchor_payload.get("tz")
    if isinstance(tz_value, ZoneInfo):
        tz_obj = tz_value
        tz_key = tz_obj.key
    elif isinstance(tz_value, str) and tz_value:
        tz_obj = ZoneInfo(tz_value)
        tz_key = tz_value
    else:
        tz_obj = ZoneInfo("UTC")
        tz_key = tz_obj.key

    weekend_value = anchor_payload.get("weekend") or set()
    weekend_set: Set[int] = {int(day) % 7 for day in weekend_value}

    holidays_value = anchor_payload.get("holidays") or set()
    holidays_set: Set[date] = set()
    for entry in holidays_value:
        if isinstance(entry, date):
            holidays_set.add(entry)
        elif entry:
            try:
                holidays_set.add(datetime.fromisoformat(str(entry)).date())
            except ValueError:
                continue

    target_index = int(anchor_payload.get("target_index", 0))
    target_date_value = anchor_payload.get("target_date")
    if isinstance(target_date_value, date):
        target_date = target_date_value
    elif target_date_value:
        try:
            target_date = datetime.fromisoformat(str(target_date_value)).date()
        except ValueError:
            target_date = None
    else:
        target_date = None

    target_within_window = bool(anchor_payload.get("target_within_window"))
    window_available = bool(anchor_payload.get("window_available"))
    blocked_details = anchor_payload.get("blocked_details") or ()
    attempted_index = anchor_payload.get("attempted_index")
    window_reason = anchor_payload.get("window_reason")
    last_adjustment = anchor_payload.get("last_adjustment")

    window_start, window_end = window
    handoff_min, handoff_max = handoff_range

    item_map: Dict[str, Dict[str, object]] = {}
    for item in items:
        field = str(item.get("field"))
        if not field:
            continue
        item_map[field] = deepcopy(item)

    middle_items = [value for key, value in item_map.items() if key not in {opener_field, closer_field}]
    middle_items.sort(key=lambda entry: (-int(entry.get("min_days", 0)), *_domain_sort_key(entry)))

    ordered_items: List[Dict[str, object]] = []
    if opener_field and opener_field in item_map:
        ordered_items.append(deepcopy(item_map[opener_field]))
    for entry in middle_items:
        ordered_items.append(deepcopy(entry))
    if closer_field in item_map:
        ordered_items.append(deepcopy(item_map[closer_field]))

    if len(ordered_items) < 2:
        ordered_items = [deepcopy(value) for _, value in sorted(item_map.items())]

    dropped_fields: List[str] = []
    best_plan_overall: Optional[WeekdayPlan] = None
    best_logs_overall: List[Dict[str, object]] = []
    best_score_overall: Optional[Tuple[int, ...]] = None
    best_meta_overall: Dict[str, object] = {}

    best_plan_window: Optional[WeekdayPlan] = None
    best_logs_window: List[Dict[str, object]] = []
    best_score_window: Optional[Tuple[int, ...]] = None
    best_meta_window: Dict[str, object] = {}

    working_items = [deepcopy(entry) for entry in ordered_items]

    while working_items:
        _assign_base_placements(working_items)
        gap_combos = _gap_options(working_items, handoff_min=handoff_min, handoff_max=handoff_max)

        for gaps in gap_combos:
            plan_candidate, candidate_logs, _, gap_violation, _ = _build_plan_candidate(
                weekday=weekday,
                anchor_date=anchor_date,
                anchor_reason=anchor_reason,
                tz=tz_obj,
                weekend=weekend_set,
                holidays=holidays_set,
                items=deepcopy(working_items),
                gaps=gaps,
                handoff_min=handoff_min,
                handoff_max=handoff_max,
                target_index=target_index,
                last_adjustment=str(last_adjustment) if last_adjustment else None,
                enforce_span_cap=enforce_span_cap,
                max_span=max_span,
                last_window=window,
                include_notes=include_notes,
            )

            metadata = {
                "target_index": target_index,
                "target_date": target_date,
                "target_within_window": target_within_window,
                "window_available": window_available,
                "blocked_details": blocked_details,
                "attempted_index": attempted_index,
                "window_reason": window_reason,
            }

            raw_sequence = plan_candidate.get("sequence_debug", [])
            raw_closer_index = (
                int(raw_sequence[-1].get("calendar_day_index", 0)) if raw_sequence else 0
            )

            aligned_plan, forced_window_hit, window_adjusted = _force_closer_into_window(
                plan_candidate,
                metadata=metadata,
                target_window=window,
                weekend=weekend_set,
                holidays=holidays_set,
                enforce_span_cap=enforce_span_cap,
                include_notes=include_notes,
            )

            sequence = aligned_plan.get("sequence_debug", [])
            if not sequence:
                continue

            closer_entry = sequence[-1]
            closer_index = int(closer_entry.get("calendar_day_index", 0))
            closer_weekday = closer_entry.get("submit_on", {}).get("weekday")
            # New deadline logic: last_submit <= 40 and not on weekend
            last_submit_ok = closer_index <= 40 and closer_weekday not in weekend_set
            business_day_hit = closer_weekday not in weekend_set
            over_cap_raw = raw_closer_index > 40
            over_cap_final = closer_index > 40

            coverage_ok = True
            for idx in range(len(sequence) - 1):
                current = sequence[idx]
                next_entry = sequence[idx + 1]
                try:
                    sla_end = datetime.fromisoformat(
                        str(current.get("sla_window", {}).get("end", {}).get("date"))
                    ).date()
                except (TypeError, ValueError):
                    coverage_ok = False
                    break
                coverage_index = (sla_end - anchor_date).days
                if int(next_entry.get("calendar_day_index", 0)) > coverage_index:
                    coverage_ok = False
                    break

            total_effective = int(aligned_plan.get("summary", {}).get("total_effective_days", 0))
            items_count = len(sequence)

            adjustments = closer_entry.get("explainer", {}).get("adjustments") or []
            middles_used = [entry.get("field") for entry in sequence[1:-1]]

            meets_deadline = bool(last_submit_ok)
            meets_coverage = bool(total_effective >= 45)
            meets_gap = bool(not gap_violation)
            meets_cap = bool(not over_cap_final)
            meets_core = meets_deadline and meets_coverage and meets_gap and meets_cap

            score = (
                0 if meets_deadline else 1,
                0 if meets_coverage else 1,
                0 if meets_gap else 1,
                0 if meets_cap else 1,
                (items_count if meets_core else 99),  # prefer fewer items only once core met
                -total_effective,                      # within same count, prefer more coverage
                abs(40 - closer_index),                # tie-breaker nearer to 40
            )

            candidate_meta = {
                "deadline_satisfied": last_submit_ok,  # Renamed from window_hit
                "business_day_hit": business_day_hit,
                "coverage_ok": coverage_ok,
                "gap_violation": gap_violation,
                "window_adjusted": bool(window_adjusted),
                "total_effective_days": total_effective,
                "closer_index": closer_index,
                "raw_closer_index": raw_closer_index,
                "over_cap_raw": over_cap_raw,
                "over_cap_final": over_cap_final,
                "forced_window_hit": bool(forced_window_hit),
                "adjustments": list(dict.fromkeys(adjustments)),
                "middles_used": [field for field in middles_used if field],
                "dropped": list(dropped_fields),
            }
            if window_adjusted and raw_closer_index != closer_index:
                candidate_meta["capped_to_index"] = closer_index

            if last_submit_ok and coverage_ok and not gap_violation and not over_cap_final:
                if best_score_window is None or score < best_score_window:
                    best_score_window = score
                    best_plan_window = aligned_plan
                    best_logs_window = candidate_logs
                    best_meta_window = candidate_meta
            else:
                if best_score_overall is None or score < best_score_overall:
                    best_score_overall = score
                    best_plan_overall = aligned_plan
                    best_logs_overall = candidate_logs
                    best_meta_overall = candidate_meta

        if best_plan_window is not None:
            break

        if len(working_items) <= 2:
            break

        drop_idx = _select_middle_index(working_items)
        if drop_idx is None:
            break
        removed = working_items.pop(drop_idx)
        dropped_fields.append(str(removed.get("field")))

    if best_plan_window is not None:
        return best_plan_window, best_logs_window, best_meta_window

    if best_plan_overall is not None:
        return best_plan_overall, best_logs_overall, best_meta_overall

    fallback_plan: WeekdayPlan = {
        "schema_version": 2,
        "anchor": {
            "weekday": weekday,
            "date": anchor_date.isoformat(),
            "reason": anchor_reason,
        },
        "timezone": tz_key,
        "sequence_compact": [],
        "sequence_debug": [],
        "calendar_span_days": 0,
        "last_calendar_day_index": 0,
        "summary": {
            "first_submit": 0,
            "last_submit": 0,
            "last_submit_in_window": False,
            "total_items": 0,
            "total_effective_days": 0,
            "final_submit_date": None,
            "final_sla_end_date": None,
            "distance_to_45": CAP_REFERENCE_DAY,
        },
    }

    return fallback_plan, [], {"deadline_satisfied": False, "dropped": list(dropped_fields)}


_DOMAIN_PRIORITY_OVERRIDES: Dict[str, int] = {
    "seven_year_history": 0,
    "two_year_payment_history": 1,
    "date_of_last_activity": 2,
    "payment_status": 4,
}
_DOMAIN_HISTORY_FALLBACK = 3
_DOMAIN_PAYMENT_STATUS_FALLBACK = 5
_DOMAIN_DEFAULT_PRIORITY = 6


def _domain_priority(item: Dict[str, object]) -> int:
    field = str(item.get("field", "")).strip().lower()
    if field in _DOMAIN_PRIORITY_OVERRIDES:
        return _DOMAIN_PRIORITY_OVERRIDES[field]

    category = str(item.get("category", "")).strip().lower()
    if "history" in field or category == "history":
        return _DOMAIN_HISTORY_FALLBACK
    if "payment_status" in field or category == "payment_status":
        return _DOMAIN_PAYMENT_STATUS_FALLBACK
    return _DOMAIN_DEFAULT_PRIORITY


def _domain_sort_key(item: Dict[str, object]) -> Tuple[int, str]:
    return (_domain_priority(item), str(item.get("field")))


def _closer_priority_key(item: Dict[str, object]) -> Tuple[int, int, int, str]:
    min_days = max(int(item.get("min_days", 0)), 0)
    score_dict = item.get("score")
    if isinstance(score_dict, dict):
        score_total = score_dict.get("total")
    else:
        score_total = None
    try:
        score_value = int(score_total) if score_total is not None else int(item.get("strength_value", 0))
    except (TypeError, ValueError):
        score_value = int(item.get("strength_value", 0))
    domain_priority = _domain_priority(item)
    return (-min_days, domain_priority, -score_value, str(item.get("field")))


def _opener_priority_key(item: Dict[str, object]) -> Tuple[int, int, int, str]:
    min_days = max(int(item.get("min_days", 0)), 0)
    score_dict = item.get("score")
    if isinstance(score_dict, dict):
        score_total = score_dict.get("total")
    else:
        score_total = None
    try:
        score_value = int(score_total) if score_total is not None else int(item.get("strength_value", 0))
    except (TypeError, ValueError):
        score_value = int(item.get("strength_value", 0))
    domain_priority = _domain_priority(item)
    return (-min_days, domain_priority, -score_value, str(item.get("field")))


def _evaluate_sequence_for_selection(
    sequence: Sequence[Dict[str, object]],
    *,
    weekday: int,
    run_dt: datetime,
    tz: ZoneInfo,
    weekend: Set[int],
    holidays: Set[date],
    target_window: Tuple[int, int],
    max_span: int,
    enforce_span_cap: bool,
    handoff_min: int,
    handoff_max: int,
    include_notes: bool,
    opener_field: Optional[str],
    closer_field: Optional[str],
) -> Tuple[WeekdayPlan, bool]:
    plan, _, success, _ = _plan_for_weekday(
        weekday,
        run_dt,
        tz,
        weekend,
        holidays,
        [deepcopy(item) for item in sequence],
        target_window=target_window,
        max_span=max_span,
        enforce_span_cap=enforce_span_cap,
        handoff_min=handoff_min,
        handoff_max=handoff_max,
        include_notes=include_notes,
        opener_field=opener_field,
        closer_field=str(closer_field or ""),
    )
    return plan, success


def _select_findings_varlen(
    items: List[Dict[str, object]],
    *,
    run_dt: datetime,
    tz: ZoneInfo,
    weekend: Set[int],
    holidays: Set[date],
    target_window: Tuple[int, int],
    max_span: int,
    enforce_span_cap: bool,
    handoff_min: int,
    handoff_max: int,
    target_effective_days: int,
    min_increment_days: int,
    dedup_by: str,
    include_supporters: bool,
    include_notes: bool,
) -> Tuple[
    List[Dict[str, object]],
    List[Dict[str, object]],
    List[DedupNote],
    int,
    RoleSelectionMeta,
]:
    selection_logs: List[Dict[str, object]] = []
    dedup_map: Dict[str, Dict[str, object]] = {}
    dedup_records: Dict[str, Dict[str, object]] = {}
    dedup_dropped = 0

    for item in items:
        if not include_supporters and str(item.get("role")) == "supporter":
            selection_logs.append(
                {
                    "event": "candidate_rejected",
                    "decision": item.get("field"),
                    "reason": "supporters_disabled",
                    "ts": _utc_now_iso(),
                }
            )
            continue

        key = _dedup_key_for(item, dedup_by)
        incumbent = dedup_map.get(key)
        if incumbent is None:
            dedup_map[key] = item
            continue

        if _is_stronger(item, incumbent):
            selection_logs.append(
                {
                    "event": "candidate_rejected",
                    "decision": incumbent.get("field"),
                    "reason": "duplicate",
                    "ts": _utc_now_iso(),
                }
            )
            dedup_map[key] = item
            record = dedup_records.setdefault(
                key,
                {
                    "kept": str(item.get("field")),
                    "dropped": set(),
                    "reason": f"dedup: {dedup_by}",
                },
            )
            record["kept"] = str(item.get("field"))
            record["dropped"].add(str(incumbent.get("field")))
            dedup_dropped += 1
        else:
            selection_logs.append(
                {
                    "event": "candidate_rejected",
                    "decision": item.get("field"),
                    "reason": "duplicate",
                    "ts": _utc_now_iso(),
                }
            )
            record = dedup_records.setdefault(
                key,
                {
                    "kept": str(incumbent.get("field")),
                    "dropped": set(),
                    "reason": f"dedup: {dedup_by}",
                },
            )
            record["dropped"].add(str(item.get("field")))
            dedup_dropped += 1

    pool = list(dedup_map.values())
    if len(pool) < 2:
        raise PlannerConfigurationError(
            "Variable-length planner requires at least two primary findings after deduplication"
        )

    sorted_pool = sorted(pool, key=_selection_sort_key)
    strong_pool = [
        item
        for item in sorted_pool
        if _normalized_decision(str(item.get("default_decision", item.get("decision", ""))))
        == "strong_actionable"
    ]

    if not sorted_pool:
        raise PlannerConfigurationError("No eligible findings available after deduplication")

    # NEW SELECTION LOGIC: Choose closer first (max business_sla_days), then opener (best score with days<=closer preference)
    
    # Step 1: Define closer candidates (strong_actionable + supportive_needs_companion)
    closer_candidates = [
        item for item in sorted_pool
        if _normalized_decision(str(item.get("default_decision", item.get("decision", ""))))
        in {"strong_actionable", "supportive_needs_companion"}
    ]
    if not closer_candidates:
        closer_candidates = sorted_pool  # Fallback if no strong/supportive items
    
    # Step 2: Choose closer = item with maximum business_sla_days, break ties by score
    max_sla = max(max(int(c.get("min_days", 0)), 0) for c in closer_candidates)
    closer_pool = [c for c in closer_candidates if max(int(c.get("min_days", 0)), 0) == max_sla]
    closer_candidate = max(closer_pool, key=lambda item: (
        int(item.get("strength_value", 0)),
        int(item.get("min_days", 0)),
        -int(item.get("order_index", 0))
    ))
    closer_field_name = str(closer_candidate.get("field"))
    closer_sla = max(int(closer_candidate.get("min_days", 0)), 0)
    closers_eligible_count = len(closer_candidates)
    
    # Step 3: Define opener candidates (strong_actionable only)
    opener_candidates = [
        item for item in sorted_pool
        if _normalized_decision(str(item.get("default_decision", item.get("decision", ""))))
        == "strong_actionable"
    ]
    if not opener_candidates:
        opener_candidates = sorted_pool  # Fallback
    openers_eligible_count = len(opener_candidates)
    
    # Step 4: Choose opener = best score strong_actionable, preferring items with days <= closer_sla
    opener_filtered = [o for o in opener_candidates if max(int(o.get("min_days", 0)), 0) <= closer_sla]
    if opener_filtered:
        # Prefer items with days <= closer
        opener_candidate = max(opener_filtered, key=lambda item: (
            int(item.get("strength_value", 0)),
            int(item.get("min_days", 0)),
            -int(item.get("order_index", 0))
        ))
    else:
        # Fallback: choose best score from all opener_candidates
        opener_candidate = max(opener_candidates, key=lambda item: (
            int(item.get("strength_value", 0)),
            int(item.get("min_days", 0)),
            -int(item.get("order_index", 0))
        ))
    opener_field_name = str(opener_candidate.get("field"))
    
    # Step 5: Ensure opener != closer when possible
    if opener_field_name == closer_field_name and len(sorted_pool) > 1:
        # Keep closer, recompute opener from remaining candidates
        remaining_opener_candidates = [o for o in opener_candidates if str(o.get("field")) != closer_field_name]
        if remaining_opener_candidates:
            opener_filtered_remaining = [o for o in remaining_opener_candidates if max(int(o.get("min_days", 0)), 0) <= closer_sla]
            if opener_filtered_remaining:
                opener_candidate = max(opener_filtered_remaining, key=lambda item: (
                    int(item.get("strength_value", 0)),
                    int(item.get("min_days", 0)),
                    -int(item.get("order_index", 0))
                ))
            else:
                opener_candidate = max(remaining_opener_candidates, key=lambda item: (
                    int(item.get("strength_value", 0)),
                    int(item.get("min_days", 0)),
                    -int(item.get("order_index", 0))
                ))
            opener_field_name = str(opener_candidate.get("field"))
    
    # Validate closer has max SLA among all items
    all_slas = [max(int(item.get("min_days", 0)), 0) for item in sorted_pool]
    if closer_sla < max(all_slas):
        raise PlannerConfigurationError(
            f"Closer {closer_field_name} (SLA={closer_sla}) is not the max-SLA candidate (max={max(all_slas)})"
        )
    
    domain_tiebreak_applied = False  # No longer using domain tiebreak in new logic
    selection_reason = "closer=max_business_sla_days; opener=best_score_strong_with_days_le_closer_preference"

    chosen_sequence: List[Dict[str, object]] = []
    chosen_fields: Set[str] = set()

    for base_candidate in (opener_candidate, closer_candidate):
        field_name = str(base_candidate.get("field"))
        if field_name in chosen_fields:
            continue
        chosen_sequence.append(deepcopy(base_candidate))
        chosen_fields.add(field_name)

    remaining: List[Dict[str, object]] = [
        deepcopy(item) for item in sorted_pool if str(item.get("field")) not in chosen_fields
    ]

    base_plan, _ = _evaluate_sequence_for_selection(
        chosen_sequence,
        weekday=0,
        run_dt=run_dt,
        tz=tz,
        weekend=weekend,
        holidays=holidays,
        target_window=target_window,
        max_span=max_span,
        enforce_span_cap=enforce_span_cap,
        handoff_min=handoff_min,
        handoff_max=handoff_max,
        include_notes=include_notes,
        opener_field=opener_field_name,
        closer_field=closer_field_name,
    )
    current_total = base_plan["summary"].get("total_effective_days", 0)
    current_window_hit = bool(base_plan["summary"].get("last_submit_in_window", False))
    current_last_index = int(base_plan["summary"].get("last_submit", 0))

    while remaining:
        eligible: List[Dict[str, object]] = []
        rejected_batch: List[Dict[str, str]] = []

        for candidate in remaining:
            tentative_sequence = chosen_sequence[:-1] + [deepcopy(candidate)] + [chosen_sequence[-1]]
            plan, success = _evaluate_sequence_for_selection(
                tentative_sequence,
                weekday=0,
                run_dt=run_dt,
                tz=tz,
                weekend=weekend,
                holidays=holidays,
                target_window=target_window,
                max_span=max_span,
                enforce_span_cap=enforce_span_cap,
                handoff_min=handoff_min,
                handoff_max=handoff_max,
                include_notes=include_notes,
                opener_field=opener_field_name,
                closer_field=closer_field_name,
            )

            new_total = plan["summary"].get("total_effective_days", 0)
            delta = new_total - current_total
            hits_window = bool(plan["summary"].get("last_submit_in_window", False))
            last_index = int(plan["summary"].get("last_submit", 0))
            improves_window = hits_window and not current_window_hit

            if not success:
                rejected_batch.append(
                    {"decision": candidate.get("field"), "reason": "breaks_last_window"}
                )
                continue

            if delta < 0:
                rejected_batch.append(
                    {
                        "decision": candidate.get("field"),
                        "reason": "delta_negative",
                        "delta": delta,
                    }
                )
                continue

            if delta < min_increment_days and not improves_window:
                rejected_batch.append(
                    {"decision": candidate.get("field"), "reason": "delta_below_min", "delta": delta}
                )
                continue

            within_target = new_total <= max(target_effective_days, 0)
            eligible.append(
                {
                    "candidate": candidate,
                    "plan": plan,
                    "delta": delta,
                    "total": new_total,
                    "within_target": within_target,
                    "hits_window": hits_window,
                    "improves_window": improves_window,
                    "last_index": last_index,
                }
            )

        timestamp = _utc_now_iso()
        if not eligible:
            for entry in rejected_batch:
                payload = {
                    "event": "candidate_rejected",
                    "decision": entry.get("decision"),
                    "reason": entry.get("reason"),
                    "ts": timestamp,
                }
                if "delta" in entry:
                    payload["delta_days"] = entry["delta"]
                selection_logs.append(payload)
            break

        for entry in rejected_batch:
            payload = {
                "event": "candidate_rejected",
                "decision": entry.get("decision"),
                "reason": entry.get("reason"),
                "ts": timestamp,
            }
            if "delta" in entry:
                payload["delta_days"] = entry["delta"]
            selection_logs.append(payload)

        eligible.sort(
            key=lambda record: (
                0 if record["improves_window"] else 1,
                0 if record["hits_window"] else 1,
                0 if record["within_target"] else 1,
                -record["delta"],
                abs(target_window[1] - record["last_index"]),
                -int(record["candidate"].get("strength_value", 0)),
                int(record["candidate"].get("order_index", 0)),
            )
        )
        chosen_record = eligible[0]
        candidate = chosen_record["candidate"]

        chosen_sequence = chosen_sequence[:-1] + [deepcopy(candidate)] + [chosen_sequence[-1]]
        current_total = chosen_record["total"]
        current_window_hit = bool(chosen_record.get("hits_window", False))
        current_last_index = int(chosen_record.get("last_index", current_last_index))
        remaining = [item for item in remaining if item["field"] != candidate["field"]]

        selection_logs.append(
            {
                "event": "candidate_accepted",
                "decision": candidate.get("field"),
                "delta_days": chosen_record["delta"],
                "running_total": current_total,
                "hits_window": chosen_record.get("hits_window", False),
                "improves_window": chosen_record.get("improves_window", False),
                "last_index": chosen_record.get("last_index", current_last_index),
                "ts": _utc_now_iso(),
            }
        )

    dedup_notes: List[DedupNote] = []
    for record in dedup_records.values():
        dropped = sorted({entry for entry in record.get("dropped", set()) if entry})
        if not dropped:
            continue
        dedup_notes.append(
            {
                "kept": str(record.get("kept")),
                "dropped": dropped,
                "reason": str(record.get("reason", f"dedup: {dedup_by}")),
            }
        )

    dedup_notes.sort(key=lambda note: (note["kept"], note["reason"]))

    role_meta: RoleSelectionMeta = {
        "closer_field": closer_field_name,
        "opener_field": opener_field_name,
        "domain_tiebreak_applied": domain_tiebreak_applied,
        "openers_eligible": openers_eligible_count,
        "closers_eligible": closers_eligible_count,
        "reason": selection_reason,
    }

    return (
        chosen_sequence,
        selection_logs,
        dedup_notes,
        dedup_dropped,
        role_meta,
    )


def _select_middle_index(items: List[Dict[str, object]]) -> Optional[int]:
    if len(items) <= 2:
        return None
    # choose weakest middle item (minimum strength, keeping order from end)
    middle_candidates = list(range(1, len(items) - 1))
    if not middle_candidates:
        return None
    weakest_idx = min(
        middle_candidates,
        key=lambda idx: (items[idx]["strength_value"], items[idx]["min_days"], idx),
    )
    return weakest_idx


def _empty_plan(
    weekday: int,
    run_dt: datetime,
    tz: ZoneInfo,
    weekend: Set[int],
    holidays: Set[date],
) -> WeekdayPlan:
    anchor_date = next_occurrence_of_weekday(run_dt, weekday % 7, tz)
    anchor_date = _advance_to_business_day(anchor_date, weekend, holidays)
    return {
        "schema_version": 2,
        "anchor": {
            "weekday": weekday % 7,
            "date": anchor_date.isoformat(),
            "reason": f"next occurrence of weekday {weekday % 7} at or after run time",
        },
        "timezone": tz.key,
        "sequence_compact": [],
        "sequence_debug": [],
        "calendar_span_days": 0,
        "last_calendar_day_index": 0,
        "summary": {
            "first_submit": 0,
            "last_submit": 0,
            "last_submit_in_window": False,
            "total_items": 0,
            "total_effective_days": 0,
            "total_effective_days_unbounded": 0,
            "final_submit_date": None,
            "final_sla_end_date": None,
            "distance_to_45": CAP_REFERENCE_DAY,
        },
        "skipped": [],
    }


def _gap_options(
    items: Sequence[Dict[str, object]],
    *,
    handoff_min: int,
    handoff_max: int,
) -> List[Tuple[int, ...]]:
    if len(items) <= 1:
        return [tuple()]

    ranges: List[Tuple[int, ...]] = []
    for idx in range(len(items) - 1):
        prev_min = max(int(items[idx].get("min_days", 0)), 0)
        options: List[int] = [
            gap for gap in range(handoff_min, handoff_max + 1) if gap <= prev_min or prev_min == 0
        ]
        if not options:
            if prev_min > 0:
                options = [min(prev_min, handoff_min)]
            else:
                options = [0]
        ranges.append(tuple(options))

    combos: List[Tuple[int, ...]] = [tuple()]
    for bucket in ranges:
        next_combos: List[Tuple[int, ...]] = []
        for prefix in combos:
            for value in bucket:
                next_combos.append(prefix + (value,))
        combos = next_combos
    return combos or [tuple()]


def _build_plan_candidate(
    *,
    weekday: int,
    anchor_date: date,
    anchor_reason: str,
    tz: ZoneInfo,
    weekend: Set[int],
    holidays: Set[date],
    items: Sequence[Dict[str, object]],
    gaps: Tuple[int, ...],
    handoff_min: int,
    handoff_max: int,
    target_index: int,
    last_adjustment: Optional[str],
    enforce_span_cap: bool,
    max_span: int,
    last_window: Tuple[int, int],
    include_notes: bool,
) -> Tuple[WeekdayPlan, List[Dict[str, object]], bool, bool, int]:
    sequence: List[SequenceItem] = []
    logs: List[Dict[str, object]] = []
    submit_history: List[date] = []
    sla_history: List[date] = []

    gap_violation = False
    idx_gap = 0

    for idx, item in enumerate(items, start=1):
        if idx == 1:
            submit_date = anchor_date
            actual_gap = None
            requested_gap = None
            prev_index = 0
            raw_handoff_calendar = 0
            handoff_before_prev = 0
        else:
            requested_gap = gaps[idx_gap]
            idx_gap += 1
            prev_submit = submit_history[-1]
            prev_sla_end = sla_history[-1]
            prev_min = max(int(sequence[-1]["min_days"]), 0)

            effective_gap = min(requested_gap, prev_min) if prev_min > 0 else 0
            candidate = subtract_business_days_date(prev_sla_end, effective_gap, weekend, holidays)
            if candidate < prev_submit:
                candidate = prev_submit
            submit_date = candidate
            actual_gap = business_days_between(submit_date, prev_sla_end, weekend, holidays)
            if actual_gap < handoff_min or actual_gap > handoff_max:
                gap_violation = True

            logs.append(
                {
                    "event": "cadence_gap_chosen",
                    "weekday": weekday % 7,
                    "field": item.get("field"),
                    "from_field": sequence[-1]["field"],
                    "gap_business_days": actual_gap,
                    "requested_gap": requested_gap,
                    "allowed_range": [handoff_min, handoff_max],
                }
            )

            prev_index = (prev_submit - anchor_date).days
            raw_handoff_calendar = (prev_sla_end - submit_date).days
            if raw_handoff_calendar < 0:
                handoff_before_prev = 0
                logs.append(
                    {
                        "event": "after_sla_end_shifted",
                        "weekday": weekday % 7,
                        "idx": idx,
                        "field": item.get("field"),
                        "from_field": sequence[-1]["field"] if sequence else None,
                        "submit_on": submit_date.isoformat(),
                        "previous_sla_end": prev_sla_end.isoformat(),
                        "days_after": abs(raw_handoff_calendar),
                    }
                )
            else:
                handoff_before_prev = raw_handoff_calendar

        sla_end = advance_business_days_date(submit_date, item["min_days"], weekend, holidays)
        calendar_index = (submit_date - anchor_date).days

        if idx == 1:
            delta_from_prev_days = 0
        else:
            delta_from_prev_days = max(calendar_index - prev_index, 0)

        placement_value = item["placement"]
        adjustments: List[str] = []
        if idx == len(items) and last_adjustment:
            placement_value = last_adjustment
            adjustments.append(last_adjustment)

        if calendar_index < 0:
            gap_violation = True

        explainer: SequenceExplainer = {
            "placement": placement_value,
            "base_placement": item["placement"],
            "why_here": str(item.get("why_here", "")),
            "score": item["score"],
            "strength_metric": item["strength_metric"],
            "strength_value": item["strength_value"],
        }

        if idx == 1:
            explainer["handoff_rule"] = "anchor submission"
        else:
            explainer["handoff_rule"] = (
                f"next starts at ({actual_gap} business days before previous SLA end) "
                f"with range [{handoff_min}..{handoff_max}]"
            )

        if adjustments:
            explainer["adjustments"] = adjustments

        entry: SequenceItem = {
            "idx": idx,
            "field": item["field"],
            "role": item["role"],
            "min_days": item["min_days"],
            "submit_on": _serialize_day(submit_date),
            "submit": {
                "date": submit_date.isoformat(),
                "weekday": ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][submit_date.weekday()],
            },
            "sla_window": {
                "start": _serialize_day(submit_date),
                "end": _serialize_day(sla_end),
            },
            "calendar_day_index": calendar_index,
            "delta_from_prev_days": delta_from_prev_days,
            "handoff_days_before_prev_sla_end": handoff_before_prev,
            "remaining_to_45_cap": max(CAP_REFERENCE_DAY - calendar_index, 0),
            "decision": str(item.get("decision", item["field"])),
            "category": str(item.get("category", "")),
            "explainer": explainer,
        }

        sequence.append(entry)
        submit_history.append(submit_date)
        sla_history.append(sla_end)

        logs.append(
            {
                "event": "schedule_item",
                "weekday": weekday % 7,
                "idx": idx,
                "field": item["field"],
                "submit_on": submit_date.isoformat(),
                "calendar_day_index": calendar_index,
                "placement": placement_value,
            }
        )

    total_effective, total_unbounded, summary_payload = _enrich_sequence_with_contributions(
        sequence,
        submit_history=submit_history,
        sla_history=sla_history,
        anchor_date=anchor_date,
        last_window=last_window,
        weekend=weekend,
        enforce_span_cap=enforce_span_cap,
        include_notes=include_notes,
    )

    last_index = sequence[-1]["calendar_day_index"] if sequence else 0
    calendar_span = last_index

    success = (
        bool(sequence)
        and last_index == target_index
        and not gap_violation
        and (not enforce_span_cap or calendar_span <= max_span)
    )

    plan_payload: WeekdayPlan = {
        "schema_version": 2,
        "anchor": {
            "weekday": weekday % 7,
            "date": anchor_date.isoformat(),
            "reason": anchor_reason,
        },
        "timezone": tz.key,
        "sequence_debug": sequence,
        "sequence_compact": [],
        "sequence": [],
        "calendar_span_days": calendar_span,
        "last_calendar_day_index": last_index,
        "summary": summary_payload,
    }

    _refresh_sequence_views(plan_payload, enforce_span_cap=enforce_span_cap)

    plan_payload["summary"]["total_effective_days"] = total_effective
    if total_unbounded is not None:
        plan_payload["summary"]["total_effective_days_unbounded"] = total_unbounded

    return plan_payload, logs, success, gap_violation, last_index


def _build_sequence_compact(
    sequence_debug: Sequence[SequenceItem],
    *,
    enforce_span_cap: bool,
    anchor_date_str: Optional[str] = None,
) -> List[SequenceCompactEntry]:
    compact: List[SequenceCompactEntry] = []
    anchor_date: Optional[datetime.date] = None
    if anchor_date_str:
        try:
            anchor_date = date.fromisoformat(anchor_date_str)
        except Exception:
            anchor_date = None
    for entry in sequence_debug:
        submit_info = entry.get("submit") or {}
        submit_on = entry.get("submit_on") or {}
        sla_window = entry.get("sla_window") or {}
        start_payload = sla_window.get("start") if isinstance(sla_window, dict) else None
        end_payload = sla_window.get("end") if isinstance(sla_window, dict) else None
        timeline_payload = entry.get("timeline") or {}

        submit_date = str(submit_info.get("date") or submit_on.get("date") or "")
        submit_weekday = str(submit_info.get("weekday") or submit_on.get("weekday_name") or "")

        timeline_from = int(timeline_payload.get("from_day", entry.get("calendar_day_index", 0)))
        timeline_to = int(timeline_payload.get("to_day", timeline_from))

        effective_days = int(entry.get("effective_contribution_days", 0))
        effective_unbounded_value = entry.get("effective_contribution_days_unbounded")
        if effective_unbounded_value is None:
            effective_unbounded_value = effective_days
        effective_unbounded = int(effective_unbounded_value)
        cumulative = int(entry.get("running_total_days_after", entry.get("running_total_days", 0)))
        running_unbounded_after = entry.get("running_total_days_unbounded_after")
        if running_unbounded_after is None or enforce_span_cap:
            running_unbounded_after = cumulative
        cumulative_unbounded = int(running_unbounded_after)

        # Mirror an overlap field for convenience (effective_unbounded overlap)
        overlap_days_with_prev = 0
        if not enforce_span_cap:
            try:
                overlap_days_with_prev = int(entry.get("overlap_effective_unbounded_with_prev", 0))
            except Exception:
                overlap_days_with_prev = 0

        # Compute unbounded timeline indices (canonical inbound window)
        calendar_day_index = int(entry.get("calendar_day_index", timeline_from))
        sla_start_index = calendar_day_index  # fallback
        sla_end_index = timeline_to  # fallback
        if anchor_date and isinstance(end_payload, dict):
            try:
                end_date_str = str(end_payload.get("date", ""))
                if end_date_str:
                    end_date = date.fromisoformat(end_date_str)
                    sla_end_index = (end_date - anchor_date).days
            except Exception:
                pass
            try:
                start_date_str = str((start_payload or {}).get("date", ""))
                if start_date_str:
                    start_date = date.fromisoformat(start_date_str)
                    sla_start_index = (start_date - anchor_date).days
            except Exception:
                sla_start_index = calendar_day_index
        # Fallback: if SLA end index collapsed to submit index but unbounded days imply longer span
        if sla_end_index == calendar_day_index and effective_unbounded > int(sequence_entry.get("effective_contribution_days", 0)):
            raw_calendar_sla = int(sequence_entry.get("raw_calendar_sla_days", effective_unbounded))
            if raw_calendar_sla >= effective_unbounded:
                sla_end_index = calendar_day_index + raw_calendar_sla

        compact.append({
                "idx": int(entry.get("idx", len(compact) + 1)),
                "field": str(entry.get("field", "")),
                "role": str(entry.get("role", "")),
                "submit_date": submit_date,
                "submit_weekday": submit_weekday,
                "window": {
                    "start_date": str((start_payload or {}).get("date", submit_date)),
                    "end_date": str((end_payload or {}).get("date", submit_date)),
                },
                "timeline": {
                    "from_day": timeline_from,
                    "to_day": timeline_to,
                },
                # Canonical unbounded timeline reflecting full SLA span
                "timeline_unbounded": {
                    "from_day_unbounded": calendar_day_index,
                    "to_day_unbounded": sla_end_index,
                    "sla_start_index": sla_start_index,
                    "sla_end_index": sla_end_index,
                },
                "days": {
                    "effective": effective_days,
                    "effective_unbounded": effective_unbounded,
                    "cumulative": cumulative,
                    "cumulative_unbounded": cumulative_unbounded,
                },
                "overlap_days_with_prev": int(overlap_days_with_prev),
                "is_closer": bool(entry.get("is_closer", False)),
                "why_here": str((entry.get("explainer") or {}).get("why_here", "")),
            })

    return compact


def _refresh_sequence_views(plan: WeekdayPlan, *, enforce_span_cap: bool) -> None:
    sequence_debug = plan.get("sequence_debug")
    if sequence_debug is None:
        legacy_sequence = plan.get("sequence")
        if isinstance(legacy_sequence, list):
            sequence_debug = legacy_sequence  # type: ignore[assignment]
        else:
            sequence_debug = []

    if not isinstance(sequence_debug, list):
        sequence_debug = []

    plan["sequence_debug"] = sequence_debug
    anchor_date_str = None
    try:
        anchor_date_str = str((plan.get("anchor") or {}).get("date") or "") or None
    except Exception:
        anchor_date_str = None
    plan["sequence_compact"] = _build_sequence_compact(
        sequence_debug,
        enforce_span_cap=enforce_span_cap,
        anchor_date_str=anchor_date_str,
    )
    if "sequence" in plan:
        plan.pop("sequence", None)


def _resolve_booster_flag(enable_boosters: Optional[bool]) -> bool:
    if enable_boosters is not None:
        return bool(enable_boosters)
    value = os.getenv("ENABLE_STRATEGY_BOOSTERS")
    if value is None:
        return False
    lowered = value.strip().lower()
    if lowered in _TRUE_FLAG_VALUES:
        return True
    if lowered in _FALSE_FLAG_VALUES:
        return False
    return False


def _collect_booster_candidates(
    prepared_items: Sequence[Dict[str, object]],
    used_fields: Set[str],
) -> List[Dict[str, object]]:
    candidates: List[Dict[str, object]] = []
    for item in prepared_items:
        field = str(item.get("field", ""))
        if not field or field in used_fields:
            continue

        min_days = max(int(item.get("min_days", 0)), 0)
        if min_days < 3:
            continue

        decision_norm = _normalized_decision(str(item.get("decision")))
        default_norm = _normalized_decision(str(item.get("default_decision")))
        normalized_meta = _normalized_decision(str(item.get("normalized_decision", decision_norm)))
        normalized_decisions = {decision_norm, default_norm, normalized_meta}

        if not normalized_decisions.intersection(_BOOSTER_ALLOWED_DECISIONS):
            continue
        if "neutral_context_only" in normalized_decisions or "no_case" in normalized_decisions:
            continue

        candidate = dict(item)
        candidate["normalized_decisions"] = normalized_decisions
        candidates.append(candidate)

    candidates.sort(
        key=lambda entry: (
            -int(entry.get("strength_value", entry.get("min_days", 0))),
            -int(entry.get("min_days", 0)),
            str(entry.get("field", "")),
        )
    )
    return candidates


def _adjust_booster_calendar_day(
    desired: int,
    *,
    min_day: int,
    max_day: int,
    anchor_date: date,
    weekend: Set[int],
    holidays: Set[date],
    no_weekend_submit: bool,
) -> Optional[int]:
    weekend_set = {int(day) % 7 for day in weekend}
    holidays_set = set(holidays)
    upper_bound = min(max_day, CAP_REFERENCE_DAY)
    lower_bound = max(min_day, 0)

    def _is_valid(offset: int) -> bool:
        if offset < lower_bound or offset > upper_bound:
            return False
        candidate_date = anchor_date + timedelta(days=offset)
        if no_weekend_submit and candidate_date.weekday() in weekend_set:
            return False
        if candidate_date in holidays_set:
            return False
        return True

    if _is_valid(desired):
        return desired

    for offset in range(desired - 1, lower_bound - 1, -1):
        if _is_valid(offset):
            return offset

    for offset in range(desired + 1, upper_bound + 1):
        if _is_valid(offset):
            return offset

    return None


def _build_sequence_boosters(
    plan: WeekdayPlan,
    *,
    candidates: Sequence[Dict[str, object]],
    field_info_map: Dict[str, Dict[str, object]],
    weekend: Set[int],
    holidays: Set[date],
    no_weekend_submit: bool,
) -> Tuple[List[BoosterHeader], List[BoosterStep]]:
    if not candidates:
        return [], []

    sequence_compact = plan.get("sequence_compact", [])
    if not isinstance(sequence_compact, list) or len(sequence_compact) < 2:
        return [], []

    anchor_payload = plan.get("anchor", {})
    anchor_date_raw = anchor_payload.get("date") if isinstance(anchor_payload, dict) else None
    if not anchor_date_raw:
        return [], []
    try:
        anchor_date = datetime.fromisoformat(str(anchor_date_raw)).date()
    except ValueError:
        return [], []

    weekend_set = {int(day) % 7 for day in weekend} or {5, 6}
    holidays_set = set(holidays)

    field_reason_map = {
        str(field): str(info.get("reason_code")) if info.get("reason_code") is not None else None
        for field, info in field_info_map.items()
    }
    field_category_map = {
        str(field): str(info.get("category", ""))
        for field, info in field_info_map.items()
    }

    anchors: List[Dict[str, object]] = []
    for idx in range(1, len(sequence_compact)):
        prev_entry = sequence_compact[idx - 1]
        next_entry = sequence_compact[idx]
        anchor_field = str(next_entry.get("field"))
        anchor_idx = int(next_entry.get("idx", idx + 1))
        anchors.append(
            {
                "prev": prev_entry,
                "curr": next_entry,
                "anchor_idx": anchor_idx,
                "anchor_field": anchor_field,
                "anchor_reason": field_reason_map.get(anchor_field),
                "anchor_category": field_category_map.get(anchor_field),
            }
        )

    if not anchors:
        return [], []

    booster_headers: List[BoosterHeader] = []
    booster_steps: List[BoosterStep] = []
    assignment_counts: Dict[int, int] = {}
    order_idx = 1

    for candidate in candidates:
        candidate_field = str(candidate.get("field", ""))
        if not candidate_field:
            continue
        candidate_reason = candidate.get("reason_code")
        if candidate_reason is not None:
            candidate_reason = str(candidate_reason)
        candidate_category = str(candidate.get("category", "")) or None
        min_days = max(int(candidate.get("min_days", 0)), 0)

        best_anchor: Optional[Dict[str, object]] = None
        best_score: Optional[Tuple[int, int, int, int]] = None

        for anchor in anchors:
            anchor_idx = anchor["anchor_idx"]  # type: ignore[index]
            anchor_reason = anchor.get("anchor_reason")
            anchor_category = anchor.get("anchor_category")

            reason_score = 0 if candidate_reason and candidate_reason == anchor_reason else 1
            category_score = 0 if candidate_category and candidate_category == anchor_category else 1
            load_score = assignment_counts.get(anchor_idx, 0)
            tie_breaker = int(anchor_idx)
            score = (reason_score, category_score, load_score, tie_breaker)
            if best_score is None or score < best_score:
                best_score = score
                best_anchor = anchor

        if best_anchor is None:
            continue

        prev_entry = best_anchor["prev"]  # type: ignore[index]
        curr_entry = best_anchor["curr"]  # type: ignore[index]
        anchor_idx_value = int(best_anchor["anchor_idx"])  # type: ignore[index]
        anchor_field_value = str(best_anchor.get("anchor_field", "")) or None
        anchor_reason_value = best_anchor.get("anchor_reason")
        anchor_category_value = best_anchor.get("anchor_category")

        prev_timeline = prev_entry.get("timeline", {}) if isinstance(prev_entry, dict) else {}
        curr_timeline = curr_entry.get("timeline", {}) if isinstance(curr_entry, dict) else {}
        prev_from = int(prev_timeline.get("from_day", prev_timeline.get("to_day", 0)))
        next_from = int(curr_timeline.get("from_day", curr_timeline.get("to_day", prev_from)))

        desired_day = max(prev_from, next_from - min_days)
        if desired_day > next_from:
            desired_day = next_from
        desired_day = max(desired_day, 0)

        adjusted_day = _adjust_booster_calendar_day(
            desired_day,
            min_day=prev_from,
            max_day=next_from,
            anchor_date=anchor_date,
            weekend=weekend_set,
            holidays=holidays_set,
            no_weekend_submit=no_weekend_submit,
        )
        if adjusted_day is None:
            continue

        submit_date = anchor_date + timedelta(days=adjusted_day)
        submit_weekday = _WEEKDAY_ABBREVIATIONS[submit_date.weekday()]
        window_end_date = advance_business_days_date(submit_date, min_days, weekend_set, holidays_set)

        curr_days = curr_entry.get("days", {}) if isinstance(curr_entry, dict) else {}
        cumulative = int(curr_days.get("cumulative", curr_days.get("effective", 0)))
        cumulative_unbounded = int(
            curr_days.get("cumulative_unbounded", curr_days.get("cumulative", cumulative))
        )

        bundle_key = None
        if anchor_idx_value and anchor_reason_value:
            bundle_key = f"anchor{anchor_idx_value}_{anchor_reason_value}"

        why_components: List[str] = []
        if anchor_field_value:
            why_components.append(f"booster before {anchor_field_value}")
        if candidate_reason and candidate_reason == anchor_reason_value:
            why_components.append(f"shares reason_code {candidate_reason}")
        elif candidate_reason:
            why_components.append(f"reason_code {candidate_reason}")
        elif candidate_category:
            why_components.append(f"category {candidate_category}")
        if not why_components:
            why_components.append("booster placement")
        why_here = "; ".join(why_components)

        header_entry: BoosterHeader = {
            "field": candidate_field,
            "role": "booster",
            "order_idx": order_idx,
            "paired_with_field": anchor_field_value,
            "paired_with_idx": anchor_idx_value,
            "planned_submit_index": adjusted_day,
            "planned_submit_date": submit_date.isoformat(),
            "reason_code": candidate_reason,
            "category": candidate_category,
        }
        booster_headers.append(header_entry)

        booster_step: BoosterStep = {
            "idx": order_idx,
            "field": candidate_field,
            "role": "booster",
            "submit_date": submit_date.isoformat(),
            "submit_weekday": submit_weekday,
            "window": {
                "start_date": submit_date.isoformat(),
                "end_date": window_end_date.isoformat(),
            },
            "timeline": {
                "from_day": adjusted_day,
                "to_day": adjusted_day,
            },
            "days": {
                "effective": 0,
                "effective_unbounded": 0,
                "cumulative": cumulative,
                "cumulative_unbounded": cumulative_unbounded,
            },
            "is_booster": True,
            "anchor_idx": anchor_idx_value,
            "anchor_field": anchor_field_value,
            "anchor_reason_code": anchor_reason_value,
            "bundle_key": bundle_key,
            "why_here": why_here,
        }
        booster_steps.append(booster_step)

        assignment_counts[anchor_idx_value] = assignment_counts.get(anchor_idx_value, 0) + 1
        order_idx += 1

    return booster_headers, booster_steps


def _plan_for_weekday(
    weekday: int,
    run_dt: datetime,
    tz: ZoneInfo,
    weekend: Set[int],
    holidays: Set[date],
    items: List[Dict[str, object]],
    *,
    target_window: Tuple[int, int],
    max_span: int,
    enforce_span_cap: bool,
    handoff_min: int,
    handoff_max: int,
    include_notes: bool,
    opener_field: Optional[str],
    closer_field: str,
) -> Tuple[WeekdayPlan, List[Dict[str, object]], bool, Dict[str, object]]:
    if not items:
        plan = _empty_plan(weekday, run_dt, tz, weekend, holidays)
        return plan, [], False, {}

    anchor_date = next_occurrence_of_weekday(run_dt, weekday % 7, tz)
    anchor_date = _advance_to_business_day(anchor_date, weekend, holidays)
    anchor_reason = f"next occurrence of weekday {weekday % 7} at or after run time"

    target_date, target_index, target_info = find_business_day_in_window(
        anchor_date,
        target_window,
        weekend,
        holidays,
    )

    last_adjustment: Optional[str] = None
    blocked_details = target_info.get("blocked_details", ())
    target_within_window = bool(target_info.get("within_window"))
    window_available = bool(target_info.get("window_available"))
    attempted_index = target_info.get("attempted_index")
    window_reason: Optional[str] = None
    if blocked_details:
        if any(entry.get("reason") == "holiday" for entry in blocked_details):
            last_adjustment = "shifted_due_to_holiday"
            window_reason = "holiday_blocked"
        else:
            last_adjustment = "shifted_to_avoid_weekend"
            window_reason = "weekend_blocked"
    if not target_within_window and window_reason is None:
        window_reason = "window_unavailable"

    preferred_index = target_info.get("preferred_index")
    preferred_date = None
    if isinstance(preferred_index, int):
        preferred_date = anchor_date + timedelta(days=int(preferred_index))

    logs: List[Dict[str, object]] = [
        {
            "event": "strongest_target_day",
            "weekday": weekday % 7,
            "target": target_index,
            "preferred": preferred_index,
            "preferred_date": preferred_date.isoformat() if preferred_date else None,
            "moved_to": target_date.isoformat() if preferred_date and preferred_index != target_index else None,
            "target_date": target_date.isoformat(),
            "within_window": target_info.get("within_window"),
            "window_available": target_info.get("window_available"),
            "fallback_lower": target_info.get("fallback_lower"),
            "blocked_offsets": list(target_info.get("blocked_offsets", ())),
            "attempted": attempted_index,
        }
    ]

    if preferred_date and preferred_index != target_index:
        logs.append(
            {
                "event": "no_weekend_shift",
                "weekday": weekday % 7,
                "from": preferred_date.isoformat(),
                "to": target_date.isoformat(),
                "reason": last_adjustment or "fallback_lower",
            }
        )

    if not window_available:
        logs.append(
            {
                "event": "planner_no_suitable_target_day",
                "weekday": weekday % 7,
                "detail": window_reason or "window_unavailable",
            }
        )

    anchor_payload = {
        "weekday": weekday,
        "date": anchor_date,
        "reason": anchor_reason,
        "tz": tz,
        "weekend": weekend,
        "holidays": holidays,
        "target_index": target_index,
        "target_date": target_date,
        "target_within_window": target_within_window,
        "window_available": window_available,
        "blocked_details": blocked_details,
        "attempted_index": attempted_index,
        "window_reason": window_reason,
        "last_adjustment": last_adjustment,
        "enforce_span_cap": enforce_span_cap,
        "max_span": max_span,
        "include_notes": include_notes,
    }

    packed_plan, packed_logs, packing_meta = pack_sequence_to_target_window(
        items,
        opener_field,
        closer_field,
        anchor_payload,
        (handoff_min, handoff_max),
        window=target_window,
        enforce_span_cap=enforce_span_cap,
        max_span=max_span,
        include_notes=include_notes,
    )

    packed_plan.setdefault("constraints", None)

    all_logs = logs + packed_logs
    final_success = bool(packing_meta.get("deadline_satisfied") and packing_meta.get("coverage_ok"))

    return packed_plan, all_logs, final_success, {
        "target_index": target_index,
        "anchor_date": anchor_date,
        "target_date": target_date,
        "blocked_details": blocked_details,
        "target_within_window": target_within_window,
        "window_available": window_available,
        "attempted_index": attempted_index,
        "window_reason": window_reason,
        "packing_meta": packing_meta,
    }


def _force_closer_into_window(
    plan: WeekdayPlan,
    *,
    metadata: Dict[str, object],
    target_window: Tuple[int, int],
    weekend: Set[int],
    holidays: Set[date],
    enforce_span_cap: bool,
    include_notes: bool,
) -> Tuple[WeekdayPlan, bool, bool]:
    sequence = plan.get("sequence_debug", [])
    if not sequence:
        return plan, False, False

    window_start, window_end = target_window
    final_entry = sequence[-1]
    current_index = int(final_entry.get("calendar_day_index", 0))
    if window_start <= current_index <= window_end:
        return plan, True, False

    target_date = metadata.get("target_date")
    target_index = metadata.get("target_index")
    if not isinstance(target_date, date) or target_index is None:
        return plan, window_start <= current_index <= window_end, False

    anchor_info = plan.get("anchor", {})
    anchor_date_str = anchor_info.get("date")
    if not anchor_date_str:
        return plan, window_start <= current_index <= window_end, False

    anchor_date = datetime.fromisoformat(str(anchor_date_str)).date()
    weekend_set = {int(day) % 7 for day in weekend}
    holidays_set = set(holidays)

    submit_date: Optional[date] = None
    adjusted_index = current_index

    previous_index = int(sequence[-2]["calendar_day_index"]) if len(sequence) >= 2 else 0
    upper_bound = min(window_end, current_index)

    def _backtrack_to_business_day(offset: int) -> Optional[Tuple[date, int]]:
        candidate = anchor_date + timedelta(days=offset)
        while candidate.weekday() in weekend_set or candidate in holidays_set:
            candidate -= timedelta(days=1)
            new_offset = (candidate - anchor_date).days
            if new_offset < previous_index:
                return None
            offset = new_offset
        if offset < previous_index:
            return None
        return candidate, offset

    search_offsets: List[int] = []
    try:
        raw_target = int(target_index)
        if raw_target <= upper_bound:
            search_offsets.append(raw_target)
    except (TypeError, ValueError):  # pragma: no cover - defensive guard
        pass

    for offset in range(upper_bound, previous_index - 1, -1):
        if offset not in search_offsets:
            search_offsets.append(offset)

    # ensure deterministic search order from latest to earliest without duplicates
    seen_offsets: Set[int] = set()
    ordered_offsets: List[int] = []
    for offset in search_offsets:
        if offset in seen_offsets:
            continue
        seen_offsets.add(offset)
        ordered_offsets.append(offset)

    for offset in ordered_offsets:
        result = _backtrack_to_business_day(offset)
        if result is None:
            continue
        submit_date, adjusted_index = result
        break

    if submit_date is None:
        within_original_window = window_start <= current_index <= window_end
        return plan, within_original_window, False

    final_entry["submit_on"] = _serialize_day(submit_date)
    final_entry.setdefault("submit", {})
    final_entry["submit"]["date"] = submit_date.isoformat()
    final_entry["submit"]["weekday"] = final_entry["submit_on"]["weekday_name"]
    final_entry["calendar_day_index"] = adjusted_index
    # Legacy window fields removed - using ≤40 deadline semantics instead
    final_entry["remaining_to_45_cap"] = max(CAP_REFERENCE_DAY - adjusted_index, 0)

    sla_end = advance_business_days_date(submit_date, int(final_entry.get("min_days", 0)), weekend_set, holidays_set)
    final_entry.setdefault("sla_window", {})
    final_entry["sla_window"]["start"] = _serialize_day(submit_date)
    final_entry["sla_window"]["end"] = _serialize_day(sla_end)

    if len(sequence) >= 2:
        prev_entry = sequence[-2]
        prev_submit = datetime.fromisoformat(str(prev_entry["submit"]["date"])).date()
        prev_sla_end = datetime.fromisoformat(str(prev_entry["sla_window"]["end"]["date"])).date()
        final_entry["delta_from_prev_days"] = max((submit_date - prev_submit).days, 0)
        final_entry["handoff_days_before_prev_sla_end"] = max((prev_sla_end - submit_date).days, 0)
    else:
        final_entry["delta_from_prev_days"] = 0
        final_entry["handoff_days_before_prev_sla_end"] = 0

    explainer = final_entry.setdefault("explainer", {})
    adjustments = explainer.setdefault("adjustments", [])
    if "aligned_to_target_window" not in adjustments:
        adjustments.append("aligned_to_target_window")

    submit_history = [datetime.fromisoformat(str(entry["submit"]["date"])).date() for entry in sequence]
    sla_history = [datetime.fromisoformat(str(entry["sla_window"]["end"]["date"])).date() for entry in sequence]

    total_effective, total_unbounded, summary_payload = _enrich_sequence_with_contributions(
        sequence,
        submit_history=submit_history,
        sla_history=sla_history,
        anchor_date=anchor_date,
        last_window=target_window,
        weekend=weekend_set,
        enforce_span_cap=enforce_span_cap,
        include_notes=include_notes,
    )

    plan["summary"] = summary_payload
    plan["summary"]["total_effective_days"] = total_effective
    if total_unbounded is not None:
        plan["summary"]["total_effective_days_unbounded"] = total_unbounded

    if sequence:
        last_idx = int(sequence[-1]["calendar_day_index"])
        plan["calendar_span_days"] = last_idx
        plan["last_calendar_day_index"] = last_idx
        plan["summary"]["last_submit"] = last_idx
        # Note: last_submit_in_window is set by _enrich_sequence_with_contributions
        # using the rule: last_submit <= 40 and not weekend
        # We do NOT recompute it here with window bounds

    adjusted_flag = adjusted_index != current_index
    # Check if within deadline using the enriched summary value
    within_window_flag = plan.get("summary", {}).get("last_submit_in_window", False)

    _refresh_sequence_views(plan, enforce_span_cap=enforce_span_cap)

    return plan, within_window_flag, adjusted_flag


def optimize_overlap_for_inbound_cap(
    plan: "WeekdayPlan",
    *,
    max_unbounded_inbound_day: int = 50,
    weekend: Set[int],
    holidays: Set[date],
    enforce_span_cap: bool,
    include_notes: bool,
) -> "WeekdayPlan":
    """Post-process a single finalized plan to enforce hard inbound cap (≤ target).

    Rules:
    - Only applies when enforce_span_cap is False and the plan summary contains
      total_effective_days_unbounded strictly greater than max_unbounded_inbound_day.
    - Moves later disputes earlier (never later) to increase SLA overlap.
    - Preserves invariants: non-decreasing submit order, no weekend/holiday submits,
      handoff_days_before_prev_sla_end >= 1, and deadline rule (last_submit <= 40 and not weekend).
        - Hard cap: final total_effective_days_unbounded must be ≤ max_unbounded_inbound_day when legally movable.
    """
    import copy

    # Preconditions
    if enforce_span_cap:
        return plan

    summary = plan.get("summary", {})
    unbounded = summary.get("total_effective_days_unbounded")
    if unbounded is None:
        # Nothing to do when unbounded metric is not available
        return plan
    try:
        inbound_unbounded = int(unbounded)
    except (TypeError, ValueError):  # pragma: no cover - defensive guard
        return plan
    if inbound_unbounded <= int(max_unbounded_inbound_day):
        return plan

    # Preserve the original value for reporting/metadata
    original_inbound_unbounded = inbound_unbounded
    # Deep copy not required for revert anymore; kept for possible diagnostics
    original_plan = copy.deepcopy(plan)

    sequence = plan.get("sequence_debug", [])
    if not isinstance(sequence, list) or len(sequence) < 2:
        return plan

    anchor_payload = plan.get("anchor", {})
    anchor_date_str = anchor_payload.get("date")
    if not anchor_date_str:
        return plan
    try:
        anchor_date = datetime.fromisoformat(str(anchor_date_str)).date()
    except (TypeError, ValueError):  # pragma: no cover - defensive guard
        return plan

    weekend_set = {int(day) % 7 for day in weekend}
    holidays_set = set(holidays)

    # Overlap-based model: compute required total overlap
    # For N items: base_overlap = (N - 1) enforces handoff >= 1 per connection
    # For hard cap: required_overlap = max((S - 50), (N - 1))
    N = len(sequence)
    sum_items_unbounded = sum(int(e.get("effective_contribution_days_unbounded", 0)) for e in sequence)
    base_overlap_min = N - 1  # Minimum 1 day overlap per connection
    
    # Determine required total overlap to achieve cap
    required_overlap = max(sum_items_unbounded - int(max_unbounded_inbound_day), base_overlap_min)
    current_overlap = sum_items_unbounded - inbound_unbounded
    
    # How much additional overlap needed beyond current?
    additional_overlap_needed = required_overlap - current_overlap
    if additional_overlap_needed <= 0:
        # Already at or below target
        return plan

    def _is_business_day(d: date) -> bool:
        return (d.weekday() not in weekend_set) and (d not in holidays_set)

    def _business_days_back(from_date: date, limit_date: date, amount: int) -> date:
        """Move back up to `amount` business days from from_date, not before limit_date.

        limit_date is the strict lower bound (prev submit date). Result must be > limit_date.
        """
        if amount <= 0:
            return from_date
        moved = 0
        current = from_date
        while moved < amount:
            candidate = current - timedelta(days=1)
            # Stop if we would violate strict monotonicity (must be strictly after limit)
            if not (candidate > limit_date):
                break
            if _is_business_day(candidate):
                current = candidate
                moved += 1
            else:
                current = candidate
                # weekend/holiday days don't count toward business-day move amount
                continue
        return current

    # Respect handoff_range limits from constraints (maximum allowed overlap)
    constraints = plan.get("constraints", {})
    handoff_range = constraints.get("handoff_range") or [1, 3]
    try:
        handoff_min = int(handoff_range[0])
        handoff_max = int(handoff_range[1])
    except Exception:  # pragma: no cover - defensive guard
        handoff_min, handoff_max = 1, 3

    # Shift items earlier to increase overlap, enforcing handoff >= 1
    for idx in range(N - 1):
        prev = sequence[idx]
        curr = sequence[idx + 1]
        try:
            prev_submit = datetime.fromisoformat(str(prev["submit"]["date"]).strip()).date()
            curr_submit = datetime.fromisoformat(str(curr["submit"]["date"]).strip()).date()
            prev_sla_end = datetime.fromisoformat(str(prev["sla_window"]["end"]["date"]).strip()).date()
        except Exception:  # pragma: no cover
            continue

        # Recalculate current state after previous shifts
        current_overlap = sum_items_unbounded - inbound_unbounded
        if current_overlap >= required_overlap:
            break  # Target achieved
        
        additional_needed = required_overlap - current_overlap
        if additional_needed <= 0:
            break

        # Try to shift curr earlier to increase overlap
        # CRITICAL: Must maintain handoff_days_before_prev_sla_end >= 1
        working_date = curr_submit
        overlap_gained = 0
        
        while overlap_gained < additional_needed and inbound_unbounded > max_unbounded_inbound_day:
            # Find next earlier business day
            candidate = working_date - timedelta(days=1)
            while not _is_business_day(candidate):
                candidate -= timedelta(days=1)
            
            # ENFORCE INVARIANT: handoff >= 1
            # candidate must be strictly BEFORE prev_sla_end by at least 1 day
            if not (candidate < prev_sla_end):
                # Would violate handoff >= 1
                break
            if not (candidate > prev_submit):
                # Would violate strict monotonic order
                break
            
            # This shift is legal - apply it
            days_moved = (working_date - candidate).days
            working_date = candidate
            overlap_gained += days_moved
            inbound_unbounded -= days_moved

        if working_date == curr_submit or not (working_date > prev_submit):
            continue  # No valid shift applied

        # Update current entry placement
        curr["submit_on"] = _serialize_day(working_date)
        curr.setdefault("submit", {})
        curr["submit"]["date"] = working_date.isoformat()
        curr["submit"]["weekday"] = curr["submit_on"]["weekday_name"]
        curr_index = max((working_date - anchor_date).days, 0)
        curr["calendar_day_index"] = curr_index

        # Recompute SLA window for shifted entry
        min_days = int(curr.get("min_days", 0))
        sla_end = advance_business_days_date(working_date, min_days, weekend_set, holidays_set)
        curr.setdefault("sla_window", {})
        curr["sla_window"]["start"] = _serialize_day(working_date)
        curr["sla_window"]["end"] = _serialize_day(sla_end)
        curr["remaining_to_45_cap"] = max(CAP_REFERENCE_DAY - curr_index, 0)

    # Recompute deltas/overlaps for full sequence (stable and easy to reason about)
    for j in range(1, len(sequence)):
        prev = sequence[j - 1]
        curr = sequence[j]
        try:
            prev_submit = datetime.fromisoformat(str(prev["submit"]["date"]).strip()).date()
            prev_sla_end = datetime.fromisoformat(str(prev["sla_window"]["end"]["date"]).strip()).date()
            curr_submit = datetime.fromisoformat(str(curr["submit"]["date"]).strip()).date()
        except Exception:  # pragma: no cover - defensive guard
            continue
        curr["delta_from_prev_days"] = max((curr_submit - prev_submit).days, 0)
        curr["handoff_days_before_prev_sla_end"] = max((prev_sla_end - curr_submit).days, 0)

    # Re-enrich to recompute contributions and summary
    submit_history = []
    sla_history = []
    for e in sequence:
        try:
            s = datetime.fromisoformat(str(e["submit"]["date"]).strip()).date()
            submit_history.append(s)
        except Exception:  # pragma: no cover - defensive guard
            submit_history.append(anchor_date)
        try:
            end = datetime.fromisoformat(str(e["sla_window"]["end"]["date"]).strip()).date()
            sla_history.append(end)
        except Exception:  # pragma: no cover - defensive guard
            sla_history.append(anchor_date)

    # Resolve last_window for enrichment; only upper bound matters per semantics
    constraints = plan.get("constraints", {})  # re-fetch (may be same ref)
    upper = int(constraints.get("max_last_submit_day", 40))
    last_window = (0, upper)

    total_effective, total_unbounded, summary_payload = _enrich_sequence_with_contributions(
        sequence,
        submit_history=submit_history,
        sla_history=sla_history,
        anchor_date=anchor_date,
        last_window=last_window,
        weekend=weekend_set,
        enforce_span_cap=enforce_span_cap,
        include_notes=include_notes,
    )

    # No revert on under-run: If total_unbounded < cap, we keep it.

    plan["summary"] = summary_payload
    plan["summary"]["total_effective_days"] = total_effective
    if total_unbounded is not None:
        plan["summary"]["total_effective_days_unbounded"] = total_unbounded
    # Metadata for hard cap behavior (overlap-based model)
    final_unbounded = int(plan["summary"].get("total_effective_days_unbounded", total_unbounded or 0) or 0)
    final_overlap = int(plan["summary"].get("total_overlap_unbounded_days", 0))
    
    plan["summary"]["inbound_cap_hard"] = True
    plan["summary"]["inbound_cap_target"] = int(max_unbounded_inbound_day)
    # Do not emit inbound_cap_before; keep only sum, overlap, final unbounded
    plan["summary"]["inbound_cap_after"] = int(final_unbounded)
    plan["summary"]["inbound_cap_applied"] = bool(
        original_inbound_unbounded > max_unbounded_inbound_day and final_unbounded < original_inbound_unbounded
    )
    
    # Add overlap-based diagnostics
    plan["summary"]["inbound_cap_required_overlap"] = required_overlap
    plan["summary"]["inbound_cap_base_overlap_min"] = base_overlap_min
    plan["summary"]["inbound_cap_sum_items_unbounded"] = sum_items_unbounded
    
    if final_unbounded > max_unbounded_inbound_day:
        # We could not legally move enough to reach the cap; annotate unachievable case
        plan["summary"]["inbound_cap_unachievable"] = True
        plan["summary"]["inbound_cap_reason"] = "no_further_legal_overlap_increase"

    if sequence:
        last_idx = int(sequence[-1].get("calendar_day_index", 0))
        plan["calendar_span_days"] = last_idx
        plan["last_calendar_day_index"] = last_idx
        plan["summary"]["last_submit"] = last_idx

    _refresh_sequence_views(plan, enforce_span_cap=enforce_span_cap)

    return plan


def simplify_plan_for_public(plan: WeekdayPlan) -> WeekdayPlan:
    """Trim plan payload removing bounded timeline constructs and internal contribution fields.
    Keeps only required overlap & unbounded timeline semantics per business specification.
    """
    # Use sequence_debug as source of truth then rebuild minimal sequence list.
    seq_debug = plan.get("sequence_debug") or []
    anchor_date_str = str((plan.get("anchor") or {}).get("date", ""))
    try:
        anchor_date = date.fromisoformat(anchor_date_str) if anchor_date_str else None
    except Exception:
        anchor_date = None
    
    # Helper to compute calendar date and weekday from anchor + offset
    weekday_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    def _compute_date_weekday(day_index: int) -> tuple[str, str]:
        if anchor_date is None:
            return ("", "")
        target_date = anchor_date + timedelta(days=day_index)
        weekday_name = weekday_names[target_date.weekday()]
        return (target_date.isoformat(), weekday_name)
    
    has_valid_anchor = anchor_date is not None
    
    minimal_items: List[Dict[str, object]] = []
    for entry in seq_debug:
        idx = int(entry.get("idx", len(minimal_items) + 1))
        submit_index = int(entry.get("calendar_day_index", 0))
        sla_window = entry.get("sla_window", {}) or {}
        start_payload = sla_window.get("start") or {}
        end_payload = sla_window.get("end") or {}
        sla_start_index = submit_index
        sla_end_index = submit_index
        if anchor_date and end_payload.get("date"):
            try:
                end_date = date.fromisoformat(str(end_payload.get("date")))
                sla_end_index = (end_date - anchor_date).days
            except Exception:
                pass
        if anchor_date and start_payload.get("date"):
            try:
                start_date = date.fromisoformat(str(start_payload.get("date")))
                sla_start_index = (start_date - anchor_date).days
            except Exception:
                sla_start_index = submit_index
        raw_calendar = int(entry.get("raw_calendar_sla_days", 0))
        eff_unbounded = int(entry.get("effective_contribution_days_unbounded", raw_calendar))
        # Always prefer the identity-based unbounded effective span; do not shorten to raw_calendar
        if sla_end_index == submit_index and eff_unbounded > 0:
            sla_end_index = submit_index + eff_unbounded
        overlap_unbounded = int(entry.get("overlap_unbounded_days_with_prev", 0)) if idx > 1 else 0
        
        item: Dict[str, object] = {
            "idx": idx,
            "field": entry.get("field"),
            "planned_submit_index": submit_index,
        }
        
        # Only add date fields if we have a valid anchor
        if has_valid_anchor:
            submit_date, submit_weekday = _compute_date_weekday(submit_index)
            unbounded_end_date, unbounded_end_weekday = _compute_date_weekday(sla_end_index)
            item["submit_date"] = submit_date
            item["submit_weekday"] = submit_weekday
            item["unbounded_end_date"] = unbounded_end_date
            item["unbounded_end_weekday"] = unbounded_end_weekday
        
        item["timeline_unbounded"] = {
            "from_day_unbounded": submit_index,
            "to_day_unbounded": sla_end_index,
            "sla_start_index": sla_start_index,
            "sla_end_index": sla_end_index,
        }
        
        if idx > 1:
            item["overlap_unbounded_days_with_prev"] = overlap_unbounded
        minimal_items.append(item)
    # Replace sequence_compact & remove sequence_debug
    plan["sequence_compact"] = minimal_items
    if "sequence_debug" in plan:
        plan.pop("sequence_debug", None)
    # Simplify inventory_selected similarly
    inv_header = plan.get("inventory_header") or {}
    inv_sel = inv_header.get("inventory_selected") or []
    simplified_inv_sel: List[Dict[str, object]] = []
    for pos, e in enumerate(inv_sel, start=1):
        submit_index = int(e.get("planned_submit_index", 0))
        tl_unb = e.get("timeline_unbounded") or {}
        from_unb = int(tl_unb.get("from_day_unbounded", submit_index))
        to_unb = int(tl_unb.get("to_day_unbounded", tl_unb.get("sla_end_index", submit_index)))
        sla_start_index = int(tl_unb.get("sla_start_index", from_unb))
        sla_end_index = int(tl_unb.get("sla_end_index", to_unb))
        
        simplified: Dict[str, object] = {
            "field": e.get("field"),
            "planned_submit_index": submit_index,
        }
        
        # Only add date fields if we have a valid anchor
        if has_valid_anchor:
            submit_date, submit_weekday = _compute_date_weekday(submit_index)
            unbounded_end_date, unbounded_end_weekday = _compute_date_weekday(sla_end_index)
            simplified["submit_date"] = submit_date
            simplified["submit_weekday"] = submit_weekday
            simplified["unbounded_end_date"] = unbounded_end_date
            simplified["unbounded_end_weekday"] = unbounded_end_weekday
        
        simplified["timeline_unbounded"] = {
            "from_day_unbounded": from_unb,
            "to_day_unbounded": to_unb,
            "sla_start_index": sla_start_index,
            "sla_end_index": sla_end_index,
        }
        
        if pos > 1:
            overlap_val = e.get("overlap_days_with_prev") or e.get("overlap_unbounded_days_with_prev") or 0
            simplified["overlap_unbounded_days_with_prev"] = int(overlap_val)
        simplified_inv_sel.append(simplified)
    
    # MERGE SKELETON #2 ITEMS INTO INVENTORY_SELECTED
    # Add enrichment_sequence items to inventory_selected and reorder chronologically
    enrichment_sequence = plan.get("enrichment_sequence", [])
    if enrichment_sequence and isinstance(enrichment_sequence, list):
        # Track which fields are already in inventory_selected (from Skeleton #1)
        s1_fields = {str(item.get("field")) for item in simplified_inv_sel if "field" in item}
        
        # Add Skeleton #2 items, avoiding duplicates
        for enrich_item in enrichment_sequence:
            field = str(enrich_item.get("field", ""))
            if field and field not in s1_fields:
                # Project enrichment item to inventory_selected format
                projected_item: Dict[str, object] = {
                    "field": field,
                    "planned_submit_index": int(enrich_item.get("planned_submit_index", 0)),
                }
                
                # Copy date fields if present
                if "submit_date" in enrich_item:
                    projected_item["submit_date"] = enrich_item["submit_date"]
                if "submit_weekday" in enrich_item:
                    projected_item["submit_weekday"] = enrich_item["submit_weekday"]
                if "unbounded_end_date" in enrich_item:
                    projected_item["unbounded_end_date"] = enrich_item["unbounded_end_date"]
                if "unbounded_end_weekday" in enrich_item:
                    projected_item["unbounded_end_weekday"] = enrich_item["unbounded_end_weekday"]
                
                # Copy timeline_unbounded
                if "timeline_unbounded" in enrich_item:
                    projected_item["timeline_unbounded"] = dict(enrich_item["timeline_unbounded"])
                
                # Copy SLA days
                if "business_sla_days" in enrich_item:
                    projected_item["business_sla_days"] = enrich_item["business_sla_days"]
                
                # Copy optional handoff metadata
                for handoff_key in ["between_skeleton1_indices", "handoff_reference_day", "half_sla_offset"]:
                    if handoff_key in enrich_item:
                        projected_item[handoff_key] = enrich_item[handoff_key]
                
                simplified_inv_sel.append(projected_item)
        
        # Sort combined list by planned_submit_index (ascending), preserving order for ties
        simplified_inv_sel.sort(key=lambda x: int(x.get("planned_submit_index", 0)))
        
        # Recompute idx to reflect chronological order
        for idx, item in enumerate(simplified_inv_sel, start=1):
            item_copy = dict(item)
            item_copy_clean = {k: v for k, v in item_copy.items() if k != "idx"}  # Remove old idx if present
            item_copy_clean["idx"] = idx  # Add new sequential idx
            simplified_inv_sel[idx - 1] = item_copy_clean
    
    if inv_header:
        inv_header["inventory_selected"] = simplified_inv_sel
        plan["inventory_header"] = inv_header
    
    # Preserve enrichment_sequence and enrichment_debug (Skeleton #2)
    # These are already in compact format from _project_compact() in _enrich_with_skeleton2()
    enrichment_sequence = plan.get("enrichment_sequence")
    enrichment_debug = plan.get("enrichment_debug")
    
    # Prune summary to allowed fields only
    summary = plan.get("summary") or {}
    allowed_keys = [
        "inbound_cap_sum_items_unbounded",
        "total_overlap_unbounded_days",
        "total_effective_days_unbounded",
        "inbound_cap_after",
        "enrichment_stats",  # Include enrichment stats in summary
    ]
    plan["summary"] = {k: summary.get(k) for k in allowed_keys if k in summary}
    
    # Re-attach enrichment fields after pruning
    if enrichment_sequence is not None:
        plan["enrichment_sequence"] = enrichment_sequence
    if enrichment_debug is not None:
        plan["enrichment_debug"] = enrichment_debug
    
    # Remove other top-level transient fields
    for k in ["calendar_span_days", "last_calendar_day_index"]:
        plan.pop(k, None)
    return plan


def _find_nearest_business_day_to_40(
    *,
    anchor_date: date,
    weekend: Set[int],
    holidays: Set[date],
    max_search_distance: int = 7,
) -> Optional[int]:
    """Find the nearest business day to day 40, preferring earlier day on tie.
    
    Args:
        anchor_date: Plan anchor date
        weekend: Set of weekend day indices (0=Mon, 6=Sun)
        holidays: Set of holiday dates
        max_search_distance: Maximum distance to search from day 40
        
    Returns:
        Calendar day index (0-based) of nearest business day, or None if not found
    """
    target = 40
    weekend_set = {int(day) % 7 for day in weekend}
    
    # Check target first
    try:
        target_date = anchor_date + timedelta(days=target)
        if target_date.weekday() not in weekend_set and target_date not in holidays:
            return target
    except Exception:
        pass
    
    # Search outward: (39,41), (38,42), ... prefer earlier on tie
    for distance in range(1, max_search_distance + 1):
        # Try earlier first (tie-breaker)
        earlier = target - distance
        if earlier >= 0:
            try:
                earlier_date = anchor_date + timedelta(days=earlier)
                if earlier_date.weekday() not in weekend_set and earlier_date not in holidays:
                    return earlier
            except Exception:
                pass
        
        # Try later
        later = target + distance
        try:
            later_date = anchor_date + timedelta(days=later)
            if later_date.weekday() not in weekend_set and later_date not in holidays:
                return later
        except Exception:
            pass
    
    return None


def _get_strongest_unused_leftover(
    leftover_items: List[Dict[str, object]],
    used_fields: Set[str],
    min_sla_days: int,
) -> Optional[Dict[str, object]]:
    """Select the strongest unused leftover item by business_sla_days.
    
    Args:
        leftover_items: List of leftover items (already sorted by strength)
        used_fields: Set of fields already used (Skeleton #1 + handoff enrichments)
        min_sla_days: Minimum SLA threshold
        
    Returns:
        Strongest unused item, or None if no eligible candidates
    """
    for candidate in leftover_items:
        candidate_field = str(candidate.get("field", ""))
        if candidate_field in used_fields:
            continue
        
        min_days = max(int(candidate.get("min_days", 0)), 0)
        if min_days < min_sla_days:
            continue
        
        return candidate
    
    return None


def _attempt_day40_strongest_enrichment(
    plan: WeekdayPlan,
    leftover_items: List[Dict[str, object]],
    used_fields: Set[str],
    *,
    skeleton2_config: Dict[str, object],
    weekend: Set[int],
    holidays: Set[date],
    anchor_date: date,
    enrich_idx: int,
) -> tuple[Optional[EnrichmentItem], Dict[str, int]]:
    """Attempt to insert one additional enrichment item anchored to day 40.
    
    This rule:
    1. Checks guard condition: pre_closer unbounded_end < 37
    2. Selects strongest unused leftover
    3. Places at day 40 (or nearest business day)
    4. Returns item and updated stats
    
    Args:
        plan: Current plan with sequence_compact
        leftover_items: List of leftover items (already sorted by strength)
        used_fields: Set of fields already used (Skeleton #1 + handoff enrichments)
        skeleton2_config: Skeleton #2 config dict
        weekend: Set of weekend day indices
        holidays: Set of holiday dates
        anchor_date: Plan anchor date
        enrich_idx: Next enrichment item index
        
    Returns:
        Tuple of (enrichment_item or None, stats_dict)
    """
    stats = {
        "attempted_day40": 0,
        "accepted_day40": 0,
        "rejected_day40_guard": 0,
        "rejected_day40_no_unused": 0,
        "rejected_day40_no_business_day": 0,
        "rejected_day40_env_disabled": 0,
    }
    
    # Check ENV flag
    if not skeleton2_config.get("enable_day40_strongest", False):
        stats["rejected_day40_env_disabled"] = 1
        return None, stats
    
    stats["attempted_day40"] = 1
    
    # Guard condition: check pre_closer unbounded_end < 37
    sequence = plan.get("sequence_compact") or plan.get("sequence_debug") or []
    if not isinstance(sequence, list) or len(sequence) < 2:
        stats["rejected_day40_guard"] = 1
        return None, stats
    
    pre_closer = sequence[-2]
    
    # Get pre_closer unbounded end day
    try:
        timeline_unbounded = pre_closer.get("timeline_unbounded", {})
        pre_closer_unbounded_end = int(timeline_unbounded.get("to_day_unbounded", 999))
    except Exception:
        # Fallback: derive from unbounded_end_date
        try:
            unbounded_end_str = str(pre_closer.get("unbounded_end_date", ""))
            if unbounded_end_str:
                unbounded_end_date = date.fromisoformat(unbounded_end_str)
                pre_closer_unbounded_end = (unbounded_end_date - anchor_date).days
            else:
                pre_closer_unbounded_end = 999
        except Exception:
            pre_closer_unbounded_end = 999
    
    # Guard: if pre_closer unbounded end >= 37, rule is disabled
    if pre_closer_unbounded_end >= 37:
        stats["rejected_day40_guard"] = 1
        return None, stats
    
    # Select strongest unused leftover
    min_sla_days = max(skeleton2_config.get("min_sla_days", 5), 1)
    candidate = _get_strongest_unused_leftover(leftover_items, used_fields, min_sla_days)
    
    if candidate is None:
        stats["rejected_day40_no_unused"] = 1
        return None, stats
    
    # Find nearest business day to day 40
    insert_index = _find_nearest_business_day_to_40(
        anchor_date=anchor_date,
        weekend=weekend,
        holidays=holidays,
    )
    
    if insert_index is None:
        stats["rejected_day40_no_business_day"] = 1
        return None, stats
    
    # Build enrichment item with compact dates
    candidate_field = str(candidate.get("field", ""))
    min_days = max(int(candidate.get("min_days", 0)), 0)
    weekend_set = {int(day) % 7 for day in weekend}
    
    try:
        insert_date = anchor_date + timedelta(days=insert_index)
        sla_end_date = advance_business_days_date(
            insert_date, min_days, weekend_set, {d for d in holidays}
        )
    except Exception:
        stats["rejected_day40_no_business_day"] = 1
        return None, stats
    
    # Compute timeline_unbounded: from submit index to SLA end index
    sla_end_index = (sla_end_date - anchor_date).days
    
    enrichment_item: EnrichmentItem = {
        "idx": enrich_idx,
        "field": candidate_field,
        "enrichment_type": "skeleton2",
        "calendar_day_index": insert_index,
        "planned_submit_index": insert_index,
        "submit_date": insert_date.isoformat(),  # type: ignore
        "submit_weekday": _WEEKDAY_ABBREVIATIONS[insert_date.weekday()],  # type: ignore
        "unbounded_end_date": sla_end_date.isoformat(),  # type: ignore
        "unbounded_end_weekday": _WEEKDAY_ABBREVIATIONS[sla_end_date.weekday()],  # type: ignore
        "business_sla_days": min_days,
        "timeline_unbounded": {  # type: ignore
            "from_day_unbounded": insert_index,
            "to_day_unbounded": sla_end_index,
            "sla_start_index": insert_index,
            "sla_end_index": sla_end_index,
        },
        # Debug fields (removed in compact output)
        "submit_on": _serialize_day(insert_date),
        "sla_window": {
            "start": _serialize_day(insert_date),
            "end": _serialize_day(sla_end_date),
        },
        "min_days": min_days,
        "strength_value": int(candidate.get("strength_value", 0)),
        "role": str(candidate.get("role", "supporter")),
        "decision": str(candidate.get("decision", "")),
        "category": str(candidate.get("category", "")),
        "pre_closer_field": str(pre_closer.get("field", "")),  # type: ignore
        "pre_closer_unbounded_end": pre_closer_unbounded_end,  # type: ignore
    }
    
    stats["accepted_day40"] = 1
    return enrichment_item, stats

def _enrich_with_skeleton2(
    plan: WeekdayPlan,
    leftover_items: List[Dict[str, object]],
    *,
    skeleton2_config: Dict[str, object],
    prepared_items: List[Dict[str, object]],
    weekend: Set[int],
    holidays: Set[date],
    anchor_date: date,
) -> WeekdayPlan:
    """Enrich a finalized plan with Skeleton #2 disputes around handoff points.
    
    This function:
    1. Extracts handoff points from sequence_debug (where next core item submits)
    2. For each handoff, tries to place leftover items using half-SLA placement
    3. Respects constraints: no weekend, no bounds extension, no dedup, bureau match
    4. Returns plan with enrichment_sequence populated (Skeleton #1 unchanged)
    
    Args:
        plan: Finalized WeekdayPlan with sequence_debug (Skeleton #1)
        leftover_items: List of items not in Skeleton #1 chosen_sequence
        skeleton2_config: Config dict with enabled, max_items_per_handoff, min_sla_days, 
                         enforce_cadence, placement_mode
        prepared_items: Full prepared items list (for field info)
        weekend: Set of weekend day indices (0=Mon, 6=Sun)
        holidays: Set of holiday dates
        anchor_date: Plan anchor date (reference for calendar_day_index)
    
    Returns:
        Modified plan with enrichment_sequence field populated
    """
    import math
    
    # Early exit if disabled or no plan
    if not plan or not skeleton2_config.get("enabled"):
        plan.setdefault("enrichment_sequence", [])
        return plan
    
    sequence_debug = plan.get("sequence_debug", [])
    if not isinstance(sequence_debug, list) or len(sequence_debug) < 2:
        plan.setdefault("enrichment_sequence", [])
        return plan
    
    # Filter leftover items by SLA threshold
    min_sla_days = max(skeleton2_config.get("min_sla_days", 5), 1)
    eligible_leftover = [
        item for item in leftover_items
        if max(int(item.get("min_days", 0)), 0) >= min_sla_days
    ]
    if not eligible_leftover:
        plan.setdefault("enrichment_sequence", [])
        return plan
    
    # Sort leftover by strength (descending)
    eligible_leftover.sort(
        key=lambda item: (
            -int(item.get("strength_value", 0)),
            -int(item.get("min_days", 0)),
            int(item.get("order_index", 0)),
        )
    )
    
    # Get fields already in Skeleton #1
    skeleton1_fields = {str(entry.get("field")) for entry in sequence_debug}
    
    # Get field metadata from prepared_items
    field_meta_map = {str(item.get("field")): item for item in prepared_items}
    
    # Extract handoff points (where next item submits)
    handoffs = []
    for idx in range(len(sequence_debug) - 1):
        curr_entry = sequence_debug[idx]
        next_entry = sequence_debug[idx + 1]
        handoff_index = int(next_entry.get("calendar_day_index", 0))
        handoffs.append({
            "boundary_idx": idx,
            "handoff_index": handoff_index,
            "curr_entry": curr_entry,
            "next_entry": next_entry,
        })
    
    weekend_set = {int(day) % 7 for day in weekend}
    enrichment_sequence: List[EnrichmentItem] = []
    enrichment_stats = {
        "attempted": 0,
        "accepted": 0,
        "rejected_dedup": 0,
        "rejected_bounds": 0,
        "rejected_weekend": 0,
        "rejected_sla": 0,
        "rejected_fit": 0,
        "rejected_bureau": 0,
    }
    
    max_per_handoff = max(skeleton2_config.get("max_items_per_handoff", 1), 1)
    placement_mode = (skeleton2_config.get("placement_mode", "half_sla_centered") or "half_sla_centered").strip().lower()
    
    enrich_idx = 1  # Start at idx 1 (Skeleton #1 indices start at 1)

    # ========================================
    # PHASE 3 (Reordered to execute first): Day-40 Strongest Leftover Rule
    # ========================================
    day40_item, day40_stats = _attempt_day40_strongest_enrichment(
        plan,
        eligible_leftover,
        skeleton1_fields,
        skeleton2_config=skeleton2_config,
        weekend=weekend,
        holidays=holidays,
        anchor_date=anchor_date,
        enrich_idx=enrich_idx,
    )
    # Merge day-40 stats into enrichment_stats
    enrichment_stats.update(day40_stats)
    # Add day-40 item if successful and advance state
    if day40_item is not None:
        enrichment_sequence.append(day40_item)
        # Reserve idx and mark field as used to prevent reuse by handoffs
        enrich_idx += 1
        try:
            used_field = str(day40_item.get("field", ""))
            if used_field:
                skeleton1_fields.add(used_field)
        except Exception:
            pass

    # ========================================
    # PHASE 2 (Runs after Day-40 now): existing handoff enrichment
    # ========================================
    for handoff_info in handoffs:
        curr_count = 0
        boundary_idx = handoff_info["boundary_idx"]
        handoff_index = handoff_info["handoff_index"]
        next_entry = handoff_info["next_entry"]
        curr_entry = handoff_info["curr_entry"]
        
        prev_sla_end_date = None
        try:
            sla_end_str = str((curr_entry.get("sla_window", {}).get("end", {}).get("date")) or "")
            if sla_end_str:
                prev_sla_end_date = date.fromisoformat(sla_end_str)
        except Exception:
            pass
        
        for candidate in eligible_leftover:
            if curr_count >= max_per_handoff:
                break
            
            candidate_field = str(candidate.get("field", ""))
            enrichment_stats["attempted"] += 1
            
            # Constraint: not in Skeleton #1
            if candidate_field in skeleton1_fields:
                enrichment_stats["rejected_dedup"] += 1
                continue
            
            # Constraint: SLA check (already filtered, but double-check)
            min_days = max(int(candidate.get("min_days", 0)), 0)
            if min_days < min_sla_days:
                enrichment_stats["rejected_sla"] += 1
                continue
            
            # Placement: half-SLA centered
            if placement_mode == "half_sla_centered":
                back_days = math.ceil(min_days / 2)
                insert_index = handoff_index - back_days
            else:
                insert_index = handoff_index - min_days
            
            # Constraint: within bounds (> previous submit, < handoff)
            prev_submit_index = int(curr_entry.get("calendar_day_index", 0))
            if not (prev_submit_index < insert_index < handoff_index):
                enrichment_stats["rejected_bounds"] += 1
                continue
            
            # Constraint: maintain at least 1 day overlap with previous (if applicable)
            if prev_sla_end_date is not None:
                prev_sla_end_index = (prev_sla_end_date - anchor_date).days
                if insert_index >= prev_sla_end_index:
                    enrichment_stats["rejected_fit"] += 1
                    continue
            
            # Constraint: no weekend submit
            try:
                insert_date = anchor_date + timedelta(days=insert_index)
                if insert_date.weekday() in weekend_set:
                    enrichment_stats["rejected_weekend"] += 1
                    continue
            except Exception:
                pass
            
            # Constraint: not extending past plan bounds
            last_index = int(plan.get("last_calendar_day_index", 40))
            sla_end_index = insert_index + min_days
            if sla_end_index > CAP_REFERENCE_DAY:
                enrichment_stats["rejected_bounds"] += 1
                continue
            
            # Build enrichment item
            try:
                insert_date = anchor_date + timedelta(days=insert_index)
                sla_end_date = advance_business_days_date(
                    insert_date, min_days, weekend_set, {d for d in holidays}
                )
            except Exception:
                enrichment_stats["rejected_fit"] += 1
                continue
            
            # Compute sla_end_index for timeline_unbounded
            sla_end_index = (sla_end_date - anchor_date).days
            
            enrichment_item: EnrichmentItem = {
                "idx": enrich_idx,
                "field": candidate_field,
                "enrichment_type": "skeleton2",
                "calendar_day_index": insert_index,
                "planned_submit_index": insert_index,
                "submit_date": insert_date.isoformat(),  # type: ignore
                "submit_weekday": _WEEKDAY_ABBREVIATIONS[insert_date.weekday()],  # type: ignore
                "unbounded_end_date": sla_end_date.isoformat(),  # type: ignore
                "unbounded_end_weekday": _WEEKDAY_ABBREVIATIONS[sla_end_date.weekday()],  # type: ignore
                "business_sla_days": min_days,
                "timeline_unbounded": {  # type: ignore
                    "from_day_unbounded": insert_index,
                    "to_day_unbounded": sla_end_index,
                    "sla_start_index": insert_index,
                    "sla_end_index": sla_end_index,
                },
                "between_skeleton1_indices": [boundary_idx, boundary_idx + 1],  # type: ignore
                "handoff_reference_day": handoff_index,  # type: ignore
                "half_sla_offset": back_days if placement_mode == "half_sla_centered" else min_days,  # type: ignore
                # Debug fields (removed in compact output)
                "submit_on": _serialize_day(insert_date),
                "sla_window": {
                    "start": _serialize_day(insert_date),
                    "end": _serialize_day(sla_end_date),
                },
                "min_days": min_days,
                "strength_value": int(candidate.get("strength_value", 0)),
                "role": str(candidate.get("role", "supporter")),
                "decision": str(candidate.get("decision", "")),
                "category": str(candidate.get("category", "")),
            }
            
            enrichment_sequence.append(enrichment_item)
            enrichment_stats["accepted"] += 1
            enrich_idx += 1
            curr_count += 1
            
            # Mark field as "used" to prevent duplicates in other handoffs
            skeleton1_fields.add(candidate_field)
    
    # Populate plan
    # Build compact view for consumer output; keep full detail in verbose mode under enrichment_debug
    constraints = plan.get("constraints", {}) if isinstance(plan.get("constraints"), dict) else {}
    output_mode_val = str(constraints.get("output_mode", "compact")).strip().lower()

    def _project_compact(item: EnrichmentItem) -> EnrichmentItem:
        # Consumer-facing compact view with computed dates, no debug fields
        base: EnrichmentItem = {
            "idx": int(item.get("idx", 0)),
            "field": str(item.get("field", "")),
            "planned_submit_index": int(item.get("planned_submit_index", item.get("calendar_day_index", 0))),
            "submit_date": str(item.get("submit_date", "")),  # type: ignore
            "submit_weekday": str(item.get("submit_weekday", "")),  # type: ignore
            "unbounded_end_date": str(item.get("unbounded_end_date", "")),  # type: ignore
            "unbounded_end_weekday": str(item.get("unbounded_end_weekday", "")),  # type: ignore
            "business_sla_days": int(item.get("business_sla_days", item.get("min_days", 0))),
        }
        
        # Include timeline_unbounded if present
        if "timeline_unbounded" in item:
            base["timeline_unbounded"] = dict(item["timeline_unbounded"])  # type: ignore
        
        # Include optional handoff fields if this is a handoff placement
        if "between_skeleton1_indices" in item:
            base["between_skeleton1_indices"] = list(item["between_skeleton1_indices"])  # type: ignore
        if "handoff_reference_day" in item:
            base["handoff_reference_day"] = int(item["handoff_reference_day"])  # type: ignore
        if "half_sla_offset" in item:
            base["half_sla_offset"] = int(item["half_sla_offset"])  # type: ignore
        
        return base

    enrich_full = enrichment_sequence
    enrich_compact = [_project_compact(it) for it in enrich_full]

    if output_mode_val == "verbose":
        plan["enrichment_debug"] = {"sequence": enrich_full}
    else:
        plan.pop("enrichment_debug", None)

    plan["enrichment_sequence"] = enrich_compact
    plan.setdefault("summary", {})["enrichment_stats"] = enrichment_stats
    
    return plan


def _rebalance_overlap_distribution(plan: WeekdayPlan, *, weekend: Set[int]) -> None:
    """Rebalance overlap across boundaries when overlap cap is active.

    Business rules implemented (only for per-bureau weekday plans before simplification):
    - If total_items < 2 or cap not active, no changes.
    - required_overlap = sum_items_unbounded - inbound_after.
    - Baseline minimum per boundary = 1 day overlap (if feasible).
    - If required_overlap < (N-1): mark infeasible flag and exit.
    - Else distribute extra overlap as evenly as possible:
        extra = required_overlap - (N-1)
        extra_per_boundary = extra // (N-1)
        remainder = extra % (N-1)
        target_overlap[j] = 1 + extra_per_boundary + (1 if j < remainder else 0)
      (boundaries indexed 0..N-2 for items (i->i+1)).
    - Attempt to shift later items earlier to meet target overlaps without violating:
        * ordering (calendar_day_index strictly increasing)
        * weekend submit prohibition
        * last_submit_window upper bound
        * max_calendar_span (implicit via last submit index)
    - We do not attempt complex business-day handoff recalculation; we honor min incremental ordering only.
    - If a boundary cannot reach at least 1 day overlap, record boundary index in infeasible list.

    NOTE: This operates on `sequence_debug` if present; otherwise on `sequence_compact` minimal form.
          After adjustments we recompute overlap fields unbounded and update summary identity components.
    """
    try:
        seq = plan.get("sequence_debug") or plan.get("sequence_compact") or []
        if not isinstance(seq, list) or len(seq) < 2:
            return
        summary = plan.get("summary") or {}
        # Identity components present only after initial computation; safeguard.
        sum_items = summary.get("inbound_cap_sum_items_unbounded") or summary.get("_debug_sum_items_unbounded")
        # Use inbound_cap_target (the actual target like 50) if available; fallback to inbound_cap_after
        # This ensures weekday plans use correct target, not natural structural total (like 54)
        inbound_target = summary.get("inbound_cap_target") or summary.get("inbound_cap_after") or summary.get("total_effective_days_unbounded")
        if sum_items is None or inbound_target is None:
            return
        sum_items = int(sum_items)
        inbound_target = int(inbound_target)
        required_overlap = sum_items - inbound_target
        boundaries = len(seq) - 1
        if required_overlap <= 0:
            return  # cap not active
        # Compute current overlaps
        def _span(entry):
            # Prefer unbounded span fields if present
            tl_unb = entry.get("timeline_unbounded") or {}
            from_u = tl_unb.get("from_day_unbounded")
            to_u = tl_unb.get("to_day_unbounded")
            if from_u is not None and to_u is not None:
                return int(to_u) - int(from_u)
            # Fallback to raw calendar span if available
            return int(entry.get("effective_contribution_days_unbounded") or entry.get("effective_contribution_days") or 0)
        # Baseline feasibility check
        if required_overlap < boundaries:
            summary["overlap_per_boundary_infeasible"] = True
            plan["summary"] = summary
            return
        extra = required_overlap - boundaries
        extra_per = extra // boundaries
        remainder = extra % boundaries
        target_overlaps = [1 + extra_per + (1 if b < remainder else 0) for b in range(boundaries)]

        # Prepare working arrays
        submit_indices = [int(e.get("planned_submit_index") or e.get("calendar_day_index") or 0) for e in seq]
        spans = [_span(e) for e in seq]

        # Weekend helper
        weekend_set = set(weekend)
        anchor_date_str = str((plan.get("anchor") or {}).get("date", ""))
        anchor_date_obj = None
        try:
            if anchor_date_str:
                from datetime import date as _date
                anchor_date_obj = _date.fromisoformat(anchor_date_str)
        except Exception:
            anchor_date_obj = None
        def _is_weekend(idx: int) -> bool:
            if anchor_date_obj is None:
                return False
            from datetime import timedelta
            d = anchor_date_obj + timedelta(days=idx)
            return d.weekday() in weekend_set

        infeasible_boundaries: List[int] = []
        # Capture initial state to detect if changes actually made
        initial_submit_indices = submit_indices[:]
        initial_total_overlap = summary.get("total_overlap_unbounded_days", 0)
        initial_cap_after = summary.get("inbound_cap_after", inbound_target)
        # Debug snapshot (pre-rebalance) when enabled
        if os.getenv("OVERLAP_DEBUG_LOG"):
            def _calc_overlap_vectors(indices, spans_):
                ovs = []
                tot = 0
                for i in range(1, len(indices)):
                    ov = max(indices[i-1] + spans_[i-1] - indices[i], 0)
                    ovs.append(ov)
                    tot += ov
                return ovs, tot
            _before_ovs, _before_total_ov = _calc_overlap_vectors(submit_indices[:], spans)
            _before_effective = sum(spans) - _before_total_ov
            print(
                f"[REB_DBG:before] plan_id={plan.get('id')} weekday={plan.get('weekday_index')} "
                f"cap_target={summary.get('inbound_cap_target')} effective_days={_before_effective} "
                f"total_overlap={_before_total_ov} overlaps={_before_ovs}"
            )
        # Sequential assignment of submit indices to meet target overlaps, treating cap as a maximum (≤ target)
        for b in range(boundaries):
            prev_submit = submit_indices[b]
            prev_end = prev_submit + spans[b]
            target = target_overlaps[b]
            # Ideal submit index for next item (higher overlap means smaller submit index)
            ideal_submit = prev_end - target
            # Enforce ordering and at-least-1 overlap feasibility window
            min_allowed = prev_submit + 1                    # must be strictly after previous submit
            max_allowed = prev_end - 1                       # at least 1 day overlap
            if max_allowed < min_allowed:
                # Completely infeasible to maintain ≥1 overlap
                infeasible_boundaries.append(b+1)
                submit_indices[b+1] = max(prev_submit + 1, submit_indices[b+1])
                continue
            # Clip ideal within feasible window
            new_submit = max(min(ideal_submit, max_allowed), min_allowed)
            # Weekend adjustment (priority: avoid weekends; prefer earlier to preserve/raise overlap)
            if plan.get("constraints", {}).get("no_weekend_submit", True):
                if _is_weekend(new_submit):
                    chosen = None
                    # Search earlier (toward higher overlap) within [min_allowed, max_allowed]
                    s = new_submit
                    while s >= min_allowed:
                        if not _is_weekend(s):
                            chosen = s
                            break
                        s -= 1
                    if chosen is None:
                        # Search later (toward lower overlap) within window
                        s = new_submit
                        while s <= max_allowed:
                            if not _is_weekend(s):
                                chosen = s
                                break
                            s += 1
                    if chosen is None:
                        # No weekday within feasible window → cannot guarantee ≥1 overlap without weekend
                        infeasible_boundaries.append(b+1)
                        # Choose the nearest weekday outside window by moving later (overlap may become 0)
                        s = prev_end
                        # move forward until weekday
                        while _is_weekend(s):
                            s += 1
                        new_submit = s
                    else:
                        new_submit = chosen
            # Ensure not violating immediate monotonicity (next pass will fix chains)
            submit_indices[b+1] = new_submit
        # Second pass: ensure ordering monotonicity (strictly increasing). If violation occurs, push forward.
        for i in range(1, len(submit_indices)):
            if submit_indices[i] <= submit_indices[i-1]:
                submit_indices[i] = submit_indices[i-1] + 1
        # Recompute overlaps; cap is treated as maximum, so do NOT trim if we exceed required_overlap
        overlaps = []
        total_overlap = 0
        for i in range(1, len(seq)):
            prev_end = submit_indices[i-1] + spans[i-1]
            curr_submit = submit_indices[i]
            ov = max(prev_end - curr_submit, 0)
            overlaps.append(ov)
            total_overlap += ov
        # No trimming: allowing inbound_cap_after ≤ inbound_target
        if os.getenv("OVERLAP_DEBUG_LOG"):
            _after_effective = sum(spans) - total_overlap
            print(
                f"[REB_DBG:after] plan_id={plan.get('id')} weekday={plan.get('weekday_index')} "
                f"cap_target={summary.get('inbound_cap_target')} effective_days={_after_effective} "
                f"total_overlap={total_overlap} overlaps={overlaps}"
            )
        # Write back modifications
        for i, entry in enumerate(seq):
            new_submit = submit_indices[i]
            entry["calendar_day_index"] = new_submit
            entry["planned_submit_index"] = new_submit
            tl_unb = entry.get("timeline_unbounded") or entry.get("timeline") or {}
            span_i = spans[i]
            tl_unb["from_day_unbounded"] = new_submit
            tl_unb["to_day_unbounded"] = new_submit + span_i
            tl_unb["sla_start_index"] = new_submit
            tl_unb["sla_end_index"] = new_submit + span_i
            entry["timeline_unbounded"] = tl_unb
            if i > 0:
                entry["overlap_effective_unbounded_with_prev"] = overlaps[i-1]
                entry["overlap_unbounded_days_with_prev"] = overlaps[i-1]  # Keep for backwards compat
        # Summary updates
        summary["total_overlap_unbounded_days"] = int(sum(overlaps))
        summary["total_effective_days_unbounded"] = int(sum_items - sum(overlaps))
        summary["inbound_cap_after"] = int(sum_items - sum(overlaps))
        if infeasible_boundaries:
            summary["overlap_boundary_infeasible"] = True
            summary["overlap_boundary_infeasible_list"] = infeasible_boundaries
        # Set flag ONLY if actual changes made
        final_total_overlap = int(sum(overlaps))
        final_cap_after = int(sum_items - sum(overlaps))
        changes_detected = (
            submit_indices != initial_submit_indices or
            final_total_overlap != initial_total_overlap or
            final_cap_after != initial_cap_after
        )
        if changes_detected:
            summary["overlap_distribution_applied"] = True
        plan["summary"] = summary
        # Mirror inventory_selected
        inv_header = plan.get("inventory_header") or {}
        inv_sel = inv_header.get("inventory_selected") or []
        if inv_sel and len(inv_sel) == len(seq):
            for i, inv_entry in enumerate(inv_sel):
                inv_entry["planned_submit_index"] = submit_indices[i]
                tl_unb = inv_entry.get("timeline_unbounded") or {}
                span_i = spans[i]
                tl_unb["from_day_unbounded"] = submit_indices[i]
                tl_unb["to_day_unbounded"] = submit_indices[i] + span_i
                tl_unb["sla_start_index"] = submit_indices[i]
                tl_unb["sla_end_index"] = submit_indices[i] + span_i
                inv_entry["timeline_unbounded"] = tl_unb
                if i > 0:
                    inv_entry["overlap_unbounded_days_with_prev"] = overlaps[i-1]
            inv_header["inventory_selected"] = inv_sel
            plan["inventory_header"] = inv_header

        # Canonical recompute of date fields from indices (anchor based)
        anchor_block = plan.get("anchor") or {}
        anchor_date_str = str(anchor_block.get("date", ""))
        try:
            from datetime import date as _date, timedelta as _td
            anchor_obj = _date.fromisoformat(anchor_date_str) if anchor_date_str else None
        except Exception:
            anchor_obj = None
        weekday_names = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
        def _derive(idx: int):
            if anchor_obj is None:
                return (None, None)
            d = anchor_obj + _td(days=idx)
            return (d.isoformat(), weekday_names[d.weekday()])
        if anchor_obj is not None:
            # sequence entries
            for entry in seq:
                submit_idx = int(entry.get("planned_submit_index") or entry.get("calendar_day_index") or 0)
                end_idx = int((entry.get("timeline_unbounded") or {}).get("sla_end_index") or submit_idx)
                submit_date, submit_weekday = _derive(submit_idx)
                end_date, end_weekday = _derive(end_idx)
                entry["submit_date"] = submit_date
                entry["submit_weekday"] = submit_weekday
                entry["unbounded_end_date"] = end_date
                entry["unbounded_end_weekday"] = end_weekday
            # inventory_selected
            if inv_sel and len(inv_sel) == len(seq):
                for inv_entry in inv_sel:
                    submit_idx = int(inv_entry.get("planned_submit_index", 0))
                    tl_unb = inv_entry.get("timeline_unbounded") or {}
                    end_idx = int(tl_unb.get("sla_end_index", submit_idx))
                    submit_date, submit_weekday = _derive(submit_idx)
                    end_date, end_weekday = _derive(end_idx)
                    inv_entry["submit_date"] = submit_date
                    inv_entry["submit_weekday"] = submit_weekday
                    inv_entry["unbounded_end_date"] = end_date
                    inv_entry["unbounded_end_weekday"] = end_weekday
                inv_header["inventory_selected"] = inv_sel
                plan["inventory_header"] = inv_header
    except Exception:
        # Fail silent to avoid breaking planner; set flag
        summary = plan.get("summary") or {}
        summary["overlap_distribution_error"] = True
        plan["summary"] = summary

def compute_optimal_plan(
    findings: List[Finding],
    mode: str = "joint_optimize",
    *,
    weekend: Set[int],
    holidays: Optional[Set[date]] = None,
    forced_start: Optional[int] = None,
    account_id: Optional[str] = None,
    timezone_name: str = "America/New_York",
    run_datetime: Optional[datetime] = None,
    max_calendar_span: int = 45,
    last_submit_window: Tuple[int, int] = (0, 40),
    no_weekend_submit: bool = True,
    include_supporters: bool = True,
    exclude_natural_text: bool = True,
    strength_metric: str = "score",
    handoff_min_business_days: int = 1,
    handoff_max_business_days: int = 3,
    enforce_span_cap: bool = False,
    target_effective_days: int = 45,
    min_increment_days: int = 1,
    dedup_by: str = "decision",
    output_mode: str = "compact",
    include_notes: bool = False,
    enable_boosters: Optional[bool] = None,
    bureau: Optional[str] = None,
) -> PlannerOutputs:
    """Compute the dispute plan bundle for downstream writers."""

    findings_list = list(findings)
    _validate_findings(findings_list)

    bureau_value = bureau.strip().lower() if isinstance(bureau, str) else None

    openers, middle, closers, skipped_rank = rank_findings(
        findings_list,
        include_supporters=include_supporters,
        exclude_natural_text=exclude_natural_text,
    )

    canonical_sequence, meta, skipped_build = build_strategy_orders(
        findings_list,
        include_supporters=include_supporters,
        exclude_natural_text=exclude_natural_text,
    )
    if not canonical_sequence:
        raise PlannerConfigurationError("No eligible findings available for planner")

    skipped_lookup: Dict[str, str] = {}
    for source in (skipped_rank, skipped_build):
        for entry in source:
            field = entry.get("field")
            reason = entry.get("reason", "filtered")
            if field and field not in skipped_lookup:
                skipped_lookup[field] = reason

    canonical_fields = {finding.field for finding in canonical_sequence}
    skipped_entries: List[Dict[str, str]] = []
    for finding in findings_list:
        if finding.field not in canonical_fields and finding.field in skipped_lookup:
            skipped_entries.append({"field": finding.field, "reason": skipped_lookup[finding.field]})

    weekend_set = {int(day) % 7 for day in weekend} or {5, 6}
    tz = _resolve_timezone(timezone_name)
    if run_datetime is None:
        run_dt = datetime.now(tz)
    else:
        run_dt = run_datetime.astimezone(tz) if run_datetime.tzinfo else run_datetime.replace(tzinfo=tz)

    holidays_set = set(holidays or set())
    holidays_list = sorted({holiday.isoformat() for holiday in holidays_set})
    weekend_list = sorted(weekend_set)
    boosters_enabled = _resolve_booster_flag(enable_boosters)

    prepared_items = _prepare_items(
        canonical_sequence,
        meta,
        strength_metric=strength_metric,
    )

    per_bureau_inventory_meta: Dict[str, Dict[str, object]] = {}
    if bureau_value:
        for idx, item in enumerate(prepared_items):
            if idx >= len(canonical_sequence):  # pragma: no cover - defensive guard
                break
            finding = canonical_sequence[idx]
            state_norm, bureau_missing, bureau_mismatch, state_present = _resolve_bureau_dispute_metadata(
                finding,
                bureau_value,
            )

            item["bureau"] = bureau_value
            if state_present and state_norm:
                item["bureau_dispute_state"] = state_norm
            item["bureau_is_missing"] = bool(bureau_missing)
            item["bureau_is_mismatch"] = bool(bureau_mismatch)

            per_bureau_inventory_meta[str(item.get("field"))] = {
                "state": state_norm if state_present else None,
                "missing": bool(bureau_missing),
                "mismatch": bool(bureau_mismatch),
            }

    inventory_all: List[InventoryAllEntry] = []
    inventory_base_map: Dict[str, InventoryAllEntry] = {}

    def _append_inventory_entry(
        *,
        field_name: str,
        default_decision_value: str,
        min_days_val: int,
        role_guess: str,
        bureau_meta: Optional[Dict[str, object]] = None,
    ) -> None:
        if field_name in inventory_base_map:
            return

        entry_payload: InventoryAllEntry = {
            "field": field_name,
            "default_decision": default_decision_value or field_name,
            "business_sla_days": min_days_val,
            "role_guess": role_guess,
        }

        if bureau_value:
            entry_payload["bureau"] = bureau_value
            if bureau_meta:
                state_value = bureau_meta.get("state")
                if state_value:
                    entry_payload["bureau_dispute_state"] = state_value
                entry_payload["bureau_is_missing"] = bool(bureau_meta.get("missing", False))
                entry_payload["bureau_is_mismatch"] = bool(bureau_meta.get("mismatch", False))

        inventory_all.append(entry_payload)
        inventory_base_map[field_name] = dict(entry_payload)

    for item in prepared_items:
        field_name = str(item.get("field"))
        if not field_name:
            continue
        min_days_val = int(item.get("min_days", 0))
        default_decision_value = str(item.get("default_decision", item.get("decision", "")))
        normalized_decision = _normalized_decision(default_decision_value)
        role_guess = "opener" if normalized_decision == "strong_actionable" else "supporter"

        bureau_meta = per_bureau_inventory_meta.get(field_name) if bureau_value else None
        _append_inventory_entry(
            field_name=field_name,
            default_decision_value=default_decision_value or str(item.get("decision", field_name)),
            min_days_val=min_days_val,
            role_guess=role_guess,
            bureau_meta=bureau_meta,
        )

    if not include_supporters and skipped_rank:
        supporters_disabled = {
            str(entry.get("field"))
            for entry in skipped_rank
            if entry.get("reason") == "supporters_disabled"
        }
        if supporters_disabled:
            finding_map = {finding.field: finding for finding in findings_list if getattr(finding, "field", None)}
            for supporter_field in supporters_disabled:
                finding = finding_map.get(supporter_field)
                if finding is None:
                    continue
                min_days_val = max(int(getattr(finding, "min_days", 0) or 0), 0)
                default_decision_value = str(getattr(finding, "default_decision", "") or "")
                bureau_meta: Optional[Dict[str, object]] = None
                if bureau_value:
                    state_norm, bureau_missing, bureau_mismatch, state_present = _resolve_bureau_dispute_metadata(
                        finding,
                        bureau_value,
                    )
                    bureau_meta = {
                        "state": state_norm if state_present else None,
                        "missing": bool(bureau_missing),
                        "mismatch": bool(bureau_mismatch),
                    }
                    per_bureau_inventory_meta[supporter_field] = dict(bureau_meta)

                _append_inventory_entry(
                    field_name=supporter_field,
                    default_decision_value=default_decision_value or supporter_field,
                    min_days_val=min_days_val,
                    role_guess="supporter",
                    bureau_meta=bureau_meta,
                )

    inventory_all.sort(
        key=lambda entry: (-entry["business_sla_days"], entry["field"])
    )

    # PHASE 5: Constraints output structure
    # last_submit_window is kept for backward compatibility, but only the upper bound is enforced.
    # Plans are accepted if last_submit <= window[1] (typically 40) and not on weekend.
    # The lower bound is deprecated and not enforced.
    constraints = {
        "max_calendar_span": max_calendar_span,
        "last_submit_window": [0, last_submit_window[1]],  # Normalized to [0, upper_bound]
        "max_last_submit_day": last_submit_window[1],  # Only upper bound enforced (≤40 semantics)
        "no_weekend_submit": no_weekend_submit,
        "handoff_range": [handoff_min_business_days, handoff_max_business_days],
        "enforce_span_cap": enforce_span_cap,
        "target_effective_days": target_effective_days,
        "min_increment_days": min_increment_days,
        "dedup_by": dedup_by,
        "output_mode": output_mode,
        "include_notes": include_notes,
    }

    if len(prepared_items) < 2:
        weekday_plans: Dict[int, WeekdayPlan] = {}
        schedule_logs: List[Dict[str, object]] = [
            {
                "event": "planner_env",
                "last_window": list(last_submit_window),
                "handoff_range": [handoff_min_business_days, handoff_max_business_days],
                "timezone": tz.key,
                "enforce_span_cap": enforce_span_cap,
                "max_calendar_span": max_calendar_span,
                "include_supporters": include_supporters,
                "exclude_natural_text": exclude_natural_text,
                "no_weekend_submit": no_weekend_submit,
                "strength_metric": strength_metric,
                "target_effective_days": target_effective_days,
                "min_increment_days": min_increment_days,
                "dedup_by": dedup_by,
                "output_mode": output_mode,
                "include_notes": include_notes,
            },
            {
                "event": "planner_impossible_window",
                "reason": "insufficient_items",
            },
        ]

        inventory_header: InventoryHeader = {
            "inventory_all": deepcopy(inventory_all),
            "inventory_selected": [],
        }

        for weekday in range(5):
            plan = _empty_plan(weekday, run_dt, tz, weekend_set, holidays_set)
            plan["reason"] = "planner_impossible_window"
            plan["constraints"] = dict(constraints)
            plan["skipped"] = skipped_entries
            plan["inventory_boosters"] = []
            plan["sequence_boosters"] = []
            if not enforce_span_cap:
                plan["summary"]["total_effective_days_unbounded"] = 0
            weekday_plans[weekday] = _with_inventory_header(plan, inventory_header)

        summary_block: Dict[str, object] = {
            "first_submit": 0,
            "last_submit": 0,
            "last_submit_in_window": False,
            "total_items": 0,
            "total_effective_days": 0,
            "final_submit_date": None,
            "final_sla_end_date": None,
            "distance_to_45": CAP_REFERENCE_DAY,
        }
        if not enforce_span_cap:
            summary_block["total_effective_days_unbounded"] = 0

        master_plan: MasterPlan = {
            "schema_version": 1,
            "generated_at": _utc_now_iso(),
            "timezone": tz.key,
            "mode_used": mode,
            "weekend": weekend_list,
            "holidays": holidays_list,
            "best_overall": {
                "start_weekday": forced_start or 0,
                "calendar_span_days": 0,
                "order": [item["field"] for item in prepared_items],
                "last_calendar_day_index": 0,
            },
            "by_weekday": {str(idx): weekday_plans[idx] for idx in range(5)},
            "meta": {
                "openers_count": len(openers),
                "supporters_count": len(middle),
                "closers_count": len(closers),
            },
            "reason": "planner_impossible_window",
            "calendar_span_days": 0,
            "last_calendar_day_index": 0,
            "summary": summary_block,
            "constraints": dict(constraints),
            "skipped": skipped_entries,
            "inventory_header": deepcopy(inventory_header),
            "inventory_boosters": [],
            "sequence_boosters": [],
        }

        if bureau_value:
            master_plan["bureau"] = bureau_value

        schedule_logs.append(
            {
                "event": "inventory_snapshot",
                "all": len(inventory_all),
                "selected": 0,
                "dedup_dropped": 0,
                "ts": _utc_now_iso(),
            }
        )

        if boosters_enabled:
            schedule_logs.append(
                {
                    "event": "planner_boosters_summary",
                    "mode": mode,
                    "boosters_count": 0,
                    "boosters_fields": [],
                }
            )

        if account_id:
            for event in schedule_logs:
                event.setdefault("account", account_id)
        if bureau_value:
            for event in schedule_logs:
                event.setdefault("bureau", bureau_value)

        return {
            "master": master_plan,
            "weekday_plans": weekday_plans,
            "schedule_logs": schedule_logs,
            "best_weekday": forced_start or 0,
            "inventory_boosters": [],
            "sequence_boosters": [],
        }

    sorted_items = sorted(
        prepared_items,
        key=lambda item: (item["strength_value"], item["min_days"], item["field"]),
        reverse=True,
    )

    (
        base_sequence,
        selection_logs,
        dedup_notes,
        dedup_dropped,
        role_selection,
    ) = _select_findings_varlen(
        sorted_items,
        run_dt=run_dt,
        tz=tz,
        weekend=weekend_set,
        holidays=holidays_set,
        target_window=last_submit_window,
        max_span=max_calendar_span,
        enforce_span_cap=enforce_span_cap,
        handoff_min=handoff_min_business_days,
        handoff_max=handoff_max_business_days,
        target_effective_days=target_effective_days,
        min_increment_days=min_increment_days,
        dedup_by=dedup_by,
        include_supporters=include_supporters,
        include_notes=include_notes,
    )

    closer_field = role_selection["closer_field"]
    opener_field = role_selection["opener_field"]
    domain_tiebreak_applied = role_selection["domain_tiebreak_applied"]
    openers_eligible_count = role_selection["openers_eligible"]
    closers_eligible_count = role_selection["closers_eligible"]
    selection_reason = role_selection["reason"]

    weekday_plans: Dict[int, WeekdayPlan] = {}
    schedule_logs: List[Dict[str, object]] = [
        {
            "event": "planner_env",
            "last_window": list(last_submit_window),
            "handoff_range": [handoff_min_business_days, handoff_max_business_days],
            "timezone": tz.key,
            "enforce_span_cap": enforce_span_cap,
            "max_calendar_span": max_calendar_span,
            "include_supporters": include_supporters,
            "exclude_natural_text": exclude_natural_text,
            "no_weekend_submit": no_weekend_submit,
            "strength_metric": strength_metric,
            "target_effective_days": target_effective_days,
            "min_increment_days": min_increment_days,
            "dedup_by": dedup_by,
            "output_mode": output_mode,
            "include_notes": include_notes,
        }
    ]

    weekday_meta: Dict[int, Dict[str, object]] = {}
    available_weekdays: List[int] = []

    for weekday in range(5):
        plan, logs, success, info = _plan_for_weekday(
            weekday,
            run_dt,
            tz,
            weekend_set,
            holidays_set,
            base_sequence,
            target_window=last_submit_window,
            max_span=max_calendar_span,
            enforce_span_cap=enforce_span_cap,
            handoff_min=handoff_min_business_days,
            handoff_max=handoff_max_business_days,
            include_notes=include_notes,
            opener_field=opener_field,
            closer_field=closer_field,
        )

        plan["constraints"] = dict(constraints)
        plan["skipped"] = skipped_entries
        if not success:
            plan["reason"] = "planner_impossible_window"

        weekday_plans[weekday] = plan
        weekday_meta[weekday] = {
            "success": success,
            "target_index": info.get("target_index"),
            "target_date": info.get("target_date"),
            "last_index": plan.get("last_calendar_day_index", 0),
            "calendar_span": plan.get("calendar_span_days", 0),
            "target_within_window": info.get("target_within_window"),
            "attempted_index": info.get("attempted_index"),
            "window_available": info.get("window_available"),
            "window_reason": info.get("window_reason"),
            "blocked_details": info.get("blocked_details", ()),
            "packing_meta": info.get("packing_meta", {}),
        }

        schedule_logs.extend(logs)

        if plan.get("sequence_debug"):
            available_weekdays.append(weekday)

    window_align_logs: List[Dict[str, object]] = []
    for weekday, plan in list(weekday_plans.items()):
        meta = weekday_meta.get(weekday, {})
        packing_meta = meta.get("packing_meta") if isinstance(meta, dict) else None
        if packing_meta and packing_meta.get("deadline_satisfied"):
            continue

        adjusted_plan, aligned, modified = _force_closer_into_window(
            plan,
            metadata=meta,
            target_window=last_submit_window,
            weekend=weekend_set,
            holidays=holidays_set,
            enforce_span_cap=enforce_span_cap,
            include_notes=include_notes,
        )
        weekday_plans[weekday] = adjusted_plan
        if aligned:
            meta = weekday_meta.setdefault(weekday, {})
            meta["success"] = True
            meta["target_within_window"] = True
            meta["window_reason"] = None
            meta["target_index"] = adjusted_plan.get("last_calendar_day_index", meta.get("target_index"))
            meta["last_index"] = adjusted_plan.get("last_calendar_day_index", meta.get("last_index"))
            meta["calendar_span"] = adjusted_plan.get("calendar_span_days", meta.get("calendar_span"))
            meta["attempted_index"] = adjusted_plan.get("last_calendar_day_index", meta.get("attempted_index"))
            try:
                meta["target_date"] = datetime.fromisoformat(
                    str(adjusted_plan.get("sequence_debug", [])[-1]["submit"]["date"])
                ).date()
            except (IndexError, KeyError, ValueError, TypeError):  # pragma: no cover - defensive guard
                pass
        if modified:
            window_align_logs.append(
                {
                    "event": "closer_window_adjust",
                    "weekday": weekday,
                    "reason": "aligned_to_window",
                    "final_day": int(adjusted_plan.get("last_calendar_day_index", 0)),
                    "ts": _utc_now_iso(),
                }
            )

    schedule_logs.extend(window_align_logs)

    # Resolve best-weekday selection flag early (needed for per-weekday optimization)
    # DEFAULT: per-weekday mode (False) so optimizer runs unless explicitly enabled
    best_weekday_enabled_env = os.getenv("STRATEGY_BEST_WEEKDAY_ENABLED")
    best_weekday_enabled = False  # Changed default from True to False
    if best_weekday_enabled_env is not None:
        lowered = best_weekday_enabled_env.strip().lower()
        if lowered in _FALSE_FLAG_VALUES:
            best_weekday_enabled = False
        elif lowered in _TRUE_FLAG_VALUES:
            best_weekday_enabled = True

    # Capture optimizer environment and weekdays present (no extra schedule logs)

    # Phase: Inbound-cap post-processing for per-weekday mode (when best-weekday is disabled)
    # Apply optimizer to ALL weekday plans independently before best-weekday selection
    if not best_weekday_enabled and not enforce_span_cap:
        for weekday in list(weekday_plans.keys()):
            plan = weekday_plans[weekday]
            unbounded_val = plan.get("summary", {}).get("total_effective_days_unbounded")
            try:
                needs_cap = unbounded_val is not None and int(unbounded_val) > 50
            except (TypeError, ValueError):
                needs_cap = False
            if needs_cap:
                optimized_plan = optimize_overlap_for_inbound_cap(
                    deepcopy(plan),
                    max_unbounded_inbound_day=50,
                    weekend=weekend_set,
                    holidays=holidays_set,
                    enforce_span_cap=enforce_span_cap,
                    include_notes=include_notes,
                )
                weekday_plans[weekday] = optimized_plan

    opener_field_str = str(opener_field) if opener_field else ""
    closer_field_str = str(closer_field)

    for plan in weekday_plans.values():
        sequence_debug = plan.get("sequence_debug", [])
        for entry in sequence_debug:
            field_value = str(entry.get("field"))
            is_closer_entry = field_value == closer_field_str
            is_opener_entry = field_value == opener_field_str
            if is_closer_entry:
                entry["is_closer"] = True
            else:
                entry.pop("is_closer", None)
            if is_closer_entry:
                role_value = "closer"
            elif is_opener_entry:
                role_value = "opener"
            else:
                role_value = "supporter"
            entry["role"] = role_value

        _refresh_sequence_views(plan, enforce_span_cap=enforce_span_cap)

    for entry in inventory_all:
        field_value = str(entry.get("field"))
        if field_value == opener_field_str:
            role_value = "opener"
        elif field_value == closer_field_str:
            role_value = "closer"
        else:
            role_value = "supporter"
        entry["role_guess"] = role_value
        entry.pop("is_closer", None)

        base_entry = inventory_base_map.get(field_value)
        if base_entry is None:
            continue
        base_entry["role_guess"] = role_value
        base_entry.pop("is_closer", None)

    # Build candidates preferring plans that satisfy the deadline
    all_candidates = [(idx, weekday_plans[idx]) for idx in weekday_plans.keys()]
    
    def _selection_key(item: Tuple[int, WeekdayPlan]) -> Tuple[int, int, int, int, int, int]:
        idx, plan = item
        summary = plan.get("summary", {})
        in_window = bool(summary.get("last_submit_in_window", False))
        total_eff = int(summary.get("total_effective_days", 0))
        total_items = int(summary.get("total_items", 0))
        last_idx = int(plan.get("last_calendar_day_index", 0))
        distance_to_40 = abs(40 - last_idx) if in_window else 999
        meets_coverage = bool(total_eff >= 45)
        meets_core = in_window and meets_coverage
        return (
            0 if in_window else 1,           # must satisfy deadline first
            0 if meets_coverage else 1,      # then ensure coverage ≥45
            (total_items if meets_core else 999),  # prefer fewer items once core is met
            -total_eff,                       # then prefer more coverage
            distance_to_40,                   # nearer to 40
            idx,                              # stable tie-breaker
        )
    
    success_candidates = [
        (idx, plan) for idx, plan in all_candidates
        if plan.get("summary", {}).get("last_submit_in_window", False)
    ]

    if forced_start is not None:
        if forced_start < 0 or forced_start > 4:
            raise PlannerConfigurationError(
                "forced_start must be between 0 and 4 for strategy planner"
            )
        best_weekday = forced_start
    elif best_weekday_enabled and success_candidates:
        best_weekday = min(success_candidates, key=_selection_key)[0]
    elif best_weekday_enabled and all_candidates:
        # Fallback: pick best among all plans even if none satisfy deadline
        best_weekday = min(all_candidates, key=_selection_key)[0]
    else:
        best_weekday = forced_start or 0

    best_plan = weekday_plans.get(best_weekday)
    if best_plan is None:
        best_plan = _empty_plan(best_weekday, run_dt, tz, weekend_set, holidays_set)
        best_plan["constraints"] = dict(constraints)
        best_plan["reason"] = "planner_impossible_window"
        weekday_plans[best_weekday] = best_plan

    # Phase: Inbound-cap post-processing for the chosen best plan (only when best-weekday is enabled)
    if best_weekday_enabled and not enforce_span_cap:
        unbounded_val = best_plan.get("summary", {}).get("total_effective_days_unbounded")
        try:
            needs_cap = unbounded_val is not None and int(unbounded_val) > 50
        except (TypeError, ValueError):
            needs_cap = False
        if needs_cap:
            best_plan = optimize_overlap_for_inbound_cap(
                deepcopy(best_plan),
                max_unbounded_inbound_day=50,
                weekend=weekend_set,
                holidays=holidays_set,
                enforce_span_cap=enforce_span_cap,
                include_notes=include_notes,
            )
            # Replace in weekday_plans
            weekday_plans[best_weekday] = best_plan

    # PHASE 2.5: Skeleton #2 Enrichment Layer
    # Apply enrichment to best plan + all weekday plans if enabled
    from .config import load_planner_env as load_env_for_skeleton2
    skeleton2_applied = False
    skeleton2_leftover_count = 0
    skeleton2_added_count = 0
    try:
        skeleton2_env = load_env_for_skeleton2()
        if skeleton2_env.skeleton2_enabled:
            # Reconstruct leftover items from prepared_items
            chosen_fields = {str(entry.get("field")) for entry in best_plan.get("sequence_debug", [])}
            leftover_items = [item for item in prepared_items if str(item.get("field")) not in chosen_fields]
            skeleton2_leftover_count = len(leftover_items)
            
            # Get anchor date from best plan
            anchor_payload = best_plan.get("anchor", {})
            anchor_date_str = anchor_payload.get("date")
            anchor_date_obj = None
            if anchor_date_str:
                try:
                    anchor_date_obj = date.fromisoformat(str(anchor_date_str))
                except Exception:
                    anchor_date_obj = None
            
            if anchor_date_obj and leftover_items:
                # Build skeleton2 config
                skeleton2_config = {
                    "enabled": skeleton2_env.skeleton2_enabled,
                    "max_items_per_handoff": skeleton2_env.skeleton2_max_items_per_handoff,
                    "min_sla_days": skeleton2_env.skeleton2_min_sla_days,
                    "enforce_cadence": skeleton2_env.skeleton2_enforce_cadence,
                    "placement_mode": skeleton2_env.skeleton2_placement_mode,
                    "enable_day40_strongest": skeleton2_env.skeleton2_enable_day40_strongest,
                }

                # Pre-enrichment metrics
                handoffs_count = max(len(best_plan.get("sequence_debug", [])) - 1, 0)
                eligible_count = len([item for item in leftover_items if int(item.get("min_days", 0)) >= int(skeleton2_config["min_sla_days"])])
                schedule_logs.append({
                    "event": "skeleton2_hook_reached",
                    "enabled": True,
                    "leftover_items": len(leftover_items),
                    "handoffs_found": handoffs_count,
                    "eligible_candidates": eligible_count,
                })
                # Log explicit phase ordering for Skeleton #2
                schedule_logs.append({
                    "event": "skeleton2_phase_order",
                    "order": "day40_first_then_handoffs",
                })
                
                # Apply to best plan
                best_plan = _enrich_with_skeleton2(
                    best_plan,
                    leftover_items,
                    skeleton2_config=skeleton2_config,
                    prepared_items=prepared_items,
                    weekend=weekend_set,
                    holidays=holidays_set,
                    anchor_date=anchor_date_obj,
                )
                weekday_plans[best_weekday] = best_plan

                # Post-enrichment stats (best)
                stats = (best_plan.get("summary", {}) or {}).get("enrichment_stats", {})
                if isinstance(stats, dict):
                    schedule_logs.append({
                        "event": "skeleton2_summary_best",
                        **{k: int(stats.get(k, 0)) for k in [
                            "attempted","accepted","rejected_dedup","rejected_bounds","rejected_weekend","rejected_sla","rejected_fit","rejected_bureau",
                            "attempted_day40","accepted_day40","rejected_day40_guard","rejected_day40_no_unused","rejected_day40_no_business_day","rejected_day40_env_disabled"
                        ]},
                    })
                
                # Apply to all weekday plans
                for weekday_idx in range(5):
                    if weekday_idx in weekday_plans and weekday_idx != best_weekday:
                        w_anchor_payload = weekday_plans[weekday_idx].get("anchor", {})
                        w_anchor_str = w_anchor_payload.get("date")
                        w_anchor_obj = None
                        if w_anchor_str:
                            try:
                                w_anchor_obj = date.fromisoformat(str(w_anchor_str))
                            except Exception:
                                w_anchor_obj = None
                        
                        if w_anchor_obj:
                            weekday_plans[weekday_idx] = _enrich_with_skeleton2(
                                weekday_plans[weekday_idx],
                                leftover_items,
                                skeleton2_config=skeleton2_config,
                                prepared_items=prepared_items,
                                weekend=weekend_set,
                                holidays=holidays_set,
                                anchor_date=w_anchor_obj,
                            )
                            w_stats = (weekday_plans[weekday_idx].get("summary", {}) or {}).get("enrichment_stats", {})
                            if isinstance(w_stats, dict):
                                schedule_logs.append({
                                    "event": "skeleton2_summary_weekday",
                                    "weekday": weekday_idx,
                                    **{k: int(w_stats.get(k, 0)) for k in [
                                        "attempted","accepted","rejected_dedup","rejected_bounds","rejected_weekend","rejected_sla","rejected_fit","rejected_bureau",
                                        "attempted_day40","accepted_day40","rejected_day40_guard","rejected_day40_no_unused","rejected_day40_no_business_day","rejected_day40_env_disabled"
                                    ]},
                                })
                
                # Mark S2 as successfully applied with item count
                skeleton2_applied = True
                best_plan_enrich = (best_plan.get("summary", {}) or {}).get("enrichment_stats", {})
                if isinstance(best_plan_enrich, dict):
                    skeleton2_added_count = int(best_plan_enrich.get("accepted", 0))
    except Exception as e:
        # Fail silent: Skeleton #2 is optional enrichment
        import traceback
        if os.getenv("DEBUG_SKELETON2"):
            print(f"DEBUG: Skeleton #2 enrichment error: {e}")
            traceback.print_exc()
    
    # Log S2 runtime application event
    schedule_logs.append({
        "event": "skeleton2_applied_runtime",
        "enabled": skeleton2_applied,
        "leftover": skeleton2_leftover_count,
        "added": skeleton2_added_count,
        "day40_added": 1 if skeleton2_applied and skeleton2_added_count > 0 else 0,
    })

    # PHASE 3: Ensure inventory_selected is built from the final enriched sequence
    # best_plan contains the winner AFTER _force_closer_into_window and enrichment
    best_sequence = best_plan.get("sequence_debug", [])
    assert isinstance(best_sequence, list), "best_sequence must be a list"

    packing_meta = weekday_meta.get(best_weekday, {}).get("packing_meta") if isinstance(weekday_meta.get(best_weekday), dict) else None
    if isinstance(packing_meta, dict) and best_sequence:
        middles_used = packing_meta.get("middles_used")
        if not isinstance(middles_used, list):
            middles_used = [entry["field"] for entry in best_sequence[1:-1]]
        adjustments = packing_meta.get("adjustments")
        if not isinstance(adjustments, list):
            adjustments = []
        summary_block = best_plan.get("summary", {})
        schedule_logs.append(
            {
                "event": "packing_summary",
                "target_window": list(last_submit_window),
                "final_closer_index": int(best_sequence[-1].get("calendar_day_index", 0)),
                "final_total_effective_days": int(best_plan.get("summary", {}).get("total_effective_days", 0)),
                "final_total_effective_days_unbounded": summary_block.get("total_effective_days_unbounded"),
                "over_45_by_days": summary_block.get("over_45_by_days"),
                "middles_used": middles_used,
                "dropped": packing_meta.get("dropped", []),
                "adjustments": adjustments,
                "raw_closer_index": packing_meta.get("raw_closer_index"),
                "capped_to_index": packing_meta.get("capped_to_index"),
                "window_adjusted": packing_meta.get("window_adjusted"),
                "over_cap_raw": packing_meta.get("over_cap_raw"),
                "over_cap_final": packing_meta.get("over_cap_final"),
            }
        )

    # Build inventory_selected from the best plan's final sequence_debug
    # This ensures consistency: planned_submit_index/date and effective_contribution_days
    # must match the values in sequence_debug exactly
    inventory_header = _build_inventory_header_from_sequence(
        sequence=best_sequence,
        inventory_base_map=inventory_base_map,
        inventory_all=inventory_all,
        closer_field=closer_field,
        bureau_value=bureau_value,
        dedup_notes=dedup_notes,
    )

    used_fields = {entry["field"] for entry in inventory_header["inventory_selected"]}
    booster_candidates = _collect_booster_candidates(prepared_items, used_fields)
    field_info_map = {str(item.get("field")): dict(item) for item in prepared_items if item.get("field")}

    inventory_boosters: List[BoosterHeader] = []
    sequence_boosters: List[BoosterStep] = []
    if boosters_enabled and booster_candidates and best_sequence:
        inventory_boosters, sequence_boosters = _build_sequence_boosters(
            best_plan,
            candidates=booster_candidates,
            field_info_map=field_info_map,
            weekend=weekend_set,
            holidays=holidays_set,
            no_weekend_submit=no_weekend_submit,
        )

    role_summary_ts = _utc_now_iso()
    opener_count = sum(1 for entry in inventory_all if entry.get("role_guess") == "opener")
    supporter_count = sum(1 for entry in inventory_all if entry.get("role_guess") == "supporter")
    schedule_logs.append(
        {
            "event": "role_assignment_summary",
            "opener": opener_field_str,
            "closer": closer_field_str,
            "openers_eligible": openers_eligible_count,
            "openers_marked": opener_count,
            "closers_eligible": closers_eligible_count,
            "supporters_eligible": supporter_count,
            "domain_tiebreak_applied": domain_tiebreak_applied,
            "reason": selection_reason,
            "source": "role_selector",
            "ts": role_summary_ts,
        }
    )

    for idx, plan in weekday_plans.items():
        plan["inventory_boosters"] = deepcopy(inventory_boosters)
        if idx == best_weekday:
            plan["sequence_boosters"] = deepcopy(sequence_boosters)
        else:
            plan["sequence_boosters"] = []

    # Build inventory_header per weekday and optionally apply inbound-cap to each weekday independently
    for weekday_idx, plan in list(weekday_plans.items()):
        plan_sequence = plan.get("sequence_debug", [])
        plan_inventory_header = _build_inventory_header_from_sequence(
            sequence=plan_sequence,
            inventory_base_map=inventory_base_map,
            inventory_all=inventory_all,
            closer_field=closer_field,
            bureau_value=bureau_value,
            dedup_notes=dedup_notes,
        )
        plan_with_header = _with_inventory_header(plan, plan_inventory_header)

        # When best-weekday is disabled, apply inbound-cap to every weekday plan
        if not best_weekday_enabled and not enforce_span_cap:
            ub_val = plan_with_header.get("summary", {}).get("total_effective_days_unbounded")
            try:
                needs_cap = ub_val is not None and int(ub_val) > 50
            except (TypeError, ValueError):
                needs_cap = False
            if needs_cap:
                optimized = optimize_overlap_for_inbound_cap(
                    deepcopy(plan_with_header),
                    max_unbounded_inbound_day=50,
                    weekend=weekend_set,
                    holidays=holidays_set,
                    enforce_span_cap=enforce_span_cap,
                    include_notes=include_notes,
                )
                # Rebuild inventory header after optimization
                opt_seq = optimized.get("sequence_debug", [])
                opt_header = _build_inventory_header_from_sequence(
                    sequence=opt_seq,
                    inventory_base_map=inventory_base_map,
                    inventory_all=inventory_all,
                    closer_field=closer_field,
                    bureau_value=bureau_value,
                    dedup_notes=dedup_notes,
                )
                plan_with_header = _with_inventory_header(optimized, opt_header)
                # No extra schedule logs for cap events

        weekday_plans[weekday_idx] = plan_with_header

    best_plan = weekday_plans.get(best_weekday, best_plan)
    best_sequence = best_plan.get("sequence_debug", [])
    # Rebuild inventory header from potentially optimized best_sequence to ensure sync
    inventory_header = _build_inventory_header_from_sequence(
        sequence=best_sequence,
        inventory_base_map=inventory_base_map,
        inventory_all=inventory_all,
        closer_field=closer_field,
        bureau_value=bureau_value,
        dedup_notes=dedup_notes,
    )
    canonical_order = [entry["field"] for entry in best_sequence]

    best_overall: BestOverall = {
        "start_weekday": best_weekday,
        "calendar_span_days": int(best_plan.get("calendar_span_days", 0)),
        "order": canonical_order,
        "last_calendar_day_index": int(best_plan.get("last_calendar_day_index", 0)),
    }

    master_summary = {
        "first_submit": best_sequence[0]["calendar_day_index"] if best_sequence else 0,
        "last_submit": best_overall["last_calendar_day_index"],
        "last_submit_in_window": bool(best_sequence)
        and last_submit_window[0] <= best_overall["last_calendar_day_index"] <= last_submit_window[1]
        and (best_sequence[-1]["submit_on"]["weekday"] not in weekend_set),
        "total_items": len(best_sequence),
    }

    master_plan: MasterPlan = {
        "schema_version": 1,
        "generated_at": _utc_now_iso(),
        "timezone": tz.key,
        "mode_used": mode,
        "weekend": weekend_list,
        "holidays": holidays_list,
        "best_overall": best_overall,
        "inventory_header": deepcopy(inventory_header),
        "by_weekday": {str(idx): weekday_plans[idx] for idx in range(5)},
        "meta": {
            "openers_count": len(openers),
            "supporters_count": len(middle),
            "closers_count": len(closers),
        },
        "calendar_span_days": best_overall["calendar_span_days"],
        "last_calendar_day_index": best_overall["last_calendar_day_index"],
        "summary": deepcopy(best_plan.get("summary", master_summary)),
        "constraints": dict(constraints),
        "skipped": skipped_entries,
    }

    if bureau_value:
        master_plan["bureau"] = bureau_value

    master_plan["inventory_boosters"] = deepcopy(inventory_boosters)
    master_plan["sequence_boosters"] = deepcopy(sequence_boosters)

    if not weekday_meta.get(best_weekday, {}).get("success"):
        master_plan["reason"] = "planner_impossible_window"

    meta_for_best = weekday_meta.get(best_weekday, {})
    final_day_index = best_overall["last_calendar_day_index"]
    attempted_day = meta_for_best.get("attempted_index")
    window_reason = meta_for_best.get("window_reason")
    target_within_window = bool(meta_for_best.get("target_within_window", True))

    if best_sequence:
        closer_schedule_ts = _utc_now_iso()
        schedule_logs.append(
            {
                "event": "closer_scheduled",
                "field": closer_field,
                "target_window": list(last_submit_window),
                "scheduled_day": final_day_index,
                "weekday": best_sequence[-1]["submit_on"].get("weekday_name"),
                "ts": closer_schedule_ts,
            }
        )

        if not target_within_window:
            attempted_value = attempted_day if attempted_day is not None else last_submit_window[1]
            warning_ts = _utc_now_iso()
            reason_text = window_reason or "window_unavailable"
            schedule_logs.append(
                {
                    "event": "timeline_warning",
                    "target_window": list(last_submit_window),
                    "attempted_day": attempted_value,
                    "reason": reason_text,
                    "ts": warning_ts,
                }
            )
            schedule_logs.append(
                {
                    "event": "closer_window_adjust",
                    "reason": reason_text,
                    "attempted_day": attempted_value,
                    "final_day": final_day_index,
                    "ts": warning_ts,
                }
            )

    # Drop bounded timeline event emission per simplification requirements
    summary_block = master_plan.get("summary", {})
    timeline_logs: List[Dict[str, object]] = []
    schedule_logs.extend(selection_logs)
    schedule_logs.append(
        {
            "event": "inventory_snapshot",
            "all": len(inventory_all),
            "selected": len(inventory_header["inventory_selected"]),
            "dedup_dropped": dedup_dropped,
            "ts": _utc_now_iso(),
        }
    )
    # Do not append removed timeline logs

    if boosters_enabled:
        schedule_logs.append(
            {
                "event": "planner_boosters_summary",
                "mode": mode,
                "boosters_count": len(inventory_boosters),
                "boosters_fields": [entry["field"] for entry in inventory_boosters],
            }
        )

    if account_id:
        for event in schedule_logs:
            event.setdefault("account", account_id)
    if bureau_value:
        for event in schedule_logs:
            event.setdefault("bureau", bureau_value)

    # Apply overlap redistribution on full internal representation (with sequence_debug + cap metadata)
    # BEFORE simplify_plan_for_public strips internal fields
    try:
        _rebalance_overlap_distribution(master_plan, weekend=weekend)
        for w_idx, w_plan in list(weekday_plans.items()):
            _rebalance_overlap_distribution(w_plan, weekend=weekend)
    except Exception as e:
        import traceback
        print(f"DEBUG: Rebalance error: {e}")
        traceback.print_exc()
        pass

    # NOW simplify for public output (after rebalance has seen full state)
    for w_idx, w_plan in list(weekday_plans.items()):
        # No deepcopy - simplify the already-rebalanced plan in-place
        weekday_plans[w_idx] = simplify_plan_for_public(w_plan)
    master_plan["by_weekday"] = {str(idx): weekday_plans[idx] for idx in range(5)}
    master_plan = simplify_plan_for_public(master_plan)
    return {
        "master": master_plan,
        "weekday_plans": weekday_plans,
        "schedule_logs": schedule_logs,
        "best_weekday": best_weekday,
    }
