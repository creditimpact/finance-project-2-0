import os
import pytest
from backend.strategy.planner import compute_optimal_plan
from backend.strategy.types import Finding


def _build_basic_findings():
    # opener + closer + two leftovers to enable enrichment
    return [
        Finding(
            field="payment_status",
            min_days=19,
            duration_unit="business_days",
            default_decision="strong_actionable",
            category="status",
        ),
        Finding(
            field="seven_year_history",
            min_days=19,
            duration_unit="business_days",
            default_decision="strong_actionable",
            category="history",
        ),
        Finding(
            field="date_reported",
            min_days=10,
            duration_unit="business_days",
            default_decision="supportive_needs_companion",
            category="status",
        ),
        Finding(
            field="last_payment",
            min_days=12,
            duration_unit="business_days",
            default_decision="supportive_needs_companion",
            category="status",
        ),
    ]


def _set_env_s2(enable_day40=True):
    os.environ["PLANNER_ENABLE_SKELETON2"] = "1"
    os.environ["PLANNER_SKELETON2_ENABLE_DAY40_STRONGEST"] = "1" if enable_day40 else "0"
    os.environ["PLANNER_SKELETON2_MAX_ITEMS_PER_HANDOFF"] = "1"
    os.environ["PLANNER_SKELETON2_MIN_SLA_DAYS"] = "5"
    os.environ["PLANNER_SKELETON2_ENFORCE_CADENCE"] = "0"


def _allowed_keys_for(reason: str):
    # Consumer-facing compact output: dates + timeline, no debug/placement fields
    base = {
        "idx", "field", "planned_submit_index",
        "submit_date", "submit_weekday", "unbounded_end_date", "unbounded_end_weekday",
        "business_sla_days", "timeline_unbounded"
    }
    if reason.startswith("handoff_"):
        base |= {"between_skeleton1_indices", "handoff_reference_day", "half_sla_offset"}
    return base


def test_compact_enrichment_schema_includes_dates_and_excludes_debug():
    _set_env_s2()
    # Force compact mode
    os.environ["PLANNER_OUTPUT_MODE"] = "compact"

    plan = compute_optimal_plan(
        _build_basic_findings(),
        mode="per_bureau_joint_optimize",
        weekend={5, 6},
        holidays=set(),
        timezone_name="America/New_York",
        enforce_span_cap=False,
        handoff_min_business_days=1,
        handoff_max_business_days=3,
        bureau="equifax",
        output_mode="compact",
    )

    # Check all weekday plans for allowed keys and required dates
    for wd in range(5):
        wd_plan = plan["weekday_plans"][wd]
        enrich = wd_plan.get("enrichment_sequence", [])
        for item in enrich:
            # Required consumer-facing fields
            assert "submit_date" in item, "Missing submit_date in compact output"
            assert "submit_weekday" in item, "Missing submit_weekday in compact output"
            assert "unbounded_end_date" in item, "Missing unbounded_end_date in compact output"
            assert "unbounded_end_weekday" in item, "Missing unbounded_end_weekday in compact output"
            assert "timeline_unbounded" in item, "Missing timeline_unbounded in compact output"
            
            # Ensure debug/metadata keys are NOT present
            for disallowed in [
                "placement_reason", "calendar_day_index", "day40_target_index", "day40_adjustment_reason",
                "submit_on", "sla_window", "min_days", "strength_value",
                "role", "decision", "category", "pre_closer_field", "pre_closer_unbounded_end",
            ]:
                assert disallowed not in item, f"{disallowed} must not appear in compact output"
            
            # Validate allowed keys: base + optional handoff keys if they exist
            base = {"idx", "field", "planned_submit_index", "submit_date", "submit_weekday",
                    "unbounded_end_date", "unbounded_end_weekday", "business_sla_days", "timeline_unbounded"}
            optional_handoff = {"between_skeleton1_indices", "handoff_reference_day", "half_sla_offset"}
            allowed = base | (optional_handoff if any(k in item for k in optional_handoff) else set())
            assert set(item.keys()).issubset(allowed), f"Unexpected keys: {set(item.keys()) - allowed}"


def test_verbose_mode_keeps_debug_and_compact_output():
    _set_env_s2()
    os.environ["PLANNER_OUTPUT_MODE"] = "verbose"

    plan = compute_optimal_plan(
        _build_basic_findings(),
        mode="per_bureau_joint_optimize",
        weekend={5, 6},
        holidays=set(),
        timezone_name="America/New_York",
        enforce_span_cap=False,
        handoff_min_business_days=1,
        handoff_max_business_days=3,
        bureau="experian",
        output_mode="verbose",
    )

    # In verbose mode: enrichment_sequence is compact (with dates, no debug), enrichment_debug has full items
    for wd in range(5):
        wd_plan = plan["weekday_plans"][wd]
        enrich_compact = wd_plan.get("enrichment_sequence", [])
        enrich_debug = wd_plan.get("enrichment_debug", {}).get("sequence")
        
        # Compact is still compact (dates + timeline only, no debug fields)
        for item in enrich_compact:
            assert "submit_date" in item, "Compact must include submit_date"
            assert "placement_reason" not in item, "Compact must NOT include placement_reason"
            assert "day40_target_index" not in item, "Compact must NOT include day40_target_index"
        
        # Debug sequence must exist if any enrichment exists
        if enrich_compact:
            assert isinstance(enrich_debug, list) and enrich_debug, "enrichment_debug.sequence must exist in verbose mode"
            # At least one debug item should include rich debug fields like submit_on
            assert any("submit_on" in d for d in enrich_debug), "verbose debug should include rich keys like submit_on"
