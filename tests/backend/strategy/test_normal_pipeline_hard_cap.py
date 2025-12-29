"""
Regression test: Verify hard-cap + overlap works through the normal pipeline.
This simulates the exact code path used by `python -m backend.strategy.runner --sid <sid>`.
"""
import os
import pytest
from backend.strategy.planner import compute_optimal_plan
from backend.strategy.types import Finding


def test_normal_pipeline_applies_hard_cap_and_overlap():
    """
    Simulates SID 88c4ee20: opener 26 days + closer 27 days → 53 unbounded.
    
    Uses the NORMAL pipeline code path (without explicitly setting STRATEGY_BEST_WEEKDAY_ENABLED).
    The default must be per-weekday mode so optimizer runs.
    
    Asserts:
    - total_effective_days_unbounded ≤ 50
    - inbound_cap_* metadata present
    - Overlap fields populated and identity holds
    """
    # Two 19-business-day findings: opener + closer
    findings = [
        Finding(
            field="payment_status",
            min_days=19,
            duration_unit="business_days",
            default_decision="strong_actionable",
            reason_code="R1",
            category="status",
            documents=["doc1"],
            is_missing=False,
            is_mismatch=True,
            bureau_dispute_state={"experian": "conflict"},
        ),
        Finding(
            field="seven_year_history",
            min_days=19,
            duration_unit="business_days",
            default_decision="strong_actionable",
            reason_code="R2",
            category="history",
            documents=["doc2", "doc3"],
            is_missing=False,
            is_mismatch=True,
            bureau_dispute_state={"experian": "conflict"},
        ),
    ]

    # Temporarily unset STRATEGY_BEST_WEEKDAY_ENABLED to simulate "not explicitly set" case
    original_val = os.getenv("STRATEGY_BEST_WEEKDAY_ENABLED")
    try:
        # Remove env var to test default behavior
        if "STRATEGY_BEST_WEEKDAY_ENABLED" in os.environ:
            del os.environ["STRATEGY_BEST_WEEKDAY_ENABLED"]
        
        result = compute_optimal_plan(
            findings,
            mode="per_bureau_joint_optimize",
            weekend={5, 6},
            holidays=set(),
            timezone_name="America/New_York",
            enforce_span_cap=False,  # Must be False for optimizer to run
            handoff_min_business_days=1,
            handoff_max_business_days=3,
            bureau="experian",
        )
    finally:
        # Restore original value
        if original_val is None:
            os.environ.pop("STRATEGY_BEST_WEEKDAY_ENABLED", None)
        else:
            os.environ["STRATEGY_BEST_WEEKDAY_ENABLED"] = original_val

    # Check every weekday plan
    for weekday_idx in range(5):
        plan = result["weekday_plans"][weekday_idx]
        summary = plan["summary"]
        
        # Hard cap must be applied
        unbounded = summary.get("total_effective_days_unbounded")
        assert unbounded is not None, f"wd{weekday_idx}: unbounded must be present"
        assert unbounded <= 50, f"wd{weekday_idx}: unbounded must be ≤50, got {unbounded}"
        
        # Metadata must be present if cap was applied
        if unbounded < 53:  # If it was capped from original 53
            assert summary.get("inbound_cap_hard") is True, f"wd{weekday_idx}: inbound_cap_hard missing"
            assert summary.get("inbound_cap_applied") is True, f"wd{weekday_idx}: inbound_cap_applied missing"
            assert summary.get("inbound_cap_before", 0) >= 51, f"wd{weekday_idx}: inbound_cap_before should be ≥51"
            assert summary.get("inbound_cap_after") == unbounded, f"wd{weekday_idx}: inbound_cap_after mismatch"
        
        # Overlap fields must be present
        sequence_debug = plan.get("sequence_debug", [])
        if len(sequence_debug) >= 2:
            closer = sequence_debug[1]
            assert "overlap_raw_days_with_prev" in closer, f"wd{weekday_idx}: overlap_raw_days_with_prev missing"
            assert "overlap_effective_unbounded_with_prev" in closer, f"wd{weekday_idx}: overlap_effective_unbounded_with_prev missing"
            
            # Check identity: total_unbounded = sum(item unbounded) - total_overlap
            total_overlap = int(summary.get("total_overlap_unbounded_days", 0))
            sum_unbounded = sum(int(e.get("effective_contribution_days_unbounded", 0)) for e in sequence_debug)
            identity_lhs = int(unbounded)
            identity_rhs = sum_unbounded - total_overlap
            assert identity_lhs == identity_rhs, (
                f"wd{weekday_idx}: Identity broken: {identity_lhs} != {sum_unbounded} - {total_overlap}"
            )
        
        # Sequence compact must have overlap field
        sequence_compact = plan.get("sequence_compact", [])
        if len(sequence_compact) >= 2:
            assert "overlap_days_with_prev" in sequence_compact[1], f"wd{weekday_idx}: overlap_days_with_prev missing in compact"
        
        # Inventory selected must have overlap field for 2nd+ items
        inventory_selected = plan.get("inventory_header", {}).get("inventory_selected", [])
        if len(inventory_selected) >= 2:
            assert "overlap_days_with_prev" in inventory_selected[1], f"wd{weekday_idx}: overlap_days_with_prev missing in inventory"

    print("✅ Normal pipeline (default behavior) applies hard-cap + overlap correctly")
