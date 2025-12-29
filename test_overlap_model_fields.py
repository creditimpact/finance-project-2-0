"""Test overlap-based inbound cap model with all new fields."""
import pytest
from backend.strategy.planner import compute_optimal_plan
from backend.strategy.types import Finding


def test_overlap_based_model_fields():
    """Verify overlap-based model adds all required fields and enforces invariants."""
    
    # Create a scenario that will trigger the hard cap (total > 50)
    findings = [
        Finding(
            field="payment_history",
            category="mixed",
            decision="dispute_primary",
            strength_value=80,
            min_days=24,  # Will contribute ~24 unbounded
            raw_text_value="Late payment",
            order_index=0,
        ),
        Finding(
            field="account_status",
            category="negative",
            decision="dispute_primary",
            strength_value=75,
            min_days=30,  # Will contribute ~30 unbounded
            raw_text_value="Charged off",
            order_index=1,
        ),
    ]
    
    result = compute_optimal_plan(
        findings,
        mode="joint_optimize",
        weekend={6, 0},  # Sat, Sun
        holidays=set(),
        timezone_name="America/New_York",
        max_calendar_span=45,
        last_submit_window=(0, 40),
        no_weekend_submit=True,
        include_supporters=False,
        exclude_natural_text=True,
    )
    
    for bureau_key in ["equifax", "experian", "transunion"]:
        plans = result.get(bureau_key, {})
        for wd_key in ["wd0", "wd1", "wd2", "wd3", "wd4"]:
            plan = plans.get(wd_key)
            if not plan:
                continue
            
            summary = plan.get("summary", {})
            sequence = plan.get("sequence_debug", [])
            
            if not sequence or len(sequence) < 2:
                continue
            
            print(f"\n{bureau_key}/{wd_key}:")
            print(f"  N={len(sequence)}")
            
            # Check new summary fields exist
            assert "inbound_cap_sum_items_unbounded" in summary
            assert "inbound_cap_required_overlap" in summary
            assert "inbound_cap_base_overlap_min" in summary
            assert "_debug_sum_items_unbounded" in summary
            assert "_debug_calculated_inbound" in summary
            assert "_debug_identity_valid" in summary
            
            sum_items = summary["inbound_cap_sum_items_unbounded"]
            total_overlap = summary["total_overlap_unbounded_days"]
            total_unbounded = summary["total_effective_days_unbounded"]
            required_overlap = summary["inbound_cap_required_overlap"]
            base_min = summary["inbound_cap_base_overlap_min"]
            
            print(f"  sum_items={sum_items}")
            print(f"  total_overlap={total_overlap}")
            print(f"  total_unbounded={total_unbounded}")
            print(f"  required_overlap={required_overlap}")
            print(f"  base_min={base_min}")
            
            # Verify base_overlap_min = N - 1
            N = len(sequence)
            assert base_min == N - 1, f"base_overlap_min should be {N-1}, got {base_min}"
            
            # Verify required_overlap = max((sum - 50), (N - 1))
            expected_required = max(sum_items - 50, N - 1)
            assert required_overlap == expected_required, \
                f"required_overlap should be {expected_required}, got {required_overlap}"
            
            # Verify identity: total_unbounded = sum_items - total_overlap
            assert summary["_debug_identity_valid"] is True
            assert sum_items - total_overlap == total_unbounded
            
            # Check per-item fields
            for idx, item in enumerate(sequence):
                # All items should have running_unbounded_at_submit
                assert "running_unbounded_at_submit" in item
                
                if idx == 0:
                    # First item should have running_unbounded_at_submit = 0
                    assert item["running_unbounded_at_submit"] == 0
                else:
                    # Subsequent items should have overlap_unbounded_days_with_prev
                    assert "overlap_unbounded_days_with_prev" in item
                    assert "handoff_days_before_prev_sla_end" in item
                    
                    # CRITICAL INVARIANT: handoff >= 1
                    handoff = item["handoff_days_before_prev_sla_end"]
                    assert handoff >= 1, \
                        f"Item {idx} handoff={handoff}, violates handoff >= 1 invariant"
                    
                    print(f"    [{idx+1}] handoff={handoff} overlap={item['overlap_unbounded_days_with_prev']}")
            
            # Check inventory_selected mirrors overlap
            inv_selected = plan.get("inventory_header", {}).get("inventory_selected", [])
            assert len(inv_selected) == len(sequence)
            
            for idx, inv_item in enumerate(inv_selected):
                if idx > 0:
                    assert "overlap_days_with_prev" in inv_item
            
            # If cap was applied, verify unbounded <= 50
            if summary.get("inbound_cap_applied"):
                assert total_unbounded <= 50, \
                    f"Cap applied but unbounded={total_unbounded} > 50"
            
            print(f"  ✅ All checks passed for {bureau_key}/{wd_key}")


if __name__ == "__main__":
    test_overlap_based_model_fields()
    print("\n✅ ALL TESTS PASSED")
