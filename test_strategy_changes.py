"""
Test script to verify the strategy planner changes:
1. last_submit_window changed from [37, 40] to [0, 40]
2. Closer selection prioritizes max business_sla_days
3. Opener selection prefers strong_actionable with days <= closer
"""
from datetime import datetime
from zoneinfo import ZoneInfo

from backend.strategy.planner import _select_findings_varlen, _normalized_decision


def test_new_selection_logic():
    """Test that the new selection logic works as expected."""
    
    # Create mock items with different SLA days and decisions
    items = [
        {
            "field": "payment_status",
            "min_days": 19,
            "default_decision": "strong_actionable",
            "decision": "strong_actionable",
            "strength_value": 20,
            "strength_metric": "score",
            "order_index": 1,
            "role": "opener",
            "category": "status",
            "score": {"base": 19, "bonuses": {}, "total": 19},
            "placement": "second_strongest_first",
            "why_here": "Top-scoring opener",
        },
        {
            "field": "account_status",
            "min_days": 10,
            "default_decision": "strong_actionable",
            "decision": "strong_actionable",
            "strength_value": 12,
            "strength_metric": "score",
            "order_index": 2,
            "role": "supporter",
            "category": "status",
            "score": {"base": 10, "bonuses": {}, "total": 10},
            "placement": "second_strongest_first",
            "why_here": "Supporter",
        },
        {
            "field": "date_of_last_activity",
            "min_days": 10,
            "default_decision": "supportive_needs_companion",
            "decision": "supportive_needs_companion",
            "strength_value": 12,
            "strength_metric": "score",
            "order_index": 3,
            "role": "supporter",
            "category": "activity",
            "score": {"base": 10, "bonuses": {}, "total": 10},
            "placement": "second_strongest_first",
            "why_here": "Supporter",
        },
        {
            "field": "payment_amount",
            "min_days": 5,
            "default_decision": "strong_actionable",
            "decision": "strong_actionable",
            "strength_value": 6,
            "strength_metric": "score",
            "order_index": 4,
            "role": "supporter",
            "category": "terms",
            "score": {"base": 5, "bonuses": {}, "total": 5},
            "placement": "second_strongest_first",
            "why_here": "Supporter",
        },
    ]
    
    run_dt = datetime(2025, 11, 20, 9, 0, tzinfo=ZoneInfo("America/New_York"))
    
    try:
        chosen_sequence, logs, dedup_notes, dedup_dropped, role_meta = _select_findings_varlen(
            items=items,
            run_dt=run_dt,
            tz=ZoneInfo("America/New_York"),
            weekend={5, 6},
            holidays=set(),
            target_window=(0, 40),  # Changed from (37, 40)
            max_span=45,
            enforce_span_cap=False,
            handoff_min=1,
            handoff_max=3,
            target_effective_days=45,
            min_increment_days=1,
            dedup_by="decision",
            include_supporters=True,
            include_notes=False,
        )
        
        print("✅ Selection completed successfully!")
        print(f"\nChosen sequence ({len(chosen_sequence)} items):")
        for item in chosen_sequence:
            print(f"  - {item['field']}: {item['min_days']} days, decision={item.get('decision', item.get('default_decision'))}")
        
        print(f"\nRole metadata:")
        print(f"  Opener: {role_meta['opener_field']}")
        print(f"  Closer: {role_meta['closer_field']}")
        print(f"  Selection reason: {role_meta['reason']}")
        print(f"  Openers eligible: {role_meta['openers_eligible']}")
        print(f"  Closers eligible: {role_meta['closers_eligible']}")
        
        # Verify expectations
        opener_item = next(item for item in chosen_sequence if item['field'] == role_meta['opener_field'])
        closer_item = next(item for item in chosen_sequence if item['field'] == role_meta['closer_field'])
        
        # Closer should be payment_status (19 days, highest SLA)
        assert role_meta['closer_field'] == 'payment_status', f"Expected closer to be payment_status, got {role_meta['closer_field']}"
        assert closer_item['min_days'] == 19, f"Expected closer to have 19 days, got {closer_item['min_days']}"
        
        # Opener should be strong_actionable with days <= closer (payment_status has 19 days, so opener should be <= 19)
        opener_decision = _normalized_decision(str(opener_item.get('default_decision', opener_item.get('decision', ''))))
        assert opener_decision == 'strong_actionable', f"Expected opener to be strong_actionable, got {opener_decision}"
        assert opener_item['min_days'] <= 19, f"Expected opener days <= 19, got {opener_item['min_days']}"
        
        print("\n✅ All assertions passed!")
        print("\nExpected behavior verified:")
        print("  ✓ Closer selected based on max business_sla_days (19 days)")
        print("  ✓ Opener selected from strong_actionable items")
        print("  ✓ Opener has days <= closer")
        print("  ✓ Target window is [0, 40] (no lower bound enforcement)")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_last_submit_window_constraint():
    """Test that last_submit_in_window now only checks <= 40."""
    from backend.strategy.planner import _enrich_sequence_with_contributions
    from backend.strategy.calendar import advance_business_days_date
    
    anchor_date = datetime(2025, 11, 20).date()
    
    # Create a mock sequence with last item at day 35 (previously would fail if outside 37-40)
    sequence = [
        {
            "calendar_day_index": 0,
            "submit_on": {"date": "2025-11-20", "weekday": 3},
            "sla_window": {"start": {"date": "2025-11-20"}, "end": {"date": "2025-12-17"}},
            "min_days": 19,
        },
        {
            "calendar_day_index": 35,  # Previously this would fail last_window check if window was [37,40]
            "submit_on": {"date": "2025-12-25", "weekday": 3},
            "sla_window": {"start": {"date": "2025-12-25"}, "end": {"date": "2026-01-10"}},
            "min_days": 10,
        },
    ]
    
    submit_history = [datetime(2025, 11, 20).date(), datetime(2025, 12, 25).date()]
    sla_history = [datetime(2025, 12, 17).date(), datetime(2026, 1, 10).date()]
    
    try:
        total_effective, total_unbounded, summary = _enrich_sequence_with_contributions(
            sequence,
            submit_history=submit_history,
            sla_history=sla_history,
            anchor_date=anchor_date,
            last_window=(0, 40),  # New window
            weekend={5, 6},
            enforce_span_cap=False,
            include_notes=False,
        )
        
        print("\n✅ last_submit_window constraint test passed!")
        print(f"  Last submit: day {summary['last_submit']}")
        print(f"  last_submit_in_window: {summary['last_submit_in_window']}")
        
        # Should be True because 35 <= 40 (no lower bound check)
        assert summary['last_submit_in_window'] == True, \
            f"Expected last_submit_in_window=True for day 35, got {summary['last_submit_in_window']}"
        
        print("  ✓ Day 35 submission is valid (previously would fail [37,40] window)")
        
        return True
        
    except Exception as e:
        print(f"❌ last_submit_window test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("=" * 80)
    print("Testing Strategy Planner Changes")
    print("=" * 80)
    
    print("\n" + "=" * 80)
    print("Test 1: New Selection Logic")
    print("=" * 80)
    test1_passed = test_new_selection_logic()
    
    print("\n" + "=" * 80)
    print("Test 2: last_submit_window Constraint")
    print("=" * 80)
    test2_passed = test_last_submit_window_constraint()
    
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    if test1_passed and test2_passed:
        print("✅ All tests passed!")
        exit(0)
    else:
        print("❌ Some tests failed")
        exit(1)
