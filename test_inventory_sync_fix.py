"""Test to verify inventory_selected synchronization with sequence_debug."""

from datetime import datetime
from zoneinfo import ZoneInfo
from backend.strategy.planner import compute_optimal_plan
from backend.strategy.types import Finding


def test_real_world_inventory_sync():
    """Verify inventory_selected matches sequence_debug across all weekday plans."""
    
    # Create realistic test findings
    findings = [
        Finding(
            field="payment_status",
            category="status",
            min_days=19,
            duration_unit="business_days",
            default_decision="strong_actionable",
        ),
        Finding(
            field="account_status",
            category="status",
            min_days=10,
            duration_unit="business_days",
            default_decision="strong_actionable",
        ),
        Finding(
            field="date_of_last_activity",
            category="status",
            min_days=10,
            duration_unit="business_days",
            default_decision="strong_actionable",
        ),
    ]

    run_dt = datetime(2025, 11, 20, 9, 0, tzinfo=ZoneInfo("America/New_York"))
    plan = compute_optimal_plan(
        findings,
        weekend={5, 6},
        holidays=None,
        timezone_name="America/New_York",
        run_datetime=run_dt,
        max_calendar_span=60,
        last_submit_window=(37, 40),
        include_supporters=True,
        exclude_natural_text=True,
        strength_metric="score",
        handoff_min_business_days=1,
        handoff_max_business_days=3,
        enforce_span_cap=False,
        dedup_by="field",
    )

    # Check master plan
    master_header = plan["master"]["inventory_header"]
    master_selected = master_header["inventory_selected"]
    
    best_weekday = plan["best_weekday"]
    best_plan = plan["weekday_plans"][best_weekday]
    best_sequence = best_plan["sequence_debug"]
    
    # Master inventory should match best plan's sequence
    assert len(master_selected) == len(best_sequence), "Master inventory count mismatch"
    
    for inv_entry, seq_entry in zip(master_selected, best_sequence):
        assert inv_entry["field"] == seq_entry["field"], f"Field mismatch: {inv_entry['field']} != {seq_entry['field']}"
        assert inv_entry["planned_submit_index"] == seq_entry["calendar_day_index"], \
            f"Index mismatch for {inv_entry['field']}: {inv_entry['planned_submit_index']} != {seq_entry['calendar_day_index']}"
        assert inv_entry["planned_submit_date"] == seq_entry["submit"]["date"], \
            f"Date mismatch for {inv_entry['field']}: {inv_entry['planned_submit_date']} != {seq_entry['submit']['date']}"
        assert inv_entry["effective_contribution_days"] == seq_entry["effective_contribution_days"], \
            f"Effective days mismatch for {inv_entry['field']}: {inv_entry['effective_contribution_days']} != {seq_entry['effective_contribution_days']}"
        assert inv_entry["running_total_after"] == seq_entry["running_total_days_after"], \
            f"Running total mismatch for {inv_entry['field']}: {inv_entry['running_total_after']} != {seq_entry['running_total_days_after']}"
    
    print(f"✅ Master inventory_selected synced with best plan (weekday {best_weekday})")
    
    # Check each weekday plan
    for weekday_idx, weekday_plan in plan["weekday_plans"].items():
        weekday_header = weekday_plan["inventory_header"]
        weekday_selected = weekday_header["inventory_selected"]
        weekday_sequence = weekday_plan["sequence_debug"]
        
        assert len(weekday_selected) == len(weekday_sequence), \
            f"Weekday {weekday_idx}: inventory count {len(weekday_selected)} != sequence count {len(weekday_sequence)}"
        
        for inv_entry, seq_entry in zip(weekday_selected, weekday_sequence):
            assert inv_entry["field"] == seq_entry["field"]
            assert inv_entry["planned_submit_index"] == seq_entry["calendar_day_index"], \
                f"WD{weekday_idx} {inv_entry['field']}: index {inv_entry['planned_submit_index']} != {seq_entry['calendar_day_index']}"
            assert inv_entry["planned_submit_date"] == seq_entry["submit"]["date"], \
                f"WD{weekday_idx} {inv_entry['field']}: date {inv_entry['planned_submit_date']} != {seq_entry['submit']['date']}"
            assert inv_entry["effective_contribution_days"] == seq_entry["effective_contribution_days"], \
                f"WD{weekday_idx} {inv_entry['field']}: eff {inv_entry['effective_contribution_days']} != {seq_entry['effective_contribution_days']}"
        
        print(f"✅ Weekday {weekday_idx} inventory_selected synced with its own sequence")
    
    # Check constraints reflect new semantics
    constraints = plan["master"]["constraints"]
    assert constraints["last_submit_window"] == [0, 40], f"Expected [0, 40], got {constraints['last_submit_window']}"
    assert constraints["max_last_submit_day"] == 40, f"Expected max_last_submit_day=40, got {constraints.get('max_last_submit_day')}"
    print(f"✅ Constraints reflect new ≤40 deadline semantics")
    
    # Check legacy fields removed
    for weekday_plan in plan["weekday_plans"].values():
        for seq_entry in weekday_plan["sequence_debug"]:
            assert "remaining_to_last_window_start" not in seq_entry, "Legacy field still present!"
            assert "remaining_to_last_window_end" not in seq_entry, "Legacy field still present!"
    print(f"✅ Legacy window fields removed from all sequences")
    
    print("\n🎉 All synchronization checks passed!")


if __name__ == "__main__":
    test_real_world_inventory_sync()
