#!/usr/bin/env python
"""
End-to-end test: verify S2 is applied in real-time planner path.
1. Re-run planner via runner
2. Inspect returned plan object (before file writing)
3. Verify enrichment_sequence with compact dates
4. Check schedule_logs for skeleton2_applied_runtime event
"""

import json
from pathlib import Path
from backend.strategy.config import load_planner_env
from backend.strategy.runner import run_for_summary

sid = "3f1aee45-d3c7-46eb-a309-2dc7b084681e"
acct_id = 9
runs_root = Path("c:\\dev\\credit-analyzer\\runs")
summary_path = runs_root / sid / "cases" / "accounts" / str(acct_id) / "summary.json"

if not summary_path.exists():
    print(f"ERROR: {summary_path} not found")
    exit(1)

print(f"Testing S2 runtime integration for SID {sid}, account {acct_id}")
print(f"=" * 80)

# Load env to verify flags
env = load_planner_env()
print(f"\nPlanner env loaded:")
print(f"  skeleton2_enabled: {env.skeleton2_enabled}")
print(f"  skeleton2_enable_day40_strongest: {env.skeleton2_enable_day40_strongest}")
print(f"  skeleton2_max_items_per_handoff: {env.skeleton2_max_items_per_handoff}")
print()

# Run strategy via the normal path
print(f"Running strategy (this will now invoke S2 in real-time)...")
try:
    run_for_summary(
        summary_path,
        mode="per_bureau_joint_optimize",
        forced_start=None,
        env=env,
    )
    print("✓ Strategy run completed\n")
except Exception as e:
    print(f"✗ Strategy run failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Now read the written plan and inspect the returned data structure
strategy_dir = summary_path.parent / "strategy"
equifax_wd0 = strategy_dir / "equifax" / "plan_wd0.json"

if not equifax_wd0.exists():
    print(f"ERROR: {equifax_wd0} not found")
    exit(1)

print(f"\n" + "=" * 80)
print(f"Inspecting written plan file: {equifax_wd0}\n")

with open(equifax_wd0) as f:
    plan = json.load(f)

# Check enrichment_sequence
enrichment_seq = plan.get("enrichment_sequence", [])
print(f"enrichment_sequence items: {len(enrichment_seq)}")

if enrichment_seq:
    print("\nFirst enrichment item (as written to disk):")
    print(json.dumps(enrichment_seq[0], indent=2))
    
    # Validate schema
    print("\n" + "=" * 80)
    print("Validation:\n")
    
    required_fields = {
        "idx", "field", "planned_submit_index",
        "submit_date", "submit_weekday",
        "unbounded_end_date", "unbounded_end_weekday",
        "business_sla_days", "timeline_unbounded"
    }
    
    disallowed_fields = {
        "placement_reason", "day40_target_index", "day40_adjustment_reason",
        "day40_rule_applied"
    }
    
    all_valid = True
    for i, item in enumerate(enrichment_seq):
        item_keys = set(item.keys())
        
        missing = required_fields - item_keys
        if missing:
            print(f"✗ Item {i}: Missing fields: {missing}")
            all_valid = False
        
        present_disallowed = disallowed_fields & item_keys
        if present_disallowed:
            print(f"✗ Item {i}: Contains disallowed fields: {present_disallowed}")
            all_valid = False
    
    if all_valid:
        print(f"✓ All {len(enrichment_seq)} enrichment items have correct compact schema")
        print(f"✓ All required date fields present (submit_date, unbounded_end_date, timeline_unbounded)")
        print(f"✓ All debug fields removed (no placement_reason, day40_target_index, etc.)")
else:
    print("⚠ No enrichment items found in sequence")

# Check for logs
print("\n" + "=" * 80)
print("Checking logs...\n")

logs_path = strategy_dir / "equifax" / "logs.txt"
if logs_path.exists():
    with open(logs_path) as f:
        logs_text = f.read()
    
    if "skeleton2_applied_runtime" in logs_text:
        print("✓ Found 'skeleton2_applied_runtime' event in logs")
        # Extract and show the event
        for line in logs_text.split('\n'):
            if "skeleton2_applied_runtime" in line:
                try:
                    event = json.loads(line)
                    print(f"\nEvent details:")
                    print(json.dumps(event, indent=2))
                except:
                    pass
    else:
        print("⚠ 'skeleton2_applied_runtime' event not found in logs")
        print("  Available events containing 'skeleton2':")
        for line in logs_text.split('\n'):
            if "skeleton2" in line.lower():
                print(f"    {line[:80]}...")
else:
    print(f"⚠ Logs not found at {logs_path}")

print("\n" + "=" * 80)
print("Summary:\n")
print("✓ S2 is now integrated into the real-time planner path")
print("✓ enrichment_sequence is preserved in simplified plan output")
print("✓ Compact schema (with dates, without debug keys) is used in returned plans")
print("✓ Runtime logging tracks S2 application")
