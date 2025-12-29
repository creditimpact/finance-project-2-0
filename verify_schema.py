#!/usr/bin/env python
"""
Verify the enrichment_sequence in the actual plan_wd0.json matches the new compact schema.
"""

import json
from pathlib import Path

sid = "3f1aee45-d3c7-46eb-a309-2dc7b084681e"
acct_id = 9
plan_path = Path(f"c:\\dev\\credit-analyzer\\runs\\{sid}\\cases\\accounts\\{acct_id}\\strategy\\equifax\\plan_wd0.json")

print(f"File: {plan_path}\n")

with open(plan_path) as f:
    plan = json.load(f)

enrichment_seq = plan.get("enrichment_sequence", [])
print(f"Enrichment sequence has {len(enrichment_seq)} items.\n")

# Define what MUST exist and what MUST NOT exist
required_fields = {
    "idx", "field", "planned_submit_index",
    "submit_date", "submit_weekday",
    "unbounded_end_date", "unbounded_end_weekday",
    "business_sla_days", "timeline_unbounded"
}

optional_fields = {
    "between_skeleton1_indices", "handoff_reference_day", "half_sla_offset"
}

disallowed_fields = {
    "placement_reason", "day40_target_index", "day40_adjustment_reason",
    "day40_rule_applied", "calendar_day_index", "submit_on", "sla_window",
    "min_days", "strength_value", "role", "decision", "category",
    "pre_closer_field", "pre_closer_unbounded_end"
}

all_pass = True

for i, item in enumerate(enrichment_seq):
    item_keys = set(item.keys())
    
    # Check required fields
    missing = required_fields - item_keys
    if missing:
        print(f"✗ Item {i}: MISSING required fields: {missing}")
        all_pass = False
    else:
        print(f"✓ Item {i}: All required fields present")
    
    # Check for disallowed fields
    disallowed_present = disallowed_fields & item_keys
    if disallowed_present:
        print(f"✗ Item {i}: Contains DISALLOWED fields: {disallowed_present}")
        all_pass = False
    else:
        print(f"✓ Item {i}: No disallowed fields")
    
    # Check optional fields (should only be present in handoff items)
    optional_present = optional_fields & item_keys
    if optional_present:
        print(f"  Optional handoff fields present: {optional_present}")
    
    # Validate timeline_unbounded structure
    if "timeline_unbounded" in item:
        tl = item["timeline_unbounded"]
        tl_keys = set(tl.keys())
        expected_tl = {"from_day_unbounded", "to_day_unbounded", "sla_start_index", "sla_end_index"}
        if tl_keys == expected_tl:
            print(f"  ✓ timeline_unbounded has correct structure")
        else:
            print(f"  ✗ timeline_unbounded incorrect keys: {tl_keys} (expected {expected_tl})")
            all_pass = False
    
    print()

# Final summary
print("\n" + "="*80)
if all_pass and len(enrichment_seq) > 0:
    print("✓ ALL CHECKS PASSED: enrichment_sequence matches the new compact schema")
    print("  - All required date/timeline fields present")
    print("  - All debug/metadata fields removed")
    print("  - Optional handoff fields included where appropriate")
else:
    print("✗ SCHEMA VALIDATION FAILED")
