#!/usr/bin/env python
"""
Quick test of S2 enrichment output format to verify it matches S1 style with dates.
"""

import os
import json
from datetime import date
from backend.strategy.planner import compute_optimal_plan
from backend.strategy.types import Finding

# Enable S2 with day-40 rule
os.environ["PLANNER_ENABLE_SKELETON2"] = "1"
os.environ["PLANNER_SKELETON2_ENABLE_DAY40_STRONGEST"] = "1"
os.environ["PLANNER_SKELETON2_MAX_ITEMS_PER_HANDOFF"] = "2"
os.environ["PLANNER_SKELETON2_MIN_SLA_DAYS"] = "5"
os.environ["PLANNER_SKELETON2_ENFORCE_CADENCE"] = "0"

# Build findings
findings = [
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
    Finding(
        field="account_age",
        min_days=8,
        duration_unit="business_days",
        default_decision="supportive_needs_companion",
        category="account",
    ),
]

# Compute plan
plan = compute_optimal_plan(
    findings,
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

# Extract and show enrichment sequence from one weekday
wd_plan = plan["weekday_plans"][0]
enrichment = wd_plan.get("enrichment_sequence", [])

print(f"\n=== Skeleton #2 Enrichment Sequence (Compact Output) ===")
print(f"Total items: {len(enrichment)}\n")

if enrichment:
    # Show structure of first item
    first_item = enrichment[0]
    print(f"First item keys: {sorted(first_item.keys())}\n")
    
    # Pretty print first few items
    for i, item in enumerate(enrichment[:3]):
        print(f"Item {i}:")
        print(json.dumps(item, indent=2, default=str))
        print()
else:
    print("No enrichment items generated.")

# Validate structure
required_fields = {"idx", "field", "planned_submit_index", "submit_date", "submit_weekday", 
                  "unbounded_end_date", "unbounded_end_weekday", "business_sla_days", "timeline_unbounded"}
disallowed_fields = {"placement_reason", "day40_target_index", "day40_adjustment_reason", 
                     "calendar_day_index", "submit_on", "sla_window"}

print("\n=== Validation ===")
if enrichment:
    all_present = all(all(f in item for f in required_fields) for item in enrichment)
    print(f"✓ All required date fields present: {all_present}")
    
    no_debug = all(not any(f in item for f in disallowed_fields) for item in enrichment)
    print(f"✓ No debug fields present: {no_debug}")
    
    has_handoff_optional = any(any(f in item for f in ["between_skeleton1_indices", "handoff_reference_day"]) for item in enrichment)
    print(f"Handoff optional fields: {has_handoff_optional}")
else:
    print("No enrichment items to validate.")
