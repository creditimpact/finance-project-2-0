#!/usr/bin/env python
"""
Test PLANNER_OUTPUT_OMIT_SUMMARY_AND_CONSTRAINTS flag behavior.
Direct unit test of _project_plan_for_output() function.
"""

import json
from backend.strategy.io import _project_plan_for_output

# Create a mock plan with summary and constraints
mock_plan = {
    "anchor": {"date": "2025-12-29", "weekday": 0},
    "sequence_compact": [
        {"idx": 1, "field": "payment_status", "planned_submit_index": 0}
    ],
    "enrichment_sequence": [
        {"idx": 1, "field": "extra_field", "planned_submit_index": 39}
    ],
    "summary": {
        "total_effective_days_unbounded": 50,
        "inbound_cap_sum_items_unbounded": 2,
        "enrichment_stats": {"accepted": 1}
    },
    "constraints": {
        "max_calendar_span": 45,
        "last_submit_window": [0, 40],
        "no_weekend_submit": True,
        "output_mode": "compact"
    },
    "inventory_header": {
        "inventory_all": [],
        "inventory_selected": []
    }
}

print("="*80)
print("Test 1: omit_summary_and_constraints=False (default)")
print("="*80 + "\n")

result1 = _project_plan_for_output(mock_plan, omit_summary_and_constraints=False)

print(f"Keys in output: {list(result1.keys())}")
print(f"  Has 'summary': {'summary' in result1}")
print(f"  Has 'constraints': {'constraints' in result1}")
print(f"  Has 'sequence_compact': {'sequence_compact' in result1}")
print(f"  Has 'enrichment_sequence': {'enrichment_sequence' in result1}")

test1_pass = ("summary" in result1) and ("constraints" in result1)
if test1_pass:
    print("\n✓ Test 1 PASSED: Both summary and constraints present")
else:
    print("\n✗ Test 1 FAILED: Expected summary and constraints to be present")

print("\n" + "="*80)
print("Test 2: omit_summary_and_constraints=True")
print("="*80 + "\n")

result2 = _project_plan_for_output(mock_plan, omit_summary_and_constraints=True)

print(f"Keys in output: {list(result2.keys())}")
print(f"  Has 'summary': {'summary' in result2}")
print(f"  Has 'constraints': {'constraints' in result2}")
print(f"  Has 'sequence_compact': {'sequence_compact' in result2}")
print(f"  Has 'enrichment_sequence': {'enrichment_sequence' in result2}")

test2_pass = ("summary" not in result2) and ("constraints" not in result2)
if test2_pass:
    print("\n✓ Test 2 PASSED: Both summary and constraints removed")
else:
    print("\n✗ Test 2 FAILED: Expected summary and constraints to be absent")

# Verify other keys are preserved
test3_pass = ("sequence_compact" in result2) and ("enrichment_sequence" in result2)
if test3_pass:
    print("\n✓ Test 3 PASSED: Consumer-facing keys preserved (sequence_compact, enrichment_sequence)")
else:
    print("\n✗ Test 3 FAILED: Some consumer keys were removed")

# Verify original is not modified (shallow copy test)
test4_pass = ("summary" in mock_plan) and ("constraints" in mock_plan)
if test4_pass:
    print("✓ Test 4 PASSED: Original plan not modified")
else:
    print("✗ Test 4 FAILED: Original plan was modified")

print("\n" + "="*80)
if test1_pass and test2_pass and test3_pass and test4_pass:
    print("✓ All tests PASSED")
    print("\nSample output when flag enabled:\n")
    print(json.dumps(result2, indent=2))
    exit(0)
else:
    print("✗ Some tests FAILED")
    exit(1)
