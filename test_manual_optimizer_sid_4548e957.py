"""
Manual test: Load actual plan from SID 4548e957 and invoke optimizer directly.
Expected: optimizer reduces unbounded from 51 to ≤50 and adds metadata.
If this works, the bug is in the serialization path (optimizer not called on write).
"""
import json
from backend.strategy.planner import optimize_overlap_for_inbound_cap

# Load actual plan from the SID
plan_path = r"c:\dev\credit-analyzer\runs\4548e957-5918-4b06-97ce-ce50e75830a2\cases\accounts\9\strategy\experian\plan_wd0.json"
with open(plan_path, "r") as f:
    plan = json.load(f)

print("=" * 80)
print("BEFORE OPTIMIZATION")
print("=" * 80)
print(f"Unbounded: {plan['summary']['total_effective_days_unbounded']}")
print(f"Closer field: {plan['inventory_header']['inventory_selected'][1]['field']}")
print(f"Closer submit date: {plan['inventory_header']['inventory_selected'][1]['planned_submit_date']}")
print(f"Closer submit index: {plan['inventory_header']['inventory_selected'][1]['planned_submit_index']}")
print(f"Closer effective_unbounded: {plan['inventory_header']['inventory_selected'][1]['effective_contribution_days_unbounded']}")
print(f"Metadata fields present: {[k for k in plan['summary'].keys() if k.startswith('inbound_cap_')]}")
print()

# Call the optimizer directly (need weekend/holidays/span_cap params)
from datetime import date

optimized = optimize_overlap_for_inbound_cap(
    plan=plan,
    max_unbounded_inbound_day=50,
    weekend={5, 6},  # Sat, Sun
    holidays=set(),
    enforce_span_cap=False,
    include_notes=False
)

print("=" * 80)
print("AFTER OPTIMIZATION")
print("=" * 80)
print(f"Unbounded: {optimized['summary']['total_effective_days_unbounded']}")
print(f"Closer field: {optimized['inventory_header']['inventory_selected'][1]['field']}")
print(f"Closer submit date: {optimized['inventory_header']['inventory_selected'][1]['planned_submit_date']}")
print(f"Closer submit index: {optimized['inventory_header']['inventory_selected'][1]['planned_submit_index']}")
print(f"Closer effective_unbounded: {optimized['inventory_header']['inventory_selected'][1]['effective_contribution_days_unbounded']}")
print(f"Metadata fields present: {[k for k in optimized['summary'].keys() if k.startswith('inbound_cap_')]}")
print()

# Print metadata values
if "inbound_cap_applied" in optimized["summary"]:
    print("Metadata values:")
    for k, v in optimized["summary"].items():
        if k.startswith("inbound_cap_"):
            print(f"  {k}: {v}")
    print()

# Assertions
assert optimized['summary']['total_effective_days_unbounded'] <= 50, \
    f"Expected unbounded ≤50, got {optimized['summary']['total_effective_days_unbounded']}"
assert "inbound_cap_applied" in optimized["summary"], "Expected metadata field 'inbound_cap_applied'"
assert optimized["summary"]["inbound_cap_applied"] is True, "Expected inbound_cap_applied=True"
assert optimized["summary"]["inbound_cap_before"] == 51, "Expected inbound_cap_before=51"

print("✅ All assertions passed!")
print("🔍 CONCLUSION: Optimizer works correctly in isolation.")
print("🐛 BUG: Serialization path does NOT call optimizer or uses stale plan.")
