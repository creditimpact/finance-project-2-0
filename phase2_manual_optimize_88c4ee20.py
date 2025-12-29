"""
Phase 2: Manually run optimizer on SID 88c4ee20 plan to prove it works in isolation.
"""
import json
from pathlib import Path
from copy import deepcopy
from backend.strategy.planner import optimize_overlap_for_inbound_cap

def load_plan(path: str):
    return json.loads(Path(path).read_text(encoding="utf-8"))

# Load actual plan from SID 88c4ee20 (account 9, experian, Thursday/wd3)
plan_path = r"c:\dev\credit-analyzer\runs\88c4ee20-cd6f-45f1-a9bb-ebbfb1aa1f13\cases\accounts\9\strategy\experian\plan_wd3.json"
plan = load_plan(plan_path)

original_unbounded = plan["summary"]["total_effective_days_unbounded"]
print("=" * 80)
print("PHASE 2: Manual optimizer invocation on SID 88c4ee20")
print("=" * 80)
print(f"BEFORE: total_effective_days_unbounded = {original_unbounded}")

# Get timezone and holidays from plan
weekend = {5, 6}  # Sat, Sun (standard)
holidays = set()  # No holidays in this plan

optimized = optimize_overlap_for_inbound_cap(
    deepcopy(plan),
    max_unbounded_inbound_day=50,
    weekend=weekend,
    holidays=holidays,
    enforce_span_cap=plan["constraints"]["enforce_span_cap"],
    include_notes=plan["constraints"]["include_notes"],
)

new_unbounded = optimized["summary"]["total_effective_days_unbounded"]
print(f"AFTER:  total_effective_days_unbounded = {new_unbounded}")

print(f"\nMetadata:")
print(f"  inbound_cap_before: {optimized['summary'].get('inbound_cap_before')}")
print(f"  inbound_cap_after: {optimized['summary'].get('inbound_cap_after')}")
print(f"  inbound_cap_applied: {optimized['summary'].get('inbound_cap_applied')}")
print(f"  total_overlap_unbounded_days: {optimized['summary'].get('total_overlap_unbounded_days')}")

# Closer details
sequence_debug = optimized.get("sequence_debug", [])
if len(sequence_debug) >= 2:
    closer = sequence_debug[1]
    print(f"\nCloser after optimization:")
    print(f"  calendar_day_index: {closer.get('calendar_day_index')}")
    print(f"  handoff_days_before_prev_sla_end: {closer.get('handoff_days_before_prev_sla_end')}")
    print(f"  overlap_raw_days_with_prev: {closer.get('overlap_raw_days_with_prev')}")
    print(f"  overlap_effective_unbounded_with_prev: {closer.get('overlap_effective_unbounded_with_prev')}")

# Assertions
assert new_unbounded <= 50, f"Expected unbounded ≤50, got {new_unbounded}"
assert optimized["summary"].get("inbound_cap_applied") is True, "Expected inbound_cap_applied=True"
assert optimized["summary"].get("inbound_cap_before") == original_unbounded
print(f"\n✅ Optimizer works: {original_unbounded} → {new_unbounded}")
