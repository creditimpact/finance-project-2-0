"""
Manual test: Load actual plan from SID 06cea705 and invoke optimizer directly.
Expected: optimizer reduces unbounded from 51 to ≤50 and adds metadata.
Also prints handoff_days_before_prev_sla_end for visibility.
"""
import json
from datetime import date
from backend.strategy.planner import optimize_overlap_for_inbound_cap
from copy import deepcopy

# Load actual plan from the SID (account 9, experian Monday)
plan_path = r"c:\dev\credit-analyzer\runs\06cea705-09d5-41e6-8215-de4c34b050e0\cases\accounts\9\strategy\experian\plan_wd0.json"
with open(plan_path, "r", encoding="utf-8") as f:
    plan = json.load(f)

print("=" * 80)
print("BEFORE OPTIMIZATION (SID 06cea705, wd0)")
print("=" * 80)
print(f"Unbounded: {plan['summary'].get('total_effective_days_unbounded')}")
sel = plan.get("inventory_header", {}).get("inventory_selected", [])
if len(sel) >= 2:
    print(f"Opener effective_unbounded: {sel[0].get('effective_contribution_days_unbounded')}")
    print(f"Closer effective_unbounded: {sel[1].get('effective_contribution_days_unbounded')}")

print("handoff_days_before_prev_sla_end:")
for e in plan.get("sequence_debug", []):
    idx = e.get("idx")
    hod = e.get("handoff_days_before_prev_sla_end")
    print(f"  idx={idx} handoff_days_before_prev_sla_end={hod}")

print(f"Metadata fields present: {[k for k in plan['summary'].keys() if k.startswith('inbound_cap_')]}")
print()

# Call the optimizer directly with same parameters used by planner
optimized = optimize_overlap_for_inbound_cap(
    plan=deepcopy(plan),
    max_unbounded_inbound_day=50,
    weekend={5, 6},  # Sat, Sun
    holidays=set(),
    enforce_span_cap=False,
    include_notes=False,
)

print("=" * 80)
print("AFTER OPTIMIZATION")
print("=" * 80)
print(f"Unbounded: {optimized['summary'].get('total_effective_days_unbounded')}")
sel2 = optimized.get("inventory_header", {}).get("inventory_selected", [])
if len(sel2) >= 2:
    print(f"Opener effective_unbounded: {sel2[0].get('effective_contribution_days_unbounded')}")
    print(f"Closer effective_unbounded: {sel2[1].get('effective_contribution_days_unbounded')}")

print("handoff_days_before_prev_sla_end:")
for e in optimized.get("sequence_debug", []):
    idx = e.get("idx")
    hod = e.get("handoff_days_before_prev_sla_end")
    print(f"  idx={idx} handoff_days_before_prev_sla_end={hod}")

print(f"Metadata fields present: {[k for k in optimized['summary'].keys() if k.startswith('inbound_cap_')]}")
print("Metadata values:")
for k, v in optimized["summary"].items():
    if k.startswith("inbound_cap_"):
        print(f"  {k}: {v}")

# Assertions
before = int(plan['summary'].get('total_effective_days_unbounded', 0) or 0)
after = int(optimized['summary'].get('total_effective_days_unbounded', 0) or 0)
assert after <= 50, f"Expected unbounded ≤50, got {after}"
assert "inbound_cap_applied" in optimized["summary"], "Expected inbound_cap_applied in metadata"
assert optimized["summary"]["inbound_cap_applied"] is True, "Expected inbound_cap_applied=True"
ic_before = int(optimized["summary"].get("inbound_cap_before", -1) or -1)
assert ic_before == 51, f"Expected inbound_cap_before=51, got {ic_before}"
assert ic_before == before, f"Expected inbound_cap_before==before ({before}), got {ic_before}"
print("✅ Optimizer reduces 51→≤50 with metadata on SID 06cea705 (wd0)")
