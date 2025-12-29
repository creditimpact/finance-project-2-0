"""
Manual test: Load actual plan from SID d344b1f0 and invoke optimizer directly.
Prove that optimizer reduces unbounded from 51 to ≤50 and adds metadata.
"""
import json
from pathlib import Path
from copy import deepcopy
from backend.strategy.planner import optimize_overlap_for_inbound_cap

def load_plan(path: str):
    return json.loads(Path(path).read_text(encoding="utf-8"))

# Load actual plan from the SID (account 9, experian Monday)
plan_path = r"c:\dev\credit-analyzer\runs\d344b1f0-2366-4e64-8266-38554cf3cbac\cases\accounts\9\strategy\experian\plan_wd0.json"
plan = load_plan(plan_path)

print("=" * 80)
print("BEFORE OPTIMIZATION (SID d344b1f0, account 9, experian, wd0)")
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
print(f"total_overlap_unbounded_days: {plan['summary'].get('total_overlap_unbounded_days')}")
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

print(f"total_overlap_unbounded_days: {optimized['summary'].get('total_overlap_unbounded_days')}")

# Closer overlap fields in sequence_debug
closer_seq = optimized.get("sequence_debug", [])[1] if len(optimized.get("sequence_debug", [])) > 1 else {}
print(f"Closer overlap_raw_days_with_prev: {closer_seq.get('overlap_raw_days_with_prev')}")
print(f"Closer overlap_effective_unbounded_with_prev: {closer_seq.get('overlap_effective_unbounded_with_prev')}")

# Assertions
before = int(plan['summary'].get('total_effective_days_unbounded', 0) or 0)
after = int(optimized['summary'].get('total_effective_days_unbounded', 0) or 0)
assert after <= 50, f"Expected unbounded ≤50, got {after}"
assert "inbound_cap_applied" in optimized["summary"], "Expected inbound_cap_applied in metadata"
assert optimized["summary"]["inbound_cap_applied"] is True, "Expected inbound_cap_applied=True"
ic_before = int(optimized["summary"].get("inbound_cap_before", -1) or -1)
assert ic_before == 51, f"Expected inbound_cap_before=51, got {ic_before}"
assert ic_before == before, f"Expected inbound_cap_before==before ({before}), got {ic_before}"
print("\n✅ Optimizer reduces 51→≤50 with metadata on SID d344b1f0 (wd0)")
