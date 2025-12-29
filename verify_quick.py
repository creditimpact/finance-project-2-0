"""Quick verification of overlap model."""
import json
from pathlib import Path

plan_path = Path(r"c:\dev\credit-analyzer\runs\88c4ee20-cd6f-45f1-a9bb-ebbfb1aa1f13\cases\accounts\9\strategy\experian\plan_wd3.json")

with open(plan_path) as f:
    plan = json.load(f)

s = plan["summary"]
seq = plan["sequence_debug"]

print("SUMMARY:")
print(f"  unbounded: {s.get('total_effective_days_unbounded')}")
print(f"  overlap: {s.get('total_overlap_unbounded_days')}")
print(f"  sum_items: {s.get('inbound_cap_sum_items_unbounded')}")
print(f"  required_overlap: {s.get('inbound_cap_required_overlap')}")
print(f"  base_overlap_min: {s.get('inbound_cap_base_overlap_min')}")
print(f"  before: {s.get('inbound_cap_before')}")
print(f"  after: {s.get('inbound_cap_after')}")
print(f"  applied: {s.get('inbound_cap_applied')}")
print(f"  unachievable: {s.get('inbound_cap_unachievable')}")

print("\nIDENTITY CHECK:")
sum_items = s.get('inbound_cap_sum_items_unbounded', 0)
overlap = s.get('total_overlap_unbounded_days', 0)
unbounded = s.get('total_effective_days_unbounded', 0)
calc = sum_items - overlap
print(f"  {sum_items} - {overlap} = {calc}")
print(f"  actual = {unbounded}")
print(f"  ✅ VALID" if calc == unbounded else "❌ INVALID")

print("\nPER-ITEM FIELDS:")
for idx, item in enumerate(seq):
    print(f"\n  [{idx+1}] {item['field']}:")
    print(f"    effective_unbounded: {item.get('effective_contribution_days_unbounded')}")
    print(f"    running_at_submit: {item.get('running_unbounded_at_submit')}")
    if idx > 0:
        print(f"    overlap_with_prev: {item.get('overlap_unbounded_days_with_prev')}")
        print(f"    handoff: {item.get('handoff_days_before_prev_sla_end')}")

print("\nHANDOFF >= 1 CHECK:")
for idx, item in enumerate(seq):
    if idx > 0:
        h = item.get('handoff_days_before_prev_sla_end', 0)
        status = "✅" if h >= 1 else "❌ VIOLATION"
        print(f"  [{idx+1}] handoff={h} {status}")
