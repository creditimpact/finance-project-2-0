"""Verify SID 88c4ee20 plan_wd3.json after default-value fix."""
import json
from pathlib import Path

plan_path = Path(r"c:\dev\credit-analyzer\runs\88c4ee20-cd6f-45f1-a9bb-ebbfb1aa1f13\cases\accounts\9\strategy\experian\plan_wd3.json")

with open(plan_path) as f:
    plan = json.load(f)

summary = plan["summary"]
print("=" * 60)
print("FRESHLY WRITTEN PLAN (after default-value fix)")
print("=" * 60)
print(f"total_effective_days_unbounded: {summary.get('total_effective_days_unbounded')}")
print(f"inbound_cap_hard: {summary.get('inbound_cap_hard')}")
print(f"inbound_cap_before: {summary.get('inbound_cap_before')}")
print(f"inbound_cap_after: {summary.get('inbound_cap_after')}")
print(f"inbound_cap_applied: {summary.get('inbound_cap_applied')}")
print(f"total_overlap_unbounded_days: {summary.get('total_overlap_unbounded_days')}")

# Find the closer dispute (last item in sequence)
closer = plan["sequence_debug"][-1]
print("\n" + "=" * 60)
print("CLOSER DISPUTE (submit_batch_index=7)")
print("=" * 60)
print(f"calendar_day_index: {closer['calendar_day_index']}")
print(f"handoff_days_before_prev_sla_end: {closer.get('handoff_days_before_prev_sla_end')}")
print(f"overlap_raw_days_with_prev: {closer.get('overlap_raw_days_with_prev')}")
print(f"overlap_effective_unbounded_with_prev: {closer.get('overlap_effective_unbounded_with_prev')}")

print("\n" + "=" * 60)
print("VERIFICATION")
print("=" * 60)
if summary.get("total_effective_days_unbounded", 999) <= 50:
    print("✅ unbounded ≤ 50")
else:
    print(f"❌ unbounded > 50: {summary.get('total_effective_days_unbounded')}")

if summary.get("inbound_cap_applied") is True:
    print("✅ inbound_cap_applied = True")
else:
    print(f"❌ inbound_cap_applied not True: {summary.get('inbound_cap_applied')}")

if summary.get("inbound_cap_before") is not None:
    print(f"✅ inbound_cap_before = {summary.get('inbound_cap_before')}")
else:
    print("❌ inbound_cap_before missing")

if closer.get("handoff_days_before_prev_sla_end", 0) > 1:
    print(f"✅ handoff increased: {closer.get('handoff_days_before_prev_sla_end')}")
else:
    print(f"❌ handoff NOT increased: {closer.get('handoff_days_before_prev_sla_end')}")

if closer.get("overlap_raw_days_with_prev", 0) > 1:
    print(f"✅ overlap increased: {closer.get('overlap_raw_days_with_prev')}")
else:
    print(f"❌ overlap NOT increased: {closer.get('overlap_raw_days_with_prev')}")
