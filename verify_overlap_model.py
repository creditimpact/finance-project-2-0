"""Verify overlap-based inbound cap model on SID 88c4ee20."""
import sys
import os
import json
from pathlib import Path

# Run the pipeline first
print("=" * 70)
print("STEP 1: Running pipeline for SID 88c4ee20...")
print("=" * 70)

os.environ["RUNS_ROOT"] = r"c:\dev\credit-analyzer\runs"
os.system('python -m backend.strategy.runner --sid 88c4ee20-cd6f-45f1-a9bb-ebbfb1aa1f13')

print("\n" + "=" * 70)
print("STEP 2: Verifying overlap-based model...")
print("=" * 70)

plan_path = Path(r"c:\dev\credit-analyzer\runs\88c4ee20-cd6f-45f1-a9bb-ebbfb1aa1f13\cases\accounts\9\strategy\experian\plan_wd3.json")

with open(plan_path) as f:
    plan = json.load(f)

summary = plan["summary"]
sequence = plan["sequence_debug"]

print(f"\n📊 SUMMARY FIELDS:")
print(f"  total_effective_days_unbounded: {summary.get('total_effective_days_unbounded')}")
print(f"  total_overlap_unbounded_days: {summary.get('total_overlap_unbounded_days')}")
print(f"  inbound_cap_sum_items_unbounded: {summary.get('inbound_cap_sum_items_unbounded')}")
print(f"  inbound_cap_required_overlap: {summary.get('inbound_cap_required_overlap')}")
print(f"  inbound_cap_base_overlap_min: {summary.get('inbound_cap_base_overlap_min')}")
print(f"  inbound_cap_before: {summary.get('inbound_cap_before')}")
print(f"  inbound_cap_after: {summary.get('inbound_cap_after')}")
print(f"  inbound_cap_applied: {summary.get('inbound_cap_applied')}")
print(f"  inbound_cap_unachievable: {summary.get('inbound_cap_unachievable')}")

print(f"\n  _debug_sum_items_unbounded: {summary.get('_debug_sum_items_unbounded')}")
print(f"  _debug_calculated_inbound: {summary.get('_debug_calculated_inbound')}")
print(f"  _debug_identity_valid: {summary.get('_debug_identity_valid')}")

print(f"\n🔍 SEQUENCE DEBUG (per-item fields):")
for idx, item in enumerate(sequence):
    print(f"\n  Item {idx + 1}: {item['field']}")
    print(f"    effective_contribution_days_unbounded: {item.get('effective_contribution_days_unbounded')}")
    print(f"    running_unbounded_at_submit: {item.get('running_unbounded_at_submit')}")
    print(f"    running_total_days_unbounded_after: {item.get('running_total_days_unbounded_after')}")
    
    if idx > 0:
        print(f"    overlap_unbounded_days_with_prev: {item.get('overlap_unbounded_days_with_prev')}")
        print(f"    overlap_raw_days_with_prev: {item.get('overlap_raw_days_with_prev')}")
        print(f"    handoff_days_before_prev_sla_end: {item.get('handoff_days_before_prev_sla_end')}")

print(f"\n📦 INVENTORY SELECTED (overlap fields):")
inventory_selected = plan["inventory_header"]["inventory_selected"]
for idx, item in enumerate(inventory_selected):
    if idx > 0:
        print(f"  Item {idx + 1}: {item['field']}")
        print(f"    overlap_days_with_prev: {item.get('overlap_days_with_prev')}")

print(f"\n✅ INVARIANT CHECKS:")

# Check 1: Identity formula
sum_items = summary.get('inbound_cap_sum_items_unbounded', 0)
total_overlap = summary.get('total_overlap_unbounded_days', 0)
total_unbounded = summary.get('total_effective_days_unbounded', 0)
calculated = sum_items - total_overlap

print(f"\n  Identity: total_unbounded = sum(items) - overlap")
print(f"    sum(items) = {sum_items}")
print(f"    overlap = {total_overlap}")
print(f"    calculated = {calculated}")
print(f"    actual = {total_unbounded}")
print(f"    ✅ VALID" if calculated == total_unbounded else f"    ❌ INVALID")

# Check 2: Handoff >= 1 for all connections
print(f"\n  Handoff >= 1 for all connections:")
all_handoff_valid = True
for idx, item in enumerate(sequence):
    if idx > 0:
        handoff = item.get('handoff_days_before_prev_sla_end', 0)
        if handoff < 1:
            print(f"    ❌ Item {idx + 1}: handoff = {handoff} (VIOLATION)")
            all_handoff_valid = False
if all_handoff_valid:
    print(f"    ✅ All handoffs >= 1")

# Check 3: Base overlap minimum
N = len(sequence)
base_min = summary.get('inbound_cap_base_overlap_min', 0)
print(f"\n  Base overlap minimum: (N-1) = ({N}-1) = {N-1}")
print(f"    inbound_cap_base_overlap_min = {base_min}")
print(f"    ✅ CORRECT" if base_min == N - 1 else f"    ❌ INCORRECT")

# Check 4: Required overlap calculation
required = summary.get('inbound_cap_required_overlap', 0)
expected_required = max(sum_items - 50, N - 1)
print(f"\n  Required overlap: max((sum_items - 50), (N-1))")
print(f"    max(({sum_items} - 50), {N - 1}) = {expected_required}")
print(f"    inbound_cap_required_overlap = {required}")
print(f"    ✅ CORRECT" if required == expected_required else f"    ❌ INCORRECT")

# Check 5: Final unbounded <= 50
print(f"\n  Inbound cap: unbounded <= 50")
print(f"    total_effective_days_unbounded = {total_unbounded}")
print(f"    ✅ PASS" if total_unbounded <= 50 else f"    ❌ FAIL")

print("\n" + "=" * 70)
print("VERIFICATION COMPLETE")
print("=" * 70)
