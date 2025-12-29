"""
Phase 1: Load and inspect real plan for SID 88c4ee20, weekday 3 (Thursday).
"""
import json
from pathlib import Path

def load_plan(path: str):
    return json.loads(Path(path).read_text(encoding="utf-8"))

# Load actual plan from SID 88c4ee20 (account 9, experian, Thursday/wd3)
plan_path = r"c:\dev\credit-analyzer\runs\88c4ee20-cd6f-45f1-a9bb-ebbfb1aa1f13\cases\accounts\9\strategy\experian\plan_wd3.json"
plan = load_plan(plan_path)

print("=" * 80)
print("PHASE 1: Current state of SID 88c4ee20 plan_wd3.json")
print("=" * 80)

# Summary
summary = plan.get("summary", {})
print(f"summary['total_effective_days_unbounded']: {summary.get('total_effective_days_unbounded')}")
print(f"summary.get('inbound_cap_hard'): {summary.get('inbound_cap_hard')}")
print(f"summary.get('inbound_cap_before'): {summary.get('inbound_cap_before')}")
print(f"summary.get('inbound_cap_after'): {summary.get('inbound_cap_after')}")
print(f"summary.get('total_overlap_unbounded_days'): {summary.get('total_overlap_unbounded_days')}")

# Closer in sequence_debug
sequence_debug = plan.get("sequence_debug", [])
if len(sequence_debug) >= 2:
    closer = sequence_debug[1]
    print(f"\nCloser in sequence_debug:")
    print(f"  calendar_day_index: {closer.get('calendar_day_index')}")
    print(f"  effective_contribution_days_unbounded: {closer.get('effective_contribution_days_unbounded')}")
    print(f"  handoff_days_before_prev_sla_end: {closer.get('handoff_days_before_prev_sla_end')}")
    print(f"  overlap_raw_days_with_prev: {closer.get('overlap_raw_days_with_prev')}")
    print(f"  overlap_effective_unbounded_with_prev: {closer.get('overlap_effective_unbounded_with_prev')}")

# Closer in sequence_compact
sequence_compact = plan.get("sequence_compact", [])
if len(sequence_compact) >= 2:
    closer_compact = sequence_compact[1]
    print(f"\nCloser in sequence_compact:")
    print(f"  overlap_days_with_prev: {closer_compact.get('overlap_days_with_prev')}")

# Closer in inventory_selected
inventory_selected = plan.get("inventory_header", {}).get("inventory_selected", [])
if len(inventory_selected) >= 2:
    closer_inv = inventory_selected[1]
    print(f"\nCloser in inventory_selected:")
    print(f"  overlap_days_with_prev: {closer_inv.get('overlap_days_with_prev')}")

print("\n" + "=" * 80)
