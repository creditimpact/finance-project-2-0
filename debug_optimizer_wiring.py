"""
Debug script: Trace optimizer execution in compute_optimal_plan for SID d344b1f0.
"""
import os
import sys
from pathlib import Path

# Set environment before importing
os.environ["STRATEGY_BEST_WEEKDAY_ENABLED"] = "0"
os.environ["RUNS_ROOT"] = r"c:\dev\credit-analyzer\runs"

from backend.strategy.runner import run_for_summary
from backend.strategy.config import load_planner_env

# Point to the exact summary
summary_path = Path(r"c:\dev\credit-analyzer\runs\d344b1f0-2366-4e64-8266-38554cf3cbac\cases\accounts\9\summary.json")

print("=" * 80)
print(f"Running planner for: {summary_path}")
print(f"STRATEGY_BEST_WEEKDAY_ENABLED={os.getenv('STRATEGY_BEST_WEEKDAY_ENABLED')}")
print("=" * 80)

env = load_planner_env()
run_for_summary(summary_path, mode="per_bureau_joint_optimize", forced_start=None, env=env)

print("\n" + "=" * 80)
print("Planner completed. Now checking written plan...")
print("=" * 80)

# Load the written plan
import json
plan_path = Path(r"c:\dev\credit-analyzer\runs\d344b1f0-2366-4e64-8266-38554cf3cbac\cases\accounts\9\strategy\experian\plan_wd0.json")
plan = json.loads(plan_path.read_text(encoding="utf-8"))

print(f"summary.total_effective_days_unbounded: {plan['summary'].get('total_effective_days_unbounded')}")
print(f"summary.inbound_cap_hard: {plan['summary'].get('inbound_cap_hard')}")
print(f"summary.inbound_cap_before: {plan['summary'].get('inbound_cap_before')}")
print(f"summary.inbound_cap_after: {plan['summary'].get('inbound_cap_after')}")
print(f"summary.inbound_cap_applied: {plan['summary'].get('inbound_cap_applied')}")
print(f"summary.total_overlap_unbounded_days: {plan['summary'].get('total_overlap_unbounded_days')}")

seq = plan.get("sequence_debug", [])
if len(seq) >= 2:
    closer = seq[1]
    print(f"closer.handoff_days_before_prev_sla_end: {closer.get('handoff_days_before_prev_sla_end')}")
    print(f"closer.overlap_raw_days_with_prev: {closer.get('overlap_raw_days_with_prev')}")
    print(f"closer.overlap_effective_unbounded_with_prev: {closer.get('overlap_effective_unbounded_with_prev')}")

inv_sel = plan.get("inventory_header", {}).get("inventory_selected", [])
if len(inv_sel) >= 2:
    print(f"inventory_selected[1].overlap_days_with_prev: {inv_sel[1].get('overlap_days_with_prev')}")
