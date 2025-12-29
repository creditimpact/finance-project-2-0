#!/usr/bin/env python
"""
Re-run planner for SID 3f1aee45 account 9 with S2 flags enabled.
Load the summary.json, run the strategy, and write updated plan files.
"""

import os
import json
from pathlib import Path

# Enable S2 with day-40 strongest
os.environ["PLANNER_ENABLE_SKELETON2"] = "1"
os.environ["PLANNER_SKELETON2_ENABLE_DAY40_STRONGEST"] = "1"
os.environ["PLANNER_SKELETON2_MAX_ITEMS_PER_HANDOFF"] = "2"
os.environ["PLANNER_SKELETON2_MIN_SLA_DAYS"] = "5"
os.environ["PLANNER_SKELETON2_ENFORCE_CADENCE"] = "0"

# Import after setting env
from backend.strategy.config import load_planner_env
from backend.strategy.runner import run_for_summary

# Paths
sid = "3f1aee45-d3c7-46eb-a309-2dc7b084681e"
acct_id = 9
runs_root = Path("c:\\dev\\credit-analyzer\\runs")
summary_path = runs_root / sid / "cases" / "accounts" / str(acct_id) / "summary.json"

if not summary_path.exists():
    print(f"ERROR: {summary_path} not found")
    exit(1)

print(f"Re-running strategy for SID {sid}, account {acct_id}")
print(f"PLANNER_ENABLE_SKELETON2={os.environ.get('PLANNER_ENABLE_SKELETON2')}")
print(f"PLANNER_SKELETON2_ENABLE_DAY40_STRONGEST={os.environ.get('PLANNER_SKELETON2_ENABLE_DAY40_STRONGEST')}")

# Load env and run
env = load_planner_env()
print(f"Planner env loaded: skeleton2_enabled={env.skeleton2_enabled}, day40_strongest={env.skeleton2_enable_day40_strongest}")

# Run strategy
try:
    run_for_summary(
        summary_path,
        mode="per_bureau_joint_optimize",
        forced_start=None,
        env=env,
    )
    print("\n✓ Strategy run completed successfully")
except Exception as e:
    print(f"\n✗ Strategy run failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# List generated files
strategy_dir = summary_path.parent / "strategy"
for bureau in ["equifax", "experian", "transunion"]:
    bureau_dir = strategy_dir / bureau
    if bureau_dir.exists():
        files = sorted(bureau_dir.glob("plan_wd*.json"))
        print(f"{bureau}: Generated {len(files)} plan files")
