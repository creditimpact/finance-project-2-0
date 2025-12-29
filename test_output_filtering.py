#!/usr/bin/env python
"""
Test PLANNER_OUTPUT_OMIT_SUMMARY_AND_CONSTRAINTS flag behavior.
Verify summary and constraints are present when flag=0, absent when flag=1.
"""

import json
import os
from pathlib import Path
from backend.strategy.config import load_planner_env
from backend.strategy.runner import run_for_summary

sid = "3f1aee45-d3c7-46eb-a309-2dc7b084681e"
acct_id = 9
runs_root = Path("c:\\dev\\credit-analyzer\\runs")
summary_path = runs_root / sid / "cases" / "accounts" / str(acct_id) / "summary.json"

def verify_output(flag_value, expect_keys_absent=False):
    """Run planner with flag and verify output."""
    
    print(f"\n{'='*80}")
    print(f"Testing with PLANNER_OUTPUT_OMIT_SUMMARY_AND_CONSTRAINTS={flag_value}")
    print(f"{'='*80}\n")
    
    # Set the flag
    os.environ["PLANNER_OUTPUT_OMIT_SUMMARY_AND_CONSTRAINTS"] = str(flag_value)
    
    # Load env to verify flag
    env = load_planner_env()
    print(f"Flag loaded: output_omit_summary_and_constraints={env.output_omit_summary_and_constraints}")
    
    # Run strategy
    print(f"Running strategy...")
    try:
        run_for_summary(
            summary_path,
            mode="per_bureau_joint_optimize",
            forced_start=None,
            env=env,
        )
        print("✓ Strategy run completed\n")
    except Exception as e:
        print(f"✗ Strategy run failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Read and inspect output
    strategy_dir = summary_path.parent / "strategy"
    equifax_wd0 = strategy_dir / "equifax" / "plan_wd0.json"
    
    if not equifax_wd0.exists():
        print(f"ERROR: {equifax_wd0} not found")
        return False
    
    with open(equifax_wd0) as f:
        plan = json.load(f)
    
    # Check for summary and constraints
    has_summary = "summary" in plan
    has_constraints = "constraints" in plan
    
    print(f"Output inspection:")
    print(f"  Has 'summary' key: {has_summary}")
    print(f"  Has 'constraints' key: {has_constraints}")
    
    # Verify expectations
    if expect_keys_absent:
        success = (not has_summary) and (not has_constraints)
        if success:
            print(f"  ✓ Both keys absent as expected")
        else:
            print(f"  ✗ Expected keys to be absent, but found:")
            if has_summary:
                print(f"    - 'summary' present (should be absent)")
            if has_constraints:
                print(f"    - 'constraints' present (should be absent)")
    else:
        success = has_summary and has_constraints
        if success:
            print(f"  ✓ Both keys present as expected")
        else:
            print(f"  ✗ Expected keys to be present, but:")
            if not has_summary:
                print(f"    - 'summary' missing")
            if not has_constraints:
                print(f"    - 'constraints' missing")
    
    # Show key counts
    print(f"\nTop-level keys in output ({len(plan)} total):")
    for key in sorted(plan.keys())[:10]:
        print(f"  - {key}")
    if len(plan) > 10:
        print(f"  ... ({len(plan) - 10} more)")
    
    return success

# Test 1: Default (flag=0) - keys should be present
result1 = verify_output(0, expect_keys_absent=False)

# Test 2: Enabled (flag=1) - keys should be absent
result2 = verify_output(1, expect_keys_absent=True)

print(f"\n{'='*80}")
print(f"Summary:\n")
if result1:
    print(f"✓ Test 1 (flag=0): Keys present - PASSED")
else:
    print(f"✗ Test 1 (flag=0): Keys present - FAILED")

if result2:
    print(f"✓ Test 2 (flag=1): Keys absent - PASSED")
else:
    print(f"✗ Test 2 (flag=1): Keys absent - FAILED")

if result1 and result2:
    print(f"\n✓ All tests PASSED")
    exit(0)
else:
    print(f"\n✗ Some tests FAILED")
    exit(1)
