#!/usr/bin/env python
"""
Test script to verify inventory_selected merge with S2 enrichment items.
"""
import json
import sys
from pathlib import Path

# Real SID used for testing
TEST_SID = "7e9f27d9-2494-4751-b781-79bb1cd72803"

def load_plan_from_file():
    """Load a pre-computed plan from JSON file."""
    # Look for plan files in data directory
    data_dir = Path("/data/plans")
    
    # Find most recent plan for this SID
    plan_files = list(Path.glob(data_dir, f"*{TEST_SID}*plan*.json"))
    if not plan_files:
        # Try alternate location
        plan_files = list(Path.glob(Path.cwd(), f"*{TEST_SID}*.json"))
    
    if plan_files:
        latest = sorted(plan_files)[-1]
        print(f"Loading plan from: {latest}")
        with open(latest) as f:
            return json.load(f)
    
    print(f"No plan file found for SID {TEST_SID}")
    print(f"Searched in: {data_dir}")
    return None

def test_inventory_merge():
    """Verify inventory_selected merge with S2 enrichment items."""
    print(f"\n{'='*80}")
    print(f"Testing inventory_selected merge with S2 enrichment items")
    print(f"SID: {TEST_SID}")
    print(f"{'='*80}\n")
    
    # Try to load plan from file
    plan = load_plan_from_file()
    if not plan:
        print("ERROR: Could not load plan. Running live computation instead...")
        # Fall back to computing directly
        return False
    
    # Extract inventory_selected
    inv_header = plan.get("inventory_header") or {}
    inv_selected = inv_header.get("inventory_selected") or []
    
    # Extract enrichment_sequence (S2 source)
    enrichment_seq = plan.get("enrichment_sequence") or []
    
    print(f"\nInventory Selected ({len(inv_selected)} items):")
    print(f"  Enrichment Sequence ({len(enrichment_seq)} S2 items available)")
    
    # Display inventory_selected with key fields
    print("\n" + "-" * 80)
    print(f"{'idx':<5} {'field':<30} {'planned_submit_index':<22} {'submit_date':<15}")
    print("-" * 80)
    
    for item in inv_selected:
        idx = item.get("idx", "?")
        field = str(item.get("field", ""))[:30]
        planned_idx = item.get("planned_submit_index", "?")
        submit_date = item.get("submit_date", "?")
        print(f"{idx:<5} {field:<30} {str(planned_idx):<22} {str(submit_date):<15}")
    
    print("-" * 80)
    
    # Verify requirements
    print("\n" + "=" * 80)
    print("VERIFICATION CHECKS:")
    print("=" * 80)
    
    checks = []
    
    # 1. Check inventory_selected is not empty
    if inv_selected:
        checks.append(("✓", "inventory_selected has items", True))
    else:
        checks.append(("✗", "inventory_selected is empty", False))
    
    # 2. Check idx values are sequential and 1-based
    if inv_selected:
        idx_values = [item.get("idx") for item in inv_selected]
        expected_idx = list(range(1, len(inv_selected) + 1))
        if idx_values == expected_idx:
            checks.append(("✓", "idx values are sequential 1-based", True))
        else:
            checks.append(("✗", f"idx values incorrect. Got {idx_values}, expected {expected_idx}", False))
    
    # 3. Check planned_submit_index is in ascending order
    if inv_selected:
        planned_indices = [item.get("planned_submit_index", 0) for item in inv_selected]
        if planned_indices == sorted(planned_indices):
            checks.append(("✓", "planned_submit_index in ascending order", True))
        else:
            checks.append(("✗", f"planned_submit_index not sorted. Got {planned_indices}", False))
    
    # 4. Check for duplicate fields
    if inv_selected:
        fields = [item.get("field") for item in inv_selected]
        unique_fields = set(fields)
        if len(fields) == len(unique_fields):
            checks.append(("✓", "No duplicate fields", True))
        else:
            dup_fields = [f for f in fields if fields.count(f) > 1]
            checks.append(("✗", f"Duplicate fields found: {set(dup_fields)}", False))
    
    # 5. Check for required fields in S2 items (if any)
    s2_items = [item for item in inv_selected if item.get("planned_submit_index", 0) >= 41]
    if s2_items:
        # All S2 items should have required fields
        s2_complete = True
        for item in s2_items:
            required = ["field", "planned_submit_index"]
            if not all(k in item for k in required):
                s2_complete = False
                break
        if s2_complete:
            checks.append(("✓", f"S2 items have required fields ({len(s2_items)} items)", True))
        else:
            checks.append(("✗", f"S2 items missing required fields", False))
    else:
        checks.append(("ℹ", "No S2 items found in inventory_selected", None))
    
    # Print check results
    print()
    for icon, msg, result in checks:
        if result is None:
            print(f"  {icon}  {msg}")
        else:
            status_text = "PASS" if result else "FAIL"
            print(f"  {icon}  {msg:<60} [{status_text}]")
    
    # Summary
    passed = sum(1 for _, _, r in checks if r is True)
    failed = sum(1 for _, _, r in checks if r is False)
    
    print("\n" + "=" * 80)
    print(f"SUMMARY: {passed} passed, {failed} failed")
    print("=" * 80)
    
    if failed == 0:
        print("\n✓ All checks passed!")
        return True
    else:
        print(f"\n✗ {failed} check(s) failed!")
        return False

if __name__ == "__main__":
    success = test_inventory_merge()
    sys.exit(0 if success else 1)
