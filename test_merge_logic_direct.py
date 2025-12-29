#!/usr/bin/env python
"""
Test the inventory_selected merge logic directly.
"""
import sys
from pathlib import Path

# Add repo to path
sys.path.insert(0, str(Path(__file__).parent))

def test_merge_logic():
    """Test the merge logic that was added to simplify_plan_for_public."""
    
    print("\n" + "="*80)
    print("Testing inventory_selected merge logic")
    print("="*80)
    
    # Simulate S1 inventory_selected (from sequence_debug)
    simplified_inv_sel = [
        {
            "idx": 1,
            "field": "account_number",
            "planned_submit_index": 10,
            "submit_date": "2025-01-20",
            "submit_weekday": "Monday",
            "timeline_unbounded": {"from_day_unbounded": 10, "to_day_unbounded": 20},
        },
        {
            "idx": 2,
            "field": "date_opened",
            "planned_submit_index": 15,
            "submit_date": "2025-01-25",
            "submit_weekday": "Saturday",
            "timeline_unbounded": {"from_day_unbounded": 15, "to_day_unbounded": 25},
        },
    ]
    
    # Simulate enrichment_sequence (S2 items from skeleton2 enrichment)
    enrichment_sequence = [
        {
            "field": "balance_owed",
            "planned_submit_index": 42,  # Day 42 (S2)
            "submit_date": "2025-02-11",
            "submit_weekday": "Tuesday",
            "unbounded_end_date": "2025-02-21",
            "unbounded_end_weekday": "Friday",
            "timeline_unbounded": {
                "from_day_unbounded": 42,
                "to_day_unbounded": 52,
                "sla_start_index": 42,
                "sla_end_index": 52,
            },
            "business_sla_days": 10,
        },
        {
            "field": "account_type",
            "planned_submit_index": 5,  # Day 5 (would be S1)
            "submit_date": "2025-01-15",
            "submit_weekday": "Wednesday",
        },
    ]
    
    print("\nBefore merge:")
    print(f"  S1 items: {len(simplified_inv_sel)}")
    for item in simplified_inv_sel:
        print(f"    idx={item['idx']} field={item['field']} planned_submit_index={item['planned_submit_index']}")
    
    print(f"\n  S2 items available: {len(enrichment_sequence)}")
    for item in enrichment_sequence:
        print(f"    field={item['field']} planned_submit_index={item['planned_submit_index']}")
    
    # NOW EXECUTE THE MERGE LOGIC (from planner.py)
    print("\n" + "-"*80)
    print("Executing merge logic...")
    print("-"*80)
    
    # Track which fields are already in inventory_selected (from Skeleton #1)
    s1_fields = {str(item.get("field")) for item in simplified_inv_sel if "field" in item}
    print(f"\nS1 fields: {s1_fields}")
    
    # Add Skeleton #2 items, avoiding duplicates
    for enrich_item in enrichment_sequence:
        field = str(enrich_item.get("field", ""))
        if field and field not in s1_fields:
            print(f"\nAdding S2 item: {field} (planned_submit_index={enrich_item.get('planned_submit_index')})")
            # Project enrichment item to inventory_selected format
            projected_item = {
                "field": field,
                "planned_submit_index": int(enrich_item.get("planned_submit_index", 0)),
            }
            
            # Copy date fields if present
            if "submit_date" in enrich_item:
                projected_item["submit_date"] = enrich_item["submit_date"]
            if "submit_weekday" in enrich_item:
                projected_item["submit_weekday"] = enrich_item["submit_weekday"]
            if "unbounded_end_date" in enrich_item:
                projected_item["unbounded_end_date"] = enrich_item["unbounded_end_date"]
            if "unbounded_end_weekday" in enrich_item:
                projected_item["unbounded_end_weekday"] = enrich_item["unbounded_end_weekday"]
            
            # Copy timeline_unbounded
            if "timeline_unbounded" in enrich_item:
                projected_item["timeline_unbounded"] = dict(enrich_item["timeline_unbounded"])
            
            # Copy SLA days
            if "business_sla_days" in enrich_item:
                projected_item["business_sla_days"] = enrich_item["business_sla_days"]
            
            # Copy optional handoff metadata
            for handoff_key in ["between_skeleton1_indices", "handoff_reference_day", "half_sla_offset"]:
                if handoff_key in enrich_item:
                    projected_item[handoff_key] = enrich_item[handoff_key]
            
            simplified_inv_sel.append(projected_item)
        else:
            if field in s1_fields:
                print(f"\nSkipping S2 item (duplicate): {field}")
            else:
                print(f"\nSkipping S2 item (no field): {enrich_item}")
    
    # Sort combined list by planned_submit_index (ascending), preserving order for ties
    print(f"\nBefore sort: {[item.get('planned_submit_index') for item in simplified_inv_sel]}")
    simplified_inv_sel.sort(key=lambda x: int(x.get("planned_submit_index", 0)))
    print(f"After sort:  {[item.get('planned_submit_index') for item in simplified_inv_sel]}")
    
    # Recompute idx to reflect chronological order
    print(f"\nRecomputing idx...")
    for idx, item in enumerate(simplified_inv_sel, start=1):
        item_copy = dict(item)
        item_copy_clean = {k: v for k, v in item_copy.items() if k != "idx"}  # Remove old idx if present
        item_copy_clean["idx"] = idx  # Add new sequential idx
        simplified_inv_sel[idx - 1] = item_copy_clean
    
    print("\nAfter merge:")
    print(f"  Total items: {len(simplified_inv_sel)}")
    print(f"\n  {'idx':<5} {'field':<30} {'planned_submit_index':<22}")
    print(f"  {'-'*57}")
    for item in simplified_inv_sel:
        idx = item.get("idx", "?")
        field = str(item.get("field", ""))[:30]
        planned_idx = item.get("planned_submit_index", "?")
        print(f"  {idx:<5} {field:<30} {str(planned_idx):<22}")
    
    # Verification checks
    print("\n" + "="*80)
    print("VERIFICATION CHECKS:")
    print("="*80)
    
    checks = []
    
    # 1. Correct total count
    expected_count = 4  # 2 S1 + 2 S2 (both S2 items added: balance_owed and account_type)
    if len(simplified_inv_sel) == expected_count:
        checks.append(("✓", f"Total item count is {expected_count}", True))
    else:
        checks.append(("✗", f"Expected {expected_count} items, got {len(simplified_inv_sel)}", False))
    
    # 2. Check idx values are sequential and 1-based
    idx_values = [item.get("idx") for item in simplified_inv_sel]
    expected_idx = list(range(1, len(simplified_inv_sel) + 1))
    if idx_values == expected_idx:
        checks.append(("✓", f"idx values are sequential 1-based: {idx_values}", True))
    else:
        checks.append(("✗", f"idx values incorrect. Got {idx_values}, expected {expected_idx}", False))
    
    # 3. Check planned_submit_index is in ascending order
    planned_indices = [item.get("planned_submit_index", 0) for item in simplified_inv_sel]
    if planned_indices == sorted(planned_indices):
        checks.append(("✓", f"planned_submit_index in ascending order: {planned_indices}", True))
    else:
        checks.append(("✗", f"planned_submit_index not sorted. Got {planned_indices}", False))
    
    # 4. Check for duplicate fields
    fields = [item.get("field") for item in simplified_inv_sel]
    unique_fields = set(fields)
    if len(fields) == len(unique_fields):
        checks.append(("✓", f"No duplicate fields: {fields}", True))
    else:
        dup_fields = [f for f in fields if fields.count(f) > 1]
        checks.append(("✗", f"Duplicate fields found: {set(dup_fields)}", False))
    
    # 5. Check S2 item is present
    s2_item = next((item for item in simplified_inv_sel if item.get("field") == "balance_owed"), None)
    if s2_item:
        checks.append(("✓", f"S2 item (balance_owed) present with planned_submit_index={s2_item.get('planned_submit_index')}", True))
    else:
        checks.append(("✗", "S2 item (balance_owed) not found", False))
    
    # 6. Check S2 item has required fields
    if s2_item:
        required_fields = ["field", "planned_submit_index", "submit_date", "timeline_unbounded"]
        missing = [f for f in required_fields if f not in s2_item]
        if not missing:
            checks.append(("✓", f"S2 item has all required fields: {required_fields}", True))
        else:
            checks.append(("✗", f"S2 item missing fields: {missing}", False))
    
    # Print check results
    print()
    for icon, msg, result in checks:
        status_text = "PASS" if result else "FAIL"
        print(f"  {icon}  {msg:<70} [{status_text}]")
    
    # Summary
    passed = sum(1 for _, _, r in checks if r is True)
    failed = sum(1 for _, _, r in checks if r is False)
    
    print("\n" + "="*80)
    print(f"SUMMARY: {passed} passed, {failed} failed")
    print("="*80)
    
    if failed == 0:
        print("\n✓ All checks passed!")
        return True
    else:
        print(f"\n✗ {failed} check(s) failed!")
        return False

if __name__ == "__main__":
    success = test_merge_logic()
    sys.exit(0 if success else 1)
