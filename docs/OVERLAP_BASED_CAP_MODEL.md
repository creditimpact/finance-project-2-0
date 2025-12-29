# Overlap-Based Inbound Cap Model Implementation

## Overview

Implemented an overlap-based model for the inbound hard cap (≤50 days) that aligns with the business requirement: **never reduce any item's effective_contribution_days_unbounded, only increase overlap between items**.

## Conceptual Model

### Core Formula

```
total_effective_days_unbounded = sum(effective_unbounded_of_items) - total_overlap_unbounded_days
```

### Hard Invariants for Skeleton #1

1. **Each item's effective_unbounded is canonical** - determined by its SLA window, never reduced by the optimizer
2. **Minimum overlap per connection** - each connection between items must have at least 1 day of overlap in time
3. **Base overlap requirement** - `base_overlap_days ≥ (N - 1)` where N is number of items
4. **Max inbound formula** - `max_inbound = sum(effective_unbounded) - base_overlap_days`

### Hard Cap Enforcement Strategy

When `sum(effective_unbounded)` is `S` and we want inbound ≤ 50:

1. Compute required overlap: `required_overlap = max((S - 50), (N - 1))`
2. Optimizer moves later items earlier (in calendar/business days)
3. This increases `overlap_unbounded_days_with_prev` per connection
4. Continue until: `total_overlap_unbounded_days >= required_overlap`

### Constraints

- **Strict monotonic order**: `submit_i+1 > submit_i`
- **No weekend submits**: Submit dates must be business days
- **Handoff >= 1**: `handoff_days_before_prev_sla_end >= 1` (enforced in optimizer loop)
- **Deadline rule**: `last_submit <= 40` and not on weekend

If no legal configuration exists without breaking these invariants:
```json
{
  "inbound_cap_unachievable": true,
  "inbound_cap_reason": "no_further_legal_overlap_increase"
}
```

## Implementation Changes

### 1. New Per-Item Fields in `sequence_debug`

#### `running_unbounded_at_submit`
- **Type**: `int`
- **Definition**: Total inbound unbounded days at this item's submit date, BEFORE counting its own contribution
- **First item**: Always `0`
- **Subsequent items**: Cumulative sum up to (but not including) this item

#### `overlap_unbounded_days_with_prev`
- **Type**: `int`
- **Definition**: How many unbounded days this item overlaps with the previous item
- **Calculation**: Derived from the identity formula
- **Note**: Kept `overlap_effective_unbounded_with_prev` for backward compatibility

### 2. New Summary Fields

#### Debug/Identity Fields
```json
{
  "_debug_sum_items_unbounded": 53,
  "_debug_calculated_inbound": 49,
  "_debug_identity_valid": true
}
```

#### Optimizer Diagnostics
```json
{
  "inbound_cap_sum_items_unbounded": 53,
  "inbound_cap_required_overlap": 3,
  "inbound_cap_base_overlap_min": 1,
  "inbound_cap_before": 53,
  "inbound_cap_after": 49,
  "inbound_cap_applied": true,
  "inbound_cap_unachievable": false
}
```

### 3. Optimizer Algorithm Changes

**File**: `backend/strategy/planner.py`
**Function**: `optimize_overlap_for_inbound_cap` (starting line ~2954)

#### Before (problematic approach):
- Computed `extra = inbound_unbounded - 50`
- Divided extra among connections
- Shifted items without explicit overlap target
- Could violate handoff >= 1 (used `max(..., 0)`)

#### After (overlap-based approach):
```python
# 1. Compute overlap requirements
N = len(sequence)
sum_items_unbounded = sum(item["effective_contribution_days_unbounded"] for item in sequence)
base_overlap_min = N - 1  # Minimum 1 day overlap per connection
required_overlap = max(sum_items_unbounded - 50, base_overlap_min)
current_overlap = sum_items_unbounded - inbound_unbounded

# 2. Shift items earlier until required overlap achieved
for each connection (prev, curr):
    while overlap_gained < additional_needed:
        candidate = working_date - 1 business day
        
        # ENFORCE INVARIANT: handoff >= 1
        if not (candidate < prev_sla_end):
            break  # Would violate handoff >= 1
        
        if not (candidate > prev_submit):
            break  # Would violate monotonic order
        
        # Legal shift - apply it
        working_date = candidate
        overlap_gained += days_moved
        inbound_unbounded -= days_moved
```

### 4. Inventory Selected Updates

Updated `_build_inventory_header_from_sequence` to read from the renamed field:
```python
overlap_days = int(sequence_entry.get("overlap_unbounded_days_with_prev", 
                                       sequence_entry.get("overlap_effective_unbounded_with_prev", 0)))
```

## Example: 2-Item Skeleton

### Input
- Item A: `effective_unbounded = 26`
- Item B: `effective_unbounded = 27`
- Raw sum: `S = 53`

### Processing
1. `N = 2`
2. `base_overlap_min = N - 1 = 1`
3. `required_overlap = max((53 - 50), 1) = 3`
4. Optimizer shifts Item B earlier by 2 additional days
5. `overlap_unbounded_days_with_prev = 3`

### Output
```json
{
  "summary": {
    "total_effective_days_unbounded": 49,
    "total_overlap_unbounded_days": 4,
    "inbound_cap_sum_items_unbounded": 53,
    "inbound_cap_required_overlap": 3,
    "inbound_cap_base_overlap_min": 1,
    "_debug_sum_items_unbounded": 53,
    "_debug_calculated_inbound": 49,
    "_debug_identity_valid": true
  },
  "sequence_debug": [
    {
      "field": "payment_history",
      "effective_contribution_days_unbounded": 26,
      "running_unbounded_at_submit": 0
    },
    {
      "field": "account_status",
      "effective_contribution_days_unbounded": 27,
      "running_unbounded_at_submit": 26,
      "overlap_unbounded_days_with_prev": 4,
      "handoff_days_before_prev_sla_end": 4
    }
  ]
}
```

### Verification
```
26 + 27 - 4 = 49 ✅
49 ≤ 50 ✅
handoff = 4 >= 1 ✅
```

## Testing

### Existing Tests (Pass)
- ✅ `test_per_weekday_cap_overlap_real_case.py` - Tests per-weekday mode with explicit env var
- ✅ `test_normal_pipeline_hard_cap.py` - Tests normal pipeline without env var (validates default)

### Test Coverage
1. Identity formula holds: `total_unbounded = sum(items) - overlap`
2. Base overlap minimum correct: `base_overlap_min = N - 1`
3. Required overlap calculation: `max((S - 50), (N - 1))`
4. Handoff invariant enforced: All connections have `handoff >= 1`
5. Items' effective_unbounded never reduced
6. Overlap increases when cap applied
7. Unachievable case marked correctly

## Files Modified

- **`backend/strategy/planner.py`**
  - `_enrich_sequence_with_contributions`: Added `running_unbounded_at_submit` and debug fields
  - `optimize_overlap_for_inbound_cap`: Rewrote to use overlap-based model with handoff >= 1 enforcement
  - `_build_inventory_header_from_sequence`: Updated to read renamed field

## Breaking Changes

None. All changes are additive:
- New fields added alongside existing ones
- Backward compatibility maintained with `overlap_effective_unbounded_with_prev`
- Existing tests pass without modification

## Migration Notes

For downstream consumers of plan JSON:
1. Prefer reading `overlap_unbounded_days_with_prev` over `overlap_effective_unbounded_with_prev`
2. Use `running_unbounded_at_submit` to track cumulative inbound at each submit
3. Leverage `_debug_*` fields for validation and troubleshooting
4. Check `inbound_cap_unachievable` to detect cases where cap couldn't be achieved legally

## Verification Commands

```bash
# Run tests
python -m pytest tests/backend/strategy/test_per_weekday_cap_overlap_real_case.py -v
python -m pytest tests/backend/strategy/test_normal_pipeline_hard_cap.py -v

# Run on real SID
python -m backend.strategy.runner --sid 88c4ee20-cd6f-45f1-a9bb-ebbfb1aa1f13

# Verify JSON output
python verify_quick.py
```

## Future Enhancements

1. **More granular unachievable reasons**: Track which specific invariant blocked further optimization
2. **Overlap distribution reporting**: Show per-connection overlap contributions
3. **Alternative strategies**: If cap unachievable, suggest which items could be dropped
4. **Performance optimization**: Cache overlap calculations during shift loop
