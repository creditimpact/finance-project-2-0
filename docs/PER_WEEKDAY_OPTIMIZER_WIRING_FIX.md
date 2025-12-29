# Per-Weekday Optimizer Wiring Fix Summary

## Problem Statement

SID `4548e957-5918-4b06-97ce-ce50e75830a2` showed `unbounded=51` with **no metadata fields** despite Phase 2 implementation of the hard cap optimizer. Manual testing proved the optimizer logic worked correctly (51→50 with metadata), indicating a **wiring issue** in the serialization path.

## Root Cause Analysis

### Discovery Process

1. **Baseline Verification**: Read `plan_wd0.json` for account 9:
   - Confirmed `total_effective_days_unbounded: 51`
   - No `inbound_cap_*` metadata fields present
   - Opener: 24 days, Closer: 27 days (Thu 2025-12-18)

2. **Manual Optimizer Test**: Created `test_manual_optimizer_sid_4548e957.py`:
   - Loaded actual plan JSON from SID
   - Called `optimize_overlap_for_inbound_cap` directly
   - **Result**: Successfully reduced 51→50 with all metadata fields
   - **Conclusion**: Optimizer logic works correctly in isolation

3. **Serialization Path Trace**: Examined `compute_optimal_plan` in `planner.py`:
   - Lines 3708-3715: Found optimizer call **only when `best_weekday_enabled=True`**
   - For per-weekday mode (`STRATEGY_BEST_WEEKDAY_ENABLED=0`), optimizer was never invoked
   - Bug: Each weekday plan written to disk without post-processing

### Root Cause

**The optimizer was only applied to the best-weekday plan in joint-optimize mode, but NOT to individual weekday plans in per-weekday mode.**

```python
# OLD CODE (lines 3708-3715)
# Phase: Inbound-cap post-processing for the chosen best plan (only when best-weekday is enabled)
if best_weekday_enabled and not enforce_span_cap:
    unbounded_val = best_plan.get("summary", {}).get("total_effective_days_unbounded")
    try:
        needs_cap = unbounded_val is not None and int(unbounded_val) > 50
    except (TypeError, ValueError):
        needs_cap = False
    if needs_cap:
        best_plan = optimize_overlap_for_inbound_cap(...)
        weekday_plans[best_weekday] = best_plan
```

## Implementation Fix

### Code Changes

**File**: `backend/strategy/planner.py`

**Change 1**: Moved `best_weekday_enabled` resolution earlier (after line 3602)
- Needed to determine mode before per-weekday optimization loop
- Removed duplicate resolution block at line 3675

**Change 2**: Added per-weekday optimizer loop (lines 3603-3625):
```python
# Phase: Inbound-cap post-processing for per-weekday mode (when best-weekday is disabled)
# Apply optimizer to ALL weekday plans independently before best-weekday selection
if not best_weekday_enabled and not enforce_span_cap:
    for weekday in list(weekday_plans.keys()):
        plan = weekday_plans[weekday]
        unbounded_val = plan.get("summary", {}).get("total_effective_days_unbounded")
        try:
            needs_cap = unbounded_val is not None and int(unbounded_val) > 50
        except (TypeError, ValueError):
            needs_cap = False
        if needs_cap:
            optimized_plan = optimize_overlap_for_inbound_cap(
                deepcopy(plan),
                max_unbounded_inbound_day=50,
                weekend=weekend_set,
                holidays=holidays_set,
                enforce_span_cap=enforce_span_cap,
                include_notes=include_notes,
            )
            weekday_plans[weekday] = optimized_plan
```

**Change 3**: Retained existing best-weekday optimizer call (lines 3736-3747)
- Ensures joint-optimize mode still works
- No changes to existing behavior when `best_weekday_enabled=True`

## Test Coverage

### Regression Test Added

**File**: `tests/backend/strategy/test_inbound_cap_and_minimal_disputes.py`

**Test**: `test_per_weekday_mode_applies_optimizer_to_all_plans`
- Uses exact SID scenario: opener=24, closer=27 → 51 unbounded
- Sets `STRATEGY_BEST_WEEKDAY_ENABLED=0` via context manager
- Asserts:
  - All weekday plans have `unbounded ≤ 50`
  - Weekday 0 (Monday anchor) shows metadata: `inbound_cap_applied=True`, `before=51`, `after=50`

### Test Results

```
tests/backend/strategy/test_inbound_cap_and_minimal_disputes.py
  test_inbound_cap_adjusts_and_preserves_invariants     PASSED
  test_no_adjust_when_unbounded_leq_50                  PASSED
  test_two_items_preferred_over_three_when_core_met     PASSED
  test_inbound_cap_optimizer_applies_and_caps_unbounded PASSED
  test_minimal_item_strategy_skeleton_1                 PASSED
  test_minimal_item_strategy_skeleton_2_and_3           PASSED
  test_weekend_step_accepts_underrun                    PASSED
  test_per_weekday_mode_applies_optimizer_to_all_plans  PASSED

8 passed in 0.10s
```

## Validation

### Manual Test Script

Created `test_manual_optimizer_sid_4548e957.py` for ad-hoc verification:
- Loads actual plan from SID 4548e957
- Calls optimizer directly with correct parameters
- Asserts: unbounded ≤50, metadata present, `before=51`, `after=50`
- Output confirms optimizer works correctly

### Expected Behavior After Fix

When re-running SID `4548e957-5918-4b06-97ce-ce50e75830a2` with `STRATEGY_BEST_WEEKDAY_ENABLED=0`:

**Before fix**:
```json
{
  "summary": {
    "total_effective_days_unbounded": 51
  }
}
```

**After fix**:
```json
{
  "summary": {
    "total_effective_days_unbounded": 50,
    "inbound_cap_hard": true,
    "inbound_cap_target": 50,
    "inbound_cap_before": 51,
    "inbound_cap_after": 50,
    "inbound_cap_applied": true
  }
}
```

## Files Modified

1. **`backend/strategy/planner.py`**:
   - Lines 3603-3625: Added per-weekday optimizer loop
   - Lines 3604-3613: Moved `best_weekday_enabled` resolution earlier
   - Line 3688: Removed duplicate flag resolution

2. **`tests/backend/strategy/test_inbound_cap_and_minimal_disputes.py`**:
   - Lines 287-314: Added `test_per_weekday_mode_applies_optimizer_to_all_plans`

3. **`test_manual_optimizer_sid_4548e957.py`** (new file):
   - Manual testing script for ad-hoc validation

## Deployment Notes

- No configuration changes required
- No database migrations
- No API contract changes
- Backward compatible: best-weekday mode unchanged
- Per-weekday mode now correctly applies hard cap to all plans

## Next Steps

1. **Verify fix in production**: Re-run SID 4548e957 and confirm `unbounded=50` with metadata
2. **Monitor for edge cases**: Track `inbound_cap_unachievable` flag in per-weekday runs
3. **Consider Phase 3 (optional)**: Minimal-disputes optimizer for Phase 1-2-3 sequence

---

**Status**: ✅ Complete
**Date**: 2025-01-22
**Author**: GitHub Copilot
**Severity**: High (production wiring bug)
**Impact**: All per-weekday runs now enforce hard ≤50 cap with metadata
