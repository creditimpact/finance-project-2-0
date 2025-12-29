# Test Fixes Summary - Deadline Semantics Implementation

## Overview
After implementing the new deadline semantics (≤40 instead of [37,40] window), several existing tests had outdated expectations that needed updating to reflect the new behavior.

## Test Files Modified

### 1. `test_summary_block.py`
**Issue**: Test referenced deprecated log field name `in_window_37_40`
**Fix**: Updated to use new field name `deadline_satisfied`
**Lines Changed**: Line checking `summary_event["in_window_37_40"]`

### 2. `test_inventory_header.py` (5 fixes)

#### Fix 1: `test_closer_is_max_sla_remaining`
**Issue**: Expected `closers_eligible` count to be `len(inventory_all) - 1`
**Actual**: Now correctly `len(inventory_all)` (all items are closer candidates)
**Reason**: New logic doesn't exclude opener from closer candidate pool initially

#### Fix 2: `test_no_strongs_still_picks_single_opener_and_closer` (3 changes)
**Issue 1**: Expected opener to have max SLA
**Actual**: Closer has max SLA (15), opener has second-best (11)
**Fix**: Updated assertions to verify closer=max_sla and opener≤closer

**Issue 2**: Wrong reason string format
**Actual**: `"closer=max_business_sla_days; opener=best_score_strong_with_days_le_closer_preference"`
**Fix**: Updated expected reason string

**Issue 3**: Wrong `closers_eligible` count
**Fix**: Updated to expect `len(inventory_all)` instead of `len(inventory_all) - 1`

#### Fix 3: `test_closer_scheduled_in_window`
**Issue**: Expected closer in legacy window [37,40]
**Actual**: Closer scheduled at day 25 (still satisfies ≤40)
**Fix**: Changed assertion from `assert 37 <= closer_day <= 40` to `assert closer_day <= 40`
**Added Comment**: Explaining new deadline semantics (≤40 instead of [37,40])

#### Fix 4: `test_closer_window_adjustment_when_blocked`
**Issue**: Expected `planner_impossible_window` error when [37,40] blocked by holidays
**Actual**: Planner succeeds by scheduling earlier (before day 37, still ≤40)
**Fix**: 
- Removed expectation of "planner_impossible_window" reason
- Removed expectation of "timeline_warning" and "closer_window_adjust" log events
- Changed closer assertion from `assert not (37 <= day <= 40)` to `assert day < 37 and day <= 40`
**Reason**: Under new semantics, blocking [37,40] doesn't prevent planning - items can be scheduled earlier

## Summary of Changes

### Behavioral Changes Reflected in Tests:
1. **Deadline Check**: Changed from "must be in [37,40]" to "must be ≤40"
2. **Log Field Names**: `in_window_37_40` → `deadline_satisfied`
3. **Closer Candidates**: All items are now initially considered as closer candidates
4. **Window Blocking**: Blocking [37,40] no longer causes planning failure
5. **Reason Strings**: Updated to reflect new selection logic format

### Test Results:
- **Before fixes**: 5 failures, 35 passed
- **After fixes**: 40 passed, 0 failures ✅

### Files Modified:
1. `tests/backend/strategy/test_summary_block.py` - 1 change
2. `tests/backend/strategy/test_inventory_header.py` - 7 changes across 4 tests

## Verification
All strategy tests now pass with new deadline semantics:
```
tests/backend/strategy/ - 40 passed in 1.01s
```

This includes:
- 7 new deadline semantics tests (test_deadline_semantics.py)
- 33 existing tests updated for compatibility
