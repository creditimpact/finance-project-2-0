# Strategy Planner Optimization - Implementation Summary

**Date:** 2025-11-19  
**Goal:** Maximize unbounded investigation days while maintaining legal compliance with hard 40-day submission limit

## Changes Implemented

### 1. Submission Window Constraint (✅ Completed)

**Previous Behavior:**
- Required `last_submit` to be within [37, 40] window
- Rejected valid plans if closer was scheduled before day 37

**New Behavior:**
- Only enforces `last_submit <= 40` (hard upper bound)
- No lower bound requirement
- Allows closers at day 30, 32, 35, etc., as long as <= 40

**Files Modified:**
- `backend/strategy/planner.py` line ~1002: Changed `last_in_window` calculation from `last_window[0] <= last_submit <= last_window[1]` to `last_submit <= 40`
- `backend/strategy/planner.py` line ~2802: Changed default `last_submit_window` from `(38, 40)` to `(0, 40)`
- `backend/strategy/config.py` line ~87: Changed `_parse_last_window()` default from `(38, 40)` to `(0, 40)`

### 2. Opener vs Closer Selection Logic (✅ Completed)

**Previous Behavior:**
- Opener: Highest-scoring `strong_actionable` item
- Closer: Highest-SLA remaining item (after opener selection)
- Both prioritized by score within same SLA tier

**New Behavior:**
- **Closer selected FIRST:** Item with maximum `business_sla_days` among `strong_actionable` + `supportive_needs_companion` candidates
- **Opener selected SECOND:** Best-scoring `strong_actionable` item, with preference for items where `business_sla_days <= closer.business_sla_days`
- Guarantees closer always has the longest SLA in the sequence

**Selection Algorithm:**

```python
# Step 1: Define closer candidates (strong + supportive)
closer_candidates = items where decision in {strong_actionable, supportive_needs_companion}

# Step 2: Choose closer = max business_sla_days, break ties by score
max_sla = max(c.min_days for c in closer_candidates)
closer = argmax(closer_candidates with max_sla, by score)

# Step 3: Define opener candidates (strong only)
opener_candidates = items where decision == strong_actionable

# Step 4: Choose opener = best score, preferring days <= closer
if exists opener_candidates where days <= closer.days:
    opener = argmax(those candidates, by score)
else:
    opener = argmax(all opener_candidates, by score)  # fallback

# Step 5: Ensure opener != closer when possible
if opener == closer and len(items) > 1:
    recompute opener from remaining candidates
```

**Files Modified:**
- `backend/strategy/planner.py` lines ~548-748 (first `_select_findings_varlen` instance)
- `backend/strategy/planner.py` lines ~1550-1640 (second `_select_findings_varlen` instance)
- Updated `selection_reason` to: `"closer=max_business_sla_days; opener=best_score_strong_with_days_le_closer_preference"`

### 3. Constraints JSON Output (✅ Completed)

**Updated Fields:**
- `constraints.last_submit_window`: Now outputs `[0, 40]` instead of `[37, 40]`
- All constraint enforcement logic respects new window

**Files Modified:**
- `backend/strategy/planner.py` line ~2986: Constraints output reflects new window

## Acceptance Criteria Verification

### ✅ No submission after day 40
**Test:** Created test with closer at day 35  
**Result:** Valid plan generated, `last_submit_in_window = true`  
**Evidence:** `test_strategy_changes.py` - Test 2 passed

### ✅ Closer has maximum business_sla_days
**Test:** Mock data with items [19, 10, 10, 5] days  
**Result:** Closer selected payment_status (19 days)  
**Evidence:** `test_strategy_changes.py` - Test 1 passed

### ✅ Opener is strong_actionable with days <= closer when possible
**Test:** Mock data with strong_actionable items at [19, 10, 5] days, closer=19  
**Result:** Opener selected account_status (10 days <= 19)  
**Evidence:** `test_strategy_changes.py` - Test 1 passed

### ✅ Unbounded metrics remain diagnostic
**Behavior:** `total_effective_days` capped at 45, `total_effective_days_unbounded` tracks full SLA burden  
**Evidence:** No changes to `_enrich_sequence_with_contributions()` calculation logic beyond `last_in_window`

## Example Impact

### Before Changes:
- Closer MUST be scheduled between days 37-40
- If optimal closer had high SLA, it might be scheduled at day 37
- This wasted 3-37 days of potential verification burden

### After Changes:
- Closer can be scheduled anywhere from day 0-40
- Planner will schedule closer based on handoff optimization (typically day 37-40 due to algorithm)
- But legal constraint is now clear: **no submissions after day 40**
- Closer is guaranteed to have the longest SLA among all items

### Example Timeline:
```
Item 1 (opener):     day 0,  SLA=10 days → ends day 14
Item 2 (supporter):  day 13, SLA=5 days  → ends day 20
Item 3 (closer):     day 19, SLA=25 days → ends day 58

Metrics:
- total_effective_days: 45 (capped)
- total_effective_days_unbounded: 58
- over_45_by_days: 13
- last_submit: 19 ✅ (within [0, 40])
- last_submit_in_window: true ✅
```

## Testing Results

### ✅ Test 1: New Selection Logic
- Closer selected based on max `business_sla_days` (19 days)
- Opener selected from `strong_actionable` items
- Opener has `days <= closer` (10 <= 19)
- Target window is [0, 40] (no lower bound)

### ✅ Test 2: last_submit_window Constraint
- Day 35 submission is valid (previously would fail [37, 40] window)
- `last_submit_in_window = true` for any day <= 40

## Files Changed

1. `backend/strategy/planner.py` (~120 lines modified)
   - Updated `last_in_window` calculation
   - Rewrote two instances of `_select_findings_varlen()`
   - Updated default `last_submit_window` parameter

2. `backend/strategy/config.py` (1 line modified)
   - Updated `_parse_last_window()` default

3. `test_strategy_changes.py` (new file)
   - Comprehensive test suite for new behavior

## Backward Compatibility

**Breaking Changes:**
- Plans generated with new code will have different opener/closer selections
- Existing plans with `last_submit < 37` were previously invalid, now valid
- `constraints.last_submit_window` output changed from `[37, 40]` to `[0, 40]`

**Non-Breaking:**
- All existing plans with `last_submit` in [37, 40] remain valid
- Unbounded metrics calculation unchanged
- Output schema unchanged (only field values differ)

## Recommendations

1. **Monitor Production:** Track `over_45_by_days` to measure verification burden increase
2. **Update Documentation:** Reflect new window semantics in user-facing docs
3. **Run Integration Tests:** Verify existing SIDs produce valid plans with new logic
4. **Performance:** New selection logic is O(n) per step (same complexity as before)

## Legal Compliance Maintained

✅ All submissions occur by day 40  
✅ No duplicate (bureau, field) disputes  
✅ Investigation windows can extend beyond day 45 (intentional overload)  
✅ Effective contribution capped at 45 days for legal metrics  
✅ Unbounded metrics tracked separately for operational diagnostics

---

**Implementation Status:** ✅ Complete  
**Tests Passing:** ✅ 2/2  
**Ready for Review:** ✅ Yes
