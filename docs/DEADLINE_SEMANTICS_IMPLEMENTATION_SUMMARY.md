# Strategy Planner Deadline Semantics Update - Implementation Summary

## Overview
Successfully implemented a unified deadline rule for the strategy planner: **last_submit ≤ 40 and not on weekend**.

Removed legacy [37, 40] window-hit semantics that required plans to hit a specific target day within a range.

## Changes Made

### Phase 1: Clean Up last_submit_in_window Semantics

**File: backend/strategy/planner.py**

1. **_force_closer_into_window (line ~2780)**
   - **REMOVED**: Line that overwrote `last_submit_in_window` using `[window_start, window_end]` bounds
   - **ADDED**: Comment explaining that `last_submit_in_window` is set by enrichment using ≤40 rule
   - **CHANGED**: `within_window_flag` now reads from enriched summary instead of recomputing

2. **_enrich_sequence_with_contributions (line ~1033)**
   - **VERIFIED**: Already correctly implements: `last_in_window = last_submit <= 40 and last_weekday not in weekend`
   - No changes needed - this was the canonical source of truth

### Phase 2: Replace Legacy window_hit Logic

**File: backend/strategy/planner.py**

3. **pack_sequence_to_target_window (line ~1265)**
   - **REPLACED**: `window_hit = window_start <= closer_index <= window_end`
   - **WITH**: `last_submit_ok = closer_index <= 40 and closer_weekday not in weekend_set`
   - **UPDATED**: Scoring tuple to use `last_submit_ok` instead of `window_hit`
   - **UPDATED**: Distance calculation changed from `abs(window_end - closer_index)` to `abs(40 - closer_index)`
   - **RENAMED**: `candidate_meta["window_hit"]` to `candidate_meta["deadline_satisfied"]`

4. **_plan_for_weekday (line ~2593)**
   - **RENAMED**: Log event from `"planner_no_day_37_40"` to `"planner_no_suitable_target_day"`

5. **Master plan selection (line ~3320)**
   - **REPLACED**: Selection key based on window midpoint
   - **WITH**: New selection key prioritizing:
     1. `last_submit_in_window` (≤40 deadline satisfied)
     2. Higher `total_effective_days`
     3. Proximity to 40 (not to window midpoint)
   - **CHANGED**: Success candidates now filter on enriched `last_submit_in_window` flag

6. **Timeline logs (line ~3609)**
   - **RENAMED**: `"in_window_37_40"` to `"deadline_satisfied"`

7. **Metadata references**
   - **UPDATED**: All references to `packing_meta.get("window_hit")` changed to `packing_meta.get("deadline_satisfied")`
   - **UPDATED**: Fallback metadata return changed from `{"window_hit": False}` to `{"deadline_satisfied": False}`

### Phase 3: Ensure inventory_selected Consistency

**File: backend/strategy/planner.py**

8. **best_sequence assignment (line ~3357)**
   - **ADDED**: Defensive assertion: `assert isinstance(best_sequence, list)`
   - **ADDED**: Comment clarifying that `best_sequence` comes from final enriched `best_plan`

9. **inventory_selected builder (line ~3390)**
   - **ADDED**: Comment documenting consistency guarantee between `inventory_selected` and `sequence_debug`

### Phase 4: Supporters Acceptance

**File: backend/strategy/planner.py**

10. **_evaluate_sequence_for_selection** 
    - No code changes needed
    - This function returns `success` flag from `_plan_for_weekday`
    - With Phase 2 changes, `success` now based on ≤40 rule via `packing_meta["deadline_satisfied"]`
    - Supporters are now accepted/rejected based on whether adding them maintains `last_submit ≤ 40 and not weekend`

### Phase 5: Documentation & Constraints

**File: backend/strategy/planner.py**

11. **Module docstring (line ~1)**
    - **ADDED**: Comprehensive documentation explaining:
      - New deadline semantics (≤40, not weekend)
      - Legacy window parameter kept for backward compatibility
      - Consistency guarantee across all output views

12. **Constraints output (line ~3011)**
    - **ADDED**: Comment clarifying that `last_submit_window` only enforces upper bound
    - Lower bound is deprecated and not enforced

### Phase 6: Tests

**File: tests/backend/strategy/test_deadline_semantics.py (NEW)**

13. **Created comprehensive test suite** with 7 tests:
    - `test_inventory_selected_consistency`: Validates inventory_selected matches sequence_debug exactly
    - `test_deadline_satisfied_below_40`: Confirms plans with last_submit ≤ 40 are accepted
    - `test_deadline_not_satisfied_above_40`: Confirms plans with last_submit > 40 are rejected
    - `test_supporter_acceptance_within_deadline`: Validates supporters accepted when maintaining ≤40
    - `test_no_legacy_window_hit_requirement`: Confirms no lower bound requirement
    - `test_opener_closer_rules_preserved`: Validates opener/closer selection unchanged
    - `test_timeline_log_renamed`: Confirms legacy event names removed

All tests **PASS** ✅

## Key Behavioral Changes

### Before (Legacy)
- Plans required last_submit to fall within `[window_start, window_end]` range (typically [37, 40])
- Specific target day hit was required
- Plans with last_submit < 37 were marked as invalid
- Supporters rejected if they shifted closer off target day

### After (New)
- Plans only require `last_submit ≤ 40` and not on weekend
- No lower bound enforced
- No specific target day requirement
- Supporters accepted as long as they maintain ≤40 deadline
- More plans are valid, leading to better coverage

## Consistency Guarantees

The following are now guaranteed to be consistent:
- `inventory_header.inventory_selected[i].planned_submit_index` == `sequence_debug[i].calendar_day_index`
- `inventory_header.inventory_selected[i].planned_submit_date` == `sequence_debug[i].submit.date`
- `inventory_header.inventory_selected[i].effective_contribution_days` == `sequence_debug[i].effective_contribution_days`
- All effective days and running totals match across views

## Preserved Behavior

The following rules remain UNCHANGED:
- ✅ Closer = field with max `business_sla_days` (including supporters)
- ✅ Opener = strongest `strong_actionable` with `business_sla_days ≤ closer`
- ✅ SLA overlap enforced via `handoff_days_before_prev_sla_end ≥ 1`
- ✅ Timeline `from_day/to_day` are non-overlapping effective contribution windows
- ✅ Enrichment computes effective days correctly

## Log/Event Name Changes

| Old Name | New Name | Purpose |
|----------|----------|---------|
| `planner_no_day_37_40` | `planner_no_suitable_target_day` | Remove hardcoded window numbers |
| `in_window_37_40` | `deadline_satisfied` | Clarify semantic meaning |
| `window_hit` (metadata) | `deadline_satisfied` (metadata) | Align with new semantics |

## Backward Compatibility

- `last_submit_window` parameter still accepted (for compatibility)
- Only upper bound (typically 40) is enforced
- Lower bound is ignored
- Existing callers will see improved behavior (more valid plans)

## Testing

All tests pass (40 total) ✅, including:

**New Tests** (`test_deadline_semantics.py` - 7 tests):
- Deadline semantics
- Consistency guarantees
- Supporter acceptance
- Opener/closer rules
- Log event renaming

**Existing Tests** (33 tests):
- All passing after updating 5 tests with outdated expectations
- See `TEST_FIXES_SUMMARY.md` for details

## Files Modified

1. `backend/strategy/planner.py` - Core implementation (~12 changes)
2. `tests/backend/strategy/test_deadline_semantics.py` - New test suite (7 tests, all passing)
3. `tests/backend/strategy/test_summary_block.py` - Updated log field name
4. `tests/backend/strategy/test_inventory_header.py` - Updated 5 test expectations
5. `TEST_FIXES_SUMMARY.md` - Documentation of test updates

## Summary

**We now use a single deadline rule: last_submit ≤ 40 and not weekend.**

Legacy window [37, 40] hit semantics are fully removed. All code paths (scoring, selection, enrichment, output) are unified around this rule.
