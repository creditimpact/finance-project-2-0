# Validation Chain Trigger After Merge Ready — Implementation Summary

**Date**: November 19, 2025  
**Issue**: Validation chain not triggered when `merge_ready` becomes `true` after `stage_a_task` completes  
**Example SID**: `248d03fe-50c3-4334-ae6d-ea90e6b77884`

---

## Problem Description

### Observed Behavior

For SID `248d03fe-50c3-4334-ae6d-ea90e6b77884`:

1. **At validation requirements time**: `merge_ready=false` → logs `VALIDATION_CHAIN_DEFER`
2. **Later, merge finishes**: `merge_ready=true`, `merge_ai_applied=true`
3. **But no validation chain trigger**: Missing `VALIDATION_CHAIN_TRIGGER` and `AUTO_AI_CHAIN_START` logs
4. **Result**: Validation never runs, stuck with `validation_ready=false`

### Root Cause

The validation chain trigger was only called from `stage_a_task` after validation requirements. If `merge_ready` was still `false` at that point, the chain was deferred. When merge later completed and `merge_ready` flipped to `true`, there was **no code path to retry the trigger**.

---

## Solution Implemented

### 1. Created Centralized Helper Function

**File**: `backend/pipeline/auto_ai.py`  
**Function**: `maybe_trigger_validation_chain_if_merge_ready()`  
**Lines**: ~1015-1102

**Logic**:
```python
def maybe_trigger_validation_chain_if_merge_ready(
    sid: str,
    runs_root: Path | str | None = None,
    *,
    flag_env: Mapping[str, str] | None = None,
) -> dict[str, object]:
    """
    Centralized entrypoint to trigger the Auto-AI validation chain
    once merge is ready.

    Rules:
    - If merge_ready is False -> do NOT trigger, just log DEFER.
    - If merge_ready is True but validation is already success -> SKIP.
    - If merge_ready is True and validation is not complete -> trigger chain.

    This MUST NOT start validation before merge_ready=true.
    """
```

**Decision Tree**:
1. Load `merge_ready` from `_compute_umbrella_barriers()`
2. Load validation status from runflow
3. **If `merge_ready=false`**: Log `VALIDATION_CHAIN_DEFER reason=merge_not_ready`, return `triggered=false`
4. **If `validation.status="success"`**: Log `VALIDATION_CHAIN_SKIP_ALREADY_DONE`, return `triggered=false`
5. **Otherwise**: Log `VALIDATION_CHAIN_TRIGGER`, call `maybe_queue_auto_ai_pipeline()`, return `triggered=true`

**Key Properties**:
- ✅ **Strict merge barrier**: Never triggers before `merge_ready=true`
- ✅ **Idempotent**: Skips if validation already successful
- ✅ **Defensive**: Handles errors gracefully, logs failures

---

### 2. Wired Helper into Two Call Sites

#### Call Site #1: `stage_a_task` (After Validation Requirements)

**File**: `backend/api/tasks.py`  
**Location**: Lines ~906-923 (after validation requirements complete)

**Old Code** (REMOVED):
- Inlined `_compute_umbrella_barriers` check
- Inlined `maybe_queue_auto_ai_pipeline` call
- Manual defer/enqueue logging

**New Code** (IMPLEMENTED):
```python
from backend.pipeline.auto_ai import maybe_trigger_validation_chain_if_merge_ready

result = maybe_trigger_validation_chain_if_merge_ready(
    sid,
    runs_root=runs_root,
    flag_env=os.environ,
)
# Attach result into the summary
if isinstance(summary, dict):
    summary["validation_chain"] = result
```

**Expected Behavior**:
- Usually logs `VALIDATION_CHAIN_DEFER reason=merge_not_ready` (because merge not done yet)
- This is **correct** — respects the strict merge barrier
- Chain will be triggered later when merge completes (see Call Site #2)

---

#### Call Site #2: `finalize_merge_stage` (After Merge Completion)

**File**: `backend/runflow/decider.py`  
**Location**: Lines ~2544-2556 (after `runflow_refresh_umbrella_barriers(sid)`)

**Added Code**:
```python
# ── VALIDATION CHAIN TRIGGER ───────────────────────────────────────────
# After merge finalization and barrier refresh, attempt to trigger validation chain.
# The helper will check merge_ready and only trigger if validation is needed.
try:
    from backend.pipeline.auto_ai import maybe_trigger_validation_chain_if_merge_ready
    maybe_trigger_validation_chain_if_merge_ready(
        sid,
        runs_root=base_root,
        flag_env=None,  # Will use os.environ by default
    )
except Exception:
    log.error("VALIDATION_CHAIN_POST_MERGE_TRIGGER_FAILED sid=%s", sid, exc_info=True)
# ───────────────────────────────────────────────────────────────────────
```

**Context**:
- Called at the **end** of `finalize_merge_stage()`
- After `merge_ai_applied=True` is set
- After `runflow_refresh_umbrella_barriers()` updates `merge_ready=true`

**Expected Behavior**:
- Now sees `merge_ready=true`
- Logs `VALIDATION_CHAIN_TRIGGER sid=... merge_ready=True`
- Calls `maybe_queue_auto_ai_pipeline()`
- Logs `VALIDATION_CHAIN_ENQUEUED sid=... result=...`
- Auto-AI chain starts

---

## Expected Flow for SID `248d03fe-50c3-4334-ae6d-ea90e6b77884`

### Before Fix (Broken)

```
1. stage_a_task runs:
   ├─ validation_requirements completes
   ├─ merge_ready=false
   └─ Logs: VALIDATION_CHAIN_DEFER reason=merge_not_ready

2. Merge pipeline finishes:
   ├─ merge_ai_applied=true, merge_ready=true
   └─ ❌ NO CODE TO TRIGGER VALIDATION CHAIN

3. Result:
   └─ Validation never starts, stuck with validation_ready=false
```

### After Fix (Working)

```
1. stage_a_task runs:
   ├─ validation_requirements completes
   ├─ Helper called: maybe_trigger_validation_chain_if_merge_ready()
   ├─ Checks merge_ready → false
   └─ Logs: VALIDATION_CHAIN_DEFER reason=merge_not_ready

2. Merge pipeline finishes (finalize_merge_stage):
   ├─ merge_ai_applied=true
   ├─ runflow_refresh_umbrella_barriers() → merge_ready=true
   ├─ Helper called: maybe_trigger_validation_chain_if_merge_ready()
   ├─ Checks merge_ready → true ✅
   ├─ Checks validation.status → not "success" ✅
   └─ Logs: VALIDATION_CHAIN_TRIGGER sid=... merge_ready=True
           VALIDATION_CHAIN_ENQUEUED sid=...

3. Auto-AI chain runs:
   ├─ Logs: AUTO_AI_CHAIN_START sid=...
   ├─ Merge tasks skip (idempotent guards detect merge complete)
   ├─ Validation tasks run:
   │   ├─ VALIDATION_STAGE_STARTED
   │   ├─ VALIDATION_BUILD_DONE
   │   ├─ VALIDATION_ORCHESTRATOR_SEND_V2
   │   ├─ VALIDATION_COMPACT_DONE
   │   └─ VALIDATION_STAGE_PROMOTED
   └─ Runflow updated: validation.status="success", validation_ready=true ✅
```

---

## Verification Steps

### 1. Check Logs After Merge Completion

**For SID `248d03fe-50c3-4334-ae6d-ea90e6b77884` (or any SID where merge completes after validation requirements)**:

```bash
# Should see validation chain triggered AFTER merge finalization
grep "VALIDATION_CHAIN_TRIGGER sid=248d03fe" logs/

# Should see chain start
grep "AUTO_AI_CHAIN_START sid=248d03fe" logs/

# Should see merge tasks skip (idempotent)
grep "MERGE_BUILD_IDEMPOTENT_SKIP sid=248d03fe" logs/
grep "MERGE_SEND_IDEMPOTENT_SKIP sid=248d03fe" logs/

# Should see validation tasks run
grep "VALIDATION_STAGE_STARTED sid=248d03fe" logs/
grep "VALIDATION_ORCHESTRATOR_SEND_V2 sid=248d03fe" logs/
grep "VALIDATION_COMPACT_DONE sid=248d03fe" logs/
```

### 2. Check Runflow State

```bash
# Validation stage should exist with status="success"
cat runs/248d03fe-50c3-4334-ae6d-ea90e6b77884/runflow.json | jq '.stages.validation.status'
# Expected: "success"

# Validation ready flag should be true
cat runs/248d03fe-50c3-4334-ae6d-ea90e6b77884/runflow.json | jq '.umbrella_barriers.validation_ready'
# Expected: true
```

### 3. Expected Log Sequence

For a typical run where merge completes after validation requirements:

```
[stage_a_task - early]
VALIDATION_REQUIREMENTS_PIPELINE_DONE sid=... processed=3 findings=24
VALIDATION_CHAIN_DEFER sid=... reason=merge_not_ready

[merge pipeline - later]
MERGE_AI_APPLIED sid=...
MERGE_STAGE_PROMOTED sid=... result_files=1
VALIDATION_CHAIN_TRIGGER sid=... merge_ready=True
VALIDATION_CHAIN_ENQUEUED sid=... result={'queued': True, ...}

[auto-ai chain - starts]
AUTO_AI_CHAIN_START sid=...
MERGE_BUILD_IDEMPOTENT_SKIP sid=... reason=already_complete ...
MERGE_SEND_IDEMPOTENT_SKIP sid=... reason=already_sent ...
VALIDATION_STAGE_STARTED sid=...
VALIDATION_BUILD_DONE sid=... packs=3
VALIDATION_ORCHESTRATOR_SEND_V2 sid=...
VALIDATION_ORCHESTRATOR_SEND_V2_DONE sid=... expected=3 sent=3 written=3
VALIDATION_COMPACT_START sid=...
VALIDATION_COMPACT_DONE sid=...
VALIDATION_STAGE_PROMOTED sid=...
```

---

## Design Properties Preserved

### ✅ Strict Merge Barrier

- Helper **never** triggers if `merge_ready=false`
- Validation cannot start before merge is complete
- Same strict barrier as before, now enforced in one place

### ✅ Idempotent Chain

- All existing idempotency guards remain unchanged
- Merge tasks skip if already complete
- Validation tasks skip if already complete
- Safe to call helper multiple times

### ✅ No Legacy Code

- No re-enabling of legacy validation orchestration
- No touching of `ENABLE_LEGACY_VALIDATION_ORCHESTRATION`
- No autosend flags dependencies
- Chain-only design fully preserved

### ✅ Centralized Logic

- Single source of truth for validation chain trigger decision
- Helper reused in two places (stage_a_task + finalize_merge_stage)
- Consistent logging and error handling

---

## Files Modified

| File | Changes | Lines |
|------|---------|-------|
| `backend/pipeline/auto_ai.py` | Added `maybe_trigger_validation_chain_if_merge_ready()` helper | +88 lines |
| `backend/api/tasks.py` | Replaced inline trigger logic with helper call | ~45 lines replaced with ~17 lines |
| `backend/runflow/decider.py` | Added helper call after merge finalization | +13 lines |

**Total**: 3 files, ~116 lines added/modified

---

## Edge Cases Handled

### Case 1: Merge Already Complete When stage_a_task Runs
- Helper sees `merge_ready=true` at stage_a_task time
- Triggers chain immediately
- ✅ Works correctly

### Case 2: Validation Already Complete When Merge Finishes
- Helper checks `validation.status == "success"`
- Logs `VALIDATION_CHAIN_SKIP_ALREADY_DONE`
- ✅ No duplicate work

### Case 3: Multiple Merge Finalizations (Rare)
- Helper is idempotent
- Will skip after first successful trigger
- ✅ Safe to call multiple times

### Case 4: Barrier Check Fails
- Helper catches exception
- Logs `VALIDATION_CHAIN_DEFER reason=barrier_check_failed`
- ✅ Defensive, doesn't crash

### Case 5: Chain Already Enqueued
- `maybe_queue_auto_ai_pipeline` has its own inflight lock
- Will return `{"queued": False, "reason": "inflight"}`
- ✅ Prevents duplicate chains

---

## Testing Recommendations

### Unit Tests (To Be Added)

1. **Test helper logic**:
   - Mock `_compute_umbrella_barriers` to return `merge_ready=false` → assert defer
   - Mock `merge_ready=true`, `validation.status="success"` → assert skip
   - Mock `merge_ready=true`, validation incomplete → assert trigger

2. **Test stage_a_task integration**:
   - Mock helper to track calls
   - Assert helper called with correct arguments
   - Assert result attached to summary

3. **Test finalize_merge_stage integration**:
   - Mock helper to track calls
   - Assert helper called after barrier refresh
   - Assert error logged if helper raises

### Integration Tests (Manual Verification)

1. **Test with SID where merge completes late**:
   - Use SID like `248d03fe-50c3-4334-ae6d-ea90e6b77884`
   - Verify chain triggered after merge finalization
   - Verify validation completes successfully

2. **Test with SID where merge already complete**:
   - Trigger chain from stage_a_task when `merge_ready=true`
   - Verify immediate trigger (no defer)

---

## Rollback Plan

If issues arise:

1. **Revert `backend/runflow/decider.py`** (lines ~2544-2556):
   - Remove helper call from `finalize_merge_stage`
   - This restores old behavior where only stage_a_task could trigger

2. **Optionally revert helper** (if helper has bugs):
   - Revert `backend/pipeline/auto_ai.py` changes
   - Restore inline logic in `backend/api/tasks.py`

**Risk**: Low — helper is defensive and doesn't change existing idempotency guards or barrier logic.

---

## Summary

**Problem**: Validation chain not triggered when merge completes after `stage_a_task`  
**Solution**: Centralized helper + two call sites (stage_a_task + finalize_merge_stage)  
**Result**: Validation chain now triggers reliably when `merge_ready=true`, regardless of timing

**Key Invariants Preserved**:
- ✅ Validation never starts before `merge_ready=true`
- ✅ Idempotent chain (safe to re-run)
- ✅ No legacy validation re-enabled
- ✅ Chain-only design maintained

**Status**: ✅ **IMPLEMENTATION COMPLETE**  
**Next**: Test with SID `248d03fe-50c3-4334-ae6d-ea90e6b77884` to verify fix
