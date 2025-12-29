# Validation V2 Applied Invariant Fix

**Date**: 2025-11-17  
**SID**: d8919daf-20f7-4ee0-b0b9-aeaa55b5d0c0

## Problem

Manifest and runflow were inconsistent for validation V2 in orchestrator mode:

### Manifest (`ai.status.validation`)
```json
{
  "results_total": 3,
  "results_applied": 3,
  "results_apply_ok": true,
  "validation_ai_applied": true
}
```

### Runflow (`stages.validation`)
```json
{
  "status": "success",
  "expected_results": 3,
  "results_received": 3,
  "validation_ai_required": true,
  "validation_ai_completed": true,
  "validation_ai_applied": false
}
```

### Umbrella Barriers
```json
{
  "validation_ready": true
}
```

**The logical inconsistency:**
- `validation_ai_required = true`
- `validation_ai_applied = false`
- `validation.status = "success"`
- `umbrella_barriers.validation_ready = true`

This violates the invariant: **If AI validation is required, the stage cannot be "success" and "ready" unless `validation_ai_applied` is true.**

## Root Cause

1. **Refresh after apply didn't persist manifest V2 stats into runflow.**
   - `refresh_validation_stage_from_index` was calling `_apply_validation_stage_promotion`, which reads disk indices but does not directly inject manifest V2 fields.
   - Terminal-run guard in promotion path skipped updates when `run_state=AWAITING_CUSTOMER_INPUT` and status already `success`.

2. **Status computation did not enforce applied invariant.**
   - When `validation_ai_required=True` and `validation_ai_applied=False`, status could still be `success`.
   - Umbrella readiness computed from `_validation_stage_ready`, which checked `ready_latched` before checking applied.

3. **Normalization/merge could overwrite applied flag.**
   - Even when set in promotion, subsequent merge or normalization could discard it.

## Solution

### A. Enforce Hard Invariant in Status Computation

**File:** `backend/runflow/decider.py`

**Changes:**
1. In `_apply_validation_stage_promotion` (line ~3913):
   - After normalization and re-injection of `validation_ai_applied`, add a **hard invariant** check:
     - If `validation_ai_required=True` and `validation_ai_applied=False`, downgrade `status` to `"results_pending"` and set summary error hint.

2. In `refresh_validation_stage_from_index` (orchestrator V2 path):
   - Read `ai.status.validation` from manifest directly.
   - Write `expected_results`, `results_received`, and `validation_ai_applied` into runflow stage/metrics/summary.
   - Enforce status invariant **after** merging with latest snapshot:
     - If `required=True` and `applied=False`, set `status="results_pending"`.
     - If `required=True` and `applied=True` and counts match, set `status="success"`.

### B. Enforce Invariant in Readiness Check

**File:** `backend/runflow/decider.py`

**Changes:**
1. In `_validation_stage_ready`:
   - Check applied invariant **before** honoring `ready_latched`:
     - If `validation_ai_required=True` and `validation_ai_applied=False`, return `False`.
   - Fall back to manifest `ai.status.validation` to determine `validation_ai_applied` if stage/metrics/summary don't have it.
   - Require `applied >= total` when counts exist.

2. In `_compute_umbrella_barriers` (line ~5492):
   - After computing `validation_ready` from `_validation_stage_ready`, add a **double-check**:
     - Read `validation_ai_required` and `validation_ai_applied` from stage.
     - If `required=True` and `applied=False`, read manifest fallback.
     - If manifest confirms `applied=False`, **force `validation_ready=False`** regardless of latch or prior state.

### C. Manifest as Source of Truth in V2 Mode

**File:** `backend/runflow/decider.py`

**Changes:**
1. `refresh_validation_stage_from_index`:
   - In orchestrator mode, skip legacy promotion path entirely.
   - Read V2 stats from manifest: `results_total`, `results_applied`, `results_apply_ok`, `validation_ai_applied`.
   - Compute `applied_flag = apply_ok OR validation_ai_applied` with count validation.
   - Write into runflow stage/metrics/summary.
   - Merge with latest runflow, then enforce invariant post-merge.
   - Persist and log: `VALIDATION_STAGE_V2_REFRESH`.

## Testing

### Test Case 1: Applied = True
**Setup:**
- Manifest: `results_applied=3`, `validation_ai_applied=true`
- Run: `refresh_validation_stage_from_index(sid)` → `reconcile_umbrella_barriers(sid)`

**Expected:**
```
status: "success"
validation_ai_applied: true
validation_ready: true
```

**Result:** ✅ PASS

### Test Case 2: Applied = False
**Setup:**
- Manifest: `validation_ai_applied=false`, `results_apply_ok=false`
- Run: `refresh_validation_stage_from_index(sid)` → `reconcile_umbrella_barriers(sid)`

**Expected:**
```
status: "results_pending"
validation_ai_applied: false
validation_ready: false
```

**Result:** ✅ PASS

### Test Case 3: Verification Script
**Command:**
```bash
python verify_applied_flag.py d8919daf-20f7-4ee0-b0b9-aeaa55b5d0c0
```

**Result:**
```
✅ VALIDATION PASSED: All fields consistent!
```

## Invariant Guarantees

After this fix, the following invariants are **guaranteed**:

1. **Status Invariant:**
   - If `validation_ai_required=True` and `validation_ai_applied=False`, then `status != "success"`.
   - Status will be downgraded to `"results_pending"`.

2. **Readiness Invariant:**
   - If `validation_ai_required=True` and `validation_ai_applied=False`, then `validation_ready = False`.
   - This overrides any `ready_latched` flag.

3. **Manifest Priority:**
   - In orchestrator mode, manifest `ai.status.validation` is the authoritative source for `validation_ai_applied`.
   - Runflow refresh reads from manifest and persists into runflow.

4. **Umbrella Consistency:**
   - Umbrella barriers recompute `validation_ready` from `_validation_stage_ready`, which checks the applied invariant.
   - If stage/metrics/summary say applied=False, umbrella reads manifest fallback to confirm.

## Files Modified

- `backend/runflow/decider.py`:
  - `_validation_stage_ready`: Added applied invariant check before latch.
  - `_apply_validation_stage_promotion`: Added hard invariant after normalization.
  - `refresh_validation_stage_from_index`: V2 orchestrator path reads manifest, writes to runflow, enforces invariant post-merge.
  - `_compute_umbrella_barriers`: Added manifest fallback check for applied invariant enforcement.

## Verification

Run verification script for any SID:
```bash
python verify_applied_flag.py <sid>
```

Expected output when consistent:
```
✅ VALIDATION PASSED: All fields consistent!
```

## Summary

The fix ensures that:
- **Manifest is source of truth** for V2 apply status.
- **Status cannot be "success"** when required but not applied.
- **Validation_ready is False** when required but not applied.
- **Runflow refresh updates status** based on manifest applied flag.
- **Umbrella recomputation enforces invariant** even when terminal-run or latched flags exist.

All tests pass. The invariant is now **hard-enforced** across the entire validation pipeline.
