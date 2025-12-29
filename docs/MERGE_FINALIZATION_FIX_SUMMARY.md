# Merge Finalization & Validation Gating Fix - Implementation Summary

**Date:** 2025-01-15  
**Bug:** Validation could start/complete before merge AI results were fully applied to runflow  
**Root Cause:** `merge_ready` barrier opened when result files appeared on disk, not when results applied  
**Solution:** New `merge_ai_applied` flag marks when merge results are truly finalized  

---

## 🎯 Overview

This implementation fixes the timing bug discovered in SID `83830ae4-6406-4a7e-ad80-a3f721a3787b`, where validation completed 8 seconds BEFORE merge was ready. The bug occurred because:

1. **Old behavior:** `merge_ready = True` when result files exist on disk
2. **Problem:** Files appear BEFORE `finalize_merge_stage()` applies them to runflow
3. **Result:** Validation could read incomplete/stale merge data

**New behavior:** Non-zero-packs cases require `merge_ai_applied=True` flag (set AFTER results applied).  
**Zero-packs fast path:** Unchanged - `empty_ok` still bypasses all merge checks instantly.

---

## 📝 Implementation Tasks (All Completed)

### ✅ Task 1: Add `merge_ai_applied` Flag in finalize_merge_stage

**File:** `backend/runflow/decider.py`  
**Location:** After `record_stage()` call in `finalize_merge_stage` function (~line 2491)

**Changes:**
```python
# After record_stage() completes successfully
runflow_data = _load_runflow(sid)
if runflow_data is not None:
    stages = _ensure_stages_dict(runflow_data)
    merge_stage = stages.get("merge")
    if isinstance(merge_stage, dict):
        merge_stage["merge_ai_applied"] = True
        merge_stage["merge_ai_applied_at"] = _now_iso()
        _save_runflow(sid, runflow_data)
        log.info("MERGE_AI_APPLIED sid=%s", sid)
```

**Purpose:** Mark the exact moment when merge AI results are fully applied to runflow.

---

### ✅ Task 2: Update `_compute_umbrella_barriers` Logic

**File:** `backend/runflow/decider.py`  
**Location:** After merge_ready determination, before validation_ready logic (~line 3947)

**Changes:**
```python
# For non-zero-packs cases, require merge_ai_applied flag
if merge_ready and not merge_empty_ok:
    merge_ai_applied = merge_stage.get("merge_ai_applied", False) if isinstance(merge_stage, Mapping) else False
    if not merge_ai_applied:
        log.info(
            "MERGE_NOT_AI_APPLIED sid=%s merge_ready_disk=%s merge_empty_ok=%s",
            run_dir.name,
            merge_ready_disk,
            merge_empty_ok,
        )
        merge_ready = False
```

**Critical:** Zero-packs fast path unaffected - `merge_empty_ok` check happens FIRST.

---

### ✅ Task 3: Gate Validation Orchestrator on merge_ready

**File:** `backend/pipeline/validation_orchestrator.py`  
**Location:** Start of `run_for_sid()` method, after orchestrator mode check

**Changes:**
```python
# Import _compute_umbrella_barriers at top
from backend.runflow.decider import _compute_umbrella_barriers

# At start of run_for_sid()
barriers = _compute_umbrella_barriers(run_dir)
merge_ready = barriers.get("merge_ready", False)
if not merge_ready:
    logger.info(
        "VALIDATION_ORCHESTRATOR_DEFERRED sid=%s reason=merge_not_ready barriers=%s",
        sid,
        barriers,
    )
    return {"sid": sid, "deferred": True, "reason": "merge_not_ready"}
```

**Effect:** Validation cannot start until merge is fully ready.

---

### ✅ Task 4: Defensive Checks

**Files:**
- `backend/runflow/decider.py` - `_maybe_enqueue_validation_fastpath` function
- `devtools/run_validation_orchestrator.py`

**Changes:**

#### Fastpath Check (decider.py, ~line 740):
```python
# At start of _maybe_enqueue_validation_fastpath
merge_empty_ok = _stage_empty_ok(merge_stage)
if not merge_empty_ok:
    merge_ai_applied = merge_stage.get("merge_ai_applied", False)
    if not merge_ai_applied:
        log.info(
            "VALIDATION_FASTPATH_SKIP sid=%s reason=merge_not_ai_applied empty_ok=%s",
            sid,
            merge_empty_ok,
        )
        return False
```

#### Devtools Warning (run_validation_orchestrator.py):
```python
# Before running orchestrator
barriers = _compute_umbrella_barriers(run_dir)
merge_ready = barriers.get("merge_ready", False)
if not merge_ready:
    print(
        f"⚠️  WARNING: merge_ready=False for sid={sid}. "
        f"Validation may fail or produce incomplete results.",
        file=sys.stderr,
    )
```

---

### ✅ Task 5: Migration/Backfill Script

**File:** `scripts/repair_merge_ai_applied_from_run.py` (NEW)

**Purpose:** Backfill `merge_ai_applied=True` for existing completed runs.

**Usage:**
```bash
# Dry run (preview)
python scripts/repair_merge_ai_applied_from_run.py --runs-root runs --dry-run

# All runs
python scripts/repair_merge_ai_applied_from_run.py --runs-root runs

# Single SID
python scripts/repair_merge_ai_applied_from_run.py --runs-root runs --sid 83830ae4-6406-4a7e-ad80-a3f721a3787b
```

**Logic:**
- Finds merge stages with `status=success` and `empty_ok=False` (non-zero-packs)
- Checks if `merge_ai_applied` is missing
- Sets flag and reconciles barriers
- Skips zero-packs cases (don't need repair)

---

### ✅ Task 6: Tests

**File:** `backend/tests/test_merge_ai_applied_fix.py` (NEW)

**Test Coverage:**

1. **Zero-packs fast path unaffected**
   - Verifies `merge_ready=True` even without `merge_ai_applied` when `empty_ok=True`

2. **Non-zero-packs requires flag**
   - Verifies `merge_ready=False` when `merge_ai_applied` is missing

3. **Non-zero-packs with flag ready**
   - Verifies `merge_ready=True` when `merge_ai_applied=True`

4. **Orchestrator gating**
   - Defers when `merge_ready=False`
   - Proceeds when `merge_ready=True`

5. **Regression test (SID 83830ae4 pattern)**
   - Reproduces exact bug scenario
   - Verifies fix prevents premature validation

**Run tests:**
```bash
pytest backend/tests/test_merge_ai_applied_fix.py -v
```

---

### ✅ Task 7: Prove Fix on Real SIDs

**File:** `scripts/prove_merge_fix.py` (NEW)

**Test SIDs:**
- `83830ae4-6406-4a7e-ad80-a3f721a3787b`: **Bug case** (validation before merge)
- `61e8cb38-8a58-42e3-9477-58485d43cb52`: **Zero-packs case** (should be unaffected)
- `9d4c385b-4688-46f1-b545-56ebb5ffff06`: **Healthy case** with packs

**Usage:**
```bash
# Test all 3 SIDs
python scripts/prove_merge_fix.py --runs-root runs

# Test specific SID
python scripts/prove_merge_fix.py --runs-root runs --sid 83830ae4-6406-4a7e-ad80-a3f721a3787b
```

**Output:**
- Before/after snapshots (JSON)
- Verification checks (pass/fail)
- Detailed results saved to `runs/merge_fix_verification_results.json`

---

## 🔍 Verification Plan

### Step 1: Run Backfill Script
```bash
# Dry run first to see what would change
python scripts/repair_merge_ai_applied_from_run.py --runs-root runs --dry-run

# Apply fix
python scripts/repair_merge_ai_applied_from_run.py --runs-root runs
```

### Step 2: Run Proof Script
```bash
python scripts/prove_merge_fix.py --runs-root runs
```

**Expected Results:**

#### Bug SID (83830ae4):
- ✅ `merge_ai_applied` set to `True`
- ✅ `merge_ready` now `True`
- ✅ Validation status unchanged (already completed)

#### Zero-packs SID (61e8cb38):
- ✅ `empty_ok=True` confirmed
- ✅ `merge_ready` was and remains `True`
- ✅ No backfill needed (skipped)

#### Healthy SID (9d4c385b):
- ✅ `merge_ai_applied=True` (either already present or set by backfill)
- ✅ `merge_ready=True`

### Step 3: Run Unit Tests
```bash
pytest backend/tests/test_merge_ai_applied_fix.py -v
```

**Expected:** All 6 tests pass.

---

## 📊 Key Log Messages

### Success Path (Non-Zero-Packs):
```
MERGE_AI_APPLIED sid=<sid>
MERGE_STAGE_PROMOTED sid=<sid> result_files=<n>
MERGE_AI_APPLIED_REPAIRED sid=<sid> before={...} after={...}
```

### Gating/Deferral:
```
MERGE_NOT_AI_APPLIED sid=<sid> merge_ready_disk=True merge_empty_ok=False
VALIDATION_ORCHESTRATOR_DEFERRED sid=<sid> reason=merge_not_ready
VALIDATION_FASTPATH_SKIP sid=<sid> reason=merge_not_ai_applied
```

### Zero-Packs Fast Path (Unchanged):
```
UMBRELLA_MERGE_OPTIONAL sid=<sid> reason=empty_merge_results
# No MERGE_NOT_AI_APPLIED log - empty_ok bypass works
```

---

## 🛡️ Safety Guarantees

1. **Zero-packs unaffected:** `empty_ok` check takes precedence, fast path preserved
2. **Idempotent operations:** Backfill script safe to run multiple times
3. **Backward compatible:** Existing logic flows unchanged, only adds gating
4. **Defensive logging:** Clear log messages for troubleshooting
5. **No data loss:** All operations read-repair pattern, never delete

---

## 📂 Files Modified/Created

### Modified:
1. `backend/runflow/decider.py`
   - `finalize_merge_stage()` - Set `merge_ai_applied` flag
   - `_compute_umbrella_barriers()` - Check flag for non-zero-packs
   - `_maybe_enqueue_validation_fastpath()` - Defensive check

2. `backend/pipeline/validation_orchestrator.py`
   - `run_for_sid()` - Gate on `merge_ready` barrier

3. `devtools/run_validation_orchestrator.py`
   - Added merge_ready warning

### Created:
1. `scripts/repair_merge_ai_applied_from_run.py` - Backfill script
2. `scripts/prove_merge_fix.py` - Verification script
3. `backend/tests/test_merge_ai_applied_fix.py` - Test suite
4. `MERGE_FINALIZATION_FIX_SUMMARY.md` - This document

---

## 🚀 Deployment Checklist

- [ ] Run unit tests: `pytest backend/tests/test_merge_ai_applied_fix.py -v`
- [ ] Run backfill dry-run: `python scripts/repair_merge_ai_applied_from_run.py --runs-root runs --dry-run`
- [ ] Review dry-run output, confirm expected repair count
- [ ] Run backfill: `python scripts/repair_merge_ai_applied_from_run.py --runs-root runs`
- [ ] Run proof script: `python scripts/prove_merge_fix.py --runs-root runs`
- [ ] Verify all 3 test SIDs pass
- [ ] Review `runs/merge_fix_verification_results.json`
- [ ] Check logs for `MERGE_AI_APPLIED` and `MERGE_NOT_AI_APPLIED` messages
- [ ] Monitor first production run with merge packs
- [ ] Verify validation orchestrator defers correctly when merge not ready

---

## 🐛 Troubleshooting

### Problem: Validation still starting too early
**Check:**
1. Is `merge_ai_applied` flag present in runflow.json?
2. Run backfill script on the SID
3. Check for `MERGE_NOT_AI_APPLIED` in logs

### Problem: Zero-packs case broken
**Check:**
1. Verify `empty_ok=True` in merge stage
2. Check logs for `UMBRELLA_MERGE_OPTIONAL reason=empty_merge_results`
3. Confirm no `MERGE_NOT_AI_APPLIED` logs for zero-packs cases

### Problem: Backfill script reports no repairs needed
**Check:**
1. Are merge stages already marked success?
2. Is `empty_ok=True` (zero-packs don't need repair)?
3. Is `merge_ai_applied` already present?

---

## 📚 References

- Investigation: `MERGE_BARRIER_INVESTIGATION.md`
- Bug SID Timeline: Lines 198-281 in investigation document
- Original authority fix: `RUNFLOW_VALIDATION_AUTHORITY_ANALYSIS.md`
- V2 strategy: `VALIDATION_V2_PRODUCTION_INTEGRATION.md`

---

**Status:** ✅ All 7 implementation tasks complete  
**Next Step:** Run verification scripts and deploy
