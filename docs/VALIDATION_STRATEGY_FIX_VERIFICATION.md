# Validation & Strategy Idempotency Fix - Verification Guide

**Date:** 2025-01-XX  
**Test SID:** `b4cf2c3e-7f7c-4f7b-9087-7bff671a2b0e`  
**Changes:** Validation/strategy short-circuit checks, monotonic completion guards

---

## Changes Summary

### 1. Validation Task Short-Circuit (`backend/pipeline/auto_ai_tasks.py`)
**Line:** ~1732-1750

**Before:**
```python
def validation_build_packs(self, prev):
    # ... setup ...
    ensure_validation_section(sid, runs_root=runs_root)
    logger.info("VALIDATION_STAGE_STARTED sid=%s", sid)
    
    results = build_validation_packs_for_run(sid, runs_root=runs_root)
    # ... always ran validation work ...
```

**After:**
```python
def validation_build_packs(self, prev):
    # ... setup ...
    ensure_validation_section(sid, runs_root=runs_root)
    
    # Short-circuit if validation already completed
    manifest = RunManifest.for_sid(sid, runs_root=runs_root, allow_create=False)
    validation_status = manifest.get_ai_stage_status("validation")
    state = validation_status.get("state")
    if state == "success":
        logger.info("VALIDATION_BUILD_SHORT_CIRCUIT sid=%s reason=already_success", sid)
        return payload  # Skip heavy work
    
    logger.info("VALIDATION_STAGE_STARTED sid=%s", sid)
    results = build_validation_packs_for_run(sid, runs_root=runs_root)
    # ...
```

**Impact:**
- ✅ Prevents re-building validation packs when already complete
- ✅ Reduces redundant AI API calls
- ✅ Saves ~30-60 seconds per duplicate task

---

### 2. Strategy Task Short-Circuit (`backend/pipeline/auto_ai_tasks.py`)
**Line:** ~1295-1320

**Before:**
```python
def strategy_planner_step(self, prev):
    # ... setup ...
    manifest.mark_strategy_started()  # ❌ Always writes started_at!
    manifest.save()
    
    stats = run_strategy_planner_for_all_accounts(sid, ...)  # Has inner short-circuit
    # ...
```

**After:**
```python
def strategy_planner_step(self, prev):
    # ... setup ...
    
    # Check BEFORE writing started_at
    strategy_status = manifest.get_ai_stage_status("strategy")
    strategy_state = strategy_status.get("state")
    if strategy_state == "success":
        logger.info("STRATEGY_TASK_SHORT_CIRCUIT sid=%s reason=already_success", sid)
        # Return cached stats without modifying manifest
        return payload
    
    # Not done yet, mark started and proceed
    manifest.mark_strategy_started()
    manifest.save()
    
    stats = run_strategy_planner_for_all_accounts(sid, ...)
    # ...
```

**Impact:**
- ✅ Prevents timestamp pollution (started_at no longer updated on re-enqueue)
- ✅ Prevents `built=true` flipping when already done
- ✅ Cleaner audit trail

---

### 3. Strategy Monotonic Completion Guard (`backend/pipeline/runs.py`)
**Line:** ~1220-1240

**Before:**
```python
def mark_strategy_started(self):
    stage_status = self.ensure_ai_stage_status("strategy")
    stage_status["state"] = stage_status.get("state") or "in_progress"  # ❌ Could revert!
    # ...
```

**After:**
```python
def mark_strategy_started(self):
    stage_status = self.ensure_ai_stage_status("strategy")
    
    # Monotonic guarantee: Never revert terminal states
    current_state = stage_status.get("state")
    if current_state in ("success", "error"):
        return self  # Don't modify terminal state
    
    stage_status["state"] = current_state or "in_progress"
    # ...
```

**Impact:**
- ✅ Once strategy reaches `state=success`, never goes back to `in_progress`
- ✅ Prevents state machine violations
- ✅ More robust error recovery

---

## PowerShell Verification Commands

### Setup
```powershell
cd c:\dev\credit-analyzer
$SID = "b4cf2c3e-7f7c-4f7b-9087-7bff671a2b0e"
$RUN_DIR = "runs\$SID"
```

---

### Test 1: Check Current State

**Goal:** Inspect manifest and runflow to see validation/strategy completion state

```powershell
# Check if files exist
Test-Path "$RUN_DIR\manifest.json"
Test-Path "$RUN_DIR\runflow.json"

# View validation state in manifest
$manifest = Get-Content "$RUN_DIR\manifest.json" -Raw | ConvertFrom-Json
$manifest.ai.status.validation.state
$manifest.ai.status.validation.merge_results_applied
$manifest.ai.packs.validation.packs_dir

# View strategy state in manifest
$manifest.ai.status.strategy.state
$manifest.ai.status.strategy.completed_at

# View validation state in runflow
$runflow = Get-Content "$RUN_DIR\runflow.json" -Raw | ConvertFrom-Json
$runflow.stages.validation.status
$runflow.stages.validation.validation_ai_completed
$runflow.stages.validation.merge_results_applied

# View strategy state in runflow
$runflow.stages.strategy.status
$runflow.stages.strategy.plans_written
```

**Expected (if already complete):**
- Manifest: `validation.state = "success"`, `strategy.state = "success"`
- Runflow: `validation.status = "success"`, `strategy.status = "success"`
- Validation paths: `packs_dir` should be non-null
- Strategy: `completed_at` should be non-null

---

### Test 2: Trigger Reconciliation

**Goal:** Force umbrella barriers to re-compute and log decision

```powershell
# Run Python reconciliation command
python -c "from backend.runflow.decider import reconcile_umbrella_barriers; import sys; sys.exit(0 if reconcile_umbrella_barriers('$SID') else 1)"

# Check logs for reconciliation output (if logging configured)
# Look for:
# - "VALIDATION_BUILD_SHORT_CIRCUIT" (should appear if validation already done)
# - "STRATEGY_TASK_SHORT_CIRCUIT" (should appear if strategy already done)
# - "UMBRELLA_RECONCILE_IDLE" (should be true if everything complete)
```

**Expected:**
- Reconciliation completes without enqueuing new tasks
- `reconciliation_idle=true` in runflow
- No "VALIDATION_BUILD_PACKS_START" or "STRATEGY_TASK_MANIFEST_START" logs

---

### Test 3: Simulate Re-Enqueueing Validation Task

**Goal:** Manually trigger validation task, verify short-circuit

```powershell
# This requires Celery worker running and configured
# WARNING: Only run if you have a test environment!

# Queue validation_build_packs task
python -c @"
from backend.pipeline.auto_ai_tasks import validation_build_packs
from celery import chain
result = validation_build_packs.apply_async(args=[{'sid': '$SID', 'runs_root': 'runs'}])
print(f'Task ID: {result.id}')
"@

# Wait 5 seconds
Start-Sleep -Seconds 5

# Check logs for "VALIDATION_BUILD_SHORT_CIRCUIT"
# Should see: VALIDATION_BUILD_SHORT_CIRCUIT sid=b4cf2c3e... reason=already_success

# Verify manifest unchanged
$manifest2 = Get-Content "$RUN_DIR\manifest.json" -Raw | ConvertFrom-Json
$manifest2.ai.status.validation.state  # Should still be "success"
```

**Expected:**
- Task completes in <1 second (short-circuit path)
- Log shows `VALIDATION_BUILD_SHORT_CIRCUIT`
- Manifest `validation.state` remains `"success"`
- No new validation packs built

---

### Test 4: Simulate Re-Enqueueing Strategy Task

**Goal:** Manually trigger strategy task, verify short-circuit and no timestamp pollution

```powershell
# Capture current started_at timestamp
$manifest_before = Get-Content "$RUN_DIR\manifest.json" -Raw | ConvertFrom-Json
$started_at_before = $manifest_before.ai.status.strategy.started_at

Write-Host "Strategy started_at BEFORE: $started_at_before"

# Queue strategy_planner_step task
python -c @"
from backend.pipeline.auto_ai_tasks import strategy_planner_step
result = strategy_planner_step.apply_async(args=[{'sid': '$SID', 'runs_root': 'runs'}])
print(f'Task ID: {result.id}')
"@

# Wait 5 seconds
Start-Sleep -Seconds 5

# Check logs for "STRATEGY_TASK_SHORT_CIRCUIT"

# Verify manifest timestamps UNCHANGED
$manifest_after = Get-Content "$RUN_DIR\manifest.json" -Raw | ConvertFrom-Json
$started_at_after = $manifest_after.ai.status.strategy.started_at

Write-Host "Strategy started_at AFTER: $started_at_after"

# Compare
if ($started_at_before -eq $started_at_after) {
    Write-Host "✅ SUCCESS: started_at unchanged (no timestamp pollution)" -ForegroundColor Green
} else {
    Write-Host "❌ FAILURE: started_at changed from $started_at_before to $started_at_after" -ForegroundColor Red
}
```

**Expected:**
- Task completes in <1 second
- Log shows `STRATEGY_TASK_SHORT_CIRCUIT`
- Manifest `strategy.started_at` **UNCHANGED** ← Key fix!
- Manifest `strategy.state` remains `"success"`

---

### Test 5: Check Reconciliation Log Noise

**Goal:** Verify reconciliation doesn't spam logs when idle

```powershell
# Run reconciliation 3 times in a row
1..3 | ForEach-Object {
    Write-Host "Reconciliation run $_"
    python -c "from backend.runflow.decider import reconcile_umbrella_barriers; reconcile_umbrella_barriers('$SID')"
    Start-Sleep -Seconds 1
}

# Check runflow reconciliation_idle flag
$runflow_final = Get-Content "$RUN_DIR\runflow.json" -Raw | ConvertFrom-Json
$runflow_final.reconciliation_idle

# Expected: reconciliation_idle = true
# Expected: Logs should NOT show "enqueuing validation" or "enqueuing strategy"
```

**Expected:**
- `reconciliation_idle = true`
- No task enqueueing
- Minimal log output (should skip autosends when idle)

---

## Success Criteria

### ✅ Validation Idempotency
- [ ] `VALIDATION_BUILD_SHORT_CIRCUIT` log appears when validation already done
- [ ] Validation task completes in <1 second when short-circuiting
- [ ] Manifest `ai.status.validation.state` remains `"success"` after re-enqueue
- [ ] No duplicate validation pack files created
- [ ] At most ONE `VALIDATION_STAGE_STARTED` log per SID per pipeline run

### ✅ Strategy Idempotency
- [ ] `STRATEGY_TASK_SHORT_CIRCUIT` log appears when strategy already done
- [ ] Strategy task completes in <1 second when short-circuiting
- [ ] Manifest `ai.status.strategy.started_at` **UNCHANGED** after re-enqueue ← Key metric
- [ ] Manifest `ai.status.strategy.state` remains `"success"` after re-enqueue
- [ ] At most ONE `STRATEGY_TASK_MANIFEST_START` log per SID per pipeline run

### ✅ Monotonic Completion
- [ ] Once validation `state="success"`, never reverts to `null` or `"in_progress"`
- [ ] Once strategy `state="success"`, never reverts to `"in_progress"`
- [ ] `mark_strategy_started()` returns early when `state` is `"success"` or `"error"`

### ✅ Reconciliation Efficiency
- [ ] `reconciliation_idle=true` when all stages complete
- [ ] No validation/strategy tasks enqueued when idle
- [ ] Reconciliation completes in <500ms when idle (just barrier checks, no heavy work)

### ✅ No Busy-Wait Periods
- [ ] After all stages complete, system reaches idle state within 10 seconds
- [ ] No 1-2 minute periods of repeated reconciliation calls
- [ ] Log volume drops significantly after reaching idle state

---

## Rollback Plan

If fixes cause regressions:

### Quick Rollback
```powershell
cd c:\dev\credit-analyzer
git diff HEAD backend/pipeline/auto_ai_tasks.py backend/pipeline/runs.py
git checkout HEAD -- backend/pipeline/auto_ai_tasks.py backend/pipeline/runs.py
```

### Selective Rollback
```powershell
# Rollback only validation short-circuit
git diff HEAD backend/pipeline/auto_ai_tasks.py | Select-String -Pattern "validation_build_packs" -Context 20

# Rollback only strategy short-circuit
git diff HEAD backend/pipeline/auto_ai_tasks.py | Select-String -Pattern "strategy_planner_step" -Context 20

# Rollback only monotonic guard
git diff HEAD backend/pipeline/runs.py | Select-String -Pattern "mark_strategy_started" -Context 15
```

---

## Known Limitations

1. **Validation paths in manifest:**
   - Fix ensures short-circuit works
   - Does NOT retroactively populate missing paths for old SIDs
   - Only NEW validations will persist paths correctly

2. **Strategy recovery path:**
   - Short-circuit added to main task entry
   - Inner `run_strategy_planner_for_all_accounts()` still has own short-circuit (good!)
   - Both checks are complementary, not redundant

3. **Reconciliation log noise:**
   - Idle detection works (no re-enqueueing)
   - But reconciliation STILL RUNS every 5 seconds when idle
   - Future enhancement: Add early return when `reconciliation_idle=true` AND no state changes

---

## Debugging Tips

### If validation still re-runs:
1. Check manifest: `ai.status.validation.state` - should be `"success"`
2. Check runflow: `stages.validation.status` - should be `"success"`
3. Verify short-circuit log appears: `grep "VALIDATION_BUILD_SHORT_CIRCUIT"`
4. Check if `RunManifest.for_sid()` can load manifest (file exists, valid JSON)

### If strategy timestamps still change:
1. Verify short-circuit happens BEFORE `mark_strategy_started()` call
2. Check log order: `STRATEGY_TASK_SHORT_CIRCUIT` should appear BEFORE `STRATEGY_TASK_MANIFEST_START`
3. If manifest modified, check stack trace to find caller
4. Verify `mark_strategy_started()` returns early when state is terminal

### If reconciliation still busy:
1. Check `reconciliation_idle` flag in runflow: `jq .reconciliation_idle runs/$SID/runflow.json`
2. Verify watchdog interval: `UMBRELLA_BARRIERS_WATCHDOG_INTERVAL_MS` (default 5000ms)
3. Check if autosend functions being called: grep for `AUTOSEND_DECISION`
4. Confirm no tasks enqueued: Check Celery queue depth

---

**End of Verification Guide**
