# Validation & Strategy Writers/Drivers Mapping
**Date:** 2025-11-16  
**SID:** b4cf2c3e-7f7c-4f7b-9087-7bff671a2b0e  
**Purpose:** Map all components that write validation/strategy state to manifest/runflow

---

## VALIDATION WRITERS

### 1. `build_validation_packs_inline()` 
**File:** `backend/ai/validation_builder.py` ~line 1995-2050  
**Called by:** `strategy_planner_step()` task (inline execution path)  
**Writes to:**
- **runflow.json:** NO direct write (relies on caller)
- **manifest.json:** ❌ **DOES NOT WRITE** (this is Bug #1)

**Short-circuit:** ❌ **NONE** - Always runs heavy validation work  
**Problem:** 
- No idempotency check
- Never calls `_update_manifest_for_run()`
- Validation completes but manifest shows null paths

---

### 2. `_update_manifest_for_run()`
**File:** `backend/ai/validation_builder.py` ~line 2275-2295  
**Called by:** Legacy task path (disabled by default)  
**Writes to:**
- **manifest.json:**
  - `ai.validation.{base, dir, accounts, accounts_dir, last_prepared_at}`
  - `ai.packs.validation.{base, dir, packs, packs_dir, results, results_dir, index, logs, last_built_at}`
  - `ai.status.validation.{built=true, sent=false, completed_at=null}`

**Short-circuit:** N/A (is a helper, not an entry point)  
**Problem:** Never called by inline path

---

### 3. `apply_validation_merge_and_update_state()`
**File:** `backend/pipeline/validation_merge_helpers.py` ~line 89-235  
**Called by:** 
- `strategy_planner_step()` after validation completes
- Umbrella/reconciliation logic

**Writes to:**
- **runflow.json:**
  - `stages.validation.merge_results.{applied, applied_at, source}`
  - `stages.validation.merge_results_applied = true`
  - `stages.validation.metrics.{validation_ai_*}`
  - `stages.validation.summary.*`
  - Calls `record_stage_force()` → updates runflow atomically

- **manifest.json:**
  - `ai.status.validation.merge_results_applied = true`
  - `ai.status.validation.merge_results.{applied, applied_at, source}`
  - Via `RunManifest.mark_validation_merge_applied()`

**Short-circuit:** ✅ **YES** - Checks if merge already applied:
```python
if runflow_merge_applied and manifest_merge_applied:
    return {"merge_applied": True, "skipped": True, "reason": "already_applied"}
```

---

### 4. `RunManifest.upsert_validation_packs_dir()`
**File:** `backend/pipeline/runs.py` ~line 842-903  
**Called by:** `_update_manifest_for_run()` (when it's actually called)  
**Writes to:**
- **manifest.json:**
  - `ai.validation.*` (all path fields)
  - `ai.packs.validation.*` (all path fields)
  - `ai.status.validation.{built=true, sent=false, completed_at=null}`
  - Saves atomically via `safe_replace()`

**Short-circuit:** N/A (is a helper method)  
**Notes:** This is the CORRECT way to persist validation paths, but inline builder skips it

---

### 5. `RunManifest.mark_validation_merge_applied()`
**File:** `backend/pipeline/runs.py` ~line 1105-1199  
**Called by:** `apply_validation_merge_and_update_state()`  
**Writes to:**
- **manifest.json:**
  - `ai.status.validation.{merge_results_applied, merge_results_applied_at, sent=true, completed_at, failed=false}`
  - `ai.status.validation.merge_results.{applied, applied_at, source}`
  - `ai.packs.validation.status.{sent=true, completed_at}`
  - Saves atomically

**Short-circuit:** Internal state check - only writes if changed

---

### 6. `record_stage_force()` with validation
**File:** `backend/runflow/decider.py` ~line 2589-2615  
**Called by:** Multiple callers (validation_merge_helpers, umbrella, etc.)  
**Writes to:**
- **runflow.json:**
  - Merges incoming `stages.validation.*` with existing data
  - Uses `_merge_runflow_snapshots()` and `_merge_stage_snapshot()`
  - Atomically writes via `_atomic_write_json()`

**Short-circuit:** N/A (generic recorder)  
**Notes:** Merge logic attempts to preserve existing data, but can be overridden by incoming snapshot

---

### 7. `_maybe_enqueue_validation_fastpath()`
**File:** `backend/runflow/decider.py` ~line 696-810  
**Called by:** `reconcile_umbrella_barriers()`  
**Triggers:** Validation task enqueue (only if enabled via env var)  
**Writes to:**
- **runflow.json:**
  - `stages.validation.{status=in_progress, sent=true, metrics, merge_context}`
  - Only if merge_zero_packs fastpath is triggered

**Short-circuit:** ✅ Checks `validation_terminal`, `validation_ready_latched`, `validation_sent_flag`  
**Notes:** **DISABLED BY DEFAULT** - requires `ENABLE_VALIDATION_FASTPATH=1`

---

## STRATEGY WRITERS

### 8. `run_strategy_planner_for_all_accounts()`
**File:** `backend/pipeline/auto_ai.py` ~line 423-640  
**Called by:** 
- `strategy_planner_step()` Celery task
- Legacy `run_validation_and_strategy_for_all_accounts()` (disabled)

**Writes to:**
- **runflow.json:** NO direct write (relies on caller)
- **manifest.json:**
  - Via `RunManifest.mark_strategy_started()` at beginning
  - Via `RunManifest.register_strategy_artifacts_for_account()` per account
  - Via `RunManifest.mark_strategy_completed()` at end
  - All saved atomically

**Short-circuit:** ✅ **YES** - Checks runflow.json at start:
```python
strategy_stage = stages_payload.get("strategy")
if isinstance(strategy_stage, Mapping):
    strategy_status = str(strategy_stage.get("status") or "").strip().lower()
    if strategy_status == "success":
        logger.info("STRATEGY_RUN_SHORT_CIRCUIT sid=%s reason=stage_success", sid)
        # Returns cached stats, does not re-run
```

**Problem:** Short-circuit is AFTER opening manifest and marking started!

---

### 9. `strategy_planner_step()` Celery task
**File:** `backend/pipeline/auto_ai_tasks.py` ~line 1263-1450  
**Called by:** 
- Strategy recovery chain (`enqueue_strategy_recovery_chain()`)
- Direct invocation (rarely)

**Writes to:**
- **runflow.json:**
  - `stages.validation.*` (via `record_stage()`)
  - `stages.frontend.*` (via `record_stage()`)
  - Eventually calls `reconcile_umbrella_barriers()`

- **manifest.json:**
  - Via `manifest.mark_strategy_started()` at task start
  - Via `manifest.register_strategy_artifacts_for_account()` per account
  - Via `manifest.mark_strategy_completed()` at task end

**Short-circuit:** ⚠️ **PARTIAL** - Has early check but AFTER some writes:
```python
manifest = RunManifest.for_sid(sid, ...)
manifest.mark_strategy_started()  # ← Writes to manifest BEFORE checking!
manifest.save()

# Only then checks:
stats = run_strategy_planner_for_all_accounts(sid, ...)
# Which has the actual short-circuit inside
```

**Problem:** 
- Writes `strategy.started_at` and `strategy.built=true` even if already done
- Should check FIRST, then mark started only if actually running

---

### 10. `RunManifest.mark_strategy_started()`
**File:** `backend/pipeline/runs.py` ~line 1220-1235  
**Called by:** 
- `strategy_planner_step()` task
- `run_strategy_planner_for_all_accounts()`

**Writes to:**
- **manifest.json:**
  - `ai.status.strategy.{built=true, sent=false, failed=false, state=in_progress, started_at}`
  - Saves atomically

**Short-circuit:** ❌ **NONE** - Always writes  
**Problem:** Overwrites existing state even if strategy already success

---

### 11. `RunManifest.mark_strategy_completed()`
**File:** `backend/pipeline/runs.py` ~line 1237-1274  
**Called by:** 
- `strategy_planner_step()` task
- `run_strategy_planner_for_all_accounts()`

**Writes to:**
- **manifest.json:**
  - `ai.status.strategy.{completed_at, failed, state, stats}`
  - Saves atomically

**Short-circuit:** N/A (called after strategy completes)

---

### 12. `RunManifest.register_strategy_artifacts_for_account()`
**File:** `backend/pipeline/runs.py` ~line 1276-1366  
**Called by:** 
- `strategy_planner_step()` task (for each account)
- `run_strategy_planner_for_all_accounts()` (for each account)

**Writes to:**
- **manifest.json:**
  - `artifacts.cases.accounts.<account_id>.strategy.*` (all bureau paths)
  - Does NOT call `save()` - relies on caller to save

**Short-circuit:** Checks if strategy dir exists, returns early if not  
**Notes:** Idempotent - safe to call multiple times

---

### 13. `enqueue_strategy_recovery_chain()`
**File:** `backend/pipeline/auto_ai_tasks.py` ~line 2102-2140  
**Called by:** `reconcile_umbrella_barriers()` (strategy recovery logic)  
**Triggers:** Chain of strategy_seed → strategy_planner → polarity → consistency → finalize  
**Writes to:**
- **runflow.json:**
  - `stages.strategy.{status=in_progress, notes=recovery_enqueued, task_id}`
  - Via `record_stage_force()` ONLY if enqueue succeeds

**Short-circuit:** ⚠️ **CALLER RESPONSIBLE**  
**Notes:** Should not be called if strategy already terminal (success/error)

---

### 14. `reconcile_umbrella_barriers()` - Strategy Recovery Path
**File:** `backend/runflow/decider.py` ~line 6000-6130  
**Triggered by:** Periodic umbrella watchdog + manual reconcile calls  
**Condition:** When validation completes but strategy not started/failed  
**Enqueues:** `enqueue_strategy_recovery_chain()` if conditions met  
**Writes to:**
- **runflow.json:**
  - `stages.strategy.*` (if enqueue succeeds)
  - Via `record_stage_force()`

**Short-circuit:** ✅ **YES** - Multiple guards:
```python
# Hard guard: do not attempt recovery for terminal strategy states
if strategy_status in {"success", "error"}:
    # Only log once, then return
    
# Guard: require validation to be complete first
if not (validation_ready and validation_merge_applied):
    return
    
# Guard: check if strategy already started
if strategy_started:
    return
```

**Problem:** Guards are GOOD, but still runs reconciliation repeatedly even when idle

---

## INTENDED FLOW vs ACTUAL FLOW

### INTENDED FLOW (Design)

**Validation:**
1. `strategy_planner_step()` calls `build_validation_packs_inline()`
2. Validation builds packs, writes to disk
3. `_update_manifest_for_run()` persists paths to manifest
4. `apply_validation_merge_and_update_state()` applies results, updates both manifest + runflow
5. **DONE** - Validation marked success, paths persisted, no further work

**Strategy:**
1. After validation completes, strategy_planner_step continues OR
2. Umbrella recovery enqueues strategy chain
3. `run_strategy_planner_for_all_accounts()` checks if already done (short-circuit)
4. If not done, runs strategy, writes artifacts, marks completed in manifest
5. Updates runflow.json via record_stage()
6. **DONE** - Strategy marked success, no further work

**Reconciliation:**
1. Runs periodically to check barriers
2. If all stages complete, sets `reconciliation_idle=true`
3. Skips heavy work when idle
4. Never re-enqueues completed stages

---

### ACTUAL FLOW (Code Reality)

**Validation:**
1. `strategy_planner_step()` calls `build_validation_packs_inline()`
2. Validation builds packs ✅
3. ❌ **SKIPS** `_update_manifest_for_run()` - manifest paths stay NULL
4. `apply_validation_merge_and_update_state()` updates runflow + partial manifest ✅
5. **INCOMPLETE** - Validation works but manifest missing paths

**Problems:**
- No short-circuit in `build_validation_packs_inline()` - always runs
- Manifest validation paths never persisted
- `ai.status.validation.built` stays `false` in manifest (even though runflow says success)

**Strategy:**
1. `strategy_planner_step()` task starts
2. ❌ **IMMEDIATELY WRITES** `mark_strategy_started()` before checking if needed
3. Then calls `run_strategy_planner_for_all_accounts()`
4. Which has short-circuit checking runflow ✅
5. If already done, returns cached stats
6. But manifest was already modified in step 2!
7. Task completes, umbrella runs, sees strategy complete
8. **PARTIAL SUCCESS** - Strategy doesn't re-run heavy work, but writes happen

**Problems:**
- `mark_strategy_started()` called BEFORE short-circuit check
- Overwrites `ai.status.strategy` timestamps even if not running
- Creates false impression of activity

**Reconciliation:**
1. Runs every 5 seconds via watchdog
2. Checks barriers, computes idle state ✅
3. Sets `reconciliation_idle=true` when appropriate ✅
4. **BUT**: Still runs full reconciliation logic every time
5. **AND**: Strategy recovery path checks if should enqueue ✅
6. **BUT**: Logs repeatedly even when idle

**Problems:**
- Reconciliation doesn't short-circuit ITSELF when idle
- Repeats full barrier computation even when nothing changes
- Creates log noise (not actual work, but looks busy)

---

## RACE CONDITIONS & OVERWRITES

### Race Scenario 1: Validation Merge vs Strategy Start

**Timeline:**
1. T1: validation completes, `apply_validation_merge_and_update_state()` updates manifest
2. T2: strategy task starts, calls `mark_strategy_started()`, writes manifest
3. T3: Both save manifest.json

**Risk:** If T2 reads stale manifest (before T1's save), T2's write overwrites T1's validation state

**Mitigation:** 
- Atomic `safe_replace()` prevents file corruption ✅
- `ensure_ai_stage_status()` preserves other stage sections ✅
- Each stage has its own section (validation vs strategy) ✅

**Verdict:** LOW RISK - Sections are separate, atomic writes work

---

### Race Scenario 2: Multiple Reconciliation Calls

**Timeline:**
1. T1: Umbrella watchdog triggers `reconcile_umbrella_barriers()`
2. T2: Task completion handler also calls `reconcile_umbrella_barriers()`
3. Both compute barriers, both write runflow.json

**Risk:** Last writer wins, but since they compute same values, no data loss

**Mitigation:**
- Idle detection prevents enqueuing duplicate tasks ✅
- runflow merging logic attempts to preserve data ✅

**Verdict:** LOW RISK - Redundant but safe

---

### Overwrite Scenario: Strategy Marks Started Over Success

**Problem:**
```python
# In strategy_planner_step():
manifest.mark_strategy_started()  # Sets state=in_progress, built=true
manifest.save()

# Later, run_strategy_planner_for_all_accounts() short-circuits
# But manifest already modified!

# At end, if short-circuited:
manifest.mark_strategy_completed(cached_stats, state="success")
# This writes state=success, but timestamps were already modified
```

**Impact:**
- `started_at` timestamp keeps getting updated even when not running
- `built=true` flips even when strategy already complete
- Not destructive, but creates false activity signals

**Verdict:** MEDIUM ISSUE - Not a race, but poor idempotency

---

## SUMMARY OF PROBLEMS

### Validation Issues
1. ❌ **No short-circuit** - Always runs heavy work
2. ❌ **Missing manifest updates** - Paths never persisted by inline builder
3. ⚠️ **Inconsistent state** - runflow says success, manifest says not built

### Strategy Issues
1. ⚠️ **Premature writes** - Marks started before checking if needed
2. ✅ **Good short-circuit** - Doesn't re-run heavy work (in inner function)
3. ⚠️ **Timestamp pollution** - Updates times even when short-circuiting

### Reconciliation Issues
1. ⚠️ **No self short-circuit** - Runs full logic even when idle
2. ✅ **Good idle detection** - Doesn't re-enqueue tasks
3. ⚠️ **Log noise** - Creates impression of activity when stable

### Race Conditions
1. ✅ **Low risk** - Atomic writes, separate sections
2. ✅ **Good guards** - Short-circuits prevent duplicate heavy work
3. ⚠️ **Redundant calls** - Multiple reconciliations but harmless

---

## NEXT STEPS (Task 2-4)

1. **Fix validation idempotency:**
   - Add short-circuit check at top of `build_validation_packs_inline()`
   - Call `_update_manifest_for_run()` after successful validation
   - Ensure manifest validation.built reflects reality

2. **Fix strategy idempotency:**
   - Move short-circuit check to TOP of `strategy_planner_step()` before any writes
   - Only call `mark_strategy_started()` if actually running strategy
   - Prevent timestamp pollution

3. **Fix reconciliation efficiency:**
   - Add early return when `reconciliation_idle=true` and no state changes
   - Reduce log level for stable/idle state
   - Keep idle detection, add idle short-circuit

4. **Ensure monotonic completion:**
   - Once validation success, never revert to not-built
   - Once strategy success, never revert to in-progress
   - Add logging for any state changes

---

**End of Mapping**
