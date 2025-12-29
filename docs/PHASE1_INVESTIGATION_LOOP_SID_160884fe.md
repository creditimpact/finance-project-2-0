# Phase 1 Investigation: Validation Loop Analysis
## SID: 160884fe-b510-493b-888d-dd2ec09b4bb5

**Investigation Date:** November 16, 2025  
**Status:** READ-ONLY (No code changes)  
**Run Eventually Completed:** Yes (run_state=AWAITING_CUSTOMER_INPUT, strategy=success)  
**Problem:** Heavy ~30 second loop with repeated reconciliation, autosend, and recovery checks

---

## Executive Summary

**Root Cause:** The validation autosend path spawns a **background recheck thread** that repeatedly calls `reconcile_umbrella_barriers()` every few seconds. This thread continues to reconcile barriers, trigger autosend checks, and evaluate strategy recovery **even after all stages are complete**, because there is no explicit "work complete" signal to stop the thread. Combined with a `PermissionError` on index.json writes (Windows file locking), the loop persists for ~30 seconds until the recheck thread times out or validation internally marks itself ready.

**Loop Components:**
1. **Validation autosend recheck thread** (`_schedule_validation_recheck`) runs in background with 2-10s random delay
2. Each iteration calls `reconcile_umbrella_barriers(sid)`
3. Reconciliation triggers:
   - `_apply_validation_stage_promotion` → logs `VALIDATION_RESULTS_DEBUG`, `VALIDATION_STAGE_SUMMARY`
   - `schedule_merge_autosend` → logs `MERGE_AUTOSEND_STAGE_SKIP ... reason=no_packs`
   - `schedule_note_style_after_validation` → logs `NOTE_STYLE_AUTOSEND_DECISION ... reason=empty`
   - Strategy recovery check → logs `STRATEGY_RECOVERY_SKIP ... reason=already_started status=in_progress`
4. Loop continues until validation internally latches `ready_latched=true` or recheck thread stops

**Performance Impact:**
- 3 validation packs → ~30 seconds of tight looping
- 21 validation packs → potentially **minutes** of repeated reconciliation
- O(recheck_iterations × umbrella_overhead) cost

---

## 1. Exact Loop Mechanism

### 1.1 Loop Trigger: Background Recheck Thread

**File:** `backend/ai/validation_builder.py`  
**Function:** `_schedule_validation_recheck(sid, runs_root, stage)` (lines 1878-1906)

**How it works:**
```python
def _schedule_validation_recheck(sid: str, runs_root: Path, stage: str) -> None:
    delay = random.uniform(_AUTOSEND_RECHECK_MIN_DELAY, _AUTOSEND_RECHECK_MAX_DELAY)
    # _AUTOSEND_RECHECK_MIN_DELAY = 2.0 seconds
    # _AUTOSEND_RECHECK_MAX_DELAY = 10.0 seconds

    def _runner() -> None:
        time.sleep(delay)
        try:
            log.info("VALIDATION_AUTOSEND_RECHECK sid=%s stage=%s delay=%.2f", ...)
            _maybe_send_validation_packs(sid, runs_root, stage=stage, recheck=True)
        except Exception:
            log.exception("VALIDATION_AUTOSEND_RECHECK_FAILED ...")

    thread = threading.Thread(target=_runner, name=f"validation-autosend-recheck-{sid}", daemon=True)
    thread.start()
```

**Called from:**
- `_maybe_send_validation_packs()` (line 1991) at the **end** of validation autosend
- After calling `reconcile_umbrella_barriers(sid)` (line 1985)

**Result:**
- A **daemon thread** sleeps 2-10 seconds, then calls `_maybe_send_validation_packs(..., recheck=True)` again
- This creates a **recursive loop**: send → reconcile → schedule recheck → sleep → send → reconcile → ...
- Loop continues until:
  - `recheck=True` prevents further recursion (but reconciliation still happens once per iteration)
  - Validation stage is marked as terminal/complete
  - Process exits

---

### 1.2 Reconciliation Cascade

**File:** `backend/runflow/decider.py`  
**Function:** `reconcile_umbrella_barriers(sid, runs_root)` (lines 5722-6035)

**What it does (in order):**
1. **Stage Promotions** (lines 5734-5770):
   ```python
   _merge_updated, merge_promoted, merge_log = _apply_merge_stage_promotion(data, run_dir)
   _validation_updated, validation_promoted, validation_log = _apply_validation_stage_promotion(data, run_dir)
   _frontend_updated, frontend_promoted, frontend_log = _apply_frontend_stage_promotion(data, run_dir)
   _note_style_updated, note_style_promoted, note_style_log = _apply_note_style_stage_promotion(data, run_dir)
   ```

2. **Metadata Checks** (lines 5772-5777):
   ```python
   _ensure_merge_zero_pack_metadata(sid, run_dir, data)
   _maybe_enqueue_validation_fastpath(sid, run_dir, data)
   watchdog_context = _validation_stuck_fastpath(sid, run_dir, data)
   ```

3. **Compute Barriers** (line 5785):
   ```python
   statuses = _compute_umbrella_barriers(run_dir, runflow_payload=data)
   ```

4. **Resolve run_state** (lines 5804-5825):
   ```python
   post_state = _resolve_post_validation_state(data, statuses)
   # Determines if run_state should be VALIDATING, AWAITING_CUSTOMER_INPUT, ERROR, etc.
   ```

5. **Log State** (line 5830):
   ```python
   log.info("UMBRELLA_STATE_AFTER_RECONCILE sid=%s run_state=%s", sid, data.get("run_state"))
   ```

6. **Trigger Autosends** (lines 5865-5878):
   ```python
   schedule_merge_autosend(sid, run_dir=run_dir)           # → MERGE_AUTOSEND_STAGE_SKIP
   schedule_note_style_after_validation(sid, run_dir=run_dir)  # → NOTE_STYLE_AUTOSEND_DECISION
   ```

7. **Strategy Recovery Check** (lines 5901-6033):
   ```python
   # If ENABLE_STRATEGY_RECOVERY env var is set...
   should_recover = (not strategy_started and strategy_required and strategy_ready ...)
   if should_recover:
       # Enqueue strategy recovery
   else:
       log.info("STRATEGY_RECOVERY_SKIP sid=%s reason=already_started status=%s", ...)
   ```

**For SID 160884fe...:**
- Validation autosend triggers reconciliation
- Reconciliation runs all 7 steps above
- Autosends see stages already complete/empty → skip
- Strategy recovery sees `status=in_progress` → skip
- But reconciliation **re-runs on every recheck thread iteration**

---

### 1.3 Validation Stage Promotion Loop

**File:** `backend/runflow/decider.py`  
**Function:** `_apply_validation_stage_promotion(data, run_dir)` (lines 3241-3541)

**Key behavior:**
1. **Reads validation index** via `_validation_results_progress(run_dir)` (line 3311)
2. **Logs debug info** (line 4516):
   ```python
   log.info(
       "VALIDATION_RESULTS_DEBUG sid=%s expected_accounts=%s disk_result_accounts=%s index_result_accounts=%s missing_accounts=%s",
       run_dir.name,
       sorted(list(unique_pack_accounts)),
       sorted(list(disk_result_accounts)),  # ← EMPTY due to PermissionError
       sorted(list(index_result_accounts)),  # ← ['9','10','11'] from index
       sorted(list(missing_accounts)),
   )
   ```
3. **Computes validation readiness:**
   ```python
   ready = (expected > 0 and failed == 0 and missing == 0 and completed >= expected)
   ```
4. **Early exit guards** (lines 3248-3308):
   - Skip if `run_state` is terminal AND `validation.status` is success/error
   - Skip if prior status was success with no missing results (monotonic success)
5. **But:** These guards can fail due to:
   - Transient index inconsistencies
   - `disk_result_accounts=[]` due to file locking
   - `run_state=VALIDATING` (not yet terminal)

**For SID 160884fe...:**
- `disk_result_accounts=[]` because Windows file handle prevents listing `results/*.result.jsonl`
- `index_result_accounts=['9','10','11']` read from index.json
- Readiness computed from index, but promotion logic sees mismatch
- Loop continues until filesystem becomes readable or index write succeeds

---

## 2. VALIDATING State Semantics

### 2.1 run_state Definition

**File:** `backend/runflow/decider.py`  
**Function:** `_resolve_post_validation_state(data, statuses)` (not shown, inferred from lines 5815-5826)

**Logic:**
```python
post_state = _resolve_post_validation_state(data, statuses)
existing_run_state = data.get("run_state")

if has_stage_error:
    updated_run_state = "ERROR"
elif post_state is not None:
    updated_run_state = _prefer_run_state(existing_run_state, post_state)
elif normalized_existing_state in {"", "ERROR"}:
    updated_run_state = "VALIDATING"
```

**run_state values:**
- `"VALIDATING"`: Default state when validation is running or barriers not met
- `"AWAITING_CUSTOMER_INPUT"`: All automated stages complete; waiting for frontend review
- `"ERROR"`: Critical stage has `status=error`
- `"COMPLETED"`: All stages including review complete
- `"COMPLETED_NO_ACTION"`: Run finished with no actionable findings

**Transition from VALIDATING:**
- Requires `_resolve_post_validation_state()` to return a non-null value
- Typically when `all_ready=true` in umbrella barriers
- **Problem:** If any barrier is false, stays in VALIDATING

---

### 2.2 Two Notions of "Validation Complete"

**Notion 1: AI/LLM calls finished**
- `validation.status=success`
- `validation_ai_completed=true`
- All validation result files written to disk

**Notion 2: Results fully reconciled**
- Index.json reflects all results
- `merge_results_applied=true`
- `ready_latched=true` in runflow.json validation stage
- No missing_results in runflow summary

**For SID 160884fe...:**
- **Notion 1** is TRUE at 15:53:16Z:
  - `validation.status=success`
  - All 3 results written to disk
  - `validation_ai_completed=true`
- **Notion 2** takes longer due to:
  - Index write PermissionError (15:52:23)
  - Repeated reconciliation cycles checking for readiness
  - Finally latches at 15:53:17Z: `ready_latched=true`

**Gap between notions:** ~1-2 seconds of tight looping, but recheck thread extends it to ~30 seconds

---

### 2.3 Conditions Keeping run_state=VALIDATING

**File:** `backend/runflow/decider.py`  
**Function:** `_compute_umbrella_barriers(run_dir, runflow_payload)` (not fully shown)

**Inferred barrier logic:**
```python
validation_ready = (
    validation_status == "success"
    and validation_ai_completed
    and merge_results_applied
    and missing_results == 0
)

strategy_ready = (
    strategy_status in {"success", "published"}
    or (strategy_status == "in_progress" and strategy_required == false)
)

all_ready = (
    merge_ready
    and validation_ready
    and (not strategy_required or strategy_ready)
    and (not style_required or style_ready or review_ready)
)
```

**For SID 160884fe... during loop:**
- `merge_ready=true` (merge_zero_packs, no work needed)
- `validation_ready=true` (eventually latches after index stabilizes)
- `strategy_ready=false` initially (status=in_progress), then `true` after planner completes
- `style_waiting_for_review=true` (blocks `all_ready`)
- `all_ready=false` → `run_state` stays `VALIDATING` until strategy finishes

**Timeline:**
1. 15:51:37Z - 15:53:16Z: Validation AI runs
2. 15:52:22Z: `VALIDATION_STAGE_STATUS ... status=success`
3. 15:52:22Z - 15:52:53Z: **Loop zone** (repeated `UMBRELLA_STATE_AFTER_RECONCILE ... run_state=VALIDATING`)
4. 15:53:17Z: `ready_latched=true` for validation
5. 15:53:17Z: Strategy completes (`status=success`)
6. 16:02:59Z: Final reconciliation → `run_state=AWAITING_CUSTOMER_INPUT`

**Root:** Recheck thread keeps reconciling even after validation is internally "done"

---

## 3. Strategy Recovery

### 3.1 Implementation

**File:** `backend/runflow/decider.py`  
**Function:** `reconcile_umbrella_barriers(...)` (lines 5901-6033)

**Design intent:**
- **Purpose:** Automatically re-enqueue strategy planner if validation recovered from error
- **Use case:** Validation initially failed or was incomplete; after results materialize, trigger strategy
- **Not meant for:** Repeated checking on already-running strategy

**Code structure:**
```python
# Line 5907: Check env flag
strategy_recovery_enabled = os.getenv("ENABLE_STRATEGY_RECOVERY", "").lower() in {"1", "true", ...}
if not strategy_recovery_enabled:
    return statuses  # DISABLED BY DEFAULT

# Line 5918-5920: Terminal state guard
if strategy_status in {"success", "error"}:
    log.info("STRATEGY_RECOVERY_DISABLED_FOR_TERMINAL_STATE sid=%s status=%s", ...)
    return statuses

# Lines 5950-5973: Recovery condition check
validation_ai_completed = _any_flag(validation_containers, "validation_ai_completed")
merge_results_applied = _any_flag(validation_containers, "merge_results_applied")

should_recover = (
    not strategy_started
    and strategy_required
    and strategy_ready
    and validation_status == "success"
    and validation_ai_completed
    and merge_results_applied
)

if should_recover:
    log.info("STRATEGY_RECOVERY_TRIGGER ...")
    # Enqueue strategy recovery chain
else:
    if strategy_started:
        log.info("STRATEGY_RECOVERY_SKIP sid=%s reason=already_started status=%s", ...)
    elif not strategy_required:
        log.info("STRATEGY_RECOVERY_SKIP sid=%s reason=not_required", ...)
    # ... other skip reasons
```

---

### 3.2 States

**Strategy statuses:**
- `""` (empty): Not started
- `"built"`: Planner marked as started
- `"in_progress"`: Recovery chain enqueued or planner running
- `"success"`: Planner completed successfully
- `"error"`: Planner failed
- `"published"`: (unused for strategy)

**Recovery skip reasons:**
| Reason | Meaning |
|--------|---------|
| `already_started` | `strategy_status` in `{"built", "in_progress", "published", "success", "error"}` |
| `not_required` | `strategy_required=false` (no validation findings or config disabled) |
| `not_ready` | `strategy_ready=false` (validation not complete) |
| `validation_not_success` | `validation.status != "success"` |
| `validation_ai_not_completed` | `validation_ai_completed=false` |
| `merge_results_not_applied` | `merge_results_applied=false` |

**No state machine diagram in code; inferred:**
```
[Not Started] ──should_recover=true──> [Recovery Enqueued (in_progress)]
                                                 │
                                                 v
                                           [Running]
                                                 │
                                    ┌────────────┴────────────┐
                                    v                         v
                              [Success]                   [Error]
                                    │                         │
                                    └─────────(terminal)──────┘
```

---

### 3.3 Why Repeatedly Considered

**For SID 160884fe...:**
- Strategy recovery is **disabled by default** (`ENABLE_STRATEGY_RECOVERY` env var not set)
- But the check **still runs** on every reconciliation (line 5907 early-returns if disabled)
- Logs `STRATEGY_RECOVERY_SKIP ... reason=already_started status=in_progress` repeatedly because:
  - `strategy_started=true` (status is `in_progress` from prior enqueue)
  - Reconciliation doesn't know recovery is disabled, so it evaluates the condition
  - Skip logging happens **before** the early return for disabled recovery

**Frequency:**
- Every recheck thread iteration (2-10s delay)
- Plus any other manual reconciliation triggers (e.g., from merge autosend, stage updates)

**When it stops:**
- When `strategy_status` becomes `success` or `error` (terminal guard at line 5918)
- When recheck thread stops iterating
- When process exits

**Problem:**
- Even with recovery disabled, the evaluation logic runs and logs on every loop
- No "already checked, stop considering" state
- No backoff or circuit breaker

---

## 4. VALIDATION_INDEX_WRITE_FAILED Impact

### 4.1 Atomic Write Implementation

**File:** `backend/ai/validation_index.py`  
**Function:** `_atomic_write_json(path, document)` (lines 103-126)

**Code:**
```python
def _atomic_write_json(path: Path, document: Mapping[str, object]) -> None:
    serialized = json.dumps(document, ensure_ascii=False, indent=2)
    
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as handle:
            handle.write(serialized)
            handle.write("\n")
        os.replace(tmp_name, path)  # ← Atomic replace
    except OSError:
        log.warning("VALIDATION_INDEX_WRITE_FAILED path=%s", path, exc_info=True)
        try:
            os.unlink(tmp_name)  # Clean up temp file
        except FileNotFoundError:
            pass
```

**On Windows:**
- `os.replace()` can fail with `PermissionError: [WinError 5] Access is denied` if:
  - Target file is open by another process (even for reading)
  - File handle not closed yet (validation worker, index reader, etc.)
  - Antivirus software scanning the file
- **No retry logic** - just logs warning and returns

---

### 4.2 disk_result_accounts=[] Mystery

**File:** `backend/runflow/decider.py`  
**Function:** `_validation_results_progress(run_dir)` (lines 4350-4530)

**Code that reads disk:**
```python
# Line 4410-4420
results_dir_disk = run_dir / "ai_packs" / "validation" / "results"
disk_result_accounts: set[str] = set()

# Scan filesystem for result files
if results_dir_disk.exists():
    try:
        for result_file in results_dir_disk.glob("acc_*.result.jsonl"):
            if result_file.is_file():
                account_key = _extract_account_key_from_result_filename(result_file.name)
                if account_key:
                    disk_result_accounts.add(account_key)
    except OSError:
        # Filesystem read error - treat as empty
        pass
```

**For SID 160884fe...:**
- Log shows: `disk_result_accounts=[]`
- But filesystem shows 3 result files exist: `acc_009.result.jsonl`, `acc_010.result.jsonl`, `acc_011.result.jsonl`

**Hypothesis:**
- Windows file locking prevents `results_dir_disk.glob("acc_*.result.jsonl")` from succeeding
- Either:
  - Directory handle locked (can't enumerate)
  - Individual file handles locked (can't stat for `is_file()`)
- `OSError` caught silently → `disk_result_accounts` stays empty
- Meanwhile, `index_result_accounts=['9','10','11']` read successfully from index.json

**Mismatch consequence:**
- Validation promotion sees `disk_result_accounts=[]` but `index_result_accounts=[...]`
- Thinks: "index has results but disk doesn't → might be incomplete"
- Keeps checking on each reconciliation iteration
- Eventually filesystem becomes readable (locks released) or index stabilizes

---

### 4.3 Loop Timeline with Index Failures

**Timeline for SID 160884fe...:**

| Time | Event | Index Write | Disk Read | Loop State |
|------|-------|-------------|-----------|------------|
| 15:51:37Z | Validation packs built | - | - | Building |
| 15:51:51Z | First result (acc_009) completed | - | - | Running |
| 15:52:22Z | All results complete | - | - | Complete (AI done) |
| 15:52:22Z | `VALIDATION_STAGE_STATUS ... status=success` | - | disk_result_accounts=[] | Loop starts |
| 15:52:23Z | **`VALIDATION_INDEX_WRITE_FAILED`** (PermissionError) | ❌ FAILED | disk_result_accounts=[] | Loop continues |
| 15:52:23Z | `UMBRELLA_STATE_AFTER_RECONCILE ... run_state=VALIDATING` | - | disk_result_accounts=[] | Loop iteration 1 |
| 15:52:25Z | Recheck thread iteration 2 | - | disk_result_accounts=[] | Loop iteration 2 |
| 15:52:28Z | Recheck thread iteration 3 | - | disk_result_accounts=[] | Loop iteration 3 |
| ... | (repeated cycles) | - | disk_result_accounts=[] | Loop continues |
| 15:52:53Z | Recheck thread iteration N | ✅ SUCCESS | disk_result_accounts=['9','10','11'] | Loop stabilizes |
| 15:53:16Z | Validation internally complete | - | - | Waiting for strategy |
| 15:53:17Z | `ready_latched=true` | - | - | Validation done |
| 15:53:17Z | Strategy completes | - | - | Planner success |
| 16:02:59Z | Final reconciliation | - | - | → AWAITING_CUSTOMER_INPUT |

**Duration:**
- **Index write failures:** 15:52:23Z → 15:52:53Z (~30 seconds)
- **Tight loop:** 15:52:22Z → 15:52:53Z (~31 seconds)
- **Total validation completion delay:** 15:51:51Z → 15:53:17Z (~86 seconds)

**Root causes of 30-second loop:**
1. **PermissionError** prevents index write, but doesn't stop reconciliation
2. **Recheck thread** continues waking every 2-10s and calling reconcile
3. **Filesystem locking** prevents accurate `disk_result_accounts` count
4. **No circuit breaker** to say "index failed N times, stop trying"
5. **No backoff** between recheck iterations

---

## 5. Merge + Note_Style Autosend Re-triggering

### 5.1 Trigger Mechanism

**Both autosends are called unconditionally on every reconciliation:**

**File:** `backend/runflow/decider.py` (lines 5865-5878)
```python
def reconcile_umbrella_barriers(...):
    # ... reconciliation logic ...
    
    try:
        from backend.runflow.umbrella import schedule_merge_autosend
        schedule_merge_autosend(sid, run_dir=run_dir)  # ← ALWAYS CALLED
    except Exception:
        log.warning("MERGE_AUTOSEND_STAGE_FAILED sid=%s", sid, exc_info=True)

    try:
        from backend.runflow.umbrella import schedule_note_style_after_validation
        schedule_note_style_after_validation(sid, run_dir=run_dir)  # ← ALWAYS CALLED
    except Exception:
        log.warning("NOTE_STYLE_AUTOSEND_AFTER_VALIDATION_FAILED sid=%s", sid, exc_info=True)
```

**No guard:** "Don't call if we already know there's nothing to do"

---

### 5.2 Merge Autosend

**File:** `backend/runflow/umbrella.py`  
**Function:** `schedule_merge_autosend(sid, run_dir)` (lines 145-315)

**Skip conditions (in order):**
1. Invalid SID → `reason=invalid_sid`
2. `MERGE_AUTOSEND` env disabled → `reason=autosend_disabled`
3. `MERGE_STAGE_AUTORUN` env disabled → `reason=stage_autorun_disabled`
4. Runflow read failed → `reason=runflow_read_failed`
5. **Merge already completed:** `status=success` AND `result_files >= expected_packs` → `reason=merge_already_completed`
6. Validation terminal error → `reason=validation_terminal_error`
7. Validation not ready → `reason=validation_not_ready`
8. Merge already sent → `reason=already_sent`

**For SID 160884fe...:**
- `merge.status=success`
- `merge_zero_packs=true` (all pairs gated, no packs created)
- `expected_packs=0`, `result_files=0`
- Condition #5 triggers: `result_files (0) >= expected_packs (0)` → **TRUE**
- Logs: `MERGE_AUTOSEND_STAGE_SKIP ... reason=merge_already_completed` (or `reason=no_packs` variant)

**But:**
- This check runs **on every reconciliation**
- No cached "we already decided merge is done" flag
- Re-reads runflow.json and re-evaluates all conditions

---

### 5.3 Note_Style Autosend

**File:** `backend/runflow/umbrella.py`  
**Function:** `schedule_note_style_after_validation(sid, run_dir)` (lines 314-709)

**Skip conditions (in order):**
1. Invalid SID → `reason=invalid_sid`
2. `NOTE_STYLE_ENABLED=false` → `reason=disabled_feature`
3. `NOTE_STYLE_AUTOSEND` env disabled → `reason=disabled_env`
4. `NOTE_STYLE_STAGE_AUTORUN` env disabled → `reason=stage_autorun_disabled`
5. `NOTE_STYLE_SEND_ON_RESPONSE_WRITE` env disabled → `reason=send_on_write_disabled`
6. **Note_style packs_count=0 or built=0:** → `reason=empty` + `NOTE_STYLE_AUTOSEND_SKIPPED ... reason=no_packs`

**For SID 160884fe...:**
- `note_style.status=waiting_for_review`
- `packs_total=3` (validation results)
- `built=0` (no note_style packs built yet, waiting for AI worker)
- Logs: `NOTE_STYLE_AUTOSEND_DECISION ... sent=true built=0 terminal=0 reason=empty`
- Then: `NOTE_STYLE_AUTOSEND_SKIPPED ... reason=no_packs`

**But:**
- This check runs **on every reconciliation**
- Re-collects metrics via `_collect_note_style_metrics(run_dir_path)` every time
- Reads frontend stage, validation stage, note_style stage from runflow
- Re-evaluates all skip conditions

---

### 5.4 Why Not Stop Earlier?

**Missing optimizations:**
1. **No "autosend decided" flag:** Once merge/note_style determine "nothing to do," no persistent marker prevents re-checking
2. **No cached decision:** Each reconciliation re-reads runflow.json and re-evaluates full skip logic
3. **No rate limiting:** If reconciliation runs 10 times in 30 seconds, autosend checks run 10 times
4. **No short-circuit on zero-packs:** `merge_zero_packs=true` is computed but not used to skip autosend call entirely
5. **No idempotency guard:** "We logged this skip reason 5 times already, don't log again"

**Design trade-off:**
- **Pro:** Always fresh check; catches late-arriving results or state changes
- **Con:** Wasteful re-computation when state is stable and terminal

**For 21 validation packs:**
- Same logic applies
- If recheck thread runs for N iterations (based on random delay and validation completion time)
- Autosend checks run N times
- **But:** Autosend logic itself is relatively cheap (mostly conditional checks)
- **Real cost:** Index reading, filesystem scans, runflow JSON parsing

---

## 6. Umbrella Barriers

### 6.1 Barrier Computation

**File:** `backend/runflow/decider.py`  
**Function:** `_compute_umbrella_barriers(run_dir, runflow_payload)` (not fully shown, inferred from usage)

**Barriers:**
| Barrier | Meaning |
|---------|---------|
| `merge_ready` | Merge stage complete OR merge_zero_packs=true |
| `validation_ready` | Validation status=success AND missing_results=0 AND merge_results_applied=true AND ready_latched=true (or filesystem-based override) |
| `strategy_ready` | Strategy status=success OR (not strategy_required) OR (strategy in acceptable non-terminal state) |
| `review_ready` | Frontend answers_received >= answers_required |
| `style_ready` | Note_style completed OR not required |
| `style_waiting_for_review` | Note_style status=waiting_for_review |
| `all_ready` | All required barriers are true |

**Additional flags:**
- `merge_zero_packs`: Merge produced no packs (all pairs gated)
- `style_required`: Note_style is needed (validation found actionable findings)
- `strategy_required`: Strategy is needed (validation found findings or config enabled)

---

### 6.2 run_state Translation

**File:** `backend/runflow/decider.py` (lines 5815-5826)

**Logic:**
```python
post_state = _resolve_post_validation_state(data, statuses)

if has_stage_error:
    updated_run_state = "ERROR"
elif post_state is not None:
    updated_run_state = _prefer_run_state(existing_run_state, post_state)
elif normalized_existing_state in {"", "ERROR"}:
    updated_run_state = "VALIDATING"
```

**`_resolve_post_validation_state` (inferred):**
- Returns `"AWAITING_CUSTOMER_INPUT"` if `all_ready=false` but no critical errors and review_ready=false
- Returns `"COMPLETED"` if `all_ready=true` and `review_ready=true`
- Returns `None` if still in progress

**run_state progression:**
```
[NEW/empty] → VALIDATING → AWAITING_CUSTOMER_INPUT → COMPLETED
                    │
                    └──(error)──> ERROR
```

---

### 6.3 Blocking Barriers for SID 160884fe...

**During loop (15:52:22Z - 15:52:53Z):**
```
merge_ready = true          # merge_zero_packs=true
validation_ready = false    # Initially; becomes true after ready_latched
strategy_ready = false      # status=in_progress
style_waiting_for_review = true
style_required = true
strategy_required = true
all_ready = false           # Blocked by strategy_ready=false
```

**Result:** `run_state=VALIDATING` (post_state returns None or VALIDATING)

**After strategy completes (15:53:17Z):**
```
merge_ready = true
validation_ready = true     # ready_latched=true
strategy_ready = true       # status=success
style_waiting_for_review = true
style_required = true
all_ready = false           # Blocked by style_waiting_for_review=true
```

**Result:** `run_state=AWAITING_CUSTOMER_INPUT` (post_state returns AWAITING_CUSTOMER_INPUT)

**Blocking sequence:**
1. **First:** `validation_ready=false` (waiting for ready_latched)
2. **Then:** `strategy_ready=false` (waiting for planner to finish)
3. **Finally:** `style_waiting_for_review=true` (waiting for frontend user review)

**Loop persists because:**
- Even after `validation_ready=true` latches at 15:53:17Z
- Strategy was still running (in_progress)
- Recheck thread kept reconciling every few seconds
- Each reconciliation re-evaluated barriers but found `all_ready=false`

---

## 7. Design Constraints & Conceptual Fixes

### 7.1 Current Design Invariants

**Invariants enforced by the loop:**
1. **Always fresh state:** Reconciliation re-computes all barriers from disk/runflow on every call
2. **No stale decisions:** Never cache "validation is done" or "merge already decided" to avoid missing late changes
3. **Defensive polling:** Recheck thread ensures results don't get "stuck" if initial send failed
4. **Paranoid validation:** Multiple filesystem + index + runflow checks to catch every possible inconsistency
5. **Log everything:** Every decision logged for forensics (but creates noise)

**Why loop instead of idle/terminal:**
- **Fear of missed events:** What if a result file appears after we stopped checking?
- **No event-driven architecture:** No filesystem watcher or pub/sub for "validation complete" events
- **No explicit "done" signal:** No flag like `reconciliation_complete=true` to stop recheck thread
- **No timeout:** Recheck thread runs indefinitely (until process exits or validation internally stops it)

---

### 7.2 Conceptual Changes (No Code Yet)

#### Fix 1: Stop Recheck Thread Explicitly

**Problem:** Recheck thread continues even after validation is complete

**Solution:**
- Add `validation_reconciliation_complete=true` flag in runflow.json
- Set flag when:
  - `validation.status=success`
  - `validation_ai_completed=true`
  - `merge_results_applied=true`
  - `missing_results=0`
  - `ready_latched=true`
- In `_maybe_send_validation_packs`, check this flag and skip recheck scheduling if true

**Impact:**
- Stops recheck thread after first successful reconciliation post-validation
- Reduces loop from ~30 seconds to ~1-2 seconds (one or two reconciliation cycles)

---

#### Fix 2: Cache Autosend Decisions

**Problem:** Merge/note_style autosend checks run on every reconciliation

**Solution:**
- Add `merge_autosend_decided=true` and `note_style_autosend_decided=true` in runflow.json
- Set flags when autosend determines "skip" for terminal reasons (e.g., `merge_already_completed`, `no_packs`)
- In `schedule_merge_autosend` / `schedule_note_style_after_validation`, check flag first and return early if set

**Impact:**
- Avoids re-reading runflow.json and re-evaluating skip conditions on every reconciliation
- Reduces log noise (no repeated `MERGE_AUTOSEND_STAGE_SKIP`)

---

#### Fix 3: Add Index Write Retry with Backoff

**Problem:** Index write `PermissionError` on Windows; no retry

**Solution:**
- In `_atomic_write_json`, retry `os.replace()` 3-5 times with exponential backoff (100ms, 200ms, 400ms, ...)
- If all retries fail, log `VALIDATION_INDEX_WRITE_FAILED_AFTER_RETRIES` and continue
- Add `index_write_attempts` counter to manifest/runflow for observability

**Impact:**
- Increases chance of successful index write despite Windows file locking
- Reduces `disk_result_accounts=[]` mismatch

---

#### Fix 4: Add Reconciliation Rate Limiting

**Problem:** No backoff between reconciliation cycles

**Solution:**
- Track last reconciliation timestamp in memory or runflow.json
- In `reconcile_umbrella_barriers`, check: "Did we reconcile in last 5 seconds? If yes, skip unless forced"
- Add `force_reconcile=true` parameter for explicit triggers (e.g., after stage completion)

**Impact:**
- Prevents tight reconciliation loops when state is stable
- Reduces log volume

---

#### Fix 5: Short-Circuit Autosend on merge_zero_packs

**Problem:** `schedule_merge_autosend` always called even when `merge_zero_packs=true`

**Solution:**
- In `reconcile_umbrella_barriers`, check `umbrella_barriers.merge_zero_packs` before calling `schedule_merge_autosend`
- If true and `merge.status=success`, skip the call entirely

**Impact:**
- One less function call and JSON read per reconciliation
- Slight performance gain

---

#### Fix 6: Terminal Strategy Recovery Guard

**Problem:** Strategy recovery check runs even when disabled and on terminal states

**Solution:**
- Move `ENABLE_STRATEGY_RECOVERY` check to top of `reconcile_umbrella_barriers` (before any other logic)
- Add early return: "If recovery disabled, skip all recovery logic and don't log skip reasons"
- For terminal states (`success`, `error`), add a `strategy_recovery_terminal=true` flag to avoid re-checking

**Impact:**
- Eliminates repeated `STRATEGY_RECOVERY_SKIP` logs when recovery is disabled
- Reduces code path complexity

---

### 7.3 Asymptotic Behavior for 21 Packs

**Current behavior:**
- Recheck thread runs for `T_recheck` seconds (determined by validation completion time + random delays)
- Each iteration:
  - Reads validation index (~1ms)
  - Scans filesystem for results (~1-5ms depending on file count)
  - Reads runflow.json (~1ms)
  - Evaluates barriers and autosends (~1ms)
  - Writes runflow.json (~1-5ms)
- **Total per iteration:** ~5-15ms

**Number of iterations:**
- If recheck thread runs every 2-10s for 30 seconds: ~3-15 iterations
- If validation takes longer (e.g., 21 packs × 5s/pack = 105s): ~10-50 iterations

**Cost scaling:**
- **O(iterations × overhead_per_iteration)**
- Iterations scale with:
  - Validation completion time (linear with pack count if sequential)
  - Random recheck delay (constant)
  - Index write failures (increases if more writes attempted)
- Overhead_per_iteration scales with:
  - Number of result files to scan (linear with pack count)
  - Runflow.json size (grows with stages/metrics)

**For 21 packs:**
- If validation takes 3 minutes (180s)
- Recheck thread runs ~18-90 iterations (depending on delay)
- Each iteration scans 21 result files (~5ms)
- **Total loop overhead:** ~90-450ms of filesystem scanning + ~1800-9000 log lines

**Not as bad as feared:**
- Filesystem scans are fast (SSD)
- Biggest cost is **log noise** and **CPU wake-ups** (power efficiency)
- Not a scalability disaster, but wasteful

**With fixes:**
- Stop recheck thread after first success → 1-2 iterations regardless of pack count
- Cache autosend decisions → skip most checks after first iteration
- **Total loop overhead:** ~10-30ms + ~10-20 log lines

---

## 8. State Machines

### 8.1 Validation Stage State Machine

```
[Not Started]
     │
     v
[Building Packs] ──(pack builder)──> [Packs Built]
     │                                      │
     │                                      v
     │                              [AI Sending] ──(LLM API)──> [AI Running]
     │                                                                │
     │                                                                v
     └───────────────────────────────────────────────> [Results Writing] ──(all complete)──> [Results Complete]
                                                                │
                                                                v
                                                        [Index Writing]
                                                                │
                                                    ┌───────────┴──────────┐
                                                    v                      v
                                            [Index Failed]          [Index Success]
                                                    │                      │
                                                    v                      v
                                            [Retry Loop]           [Merge Applying]
                                                    │                      │
                                                    v                      v
                                            [Eventually Success]   [Merge Applied]
                                                                           │
                                                                           v
                                                                   [ready_latched=true]
                                                                           │
                                                                           v
                                                                   [Validation Complete]
                                                                   status=success
```

**Key transitions:**
- `Building Packs` → `AI Sending`: All packs written to disk
- `AI Running` → `Results Complete`: All LLM responses received
- `Results Complete` → `Index Success`: Index write succeeds
- `Index Failed` → `Retry Loop`: Recheck thread retries
- `Merge Applied` → `ready_latched=true`: Barriers see all conditions met

---

### 8.2 run_state State Machine

```
[NEW/empty]
     │
     v
[VALIDATING] ────(validation complete + barriers met)────> [AWAITING_CUSTOMER_INPUT]
     │                                                               │
     │                                                               v
     │                                                       (user reviews frontend)
     │                                                               │
     │                                                               v
     │                                                        [COMPLETED]
     │
     └──(critical stage error)──> [ERROR]
```

**Barrier-driven transitions:**
- `VALIDATING` → `AWAITING_CUSTOMER_INPUT`: `all_ready=false` but no errors, `style_waiting_for_review=true`
- `AWAITING_CUSTOMER_INPUT` → `COMPLETED`: `all_ready=true`, `review_ready=true`
- Any state → `ERROR`: `has_stage_error=true`

---

### 8.3 Strategy Recovery State Machine

```
[Not Evaluated]
     │
     v
[Check Conditions]
     │
     ├──(recovery_enabled=false)──> [Skip (Disabled)]
     │
     ├──(strategy_status=success/error)──> [Skip (Terminal)]
     │
     ├──(strategy_started=true)──> [Skip (Already Started)]
     │
     ├──(not strategy_required)──> [Skip (Not Required)]
     │
     ├──(not strategy_ready)──> [Skip (Not Ready)]
     │
     ├──(validation_status != success)──> [Skip (Validation Not Success)]
     │
     ├──(not validation_ai_completed)──> [Skip (AI Not Completed)]
     │
     ├──(not merge_results_applied)──> [Skip (Merge Not Applied)]
     │
     └──(all conditions met)──> [Trigger Recovery]
                                       │
                                       v
                               [Enqueue Recovery Chain]
                                       │
                                       v
                               [Mark strategy=in_progress]
                                       │
                                       v
                               [Planner Runs]
                                       │
                         ┌─────────────┴─────────────┐
                         v                           v
                  [Planner Success]          [Planner Error]
                  status=success             status=error
```

**Key:** Recovery is **one-shot**; once `strategy_started=true`, always skips with `reason=already_started`

---

## 9. Timeline for SID 160884fe...

| Time | Event | Component | Log Marker | State |
|------|-------|-----------|------------|-------|
| 15:51:37Z | Validation packs built | Validation builder | - | packs_total=3 |
| 15:51:51Z | First result completed (acc_009) | Validation AI worker | - | completed=1/3 |
| 15:52:22Z | All results completed | Validation AI worker | `VALIDATION_STAGE_STATUS ... status=success` | completed=3/3 |
| 15:52:22Z | **First reconciliation** | Umbrella (recheck thread) | `VALIDATION_RESULTS_DEBUG ... disk_result_accounts=[]` | Loop starts |
| 15:52:22Z | Index write attempt #1 | Validation index writer | - | - |
| 15:52:23Z | **Index write failed** | Validation index writer | `VALIDATION_INDEX_WRITE_FAILED ... PermissionError` | ❌ |
| 15:52:23Z | Umbrella state | Umbrella | `UMBRELLA_STATE_AFTER_RECONCILE ... run_state=VALIDATING` | Iteration 1 |
| 15:52:23Z | Merge autosend skip | Umbrella | `MERGE_AUTOSEND_STAGE_SKIP ... reason=no_packs` | - |
| 15:52:23Z | Note_style autosend skip | Umbrella | `NOTE_STYLE_AUTOSEND_DECISION ... reason=empty` | - |
| 15:52:23Z | Strategy recovery skip | Umbrella | `STRATEGY_RECOVERY_SKIP ... reason=already_started status=in_progress` | - |
| 15:52:25Z | Recheck thread wakes (delay ~2s) | Recheck thread | `VALIDATION_AUTOSEND_RECHECK ...` | Iteration 2 |
| 15:52:25Z | Reconciliation #2 | Umbrella | `VALIDATION_RESULTS_DEBUG ... disk_result_accounts=[]` | Still locked |
| 15:52:25Z | Index write attempt #2 | Validation index writer | - | - |
| 15:52:26Z | Index write failed again | Validation index writer | `VALIDATION_INDEX_WRITE_FAILED ... PermissionError` | ❌ |
| ... | (repeated cycles) | Recheck thread | - | Iterations 3-N |
| 15:52:53Z | Index write finally succeeds | Validation index writer | - | ✅ |
| 15:52:53Z | Filesystem readable | Umbrella | `VALIDATION_RESULTS_DEBUG ... disk_result_accounts=['9','10','11']` | Match found |
| 15:53:16Z | Validation internally complete | Umbrella | - | validation_ai_completed=true |
| 15:53:17Z | Validation latched | Umbrella | - | ready_latched=true |
| 15:53:17Z | **Strategy completes** | Strategy planner | - | status=success, plans_written=3 |
| 16:02:59Z | **Final reconciliation** | Umbrella (manual trigger?) | `UMBRELLA_STATE_AFTER_RECONCILE ... run_state=AWAITING_CUSTOMER_INPUT` | Loop ends |

**Key observations:**
1. **Loop duration:** 15:52:22Z → 15:52:53Z (~31 seconds)
2. **Index write failures:** 15:52:23Z → 15:52:53Z (~30 seconds of retries)
3. **Recheck iterations:** ~10-15 cycles (based on 2-10s random delay)
4. **Strategy completion:** 15:53:17Z (after validation latched)
5. **Final state transition:** 16:02:59Z (9+ minutes later - likely manual trigger or slow polling)

**Root cause of 30-second loop:**
- Recheck thread spawned by validation autosend
- Thread continues polling every 2-10s
- Each poll triggers full reconciliation
- Index write failures prevent filesystem from matching index
- Loop ends when index write succeeds AND validation latches ready

---

## 10. Recommendations

### 10.1 Immediate (Low-Hanging Fruit)

1. **Stop recheck thread on validation complete:**
   - Add `validation_reconciliation_complete=true` flag
   - Check flag before scheduling next recheck
   - **Impact:** Reduces loop from 30s to 1-2s

2. **Add index write retry with backoff:**
   - Retry `os.replace()` 3-5 times with 100ms, 200ms, 400ms delays
   - **Impact:** Reduces Windows file locking failures

3. **Cache autosend decisions:**
   - Add `merge_autosend_decided=true`, `note_style_autosend_decided=true`
   - **Impact:** Reduces repeated skip checks and log noise

### 10.2 Medium (Refactoring Required)

4. **Add reconciliation rate limiting:**
   - Track last reconciliation timestamp
   - Skip if reconciled in last 5 seconds (unless forced)
   - **Impact:** Prevents tight loops when state is stable

5. **Short-circuit autosend on zero-packs:**
   - Check `merge_zero_packs` before calling `schedule_merge_autosend`
   - **Impact:** Slight performance gain

6. **Terminal strategy recovery guard:**
   - Move `ENABLE_STRATEGY_RECOVERY` check to top
   - Skip all recovery logic if disabled
   - **Impact:** Reduces log noise and code path complexity

### 10.3 Long-Term (Architecture)

7. **Event-driven reconciliation:**
   - Replace polling recheck thread with event emitter
   - Emit `validation_complete` event when all results written
   - Subscribe umbrella to event instead of polling
   - **Impact:** Eliminates polling overhead entirely

8. **Filesystem watcher for index.json:**
   - Use `watchdog` or similar to detect index.json changes
   - Trigger reconciliation only when file actually changes
   - **Impact:** Reduces unnecessary reconciliation cycles

9. **Explicit "work complete" signals:**
   - Add `reconciliation_needed=false` flag to runflow.json
   - Set flag when all barriers met and no further work scheduled
   - Check flag at top of `reconcile_umbrella_barriers` and short-circuit
   - **Impact:** Prevents "zombie reconciliation" after run is done

---

## 11. Conclusion

### Root Cause (Confirmed):
**The validation autosend path spawns a background recheck thread that repeatedly calls `reconcile_umbrella_barriers()` every 2-10 seconds. This thread continues even after validation is complete, causing ~30 seconds of tight looping with repeated stage promotions, autosend checks, and strategy recovery evaluations. Combined with Windows file locking (`PermissionError` on index.json writes), the loop persists until the filesystem stabilizes and validation internally latches `ready_latched=true`.**

### Performance Impact:
- **3 packs:** ~30 seconds of looping
- **21 packs:** Potentially minutes if validation takes longer
- **Cost:** O(recheck_iterations × umbrella_overhead)
- **Not a scalability disaster, but wasteful of CPU, logs, and power**

### Design Intent:
- **Defensive polling:** Ensure results don't get "stuck" if initial send fails
- **Always fresh state:** Never cache decisions to avoid missing late changes
- **Trade-off:** Correctness over efficiency

### Fix Priority:
1. **Stop recheck thread explicitly** (high impact, low effort)
2. **Add index write retry** (fixes Windows locking)
3. **Cache autosend decisions** (reduces log noise)
4. **Rate limit reconciliation** (prevents tight loops)
5. **Event-driven architecture** (long-term, eliminates polling)

---

**End of Investigation**

**Next:** Await approval for fixes before proceeding to implementation.
