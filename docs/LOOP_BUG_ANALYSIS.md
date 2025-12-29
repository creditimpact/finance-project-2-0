# Pipeline Loop & Manifest Overwrite Bug Analysis
**SID:** `b4cf2c3e-7f7c-4f7b-9087-7bff671a2b0e`  
**Date:** 2025-11-16  
**Status:** Root cause identified, fixes proposed

---

## Executive Summary

The pipeline is **NOT actually looping infinitely**. The system has reached a **stable idle state** (`reconciliation_idle=true`) but the root cause of repeated log messages is:

1. **review_ready=False** - Frontend review stage shows `answers_required=3` but has no mechanism to become "ready" without actual customer responses
2. **style_ready=False, style_waiting_for_review=True** - Note style stage is blocked waiting for review
3. **Validation paths ARE missing from manifest** - `ai.validation` and `ai.packs.validation` sections have NULL paths even though validation completed successfully

The "loop" symptoms are actually the **umbrella barriers reconciliation running repeatedly** (every 2-10 seconds via watchdog thread) but correctly **NOT re-enqueuing tasks** thanks to the idle detection logic that was already implemented.

However, there are **TWO CRITICAL BUGS**:

**Bug #1:** Validation pack paths are never persisted to `manifest.json` during validation builder execution  
**Bug #2:** Frontend `review_ready` barrier never transitions to `true` even when packs exist and are published

---

## Bug #1: Validation Paths Missing from Manifest

### Evidence

From `manifest.json`:
```json
{
  "ai": {
    "packs": {
      "validation": {
        "base": null,
        "dir": null,
        "packs": null,
        "packs_dir": null,
        "results": null,
        "results_dir": null,
        "index": null,
        "last_built_at": null,
        "logs": null,
        "status": {
          "sent": true,
          "completed_at": "2025-11-16T17:33:16Z"
        }
      }
    },
    "validation": {
      "base": null,
      "dir": null,
      "accounts": null,
      "accounts_dir": null,
      "last_prepared_at": null
    },
    "status": {
      "validation": {
        "built": false,  ← WRONG! Should be true
        "sent": true,
        "completed_at": "2025-11-16T17:33:17Z",
        "merge_results_applied": true
      }
    }
  }
}
```

Yet from `runflow.json`, validation clearly succeeded:
```json
{
  "stages": {
    "validation": {
      "status": "success",
      "metrics": {
        "packs_total": 3,
        "validation_ai_completed": true,
        "results_total": 3,
        "completed": 3,
        "failed": 0
      }
    }
  }
}
```

### Root Cause

**File:** `backend/ai/validation_builder.py`  
**Function:** `_update_manifest_for_run()` (lines 2275-2295)  
**Issue:** This function is NEVER CALLED during the inline validation builder path

The validation builder has two execution paths:

1. **Inline path** (used by strategy_planner_step):
   - Calls `build_validation_packs_inline()` 
   - Directly applies results via `apply_validation_ai_decisions_for_all_accounts()`
   - **NEVER calls `_update_manifest_for_run()`**
   - Result: validation completes but paths are never saved to manifest

2. **Task path** (legacy/disabled):
   - Would call `_update_manifest_for_run()` 
   - Properly updates manifest via `manifest.upsert_validation_packs_dir()`
   - But this path is disabled by default

**File:** `backend/ai/validation_builder.py` line 1995-2050
```python
def build_validation_packs_inline(
    sid: str,
    runs_root: Path | str,
    *,
    source: str = "inline",
) -> dict[str, Any]:
    # ... builds validation packs ...
    
    # BUG: Never calls _update_manifest_for_run() here!
    # The manifest is left with null paths even though
    # validation succeeded and files exist on disk
    
    return result
```

Compare to the legacy path that DOES update manifest (line 2275):
```python
def _update_manifest_for_run(sid: str, runs_root: Path | str) -> None:
    runs_root_path = Path(runs_root).resolve()
    base_dir = validation_base_dir(sid, runs_root=runs_root_path, create=True)
    packs_dir = validation_packs_dir(sid, runs_root=runs_root_path, create=True)
    results_dir = validation_results_dir(sid, runs_root=runs_root_path, create=True)
    index_path = validation_index_path(sid, runs_root=runs_root_path, create=True)
    log_path = validation_logs_path(sid, runs_root=runs_root_path, create=True)

    manifest_path = runs_root_path / sid / "manifest.json"
    manifest = RunManifest.load_or_create(manifest_path, sid)
    manifest.upsert_validation_packs_dir(
        base_dir,
        packs_dir=packs_dir,
        results_dir=results_dir,
        index_file=index_path,
        log_file=log_path,
    )
```

### Impact

- Manifest shows `ai.status.validation.built=false` even though validation succeeded
- Manifest shows all validation path fields as `null`
- This causes confusion and potential issues if code relies on manifest for validation paths
- However, **this does NOT cause the loop** because the umbrella barriers read from `runflow.json`, which correctly shows validation status

---

## Bug #2: review_ready Barrier Never Becomes True

### Evidence

From `runflow.json`:
```json
{
  "umbrella_barriers": {
    "merge_ready": true,
    "validation_ready": true,
    "strategy_ready": true,
    "review_ready": false,  ← STUCK FALSE
    "style_ready": false,
    "all_ready": false
  },
  "stages": {
    "frontend": {
      "status": "published",
      "packs_count": 3,
      "metrics": {
        "answers_received": 0,
        "answers_required": 3  ← Requires 3 responses
      }
    }
  }
}
```

From `manifest.json`:
```json
{
  "frontend": {
    "built": true,
    "packs_count": 3,
    "last_built_at": "2025-11-16T17:31:36Z"
  }
}
```

### Root Cause

**File:** `backend/runflow/decider.py`  
**Function:** `_compute_umbrella_barriers()` (lines 5150-5168)

```python
def _compute_umbrella_barriers(
    run_dir: Path,
    *,
    runflow_payload: Mapping[str, Any] | None = None,
) -> dict[str, bool]:
    # ...
    review_required, review_received, review_ready_disk = _frontend_responses_progress(
        run_dir
    )
    has_frontend_stage = isinstance(frontend_stage, Mapping)
    review_disk_evidence = review_required > 0 or review_received > 0
    review_ready = False
    if has_frontend_stage or review_disk_evidence:
        review_ready = review_ready_disk and review_received >= review_required  # ← BUG HERE
    if not review_ready and _stage_status_success(frontend_stage):
        metrics_payload = frontend_stage.get("metrics")
        if isinstance(metrics_payload, Mapping):
            required = _coerce_int(metrics_payload.get("answers_required"))
            received = _coerce_int(metrics_payload.get("answers_received"))
            if required is not None and received is not None and received == required:
                review_ready = True  # ← This check also requires responses
        if not review_ready and _stage_empty_ok(frontend_stage):
            review_ready = True  # ← This would work but empty_ok=true is not set on frontend stage
```

**The Problem:**
1. `answers_required=3` and `answers_received=0`
2. `review_received >= review_required` is `False`
3. The fallback check for `empty_ok` doesn't help because `frontend.empty_ok=true` in runflow but the logic doesn't recognize it properly
4. There's no logic to say "frontend is published and packs exist, so review_ready should be true even without responses"

### Why This Keeps Umbrella From Advancing

**File:** `backend/runflow/decider.py` line 5321
```python
all_ready = (
    merge_ready
    and validation_ready
    and review_ready  # ← False blocks all_ready
    and style_ready   # ← Also False (style_waiting_for_review=True)
    and strategy_ready
)
```

Since `review_ready=False` and `style_ready=False`, `all_ready` can never become `True`.

However, the system correctly handles this by:
1. Setting `run_state=AWAITING_CUSTOMER_INPUT` (correct state)
2. Setting `reconciliation_idle=true` to prevent re-enqueuing tasks
3. Only running reconciliation checks, not actual pipeline work

---

## Why There's No Actual Infinite Loop

### Idle State Detection (Already Implemented!)

**File:** `backend/runflow/decider.py` lines 5915-5970

The system already has **excellent idle detection logic**:

```python
def reconcile_umbrella_barriers(
    sid: str, runs_root: Optional[str | Path] = None
) -> dict[str, bool]:
    # ... compute barriers ...
    
    # --- Detect idle/stable state to reduce unnecessary reconciliation work ---
    validation_ready = bool(statuses.get("validation_ready"))
    merge_ready = bool(statuses.get("merge_ready"))
    strategy_ready = bool(statuses.get("strategy_ready"))
    
    reconciliation_idle = (
        validation_ready
        and merge_ready
        and (not strategy_required or strategy_ready)
        and (merge_autosend_finalized or run_state_terminal)
        and (note_style_autosend_finalized or run_state_terminal)
    )
    
    if reconciliation_idle:
        umbrella["reconciliation_idle"] = True
        log.info("UMBRELLA_IDLE_STATE_REACHED ...")
    else:
        umbrella["reconciliation_idle"] = False
    
    # Skip autosend checks if idle (autosends already finalized or terminal state)
    if not reconciliation_idle:
        schedule_merge_autosend(sid, run_dir=run_dir)
        schedule_note_style_after_validation(sid, run_dir=run_dir)
    else:
        log.debug("UMBRELLA_IDLE_SKIP_AUTOSENDS sid=%s", sid)
```

From the actual `runflow.json`:
```json
{
  "umbrella_barriers": {
    "reconciliation_idle": true,
    "reconciliation_idle_at": "2025-11-16T17:52:10Z"
  },
  "stages": {
    "merge": {
      "autosend_finalized": true
    },
    "note_style": {
      "autosend_finalized": true,
      "waiting_for_review": true
    }
  }
}
```

**This proves the idle detection is working!** The system is correctly:
- NOT re-enqueuing strategy/validation/frontend tasks
- NOT running autosend logic
- Only logging reconciliation updates when barriers change

### Why Logs Repeat

The repeated log messages you're seeing are from:
1. **Umbrella watchdog thread** - runs every 5 seconds (`UMBRELLA_BARRIERS_WATCHDOG_INTERVAL_MS=5000`)
2. **Manual reconcile calls** - from other parts of the code checking state
3. **Task completion handlers** - final tasks writing "completed" status

But crucially, **NO ACTUAL WORK IS BEING RE-ENQUEUED** thanks to:
- `reconciliation_idle=true` preventing autosends
- Short-circuit guards in tasks (e.g., `STRATEGY_RUN_SHORT_CIRCUIT ... reason=stage_success`)
- Terminal state checks (strategy/validation/merge all show `status=success`)

---

## Are Multiple Tasks Racing?

**Answer: NO, but they could have in the past**

### Evidence of Task Coordination

From logs you mentioned:
```
Task ... strategy_planner_step[...] received
STRATEGY_TASK_MANIFEST_START ...
STRATEGY_RUN_SHORT_CIRCUIT ... reason=stage_success  ← GOOD! Not re-running
```

**File:** `backend/pipeline/auto_ai_tasks.py` lines 1220-1245
```python
@celery_app.task(...)
def strategy_planner_step(...):
    # ... 
    manifest = RunManifest.for_sid(sid, runs_root=runs_root_path, allow_create=False)
    
    # Check if already completed
    strategy_status = manifest.get_ai_stage_status("strategy")
    if strategy_status.get("state") == "success":
        logger.info(
            "STRATEGY_RUN_SHORT_CIRCUIT sid=%s reason=stage_success",
            sid,
        )
        return {"sid": sid, "ok": True, "reason": "already_success"}
```

This short-circuit logic **prevents actual re-execution** even if the task is enqueued again.

### Manifest Write Safety

**File:** `backend/pipeline/runs.py` lines 77-125 (`safe_replace()`)

The manifest uses atomic writes:
1. Write to temp file
2. fsync to disk
3. `os.replace()` (atomic on both Windows and POSIX)
4. Retry logic for Windows permission errors

This prevents corruption, but doesn't prevent **logical races** where two tasks read stale data and write competing updates.

### Where Races Could Occur

**Scenario:**
1. Task A reads `manifest.json` (validation not marked as built)
2. Task B reads `manifest.json` (same stale state)
3. Task A updates strategy section, writes manifest
4. Task B updates validation section, writes manifest ← **overwrites Task A's changes**

**Mitigations:**
- Each task typically only updates its own section (strategy updates `ai.status.strategy`, validation updates `ai.status.validation`)
- The `ensure_ai_stage_status()` method preserves other sections
- Short-circuit guards prevent most race scenarios

**However:** The validation inline builder path never writes to manifest at all, so this race is avoided by omission (which is itself a bug).

---

## Fixes Required

### Fix #1: Update Validation Manifest in Inline Builder

**File:** `backend/ai/validation_builder.py`  
**Function:** `build_validation_packs_inline()`  
**Line:** ~2040 (after packs are built, before return)

**ADD THIS CODE:**
```python
def build_validation_packs_inline(
    sid: str,
    runs_root: Path | str,
    *,
    source: str = "inline",
) -> dict[str, Any]:
    # ... existing pack building logic ...
    
    # Write results
    _write_validation_builder_results(result, base_dir, logs_path)
    
    # ✅ FIX: Update manifest with validation pack paths
    try:
        _update_manifest_for_run(sid, runs_root)
        logger.info(
            "VALIDATION_MANIFEST_UPDATED sid=%s source=%s paths_persisted=True",
            sid,
            source,
        )
    except Exception:
        logger.warning(
            "VALIDATION_MANIFEST_UPDATE_FAILED sid=%s source=%s",
            sid,
            source,
            exc_info=True,
        )
    
    return result
```

This ensures validation paths are persisted even when using the inline builder.

---

### Fix #2: Correct review_ready Logic for Published Frontend

**File:** `backend/runflow/decider.py`  
**Function:** `_compute_umbrella_barriers()`  
**Lines:** 5150-5168

**CURRENT CODE:**
```python
review_ready = False
if has_frontend_stage or review_disk_evidence:
    review_ready = review_ready_disk and review_received >= review_required
if not review_ready and _stage_status_success(frontend_stage):
    metrics_payload = frontend_stage.get("metrics")
    if isinstance(metrics_payload, Mapping):
        required = _coerce_int(metrics_payload.get("answers_required"))
        received = _coerce_int(metrics_payload.get("answers_received"))
        if required is not None and received is not None and received == required:
            review_ready = True
    if not review_ready and _stage_empty_ok(frontend_stage):
        review_ready = True
```

**REPLACE WITH:**
```python
review_ready = False
if has_frontend_stage or review_disk_evidence:
    review_ready = review_ready_disk and review_received >= review_required

# ✅ FIX: Allow review_ready=true when frontend is published, even without responses
if not review_ready:
    frontend_status = _normalize_stage_status_value(
        frontend_stage.get("status") if isinstance(frontend_stage, Mapping) else None
    )
    
    if frontend_status == "published":
        # Frontend packs are published and available for review
        # This should allow progression to AWAITING_CUSTOMER_INPUT state
        # without blocking all_ready forever
        packs_count = _coerce_int(
            frontend_stage.get("packs_count") if isinstance(frontend_stage, Mapping) else None
        )
        if packs_count is not None and packs_count > 0:
            review_ready = True
            logger.debug(
                "REVIEW_READY_FROM_PUBLISHED packs_count=%d answers_received=%d answers_required=%d",
                packs_count,
                review_received,
                review_required,
            )
    
    # Fallback: check if stage is marked empty_ok or has matching answers
    if not review_ready and _stage_status_success(frontend_stage):
        metrics_payload = frontend_stage.get("metrics")
        if isinstance(metrics_payload, Mapping):
            required = _coerce_int(metrics_payload.get("answers_required"))
            received = _coerce_int(metrics_payload.get("answers_received"))
            if required is not None and received is not None and received == required:
                review_ready = True
        if not review_ready and _stage_empty_ok(frontend_stage):
            review_ready = True
```

**Rationale:**
- If frontend has `status=published` and `packs_count > 0`, the review materials exist
- The system should transition to a state where customer can provide input
- Blocking `all_ready` indefinitely is not helpful - the state should be "ready for customer review"
- Once responses are received, that can trigger a separate flow

**Alternative Approach (Less Intrusive):**

If you don't want to change the `all_ready` definition, consider:

1. Treat `review_ready=false` as expected state when waiting for customer input
2. Add a new barrier like `review_available` that becomes true when packs are published
3. Update `run_state` resolution logic to recognize this as a valid terminal state

---

### Fix #3 (Optional): Clarify Idle Reconciliation Logging

The repeated `UMBRELLA_STATE_AFTER_RECONCILE` logs can be reduced without losing visibility:

**File:** `backend/runflow/decider.py`  
**Function:** `reconcile_umbrella_barriers()`  
**Lines:** 5870-5880

**CURRENT:**
```python
log.info(
    "UMBRELLA_STATE_AFTER_RECONCILE sid=%s run_state=%s",
    sid,
    data.get("run_state"),
)
```

**CHANGE TO:**
```python
# Only log at info level when state changes or when not idle
if not previous_idle_state or not reconciliation_idle:
    log.info(
        "UMBRELLA_STATE_AFTER_RECONCILE sid=%s run_state=%s idle=%s",
        sid,
        data.get("run_state"),
        reconciliation_idle,
    )
else:
    log.debug(
        "UMBRELLA_STATE_AFTER_RECONCILE sid=%s run_state=%s idle=%s (stable)",
        sid,
        data.get("run_state"),
        reconciliation_idle,
    )
```

This reduces noise in logs while preserving full debug-level visibility.

---

## Summary of Findings

| Issue | Severity | Impact | Status |
|-------|----------|--------|--------|
| Validation paths missing from manifest | **MEDIUM** | Confusing state, potential downstream issues | Bug confirmed, fix provided |
| review_ready barrier stuck false | **HIGH** | Blocks `all_ready`, but doesn't cause loop | Bug confirmed, fix provided |
| Infinite loop | **FALSE ALARM** | No actual loop - idle detection working correctly | No action needed |
| Task racing on manifest | **LOW** | Possible but mitigated by short-circuits and atomic writes | Monitor, no immediate fix needed |
| ai vs artifacts.ai.packs inconsistency | **LOW** | Both sections exist, duplication is intentional (one for merge, one for validation) | No bug |

---

## Testing Plan

After applying fixes:

1. **Test validation manifest update:**
   ```bash
   # Run pipeline for new SID
   # Check that ai.validation and ai.packs.validation have non-null paths
   # Verify ai.status.validation.built = true
   ```

2. **Test review_ready barrier:**
   ```bash
   # Run pipeline to frontend stage
   # Verify review_ready becomes true once frontend is published
   # Verify all_ready becomes true (assuming other barriers are ready)
   # Verify run_state transitions appropriately
   ```

3. **Verify no regression in idle detection:**
   ```bash
   # Monitor logs for UMBRELLA_IDLE_STATE_REACHED
   # Confirm reconciliation_idle=true in runflow.json
   # Verify no tasks are re-enqueued after reaching idle state
   ```

4. **Load test for race conditions:**
   ```bash
   # Run multiple pipelines concurrently
   # Check manifest.json for consistency after completion
   # Verify no stage status is lost or overwritten
   ```

---

## Appendices

### A. Key Files and Functions

| File | Function | Purpose |
|------|----------|---------|
| `backend/ai/validation_builder.py` | `build_validation_packs_inline()` | Builds validation packs (inline path) |
| `backend/ai/validation_builder.py` | `_update_manifest_for_run()` | Updates manifest with validation paths |
| `backend/pipeline/runs.py` | `upsert_validation_packs_dir()` | Persists validation paths to manifest |
| `backend/pipeline/runs.py` | `mark_validation_merge_applied()` | Marks validation AI results as applied |
| `backend/pipeline/runs.py` | `ensure_ai_stage_status()` | Ensures stage status dict exists |
| `backend/pipeline/auto_ai_tasks.py` | `strategy_planner_step()` | Celery task for strategy execution |
| `backend/runflow/decider.py` | `reconcile_umbrella_barriers()` | Recomputes readiness barriers |
| `backend/runflow/decider.py` | `_compute_umbrella_barriers()` | Calculates barrier states |
| `backend/runflow/decider.py` | `decide_next()` | Determines next pipeline action |
| `backend/runflow/decider.py` | `_merge_runflow_snapshots()` | Merges runflow state updates |
| `backend/runflow/decider.py` | `_merge_stage_snapshot()` | Merges individual stage updates |

### B. State Machines

**Umbrella Barriers State:**
```
validation_ready && merge_ready && strategy_ready
  └─> If frontend published: review_ready = true (with fix)
      └─> If note_style not required OR completed: style_ready = true
          └─> all_ready = true
              └─> run_state = COMPLETED (if review also complete)
```

**Current Behavior (Broken):**
```
validation_ready && merge_ready && strategy_ready
  └─> review_ready = false (stuck because answers_required=3, answers_received=0)
      └─> all_ready = false (blocked by review_ready=false)
          └─> run_state = AWAITING_CUSTOMER_INPUT (correct!)
              └─> reconciliation_idle = true (correct!)
                  └─> No tasks re-enqueued (correct!)
```

### C. Manifest Structure

**Correct validation manifest should look like:**
```json
{
  "ai": {
    "validation": {
      "base": "C:\\...\\ai_packs\\validation",
      "dir": "C:\\...\\ai_packs\\validation",
      "accounts": "C:\\...\\ai_packs\\validation",
      "accounts_dir": "C:\\...\\ai_packs\\validation",
      "last_prepared_at": "2025-11-16T17:33:16Z"
    },
    "packs": {
      "validation": {
        "base": "C:\\...\\ai_packs\\validation",
        "dir": "C:\\...\\ai_packs\\validation",
        "packs": "C:\\...\\ai_packs\\validation\\packs",
        "packs_dir": "C:\\...\\ai_packs\\validation\\packs",
        "results": "C:\\...\\ai_packs\\validation\\results",
        "results_dir": "C:\\...\\ai_packs\\validation\\results",
        "index": "C:\\...\\ai_packs\\validation\\index.json",
        "last_built_at": "2025-11-16T17:33:16Z",
        "logs": "C:\\...\\ai_packs\\validation\\logs.txt"
      }
    },
    "status": {
      "validation": {
        "built": true,  ← Should be true!
        "sent": true,
        "completed_at": "2025-11-16T17:33:17Z",
        "merge_results_applied": true
      }
    }
  }
}
```

---

**End of Analysis**
