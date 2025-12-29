# Validation Start Gate Bug - Root Cause Analysis

## Executive Summary

**BUG**: Validation pipeline starts (builds packs, sends to AI) even when `merge_ready=false`, violating the merge barrier invariant.

**ROOT CAUSE**: `backend/api/tasks.py::build_problem_cases_task()` calls `build_validation_packs_for_run()` directly at line 965 without checking `merge_ready` barrier.

**IMPACT**: For non-zero-packs cases (merge AI expected), validation can complete before merge AI results are applied, creating timing bug where `validation_ready=true` while `merge_ready=false`.

**FIX**: Add merge_ready barrier check in `build_problem_cases_task()` before calling `build_validation_packs_for_run()`.

---

## Bug Evidence - SID 5e51359f-7ba8-4be3-8199-44d34c55a4ed

### Timeline
```
21:04:58.245434Z - Merge stage completes: packs_built=1, expected_packs=1, merge_zero_packs=false
21:04:58.503721Z - Validation stage ends: status=success, ai_packs_built=0, empty_ok=false
21:04:58.682616Z - Validation build_packs executes: eligible_accounts=3, packs_built=1, packs_skipped=2
21:05:08.469961Z - Validation promoted: status=success, validation_ready=true
                  BARRIERS: merge_ready=false, validation_ready=true ❌ VIOLATION
```

### State Analysis

**runflow.json (Final State)**:
```json
{
  "stages": {
    "validation": {
      "status": "success",
      "ready_latched": true,
      "ready_latched_at": "2025-11-18T21:05:08Z",
      "sent": true,  // ← runflow thinks validation was sent
      "empty_ok": false
    },
    "merge": {
      "status": "success",
      "empty_ok": false,
      "expected_packs": 1
    }
  },
  "umbrella_barriers": {
    "merge_ready": false,  // ❌ Merge barrier CLOSED
    "validation_ready": true,  // ❌ Validation COMPLETED
    "merge_zero_packs": false
  }
}
```

**manifest.json (AI Pipeline State)**:
```json
{
  "ai": {
    "status": {
      "validation": {
        "built": true,
        "sent": false,  // ← manifest shows validation NOT sent via auto_ai
        "completed_at": null,
        "failed": false
      }
    }
  }
}
```

**DISCREPANCY**: `runflow.sent=true` but `manifest.sent=false` indicates validation bypassed the normal auto_ai pipeline.

### Missing Logs

**NOT FOUND**:
- `VALIDATION_ORCHESTRATOR_START` - orchestrator never ran
- `VALIDATION_V2_BUILDER_ENTRY` - builder diagnostics not logged
- `VALIDATION_FASTPATH_*` - fastpath not triggered (non-zero-packs case)

**CONCLUSION**: Validation triggered via direct call to `build_validation_packs_for_run()`, bypassing all normal entry points.

---

## Root Cause - Direct Builder Invocation

### Culprit Code

**File**: `backend/api/tasks.py`  
**Function**: `build_problem_cases_task()` (line 820)  
**Call Site**: Line 965

```python
@shared_task(bind=True, autoretry_for=(), retry_backoff=False)
def build_problem_cases_task(self, prev: dict | None = None, sid: str | None = None) -> dict:
    # ... problem case building logic ...
    
    if ENABLE_VALIDATION_REQUIREMENTS:
        try:
            stats = run_validation_requirements_for_all_accounts(sid)
        except Exception:
            log.error("VALIDATION_REQUIREMENTS_PIPELINE_FAILED sid=%s", sid, exc_info=True)
        else:
            # V2: Build validation packs after requirements (triggers autosend in orchestrator mode)
            try:
                from backend.ai.validation_builder import build_validation_packs_for_run
                runs_root = _ensure_manifest_root()
                
                # ❌ NO MERGE_READY CHECK HERE!
                pack_results = build_validation_packs_for_run(sid, runs_root=runs_root)
                packs_written = sum(len(entries or []) for entries in pack_results.values())
                log.info("VALIDATION_PACKS_BUILD_DONE sid=%s packs=%d", sid, packs_written)
            except Exception:
                log.error("VALIDATION_PACKS_BUILD_FAILED sid=%s", sid, exc_info=True)
```

### Why This Is Wrong

1. **Bypasses Orchestrator**: Calls `build_validation_packs_for_run()` directly instead of using `ValidationOrchestrator.run_for_sid()`
2. **No Merge Barrier**: Does not check `merge_ready` before starting validation
3. **Non-Zero-Packs Cases**: Affects cases where merge AI results are expected (not zero-packs fast path)
4. **Timing Bug**: Validation can complete before merge AI results applied → `validation_ready=true` while `merge_ready=false`

---

## Validation Entry Points Analysis

### Complete Entry Point Map

| Entry Point | File:Line | merge_ready Check? | Used for SID 5e51359f? |
|-------------|-----------|-------------------|------------------------|
| **ValidationOrchestrator.run_for_sid()** | validation_orchestrator.py:75 | ✅ **YES** (lines 89-96) | ❌ No (no orchestrator logs) |
| **auto_ai_tasks.validation_build_packs** | auto_ai_tasks.py:1698 | ⚠️ Checks runflow success, not merge_ready | ❌ No (manifest.sent=false) |
| **auto_ai_tasks.validation_send** | auto_ai_tasks.py:1788 | ⚠️ Assumes packs already built | ❌ No (manifest.sent=false) |
| **api/tasks.py build_problem_cases_task** | api/tasks.py:965 | ❌ **NO CHECK** | ✅ **YES** (event at 21:04:58.682616Z) |
| **decider._maybe_enqueue_validation_fastpath** | decider.py:717 | ⚠️ Partial (checks merge_ai_applied) | ❌ No (non-zero-packs case) |
| **decider._watchdog_trigger_validation_fastpath** | decider.py:644 | ❌ NO (emergency requeue) | ❌ No (no watchdog logs) |

### Orchestrator Gate Implementation (CORRECT)

**File**: `backend/pipeline/validation_orchestrator.py` (lines 89-96)

```python
def run_for_sid(self, sid: str) -> dict[str, Any]:
    # ... init logic ...
    
    # ── MERGE_READY BARRIER CHECK ──────────────────────────────────────
    # Validation cannot start until merge is ready (AI results applied).
    # This prevents the timing bug where validation completes before merge.
    barriers = _compute_umbrella_barriers(run_dir)
    merge_ready = barriers.get("merge_ready", False)
    if not merge_ready:
        logger.info(
            "VALIDATION_ORCHESTRATOR_DEFERRED sid=%s reason=merge_not_ready barriers=%s",
            sid,
            barriers,
        )
        return {"sid": sid, "deferred": True, "reason": "merge_not_ready"}
    # ───────────────────────────────────────────────────────────────────
    
    # ... proceed with pack building ...
```

**This gate is CORRECT but BYPASSED** when `build_problem_cases_task()` calls `build_validation_packs_for_run()` directly.

---

## Impact Analysis

### Affected Cases

**Scope**: Non-zero-packs validation cases where:
- `merge.empty_ok = false`
- `merge.expected_packs > 0`
- `build_problem_cases_task()` executes before merge AI results arrive

**Zero-Packs Cases**: UNAFFECTED (fast path correctly uses `_maybe_enqueue_validation_fastpath` which checks `merge_ai_applied`)

### Symptom Pattern

```
merge.status = "success"
merge.empty_ok = false
merge.expected_packs = 1
merge.merge_ai_applied = false  // ❌ Results not yet applied

validation.status = "success"  // ❌ Validation completed
validation.ready_latched = true  // ❌ Validation ready
validation.empty_ok = false

umbrella_barriers.merge_ready = false  // ❌ Barrier still closed
umbrella_barriers.validation_ready = true  // ❌ But validation passed!
```

### Downstream Effects

1. **Orchestrator Deferral Broken**: Orchestrator cannot enforce merge barrier if validation already started via api/tasks
2. **Race Condition**: Validation results arrive before merge results → validation_ai_applied completes first
3. **Barrier Inversion**: System can reach state where `validation_ready=true` while `merge_ready=false`
4. **Strategy Blocking**: Strategy stage waits for merge_ready, but validation already advanced

---

## Fix Strategy

### Primary Fix: Add Merge Barrier to api/tasks.py

**Location**: `backend/api/tasks.py::build_problem_cases_task()` before line 965

**Implementation**:
```python
# V2: Build validation packs after requirements (triggers autosend in orchestrator mode)
try:
    from backend.ai.validation_builder import build_validation_packs_for_run
    from backend.runflow.decider import _compute_umbrella_barriers
    
    runs_root = _ensure_manifest_root()
    if runs_root is None:
        log.warning("VALIDATION_PACKS_SKIP sid=%s reason=no_runs_root", sid)
        return summary
    
    # ── MERGE_READY BARRIER CHECK ──────────────────────────────────────
    # Validation cannot start until merge is ready (AI results applied).
    # This prevents the timing bug where validation completes before merge.
    # Same gate used in ValidationOrchestrator.run_for_sid (validation_orchestrator.py:89-96)
    run_dir = Path(runs_root) / sid
    barriers = _compute_umbrella_barriers(run_dir)
    merge_ready = barriers.get("merge_ready", False)
    if not merge_ready:
        log.info(
            "VALIDATION_PACKS_DEFERRED sid=%s reason=merge_not_ready barriers=%s",
            sid,
            barriers,
        )
        # Return success but skip validation pack building
        if isinstance(summary, dict):
            summary["validation_packs"] = {"deferred": True, "reason": "merge_not_ready"}
        return summary
    # ───────────────────────────────────────────────────────────────────
    
    log.info("VALIDATION_V2_PIPELINE_ENTRY sid=%s runs_root=%s", sid, runs_root)
    
    # ... existing T0 injection and build logic ...
    pack_results = build_validation_packs_for_run(sid, runs_root=runs_root)
    # ... rest of existing code ...
```

### Secondary Fix: Better Orchestrator Integration

**Option 1**: Replace direct `build_validation_packs_for_run()` call with orchestrator invocation:

```python
# Instead of:
pack_results = build_validation_packs_for_run(sid, runs_root=runs_root)

# Use:
from backend.pipeline.validation_orchestrator import ValidationOrchestrator
orchestrator = ValidationOrchestrator(runs_root=runs_root)
result = orchestrator.run_for_sid(sid)

if result.get("deferred"):
    log.info("VALIDATION_ORCHESTRATOR_DEFERRED sid=%s reason=%s", sid, result.get("reason"))
    if isinstance(summary, dict):
        summary["validation_orchestrator"] = result
    return summary
```

**Option 2**: Add explicit comment that orchestrator handles deferral:

```python
# Note: If orchestrator is enabled, it will handle merge_ready deferral.
# In legacy mode (orchestrator disabled), we check merge_ready here.
if not _orchestrator_mode_enabled():
    barriers = _compute_umbrella_barriers(run_dir)
    merge_ready = barriers.get("merge_ready", False)
    if not merge_ready:
        log.info("VALIDATION_PACKS_DEFERRED sid=%s reason=merge_not_ready", sid)
        return summary
```

### Verification

**Test Case**: Run pipeline with:
- Non-zero-packs merge case
- Slow merge AI response (>10s)
- Enable `ENABLE_VALIDATION_REQUIREMENTS=true`

**Expected Behavior**:
1. `build_problem_cases_task()` completes at ~T+5s
2. Validation deferred due to `merge_ready=false`
3. Merge AI results arrive at ~T+15s → `merge_ready=true`
4. Orchestrator or watchdog triggers validation at ~T+20s
5. Validation completes at ~T+25s → `validation_ready=true`
6. **NEVER**: `validation_ready=true` while `merge_ready=false`

**Logs to Confirm**:
```
T+5s: VALIDATION_PACKS_DEFERRED sid=... reason=merge_not_ready barriers={'merge_ready': False}
T+15s: (merge results applied)
T+20s: VALIDATION_ORCHESTRATOR_START sid=... (or VALIDATION_FASTPATH_WATCHDOG_REENQUEUE)
T+25s: VALIDATION_ORCHESTRATOR_DONE sid=... ready=true
```

---

## Critical Distinctions

### What This Bug IS
- ❌ Validation **STARTING** (pack building) without merge barrier check
- ❌ Direct call to `build_validation_packs_for_run()` in `api/tasks.py`
- ❌ Bypassing orchestrator's merge_ready gate

### What This Bug IS NOT
- ✅ Validation **PROMOTION** logic (setting `status=success`, `ready_latched=true`) - promotion is CORRECT
- ✅ `_apply_validation_stage_promotion()` function - does not need changes
- ✅ Zero-packs fast path - already has `merge_ai_applied` check (line 739-751 in decider.py)
- ✅ Orchestrator gate - already correct (lines 89-96 in validation_orchestrator.py)

---

## References

### Related Documents
- `MERGE_FINALIZATION_FIX_SUMMARY.md` - Previous merge_ai_applied flag implementation
- `MERGE_GATE_BYPASS_INVESTIGATION.md` - Initial investigation (focused on promotion, refocused to starting)
- `VALIDATION_V2_PRODUCTION_INTEGRATION.md` - V2 validation architecture

### Code Locations
1. **Bug Location**: `backend/api/tasks.py:965` - `build_problem_cases_task()`
2. **Correct Gate**: `backend/pipeline/validation_orchestrator.py:89-96` - `run_for_sid()`
3. **Fastpath Gate**: `backend/runflow/decider.py:739-751` - `_maybe_enqueue_validation_fastpath()`
4. **Promotion Logic**: `backend/runflow/decider.py:2770-2900` - `_apply_validation_stage_promotion()` (NOT the bug)

### Environment Flags
- `ENABLE_VALIDATION_REQUIREMENTS` - Enables validation requirements → triggers `build_problem_cases_task()` → exposes bug
- `VALIDATION_ORCHESTRATOR_MODE` - When enabled, orchestrator provides correct gate (but bypassed by api/tasks)

---

## Next Steps

1. ✅ Apply fix to `backend/api/tasks.py::build_problem_cases_task()`
2. ⏳ Test with non-zero-packs case and slow merge AI
3. ⏳ Verify logs show proper deferral
4. ⏳ Confirm barrier invariant: `validation_ready=true` only when `merge_ready=true` (for non-zero-packs)
5. ⏳ Consider refactoring to ALWAYS use orchestrator instead of direct builder calls

---

**Date**: 2025-01-21  
**Investigator**: GitHub Copilot  
**SID Analyzed**: 5e51359f-7ba8-4be3-8199-44d34c55a4ed
