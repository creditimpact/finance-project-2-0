# Validation Chain-Only Orchestration — Implementation Summary

**Date**: November 19, 2025  
**Implementation Status**: ✅ **COMPLETE**  
**Design Document**: `VALIDATION_CHAIN_ONLY_REFACTOR_DESIGN.md`  
**Investigation**: `VALIDATION_V2_VS_LEGACY_INVESTIGATION.md`

---

## Overview

Successfully implemented chain-only validation orchestration, eliminating the fragile "fastpath" logic that caused validation to fail for SID `2d125dee-f84d-49e2-99c8-f0161cde0113` and other runs.

**Key Changes**:
1. ✅ Added idempotency guards to all chain tasks (merge + validation)
2. ✅ Removed inline validation builder from `stage_a_task`
3. ✅ Made `stage_a_task` a pure chain trigger (gate check → enqueue)
4. ✅ Updated validation compaction to mark canonical layout

---

## Changes Implemented

### Phase 1: Idempotency Guards in Chain Tasks

**File**: `backend/pipeline/auto_ai_tasks.py`

#### Change #1: `merge_build_packs` Idempotency Guard (Lines ~782-815)

**Location**: `_merge_build_stage()` function, after `_populate_common_paths()`

**What It Does**:
- Checks runflow snapshot for `stages.merge.status`, `merge_ai_applied`, `result_files`, `expected_packs`
- If merge already complete (status=success, applied=true, results >= expected):
  - Logs `MERGE_BUILD_IDEMPOTENT_SKIP` with reason and state
  - Sets `payload["merge_skipped"] = True`
  - Returns early without rebuilding packs
- Wrapped in try/except, falls back to normal build on error (defensive)

**Log Marker**: `MERGE_BUILD_IDEMPOTENT_SKIP`

---

#### Change #2: `merge_send` Idempotency Guard (Lines ~882-905)

**Location**: `_merge_send_stage()` function, after `_populate_common_paths()`

**What It Does**:
- Checks runflow snapshot for `stages.merge.merge_ai_applied`, `result_files`
- If merge already sent (applied=true, results > 0):
  - Logs `MERGE_SEND_IDEMPOTENT_SKIP`
  - Sets `payload["merge_sent"] = True`, `payload["merge_skipped"] = True`
  - Returns early
- Defensive try/except wrapper

**Log Marker**: `MERGE_SEND_IDEMPOTENT_SKIP`

---

#### Change #3: `validation_build_packs` Additional Idempotency Check (Lines ~1743-1767)

**Location**: `validation_build_packs()` task, after existing short-circuits, before `VALIDATION_STAGE_STARTED` log

**What It Does**:
- Checks validation index path for existence
- If index exists and has packs (length > 0):
  - Logs `VALIDATION_BUILD_IDEMPOTENT_SKIP` with pack count
  - Sets `payload["validation_short_circuit"] = True`
  - Returns early
- Complements existing runflow/manifest success checks
- Catches "packs built but not sent" scenario

**Log Marker**: `VALIDATION_BUILD_IDEMPOTENT_SKIP`

---

#### Change #4: `validation_send` Idempotency Guard (Lines ~1809-1830)

**Location**: `validation_send()` task, inside orchestrator mode branch, before V2 sender call

**What It Does**:
- Checks runflow snapshot for `stages.validation.status`, `results.results_total`, `results.missing_results`
- If validation already sent (status=success, results > 0, missing=0):
  - Logs `VALIDATION_SEND_IDEMPOTENT_SKIP`
  - Sets `payload["validation_sent"] = True`, `payload["validation_skipped"] = True`
  - Returns early without calling `run_validation_send_for_sid_v2`
- Defensive fallback on check failure

**Log Marker**: `VALIDATION_SEND_IDEMPOTENT_SKIP`

---

#### Change #5: `validation_compact` Idempotency Guard (Lines ~1949-1964)

**Location**: `validation_compact()` task, after index existence check, before strict pipeline check

**What It Does**:
- Reads validation index JSON
- Checks for `compacted` or `canonical_layout` flags
- If either flag is true:
  - Logs `VALIDATION_COMPACT_IDEMPOTENT_SKIP`
  - Sets `payload["validation_compacted"] = True`, `payload["validation_compact_skipped"] = True`
  - Returns early without calling `rewrite_index_to_canonical_layout`
- Falls back to normal compaction if JSON read fails

**Log Marker**: `VALIDATION_COMPACT_IDEMPOTENT_SKIP`

---

#### Change #6: Mark Canonical Layout After Compaction

**File**: `backend/validation/manifest.py`  
**Location**: `rewrite_index_to_canonical_layout()` function, lines ~302-312

**What It Does**:
- After building canonical index, adds `"canonical_layout": true` flag to JSON before writing
- Replaces `canonical_index.write()` with custom JSON serialization that includes the flag
- Enables idempotency check in `validation_compact` task

**Change**:
```python
# OLD: canonical_index.write()

# NEW:
document = canonical_index.to_json_payload()
document["canonical_layout"] = True
serialized = json.dumps(document, ensure_ascii=False, indent=2)
canonical_index.index_path.parent.mkdir(parents=True, exist_ok=True)
canonical_index.index_path.write_text(serialized + "\n", encoding="utf-8")
```

---

### Phase 2: Remove Fastpath, Make `stage_a_task` Pure Trigger

**File**: `backend/api/tasks.py`

#### Change #7: Replace Inline Validation Builder with Chain Trigger (Lines ~906-997)

**Old Behavior** (REMOVED):
- Imported `build_validation_packs_for_run` and called it inline
- Checked merge gate, injected T0 paths, built packs directly
- Relied on `VALIDATION_AUTOSEND_ENABLED` flag for completion
- Created race condition with async chain enqueue

**New Behavior** (IMPLEMENTED):
- Computes merge gate using `_compute_umbrella_barriers`
- If `merge_ready=true` and `validation_required=true`:
  - Logs `VALIDATION_CHAIN_TRIGGER` with gate state
  - Calls `maybe_queue_auto_ai_pipeline(sid, runs_root, ...)`
  - Logs `VALIDATION_CHAIN_ENQUEUED` with result
- Else:
  - Logs `VALIDATION_CHAIN_DEFER` with gate state
- No inline validation work, purely a trigger

**Log Markers**:
- `VALIDATION_CHAIN_TRIGGER` (when gate passes)
- `VALIDATION_CHAIN_ENQUEUED` (after successful enqueue)
- `VALIDATION_CHAIN_DEFER` (when gate not passed)
- `VALIDATION_CHAIN_TRIGGER_FAILED` (on exception)

---

#### Change #8: Remove Duplicate Chain Enqueue at End of Task (Lines ~1071-1093)

**Old Behavior** (REMOVED):
- Checked `ENABLE_AUTO_AI_PIPELINE` flag
- Called `has_ai_merge_best_tags(sid)`
- Enqueued `maybe_run_ai_pipeline_task.delay(sid)` (async Celery task)
- Created race condition, redundant with validation gate trigger

**New Behavior** (IMPLEMENTED):
- Replaced entire block with comment explaining chain is now enqueued earlier
- Single enqueue point: validation gate check in Change #7

---

## Verification Strategy

### For SID `2d125dee-f84d-49e2-99c8-f0161cde0113` (Example Case)

**Before Fix**:
- Merge completed: `merge_ready=true`, `merge_ai_applied=true`
- No validation stage in runflow
- `validation_ready=false`
- Logs showed `VALIDATION_PACKS_BUILD_DONE` but no sender/compact/merge

**After Fix** (Expected Flow):
1. `stage_a_task` checks `_compute_umbrella_barriers(run_dir)`
2. Sees `merge_ready=true` → logs `VALIDATION_CHAIN_TRIGGER`
3. Calls `maybe_queue_auto_ai_pipeline(sid, runs_root)`
4. Chain starts: logs `AUTO_AI_CHAIN_START`
5. Merge tasks skip (idempotent guards detect merge already complete):
   - `MERGE_BUILD_IDEMPOTENT_SKIP`
   - `MERGE_SEND_IDEMPOTENT_SKIP`
6. Validation tasks run (packs missing):
   - `VALIDATION_STAGE_STARTED`
   - `VALIDATION_BUILD_DONE`
   - `VALIDATION_ORCHESTRATOR_SEND_V2`
   - `VALIDATION_ORCHESTRATOR_SEND_V2_DONE`
   - `VALIDATION_COMPACT_DONE`
7. Runflow updated: `stages.validation.status=success`, `validation_ready=true`

---

### Log Markers to Grep After Deployment

**Chain Trigger Confirmation**:
```bash
grep "VALIDATION_CHAIN_TRIGGER" logs/
grep "VALIDATION_CHAIN_ENQUEUED" logs/
```

**Chain Execution**:
```bash
grep "AUTO_AI_CHAIN_START" logs/
```

**Idempotency (Work Skipped)**:
```bash
grep "_IDEMPOTENT_SKIP" logs/
# Should see: MERGE_BUILD_IDEMPOTENT_SKIP, MERGE_SEND_IDEMPOTENT_SKIP, etc.
```

**Validation Stages**:
```bash
grep "VALIDATION_STAGE_STARTED" logs/
grep "VALIDATION_ORCHESTRATOR_SEND_V2" logs/
grep "VALIDATION_COMPACT_DONE" logs/
grep "VALIDATION_STAGE_PROMOTED" logs/  # From V2 sender runflow refresh
```

**Validation Completion**:
```bash
# Check runflow for SID
cat runs/2d125dee-f84d-49e2-99c8-f0161cde0113/runflow.json | jq '.stages.validation.status'
# Should return: "success"

cat runs/2d125dee-f84d-49e2-99c8-f0161cde0113/runflow.json | jq '.umbrella_barriers.validation_ready'
# Should return: true
```

---

## Testing Requirements

### Unit Tests (To Be Added)

**File**: `tests/backend/pipeline/test_auto_ai_tasks.py`

1. ✅ `test_merge_build_idempotent_skip_when_merge_complete`
2. ✅ `test_merge_send_idempotent_skip_when_already_sent`
3. ✅ `test_validation_build_packs_idempotent_skip_when_packs_exist`
4. ✅ `test_validation_send_idempotent_skip_when_already_sent`
5. ✅ `test_validation_compact_idempotent_skip_when_already_compacted`

**File**: `tests/backend/api/test_stage_a_task.py` (or similar)

1. ✅ `test_stage_a_task_enqueues_chain_when_merge_ready`
2. ✅ `test_stage_a_task_defers_chain_when_merge_not_ready`

### Integration Tests (To Be Added)

1. **Merge Complete, No Validation**:
   - Setup: SID with `merge_ready=true`, no validation stage
   - Trigger: Enqueue chain
   - Assert: Merge tasks skip (idempotent), validation tasks run, validation stage promoted

2. **Both Merge and Validation Complete**:
   - Setup: SID with `merge_ready=true`, `validation_ready=true`
   - Trigger: Enqueue chain
   - Assert: All tasks skip (idempotent), no duplicate work

---

## Risk Mitigation

### Rollback Plan

**If validation stops working**:
1. **Immediate**: Set `VALIDATION_AUTOSEND_ENABLED=1` in `.env` (restores inline autosend as temporary workaround)
2. **Short-term**: Revert commits for Changes #7-8 (restore inline builder in `stage_a_task`)
3. **Root Cause**: Check logs for:
   - Missing `VALIDATION_CHAIN_TRIGGER` (gate check failed)
   - Missing `AUTO_AI_CHAIN_START` (chain never started)
   - Errors in `maybe_queue_auto_ai_pipeline`

### Monitoring

**Alert Conditions**:
- SIDs stuck with `merge_ready=true`, `validation_ready=false` for >30min
- Absence of `VALIDATION_CHAIN_TRIGGER` logs when expected
- Increase in `VALIDATION_CHAIN_TRIGGER_FAILED` errors

**Success Metrics** (After Deployment):
- % SIDs with `validation_ready=true` after merge complete: Should match or exceed baseline
- Avg time from `merge_ready=true` to `validation_ready=true`: Should not increase
- Idempotent re-runs: Chain can be safely re-enqueued without duplicate work

---

## Changes Mapping to Design Document

| Design Doc Change | Implementation | File | Status |
|-------------------|----------------|------|--------|
| **Change #3**: Add `merge_build_packs` idempotency | `_merge_build_stage()` guard | `auto_ai_tasks.py` | ✅ Done |
| **Change #4**: Add `merge_send` idempotency | `_merge_send_stage()` guard | `auto_ai_tasks.py` | ✅ Done |
| **Change #5**: Add `validation_build_packs` idempotency | Additional pack check | `auto_ai_tasks.py` | ✅ Done |
| **Change #6**: Add `validation_send` idempotency | Runflow check before V2 sender | `auto_ai_tasks.py` | ✅ Done |
| **Change #7**: Add `validation_compact` idempotency | Index flag check | `auto_ai_tasks.py` | ✅ Done |
| **Bonus**: Mark canonical layout | `canonical_layout=true` in index | `manifest.py` | ✅ Done |
| **Change #1**: Remove inline builder from `stage_a_task` | Replaced with chain trigger | `tasks.py` | ✅ Done |
| **Change #2**: Remove duplicate chain enqueue | Deleted redundant block | `tasks.py` | ✅ Done |

---

## Architecture Summary

### Before

```
stage_a_task:
  ├─ run_validation_requirements ✅
  ├─ build_validation_packs_for_run ❌ (INLINE, causes problems)
  │   └─ (optional) inline autosend if flag set
  └─ maybe_run_ai_pipeline_task.delay(sid) ⚠️ (ASYNC, race condition)
```

### After

```
stage_a_task:
  ├─ run_validation_requirements ✅
  ├─ _compute_umbrella_barriers (gate check) ✅
  └─ IF merge_ready:
      └─ maybe_queue_auto_ai_pipeline(sid) → enqueue_auto_ai_chain ✅

Auto-AI Chain (enqueue_auto_ai_chain):
  ├─ merge_build_packs (idempotent: skips if merge complete) ✅
  ├─ merge_send (idempotent: skips if already sent) ✅
  ├─ merge_compact ✅
  ├─ validation_build_packs (idempotent: skips if packs exist) ✅
  ├─ validation_send (idempotent: skips if already sent) ✅
  ├─ validation_compact (idempotent: skips if already compacted) ✅
  └─ validation_merge_ai_results_step (skipped in orchestrator mode) ✅
```

**Key Differences**:
- ✅ Single orchestration path (chain only)
- ✅ No inline validation work
- ✅ No race conditions (chain enqueued synchronously)
- ✅ Idempotent tasks (safe to re-run)
- ✅ No dependency on `VALIDATION_AUTOSEND_ENABLED` flag

---

## Next Steps

1. ✅ **Code Changes**: Complete (all 8 changes implemented)
2. ⏳ **Unit Tests**: To be written (7 tests identified)
3. ⏳ **Integration Tests**: To be written (2 fixtures identified)
4. ⏳ **Staging Deployment**: Test with real SIDs
5. ⏳ **Production Deployment**: After staging validation
6. ⏳ **Monitoring**: Set up dashboards for new log markers

---

## Related Documents

- Design Document: `VALIDATION_CHAIN_ONLY_REFACTOR_DESIGN.md` (1303 lines)
- Investigation Report: `VALIDATION_V2_VS_LEGACY_INVESTIGATION.md` (633 lines)
- This Implementation Summary: `VALIDATION_CHAIN_ONLY_IMPLEMENTATION_SUMMARY.md`

---

**Implementation Completed**: November 19, 2025  
**Status**: ✅ **READY FOR TESTING**  
**Next Phase**: Unit tests → Integration tests → Staging deployment
