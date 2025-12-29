# Runflow Validation Authority - Deep Analysis & Timeline

> Phase 3 Update (Final Design and Corrections)

- Corrected prior assumption: V2 did run for SID `14b140ab-5de5-436e-9477-811468b181cf`. The stale runflow was caused by multiple bugs now fixed: FrozenInstanceError during index update, missing `sent`→`completed` normalization, a frontend NameError during reconcile, and a late legacy `record_stage('built')` write.
- Final design (V2-authoritative): In orchestrator mode, only V2-aware paths can write `stages.validation.*`. Legacy writes are telemetry-only and cannot override promotion snapshots.
- Invariants: Once validation is promoted to `success`, it cannot be downgraded to `built`; counts and V2 flags derive from index + manifest via promotion, never hand-written.
- Backfill: A repair tool now runs V2 apply (idempotent), refreshes validation, and reconciles barriers with clear before/after logs.

See “Final Design (Phase 3)” at the end for details.

## Executive Summary

**Problem**: Validation stage in `runflow.json` shows `status: built`, `results_total: 0` despite manifest showing `applied: true`, `results_applied: 3` and result files existing on disk.

**Root Cause**: V2 validation sender never executed for SID `14b140ab-5de5-436e-9477-811468b181cf`. Legacy pipeline ran, creating packs and writing index with `status: sent`, but no AI results were generated and no V2 apply/refresh cycle occurred.

**Critical Discovery**: Index records have `status: "sent"` but promotion logic in `_validation_results_progress` only recognizes `status: "completed"` as ready for promotion.

---

## SID 14b140ab Timeline Reconstruction

### Source Files
- **Manifest**: `c:\dev\credit-analyzer\runs\14b140ab-5de5-436e-9477-811468b181cf\manifest.json`
- **Runflow**: `c:\dev\credit-analyzer\runs\14b140ab-5de5-436e-9477-811468b181cf\runflow.json`
- **Index**: `c:\dev\credit-analyzer\runs\14b140ab-5de5-436e-9477-811468b181cf\ai_packs\validation\index.json`
- **Events**: `c:\dev\credit-analyzer\runs\14b140ab-5de5-436e-9477-811468b181cf\runflow_events.jsonl`

### Timeline (All timestamps 2025-11-18)

#### T0: 17:56:12Z - Run Created
```json
{"sid": "14b140ab-5de5-436e-9477-811468b181cf", "created_at": "2025-11-18T17:56:12Z"}
```

#### T1: 17:56:24.678583Z - Merge Stage Start
- Merge scoring started
- 3 accounts processed (9, 10, 11)
- All pairs skipped: `missing_original_creditor`
- **Result**: 0 merge packs created (expected for this case)

#### T2: 17:56:24.992840Z - Date Convention Detection
```json
{"stage": "validation", "step": "detect_date_convention", "status": "success", 
 "metrics": {"confidence": 1.0, "accounts_scanned": 3},
 "out": {"path": "traces/date_convention.json"}}
```

#### T3: 17:56:25.027791Z - Validation Stage Start
```json
{"stage": "validation", "event": "start"}
```

#### T4: 17:56:25.162652Z - Requirements Processing
```json
{"stage": "validation", "step": "requirements", "status": "success",
 "metrics": {"total_accounts": 3, "processed_accounts": 3, "findings": 32, "errors": 0}}
```
- 32 validation findings generated across 3 accounts
- Requirements phase completed successfully

#### T5: 17:56:25.165652Z - Validation Stage End (First)
```json
{"t": "2025-11-18T17:56:25.165652Z", "sid": "14b140ab-5de5-436e-9477-811468b181cf",
 "stage": "validation", "event": "end", "status": "success",
 "umbrella_barriers": {"validation_ready": false},
 "summary": {"findings_count": 32, "ai_packs_built": 0, "ai_results_received": 0}}
```
- **Validation stage completed** with status `success`
- **AI packs not yet built**: `ai_packs_built: 0`
- **Umbrella barriers**: `validation_ready: false`

#### T6: 17:56:25.502635Z - Validation Packs Built
```json
{"stage": "validation", "step": "build_packs", "status": "success",
 "metrics": {"eligible_accounts": 3, "packs_built": 3, "packs_skipped": 0},
 "out": {"summary": {"eligible_accounts": 3, "packs_built": 3, "packs_skipped": 0}}}
```
- **3 packs built** for accounts 9, 10, 11
- Packs written to `ai_packs/validation/packs/val_acc_*.jsonl`
- Index created with 3 records, all with `status: "sent"`

#### T7: 17:56:41.208597Z - Validation Stage End (Second)
```json
{"t": "2025-11-18T17:56:41.208597Z", "sid": "14b140ab-5de5-436e-9477-811468b181cf",
 "stage": "validation", "event": "end", "status": "built",
 "summary": {"findings_count": 3,
   "metrics": {"packs_total": 3, "validation_ai_required": true, 
               "validation_ai_completed": false}},
 "barriers": {"validation_ready": false, "strategy_ready": false}}
```
- **Status changed to `built`**
- **AI completion**: `validation_ai_completed: false`
- **Barriers**: `validation_ready: false`

#### T8: 17:56:41.290835Z - Frontend Review
- Frontend packs refreshed (no changes)

#### T9: MISSING - V2 Sender Execution
**EXPECTED LOGS NOT FOUND**:
- No `VALIDATION_V2_AUTOSEND_TRIGGER` log
- No `VALIDATION_V2_SEND_START` logs
- No `VALIDATION_V2_SEND_DONE` logs
- No `VALIDATION_V2_APPLY_START` log
- No `VALIDATION_V2_APPLY_DONE` log
- No `VALIDATION_V2_RUNFLOW_REFRESHED` log

**Conclusion**: V2 sender never executed for this SID.

---

## Current State Analysis

### Manifest State
**File**: `manifest.json` (line 65-78)
```json
"validation": {
  "built": true,
  "sent": true,
  "completed_at": "2025-11-18T17:56:41Z",
  "failed": false,
  "state": "completed",
  "results_total": 3,
  "results_applied": 3,
  "results_unmatched": 0,
  "results_apply_at": "2025-11-18T17:56:41Z",
  "results_apply_done": true,
  "validation_ai_applied": true,
  "results_apply_ok": true,
  "applied": true
}
```

**Analysis**: Manifest shows:
- ✅ `applied: true`
- ✅ `results_applied: 3`
- ✅ `validation_ai_applied: true`
- ✅ `completed_at: "2025-11-18T17:56:41Z"`

**Issue**: These fields were likely populated by legacy code, NOT by V2 apply logic.

### Index State
**File**: `ai_packs/validation/index.json`
```json
{
  "schema_version": 2,
  "packs": [
    {
      "account_id": 9,
      "pack": "packs/val_acc_009.jsonl",
      "status": "sent",
      "result_jsonl": "results/acc_009.result.jsonl",
      "built_at": "2025-11-18T17:56:25Z"
    },
    {
      "account_id": 10,
      "pack": "packs/val_acc_010.jsonl",
      "status": "sent",
      "result_jsonl": "results/acc_010.result.jsonl",
      "built_at": "2025-11-18T17:56:25Z"
    },
    {
      "account_id": 11,
      "pack": "packs/val_acc_011.jsonl",
      "status": "sent",
      "result_jsonl": "results/acc_011.result.jsonl",
      "built_at": "2025-11-18T17:56:25Z"
    }
  ]
}
```

**Critical Issue**: All records have `status: "sent"`, NOT `"completed"`.

**Promotion Logic Expectation** (from `backend/runflow/decider.py:_validation_results_progress` line ~3468):
```python
normalized_status = status.lower() if isinstance(status, str) else ""
ready = normalized_status == "completed" and result_jsonl and result_path.exists()
```

**Result**: Promotion logic sees `status: "sent"` → `ready: False` → does not count these records as completed → `results_total: 0`.

### Result Files on Disk
**Files exist**:
- ✅ `ai_packs/validation/results/acc_009.result.jsonl`
- ✅ `ai_packs/validation/results/acc_010.result.jsonl`
- ✅ `ai_packs/validation/results/acc_011.result.jsonl`

**Sample Result** (acc_009.result.jsonl):
```json
{
  "account_id": 9,
  "decision": "supportive_needs_companion",
  "completed_at": "2025-11-18T17:56:32Z",
  "checks": {"doc_requirements_met": true, "materiality": true, "supports_consumer": true}
}
```

**Analysis**: Results have proper structure with `completed_at` timestamp. These were likely created by **legacy AI sender**, not V2.

### Runflow State
**File**: `runflow.json`
```json
{
  "stages": {
    "validation": {
      "status": "built",
      "metrics": {
        "packs_total": 3,
        "validation_ai_required": true,
        "validation_ai_completed": false
      },
      "results": {
        "results_total": 0,
        "completed": 0,
        "failed": 0
      }
    }
  },
  "last_writer": "record_stage"
}
```

**Issue**: 
- ❌ `status: "built"` (should be `"success"`)
- ❌ `results_total: 0` (should be `3`)
- ❌ `validation_ai_completed: false` (should be `true`)

**Last Writer**: `record_stage` at 17:56:41Z (from T7 event)

---

## Root Cause Analysis

### Primary Issue: V2 Sender Never Executed

**Evidence**:
1. No `VALIDATION_V2_*` logs in any log files
2. Index records have `status: "sent"` (legacy format) not `"completed"` (V2 format)
3. Result files exist but were created by legacy AI sender (not V2)
4. Manifest fields populated by legacy code path

### Secondary Issue: Status Recognition Gap

**Problem**: Even if result files exist, promotion logic only recognizes `status: "completed"`.

**Code**: `backend/runflow/decider.py:_validation_results_progress` (line ~3478)
```python
for rec in index_obj.packs:
    status = getattr(rec, "status", None)
    normalized_status = status.lower() if isinstance(status, str) else ""
    ready = normalized_status == "completed" and result_jsonl and result_path.exists()
    if ready:
        completed_count += 1
```

**Impact**: Index records with `status: "sent"` are not counted as completed, even if result files exist.

### Tertiary Issue: No Post-Apply Refresh

**Expected Flow** (from validation_sender_v2.py line ~395):
```python
if apply_success:
    refresh_validation_stage_from_index(sid, runs_root)
    log.info("VALIDATION_V2_RUNFLOW_REFRESHED sid=%s", sid)
```

**Actual**: Since V2 sender never ran, no refresh occurred after results were generated.

---

## Investigation: Why V2 Sender Didn't Trigger

### V2 Autosend Logic EXISTS ✅
**File**: `backend/ai/validation_builder.py` (lines 2515-2571)
```python
# Phase 2 orchestrator autosend: optionally send packs immediately after build completes.
# Uses new clean validation_sender_v2 inspired by note_style pattern.
# Guarded by orchestrator mode + autosend env flags.

orchestrator_mode = _flag("VALIDATION_ORCHESTRATOR_MODE", True)
autosend_enabled = (
    _flag("VALIDATION_AUTOSEND_ENABLED", False) or
    _flag("VALIDATION_SEND_ON_BUILD", False) or
    _flag("VALIDATION_STAGE_AUTORUN", False)
)

if packs_built > 0 and orchestrator_mode and autosend_enabled:
    log.info("VALIDATION_V2_AUTOSEND_TRIGGER sid=%s packs=%d", sid, packs_built)
    from backend.ai.validation_sender_v2 import run_validation_send_for_sid_v2
    stats = run_validation_send_for_sid_v2(sid, runs_root_path)
    log.info("VALIDATION_V2_AUTOSEND_DONE ...")
```

### Environment Flags ARE SET ✅
**File**: `.env`
```bash
VALIDATION_ORCHESTRATOR_MODE=1      # ✅ Set
VALIDATION_AUTOSEND_ENABLED=1       # ✅ Set
VALIDATION_STAGE_AUTORUN=1          # ✅ Set
VALIDATION_SEND_ON_BUILD=1          # ✅ Set
```

### Mystery: Why No Autosend Despite Flags?

**Observed**:
- ✅ Environment flags are set
- ✅ V2 autosend logic exists in validation_builder.py
- ✅ 3 packs were built (confirmed in runflow_events.jsonl)
- ❌ NO `VALIDATION_V2_AUTOSEND_TRIGGER` log
- ❌ NO `VALIDATION_V2_AUTOSEND_SKIP` log

**Possible Explanations**:
1. **Exception in Flag Check**: The flag check guard (`try/except`) caught an exception and logged `VALIDATION_V2_AUTOSEND_GUARD_FAILED` (not visible in SID logs)
2. **Module Import Error**: `validation_builder.py` code path not executed; legacy builder ran instead
3. **Runtime Environment Mismatch**: Environment loaded at process start doesn't match `.env` file (if app was running before flags were added)
4. **Log Destination**: V2 logs written elsewhere (not to SID-specific logs.txt)
5. **Execution Path Bypass**: `build_validation_packs_for_run` returned early or raised exception before reaching autosend block

**Critical Observation**: SID 14b140ab logs show only pack build logs, NOT V2 sender logs. This suggests:
- The `validation_builder.py` module WAS imported (pack build logs present)
- The autosend block either didn't execute OR failed silently
- No `VALIDATION_V2_AUTOSEND_SKIP` log means check didn't reach that branch

### Most Likely Cause

**Hypothesis**: The SID was created on **2025-11-18** (November 18, 2025 - likely a typo, should be 2024). If this was a test run from BEFORE environment flags were added to `.env`, the autosend wouldn't trigger.

**Alternative**: Result files show `completed_at: 2025-11-18T17:56:32Z`, suggesting legacy AI sender ran separately (possibly via Celery task queue) AFTER pack build, bypassing V2 orchestrator flow entirely.

---

## Blocking Bugs Discovered

### Bug 1: FrozenInstanceError in apply_results_v2.py
**Location**: `backend/validation/apply_results_v2.py` (during index update)
**Issue**: Attempting to mutate frozen dataclass `ValidationPackRecord`
```python
# Fails:
record.status = "completed"

# Should use:
updated_record = dataclasses.replace(record, status="completed")
```

### Bug 2: Index Status Ambiguity (sent vs completed)
**Issue**: 
- Legacy code writes `status: "sent"` when packs are sent to AI
- V2 code expects `status: "completed"` after results are applied
- Promotion logic only recognizes `"completed"` status

**Impact**: Records with status `"sent"` are never promoted, even if results exist.

### Bug 3: NameError - summary_payload in _frontend_responses_progress
**Location**: `backend/runflow/decider.py:_frontend_responses_progress` (line ~3753)
**Issue**: Variable `summary_payload` referenced before assignment in some code paths
**Impact**: Crashes when calling `reconcile_umbrella_barriers` in tests

### Bug 4: NameError - RunManifest in _ensure_manifest_root
**Issue**: Free variable reference to `RunManifest` not properly imported or shadowed by local variable
**Impact**: Crashes in manifest loading code paths

### Bug 5: RUNFLOW_UNKNOWN_STATUS for "sent" Status
**Location**: `backend/runflow/decider.py:_merge_stage_snapshot` (status priority logic)
**Issue**: Status `"sent"` not in recognized status set, triggers unknown status warning
**Impact**: Merge logic doesn't know how to prioritize `"sent"` status

---

## Comparison: Expected vs Actual Flow

### Expected V2 Flow
```
1. Requirements complete
2. Build validation packs → index with status="built"
3. VALIDATION_V2_AUTOSEND_TRIGGER
4. V2 sender: Send packs to AI → receive results
5. V2 sender: Write result files → update index status="sent"
6. V2 apply: Enrich summaries → update index status="completed"
7. V2 apply: Update manifest.ai.status.validation.applied=true
8. V2 refresh: Call refresh_validation_stage_from_index()
9. Promotion: Read index (status="completed") + manifest (applied=true)
10. Runflow updated: status="success", results_total=3, validation_ai_completed=true
```

### Actual Flow for 14b140ab
```
1. Requirements complete ✅
2. Build validation packs → index with status="sent" ✅
3. Legacy AI sender: Send packs (bypassed V2) ⚠️
4. Legacy AI sender: Write results, update manifest ⚠️
5. record_stage: Call promotion with status="sent" ❌
6. Promotion: Sees status="sent" (not "completed") → ready=False ❌
7. Runflow written: status="built", results_total=0 ❌
8. NO REFRESH: refresh_validation_stage_from_index never called ❌
```

---

## Key Insights

### 1. V2 Never Ran
- **Evidence**: No V2 logs, index has legacy status format
- **Impact**: No V2 apply cycle, no V2 refresh, no promotion to success

### 2. Status Recognition Gap
- **Issue**: Promotion logic only recognizes `status: "completed"`
- **Reality**: Index has `status: "sent"` (legacy format)
- **Impact**: Zero results counted even though result files exist

### 3. No Post-Results Refresh
- **Issue**: Even if results exist, no writer ran to promote validation stage
- **Last Writer**: `record_stage` at T7 (17:56:41Z), before results existed
- **Impact**: Runflow frozen at pre-results state

### 4. Manifest vs Runflow Divergence
- **Manifest**: Shows applied=true, results_applied=3 (legacy populated)
- **Runflow**: Shows built, results_total=0 (promotion never ran)
- **Impact**: Two sources of truth disagree

---

## Phase 1 Conclusions

### Root Cause Summary
1. **V2 sender not wired**: Integration point in tasks.py missing or disabled
2. **Legacy path executed**: Old AI sender ran, bypassing V2 apply/refresh
3. **Status mismatch**: Legacy writes `"sent"`, V2 promotion requires `"completed"`
4. **No refresh call**: Even with results on disk, no writer triggered promotion

### Repair Strategy Failure
**Why repair script didn't work for old SIDs**:
- Script calls `refresh_validation_stage_from_index`
- Refresh calls `_validation_results_progress` to count completed records
- Progress logic sees `status: "sent"` → `ready: False`
- Zero records counted → promotion computes `results_total: 0`
- Runflow updated with same stale values (no change)

### Critical Fixes Required
1. **Wire V2 autosend** in tasks.py after pack build
2. **Fix status recognition**: Either update legacy to write `"completed"` OR update promotion logic to accept `"sent"`
3. **Ensure refresh call** after V2 apply completes
4. **Fix FrozenInstanceError** to allow index status updates during apply

---

## Next Steps (Phase 2)

### Immediate Actions
1. Inspect `backend/api/tasks.py` around validation pack build to locate V2 integration point
2. Verify orchestrator mode environment setting
3. Fix status recognition: Add `"sent"` to promotion logic OR ensure V2 updates to `"completed"`
4. Fix FrozenInstanceError: Use `dataclasses.replace()` for index updates

### Integration Verification
1. Add `VALIDATION_V2_AUTOSEND_TRIGGER` after pack build in tasks.py
2. Ensure V2 sender invoked when packs > 0
3. Verify V2 sender updates index status to `"completed"` after apply
4. Verify V2 sender calls refresh after apply success

### Testing Strategy
1. Run fresh validation on SID 14b140ab
2. Verify V2 logs appear
3. Verify runflow shows success/3 results/validation_ready=true
4. Verify no regression on existing tests

---

## Files Requiring Changes

### Critical Path
1. **backend/api/tasks.py**: Add V2 autosend trigger
2. **backend/runflow/decider.py**: Update `_validation_results_progress` status recognition
3. **backend/validation/apply_results_v2.py**: Fix FrozenInstanceError with dataclasses.replace()

### Secondary
4. **backend/runflow/decider.py**: Fix summary_payload NameError
5. **backend/runflow/decider.py**: Fix RunManifest NameError
6. **backend/runflow/decider.py**: Add "sent" to recognized status set

---

---

## Complete Runflow Writer Mapping

### Core Writers (backend/runflow/decider.py)

#### 1. record_stage (Line 1467)
**Purpose**: Main stage persistence writer, records stage status/metrics/results  
**Inputs**:
- `sid`, `stage` (merge/validation/frontend/note_style)
- `status` (success/error/built/published/in_progress/empty/pending)
- `counts` (findings_count, etc.)
- `empty_ok`, `metrics`, `results`
- `runs_root`, `refresh_barriers`

**Execution Flow**:
1. Load existing runflow.json
2. Build stage_payload with status, last_at, metrics, results
3. Call ALL stage promotions: merge, validation, note_style, frontend
4. Merge promoted snapshots with existing stages
5. Update umbrella_barriers (if refresh_barriers=True, default)
6. Persist runflow.json with `last_writer: "record_stage"`

**V2-Awareness**: ✅ Calls `_apply_validation_stage_promotion` which reads manifest for V2 flags  
**Invoked By**: All pipeline stages (tasks.py, orchestrators)  
**Classification**: Main pipeline writer

#### 2. _apply_validation_stage_promotion (Line 2706)
**Purpose**: Authoritative promotion for validation stage from disk artifacts  
**Inputs**:
- `data` (runflow dict)
- `run_dir` (Path to SID folder)

**Execution Flow**:
1. Call `_validation_results_progress(run_dir)` → (total, completed, failed, ready)
2. **Early Exit**: If not ready OR completed != total, return (False, False, log_context)
3. Read manifest.json → extract `validation_ai_applied`, `validation_ai_required` flags
4. Build stage_payload with:
   - **status**: `"success"` (overrides existing)
   - **results**: `{results_total, completed, failed}`
   - **metrics**: `{packs_total, validation_ai_required, validation_ai_completed, validation_ai_applied}`
   - **`_writer`**: `"validation_promotion"` (authoritative marker)
5. Return (True, True, log_context)

**Critical Logic**:
```python
normalized_status = status.lower() if isinstance(status, str) else ""
ready = normalized_status == "completed" and result_jsonl and result_path.exists()
```
**Status Recognition**: ONLY `"completed"` status is counted. Status `"sent"` → `ready: False`.

**V2-Awareness**: ✅ Reads manifest V2 flags, sets authoritative metrics  
**Invoked By**: `record_stage`, `refresh_validation_stage_from_index`, `reconcile_umbrella_barriers`  
**Classification**: Internal promotion (authoritative for validation)

#### 3. refresh_validation_stage_from_index (Line 4174)
**Purpose**: Explicit validation refresh after V2 results/apply complete  
**Inputs**:
- `sid`
- `runs_root` (optional)

**Execution Flow**:
1. Load runflow.json
2. Call `_apply_validation_stage_promotion(data, run_dir)`
3. If not updated, return early
4. Merge promoted snapshot into latest runflow
5. Persist with `last_writer: "refresh_validation_stage"`
6. Log `VALIDATION_STAGE_PROMOTED` if promoted
7. Call `runflow_refresh_umbrella_barriers(sid)` to update barriers

**V2-Awareness**: ✅ Designed for V2 post-apply refresh  
**Invoked By**: `validation_sender_v2.py` (after successful apply), repair scripts  
**Classification**: Main pipeline (V2 refresh writer)

#### 4. reconcile_umbrella_barriers (Line 4359)
**Purpose**: Recompute ALL stage readiness and persist umbrella barriers  
**Inputs**:
- `sid`
- `runs_root` (optional)

**Execution Flow**:
1. Load runflow.json
2. Call ALL stage promotions:
   - `_apply_merge_stage_promotion(data, run_dir)`
   - `_apply_validation_stage_promotion(data, run_dir)`
   - `_apply_frontend_stage_promotion(data, run_dir)`
   - `_apply_note_style_stage_promotion(data, run_dir)`
3. Log `*_STAGE_PROMOTED` for each promoted stage
4. Evaluate global barriers (merge_ready, validation_ready, all_ready, etc.)
5. Persist umbrella_barriers with `checked_at` timestamp
6. Persist runflow.json with `last_writer: "reconcile_umbrella_barriers"`

**V2-Awareness**: ✅ Uses validation promotion which is V2-aware  
**Invoked By**: `record_stage` (via refresh_barriers), `validation_builder.py` (after pack build), repair scripts  
**Classification**: Main pipeline (barrier reconciliation)

#### 5. _merge_stage_snapshot (Line 975)
**Purpose**: Merge two stage snapshots with status priority and authoritative handling  
**Inputs**:
- `existing` (current stage dict)
- `incoming` (new stage dict)

**Merge Logic**:
- **Status**: Uses `_prefer_stage_status()` priority (success > built > error)
- **Metrics/Summary**: If `incoming._writer == "validation_promotion"`, **authoritative** (replace existing)
- **Results**: Always deep merge
- **Other Fields**: Last-write-wins

**Critical Rule**:
```python
is_validation_promotion = incoming.get("_writer") == "validation_promotion"
if is_validation_promotion and key in {"metrics", "summary"}:
    merged[key] = dict(value)  # Replace, not merge
```

**V2-Awareness**: ✅ Recognizes `_writer=validation_promotion` marker for authoritative metrics  
**Invoked By**: `record_stage`, all refresh functions  
**Classification**: Internal merge logic (core merge algorithm)

### Helper Functions

#### 6. _validation_results_progress (Line 3468)
**Purpose**: Count completed validation results from index  
**Inputs**:
- `run_dir` (Path)

**Execution Flow**:
1. Load validation index from `ai_packs/validation/index.json`
2. For each pack record:
   - Normalize status via `_normalize_terminal_status()`
   - Check if `normalized_status == "completed"` AND result file exists
   - Increment `completed` if ready
3. Return (total, completed, failed, ready)

**Critical Status Check**:
```python
normalized_status = _normalize_terminal_status(status_value, stage="validation", run_dir=run_dir)
if normalized_status != "completed":
    ready = False
    continue
```

**Issue**: Only recognizes `"completed"` status. Status `"sent"` maps to `None` → `ready: False`.

**V2-Awareness**: ⚠️ Expects `"completed"` status (V2 format), but legacy writes `"sent"`  
**Invoked By**: `_apply_validation_stage_promotion`  
**Classification**: Internal helper (results counting)

#### 7. _normalize_terminal_status (Line 1318)
**Purpose**: Normalize status strings to canonical values  
**Mapping** (from `_STATUS_NORMALIZATION` dict, lines 57-82):
```python
{
    "completed": "completed",
    "complete": "completed",
    "done": "completed",
    "success": "completed",
    "succeeded": "completed",
    "ok": "completed",
    "finished": "completed",
    "built": "completed",        # NOTE: "built" → "completed"
    "published": "completed",
    "failed": "failed",
    "failure": "failed",
    "error": "failed",
    # ...
    "skipped": "skipped",
}
```

**Critical Gap**: Status `"sent"` is NOT in mapping → returns `None` → warns `RUNFLOW_UNKNOWN_STATUS`.

**V2-Awareness**: ⚠️ Missing `"sent"` status mapping (legacy format)  
**Invoked By**: `_validation_results_progress`, all promotion logic  
**Classification**: Internal utility (status normalization)

### Other Stage Promotions (Similar Pattern)

#### 8. _apply_merge_stage_promotion (Line 2601)
- Similar pattern: calls `_merge_artifacts_progress()`, sets status=success, builds metrics/results
- Not relevant to validation V2 issue

#### 9. _apply_note_style_stage_promotion (Line 2854)
- Similar pattern for note_style results
- Uses same promotion/merge architecture

#### 10. _apply_frontend_stage_promotion (Line 3299)
- Promotes frontend review stage when answers received
- Similar promotion pattern

---

## Summary of Writer Hierarchy

### Main Writers (Persist Runflow)
1. **record_stage**: Calls all promotions, merges, persists, updates barriers
2. **refresh_validation_stage_from_index**: Validation-specific refresh (V2)
3. **refresh_note_style_stage_from_index**: Note style-specific refresh
4. **refresh_frontend_stage_from_responses**: Frontend-specific refresh
5. **reconcile_umbrella_barriers**: Runs all promotions, updates barriers

### Promotion Functions (Compute Snapshots, Don't Persist)
1. **_apply_validation_stage_promotion**: Reads index/manifest, computes validation snapshot
2. **_apply_merge_stage_promotion**: Reads merge artifacts, computes merge snapshot
3. **_apply_note_style_stage_promotion**: Reads note_style results, computes snapshot
4. **_apply_frontend_stage_promotion**: Reads frontend responses, computes snapshot

### Merge Logic (Combines Snapshots)
1. **_merge_stage_snapshot**: Merges two stage snapshots with priority rules
2. **_merge_nested_mapping**: Deep merges nested dicts

### Helper Functions (Read-Only Analysis)
1. **_validation_results_progress**: Counts completed validation results
2. **_merge_artifacts_progress**: Counts merge result files
3. **_note_style_results_progress**: Counts note_style results
4. **_frontend_responses_progress**: Counts frontend answers
5. **_normalize_terminal_status**: Maps status strings to canonical values

---

**Analysis Date**: 2025-01-23  
**SID**: 14b140ab-5de5-436e-9477-811468b181cf  
**Phase**: 1 - Deep Analysis Complete  
**Status**: Ready for Phase 2 (Bug Fixes)


## Final Design (Phase 3)

**Scope**
- Authoritative V2 control over `runflow.json` `stages.validation.*`.
- Legacy/late writers become telemetry-only under V2 mode.
- Idempotent repair path for existing runs; documentation of invariants.

**Authoritative Writers**
- Validation promotion: `_apply_validation_stage_promotion` (via `record_stage`, `refresh_validation_stage_from_index`, `reconcile_umbrella_barriers`).
- V2 apply + merge: `apply_validation_merge_and_update_state` (writes runflow merge_results.applied and minimal manifest monotonic status).
- Barrier refresh: `reconcile_umbrella_barriers` (always recomputes readiness using promotions; never trusts stale fields).

**Legacy Behavior**
- Legacy stage calls (e.g., `record_stage('validation', status='built')`) are telemetry-only when V2 orchestrator mode is ON; they cannot override promotion snapshots.
- Manifest keeps minimal monotonic status for validation: `sent`, `completed_at`, `state` (success/error), and timestamps. It no longer carries authoritative applied-counts.

**Source of Truth**
- Runflow: authoritative for validation readiness and merge application flags.
  - `stages.validation.status`: terminal status from promotions (success/error/empty).
  - `stages.validation.results`: `{results_total, completed, failed}` computed exclusively from index (`status == 'completed'` + results file exists).
  - `stages.validation.metrics.validation_ai_required|completed|applied`: promotion-derived booleans.
  - `stages.validation.merge_results_applied` and `stages.validation.merge_results.applied`: set by V2 merge helper.
- Manifest: telemetry/minimal for UI; never used as applied-count authority.

**Invariants**
- Monotonic completion: once validation is `success`, never downgrade to `built`.
- Status normalization: map equivalent terminals to `completed`/`failed`; legacy `sent` is non-terminal and not counted as completed.
- Writer precedence: incoming `_writer == 'validation_promotion'` replaces `metrics`/`summary` fields; legacy writes cannot clobber.
- Idempotency: V2 apply, refresh, and barrier reconciliation are safe to re-run any time.

**Repair/Backfill**
- Script: `scripts/repair_runflow_validation_from_manifest.py`
  - Runs V2 apply (normalizes index → completed), refreshes validation stage from index, reconciles barriers.
  - Idempotent and safe for bulk use; logs before/after snapshots.
- Quick refresh-only tool: `scripts/manual_refresh_validation.py <SID> [--runs-root PATH]`.

**Operational Commands**
- Orchestrator E2E (V2 authoritative path):
  - PowerShell
    - `Set-Item Env:RUNFLOW_NEW_SID_ON_UPLOAD_ONLY 0`
    - `$sid = [guid]::NewGuid().Guid`
    - `.venv\Scripts\python.exe devtools/run_validation_orchestrator.py $sid runs`
- Backfill/Repair existing runs:
  - All runs: `.venv\Scripts\python.exe scripts/repair_runflow_validation_from_manifest.py --runs-root runs`
  - Single SID: `.venv\Scripts\python.exe scripts/repair_runflow_validation_from_manifest.py --runs-root runs --sid <SID>`

**Expected Artifacts (after successful V2 run)**
- `manifest.json`:
  - `ai.status.validation.state: "success"`, `sent: true`, `completed_at: <ts>`; `results_applied` fields may remain 0/false (non-authoritative).
- `ai_packs/validation/index.json`:
  - `packs[n].status: "completed"` and results files present for each completed account.
- `runflow.json`:
  - `stages.validation.status: "success"`.
  - `stages.validation.results.results_total == packs_completed`.
  - `stages.validation.metrics.validation_ai_completed: true`.
  - `stages.validation.merge_results_applied: true` (after merge helper).
  - `umbrella_barriers.validation_ready: true`.

**Notes on Data-Dependent Outcomes**
- If V2 apply finds no matches for the sample dataset, `validation_ai_applied` can be `false` and `results_applied` can be `0` while the stage is still `success`; readiness still becomes `true` after merge helper sets `merge_results_applied` and barriers are refreshed.

**Gotchas & Flags**
- Ensure manifest creation is allowed for new SIDs: `RUNFLOW_NEW_SID_ON_UPLOAD_ONLY=0` (or unset).
- Orchestrator mode is forced by `devtools/run_validation_orchestrator.py`; no extra flags required.
- A PDF input is required to produce eligible accounts and packs; without inputs, the run completes with `packs=0` and will not produce an `ai_packs/validation/index.json`.
