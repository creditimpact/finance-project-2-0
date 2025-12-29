# Validation Natives Wiring Implementation

## Summary

This document describes the implementation of validation natives wiring into manifest.json and runflow.json without sending to AI. This ensures validation is the source of truth for manifest + runflow state.

## Changes Made

### 1. Fixed UnboundLocalError in reconcile_umbrella_barriers

**File:** `backend/runflow/decider.py`

**Issue:** UnboundLocalError when accessing `os.getenv("VALIDATION_ORCHESTRATOR_BYPASS")` at line 5836.

**Fix:** 
- Extracted `os.getenv()` call to a variable before the conditional check
- Added clear debug log: `UMBRELLA_ORCHESTRATOR_SKIP reconcile sid=%s`
- Simplified dict access with proper type guard for umbrellas

**Code:**
```python
bypass_value = os.getenv("VALIDATION_ORCHESTRATOR_BYPASS", "")
if _orchestrator_mode_enabled() and bypass_value.strip().lower() in {"", "0", "false", "no", "off"}:
    snapshot = get_runflow_snapshot(sid, runs_root=runs_root)
    umbrellas = snapshot.get("umbrella_barriers") if isinstance(snapshot, Mapping) else {}
    log.info("UMBRELLA_ORCHESTRATOR_SKIP reconcile sid=%s", sid)
    if not isinstance(umbrellas, Mapping):
        umbrellas = {}
    return {
        "merge_ready": bool(umbrellas.get("merge_ready", False)),
        # ... other flags
    }
```

### 2. Refined ensure_validation_paths Helper

**File:** `backend/runflow/umbrella.py`

**Enhancement:** Added manifest meta marker to indicate paths have been initialized.

**Code:**
```python
# Add meta marker to indicate paths have been initialized
meta = manifest_data.setdefault("meta", {})
meta["validation_paths_initialized"] = True
```

This makes the function:
- Strictly path-only (no status fields)
- Idempotent (safe to call multiple times)
- Self-documenting via the meta marker

### 3. Wired Manifest Update with Deduplication

**File:** `backend/ai/validation_builder.py`

**Status:** Already properly implemented in `_update_manifest_for_run()`.

The function:
- Only called when packs are actually built (`if any(result for result in results.values())`)
- Calls `ensure_validation_paths()` which sets the meta marker
- Logs `VALIDATION_PATHS_INITIALIZED` event
- Safe in orchestrator mode (no legacy dependencies)

### 4. Created record_validation_packs_built Helper

**File:** `backend/runflow/umbrella.py`

**Purpose:** Update runflow.json with validation stage showing packs have been built.

**Behavior:**
```python
def record_validation_packs_built(
    sid: str, 
    runs_root: str | Path,
    packs_count: int,
    expected_results: int | None = None,
) -> None:
    """
    Update runflow.json with validation stage showing packs have been built.
    
    Sets stages.validation with:
    - status: "packs_built"
    - packs_count: number of packs built
    - expected_results: number of results expected (defaults to packs_count)
    - results_received: 0
    - updated_at: current timestamp
    
    Does NOT change run_state (leaves it as VALIDATING).
    Idempotent - safe to call multiple times.
    """
```

**Key Features:**
- Updates `stages.validation` with clear snapshot
- Does NOT change `run_state` (stays as VALIDATING)
- Defensive error handling (non-fatal if write fails)
- Logs `VALIDATION_PACKS_BUILT_RECORDED` on success

### 5. Wired Runflow Update from Validation Builder

**File:** `backend/ai/validation_builder.py`

**Location:** In `build_validation_packs_for_run()` after manifest update

**Code:**
```python
if any(result for result in results.values()):
    _update_manifest_for_run(sid, runs_root_path)
    
    # Record validation packs built status in runflow
    try:
        from backend.runflow.umbrella import record_validation_packs_built
        record_validation_packs_built(
            sid=sid,
            runs_root=runs_root_path,
            packs_count=packs_built,
            expected_results=packs_built,  # In validation, we expect one result per pack
        )
    except Exception:  # pragma: no cover - defensive
        log.warning(
            "VALIDATION_PACKS_BUILT_RECORD_FAILED sid=%s",
            sid,
            exc_info=True,
        )
```

## Verification

### Test Script

**File:** `devtools/test_validation_natives_wiring.py`

**Usage:**
```bash
python devtools\test_validation_natives_wiring.py <SID>
```

**Checks:**
1. ✅ manifest.json has all validation natives
   - `ai.packs.validation` (base, dir, packs, packs_dir, results, results_dir, index, logs)
   - `artifacts.ai.packs.validation` (same keys)
   - `ai.validation` (base, dir)
   - `meta.validation_paths_initialized` = true

2. ✅ runflow.json has validation stage with packs_built status
   - `stages.validation.status` = "packs_built"
   - `stages.validation.packs_count` (number of packs)
   - `stages.validation.expected_results` (number of results expected)
   - `stages.validation.results_received` = 0
   - `stages.validation.updated_at` (timestamp)
   - `meta.validation_paths_initialized` = true

3. ✅ No AI sends occurred
   - Results directory is empty (no .json files)

### Expected Behavior

After running validation pack build:
1. Packs are built successfully
2. manifest.json contains complete validation natives (paths only)
3. runflow.json shows validation stage with `packs_built` status
4. No AI sends occur (results directory stays empty)
5. Run stops cleanly waiting for next step (orchestrator or manual finalization)

## Orchestrator Mode Integration

All changes are safe in orchestrator mode:
- `ensure_validation_paths()` - path-only, no status mutations
- `record_validation_packs_built()` - direct runflow write, no `record_stage()` dependency
- No calls to legacy orchestration functions
- Clean early-return in `reconcile_umbrella_barriers()` with bypass support

## Next Steps

With validation natives wired and state tracking in place:
1. ValidationOrchestrator can send packs to AI (currently suppressed)
2. Orchestrator can wait for all results deterministically
3. Orchestrator can finalize once with complete result set
4. Full deterministic pipeline: build → send → wait → finalize-once

## Log Events

Key log events to watch:
- `VALIDATION_PATHS_INITIALIZED` - Manifest paths written
- `VALIDATION_PACKS_BUILT_RECORDED` - Runflow stage updated
- `UMBRELLA_ORCHESTRATOR_SKIP` - Reconciliation skipped in orchestrator mode
- `VALIDATION_PACKS_BUILT_WRITE_FAILED` - Warning if runflow write fails (non-fatal)

## Files Modified

1. `backend/runflow/decider.py` - Fixed UnboundLocalError, improved early-return
2. `backend/runflow/umbrella.py` - Added meta marker, created record_validation_packs_built
3. `backend/ai/validation_builder.py` - Wired runflow update after pack build

## Files Created

1. `devtools/test_validation_natives_wiring.py` - Verification test script
2. `VALIDATION_NATIVES_WIRING.md` - This documentation
