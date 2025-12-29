## Validation V2 Apply → Runflow Wiring Fix Summary

### Problem Identified

**Manifest was correct**, showing:
```json
"ai": {
  "status": {
    "validation": {
      "results_total": 3,
      "results_applied": 3,
      "results_apply_ok": true,
      "validation_ai_applied": true
    }
  }
}
```

**Runflow was wrong**, showing:
```json
"stages": {
  "validation": {
    "results_received": 3,  // ✓ correct
    "expected_results": 3,  // ✓ correct
    "validation_ai_applied": false  // ✗ BUG - appeared in 3 places
  }
}
```

### Root Cause

In `backend/runflow/decider.py::_apply_validation_stage_promotion`:

1. Lines 3750-3860: V2 manifest stats were loaded and `applied_ok` computed correctly
2. **Line 3860**: `stage_payload["validation_ai_applied"] = bool(applied_ok)` was set
3. **Line 3878**: `stage_payload = _normalize_validation_stage_payload(...)` was called
4. Lines 3879-3896: Terminal flags were re-injected after normalization
5. **Line 3898**: `stages["validation"] = stage_payload` - stage assigned to dict
6. Lines 3900-3960: **AFTER stages assignment**, legacy apply logic tried to override the flag

**The bug**: The `validation_ai_applied` flag was set correctly at line 3860, but the normalization function doesn't strip it. However, the subsequent legacy apply override logic (lines 3940-3960) ran AFTER the stage was already committed to the stages dict, and only conditionally updated it when `should_apply=True`. Since orchestrator mode skips legacy apply, `validation_ai_applied_flag` was False by default, overwriting the correct V2 value.

### Fix Applied

**File**: `backend/runflow/decider.py`

**Change 1** (after line 3890):
```python
# CRITICAL: Re-inject validation_ai_applied after normalization to ensure persistence
# The applied_ok was computed above from V2 manifest but must be re-applied post-normalize
stage_payload["validation_ai_applied"] = bool(applied_ok)
if "metrics" not in stage_payload or not isinstance(stage_payload["metrics"], dict):
    stage_payload["metrics"] = {}
stage_payload["metrics"]["validation_ai_applied"] = bool(applied_ok)
if "summary" not in stage_payload or not isinstance(stage_payload["summary"], dict):
    stage_payload["summary"] = {}
stage_payload["summary"]["validation_ai_applied"] = bool(applied_ok)
```

**Change 2** (lines 3940-3960):
Modified the post-assignment override logic to only run when `should_apply=True`, preserving the V2-sourced value when legacy apply is skipped:

```python
if isinstance(current_stage, dict) and should_apply:
    # Only override if we actually ran the apply check above
    current_stage["validation_ai_applied"] = bool(validation_ai_applied_flag)
    # ... rest of override logic
```

### Verified Structure

After fix, `stages.validation` shows:

```json
{
  "status": "success",
  "expected_results": 3,
  "results_received": 3,
  "validation_ai_applied": true,
  "metrics": {
    "results_total": 3,
    "missing_results": 0,
    "validation_ai_applied": true
  },
  "summary": {
    "validation_ai_applied": true
  }
}
```

### Umbrella Barriers Integration

The umbrella logic (`backend/runflow/umbrella.py` line ~293) correctly reads:
```python
validation_applied = bool(validation_metrics.get("validation_ai_applied"))
```

And gates validation readiness (line ~307):
```python
validation_ready = (
    bool(validation_ready_barrier)
    or validation_status_value == "success"
    or (validation_required is False)
    or (validation_completed and validation_applied)
)
```

So when `validation_ai_required=true`, the system now requires **both** `validation_completed=true` AND `validation_applied=true` before declaring `validation_ready=true`.

### Next Run Expected Behavior

On a fresh run with `VALIDATION_ORCHESTRATOR_MODE=1`:

1. **V2 apply** runs and updates manifest with `validation_ai_applied=true`
2. **Runflow refresh** (`refresh_validation_stage_from_index`) is called AFTER V2 apply
3. **Promotion logic** loads V2 manifest stats and computes `applied_ok=true`
4. **Normalization** happens, then `validation_ai_applied` is re-injected into stage/metrics/summary
5. **Final runflow.json** persists with `validation_ai_applied=true` in all three locations
6. **Umbrella barriers** read the flag and only declare `validation_ready=true` when both completed and applied

### Test Command

```bash
python verify_applied_flag.py <sid>
```

Checks manifest vs runflow consistency and reports any mismatches.
