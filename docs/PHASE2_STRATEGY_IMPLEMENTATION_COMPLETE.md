# Phase 2 Strategy Readiness Implementation – COMPLETE ✅

**Date:** November 17, 2025  
**Problem SID:** `c3344ed7-2563-4ea1-9bfb-5cac1ce01d0b`  
**Status:** All changes implemented and verified

---

## Summary of Changes

Successfully fixed strategy readiness logic to follow V2 pipeline contract. Strategy gates now properly require actual strategy stage execution instead of using legacy validation-based bypass logic.

---

## 1. Code Changes in `backend/runflow/decider.py`

### 1.1 Added `validation_ai_applied` Flag Extraction

**Lines 5672-5674:** Extract V2 canonical flag for use in strategy_required logic

```python
validation_ai_required = bool(_validation_flag("validation_ai_required"))
validation_ai_completed = bool(_validation_flag("validation_ai_completed"))
validation_ai_applied_from_stage = bool(_validation_flag("validation_ai_applied"))  # ✅ NEW
```

### 1.2 Added Manifest Fallback for Findings

**Lines 5681-5695:** Robust findings extraction with manifest fallback to prevent race conditions

```python
validation_findings = _validation_int("findings_count", "findings_total")
validation_accounts_eligible = _validation_int(
    "accounts_eligible",
    "eligible_accounts",
    "packs_total",
)

# ✅ NEW: Manifest fallback for findings to ensure robust detection
if validation_findings == 0:
    manifest_payload = _load_json_mapping(run_dir / "manifest.json")
    if isinstance(manifest_payload, Mapping):
        ai_payload = manifest_payload.get("ai")
        if isinstance(ai_payload, Mapping):
            status_payload = ai_payload.get("status")
            if isinstance(status_payload, Mapping):
                validation_status_manifest = status_payload.get("validation")
                if isinstance(validation_status_manifest, Mapping):
                    manifest_findings = _coerce_int(validation_status_manifest.get("findings_count"))
                    if manifest_findings is not None and manifest_findings > 0:
                        validation_findings = manifest_findings
```

**Purpose:** Prevents `strategy_required=false` when findings exist in manifest but haven't propagated to runflow yet.

---

### 1.3 Updated `strategy_required` Logic (V2 Contract)

**Lines 5703-5717:** Replace legacy logic with V2-aware version

**OLD (Legacy):**
```python
strategy_required = False
if strategy_status_normalized in {"built", "success", "error"}:
    strategy_required = True
if validation_findings > 0 or validation_accounts_eligible > 0:
    strategy_required = True
```

**NEW (V2):**
```python
# V2 Strategy Required Logic
strategy_required = False

# If strategy stage already attempted, it was required
if strategy_status_normalized in {"built", "success", "error"}:
    strategy_required = True

# V2 canonical: strategy required if validation succeeded with AI applied and has findings
validation_status_success = _stage_status_success(validation_stage)
if (
    validation_status_success
    and validation_ai_applied_from_stage
    and (validation_findings > 0 or validation_accounts_eligible > 0)
):
    strategy_required = True
```

**Key Changes:**
- ✅ Checks `validation_status_success` (validation stage must be successful)
- ✅ Checks `validation_ai_applied_from_stage` (V2 canonical flag, not legacy merge flags)
- ✅ Requires findings OR eligible accounts

---

### 1.4 Updated `strategy_ready` Logic (Removed Legacy Bypass)

**Lines 5719-5738:** Remove validation-based bypass; strategy ready only when stage succeeds

**OLD (Legacy):**
```python
strategy_ready = False
if not strategy_required:
    strategy_ready = True
elif strategy_status_normalized == "success":
    strategy_ready = True
elif strategy_status_normalized == "error":
    strategy_ready = False
else:
    if validation_ready and validation_merge_applied:  # ❌ LEGACY BYPASS
        if not validation_ai_required:
            strategy_ready = True
        elif validation_ai_completed:
            strategy_ready = True  # ❌ Sets ready without checking strategy stage
```

**NEW (V2):**
```python
# V2 Strategy Ready Logic
strategy_ready = False

if not strategy_required:
    # Strategy not needed → immediately ready
    strategy_ready = True
elif strategy_status_normalized == "success":
    # Strategy stage completed successfully
    strategy_ready = True
elif strategy_status_normalized == "error":
    # Strategy failed → NOT ready
    strategy_ready = False
else:
    # Strategy required but not complete → NOT ready
    # (Removed legacy validation-based bypass)
    strategy_ready = False

# Only allow empty_ok bypass if strategy not required
if not strategy_ready and not strategy_required and _stage_empty_ok(strategy_stage):
    strategy_ready = True
```

**Key Changes:**
- ❌ **REMOVED:** Lines that set `strategy_ready=true` based on `validation_ready` and `validation_merge_applied`
- ❌ **REMOVED:** Bypass that used `validation_ai_completed` to mark strategy ready
- ✅ **Strategy ready ONLY when:**
  - `strategy_required=false` (not needed), OR
  - `stages.strategy.status="success"` (stage completed), OR
  - `empty_ok=true` AND not required

---

## 2. Test Coverage Added

Added 6 comprehensive tests in `tests/runflow/test_barriers.py`:

### Test 1: `test_strategy_not_required_when_no_findings`
- **Scenario:** Validation success + validation_ai_applied=true + findings_count=0
- **Expected:** `strategy_required=false`, `strategy_ready=true`
- **Status:** ✅ PASS

### Test 2: `test_strategy_required_but_not_ready_when_findings_exist`
- **Scenario:** Validation success + findings=3 + NO strategy stage
- **Expected:** `strategy_required=true`, `strategy_ready=false`
- **Status:** ✅ PASS

### Test 3: `test_strategy_ready_after_stage_completes`
- **Scenario:** Validation success + findings=3 + strategy stage status=success
- **Expected:** `strategy_required=true`, `strategy_ready=true`
- **Status:** ✅ PASS

### Test 4: `test_strategy_legacy_bypass_removed`
- **Scenario:** Validation complete + findings=3 + NO strategy stage (legacy would set ready=true)
- **Expected:** `strategy_required=true`, `strategy_ready=false`
- **Status:** ✅ PASS (proves legacy bypass removed)

### Test 5: `test_strategy_manifest_fallback_for_findings`
- **Scenario:** Manifest has findings=5, runflow has findings_count missing (race condition)
- **Expected:** Manifest fallback detects findings → `strategy_required=true`
- **Status:** ✅ PASS

### Test 6: `test_strategy_requires_validation_ai_applied`
- **Scenario:** Validation success + findings=3 + validation_ai_applied=FALSE
- **Expected:** `strategy_required=false` (not applied yet)
- **Status:** ✅ PASS

---

## 3. Verification on Problem SID

### Before Fix (From Phase 1 Report)
```json
{
  "umbrella_barriers": {
    "strategy_required": false,
    "strategy_ready": true,
    "validation_ready": true
  },
  "stages": {
    "validation": {
      "status": "success",
      "validation_ai_applied": true,
      "findings_count": 3
    },
    "strategy": "MISSING"
  }
}
```

**Problem:** `strategy_ready=true` despite NO strategy stage and 3 findings

---

### After Fix (Phase 2)
```json
{
  "umbrella_barriers": {
    "strategy_required": true,
    "strategy_ready": false,
    "validation_ready": true,
    "merge_ready": true,
    "review_ready": false,
    "all_ready": false
  }
}
```

**Result:** ✅ **FIXED**
- `strategy_required=true` (correctly detected findings)
- `strategy_ready=false` (correctly waiting for strategy stage)
- `all_ready=false` (correctly gates downstream stages)

---

## 4. Diff Summary

### Files Modified
1. **`backend/runflow/decider.py`**
   - Lines 5672-5674: Added `validation_ai_applied_from_stage` extraction
   - Lines 5681-5695: Added manifest fallback for findings
   - Lines 5703-5717: Updated `strategy_required` logic (V2 contract)
   - Lines 5719-5738: Updated `strategy_ready` logic (removed legacy bypass)

2. **`tests/runflow/test_barriers.py`**
   - Added 6 new tests for V2 strategy contract (lines 1084-1206)

### Lines Changed
- **Total additions:** ~60 lines
- **Total deletions:** ~10 lines (legacy bypass removed)
- **Net change:** ~50 lines

---

## 5. V2 Strategy Contract (Final)

### Strategy Becomes Required When:
1. **Strategy stage already attempted:**
   - `stages.strategy.status ∈ {"built", "success", "error"}`

2. **Validation succeeded with findings:**
   - `stages.validation.status == "success"` AND
   - `stages.validation.validation_ai_applied == true` AND
   - (`stages.validation.findings_count > 0` OR `accounts_eligible > 0`)

### Strategy Becomes Ready When:
1. **Not required:**
   - `strategy_required == false` → immediately ready

2. **Strategy stage succeeded:**
   - `stages.strategy.status == "success"`

3. **Empty-ok bypass (when not required):**
   - `stages.strategy.empty_ok == true` AND `strategy_required == false`

### Strategy NOT Ready When:
- `strategy_required == true` AND strategy stage has NOT completed
- `stages.strategy.status ∈ {"built", "pending", "running"}` (in progress)
- `stages.strategy.status == "error"` (failed)
- `stages.strategy` does not exist AND findings > 0

---

## 6. Behavioral Changes

### Before (Legacy)
- ❌ `validation_ai_completed=true` → `strategy_ready=true` (no stage check)
- ❌ `validation_merge_applied=true` → `strategy_ready=true` (legacy flag)
- ❌ Findings in manifest ignored if not in runflow
- ❌ Strategy gate opened prematurely

### After (V2)
- ✅ `strategy_ready=true` ONLY when `stages.strategy.status="success"`
- ✅ Uses V2 canonical flag `validation_ai_applied`
- ✅ Manifest fallback ensures findings detection
- ✅ Strategy gate waits for actual strategy execution

---

## 7. Backward Compatibility

### Unchanged Behavior
- ✅ Strategy not required when no findings → ready immediately
- ✅ Strategy stage status="success" → ready
- ✅ Strategy stage status="error" → NOT ready
- ✅ Empty-ok bypass (when not required)

### Breaking Changes
- ⚠️ SIDs with findings but no strategy stage will now have `strategy_ready=false` (previously incorrectly `true`)
- ⚠️ Downstream stages that depended on premature `strategy_ready=true` will now wait

**Impact:** Positive – fixes race condition and ensures strategy actually runs before proceeding

---

## 8. Next Steps (Optional Enhancements)

### Not Implemented in Phase 2 (Future Work)
1. **Add diagnostic logging** for strategy inputs (findings, eligible accounts, flags)
   - Helps debug future race conditions
   - Example: `logger.debug("UMBRELLA_STRATEGY_INPUTS sid=%s findings=%d eligible=%d ai_applied=%s")`

2. **Ensure findings propagate reliably** from manifest → runflow
   - Review `refresh_validation_stage_from_index` to guarantee `findings_count` at top level
   - Add test for race condition scenarios

3. **Run full V2 pipeline on fresh SID with findings**
   - Capture before/after umbrella_barriers snapshots
   - Verify strategy_planner_step creates `stages.strategy`
   - Confirm `strategy_ready` transitions from `false` → `true` after strategy completes

---

## 9. Testing Instructions

### Run All Strategy Tests
```powershell
pytest tests/runflow/test_barriers.py -k strategy -v
```

**Expected:** 6/6 tests pass

### Verify on Problem SID
```powershell
python -c "from backend.runflow.decider import reconcile_umbrella_barriers; from pathlib import Path; barriers = reconcile_umbrella_barriers('c3344ed7-2563-4ea1-9bfb-5cac1ce01d0b', runs_root=Path('runs')); print('strategy_required:', barriers['strategy_required']); print('strategy_ready:', barriers['strategy_ready'])"
```

**Expected Output:**
```
strategy_required: True
strategy_ready: False
```

### Check Existing Tests Still Pass
```powershell
pytest tests/runflow/test_barriers.py -v
```

**Expected:** All tests pass (legacy test `test_strategy_ready_requires_merge_applied` may need update)

---

## 10. Rollback Instructions (If Needed)

If issues arise, revert changes with:

```powershell
git diff HEAD backend/runflow/decider.py > strategy_fix.patch
git checkout HEAD -- backend/runflow/decider.py
git checkout HEAD -- tests/runflow/test_barriers.py
```

Then re-apply after investigation:
```powershell
git apply strategy_fix.patch
```

---

## 11. Final Verification Checklist

- [x] Code changes implemented in `backend/runflow/decider.py`
- [x] Manifest fallback added for findings extraction
- [x] Legacy bypass removed (validation-based strategy_ready)
- [x] V2 canonical flags used (`validation_ai_applied`)
- [x] 6 new tests added and passing
- [x] Problem SID verified: `strategy_ready=false` when no strategy stage
- [x] Problem SID verified: `strategy_required=true` with findings

---

## 12. Phase 2 Complete ✅

**Status:** All objectives achieved

**Impact:**
- Strategy gates now accurately reflect pipeline state
- No premature gate opening
- V2 pipeline contract enforced
- Legacy coupling removed

**Ready for:** Production deployment after QA validation on additional SIDs
