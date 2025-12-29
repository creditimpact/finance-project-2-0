# Merge Gate Bypass Investigation – SID 5e51359f-7ba8-4be3-8199-44d34c55a4ed

**Date:** 2025-11-18  
**Symptom:** Validation advanced to `status=success` and `ready_latched=true` while `merge_ready=false` for a non-zero-packs run  
**Root Cause:** `_apply_validation_stage_promotion` in decider.py promotes validation WITHOUT checking merge_ready  

---

## 🎯 Executive Summary

**The Problem:**
SID `5e51359f-7ba8-4be3-8199-44d34c55a4ed` shows:
- `validation.status = "success"`
- `validation.ready_latched = true`  
- `validation.ready_latched_at = "2025-11-18T21:05:08Z"`
- `merge_ready = false` (still false at final timestamp `21:16:03Z`)
- `merge.empty_ok = false` (non-zero-packs: expected_packs=1)

**The Root Cause:**
`_apply_validation_stage_promotion()` in `decider.py` (lines 2770-2900) sets `ready_latched=True` based ONLY on disk validation results completion, without ANY check for `merge_ready` or `merge_ai_applied`.

**The Gate Bypass:**
When validation results complete on disk, ANY call to:
- `record_stage("validation", ...)`
- `reconcile_umbrella_barriers()`  
- `refresh_validation_stage_from_index()`

will invoke `_apply_validation_stage_promotion()`, which unconditionally promotes validation to success and latches readiness, regardless of merge status.

**Impact:**
For non-zero-packs runs, validation can complete before merge AI decisions are applied, leading to:
- Race condition between validation AI and merge AI  
- Potential for validation to read stale/incomplete merge data
- Violation of intended staging order

---

## 📋 Section 1: Actual Code vs. Spec Verification

### 1.1 Merge Finalization (finalize_merge_stage)

**Location:** `backend/runflow/decider.py:2169-2530`

**Spec Claim:** "Sets `merge_ai_applied=True` after `record_stage()` completes"

**Actual Code:** ✅ **MATCHES SPEC**

```python
# Line 2504-2527
record_stage(...) # Completes successfully

# ── MERGE_AI_APPLIED FLAG ──
runflow_data = _load_runflow(sid)
if runflow_data is not None:
    stages = _ensure_stages_dict(runflow_data)
    merge_stage = stages.get("merge")
    if isinstance(merge_stage, dict):
        merge_stage["merge_ai_applied"] = True
        merge_stage["merge_ai_applied_at"] = _now_iso()
        _save_runflow(sid, runflow_data)
        log.info("MERGE_AI_APPLIED sid=%s", sid)
```

**Result:** Implementation correct ✅

---

### 1.2 Umbrella Barriers Computation (_compute_umbrella_barriers)

**Location:** `backend/runflow/decider.py:3853-4053`

**Spec Claim:** "Checks `merge_ai_applied` flag for non-zero-packs, sets `merge_ready=False` if missing"

**Actual Code:** ✅ **MATCHES SPEC**

```python
# Line 3975-3988
if merge_ready and not merge_empty_ok:
    merge_ai_applied = merge_stage.get("merge_ai_applied", False) if isinstance(merge_stage, Mapping) else False
    if not merge_ai_applied:
        log.info(
            "MERGE_NOT_AI_APPLIED sid=%s merge_ready_disk=%s merge_empty_ok=%s",
            run_dir.name,
            merge_ready_disk,
            merge_empty_ok,
        )
        merge_ready = False
```

**Result:** Implementation correct ✅

---

### 1.3 Validation Orchestrator Gating

**Location:** `backend/pipeline/validation_orchestrator.py:75-95`

**Spec Claim:** "Gates on `merge_ready` at start of `run_for_sid()`, returns deferred if False"

**Actual Code:** ✅ **MATCHES SPEC**

```python
# Line 85-96
barriers = _compute_umbrella_barriers(run_dir)
merge_ready = barriers.get("merge_ready", False)
if not merge_ready:
    logger.info(
        "VALIDATION_ORCHESTRATOR_DEFERRED sid=%s reason=merge_not_ready barriers=%s",
        sid,
        barriers,
    )
    return {"sid": sid, "deferred": True, "reason": "merge_not_ready"}
```

**Result:** Implementation correct ✅

---

### 1.4 Validation Fastpath Check

**Location:** `backend/runflow/decider.py:717-750`

**Spec Claim:** "Checks `merge_ai_applied` for non-zero-packs in `_maybe_enqueue_validation_fastpath`"

**Actual Code:** ✅ **MATCHES SPEC**

```python
# Line 739-751
merge_empty_ok = _stage_empty_ok(merge_stage)
if not merge_empty_ok:
    merge_ai_applied = merge_stage.get("merge_ai_applied", False)
    if not merge_ai_applied:
        log.info(
            "VALIDATION_FASTPATH_SKIP sid=%s reason=merge_not_ai_applied empty_ok=%s",
            sid,
            merge_empty_ok,
        )
        return False
```

**Result:** Implementation correct ✅

---

### 1.5 Devtools Warning

**Location:** `devtools/run_validation_orchestrator.py:27-37`

**Spec Claim:** "Warns if `merge_ready=False` before running orchestrator"

**Actual Code:** ✅ **MATCHES SPEC**

```python
barriers = _compute_umbrella_barriers(run_dir)
merge_ready = barriers.get("merge_ready", False)
if not merge_ready:
    print(
        f"⚠️  WARNING: merge_ready=False for sid={sid}. "
        f"Validation may fail or produce incomplete results. "
        f"Barriers: {barriers}",
        file=sys.stderr,
    )
```

**Result:** Implementation correct ✅

---

## ✅ **Conclusion for Section 1:**

All claimed implementations in `MERGE_FINALIZATION_FIX_SUMMARY.md` are present and correct in the actual code.

**However**, these implementations only cover **orchestrator-mode validation paths**. They do NOT cover the **promotion path** that runs during barrier reconciliation.

---

## 📊 Section 2: "Who Can Move Validation" Map

### 2.1 Complete Function Inventory

| Location | Function | Changes validation.status? | Sets validation.sent? | Sets validation.ready_latched? | Checks merge_ready? | Checks merge_ai_applied? |
|----------|----------|----------------------------|----------------------|--------------------------------|---------------------|-------------------------|
| `backend/pipeline/validation_orchestrator.py:75` | `ValidationOrchestrator.run_for_sid()` | ✅ Yes (via record_stage_force) | ✅ Yes | ✅ Yes | ✅ **YES** | ⚠️ Indirectly (via barriers) |
| `backend/pipeline/auto_ai_tasks.py:1002` | `validation_merge_ai_results_step()` | ✅ Yes (via record_stage_force) | ✅ Yes | ✅ Yes | ❌ **NO** | ❌ **NO** |
| `backend/runflow/decider.py:2770` | `_apply_validation_stage_promotion()` | ✅ Yes | ✅ Yes | ✅ Yes | ❌ **NO** | ❌ **NO** |
| `backend/runflow/decider.py:2169` | `finalize_merge_stage()` | ❌ No | ❌ No | ❌ No | N/A | ✅ Sets merge_ai_applied |
| `backend/ai/validation_builder.py:2383` | `build_validation_packs_for_run()` | ❌ No | ❌ No | ❌ No | N/A | N/A |
| `backend/ai/validation_builder.py:2020` | `run_validation_send_for_sid()` | ❌ No | ❌ No | ❌ No | N/A | N/A |

---

### 2.2 Classification by Safety

#### ✅ **Safe Paths** (Check merge_ready)

1. **ValidationOrchestrator.run_for_sid()**
   - **Location:** `backend/pipeline/validation_orchestrator.py:75-200`
   - **Checks:** Calls `_compute_umbrella_barriers()` at line 89, checks `merge_ready` at line 90
   - **Behavior:** Returns `{"deferred": True}` if `merge_ready=False`
   - **When Used:** Orchestrator mode (VALIDATION_ORCHESTRATOR_MODE=1)

---

#### ❌ **Unsafe Paths** (Do NOT check merge_ready)

1. **_apply_validation_stage_promotion()**
   - **Location:** `backend/runflow/decider.py:2770-2900`
   - **What It Does:** 
     - Reads validation results progress from disk
     - If `completed == total`, sets:
       - `validation.status = "success"`
       - `validation.sent = True`
       - `validation.ready_latched = True` (if previous status ≠ "success")
       - `validation.ready_latched_at = <timestamp>`
   - **Invoked By:**
     - `record_stage("validation", ...)` – calls promotion internally
     - `reconcile_umbrella_barriers()` – calls promotion
     - `refresh_validation_stage_from_index()` – calls promotion
   - **Checks Performed:** 
     - ✅ Disk result completion
     - ❌ **DOES NOT CHECK merge_ready**
     - ❌ **DOES NOT CHECK merge_ai_applied**
   - **Why Unsafe:** Any completion of validation results on disk triggers immediate promotion, regardless of merge status

2. **validation_merge_ai_results_step()**
   - **Location:** `backend/pipeline/auto_ai_tasks.py:1002-1210`
   - **What It Does:**
     - Merges validation AI results into account summaries
     - Calls `record_stage_force()` to set:
       - `validation.status = "success"`
       - `validation.sent = True`
       - `validation.ready_latched = True`
       - `validation.ready_latched_at = <timestamp>`
   - **Invoked By:** Legacy auto_ai Celery pipeline (when VALIDATION_ORCHESTRATOR_MODE=0)
   - **Checks Performed:**
     - ✅ Orchestrator mode check (skips if mode=1)
     - ❌ **DOES NOT CHECK merge_ready**
     - ❌ **DOES NOT CHECK merge_ai_applied**
   - **Why Unsafe:** Direct write to validation stage without merge gating

---

## 🔍 Section 3: SID 5e51359f Timeline Analysis

### 3.1 Key Timestamps

| Time | Event | Details |
|------|-------|---------|
| `21:04:58Z` | Merge scoring completes | Created 1 pack, merge stage written |
| `21:04:58Z` | Validation requirements run | Finds 24 requirements across 3 accounts |
| `21:04:58Z` | Validation packs built | 1 pack created, 2 skipped |
| `21:04:58Z` | Barriers reconciled | `merge_ready=false`, `validation_ready=false` |
| `21:05:03Z` | Barriers reconciled | `merge_ready=false`, `validation_ready=false` |
| **`21:05:08Z`** | **Validation ready_latched** | **`validation.ready_latched=true`, `validation.ready_latched_at="2025-11-18T21:05:08Z"`** |
| `21:05:08Z` | Barriers reconciled | `merge_ready=false`, **`validation_ready=true`** |
| `21:05:13Z` | Merge scoring re-runs | Same pack rebuilt (likely a retry or recompute) |
| ... | 680+ barrier reconciliations | All show `merge_ready=false`, `validation_ready=true` |
| `21:08:53Z` | Merge finalization | `merge.last_at="2025-11-18T21:08:53Z"` |
| `21:16:03Z` | Final barriers check | Still `merge_ready=false`, `validation_ready=true` |

---

### 3.2 Critical Moment: 21:05:08Z

**What happened:**
At `21:05:08Z`, validation transitioned from `ready_latched=false` to `ready_latched=true`.

**From runflow_events.jsonl:**
```json
{
  "ts": "2025-11-18T21:05:08.469961Z",
  "sid": "5e51359f-7ba8-4be3-8199-44d34c55a4ed",
  "stage": "validation",
  "event": "end",
  "status": "success",
  "umbrella_barriers": {
    "merge_ready": false,
    "validation_ready": true
  },
  "summary": {
    "validation_ai_completed": true,
    "validation_ai_applied": true
  }
}
```

**From runflow.json:**
```json
{
  "validation": {
    "status": "success",
    "sent": true,
    "ready_latched": true,
    "ready_latched_at": "2025-11-18T21:05:08Z",
    "metrics": {
      "validation_ai_required": true,
      "validation_ai_completed": true,
      "validation_ai_applied": false  // ← NOTE: False in final state!
    }
  }
}
```

**Analysis:**
- Validation results completed on disk (`results_total=1`, `completed=1`)
- Something called a function that invoked `_apply_validation_stage_promotion()`
- Promotion set `ready_latched=True` without checking `merge_ready`
- The event log shows `validation_ai_applied=true` momentarily, but runflow.json shows `false` (likely overwritten later)

---

### 3.3 Merge Status at Critical Moment

**At `21:05:08Z`:**
- `merge.status = "success"` (already marked success at 21:04:58Z from scoring)
- `merge.result_files = 1` (pack exists on disk)
- `merge_ai_applied = <MISSING>` (not yet set)
- `merge_ready = false` (computed by barriers)

**Merge finalization happens much later:**
- `merge.last_at = "2025-11-18T21:08:53Z"` (3 minutes 45 seconds after validation latched)
- BUT: `merge_ai_applied` is STILL missing in final runflow.json!

**Why merge_ready stayed false:**
The `_compute_umbrella_barriers` code at line 3975 checks:
```python
if merge_ready and not merge_empty_ok:
    merge_ai_applied = merge_stage.get("merge_ai_applied", False)
    if not merge_ai_applied:
        merge_ready = False
```

Since `merge_ai_applied` was never set for this SID, `merge_ready` remained `false` through all 680+ barrier reconciliations.

---

### 3.4 Who Set ready_latched?

**Function:** `_apply_validation_stage_promotion()` in `decider.py:2770`

**Invoked By:** One of these paths:
1. `record_stage("validation", ...)` – called after validation send/apply
2. `reconcile_umbrella_barriers()` – called repeatedly (680+ times)
3. `refresh_validation_stage_from_index()` – called explicitly

**Evidence:**
- Line 2793-2794 in decider.py:
  ```python
  if previous_status != "success":
      stage_payload["ready_latched"] = True
      stage_payload["ready_latched_at"] = _now_iso()
  ```
- This code runs whenever validation results are complete on disk
- NO check for `merge_ready` or `merge_ai_applied` before latching

**Most Likely Trigger:**
Given 680+ barrier reconciliations after validation completed, `reconcile_umbrella_barriers()` likely invoked promotion and set the latch.

---

## 🚨 Section 4: Gate Bypasses Identified

### 4.1 Primary Bypass: _apply_validation_stage_promotion

**Classification:** ❌ **UNSAFE**

**Location:** `backend/runflow/decider.py:2770-2900`

**Problem:**
```python
def _apply_validation_stage_promotion(
    data: dict[str, Any], run_dir: Path
) -> tuple[bool, bool, dict[str, int]]:
    total, completed, failed, ready = _validation_results_progress(run_dir)
    
    if not ready or completed != total:
        return (False, False, log_context)
    
    # ← NO MERGE_READY CHECK HERE
    
    if previous_status != "success":
        stage_payload["ready_latched"] = True  # ← UNCONDITIONAL LATCH
        stage_payload["ready_latched_at"] = _now_iso()
```

**Why This Bypasses Gate:**
- Promotion is invoked by `record_stage()`, `reconcile_umbrella_barriers()`, etc.
- These functions are called throughout the pipeline for telemetry/state sync
- ANY completion of validation results on disk triggers immediate promotion
- No awareness of merge status

**Impact for SID 5e51359f:**
- Validation results completed at ~21:05:08Z
- Promotion ran (likely via `reconcile_umbrella_barriers`)
- Set `ready_latched=True` despite `merge_ready=false`
- Barrier became `validation_ready=true`, allowing downstream stages to proceed

---

### 4.2 Secondary Bypass: validation_merge_ai_results_step (Auto AI Tasks)

**Classification:** ❌ **UNSAFE** (but not used for this SID)

**Location:** `backend/pipeline/auto_ai_tasks.py:1002-1210`

**Problem:**
```python
def validation_merge_ai_results_step(self, prev):
    if _orchestrator_mode_enabled():
        return payload  # ← Skips in orchestrator mode
    
    # ... merge validation results ...
    
    record_stage_force(
        sid,
        {"stages": {"validation": {
            "status": "success",
            "sent": True,
            "ready_latched": True,  # ← NO MERGE CHECK
            "ready_latched_at": _isoformat_timestamp(),
        }}},
        refresh_barriers=True,
    )
```

**Why This Bypasses Gate:**
- Legacy auto_ai pipeline writes validation success directly
- No check for `merge_ready` or `merge_ai_applied`
- Only skipped when orchestrator mode is ON

**Impact for SID 5e51359f:**
NOT applicable – manifest shows `validation.sent=false`, meaning auto_ai pipeline did NOT run validation send/merge for this SID. This path was not the culprit.

---

### 4.3 Bypass Summary Table

| Path | Function | merge_ready Check? | merge_ai_applied Check? | Used for SID 5e51359f? | Classification |
|------|----------|-------------------|------------------------|----------------------|----------------|
| **Promotion** | `_apply_validation_stage_promotion` | ❌ NO | ❌ NO | ✅ **YES** | ❌ **PRIMARY BYPASS** |
| **Auto AI** | `validation_merge_ai_results_step` | ❌ NO | ❌ NO | ❌ No (orchestrator mode) | ⚠️ Secondary bypass |
| **Orchestrator** | `ValidationOrchestrator.run_for_sid` | ✅ YES | ✅ Indirect | ❌ No (deferred early) | ✅ Safe |
| **Devtools** | `run_validation_orchestrator.py` | ⚠️ Warning only | ⚠️ Warning only | ❌ No evidence | ⚠️ Warns but proceeds |

---

## 🛠️ Section 5: Central Gate Design

### 5.1 Proposed Helper Function

```python
def can_advance_validation(
    run_dir: Path,
    runflow_payload: Mapping[str, Any] | None = None,
    reason: str = "",
) -> bool:
    """
    Central gate to determine if validation can advance for non-zero-packs runs.
    
    Returns:
        True if validation can proceed/be promoted
        False if validation must wait for merge
    
    Logic:
        - Zero-packs (merge.empty_ok=True): Always returns True
        - Non-zero-packs: Returns True only if merge_ready=True
    """
    barriers = _compute_umbrella_barriers(run_dir, runflow_payload)
    merge_ready = barriers.get("merge_ready", False)
    
    # Extract merge.empty_ok from runflow
    if runflow_payload is None:
        runflow_path = run_dir / "runflow.json"
        runflow_payload = _load_json_mapping(runflow_path)
    
    stages = runflow_payload.get("stages", {}) if isinstance(runflow_payload, Mapping) else {}
    merge_stage = stages.get("merge", {}) if isinstance(stages, Mapping) else {}
    empty_ok = _stage_empty_ok(merge_stage) if callable(_stage_empty_ok) else False
    
    if empty_ok:
        # Zero-packs: merge not required
        log.info(
            "VALIDATION_GATE_PASS_ZERO_PACKS sid=%s reason=%s",
            run_dir.name,
            reason or "merge_optional",
        )
        return True
    
    if merge_ready:
        # Non-zero-packs + merge ready
        log.info(
            "VALIDATION_GATE_PASS_MERGE_READY sid=%s reason=%s",
            run_dir.name,
            reason or "merge_complete",
        )
        return True
    
    # Non-zero-packs + merge NOT ready
    log.warning(
        "VALIDATION_GATE_BLOCKED sid=%s reason=%s merge_ready=%s empty_ok=%s barriers=%s",
        run_dir.name,
        reason or "unknown",
        merge_ready,
        empty_ok,
        barriers,
    )
    return False
```

---

### 5.2 Required Call Sites

#### 1. **_apply_validation_stage_promotion (CRITICAL)**

**Before:**
```python
def _apply_validation_stage_promotion(
    data: dict[str, Any], run_dir: Path
) -> tuple[bool, bool, dict[str, int]]:
    total, completed, failed, ready = _validation_results_progress(run_dir)
    
    if not ready or completed != total:
        return (False, False, log_context)
    
    # ... set status="success", ready_latched=True ...
```

**After:**
```python
def _apply_validation_stage_promotion(
    data: dict[str, Any], run_dir: Path
) -> tuple[bool, bool, dict[str, int]]:
    total, completed, failed, ready = _validation_results_progress(run_dir)
    
    if not ready or completed != total:
        return (False, False, log_context)
    
    # ── MERGE_READY GATE ──
    if not can_advance_validation(run_dir, data, reason="promotion"):
        log.info(
            "VALIDATION_PROMOTION_BLOCKED_BY_MERGE sid=%s total=%d completed=%d",
            run_dir.name,
            total,
            completed,
        )
        return (False, False, log_context)
    # ───────────────────────
    
    # ... proceed with promotion ...
```

**Impact:** Prevents ready_latched from being set until merge is ready

---

#### 2. **validation_merge_ai_results_step (Auto AI Tasks)**

**Before:**
```python
def validation_merge_ai_results_step(self, prev):
    # ... merge results ...
    
    record_stage_force(
        sid,
        {"stages": {"validation": {
            "status": "success",
            "ready_latched": True,
        }}},
    )
```

**After:**
```python
def validation_merge_ai_results_step(self, prev):
    # ... merge results ...
    
    # ── MERGE_READY GATE ──
    if not can_advance_validation(run_dir, runs_root_path, reason="auto_ai_merge"):
        logger.warning(
            "VALIDATION_FINALIZE_BLOCKED_BY_MERGE sid=%s",
            sid,
        )
        # Do NOT set ready_latched, but mark merge success as terminal to prevent retry loop
        record_stage_force(
            sid,
            {"stages": {"validation": {"validation_send_terminal": True}}},
        )
        return payload
    # ───────────────────────
    
    record_stage_force(
        sid,
        {"stages": {"validation": {
            "status": "success",
            "ready_latched": True,
        }}},
    )
```

**Impact:** Prevents auto_ai pipeline from finalizing validation before merge

---

#### 3. **ValidationOrchestrator.run_for_sid (Already Safe)**

Already has gate check at line 89-96. No changes needed. ✅

---

#### 4. **Devtools Scripts (Optional - Add Hard Block)**

**Current State:** Warns but proceeds

**Recommended:** Add `--force` flag to allow bypass, otherwise hard-block:

```python
def main() -> int:
    # ... parse args ...
    force = "--force" in sys.argv
    
    barriers = _compute_umbrella_barriers(run_dir)
    merge_ready = barriers.get("merge_ready", False)
    
    if not merge_ready and not force:
        print(
            f"❌ ERROR: merge_ready=False for sid={sid}. "
            f"Validation cannot proceed. Use --force to override.",
            file=sys.stderr,
        )
        return 1
    
    # ... proceed ...
```

---

### 5.3 Invariants to Enforce

For all non-zero-packs runs (`merge.empty_ok=False`, `expected_packs>0`):

```
INVARIANT-1: validation.sent == True 
             → (merge.empty_ok == True OR merge_ready == True)

INVARIANT-2: validation.status == "success" 
             → (merge.empty_ok == True OR merge_ready == True)

INVARIANT-3: validation.ready_latched == True 
             → (merge.empty_ok == True OR merge_ready == True)
```

---

### 5.4 Where to Assert Invariants

#### **1. In _compute_umbrella_barriers (Defensive)**

After computing `validation_ready`:

```python
# At end of _compute_umbrella_barriers, line ~4040
validation_ready = validation_ready_disk  # Already computed

# ── INVARIANT CHECK ──
validation_stage = _stage_mapping("validation")
if validation_ready and isinstance(validation_stage, Mapping):
    validation_latched = validation_stage.get("ready_latched", False)
    validation_status = validation_stage.get("status", "")
    
    if validation_latched or validation_status == "success":
        if not merge_empty_ok and not merge_ready:
            log.error(
                "VALIDATION_INVARIANT_VIOLATION sid=%s validation_latched=%s status=%s merge_ready=%s empty_ok=%s",
                run_dir.name,
                validation_latched,
                validation_status,
                merge_ready,
                merge_empty_ok,
            )
            # Force validation_ready to False to prevent downstream progression
            validation_ready = False
```

**Effect:** Prevents validation_ready from being True when invariant is violated

---

#### **2. In Tests (Regression Prevention)**

```python
def test_validation_invariant_no_latch_before_merge_ready():
    """Validate that non-zero-packs runs cannot latch validation before merge_ready."""
    run_dir = setup_test_run(merge_packs=1, validation_packs=1)
    
    # Complete validation results but NOT merge
    write_validation_results_complete(run_dir)
    
    # Trigger promotion
    runflow = load_runflow(run_dir)
    _apply_validation_stage_promotion(runflow, run_dir)
    
    # Assert validation did NOT latch
    validation_stage = runflow["stages"]["validation"]
    assert validation_stage.get("ready_latched") is not True, \
        "Validation should not latch before merge_ready for non-zero-packs"
    
    # Now complete merge and trigger promotion again
    write_merge_results_complete(run_dir)
    finalize_merge_stage(sid, run_dir)
    _apply_validation_stage_promotion(runflow, run_dir)
    
    # Assert validation NOW latched
    assert validation_stage.get("ready_latched") is True, \
        "Validation should latch after merge_ready"
```

---

#### **3. In prove_merge_fix.py (Production Verification)**

Add invariant checks to the existing proof script:

```python
def verify_invariants(sid: str, run_dir: Path) -> dict[str, bool]:
    """Check all validation/merge invariants for a SID."""
    runflow = load_runflow(run_dir)
    stages = runflow.get("stages", {})
    merge_stage = stages.get("merge", {})
    validation_stage = stages.get("validation", {})
    
    empty_ok = merge_stage.get("empty_ok", False)
    merge_ready = compute_merge_ready(merge_stage, run_dir)
    
    validation_sent = validation_stage.get("sent", False)
    validation_status = validation_stage.get("status", "")
    validation_latched = validation_stage.get("ready_latched", False)
    
    checks = {}
    
    if not empty_ok:  # Non-zero-packs only
        checks["inv1_sent"] = not validation_sent or merge_ready
        checks["inv2_status"] = (validation_status != "success") or merge_ready
        checks["inv3_latched"] = not validation_latched or merge_ready
    
    return checks
```

---

## 📝 Section 6: Deliverables Summary

### 6.1 Code Map: Validation Writers

| Function | File:Line | Changes Status? | Sets Sent? | Sets ready_latched? | Checks merge_ready? |
|----------|-----------|----------------|-----------|---------------------|---------------------|
| `_apply_validation_stage_promotion` | `decider.py:2770` | ✅ Yes | ✅ Yes | ✅ **YES** | ❌ **NO** |
| `ValidationOrchestrator.run_for_sid` | `validation_orchestrator.py:75` | ✅ Yes | ✅ Yes | ✅ **YES** | ✅ **YES** |
| `validation_merge_ai_results_step` | `auto_ai_tasks.py:1002` | ✅ Yes | ✅ Yes | ✅ **YES** | ❌ **NO** |

---

### 6.2 SID-Specific Explanation (5e51359f)

**Exact Function:** `_apply_validation_stage_promotion` (decider.py:2770)

**When:** `2025-11-18T21:05:08Z`

**Trigger:** Validation results completed on disk (1/1), invoked via `reconcile_umbrella_barriers()` or `record_stage()`

**Barriers at Time:**
- `merge_ready = false` (merge_ai_applied missing)
- `merge.empty_ok = false` (non-zero-packs)
- `validation results = 1/1 complete`

**Why Existing Gating Didn't Stop It:**
- `ValidationOrchestrator` gating: Not used (orchestrator mode likely OFF or run via different path)
- `_apply_validation_stage_promotion`: NO merge_ready check implemented
- Promotion ran unconditionally based solely on disk result completion

**Specific Line of Code:**
```python
# backend/runflow/decider.py:2793-2794
if previous_status != "success":
    stage_payload["ready_latched"] = True
    stage_payload["ready_latched_at"] = _now_iso()
```

---

### 6.3 Central Gate Design

**Function Signature:**
```python
def can_advance_validation(
    run_dir: Path,
    runflow_payload: Mapping[str, Any] | None = None,
    reason: str = "",
) -> bool
```

**Call Sites:**
1. `_apply_validation_stage_promotion` (CRITICAL – line 2780)
2. `validation_merge_ai_results_step` (auto_ai_tasks.py:1185)
3. Any future validation finalization paths

**Invariants:**
```
For merge.empty_ok == False:
  validation.sent == True       → merge_ready == True
  validation.status == "success" → merge_ready == True
  validation.ready_latched == True → merge_ready == True
```

**Enforcement:**
- Helper function returns False to block unsafe operations
- Invariant checks in `_compute_umbrella_barriers` force `validation_ready=False` if violated
- Tests verify invariants hold across all scenarios

---

## 🚀 Next Steps

1. **Implement `can_advance_validation()` helper** in `backend/runflow/decider.py`

2. **Add gate check to `_apply_validation_stage_promotion`** (CRITICAL FIX)

3. **Add gate check to `validation_merge_ai_results_step`** (auto_ai pipeline)

4. **Add invariant enforcement to `_compute_umbrella_barriers`** (defensive)

5. **Create regression test** for SID 5e51359f pattern

6. **Update existing tests** to verify invariants

7. **Run backfill script** to repair existing SIDs with violations

8. **Deploy and monitor** for `VALIDATION_GATE_BLOCKED` logs

---

## 🔗 References

- Investigation Document: `MERGE_BARRIER_INVESTIGATION.md`
- Previous Fix: `MERGE_FINALIZATION_FIX_SUMMARY.md`
- Bug SID: `5e51359f-7ba8-4be3-8199-44d34c55a4ed`
- Validation Authority: `RUNFLOW_VALIDATION_AUTHORITY_ANALYSIS.md`
- V2 Integration: `VALIDATION_V2_PRODUCTION_INTEGRATION.md`

---

**Status:** 📋 Investigation Complete – Ready for Implementation  
**Priority:** 🔴 **CRITICAL** – Affects all non-zero-packs runs where validation completes before merge finalization
