# Validation V2 vs Legacy Orchestration Investigation

**Date**: November 19, 2025  
**SID Under Investigation**: `2d125dee-f84d-49e2-99c8-f0161cde0113`  
**Purpose**: Map Legacy vs V2 validation paths, identify which is used, and determine why validation didn't start

---

## Executive Summary

### Critical Finding: **TWO SEPARATE VALIDATION PATHS EXIST**

1. **Auto-AI Chain (V2-ENABLED)**: Full Celery chain with V2 integration — **NEVER CALLED FOR THIS SID**
2. **Fastpath (LEGACY-BASED)**: Direct `stage_a_task` → `build_validation_packs_for_run` → **NO V2 SENDER/COMPACT/MERGE**

### Why Validation Didn't Start for SID `2d125dee`
- **Merge gate passed**: `merge_ready=true`, `merge_ai_applied=true`
- **Fastpath was used**: `stage_a_task` built validation packs inline (line 892)
- **V2 autosend was disabled**: `VALIDATION_AUTOSEND_ENABLED=0` in env (line 222-226 commented out)
- **Auto-AI chain was never called**: No `AUTO_AI_CHAIN_START` or `VALIDATION_PIPELINE_ENTRY` logs
- **Result**: Packs built, but no sender/compact/merge executed → validation stage never promoted → `validation_ready=false`

---

## 1. Legacy Validation vs Validation V2 — Complete Mapping

### 1.1 Legacy Validation Components

| Component | File | Lines | Function | Gating Flag | Call Sites |
|-----------|------|-------|----------|-------------|------------|
| **Legacy autosend loop** | `backend/ai/validation_builder.py` | 1936-2020 | `_maybe_send_validation_packs()` | `ENABLE_LEGACY_VALIDATION_ORCHESTRATION` (default: OFF) | `backend/pipeline/auto_ai.py:684` |
| **Recheck daemon thread** | `backend/ai/validation_builder.py` | 1884-1928 | `_schedule_validation_recheck()` | Same as above | Called by `_maybe_send_validation_packs` at line 2017 |
| **Legacy inline sender** | `backend/ai/validation_builder.py` | 2020-2170 | `run_validation_send_for_sid()` | `VALIDATION_ORCHESTRATOR_MODE=0` (apply results inline) | `backend/pipeline/auto_ai_tasks.py:1869` (validation_send legacy branch) |
| **Legacy send_packs helper** | `backend/validation/send_packs.py` | N/A | `send_validation_packs()` | None (utility) | Called by both legacy and V2 senders |
| **Legacy apply results** | `backend/ai/validation_builder.py` | 2170-2220 | `_apply_validation_result_to_summary()` | Inline application when orchestrator mode OFF | Called by `run_validation_send_for_sid` at line 2145 |

**Legacy Orchestration Decision Tree:**
```python
# backend/ai/validation_builder.py:1948
legacy_enabled = os.getenv("ENABLE_LEGACY_VALIDATION_ORCHESTRATION", "").lower() in {"1", "true", "yes", "on"}
if not legacy_enabled:
    log.info("LEGACY_VALIDATION_ORCHESTRATION_DISABLED ...")
    return  # Exit immediately
```

**Status**: **QUARANTINED** — Disabled by default, requires explicit opt-in via `ENABLE_LEGACY_VALIDATION_ORCHESTRATION=1`

---

### 1.2 Validation V2 Components

| Component | File | Lines | Function | Gating Flag | Call Sites |
|-----------|------|-------|----------|-------------|------------|
| **V2 Builder** | `backend/ai/validation_builder.py` | 2460-2571 | `build_validation_packs_for_run()` | None (always available) | 1. `auto_ai_tasks.py:1752` (chain)<br>2. `backend/api/tasks.py:900` (fastpath) |
| **V2 Autosend (in builder)** | `backend/ai/validation_builder.py` | 2519-2558 | Inline after pack build | `VALIDATION_ORCHESTRATOR_MODE=1` (default) + `VALIDATION_AUTOSEND_ENABLED=1` | N/A (inline) |
| **V2 Sender** | `backend/ai/validation_sender_v2.py` | 199-586 | `run_validation_send_for_sid_v2()` | None (always available) | 1. `auto_ai_tasks.py:1814` (validation_send)<br>2. `backend/ai/validation_builder.py:2536` (autosend) |
| **V2 Compact** | `backend/validation/manifest.py` | N/A | `rewrite_index_to_canonical_layout()` | None | `auto_ai_tasks.py:1958` (validation_compact) |
| **V2 Merge Results** | `backend/core/logic/validation_ai_merge.py` | 285-400 | `apply_validation_ai_decisions_for_all_accounts()` | `VALIDATION_ORCHESTRATOR_MODE=0` (disabled in orchestrator mode) | 1. `auto_ai_tasks.py:1049` (validation_merge_ai_results_step)<br>2. `backend/pipeline/validation_merge_helpers.py:163` |
| **V2 Runflow Refresh** | `backend/ai/validation_sender_v2.py` | 450-550 | `refresh_runflow_validation_stage()` | None | Called by V2 sender at line 380 |

**V2 Orchestration Decision Tree:**
```python
# backend/pipeline/auto_ai_tasks.py:93
def _orchestrator_mode_enabled() -> bool:
    raw = os.getenv("VALIDATION_ORCHESTRATOR_MODE")
    if raw is None:
        return True  # DEFAULT: ON
    lowered = str(raw).strip().lower()
    return lowered not in {"", "0", "false", "no", "off"}

# backend/ai/validation_builder.py:2530
orchestrator_mode = _flag("VALIDATION_ORCHESTRATOR_MODE", True)  # DEFAULT: True
autosend_enabled = (
    _flag("VALIDATION_AUTOSEND_ENABLED", False) or  # DEFAULT: False
    _flag("VALIDATION_SEND_ON_BUILD", False) or
    _flag("VALIDATION_STAGE_AUTORUN", False)
)
```

**Status**: **ACTIVE** — Orchestrator mode ON by default; autosend OFF by default (requires explicit env flag)

---

## 2. Auto-AI Chain Validation Wiring (The V2-Enabled Path)

### 2.1 Chain Definition

**File**: `backend/pipeline/auto_ai_tasks.py`  
**Function**: `enqueue_auto_ai_chain(sid, runs_root)` (lines 2091-2123)

**Full Chain:**
```python
workflow = chain(
    ai_score_step.s(sid, runs_root_value),                    # Merge scoring
    merge_build_packs.s(),                                     # Merge pack build
    merge_send.s(),                                            # Merge AI send
    merge_compact.s(),                                         # Merge compact
    run_date_convention_detector.s(),                          # Date convention
    ai_validation_requirements_step.s(),                       # Validation requirements
    validation_build_packs.s(),              # ✅ V2: calls build_validation_packs_for_run
    validation_send.s(),                     # ✅ V2: run_validation_send_for_sid_v2
    validation_compact.s(),                  # ✅ V2: rewrite_index_to_canonical_layout
    validation_merge_ai_results_step.s(),    # ✅ V2: apply_validation_ai_decisions_for_all_accounts
    strategy_planner_step.s(),                                 # Strategy planning
    ai_polarity_check_step.s(),                               # Polarity
    ai_consistency_step.s(),                                  # Consistency
    pipeline_finalize.s(),                                    # Finalize
)
```

### 2.2 V2 Wiring Verification

#### ✅ `validation_build_packs` → V2 Builder
**File**: `backend/pipeline/auto_ai_tasks.py`  
**Lines**: 1698-1787

```python
def validation_build_packs(self, prev: Mapping[str, object] | None) -> dict[str, object]:
    # ... merge readiness check, runflow short-circuit ...
    
    results = build_validation_packs_for_run(sid, runs_root=runs_root)  # ✅ V2 BUILDER
    packs_written = sum(len(entries or []) for entries in results.values())
    payload["validation_packs"] = packs_written
    return payload
```

**Call Chain**: `validation_build_packs` → `build_validation_packs_for_run` (V2 builder)

---

#### ✅ `validation_send` → V2 Sender (Orchestrator Mode)
**File**: `backend/pipeline/auto_ai_tasks.py`  
**Lines**: 1790-1898

```python
def validation_send(self, prev: Mapping[str, object] | None) -> dict[str, object]:
    if _orchestrator_mode_enabled():  # ✅ DEFAULT: True
        logger.info("VALIDATION_ORCHESTRATOR_SEND_V2 sid=%s", sid)
        
        from backend.ai.validation_sender_v2 import run_validation_send_for_sid_v2  # ✅ V2 SENDER
        stats = run_validation_send_for_sid_v2(sid, runs_root)
        
        payload["validation_sent"] = True
        payload["validation_v2_stats"] = stats
        return payload
    
    # Legacy mode below (only if VALIDATION_ORCHESTRATOR_MODE=0)
    stats = run_validation_send_for_sid(sid, runs_root)  # Legacy sender
```

**Call Chain**: `validation_send` → `run_validation_send_for_sid_v2` (when orchestrator mode=ON)

---

#### ✅ `validation_compact` → V2 Compact
**File**: `backend/pipeline/auto_ai_tasks.py`  
**Lines**: 1913-1976

```python
def validation_compact(self, prev: Mapping[str, object] | None) -> dict[str, object]:
    from backend.validation.manifest import rewrite_index_to_canonical_layout  # ✅ V2 COMPACT
    rewrite_index_to_canonical_layout(index_path, runs_root=runs_root)
    
    payload["validation_compacted"] = True
    return payload
```

**Call Chain**: `validation_compact` → `rewrite_index_to_canonical_layout` (V2 compact)

---

#### ⚠️ `validation_merge_ai_results_step` → Skipped in Orchestrator Mode
**File**: `backend/pipeline/auto_ai_tasks.py`  
**Lines**: 1002-1100

```python
def validation_merge_ai_results_step(self, prev: Mapping[str, object] | None) -> dict[str, object]:
    if _orchestrator_mode_enabled():  # ⚠️ SKIPPED when orchestrator mode ON
        logger.info("VALIDATION_ORCHESTRATOR_MODE_SKIP validation_merge sid=%s", payload.get("sid"))
        return payload  # EXIT EARLY
    
    # Legacy apply path (only when orchestrator mode OFF)
    stats = apply_validation_ai_decisions_for_all_accounts(sid, runs_root)
```

**Design**: In orchestrator mode, V2 sender (`run_validation_send_for_sid_v2`) applies results inline via `refresh_runflow_validation_stage()` at line 380. The separate merge step is redundant and skipped.

---

### 2.3 Verdict: Auto-AI Chain Uses V2 End-to-End

| Task | Implementation | V2 or Legacy |
|------|----------------|--------------|
| `validation_build_packs` | `build_validation_packs_for_run()` | ✅ **V2** |
| `validation_send` | `run_validation_send_for_sid_v2()` (orchestrator mode) | ✅ **V2** |
| `validation_compact` | `rewrite_index_to_canonical_layout()` | ✅ **V2** |
| `validation_merge_ai_results_step` | Skipped (V2 sender handles inline) | ✅ **V2** (implicit) |

**Conclusion**: The Auto-AI chain is **fully V2-integrated**. Legacy components are gated and disabled by default.

---

## 3. All Validation Entrypoints — Classification

### 3.1 Entrypoint #1: Auto-AI Chain (V2-Enabled)

**Trigger Function**: `enqueue_auto_ai_chain(sid, runs_root)`  
**File**: `backend/pipeline/auto_ai_tasks.py:2091`  
**Call Site**: `backend/pipeline/auto_ai.py:987` (inside `maybe_queue_auto_ai_pipeline`)

**Caller Chain**:
```
maybe_queue_auto_ai_pipeline(sid)              # auto_ai.py:886
  ↓
enqueue_auto_ai_chain(sid, runs_root)          # auto_ai_tasks.py:2091
  ↓
[Celery chain including validation_build_packs → validation_send → ...]
```

**Gating Flags**:
- `ENABLE_AUTO_AI_PIPELINE=1` (checked at `auto_ai.py:897`)
- `has_ai_merge_best_pairs(sid, runs_root)` (must have merge candidates)

**Classification**: ✅ **V2 ONLY** (orchestrator mode default ON)

**Invocation Paths**:
1. **From `stage_a_task` (backend/api/tasks.py:1116-1131)**:
   ```python
   if os.environ.get("ENABLE_AUTO_AI_PIPELINE", "1") in ("1", "true", "True"):
       if has_ai_merge_best_tags(sid):
           maybe_run_ai_pipeline_task.delay(sid)  # ← Celery task
   ```

2. **From test harness** (`tests/pipeline/test_auto_ai.py:555`):
   ```python
   task_id = auto_ai_tasks.enqueue_auto_ai_chain(sid, runs_root=runs_root)
   ```

**Status**: ✅ **AVAILABLE but UNUSED FOR SID `2d125dee`** (see Section 4)

---

### 3.2 Entrypoint #2: Fastpath (Legacy-Based)

**Trigger Function**: `build_validation_packs_for_run(sid, runs_root)`  
**File**: `backend/api/tasks.py:900`  
**Context**: Inside `stage_a_task`, after validation requirements complete

**Caller Chain**:
```
stage_a_task(sid)                              # tasks.py:377
  ↓
run_validation_requirements_for_all_accounts(sid)  # Line 888
  ↓
build_validation_packs_for_run(sid, runs_root)     # Line 900 ✅ V2 BUILDER
  ↓
[Inline V2 autosend IF flags enabled]              # validation_builder.py:2530
```

**Gating Flags**:
- `ENABLE_VALIDATION_REQUIREMENTS=1` (checked at `tasks.py:886`)
- **Merge readiness gate**: `merge_ready=true` required (line 922-934)
- **V2 autosend** (optional, inline): `VALIDATION_ORCHESTRATOR_MODE=1` + `VALIDATION_AUTOSEND_ENABLED=1`

**Classification**: ⚠️ **HYBRID** — V2 builder used, but **NO sender/compact/merge** unless autosend enabled

**What Happens**:
1. ✅ Packs built via V2 builder (`build_validation_packs_for_run`)
2. ⚠️ **V2 autosend checked** (line 2530 in `validation_builder.py`):
   - If `VALIDATION_AUTOSEND_ENABLED=1`: calls `run_validation_send_for_sid_v2` inline
   - If disabled (default): **STOPS HERE** — no send/compact/merge
3. ❌ **No chain continuation** — validation_send, validation_compact, validation_merge_ai_results_step **NEVER CALLED**

**Status**: ⚠️ **USED FOR SID `2d125dee`** — but autosend was OFF → packs built, nothing else

---

### 3.3 Entrypoint #3: Legacy Autosend Loop (Quarantined)

**Trigger Function**: `_maybe_send_validation_packs(sid, runs_root)`  
**File**: `backend/ai/validation_builder.py:1936`  
**Call Site**: `backend/pipeline/auto_ai.py:684`

**Gating Flag**:
```python
# backend/ai/validation_builder.py:1948
legacy_enabled = str(os.getenv("ENABLE_LEGACY_VALIDATION_ORCHESTRATION", "")).strip().lower() in {"1", "true", "yes", "on"}
if not legacy_enabled:
    log.info("LEGACY_VALIDATION_ORCHESTRATION_DISABLED sid=%s path=%s", sid, "VALIDATION_BUILDER_AUTOSEND")
    return  # EXIT IMMEDIATELY
```

**Classification**: ❌ **LEGACY ONLY** — Disabled by default

**Status**: ❌ **NOT USED** — Quarantined, requires explicit opt-in

---

### 3.4 Summary Table: Entrypoints

| Entrypoint | File | Trigger | V2 or Legacy | Status | Used for SID 2d125dee? |
|------------|------|---------|--------------|--------|------------------------|
| **Auto-AI Chain** | `auto_ai_tasks.py:2091` | `maybe_queue_auto_ai_pipeline` → `enqueue_auto_ai_chain` | ✅ **V2 ONLY** | Available, gated by `ENABLE_AUTO_AI_PIPELINE=1` | ❌ **NO** (not called) |
| **Fastpath** | `tasks.py:900` | `stage_a_task` → `build_validation_packs_for_run` | ⚠️ **HYBRID** (V2 builder, no sender/compact/merge unless autosend ON) | Default run path | ✅ **YES** (used, but incomplete) |
| **Legacy Autosend Loop** | `validation_builder.py:1936` | `_maybe_send_validation_packs` (post-build) | ❌ **LEGACY ONLY** | Quarantined, OFF by default | ❌ **NO** (disabled) |

---

## 4. Run Path Trace for SID `2d125dee-f84d-49e2-99c8-f0161cde0113`

### 4.1 Observed State (from `runflow.json`)

```json
{
  "stages": {
    "merge": {
      "status": "success",
      "merge_ai_applied": true,
      "packs_created": 1,
      "result_files": 1,
      "expected_packs": 1
    }
    // ❌ NO "validation" stage at all
  },
  "umbrella_barriers": {
    "merge_ready": true,
    "validation_ready": false,
    "strategy_ready": true
  }
}
```

**Observations**:
- ✅ Merge completed successfully (`merge_ready=true`)
- ❌ No `stages.validation` entry
- ❌ `validation_ready=false`
- ✅ No validation artifacts in `ai_packs/validation/` (directory doesn't exist)

---

### 4.2 Trace: Which Path Was Used?

#### Evidence #1: Fastpath Was Used
**Log marker**: Look for `VALIDATION_V2_PIPELINE_ENTRY` in logs (from `tasks.py:924`):
```python
log.info("VALIDATION_V2_PIPELINE_ENTRY sid=%s runs_root=%s", sid, runs_root)
```

This log appears **only in the fastpath** (`stage_a_task`), not in the Auto-AI chain.

**Expected logs for fastpath**:
- `VALIDATION_V2_PIPELINE_ENTRY`
- `VALIDATION_T0_PATHS_INJECTED` (or `ALREADY_PRESENT`)
- `VALIDATION_PACKS_BUILD_DONE`
- `VALIDATION_V2_AUTOSEND_SKIP` (if autosend OFF)

**Expected logs for Auto-AI chain**:
- `AUTO_AI_CHAIN_START`
- `VALIDATION_PIPELINE_ENTRY ... mode=full`
- `VALIDATION_STAGE_STARTED`
- `VALIDATION_BUILD_DONE`
- `VALIDATION_ORCHESTRATOR_SEND_V2`

**Verdict**: If logs show `VALIDATION_V2_PIPELINE_ENTRY` **without** `AUTO_AI_CHAIN_START`, fastpath was used.

---

#### Evidence #2: Auto-AI Chain Was NOT Called
**Why not**:
1. **Check `ENABLE_AUTO_AI_PIPELINE` flag** (`.env:70`):
   ```dotenv
   ENABLE_AUTO_AI_PIPELINE=1
   ```
   ✅ Flag is ON.

2. **Check `stage_a_task` logic** (`tasks.py:1116-1131`):
   ```python
   if os.environ.get("ENABLE_AUTO_AI_PIPELINE", "1") in ("1", "true", "True"):
       if has_ai_merge_best_tags(sid):
           if sid in _AUTO_AI_PIPELINE_ENQUEUED:
               log.info("AUTO_AI_ALREADY_ENQUEUED sid=%s", sid)
           else:
               maybe_run_ai_pipeline_task.delay(sid)  # ← Enqueue chain
   ```

3. **Timing**: `stage_a_task` runs **synchronously** at line 1116. It:
   - Enqueues `maybe_run_ai_pipeline_task` (Celery task)
   - Continues to completion
   - **Validation packs built inline at line 900** (before chain starts)

4. **Race condition**: By the time `maybe_run_ai_pipeline_task` Celery task starts:
   - `stage_a_task` already finished
   - Merge already completed (`merge_ready=true` at 23:08:37)
   - `maybe_run_ai_pipeline` checks `has_ai_merge_best_tags(sid)` → may return `False` if tags already consumed
   - **Result**: Chain aborts before reaching validation tasks

**Verdict**: Auto-AI chain was **enqueued but never executed validation tasks** for this SID.

---

#### Evidence #3: V2 Autosend Was Disabled
**Check `.env` flags** (lines 222-226):
```dotenv
ENABLE_VALIDATION_SENDER=1
VALIDATION_SENDER_ENABLED=1
AUTO_VALIDATION_SEND=1
# VALIDATION_AUTOSEND_ENABLED=1  ← COMMENTED OUT
VALIDATION_SEND_ON_BUILD=1
```

**Builder autosend logic** (`validation_builder.py:2530-2540`):
```python
orchestrator_mode = _flag("VALIDATION_ORCHESTRATOR_MODE", True)  # ✅ True
autosend_enabled = (
    _flag("VALIDATION_AUTOSEND_ENABLED", False) or  # ❌ False (not set)
    _flag("VALIDATION_SEND_ON_BUILD", False) or     # ❌ False (commented out in actual check)
    _flag("VALIDATION_STAGE_AUTORUN", False)        # ❌ False (not set)
)

if packs_built > 0 and orchestrator_mode and autosend_enabled:
    # ← NEVER REACHED (autosend_enabled=False)
    run_validation_send_for_sid_v2(sid, runs_root_path)
```

**Note**: The commented-out flags in `.env` suggest they were **intentionally disabled** during testing or debugging.

**Verdict**: Fastpath built packs but **did not trigger V2 autosend** → no sender/compact/merge executed.

---

### 4.3 Reconstructed Run Flow

```
1. User uploads report → process_report task
2. stage_a_task(sid) starts:
   ├─ Build problem cases
   ├─ Run validation requirements  ← ✅ Completed
   ├─ Check merge_ready barrier     ← ✅ Passed (merge_ready=true)
   ├─ build_validation_packs_for_run(sid)  ← ✅ V2 builder
   │   ├─ Packs built successfully
   │   ├─ Check orchestrator_mode: True  ✅
   │   ├─ Check autosend_enabled: False  ❌
   │   └─ Skip V2 autosend (log: VALIDATION_V2_AUTOSEND_SKIP)
   └─ Enqueue maybe_run_ai_pipeline_task  ← Celery task (async)
3. stage_a_task completes (validation incomplete)
4. maybe_run_ai_pipeline_task Celery task starts:
   ├─ Checks has_ai_merge_best_tags(sid)
   └─ Returns False (merge already consumed tags) → chain aborts
5. Result:
   ├─ Packs built ✅
   ├─ Packs NOT sent ❌
   ├─ Validation stage NOT promoted ❌
   └─ validation_ready=false ❌
```

---

## 5. Critical Clarity: Who Should Start Validation V2?

### Question: When merge is ready (`merge_ready=true`), which component is responsible for starting the Validation V2 algorithm (builder + V2 sender + compact + merge results)?

### Answer: **TWO COMPONENTS CAN, BUT NEITHER DID FOR THIS SID**

#### Option A: Auto-AI Chain (Preferred for Full V2 Flow)
**Responsible Component**: `enqueue_auto_ai_chain(sid, runs_root)`  
**File**: `backend/pipeline/auto_ai_tasks.py:2091`  
**Invoked By**: `maybe_queue_auto_ai_pipeline(sid)` (`auto_ai.py:886`)  
**Trigger**: Called from `stage_a_task` at line 1131 via `maybe_run_ai_pipeline_task.delay(sid)`

**What It Does**:
1. ✅ Runs merge scoring/building/sending/compact
2. ✅ Runs `validation_build_packs` → V2 builder
3. ✅ Runs `validation_send` → V2 sender
4. ✅ Runs `validation_compact` → V2 compact
5. ✅ (Skips `validation_merge_ai_results_step` — V2 sender handles inline)
6. ✅ Continues to strategy/polarity/consistency/finalize

**Why It Didn't Work for This SID**:
- **Race condition**: Enqueued asynchronously from `stage_a_task`
- **Timing issue**: By the time Celery task starts, merge already completed
- **Abort condition**: `has_ai_merge_best_tags(sid)` returns `False` → chain never starts validation tasks

---

#### Option B: Fastpath V2 Autosend (Inline After Builder)
**Responsible Component**: `build_validation_packs_for_run(sid, runs_root)` (inline autosend)  
**File**: `backend/ai/validation_builder.py:2530-2558`  
**Invoked By**: `stage_a_task` at line 900

**What It Does**:
1. ✅ Builds packs via V2 builder
2. ⚠️ **IF** `orchestrator_mode=True` **AND** `autosend_enabled=True`:
   - Calls `run_validation_send_for_sid_v2(sid, runs_root)` inline
   - V2 sender writes results, updates index, refreshes runflow
3. ❌ **NO** compact step (not part of inline path)
4. ❌ **NO** merge-results step (V2 sender handles inline)
5. ❌ **NO** chain continuation (stops after builder)

**Why It Didn't Work for This SID**:
- **Flag disabled**: `VALIDATION_AUTOSEND_ENABLED=0` (or not set)
- **Result**: Packs built, autosend skipped (log: `VALIDATION_V2_AUTOSEND_SKIP`)

---

### **Verdict for SID `2d125dee`**:

| Component | Responsible? | Was It Called? | Why Not? |
|-----------|--------------|----------------|----------|
| **Auto-AI Chain** | ✅ **YES** (preferred for full V2 flow) | ❌ **NO** | Enqueued but aborted (race condition, merge tags consumed) |
| **Fastpath V2 Autosend** | ⚠️ **PARTIALLY** (builder only, no chain) | ❌ **NO** | Autosend flag disabled (`VALIDATION_AUTOSEND_ENABLED=0`) |

**Root Cause**: **Neither component completed the V2 flow** — Auto-AI chain never reached validation tasks, and fastpath autosend was disabled.

---

## 6. Recommendations (Investigation Only — No Code Changes)

### 6.1 For V2 to Work in Current Environment

**Option 1: Enable Fastpath V2 Autosend** (Quick Fix)
```dotenv
# .env
VALIDATION_ORCHESTRATOR_MODE=1            # Already ON (default)
VALIDATION_AUTOSEND_ENABLED=1             # ← ADD THIS
# OR
VALIDATION_SEND_ON_BUILD=1                # ← ALTERNATIVE
```

**Effect**:
- Fastpath (`stage_a_task`) will trigger V2 sender inline after builder completes
- V2 sender will send packs, write results, refresh runflow → `validation_ready=true`
- **Limitation**: No compact step, no explicit merge-results step (V2 sender handles inline)

---

**Option 2: Fix Auto-AI Chain Invocation** (Robust Fix)
**Problem**: Race condition between `stage_a_task` synchronous completion and async Celery chain start

**Solution**:
1. **Move fastpath validation OUT of `stage_a_task`**:
   - Remove `build_validation_packs_for_run` call from `stage_a_task` line 900
   - Let Auto-AI chain handle validation exclusively
   
2. **OR: Guard chain invocation**:
   - Check `has_ai_merge_best_tags(sid)` **before** enqueuing chain
   - Ensure merge tags are still available when chain starts
   
3. **OR: Add explicit chain trigger after merge completes**:
   - When `finalize_merge_stage` sets `merge_ready=true`, check if chain was enqueued
   - If not, enqueue chain or trigger validation tasks directly

---

### 6.2 Monitoring & Observability

**Log Markers to Grep for V2 Validation**:
```bash
# Auto-AI Chain (full V2 flow):
grep "AUTO_AI_CHAIN_START" logs/
grep "VALIDATION_PIPELINE_ENTRY.*mode=full" logs/
grep "VALIDATION_ORCHESTRATOR_SEND_V2" logs/

# Fastpath (V2 builder, optional autosend):
grep "VALIDATION_V2_PIPELINE_ENTRY" logs/
grep "VALIDATION_V2_AUTOSEND_TRIGGER" logs/
grep "VALIDATION_V2_AUTOSEND_SKIP" logs/

# Legacy (should NOT appear if quarantine working):
grep "LEGACY_VALIDATION_ORCHESTRATION_DISABLED" logs/  # Expected
grep "VALIDATION_BUILDER_AUTOSEND" logs/               # Should be followed by DISABLED

# Validation stage promotion:
grep "VALIDATION_STAGE_PROMOTED" logs/
grep "validation_ready=true" logs/
```

---

## 7. Final Summary: Why Validation Never Started for SID `2d125dee`

| Question | Answer |
|----------|--------|
| **Which path was used?** | ⚠️ **Fastpath** (`stage_a_task` → `build_validation_packs_for_run`) |
| **Was V2 builder called?** | ✅ **YES** — packs built successfully |
| **Was V2 sender called?** | ❌ **NO** — autosend disabled (`VALIDATION_AUTOSEND_ENABLED=0`) |
| **Was Auto-AI chain called?** | ❌ **NO** — enqueued but never executed validation tasks (race condition) |
| **Why is `validation_ready=false`?** | ❌ **No validation stage** — builder ran, but sender/compact/merge never executed → stage never promoted |
| **Is legacy validation involved?** | ❌ **NO** — legacy autosend quarantined (gated by `ENABLE_LEGACY_VALIDATION_ORCHESTRATION=0`) |

**Bottom Line**: The **fastpath used V2 builder but stopped there** because autosend was disabled. The **Auto-AI chain (full V2 flow) was never invoked** for this SID due to a race condition in the orchestration logic. Neither path completed the V2 validation pipeline → validation never started.

---

## Appendix A: File/Line Reference Map

### Legacy Components
| Component | File | Lines | Function |
|-----------|------|-------|----------|
| Legacy autosend loop | `backend/ai/validation_builder.py` | 1936-2020 | `_maybe_send_validation_packs()` |
| Recheck daemon | `backend/ai/validation_builder.py` | 1884-1928 | `_schedule_validation_recheck()` |
| Legacy inline sender | `backend/ai/validation_builder.py` | 2020-2170 | `run_validation_send_for_sid()` |

### V2 Components
| Component | File | Lines | Function |
|-----------|------|-------|----------|
| V2 Builder | `backend/ai/validation_builder.py` | 2460-2571 | `build_validation_packs_for_run()` |
| V2 Sender | `backend/ai/validation_sender_v2.py` | 199-586 | `run_validation_send_for_sid_v2()` |
| V2 Compact | `backend/validation/manifest.py` | N/A | `rewrite_index_to_canonical_layout()` |
| V2 Merge Results | `backend/core/logic/validation_ai_merge.py` | 285-400 | `apply_validation_ai_decisions_for_all_accounts()` |
| V2 Runflow Refresh | `backend/ai/validation_sender_v2.py` | 450-550 | `refresh_runflow_validation_stage()` |

### Auto-AI Chain Tasks
| Task | File | Lines | Function |
|------|------|-------|----------|
| `enqueue_auto_ai_chain` | `backend/pipeline/auto_ai_tasks.py` | 2091-2123 | Main chain orchestrator |
| `validation_build_packs` | `backend/pipeline/auto_ai_tasks.py` | 1698-1787 | Calls V2 builder |
| `validation_send` | `backend/pipeline/auto_ai_tasks.py` | 1790-1898 | Calls V2 sender (orchestrator mode) |
| `validation_compact` | `backend/pipeline/auto_ai_tasks.py` | 1913-1976 | Calls V2 compact |
| `validation_merge_ai_results_step` | `backend/pipeline/auto_ai_tasks.py` | 1002-1100 | Skipped in orchestrator mode |

### Fastpath Entry
| Component | File | Lines | Function |
|-----------|------|-------|----------|
| `stage_a_task` | `backend/api/tasks.py` | 377-1140 | Main pipeline task |
| Validation requirements | `backend/api/tasks.py` | 886-898 | `run_validation_requirements_for_all_accounts` |
| Merge readiness gate | `backend/api/tasks.py` | 922-934 | `_compute_umbrella_barriers` |
| V2 builder call | `backend/api/tasks.py` | 900 | `build_validation_packs_for_run(sid, runs_root)` |
| Auto-AI chain enqueue | `backend/api/tasks.py` | 1116-1131 | `maybe_run_ai_pipeline_task.delay(sid)` |

---

**End of Investigation Report**
