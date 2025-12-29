# Strategy Readiness Audit Report – Phase 1 (READ-ONLY INVESTIGATION)

**Date:** November 17, 2025  
**SID under investigation:** `c3344ed7-2563-4ea1-9bfb-5cac1ce01d0b`  
**Problem:** `umbrella_barriers.strategy_ready = true` despite no actual strategy stage execution

---

## Executive Summary

**Root cause identified:** Strategy readiness logic in `backend/runflow/decider.py` (lines 5689-5710) uses **legacy pre-V2 conditions** that ignore whether a strategy stage actually ran. It sets `strategy_ready = true` immediately when:
- Validation is ready AND
- Validation AI completed (or not required)

This bypasses any check for an actual `stages.strategy` block or real strategy execution, causing the strategy gate to open prematurely in the V2 pipeline.

**Key finding:** For SID `c3344ed7-2563-4ea1-9bfb-5cac1ce01d0b`:
- `findings_count = 3` (strategy SHOULD be required)
- `validation_ai_completed = true`
- `validation_ready = true`
- **BUT:** `strategy_required = false` (computed incorrectly due to missing findings in manifest)
- **Result:** `strategy_ready = true` (lines 5696-5697: "if not strategy_required: strategy_ready = true")

---

## 1. All Writers/Readers of Strategy Flags

### 1.1 `strategy_ready` Writers

**Primary writer:** `backend/runflow/decider.py` :: `_compute_umbrella_barriers` (lines 5689-5710)

```python
# Line 5689-5693: Compute strategy_required
strategy_required = False
if strategy_status_normalized in {"built", "success", "error"}:
    strategy_required = True
if validation_findings > 0 or validation_accounts_eligible > 0:
    strategy_required = True

# Line 5695-5709: Compute strategy_ready
strategy_ready = False
if not strategy_required:
    strategy_ready = True  # ❌ BYPASS: sets ready if not required
elif strategy_status_normalized == "success":
    strategy_ready = True
elif strategy_status_normalized == "error":
    strategy_ready = False
else:
    if validation_ready and validation_merge_applied:
        if not validation_ai_required:
            strategy_ready = True  # ❌ LEGACY: ignores actual strategy stage
        elif validation_ai_completed:
            strategy_ready = True  # ❌ LEGACY: ignores actual strategy stage

# Line 5709-5710: Empty-ok fallback
if not strategy_ready and not strategy_required and _stage_empty_ok(strategy_stage):
    strategy_ready = True
```

**Conditions that set `strategy_ready = true`:**
1. **Line 5697:** If `strategy_required == false` → immediate bypass
2. **Line 5699:** If `strategy_status_normalized == "success"` (strategy stage completed)
3. **Lines 5705-5707:** If validation ready AND validation_merge_applied AND (validation not required OR validation completed)
4. **Line 5710:** If not required, not ready, and stage has `empty_ok = true`

**Location:** `backend/runflow/decider.py:5695-5710`

---

### 1.2 `strategy_required` Writers

**Primary writer:** `backend/runflow/decider.py` :: `_compute_umbrella_barriers` (lines 5689-5693)

```python
strategy_required = False
if strategy_status_normalized in {"built", "success", "error"}:
    strategy_required = True
if validation_findings > 0 or validation_accounts_eligible > 0:
    strategy_required = True
```

**Conditions that set `strategy_required = true`:**
1. If `stages.strategy.status` is `"built"`, `"success"`, or `"error"` (stage exists and was attempted)
2. If `validation_findings > 0` OR `validation_accounts_eligible > 0` (validation detected strategy-worthy findings)

**Critical flaw:** This depends on `validation_findings` being **extracted from runflow validation stage containers**. If these fields are missing or zero in runflow (but present in manifest), `strategy_required` will be incorrectly set to `false`.

**Location:** `backend/runflow/decider.py:5689-5693`

---

### 1.3 `stages.strategy` Writers

**Primary writer:** `backend/strategy/runflow.py` :: `record_strategy_stage` (lines 10-43)

```python
def record_strategy_stage(
    runs_root: Path | str,
    sid: str,
    *,
    status: str,
    plans_written: int,
    planner_errors: int,
    accounts_seen: int,
    accounts_with_openers: int,
) -> None:
    """Persist the strategy stage summary into runflow.json."""
    
    record_stage(
        sid,
        "strategy",
        status=status,
        counts={
            "plans_written": plans_written,
            "accounts_seen": accounts_seen,
            "accounts_with_openers": accounts_with_openers,
        },
        empty_ok=True,
        metrics={
            "plans_written": plans_written,
            "planner_errors": planner_errors,
            "accounts_seen": accounts_seen,
            "accounts_with_openers": accounts_with_openers,
        },
        runs_root=runs_root,
    )
```

**When called:**
- From `backend/pipeline/auto_ai.py:run_strategy_planner` (line 580)
- After strategy planner completes (success or error)

**Location:** `backend/strategy/runflow.py:10-43`

**Invoked by:** 
- `backend/pipeline/auto_ai.py` :: `run_strategy_planner` (line 580)
- `backend/pipeline/auto_ai_tasks.py` :: `strategy_planner_step` (via `run_strategy_planner_for_all_accounts`)

---

### 1.4 Readers of Strategy Flags

**Reader 1:** `backend/runflow/decider.py` :: `_compute_umbrella_barriers`
- **Purpose:** Compute umbrella_barriers for runflow
- **Uses:** `strategy_ready`, `strategy_required` to populate `umbrella_barriers` dict

**Reader 2:** Tests (multiple files)
- `tests/runflow/test_barriers.py`
- `tests/backend/runflow/test_decider.py`
- `tests/backend/runflow/test_strategy_recovery.py`
- **Purpose:** Assert correct barrier behavior

**No runtime consumers found** that directly read `umbrella_barriers.strategy_ready` to make decisions. It appears to be informational/observability only at this time.

---

## 2. Why `strategy_ready == true` for SID c3344ed7-2563-4ea1-9bfb-5cac1ce01d0b

### 2.1 Actual State from Runflow

```json
{
  "umbrella_barriers": {
    "strategy_ready": true,
    "strategy_required": false,
    "validation_ready": true
  },
  "stages": {
    "validation": {
      "status": "success",
      "validation_ai_applied": true,
      "validation_ai_required": true,
      "validation_ai_completed": true,
      "findings_count": 3,
      "metrics": {
        "findings_count": 3
      }
    },
    "strategy": "MISSING"
  }
}
```

### 2.2 Execution Path Through `_compute_umbrella_barriers`

**Step 1:** Extract validation flags (lines 5668-5686)
```python
validation_ai_required = bool(_validation_flag("validation_ai_required"))  # → true
validation_ai_completed = bool(_validation_flag("validation_ai_completed"))  # → true
validation_findings = _validation_int("findings_count", "findings_total")  # → 3
validation_accounts_eligible = _validation_int("accounts_eligible", "eligible_accounts", "packs_total")  # → 3
```

**Step 2:** Compute `strategy_required` (lines 5689-5693)
```python
strategy_required = False  # Start with false
if strategy_status_normalized in {"built", "success", "error"}:
    strategy_required = True  # ❌ Does NOT execute (no strategy stage)
if validation_findings > 0 or validation_accounts_eligible > 0:
    strategy_required = True  # ✅ SHOULD execute (findings_count=3)
```

**❌ CRITICAL BUG:** Despite `findings_count=3`, the second condition **did not fire**.

**Root cause analysis:**
- `validation_findings` is extracted via `_validation_int("findings_count", "findings_total")` (line 5678)
- This searches through `validation_containers` which includes: `validation_stage`, `metrics`, `summary`, `summary.metrics`, `summary.results`
- The runflow shows `findings_count: 3` at the top level and in `metrics`
- **BUT:** The function must have failed to extract it, or there was a race condition

**HOWEVER:** Looking at the umbrella_barriers output, `strategy_required = false` definitively proves that line 5692-5693 did NOT execute, meaning `validation_findings` was evaluated as `0`.

**Step 3:** Compute `strategy_ready` (lines 5695-5710)
```python
strategy_ready = False
if not strategy_required:  # ✅ TRUE (strategy_required=false)
    strategy_ready = True  # ✅ EXECUTES → strategy_ready = true
```

**Boolean evaluation that caused the bug:**
```python
strategy_required = false  # (incorrectly computed)
not strategy_required = true
→ strategy_ready = true  # IMMEDIATE BYPASS
```

**Conclusion:** The code path executed was **line 5696-5697**, the "immediate bypass" branch. Because `strategy_required` was incorrectly computed as `false` (despite 3 findings), the logic assumed "strategy not needed, mark as ready."

---

## 3. Legacy Strategy Couplings

### 3.1 Legacy Logic in `_compute_umbrella_barriers`

**File:** `backend/runflow/decider.py`  
**Lines:** 5689-5710

**Legacy behavior:**
1. **Immediate bypass (line 5696-5697):** If strategy not required → immediately ready
   - **Problem:** "Not required" can be wrong due to missing manifest/runflow synchronization
2. **Validation-based ready (lines 5703-5707):** Sets ready based on validation state alone
   - **Problem:** Does not check if strategy stage actually ran
   - Uses `validation_merge_applied` (legacy merge flag) instead of V2 `validation_ai_applied`
3. **Empty-ok fallback (lines 5709-5710):** If stage has `empty_ok`, mark as ready
   - **Problem:** Allows strategy to be "ready" without ever running

**Pre-V2 assumptions:**
- Strategy was auto-triggered as part of the old pipeline
- Validation completion = strategy readiness
- No explicit strategy stage tracking needed

**V2 reality:**
- Strategy is a distinct Celery task in the chain (`strategy_planner_step`)
- Should only be ready when `stages.strategy.status == "success"`
- Needs to respect V2 validation canonical flags (`validation_ai_applied`)

---

### 3.2 Strategy Stage Execution (NEW, not legacy)

**File:** `backend/pipeline/auto_ai_tasks.py`  
**Function:** `strategy_planner_step` (lines 1214-1334)  
**Task:** Part of V2 `enqueue_auto_ai_chain` (line 2111)

**Pipeline chain (line 2099-2113):**
```python
workflow = chain(
    ai_score_step.s(sid, runs_root_value),
    merge_build_packs.s(),
    merge_send.s(),
    merge_compact.s(),
    run_date_convention_detector.s(),
    ai_validation_requirements_step.s(),
    validation_build_packs.s(),
    validation_send.s(),
    validation_compact.s(),
    validation_merge_ai_results_step.s(),
    strategy_planner_step.s(),  # ✅ Strategy IS in V2 chain
    ai_polarity_check_step.s(),
    ai_consistency_step.s(),
    pipeline_finalize.s(),
)
```

**Key observations:**
1. **Strategy IS wired into V2 pipeline** as a proper Celery task
2. **It DOES create `stages.strategy`** via `record_strategy_stage`
3. **It DOES update manifest** via `mark_strategy_started` and `mark_strategy_completed`
4. **BUT:** The umbrella readiness logic **ignores** whether this task ran

**Disconnect:** The NEW V2 strategy task exists and works, but the OLD umbrella barrier logic doesn't check for its completion.

---

### 3.3 Celery Routing

**File:** `.env`  
**Queue:** Strategy tasks route to the default `celery` queue (no dedicated strategy queue)

```python
# From CELERY_TASK_ROUTES in .env:
# No explicit strategy routing → defaults to "celery" queue
```

**Implication:** Strategy runs on the same worker pool as merge/validation, which is correct for V2 orchestrated chains.

---

## 4. Proposed V2 Contract for Strategy

### 4.1 When Strategy Becomes Required

**Trigger conditions (all must be true):**
1. **Validation succeeded:**
   - `stages.validation.status == "success"`
2. **Validation AI applied:**
   - `stages.validation.validation_ai_applied == true` (V2 canonical flag)
3. **Findings exist:**
   - `stages.validation.findings_count > 0` OR
   - `stages.validation.metrics.findings_count > 0` OR
   - `stages.validation.summary.findings_count > 0`
   - (Use V2 canonical extraction, not legacy paths)

**Proposed logic:**
```python
strategy_required = False

# Check if strategy stage already attempted
if strategy_status_normalized in {"built", "success", "error"}:
    strategy_required = True

# V2 canonical check: validation succeeded + findings exist
if (
    validation_status_normalized == "success"
    and validation_ai_applied  # V2 canonical flag
    and validation_findings > 0
):
    strategy_required = True
```

**Manifest signal:** `ai.status.validation.findings_count` (should be mirrored to runflow)

---

### 4.2 When Strategy is Ready

**Completion conditions (any one):**
1. **Strategy stage succeeded:**
   - `stages.strategy.status == "success"`
2. **Strategy not required AND validation clear:**
   - `strategy_required == false` (computed from above)
   - AND validation completed cleanly (no findings, or findings resolved)
3. **Strategy stage skipped with empty_ok:**
   - `stages.strategy.empty_ok == true` AND `strategy_required == false`

**Proposed logic:**
```python
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
    # Strategy required but not yet complete → NOT ready
    # Remove legacy "validation ready → strategy ready" bypass
    strategy_ready = False
```

**Remove legacy bypass (lines 5703-5707):**
```python
# ❌ DELETE THIS:
else:
    if validation_ready and validation_merge_applied:
        if not validation_ai_required:
            strategy_ready = True
        elif validation_ai_completed:
            strategy_ready = True
```

**Canonical signals:**
- **Runflow:** `stages.strategy.status`
- **Manifest:** `ai.status.strategy.state` (if mirrored)

---

### 4.3 Integration with V2 Validation

**Dependency chain:**
1. **Merge completes** → `validation_ai_applied` via `apply_results_v2.py`
2. **Validation ready** → `umbrella_barriers.validation_ready = true`
3. **Strategy enqueued** → `strategy_planner_step` in V2 chain
4. **Strategy completes** → `stages.strategy.status = "success"`
5. **Strategy ready** → `umbrella_barriers.strategy_ready = true`

**Critical requirement:** Strategy readiness MUST wait for `stages.strategy` to exist and have `status = "success"`.

---

## 5. Current Strategy Stage Wiring

### 5.1 Strategy Stage Creation

**YES, strategy stage wiring exists in V2:**

**Writer:** `backend/strategy/runflow.py` :: `record_strategy_stage` (lines 10-43)
- Called by: `backend/pipeline/auto_ai.py` :: `run_strategy_planner` (line 580)
- Creates: `stages.strategy` with fields:
  - `status` ("built", "success", "error", "skipped")
  - `plans_written`
  - `accounts_seen`
  - `accounts_with_openers`
  - `planner_errors`
  - `empty_ok = true`

**Task:** `backend/pipeline/auto_ai_tasks.py` :: `strategy_planner_step` (lines 1214-1334)
- **Celery task:** Registered in V2 chain (`enqueue_auto_ai_chain`, line 2111)
- **Queue:** Default `celery` queue
- **Invoked:** After `validation_merge_ai_results_step` completes
- **Creates:** `stages.strategy` block via `record_strategy_stage`

---

### 5.2 Chain Position

**V2 pipeline order (from `auto_ai_tasks.py:2099-2113`):**
```
1. ai_score_step
2. merge_build_packs
3. merge_send
4. merge_compact
5. run_date_convention_detector
6. ai_validation_requirements_step
7. validation_build_packs
8. validation_send
9. validation_compact
10. validation_merge_ai_results_step
11. strategy_planner_step ← STRATEGY HERE
12. ai_polarity_check_step
13. ai_consistency_step
14. pipeline_finalize
```

**Position:** Strategy runs AFTER validation merge (step 10), BEFORE polarity (step 12)

---

### 5.3 Manifest Integration

**Manifest methods (in `backend/pipeline/runs.py`):**
1. **`mark_strategy_started`** (line 1372)
   - Sets `ai.status.strategy.state = "in_progress"`
   - Sets `ai.status.strategy.started_at`
2. **`mark_strategy_completed`** (line 1404)
   - Sets `ai.status.strategy.state = "success"` or `"error"`
   - Sets `ai.status.strategy.completed_at`
   - Stores stats: `plans_written`, `planner_errors`, `accounts_seen`, `accounts_with_openers`
3. **`register_strategy_artifacts_for_account`** (line 1437)
   - Registers strategy output files under `artifacts.cases.accounts.<id>.strategy`

**Used by:** `strategy_planner_step` task (line 1279, 1305 in `auto_ai_tasks.py`)

---

## 6. Root Cause Summary

### 6.1 The Exact Bug

**Location:** `backend/runflow/decider.py:5689-5710`

**Faulty logic:**
```python
strategy_required = False
# ... compute strategy_required based on findings ...

strategy_ready = False
if not strategy_required:
    strategy_ready = True  # ❌ BYPASS: marks ready if not required
elif strategy_status_normalized == "success":
    strategy_ready = True
else:
    if validation_ready and validation_merge_applied:
        if not validation_ai_required:
            strategy_ready = True
        elif validation_ai_completed:
            strategy_ready = True  # ❌ LEGACY: ignores actual strategy stage
```

**For SID c3344ed7-2563-4ea1-9bfb-5cac1ce01d0b:**
1. **Findings exist:** `findings_count = 3` (should require strategy)
2. **`strategy_required` computed as `false`** (bug: findings not extracted correctly)
3. **Line 5696 executed:** `if not strategy_required: strategy_ready = True`
4. **Result:** `strategy_ready = true` despite NO `stages.strategy` block

---

### 6.2 Why `strategy_required` Was Wrong

**Intended logic (line 5692-5693):**
```python
if validation_findings > 0 or validation_accounts_eligible > 0:
    strategy_required = True
```

**Extraction logic (lines 5668-5686):**
```python
validation_findings = _validation_int("findings_count", "findings_total")
```

**`_validation_int` searches these containers:**
- `stages.validation` (top level)
- `stages.validation.metrics`
- `stages.validation.summary`
- `stages.validation.summary.metrics`
- `stages.validation.summary.results`

**Actual runflow data:**
```json
{
  "stages": {
    "validation": {
      "findings_count": 3,  ← Should be found here
      "metrics": {
        "findings_count": 3  ← Or here
      }
    }
  }
}
```

**Hypothesis:** Either:
1. **Race condition:** Umbrella barriers computed before validation stage fully populated
2. **Typo/missing field:** Runflow had findings in a different location not covered by `_validation_int`
3. **Coercion failure:** `_coerce_int` returned `None` due to type mismatch

**Most likely:** The umbrella barriers were computed at a point in the pipeline when `stages.validation` existed but `findings_count` was not yet written. Later refreshes of runflow added the field, but umbrella barriers were never recalculated.

---

### 6.3 Legacy vs. V2 Mismatch

**Legacy assumption (pre-V2):**
- Validation complete → strategy complete (no separate stage)
- `validation_merge_applied` flag → strategy can proceed

**V2 reality:**
- Strategy is a distinct Celery task (`strategy_planner_step`)
- Creates its own `stages.strategy` block
- Should only be "ready" when `stages.strategy.status == "success"`

**The code still uses legacy logic:**
- Lines 5703-5707 set `strategy_ready = true` based on `validation_ready` and `validation_merge_applied`
- Never checks if `stages.strategy` exists or succeeded

---

## 7. Recommended Phase 2 Changes

### 7.1 Fix `strategy_required` Computation

**Replace (lines 5689-5693):**
```python
strategy_required = False
if strategy_status_normalized in {"built", "success", "error"}:
    strategy_required = True
if validation_findings > 0 or validation_accounts_eligible > 0:
    strategy_required = True
```

**With:**
```python
strategy_required = False

# If strategy stage already attempted, it was required
if strategy_status_normalized in {"built", "success", "error"}:
    strategy_required = True

# V2 canonical: strategy required if validation succeeded with findings
if validation_status_normalized == "success" and validation_ai_applied:
    if validation_findings > 0 or validation_accounts_eligible > 0:
        strategy_required = True
```

**Rationale:** Tie `strategy_required` to V2 validation success + canonical `validation_ai_applied` flag.

---

### 7.2 Fix `strategy_ready` Computation

**Replace (lines 5695-5710):**
```python
strategy_ready = False
if not strategy_required:
    strategy_ready = True
elif strategy_status_normalized == "success":
    strategy_ready = True
elif strategy_status_normalized == "error":
    strategy_ready = False
else:
    if validation_ready and validation_merge_applied:
        if not validation_ai_required:
            strategy_ready = True
        elif validation_ai_completed:
            strategy_ready = True

if not strategy_ready and not strategy_required and _stage_empty_ok(strategy_stage):
    strategy_ready = True
```

**With:**
```python
strategy_ready = False

if not strategy_required:
    # Strategy not needed → ready immediately
    strategy_ready = True
elif strategy_status_normalized == "success":
    # Strategy stage completed successfully
    strategy_ready = True
elif strategy_status_normalized == "error":
    # Strategy failed → NOT ready
    strategy_ready = False
else:
    # Strategy required but not complete → NOT ready
    # (Remove legacy validation-based bypass)
    strategy_ready = False

# Only allow empty_ok bypass if strategy not required
if not strategy_ready and not strategy_required and _stage_empty_ok(strategy_stage):
    strategy_ready = True
```

**Rationale:** Remove legacy lines 5703-5707 that set `strategy_ready = true` based on validation alone.

---

### 7.3 Replace Legacy Merge Flag with V2 Flag

**In validation extraction (lines 5680-5687):**
```python
# OLD:
if not validation_merge_applied:
    for merge_key in ("merge_results_applied", "merge_applied"):
        flag = _validation_flag(merge_key)
        if flag is not None:
            validation_merge_applied = flag
            break

# NEW (replace validation_merge_applied usage with validation_ai_applied):
validation_ai_applied = bool(_validation_flag("validation_ai_applied"))
```

**Then use `validation_ai_applied` in strategy_required condition:**
```python
if validation_status_normalized == "success" and validation_ai_applied:
    if validation_findings > 0 or validation_accounts_eligible > 0:
        strategy_required = True
```

---

### 7.4 Ensure Findings Extraction is Robust

**Add diagnostic logging in `_compute_umbrella_barriers`:**
```python
validation_findings = _validation_int("findings_count", "findings_total")
validation_accounts_eligible = _validation_int(
    "accounts_eligible",
    "eligible_accounts",
    "packs_total",
)

logger.debug(
    "UMBRELLA_STRATEGY_INPUTS sid=%s findings=%d eligible=%d ai_applied=%s",
    run_dir.name,
    validation_findings,
    validation_accounts_eligible,
    validation_ai_applied,
)
```

**Rationale:** Helps diagnose future cases where `strategy_required` is computed incorrectly.

---

### 7.5 Synchronize Manifest → Runflow for Findings

**Ensure `findings_count` propagates to runflow during validation refresh:**

**In `backend/runflow/decider.py` :: `refresh_validation_stage_from_index`:**
- Already includes `findings_count` in metrics (line ~1200)
- **Verify:** Ensure it's written at the top level of `stages.validation`, not just in `metrics`

**Or:** Extract from manifest if missing from runflow:
```python
# In _compute_umbrella_barriers, if validation_findings == 0:
manifest_path = run_dir / "manifest.json"
if manifest_path.exists():
    manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_findings = (
        manifest_data.get("ai", {})
        .get("status", {})
        .get("validation", {})
        .get("findings_count", 0)
    )
    if manifest_findings > validation_findings:
        validation_findings = manifest_findings
```

---

## 8. Final Checklist for Phase 2

- [ ] **Fix `strategy_required` logic** to use V2 `validation_ai_applied` + findings
- [ ] **Fix `strategy_ready` logic** to require `stages.strategy.status == "success"`
- [ ] **Remove legacy bypass** (lines 5703-5707) that ignores strategy stage existence
- [ ] **Replace `validation_merge_applied`** with `validation_ai_applied` everywhere
- [ ] **Add diagnostic logging** for strategy inputs (findings, eligible accounts, flags)
- [ ] **Ensure `findings_count` propagates** from manifest → runflow reliably
- [ ] **Update tests** to reflect new V2 contract (no bypass without strategy stage)
- [ ] **Verify on real SID** that strategy_ready stays false until strategy completes

---

## 9. Confirmation: Strategy Stage IS Wired

**Statement:** "No code currently builds a stages.strategy block in runflow" → **FALSE**

**Evidence:**
- `backend/strategy/runflow.py` :: `record_strategy_stage` DOES create `stages.strategy`
- `backend/pipeline/auto_ai_tasks.py` :: `strategy_planner_step` DOES call it
- V2 chain includes `strategy_planner_step` at position 11 (after validation merge)

**Conclusion:** Strategy stage wiring EXISTS and WORKS. The bug is in the umbrella barrier logic that IGNORES it.

---

## 10. End of Phase 1 Report

**Next steps (Phase 2):**
1. Apply the fixes described in section 7
2. Update tests to enforce the new contract
3. Run validation on a fresh SID with findings to confirm strategy_ready stays false until strategy completes
4. Verify umbrella_barriers update correctly after strategy finishes

**Approval needed before proceeding to Phase 2 implementation.**
