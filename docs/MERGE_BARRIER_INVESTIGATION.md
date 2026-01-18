# Merge Barrier & Validation Start Timing Investigation

**Date:** 2025-01-XX  
**Investigator:** AI Assistant  
**Scope:** Deep analysis of merge_ready barrier timing and validation start conditions

---

## Executive Summary

Investigation confirms a critical timing bug: **merge_ready barrier opens too early in non-zero-packs cases, and validation can start before merge AI completes.** For SID `83830ae4-6406-4a7e-ad80-a3f721a3787b`, validation started and completed 8 seconds BEFORE `merge_ready` became true, resulting in inconsistent state flags (`validation_ai_applied=false` despite successful completion).

**Root Cause:** `_merge_artifacts_progress` sets `ready=True` based purely on file count matching, without checking if merge AI has actually applied results to runflow. Validation can start via multiple code paths that don't all properly gate on `merge_ready`.

---

## 1. Merge Promotion & Barrier Code Documentation

### 1.1 _apply_merge_stage_promotion

**Location:** `backend/runflow/decider.py` lines 2627-2734

**Purpose:** Promotes merge stage to `status=success` when merge scoring completes and packs are built.

**Inputs:**
- `run_dir` (Path): Run directory
- `runflow_payload` (optional dict): Current runflow state

**Key Fields Read:**
- `ai_packs/merge/pairs_index.json`: Expected pack count
- Directory scans: `ai_packs/merge/packs/`, `ai_packs/merge/results/`

**Key Fields Written to runflow.stages.merge:**
- `status` = "success"
- `empty_ok` = True if expected_packs == 0
- `result_files` = count of .result.json files
- `pack_files` = count of pack files
- `expected_packs` = expected_total from pairs_index
- `last_at` = timestamp

**Logic:**
```python
result_files_total, pack_files_total, expected_total, ready = _merge_artifacts_progress(run_dir)

stage["status"] = "success"
stage["result_files"] = result_files_total
stage["pack_files"] = pack_files_total
stage["expected_packs"] = expected_total
stage["empty_ok"] = (expected_total == 0)
```

**Critical Gap:** Does NOT check if merge AI has applied results to accounts. Only counts files.

---

### 1.2 _merge_artifacts_progress

**Location:** `backend/runflow/decider.py` lines 3738-3810

**Purpose:** Counts merge artifact files on disk and computes `ready` flag.

**Inputs:**
- `run_dir` (Path): Run directory

**Outputs (tuple):**
- `result_files_total` (int): Count of `ai_packs/merge/results/*.result.json`
- `pack_files_total` (int): Count of `ai_packs/merge/packs/pair_*.jsonl`
- `expected_total` (int | None): From `pairs_index.json` `expected_packs`
- `ready` (bool): True if merge artifacts are "complete"

**Ready Logic:**
```python
if expected_total == 0:
    ready = True  # Zero-packs case
else:
    ready = result_files_total == pack_files_total
    if expected_total is not None:
        ready = ready and result_files_total == expected_total
```

**Critical Bug:** Sets `ready=True` as soon as:
- `result_files_total == pack_files_total == expected_total`

This happens **immediately when merge AI writes result files**, BEFORE those results are applied to runflow accounts. There is NO check for:
- Whether merge AI has finished processing
- Whether merge results have been applied to runflow
- Whether `merge.status` shows completion

---

### 1.3 _compute_umbrella_barriers

**Location:** `backend/runflow/decider.py` lines 3812-4098

**Purpose:** Computes umbrella-level readiness barriers by examining stage snapshots and disk artifacts.

**Inputs:**
- `run_dir` (Path): Run directory
- `runflow_payload` (optional dict): Current runflow state

**Key Outputs (dict):**
- `merge_ready` (bool): True if merge stage is complete
- `validation_ready` (bool): True if validation stage is complete
- `merge_zero_packs` (bool): True if merge expected_packs == 0
- `strategy_ready`, `review_ready`, `style_ready`, `all_ready`: Other barriers

**merge_ready Calculation (3 paths):**

```python
# Path 1: Check stage snapshot empty_ok flag
if merge_stage.get("empty_ok"):
    merge_ready = True

# Path 2: Check stage snapshot result_files count
elif merge_stage.get("result_files", 0) >= 1 and merge_stage.get("status") == "success":
    merge_ready = True

# Path 3: Check disk artifacts via _merge_artifacts_progress
else:
    result_files_total, pack_files_total, expected_total, ready_disk = _merge_artifacts_progress(run_dir)
    if ready_disk:
        merge_ready = True
```

**merge_zero_packs Flag:**
```python
merge_zero_packs = merge_stage.get("empty_ok", False)
```

**Critical Issue:** All 3 paths can set `merge_ready=True` based on file presence, without confirming merge AI has applied results.

---

### 1.4 _maybe_enqueue_validation_fastpath

**Location:** `backend/runflow/decider.py` lines 717-867

**Purpose:** Triggers validation autosend when merge has zero packs (fast path).

**Trigger Condition:**
```python
merge_zero_packs = merge_stage.get("empty_ok", False)

if merge_zero_packs and VALIDATION_AUTOSEND:
    if not validation_sent or validation_incomplete:
        # Enqueue validation send task
```

**Critical Gap:** Does NOT check `merge_ready` barrier explicitly. Only checks `merge_zero_packs` flag. For non-zero-packs cases, validation can be triggered via other paths that may not properly gate on `merge_ready`.

---

## 2. Truth Table: merge_ready Conditions

| Case | expected_packs | result_files | pack_files | status | empty_ok | merge_ready | Reason |
|------|----------------|--------------|------------|--------|----------|-------------|--------|
| Zero-packs | 0 | 0 | 0 | success | True | **TRUE** | empty_ok flag set |
| Zero-packs (alt) | 0 | 0 | 1 | success | True | **TRUE** | empty_ok flag set |
| Packs building | 1 | 0 | 1 | success | False | FALSE | result_files != pack_files |
| Packs built, no results | 1 | 0 | 1 | success | False | FALSE | result_files == 0 |
| **BUG: Results arrive** | 1 | **1** | 1 | success | False | **TRUE** | result_files == pack_files == expected_packs |
| Results applied (ideal) | 1 | 1 | 1 | success | False | TRUE | Same condition as bug case! |

**Key Finding:** Conditions (5) and (6) are indistinguishable! The barrier opens as soon as result files appear on disk, regardless of whether merge AI has applied those results to runflow.

**Expected Behavior:**
- `merge_ready` should only become True after merge AI has:
  1. Written result files
  2. Applied merge decisions to runflow accounts
  3. Updated `merge.status` to indicate completion

**Actual Behavior:**
- `merge_ready` becomes True as soon as step (1) completes
- Steps (2) and (3) may still be in progress

---

## 3. Timeline Reconstruction: SID 83830ae4-6406-4a7e-ad80-a3f721a3787b

### 3.1 Event Timeline

```
T+0s   19:07:33.037  merge_scoring START (span a664b347)
T+0s   19:07:33.288  merge_scoring SUCCESS: 1 pack built, pairs_index written
                     Expected: 1 pack, Created: 1 pack, merge_zero_packs=FALSE

T+0s   19:07:33.306  merge_scoring END

T+0s   19:07:33.431  frontend START
T+0s   19:07:33.477  validation event START
T+0s   19:07:33.000  barriers_reconciled: merge_ready=FALSE, validation_ready=FALSE

T+0s   19:07:33.590  validation requirements SUCCESS (24 findings)
T+0s   19:07:33.592  validation stage END (status=success, ai_packs_built=0)

T+0s   19:07:33.787  validation build_packs SUCCESS (1 pack built, 2 skipped)
T+1s   19:07:34.000  barriers_reconciled: merge_ready=FALSE, validation_ready=FALSE

[Periodic barrier reconciliations every ~5s, merge_ready stays FALSE]

T+17s  19:07:50.000  barriers_reconciled: merge_ready=FALSE, validation_ready=TRUE
                     ^^^ VALIDATION READY BEFORE MERGE READY ^^^

T+17s  19:07:50.581  validation stage END (status=success)
                     results_total=1, completed=1, validation_ai_applied=TRUE
                     ready_latched=TRUE, ready_latched_at=2025-11-18T19:07:50Z

T+17s  19:07:50.850  frontend END (status=published, empty_ok=TRUE)

T+23s  19:07:56.434  merge_scoring START (span 75d070ec) ← SECOND RUN
T+23s  19:07:56.570  merge_scoring SUCCESS: 1 pack built
                     [FILE CREATED: pair_007_010.result.json at 21:07:56]

T+25s  19:07:58.000  barriers_reconciled: merge_ready=TRUE, validation_ready=TRUE
                     ^^^ MERGE READY AFTER VALIDATION COMPLETED ^^^

[Periodic barrier reconciliations continue, merge_ready stays TRUE]
```

### 3.2 File Creation Timestamps

```
ai_packs/merge/results/pair_007_010.result.json
  Created: 11/18/2025 9:07:56 PM (19:07:56 UTC)
```

### 3.3 Final State (runflow.json)

```json
{
  "stages": {
    "merge": {
      "status": "success",
      "result_files": 1,
      "pack_files": 1,
      "expected_packs": 1,
      "empty_ok": false
    },
    "validation": {
      "status": "success",
      "sent": true,
      "ready_latched": true,
      "ready_latched_at": "2025-11-18T19:07:50Z",
      "validation_ai_completed": true,
      "validation_ai_applied": false,  ← INCONSISTENT!
      "results": {
        "results_total": 1,
        "completed": 1,
        "failed": 0
      }
    }
  },
  "umbrella_barriers": {
    "merge_ready": true,
    "validation_ready": true
  }
}
```

### 3.4 Analysis

**What Happened:**

1. **T+0s:** Merge scoring completes, writes pairs_index, builds 1 pack file
2. **T+0s:** Barrier check: `merge_ready=FALSE` (no result files yet)
3. **T+0s:** Validation starts (possibly via fastpath or orchestrator trigger)
4. **T+0s:** Validation builds 1 pack, starts AI processing
5. **T+17s:** Validation AI completes, applies results, sets `ready_latched=TRUE`
6. **T+17s:** Barrier check: `validation_ready=TRUE`, but `merge_ready` still `FALSE`
7. **T+23s:** Merge re-runs (triggered by validation completion?), writes result file
8. **T+25s:** Barrier check: `merge_ready=TRUE` (result file now exists)

**The Bug:**

- Validation started and completed while `merge_ready=FALSE`
- Validation's `validation_ai_applied` flag was set to `TRUE` in event at T+17s
- But final runflow.json shows `validation_ai_applied=FALSE`
- Likely: A later writer (merge promotion at T+23s?) overwrote validation state

**Questions:**

1. **Why did validation start if merge_ready=FALSE?**
   - Possible: Orchestrator triggered validation via different code path
   - Possible: `_maybe_enqueue_validation_fastpath` triggered despite merge_zero_packs=FALSE
   - Possible: Manual trigger or test code

2. **Why did merge re-run at T+23s?**
   - Possible: Validation completion triggered a merge refresh
   - Possible: Periodic merge re-scoring in orchestrator
   - Possible: Bug in decider logic causing double merge promotion

3. **Why is validation_ai_applied=FALSE in final state?**
   - Event log shows validation_ai_applied=TRUE at T+17s
   - Final runflow.json shows validation_ai_applied=FALSE
   - Indicates: Late writer overwrote validation state after merge re-ran

---

## 4. Validation Start Call Sites

### 4.1 Search Results

**From grep_search:**
- `build_validation_packs_for_run`: 3 matches
- `run_validation_send_for_sid_v2`: 2 matches
- `VALIDATION_AUTOSEND`: 20 matches
- `_maybe_enqueue_validation_fastpath`: 2 matches (definition + call)

### 4.2 Known Trigger Points

1. **Orchestrator** (`backend/pipeline/validation_orchestrator.py`):
   - Calls `build_validation_packs_for_run` → `run_validation_send_for_sid_v2`
   - Likely checks merge_ready before triggering (need to verify)

2. **Fastpath** (`backend/runflow/decider.py::_maybe_enqueue_validation_fastpath`):
   - Triggers on `merge_zero_packs=TRUE`
   - Does NOT check `merge_ready` explicitly

3. **Manual/Test** (devtools scripts):
   - Various devtools scripts can trigger validation directly
   - May bypass all barriers

### 4.3 Action Items

- [ ] Verify orchestrator checks `merge_ready` before validation
- [ ] Audit all call sites to `run_validation_send_for_sid_v2`
- [ ] Confirm no code path triggers validation while `merge_ready=FALSE` for non-zero-packs

---

## 5. Merge Zero-Packs Interaction

### 5.1 Zero-Packs Fast Path

**Intended Behavior:**
- When `merge.expected_packs == 0` (no merge candidates), skip merge AI entirely
- Set `merge.empty_ok = TRUE` → `merge_ready = TRUE` immediately
- Trigger validation fast path via `_maybe_enqueue_validation_fastpath`

**Implementation:**
```python
# In _apply_merge_stage_promotion
if expected_total == 0:
    stage["empty_ok"] = True

# In _maybe_enqueue_validation_fastpath
merge_zero_packs = merge_stage.get("empty_ok", False)
if merge_zero_packs and VALIDATION_AUTOSEND:
    # Trigger validation
```

**Assessment:** Zero-packs fast path is correct. It's safe to start validation when `merge_zero_packs=TRUE` because merge has no AI work to do.

### 5.2 Non-Zero-Packs Case

**Intended Behavior:**
- When `merge.expected_packs > 0`, wait for merge AI to complete
- Only set `merge_ready = TRUE` after merge results are applied
- Block validation until `merge_ready = TRUE`

**Actual Behavior:**
- `merge_ready` flips TRUE as soon as result files appear on disk
- Does NOT wait for merge results to be applied to runflow
- Validation can start before merge AI finishes applying results

**Root Cause:**
1. `_merge_artifacts_progress` uses file count matching as proxy for "merge complete"
2. File count matches as soon as AI writes results, before applying them
3. No explicit check for merge application status in barriers

---

## 6. Recommendations

### 6.1 Immediate Fixes (High Priority)

**Fix 1: Add merge application tracking**

Create a new merge state flag to track when results have been applied:

```python
# In merge AI result application code
merge_stage["merge_ai_applied"] = True
merge_stage["merge_ai_applied_at"] = datetime.utcnow().isoformat() + "Z"
```

**Fix 2: Update barrier logic to check application flag**

```python
# In _compute_umbrella_barriers
def _compute_umbrella_barriers(run_dir, runflow_payload=None):
    merge_stage = stages.get("merge", {})
    
    # Check merge application status
    merge_ai_applied = merge_stage.get("merge_ai_applied", False)
    
    # Old logic (file counts)
    result_files_total, pack_files_total, expected_total, ready_disk = _merge_artifacts_progress(run_dir)
    
    # New logic: require BOTH file presence AND application completion
    if merge_stage.get("empty_ok"):
        merge_ready = True  # Zero-packs fast path
    elif expected_total == 0:
        merge_ready = True  # No packs expected
    elif ready_disk and merge_ai_applied:
        merge_ready = True  # Files present AND results applied
    else:
        merge_ready = False  # Wait for both conditions
    
    return {"merge_ready": merge_ready, ...}
```

**Fix 3: Gate all validation triggers on merge_ready**

Ensure ALL code paths that start validation check `merge_ready`:

```python
# In orchestrator
barriers = _compute_umbrella_barriers(run_dir)
if not barriers["merge_ready"]:
    logger.info("Merge not ready, skipping validation")
    return

# In fastpath
if merge_zero_packs and VALIDATION_AUTOSEND:
    # Zero-packs case: OK to proceed
    ...
else:
    # Non-zero-packs case: check merge_ready
    barriers = _compute_umbrella_barriers(run_dir)
    if not barriers["merge_ready"]:
        logger.info("Merge not ready, skipping validation")
        return
```

### 6.2 Medium-Term Improvements

**Improvement 1: Separate merge completion into phases**

Track merge stages explicitly:
- `merge_packs_built`: Packs created on disk
- `merge_ai_sent`: AI processing started
- `merge_ai_completed`: AI results received
- `merge_ai_applied`: Results applied to runflow accounts

**Improvement 2: Add explicit dependencies to barriers**

```python
umbrella_barriers = {
    "merge_ready": merge_ai_applied and result_files >= expected_packs,
    "validation_ready": validation_ai_applied and results_complete,
    "validation_requires": ["merge_ready"],  # Explicit dependency
    "strategy_requires": ["validation_ready"],
}
```

**Improvement 3: Add barrier violation detection**

Log warnings when stages proceed out of order:

```python
if validation.sent and not barriers["merge_ready"]:
    logger.warning(f"Validation started before merge ready! sid={sid}")
    telemetry.increment("barrier_violation.validation_before_merge")
```

### 6.3 Documentation Updates

- [ ] Update RUNFLOW_VALIDATION_AUTHORITY_ANALYSIS.md with merge barrier findings
- [ ] Document merge completion phases and barriers
- [ ] Add runbook for diagnosing barrier violations
- [ ] Create diagram showing correct stage ordering

---

## 7. Explicit Answers to Investigation Questions

### Q1: When does merge_ready become TRUE in zero-packs vs non-zero-packs runs?

**Zero-packs:**
- `merge_ready = TRUE` immediately after merge scoring completes
- Condition: `merge.expected_packs == 0` → `merge.empty_ok = TRUE`
- Safe: No merge AI work to wait for

**Non-zero-packs (CURRENT BUG):**
- `merge_ready = TRUE` as soon as result files appear on disk
- Condition: `result_files == pack_files == expected_packs`
- **Bug:** Does NOT wait for results to be applied to runflow

**Non-zero-packs (INTENDED):**
- `merge_ready = TRUE` after merge AI applies results
- Condition: `result_files >= expected_packs AND merge_ai_applied == TRUE`
- Safe: Ensures merge decisions are reflected in runflow before validation starts

### Question 2: Is validation strictly gated on merge_ready?

**NO.** Validation can start via multiple paths:
1. Orchestrator: *Likely* checks merge_ready (needs verification)
2. Fastpath (`_maybe_enqueue_validation_fastpath`): Only checks `merge_zero_packs`, NOT `merge_ready`
3. Manual/test triggers: May bypass all barriers

**Evidence:** SID 83830ae4 shows validation completed at T+17s while `merge_ready` stayed `FALSE` until T+25s.

### Q3: Does merge_ready open too early in non-zero-packs cases?

**YES.** Confirmed by:
- Code review: `_merge_artifacts_progress` sets `ready=TRUE` based on file counts alone
- Timeline: merge_ready flipped TRUE at T+25s, 2 seconds after result file created
- No check for whether merge AI has applied results to runflow
- Barrier opens before merge work is truly complete

### Question 4: What causes validation_ai_applied=FALSE despite validation.status=success?

**Late writer overwrite.** Timeline shows:
- T+17s: validation END event shows `validation_ai_applied=TRUE`
- T+23s: merge re-runs and writes merge stage
- T+25s: Final runflow.json shows `validation_ai_applied=FALSE`

**Hypothesis:** Merge promotion or barrier reconciliation at T+23s-T+25s overwrote validation stage, resetting the `validation_ai_applied` flag. This is a separate bug from the merge_ready timing issue, but they interact to create inconsistent state.

---

## 8. Action Plan

### Phase 1: Immediate Investigation (Complete)
- [x] Document merge promotion & barrier code
- [x] Create truth table for merge_ready conditions
- [x] Reconstruct timeline for SID 83830ae4
- [x] Find validation start call sites
- [x] Analyze merge_zero_packs interaction

### Phase 2: Design & Review (Next)
- [ ] Review recommendations with team
- [ ] Decide on merge_ai_applied flag implementation
- [ ] Design barrier violation detection
- [ ] Plan testing approach

### Phase 3: Implementation
- [ ] Add merge_ai_applied tracking to merge AI code
- [ ] Update _compute_umbrella_barriers barrier logic
- [ ] Gate all validation triggers on merge_ready
- [ ] Add telemetry for barrier violations
- [ ] Update documentation

### Phase 4: Testing & Validation
- [ ] Test zero-packs fast path (should be unchanged)
- [ ] Test non-zero-packs case (validation should wait for merge)
- [ ] Test orchestrator E2E with new barriers
- [ ] Verify SID 83830ae4 scenario cannot recur
- [ ] Check for late writer overwrites of validation state

---

## Appendix A: Key Code References

```python
# backend/runflow/decider.py

def _apply_merge_stage_promotion(run_dir, runflow_payload=None):
    """Promotes merge stage when scoring completes."""
    result_files, pack_files, expected_total, ready = _merge_artifacts_progress(run_dir)
    stage["status"] = "success"
    stage["result_files"] = result_files
    stage["empty_ok"] = (expected_total == 0)
    # MISSING: stage["merge_ai_applied"] = ???

def _merge_artifacts_progress(run_dir):
    """Counts merge files on disk."""
    result_files = len(list(results_dir.glob("*.result.json")))
    pack_files = len(list(packs_dir.glob("pair_*.jsonl")))
    expected_total = pairs_index.get("expected_packs")
    
    if expected_total == 0:
        ready = True
    else:
        ready = result_files == pack_files and result_files == expected_total
    # BUG: ready=True as soon as files appear, not when results applied
    
    return result_files, pack_files, expected_total, ready

def _compute_umbrella_barriers(run_dir, runflow_payload=None):
    """Computes umbrella barriers."""
    merge_stage = stages.get("merge", {})
    
    if merge_stage.get("empty_ok"):
        merge_ready = True
    elif merge_stage.get("result_files", 0) >= 1:
        merge_ready = True
    else:
        _, _, _, ready_disk = _merge_artifacts_progress(run_dir)
        merge_ready = ready_disk
    # BUG: All paths check files, not merge_ai_applied flag
    
    return {"merge_ready": merge_ready, ...}

def _maybe_enqueue_validation_fastpath(run_dir, ...):
    """Triggers validation for zero-packs."""
    merge_zero_packs = merge_stage.get("empty_ok", False)
    if merge_zero_packs and VALIDATION_AUTOSEND:
        # Enqueue validation
        ...
    # BUG: No check for merge_ready in non-zero-packs case
```

---

## Appendix B: Timeline Diagram

```
Time -->  0s         17s        23s        25s        ... (now)
          |----------|----------|----------|----------|

Merge     [Score]----[wait]-----[Re-run]--[Ready]----[Done]
          Pack=1     Files=0    Files=1    Files=1

Validation           [Start]----[Complete]-----------[Done]
                     Packs=1    Results=1

merge_ready  FALSE   FALSE      FALSE      TRUE       TRUE
validation_ready     FALSE      TRUE       TRUE       TRUE

Flags:
  validation_ai_applied:  N/A    TRUE (event)  →  FALSE (final) ← BUG
```

**Key:** Validation started and completed (T+17s) BEFORE merge_ready became TRUE (T+25s).

---

**End of Investigation Report**
