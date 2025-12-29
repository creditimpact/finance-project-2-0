# Validation AI Auto-Merge Fix - Production Pipeline

## Problem Summary

**Symptom**: Validation AI results existed on disk (`acc_007.result.jsonl`) but `summary.json` never reflected AI decisions. Findings remained in their pre-AI state without `validation_ai`, `ai_decision`, `rationale`, or `citations` fields.

**Root Cause**: The validation **fastpath chains** in `backend/runflow/decider.py` (used in production) were missing the `validation_merge_ai_results_step`. They only called:
1. `validation_build_packs`
2. `validation_send`
3. `validation_compact`

The merge step existed and worked correctly in the main pipeline (`enqueue_auto_ai_chain`), but production runs triggered via the fastpath never executed it.

---

## Fix Applied

### 1. **Wired merge step into both fastpath chains** (`backend/runflow/decider.py`)

#### First fastpath chain (line ~370):
```python
workflow = chain(
    auto_ai_tasks.validation_build_packs.s(initial_payload),
    auto_ai_tasks.validation_send.s(),
    auto_ai_tasks.validation_compact.s(),
    auto_ai_tasks.validation_merge_ai_results_step.s(),  # ✅ ADDED
)
workflow.apply_async(queue="validation")
```

#### Watchdog fastpath chain (line ~640):
```python
workflow = chain(
    auto_ai_tasks.validation_build_packs.s(payload),
    auto_ai_tasks.validation_send.s(),
    auto_ai_tasks.validation_compact.s(),
    auto_ai_tasks.validation_merge_ai_results_step.s(),  # ✅ ADDED
)
workflow.apply_async(queue="validation")
```

### 2. **Enhanced diagnostic logging** (`backend/pipeline/auto_ai_tasks.py`)

Added comprehensive logging to `validation_merge_ai_results_step`:

```python
# Entry logging showing configuration
logger.info(
    "VALIDATION_AI_MERGE_STEP_ENTER sid=%s results_dir=%s poll_interval=%ds max_wait=%ds",
    sid, results_dir, poll_interval, max_wait,
)

# Skip logging when results incomplete
logger.info(
    "VALIDATION_AI_MERGE_SKIPPED sid=%s reason=%s results_total=%d missing_results=%d",
    sid, reason, progress.completed, max(progress.missing, 0),
)

# Applied logging showing accounts/fields updated
logger.info(
    "VALIDATION_AI_MERGE_APPLIED sid=%s accounts=%d fields=%d",
    sid, stats.get("accounts_updated", 0), stats.get("fields_updated", 0),
)
```

### 3. **Fixed wait loop helper** (`backend/pipeline/auto_ai_tasks.py`)

Corrected `_await_validation_results` to allow zero poll intervals (critical for tests):
```python
# Before: poll_interval <= 0 (rejected zero)
# After:  poll_interval < 0  (allows zero for testing)
```

### 4. **AI override preservation confirmed**

Verified `backend/core/logic/validation_requirements.py` already preserves AI data:
- `_collect_ai_overrides` extracts `ai_*` and `validation_ai` keys
- `_apply_ai_overrides_to_findings` restores them after rebuild

---

## Verification

### Tests Created/Updated

1. **`tests/backend/pipeline/test_validation_apply_results_to_summary.py`** (4 tests)
   - Wait/poll behavior
   - Idempotency
   - Missing results handling
   - Basic merge correctness

2. **`tests/backend/pipeline/test_validation_end_to_end.py`** (new)
   - End-to-end proof: results → summary with full AI enrichment
   - Verifies `validation_ai`, `ai_decision`, `rationale`, `citations`

### Test Results
```
tests/backend/pipeline/ -k validation: 5 passed
tests/backend/runflow/test_decider.py -k validation: 14 passed
```

---

## Expected Production Behavior

### Before Fix
```
[Validation packs built]
[Packs sent to AI]
[Results written: acc_007.result.jsonl]
[Compact runs]
❌ Merge step NEVER runs (not in fastpath chain)
→ summary.json stays in pre-AI state
```

### After Fix
```
[Validation packs built]
[Packs sent to AI]
[Results written: acc_007.result.jsonl]
[Compact runs]
✅ Merge step runs automatically
✅ summary.json updated with AI decisions
✅ manifest.json marked: validation_merge_applied=true
✅ runflow.json updated: validation_ai_completed=true
```

### Log Signatures (Production)

Look for these log lines in real runs:

```
INFO  VALIDATION_AI_MERGE_STEP_ENTER sid=f8f6a15a-... results_dir=.../results poll_interval=5s max_wait=90s
INFO  VALIDATION_AI_APPLIED sid=f8f6a15a-... accounts=['7'] accounts_updated=1 fields_total=2 fields_updated=2
INFO  VALIDATION_AI_MERGE_APPLIED sid=f8f6a15a-... accounts=1 fields=2
INFO  AUTO_AI_VALIDATION_MERGE_DONE sid=f8f6a15a-...
```

If merge skips due to missing results:
```
WARNING AUTO_AI_VALIDATION_RESULTS_INCOMPLETE sid=... total=1 completed=0 failed=0 missing=1
INFO    VALIDATION_AI_MERGE_SKIPPED sid=... reason=validation_results_missing results_total=0 missing_results=1
```

---

## Expected Summary Structure (After Merge)

For SID `f8f6a15a-a9c3-45b0-8334-b9b6aa6d100d`, account 007:

```json
{
  "validation_requirements": {
    "findings": [
      {
        "field": "account_type",
        "reason_code": "account_type_conflict",
        "reason_label": "Account Type Conflict",
        "send_to_ai": true,
        "decision": "supportive_needs_companion",
        "decision_source": "ai",
        "ai_decision": "supportive_needs_companion",
        "ai_rationale": "Account type shows Collection vs Auto Loan across bureaus",
        "ai_citations": ["equifax: Collection", "experian: Auto Loan"],
        "ai_legacy_decision": "supportive",
        "validation_ai": {
          "decision": "supportive_needs_companion",
          "rationale": "Account type shows Collection vs Auto Loan across bureaus",
          "citations": ["equifax: Collection", "experian: Auto Loan"],
          "legacy_decision": "supportive",
          "source": "validation_ai"
        }
      }
    ]
  }
}
```

---

## Files Modified

1. **`backend/runflow/decider.py`**
   - Added `validation_merge_ai_results_step.s()` to first fastpath chain (~line 373)
   - Added `validation_merge_ai_results_step.s()` to watchdog fastpath chain (~line 643)

2. **`backend/pipeline/auto_ai_tasks.py`**
   - Enhanced logging in `validation_merge_ai_results_step` (entry/skip/applied)
   - Fixed `_await_validation_results` poll interval guard (allow zero)

3. **`tests/backend/pipeline/test_validation_apply_results_to_summary.py`** (updated)
   - Added wait/idempotency/error tests

4. **`tests/backend/pipeline/test_validation_end_to_end.py`** (new)
   - End-to-end validation AI merge verification

---

## Rollout Checklist

- ✅ Code changes committed to `hotfix/stop-ghost-sids`
- ✅ All tests pass (pipeline + runflow)
- ✅ Logging confirms merge step entry/completion
- ⏳ Deploy and monitor for `VALIDATION_AI_MERGE_STEP_ENTER` logs
- ⏳ Verify fresh SIDs show AI-enriched `summary.json`
- ⏳ Check `manifest.json` has `validation_merge_applied: true`

---

## No Manual Intervention Required

The pipeline is now **fully automatic**:
- Packs build → AI called → Results written → **Merge runs automatically**
- No per-SID manual calls needed
- No CLI hacks required
- Works for all future validation runs

Example SID `f8f6a15a-a9c3-45b0-8334-b9b6aa6d100d` will auto-merge on next validation run (if re-triggered) or any new SID will merge automatically.
