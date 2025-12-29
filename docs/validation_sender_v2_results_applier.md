  # Validation Sender V2 + Results Applier Integration

## Overview

This document describes the Phase 2 orchestrator implementation for validation AI:
1. **Validation Sender V2**: Clean sender inspired by note_style pattern
2. **Results Applier**: Merges AI results into account summary.json files
3. **Runflow Integration**: Only marks validation.status=success after summaries are updated

## Architecture

### Flow Diagram

```
Validation Packs Built
         ↓
VALIDATION_V2_AUTOSEND_TRIGGER (validation_builder.py)
         ↓
run_validation_send_for_sid_v2() (validation_sender_v2.py)
         ↓
    Send packs to AI
         ↓
    Write results to disk (ai_packs/validation/results/)
         ↓
    Update validation index
         ↓
    Refresh runflow stage
         ↓
apply_validation_results_for_sid() (apply_results_v2.py)
         ↓
    Match results to requirement blocks in summary.json
         ↓
    Enrich requirements with AI fields:
      - ai_validation_decision
      - ai_validation_rationale
      - ai_validation_citations
      - ai_validation_checks
      - ai_validation_completed_at
         ↓
    Save updated summary.json files
         ↓
    Set manifest.ai.status.validation.results_applied = true
         ↓
Runflow promotion: validation.status = "success"
```

## Implementation Details

### 1. Validation Sender V2 (`backend/ai/validation_sender_v2.py`)

**Key Features:**
- Clean design inspired by note_style_sender.py
- Sequential per-pack sending (simple, debuggable)
- Disk-first manifest API (safe for orchestrator mode)
- Clear logging with VALIDATION_V2_* markers
- Idempotent: skips packs with existing results

**Main Function:**
```python
def run_validation_send_for_sid_v2(sid: str, runs_root: Path) -> dict[str, Any]:
    """
    Send validation packs to AI, write results, update index, apply to summaries.
    
    Returns summary with:
    - expected: number of packs expected
    - sent: number of packs sent
    - written: number of result files written
    - failed: number of failures
    - apply_stats: stats from results applier
    - apply_success: whether results were successfully applied to summaries
    """
```

**Integration Points:**
- Called automatically after packs built (validation_builder.py)
- Can be called manually for retry/testing
- Gated by VALIDATION_ORCHESTRATOR_MODE=1 and VALIDATION_AUTOSEND_ENABLED=1

### 2. Results Applier (`backend/validation/apply_results_v2.py`)

**Key Features:**
- Matches AI results to requirement blocks by reason_code + send_to_ai flag
- Enriches requirements with AI fields (decision, rationale, citations, checks)
- Idempotent: overwrites existing AI fields
- Defensive: logs warnings for unmatched results
- Clear logging with VALIDATION_V2_APPLY_* markers

**Main Function:**
```python
def apply_validation_results_for_sid(sid: str, runs_root: Path) -> dict[str, Any]:
    """
    Apply validation AI results to account summary.json files.
    
    Returns summary with:
    - accounts_total: number of unique accounts with results
    - accounts_updated: number of accounts where summaries were updated
    - results_total: number of result records processed
    - results_applied: number of results successfully merged
    - results_unmatched: number of results that didn't match any requirement
    """
```

**Matching Logic:**
For each AI result:
1. Load account's summary.json
2. Find validation_requirements.findings array
3. Match by:
   - `send_to_ai == true`
   - `reason_code == result.checks.mismatch_code`
   - (optionally) `field == result.field`
4. Enrich matched requirement with AI fields
5. Save updated summary.json

**AI Fields Added to Requirements:**
```json
{
  "field": "creditor_type",
  "reason_code": "C5_ALL_DIFF",
  "send_to_ai": true,
  "ai_validation_decision": "supportive_needs_companion",
  "ai_validation_rationale": "The discrepancies among bureau values...",
  "ai_validation_citations": [
    "equifax: miscellaneous finance",
    "experian: student loans",
    "transunion: miscellaneous education"
  ],
  "ai_validation_checks": {
    "doc_requirements_met": true,
    "materiality": true,
    "supports_consumer": true,
    "mismatch_code": "C5_ALL_DIFF"
  },
  "ai_validation_completed_at": "2025-11-17T00:48:30Z"
}
```

### 3. Runflow Integration (`backend/runflow/decider.py`)

**Key Changes:**

**a) Check results_applied flag:**
```python
# In umbrella barriers check
validation_status = manifest["ai"]["status"]["validation"]
results_applied_flag = validation_status.get("results_applied")
if results_applied_flag is True:
    validation_merge_applied = True
```

**b) Require results_applied for validation_ready:**
```python
# In _validation_stage_ready()
if orchestrator_mode and required_flag:
    results_applied = manifest["ai"]["status"]["validation"].get("results_applied")
    return (
        ready_disk and 
        total > 0 and 
        completed == total and 
        failed == 0 and 
        results_applied is True  # NEW: require summaries updated
    )
```

**Effect:**
- Validation stage only promotes to "success" after summaries are updated
- Prevents premature success marking when results exist but aren't merged
- Ensures downstream stages (strategy, note_style) only run after AI merge complete

### 4. Autotrigger Wiring (`backend/ai/validation_builder.py`)

**Integration Block:**
```python
# After packs recorded in runflow
if packs_built > 0 and orchestrator_mode and autosend_enabled:
    log.info("VALIDATION_V2_AUTOSEND_TRIGGER sid=%s packs=%d", sid, packs_built)
    stats = run_validation_send_for_sid_v2(sid, runs_root_path)
    log.info(
        "VALIDATION_V2_AUTOSEND_DONE sid=%s sent=%s written=%s apply_success=%s",
        sid,
        stats.get("sent"),
        stats.get("written"),
        stats.get("apply_success"),
    )
```

**Conditions:**
- `packs_built > 0`: has validation packs to send
- `orchestrator_mode`: VALIDATION_ORCHESTRATOR_MODE=1
- `autosend_enabled`: any of:
  - VALIDATION_AUTOSEND_ENABLED=1
  - VALIDATION_SEND_ON_BUILD=1
  - VALIDATION_STAGE_AUTORUN=1

## Configuration

**Environment Variables:**
```bash
# Phase 2 orchestrator mode (uses clean validation_sender_v2)
VALIDATION_ORCHESTRATOR_MODE=1
VALIDATION_AUTOSEND_ENABLED=1

# Legacy sender flags (kept for backward compatibility)
ENABLE_VALIDATION_SENDER=1
VALIDATION_SENDER_ENABLED=1
AUTO_VALIDATION_SEND=1
VALIDATION_STAGE_AUTORUN=1
VALIDATION_SEND_ON_BUILD=1

# Model configuration
VALIDATION_MODEL=gpt-4o-mini
VALIDATION_REQUEST_TIMEOUT=45
```

## Testing

### Manual Test Scripts

**1. Test Applier Standalone:**
```bash
python manual_apply_validation_results.py <sid>
```

**2. Test Full Integration:**
```bash
python test_validation_v2_apply.py <sid>
```

**Test Output:**
```
✅ Manifest shows results_applied=true
✅ All 3 account summaries have AI fields
✅ Runflow validation status is 'success'
✅ No missing results
🎉 ALL CHECKS PASSED!
```

### Verification Steps

**Step 1: Check validation packs exist**
```bash
ls runs/<sid>/ai_packs/validation/packs/val_acc_*.jsonl
```

**Step 2: Check validation results exist**
```bash
ls runs/<sid>/ai_packs/validation/results/acc_*.result.jsonl
```

**Step 3: Check AI fields in summary.json**
```bash
python -c "
import json
summary = json.load(open('runs/<sid>/cases/accounts/9/summary.json'))
finding = summary['validation_requirements']['findings'][0]
print(f\"Decision: {finding.get('ai_validation_decision')}\")
print(f\"Rationale: {finding.get('ai_validation_rationale')[:80]}...\")
"
```

**Step 4: Check manifest results_applied flag**
```bash
python -c "
import json
manifest = json.load(open('runs/<sid>/manifest.json'))
validation = manifest['ai']['status']['validation']
print(f\"results_applied: {validation.get('results_applied')}\")
print(f\"state: {validation.get('state')}\")
"
```

**Step 5: Check runflow validation stage**
```bash
python -c "
import json
runflow = json.load(open('runs/<sid>/runflow.json'))
validation = runflow['stages']['validation']
print(f\"status: {validation.get('status')}\")
print(f\"missing_results: {validation['metrics'].get('missing_results')}\")
"
```

## Logging Markers

**Sender V2:**
- `VALIDATION_V2_AUTOSEND_TRIGGER`: Autosend triggered after packs built
- `VALIDATION_V2_SEND_START`: Starting send for SID
- `VALIDATION_V2_SEND_DONE`: Completed send for account
- `VALIDATION_V2_SUMMARY`: Send summary (expected, sent, written, failed)
- `VALIDATION_V2_INDEX_UPDATED`: Index updated with result paths
- `VALIDATION_V2_RUNFLOW_REFRESHED`: Runflow stage refreshed
- `VALIDATION_V2_MANIFEST_UPDATED`: Manifest updated with completion status

**Applier:**
- `VALIDATION_V2_APPLY_START`: Starting apply for SID
- `VALIDATION_V2_APPLY_RESULTS_LOADED`: Loaded N result records from disk
- `VALIDATION_V2_APPLY_MATCH`: Matched result to requirement (account, field, reason_code)
- `VALIDATION_V2_APPLY_ACCOUNT_UPDATED`: Updated summary for account
- `VALIDATION_V2_APPLY_SUMMARY`: Apply summary (accounts_total, accounts_updated, results_applied, results_unmatched)
- `VALIDATION_V2_APPLY_NO_MATCH`: Warning when result doesn't match any requirement
- `VALIDATION_V2_APPLY_DONE`: Completed apply successfully

## Success Criteria

✅ **Packs Built**: Validation packs created under ai_packs/validation/packs/
✅ **Autosend Triggered**: VALIDATION_V2_AUTOSEND_TRIGGER log after packs built
✅ **Results Written**: Result files created under ai_packs/validation/results/
✅ **Index Updated**: Index records have result_jsonl paths and status="sent"
✅ **Runflow Refreshed**: Runflow shows results_total, missing_results=0
✅ **Results Applied**: AI fields merged into summary.json requirement blocks
✅ **Manifest Updated**: manifest.ai.status.validation.results_applied=true
✅ **Runflow Success**: validation.status="success" only after summaries updated
✅ **No Missing Results Error**: validation_results_missing error eliminated

## File Changes Summary

**New Files:**
- `backend/validation/apply_results_v2.py` (368 lines): Results applier module
- `test_validation_v2_apply.py` (197 lines): End-to-end test script
- `manual_apply_validation_results.py` (29 lines): Manual applier test script

**Modified Files:**
- `backend/ai/validation_sender_v2.py`:
  - Added apply_validation_results_for_sid() call after results written
  - Added results_applied flag to manifest
  - Updated completion logic to require apply_success
  
- `backend/runflow/decider.py`:
  - Updated _validation_merge_applied to check results_applied flag
  - Updated _validation_stage_ready to require results_applied in orchestrator mode
  
- `backend/ai/validation_builder.py`:
  - Autosend integration already present (completed previously)

**No Changes Needed:**
- `.env`: Orchestrator flags already set

## Design Highlights

**1. Idempotency:**
- Sender skips packs with existing results
- Applier overwrites AI fields if re-run
- Safe to retry entire flow without side effects

**2. Disk-First API:**
- Uses load_manifest_from_disk / save_manifest_to_disk
- Prevents races in orchestrator mode
- Atomic manifest updates with mutator functions

**3. Sequential Sending:**
- Simple per-pack loop (no concurrency complexity)
- Easy to debug with clear logging
- Sufficient for validation pack volumes

**4. Defensive Matching:**
- Matches results to requirements by multiple criteria
- Logs warnings for unmatched results
- Continues processing on partial failures

**5. Clear Separation:**
- Sender focuses on AI communication and result writing
- Applier focuses on summary.json enrichment
- Runflow focuses on promotion and barriers

## Future Enhancements

**Optional:**
- Add retry logic for transient AI errors
- Implement batching/concurrency for faster sending
- Add metrics emission for observability dashboards
- Consider automatic merge of results into summaries in orchestrator mode

**Not Required:**
- Current implementation meets all user requirements
- System is production-ready as-is

## Troubleshooting

**Problem: Results not applied to summaries**
```bash
# Check if applier was called
grep "VALIDATION_V2_APPLY_START" runs/<sid>/ai_packs/validation/logs.txt

# Manually run applier
python manual_apply_validation_results.py <sid>

# Check for matching issues
grep "VALIDATION_V2_APPLY_NO_MATCH" runs/<sid>/ai_packs/validation/logs.txt
```

**Problem: Runflow not showing success**
```bash
# Check results_applied flag
python -c "import json; m = json.load(open('runs/<sid>/manifest.json')); print(m['ai']['status']['validation'].get('results_applied'))"

# If false, manually run applier and check again
python manual_apply_validation_results.py <sid>
```

**Problem: Autosend not triggering**
```bash
# Check orchestrator mode flag
grep "VALIDATION_ORCHESTRATOR_MODE" .env

# Check autosend flag
grep "VALIDATION_AUTOSEND_ENABLED" .env

# Check logs for trigger
grep "VALIDATION_V2_AUTOSEND" runs/<sid>/ai_packs/validation/logs.txt
```

## Conclusion

This implementation provides a complete, clean, reliable validation AI sender with automatic results merge into account summaries. The runflow promotion logic ensures validation only shows "success" after AI results are fully integrated into the case data, preventing downstream stages from running prematurely.

All user requirements have been met:
- Clean sender inspired by note_style pattern ✅
- Automatic triggering after packs built ✅
- Results written to correct disk location ✅
- Results merged into summary.json requirement blocks ✅
- Runflow only shows success after merge complete ✅
- Clear logging throughout ✅
- Tested end-to-end successfully ✅
