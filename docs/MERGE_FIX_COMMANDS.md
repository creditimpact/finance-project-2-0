# Merge Finalization Fix - Command Reference

## 🚀 Complete Verification Workflow

### Step 1: Check Code Quality
```powershell
# No syntax errors
pytest --collect-only backend/tests/test_merge_ai_applied_fix.py
```

### Step 2: Run Unit Tests
```powershell
# All tests should pass
pytest backend/tests/test_merge_ai_applied_fix.py -v
```

### Step 3: Dry Run Backfill
```powershell
# See what would be repaired
python scripts/repair_merge_ai_applied_from_run.py --runs-root runs --dry-run
```

### Step 4: Apply Backfill
```powershell
# Fix existing runs
python scripts/repair_merge_ai_applied_from_run.py --runs-root runs
```

### Step 5: Run Proof Script
```powershell
# Verify fix on 3 test SIDs
python scripts/prove_merge_fix.py --runs-root runs
```

### Step 6: Review Results
```powershell
# Check detailed results
cat runs/merge_fix_verification_results.json | ConvertFrom-Json | ConvertTo-Json -Depth 10
```

---

## 🔍 Test Individual SIDs

### Bug Case (validation before merge)
```powershell
# SID: 83830ae4-6406-4a7e-ad80-a3f721a3787b
python scripts/prove_merge_fix.py --runs-root runs --sid 83830ae4-6406-4a7e-ad80-a3f721a3787b
```

### Zero-Packs Case (should be unaffected)
```powershell
# SID: 61e8cb38-8a58-42e3-9477-58485d43cb52
python scripts/prove_merge_fix.py --runs-root runs --sid 61e8cb38-8a58-42e3-9477-58485d43cb52
```

### Healthy Case (with merge packs)
```powershell
# SID: 9d4c385b-4688-46f1-b545-56ebb5ffff06
python scripts/prove_merge_fix.py --runs-root runs --sid 9d4c385b-4688-46f1-b545-56ebb5ffff06
```

---

## 🛠️ Manual Inspection Commands

### Check Merge Stage
```powershell
$sid = "83830ae4-6406-4a7e-ad80-a3f721a3787b"
$runflow = Get-Content "runs/$sid/runflow.json" | ConvertFrom-Json
$runflow.stages.merge | ConvertTo-Json -Depth 5
```

### Check Barriers
```powershell
python -c "from backend.runflow.decider import _compute_umbrella_barriers; from pathlib import Path; import json; print(json.dumps(_compute_umbrella_barriers(Path('runs/$sid')), indent=2))"
```

### Check Validation Stage
```powershell
$runflow.stages.validation | ConvertTo-Json -Depth 5
```

---

## 📊 Expected Output Patterns

### Before Backfill (Bug SID):
```json
{
  "merge": {
    "status": "success",
    "empty_ok": false,
    "result_files": 10,
    "merge_ai_applied": false  // ❌ MISSING
  }
}
```

### After Backfill (Bug SID):
```json
{
  "merge": {
    "status": "success",
    "empty_ok": false,
    "result_files": 10,
    "merge_ai_applied": true,  // ✅ FIXED
    "merge_ai_applied_at": "2025-01-15T10:30:00Z"
  }
}
```

### Zero-Packs (Unchanged):
```json
{
  "merge": {
    "status": "success",
    "empty_ok": true,  // ✅ Fast path preserved
    "result_files": 0
    // No merge_ai_applied needed
  }
}
```

---

## 🔬 Debugging Commands

### Check Finalize Merge Stage Logs
```powershell
Select-String -Path "logs/backend.log" -Pattern "MERGE_AI_APPLIED|MERGE_NOT_AI_APPLIED" | Select-Object -Last 20
```

### Check Validation Orchestrator Logs
```powershell
Select-String -Path "logs/backend.log" -Pattern "VALIDATION_ORCHESTRATOR_DEFERRED|MERGE_NOT_AI_APPLIED" | Select-Object -Last 20
```

### Run Validation Orchestrator with Warning
```powershell
python devtools/run_validation_orchestrator.py 83830ae4-6406-4a7e-ad80-a3f721a3787b runs
```

---

## 📈 Monitoring Production

### After Deployment - Check First Run
```powershell
# 1. Watch for merge completion
Select-String -Path "logs/backend.log" -Pattern "MERGE_AI_APPLIED sid=" -Wait

# 2. Check validation doesn't start early
Select-String -Path "logs/backend.log" -Pattern "VALIDATION_ORCHESTRATOR_DEFERRED.*merge_not_ready" -Wait

# 3. Verify barrier becomes ready
Select-String -Path "logs/backend.log" -Pattern "merge_ready.*True" -Wait
```

### Health Check Query
```powershell
# Count how many runs need backfill
python scripts/repair_merge_ai_applied_from_run.py --runs-root runs --dry-run | Select-String "repaired="
```

---

## 🎯 Success Criteria

### Tests
- ✅ All 6 unit tests pass
- ✅ No syntax errors in modified files

### Backfill
- ✅ Dry run reports expected count
- ✅ Actual run reports same count as repaired
- ✅ Logs show `MERGE_AI_APPLIED_REPAIRED` for each SID

### Proof
- ✅ Bug SID passes all checks
- ✅ Zero-packs SID shows unaffected behavior
- ✅ Healthy SID passes all checks
- ✅ Verification results file shows 3/3 passed

### Production
- ✅ New runs set `merge_ai_applied=True` automatically
- ✅ Validation defers when `merge_ready=False`
- ✅ Zero-packs remain instant
- ✅ No `MERGE_NOT_AI_APPLIED` logs for zero-packs

---

## 📚 Files to Review

### Implementation
- `backend/runflow/decider.py` - Core fix (3 changes)
- `backend/pipeline/validation_orchestrator.py` - Gating logic
- `devtools/run_validation_orchestrator.py` - Warning

### Scripts
- `scripts/repair_merge_ai_applied_from_run.py` - Backfill script
- `scripts/prove_merge_fix.py` - Verification script

### Tests
- `backend/tests/test_merge_ai_applied_fix.py` - Test suite

### Documentation
- `MERGE_FINALIZATION_FIX_SUMMARY.md` - Complete details
- `MERGE_FIX_QUICKSTART.md` - Quick reference (this file)
- `MERGE_BARRIER_INVESTIGATION.md` - Root cause analysis

---

**Ready to deploy?** Start with Step 1 above! 🚀
