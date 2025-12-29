# Merge Finalization Fix - Quick Start Guide

## 🎯 What Was Fixed

**Bug:** Validation could start BEFORE merge AI results were applied to runflow (8 second race in SID 83830ae4).

**Solution:** New `merge_ai_applied` flag ensures validation waits for merge to fully complete (non-zero-packs only).

**Zero-packs:** Unaffected - fast path preserved.

---

## ⚡ Quick Verification (5 minutes)

### 1. Run Tests
```powershell
pytest backend/tests/test_merge_ai_applied_fix.py -v
```
**Expected:** 6/6 tests pass

### 2. Dry Run Backfill
```powershell
python scripts/repair_merge_ai_applied_from_run.py --runs-root runs --dry-run
```
**Expected:** Reports count of runs needing repair (bug SID 83830ae4 should be in list)

### 3. Apply Backfill
```powershell
python scripts/repair_merge_ai_applied_from_run.py --runs-root runs
```
**Expected:** Logs show `MERGE_AI_APPLIED_REPAIRED sid=...` for each fixed run

### 4. Prove Fix
```powershell
python scripts/prove_merge_fix.py --runs-root runs
```
**Expected:** All 3 test SIDs pass verification

---

## 📋 Verification Commands

```powershell
# Test specific bug SID
python scripts/prove_merge_fix.py --runs-root runs --sid 83830ae4-6406-4a7e-ad80-a3f721a3787b

# Check results
cat runs/merge_fix_verification_results.json | ConvertFrom-Json | ConvertTo-Json -Depth 10
```

---

## 🔍 What Changed

### Core Fix (3 files):
1. **decider.py::finalize_merge_stage** - Sets `merge_ai_applied=True` after results applied
2. **decider.py::_compute_umbrella_barriers** - Checks flag for non-zero-packs before `merge_ready=True`
3. **validation_orchestrator.py::run_for_sid** - Gates validation on `merge_ready` barrier

### Defensive (2 files):
4. **decider.py::_maybe_enqueue_validation_fastpath** - Skips fastpath if flag missing
5. **run_validation_orchestrator.py** - Warns if merge not ready

### Migration (1 file):
6. **repair_merge_ai_applied_from_run.py** - Backfills flag for existing runs

### Testing (2 files):
7. **test_merge_ai_applied_fix.py** - 6 unit/integration tests
8. **prove_merge_fix.py** - Real SID verification script

---

## 🛡️ Safety Checks

### Zero-packs Fast Path Preserved?
```powershell
# Test zero-packs SID
python scripts/prove_merge_fix.py --runs-root runs --sid 61e8cb38-8a58-42e3-9477-58485d43cb52
```
**Expected:** `empty_ok=True`, `merge_ready` stays `True`, no repair needed

### Bug SID Fixed?
```powershell
# Test bug SID
python scripts/prove_merge_fix.py --runs-root runs --sid 83830ae4-6406-4a7e-ad80-a3f721a3787b
```
**Expected:** `merge_ai_applied=True`, `merge_ready=True`, checks pass

### Healthy SID Unaffected?
```powershell
# Test healthy SID
python scripts/prove_merge_fix.py --runs-root runs --sid 9d4c385b-4688-46f1-b545-56ebb5ffff06
```
**Expected:** `merge_ai_applied=True`, `merge_ready=True`, checks pass

---

## 📊 Key Log Messages to Monitor

### Success:
```
MERGE_AI_APPLIED sid=<sid>
MERGE_STAGE_PROMOTED sid=<sid> result_files=<n>
```

### Gating (Expected for incomplete merges):
```
MERGE_NOT_AI_APPLIED sid=<sid> merge_ready_disk=True merge_empty_ok=False
VALIDATION_ORCHESTRATOR_DEFERRED sid=<sid> reason=merge_not_ready
```

### Zero-packs (Fast path working):
```
UMBRELLA_MERGE_OPTIONAL sid=<sid> reason=empty_merge_results
```

---

## 🚨 Troubleshooting

### Tests fail?
```powershell
# Get detailed error output
pytest backend/tests/test_merge_ai_applied_fix.py -v -s
```

### Backfill not working?
```powershell
# Check specific SID
python scripts/repair_merge_ai_applied_from_run.py --runs-root runs --sid 83830ae4-6406-4a7e-ad80-a3f721a3787b
```

### Proof script shows failures?
```powershell
# Check detailed results
cat runs/merge_fix_verification_results.json | ConvertFrom-Json | Select-Object -ExpandProperty results
```

---

## 📚 Full Documentation

- Complete details: `MERGE_FINALIZATION_FIX_SUMMARY.md`
- Investigation: `MERGE_BARRIER_INVESTIGATION.md`
- Bug timeline: Investigation doc lines 198-281

---

**Status:** ✅ Implementation complete - ready for verification  
**Time to verify:** ~5 minutes  
**Risk level:** Low (zero-packs unaffected, defensive checks in place)
