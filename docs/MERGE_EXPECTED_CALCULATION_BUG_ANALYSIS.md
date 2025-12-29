# Merge Expected Calculation Bug Analysis

**SID**: `bf94cced-01d4-479a-b03b-ebf92623aa03`  
**Issue**: Merge stage completes successfully and updates account summaries, but `merge_ready` remains `false` because `merge_ai_applied` is never set.  
**Root Cause**: Expected calculation uses `len(pairs)` which counts bidirectional entries (2) instead of physical packs (1), causing RuntimeError before `merge_ai_applied` can be set.

---

## 1. Deep Dive: finalize_merge_stage Expected Calculation

### What Each Value Means

**From `backend/runflow/decider.py::finalize_merge_stage` (lines 2406-2468)**:

- **`result_files_total`** (int): Physical count of `*.result.json` files in `ai_packs/merge/results/` directory
  - For SID bf94cced: **1 file** (`pair_007_010.result.json`)
  
- **`pack_files_total`** (int): Physical count of `*.jsonl` files in `ai_packs/merge/packs/` directory
  - For SID bf94cced: **1 file** (`pair_007_010.jsonl`)
  
- **`expected_total`** (int | None): **Calculated value** from multiple candidate sources using `max()`
  - For SID bf94cced: **2** (INCORRECT - should be 1)

### How Expected is Calculated

**Lines 2430-2450 of `decider.py`**:

```python
expected_candidates: list[int] = []

# Candidate 1: totals.created_packs (from pairs_index.json)
if totals_created_packs is not None:
    expected_candidates.append(totals_created_packs)  # adds 1 ✅

# Candidate 2: totals.packs_built (from pairs_index.json)
packs_built_total = _maybe_int(totals.get("packs_built"))
if packs_built_total is not None:
    expected_candidates.append(packs_built_total)  # adds 1 ✅

# Candidate 3: totals.total_packs (from pairs_index.json)
total_packs_total = _maybe_int(totals.get("total_packs"))
if total_packs_total is not None:
    expected_candidates.append(total_packs_total)  # not present

# Candidate 4: fallback_created (from index_payload.created_packs)
if fallback_created is not None:
    expected_candidates.append(fallback_created)  # not present

# Candidate 5: pairs_count = len(pairs_payload) ❌ BUG HERE
if pairs_count is not None:
    expected_candidates.append(pairs_count)  # adds 2 ❌

# CALCULATION: Take maximum of all candidates
expected_total: Optional[int]
if expected_candidates:
    expected_total = max(expected_candidates)  # max([1, 1, 2]) = 2 ❌
else:
    expected_total = None
```

**For SID bf94cced**:
- `expected_candidates = [1, 1, 2]`
- `expected_total = max([1, 1, 2]) = 2`

### Why RuntimeError is Thrown

**Lines 2459-2468 of `decider.py`**:

```python
ready_counts_match = result_files_total == pack_files_total  # 1 == 1 → True ✅
if expected_total is not None:
    ready_counts_match = ready_counts_match and result_files_total == expected_total
    # True and (1 == 2) → False ❌

if not ready_counts_match:
    raise RuntimeError(
        "merge stage artifacts not ready: results=%s packs=%s expected=%s"
        % (result_files_total, pack_files_total, expected_total)
    )
    # THROWS: "merge stage artifacts not ready: results=1 packs=1 expected=2"
```

### Why merge_ai_applied is Never Set

**Lines 2505-2520 of `decider.py`**:

```python
# This code is NEVER REACHED because RuntimeError is thrown above
merge_stage["merge_ai_applied"] = True
merge_stage["merge_ai_applied_at"] = _now_iso()
log.info("MERGE_AI_APPLIED sid=%s", sid)
```

The `merge_ai_applied` flag can only be set AFTER the ready check passes. Since the RuntimeError is thrown, execution never reaches this line.

---

## 2. Pack→Accounts→Artifacts Relationship

### Physical Reality (What Actually Exists)

| Item | Count | File Path | Notes |
|------|-------|-----------|-------|
| **Merge pack** | 1 | `ai_packs/merge/packs/pair_007_010.jsonl` | Contains AI prompt data for accounts 7 and 10 |
| **Result file** | 1 | `ai_packs/merge/results/pair_007_010.result.json` | Contains AI decision: "duplicate" |
| **Accounts updated** | 2 | `cases/accounts/7/tags.json`<br>`cases/accounts/10/tags.json` | Both have merge decision tags |
| **Account summaries** | 2 | `cases/accounts/7/summary.json`<br>`cases/accounts/10/summary.json` | Both updated with merge info |

**Confirmed AI Decision (from tags.json)**:
```json
{
  "kind": "ai_decision",
  "with": 10,  // or 7
  "decision": "duplicate",
  "at": "2025-11-18T21:45:16Z",
  "flags": {
    "account_match": true,
    "debt_match": true
  }
}
```

✅ **Accounts were successfully updated** - merge stage DID complete its work.

### Index Representation (What pairs_index.json Says)

**From `runs/bf94cced.../ai_packs/merge/pairs_index.json`**:

```json
{
  "sid": "bf94cced-01d4-479a-b03b-ebf92623aa03",
  "totals": {
    "scored_pairs": 3,
    "packs_built": 1,      ✅ Correct: 1 pack built
    "created_packs": 1,    ✅ Correct: 1 pack created
    "skipped": 2
  },
  "pairs": [
    {"pair": [7, 10], "pack_file": "pair_007_010.jsonl", "score": 49},
    {"pair": [10, 7], "pack_file": "pair_007_010.jsonl", "score": 49}
  ],                       ❌ Problem: 2 entries for same pack
  "packs": [
    {"a": 7, "b": 10, "pack_file": "pair_007_010.jsonl", ...},
    {"a": 10, "b": 7, "pack_file": "pair_007_010.jsonl", ...}
  ],                       ❌ Problem: 2 entries for same pack
  "pairs_count": 2         ❌ Problem: counts both directions
}
```

### Why Bidirectional Representation Exists

**From `scripts/build_ai_merge_packs.py` lines 182-193**:

```python
seen_pairs: set[tuple[int, int]] = set()
pairs_payload: list[dict[str, object]] = []
for entry in index_entries:
    a_idx = int(entry["a"])
    b_idx = int(entry["b"])
    score_value = entry.get("score", entry.get("score_total", 0))
    pack_file = entry.get("pack_file")
    for pair in ((a_idx, b_idx), (b_idx, a_idx)):  # ← Intentional bidirectional
        if pair in seen_pairs:
            continue
        pair_entry: dict[str, object] = {
            "pair": [pair[0], pair[1]],
            "score": score_value,
        }
        if pack_file:
            pair_entry["pack_file"] = pack_file
        pairs_payload.append(pair_entry)
        seen_pairs.add(pair)
```

**Purpose**: The bidirectional representation allows **fast lookup** from either account:
- If you have account 7 and want to find all merge decisions, you can find `{"pair": [7, 10]}`
- If you have account 10, you can find `{"pair": [10, 7]}`
- Both point to the same physical pack file: `pair_007_010.jsonl`

**This is intentional design**, but it creates a **counting mismatch**:
- **Logical pairs**: 2 (bidirectional entries for lookup)
- **Physical packs**: 1 (actual file on disk)
- **Physical results**: 1 (actual result file on disk)

### Logic Table: Expected vs Actual

| Metric | Source | Value | Correct? |
|--------|--------|-------|----------|
| **Physical packs** | File count | 1 | ✅ Ground truth |
| **Physical results** | File count | 1 | ✅ Ground truth |
| **totals.created_packs** | pairs_index.json | 1 | ✅ Matches physical |
| **totals.packs_built** | pairs_index.json | 1 | ✅ Matches physical |
| **pairs array length** | len(pairs_index.json["pairs"]) | 2 | ⚠️ Bidirectional representation |
| **packs array length** | len(pairs_index.json["packs"]) | 2 | ⚠️ Bidirectional representation |
| **pairs_count** | pairs_index.json["pairs_count"] | 2 | ⚠️ Counts both directions |
| **expected (calculated)** | max([1, 1, 2]) | **2** | ❌ **BUG** - should be 1 |

### Business Logic Interpretation

**User's Intuition (CORRECT)**:
> "One merge pack represents a pair of accounts. It logically updates two account summaries. But there's still only ONE pack file and ONE result file."

**Current Code Behavior (INCORRECT)**:
> "expected uses pairs_count (2) because there are 2 lookup entries in the pairs array, so it expects 2 result files."

**The Mismatch**:
- **Index builder** creates 2 pairs entries for **lookup convenience** (intentional)
- **Expected calculation** uses `len(pairs)` to determine how many **result files** to expect (bug)
- Result: expected=2, but only 1 result file exists → RuntimeError

---

## 3. Link to merge_ai_applied and Barrier

### Full Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│ 1. AI Pipeline Runs                                                     │
│    - Sends merge pack to AI: pair_007_010.jsonl                         │
│    - AI returns decision: "duplicate"                                   │
│    - Result written: pair_007_010.result.json                           │
│    - Accounts updated: tags.json and summary.json for accounts 7 and 10 │
└────────────────────────────────┬────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ 2. finalize_merge_stage() Called                                        │
│    - Loads pairs_index.json                                             │
│    - Parses totals: {created_packs: 1, packs_built: 1}                 │
│    - Parses pairs: [{pair: [7,10], ...}, {pair: [10,7], ...}]          │
│    - Calculates pairs_count: len(pairs) = 2                             │
└────────────────────────────────┬────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ 3. Expected Calculation                                                 │
│    expected_candidates = []                                             │
│    expected_candidates.append(totals.created_packs)  # adds 1           │
│    expected_candidates.append(totals.packs_built)    # adds 1           │
│    expected_candidates.append(pairs_count)           # adds 2 ← BUG     │
│    expected_total = max([1, 1, 2]) = 2               # ← WRONG          │
└────────────────────────────────┬────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ 4. Ready Check (Lines 2459-2468)                                        │
│    result_files_total = 1  (physical count)                             │
│    pack_files_total = 1    (physical count)                             │
│    expected_total = 2      (calculated - WRONG)                         │
│                                                                          │
│    ready_counts_match = (1 == 1) and (1 == 2)  # True and False = False│
│                                                                          │
│    if not ready_counts_match:                                           │
│        raise RuntimeError("...expected=2")  ← EXECUTION STOPS HERE      │
└────────────────────────────────┬────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ 5. merge_ai_applied Flag (Lines 2505-2520)                              │
│    ❌ NEVER REACHED due to RuntimeError above                           │
│                                                                          │
│    merge_stage["merge_ai_applied"] = True   ← NEVER EXECUTED            │
│    merge_stage["merge_ai_applied_at"] = _now_iso()                      │
│    log.info("MERGE_AI_APPLIED sid=%s", sid)                             │
└────────────────────────────────┬────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ 6. Barrier Computation (_compute_umbrella_barriers)                     │
│    - Checks merge_stage.get("merge_ai_applied")                         │
│    - Finds: merge_ai_applied is MISSING (because RuntimeError)          │
│    - Result: merge_ready = false                                        │
│                                                                          │
│    Log: "MERGE_NOT_AI_APPLIED reason=merge_not_ai_applied"              │
└────────────────────────────────┬────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ 7. Validation Stage Checks Barrier                                      │
│    - Validation checks: is merge_ready true?                            │
│    - Finds: merge_ready = false                                         │
│    - Result: VALIDATION_FASTPATH_SKIP (defers validation)               │
│                                                                          │
│    Log: "reason=merge_not_ai_applied"                                   │
└─────────────────────────────────────────────────────────────────────────┘
```

### Code References

**Barrier Check** (`backend/runflow/decider.py::_compute_umbrella_barriers`):
```python
def _compute_umbrella_barriers(...):
    # ...
    merge_ai_applied = bool(merge_stage.get("merge_ai_applied"))
    if not merge_ai_applied:
        log.info("MERGE_NOT_AI_APPLIED reason=merge_not_ai_applied sid=%s", sid)
        merge_ready = False
    else:
        merge_ready = True
    # ...
```

**Validation Check** (logs show):
```
VALIDATION_FASTPATH_SKIP sid=bf94cced... reason=merge_not_ai_applied
```

### Why This is a Critical Bug

1. **Merge stage completed successfully**:
   - AI returned decision ✅
   - Result file written ✅
   - Accounts updated ✅
   - All work done ✅

2. **But finalization fails**:
   - RuntimeError thrown due to expected=2 mismatch ❌
   - merge_ai_applied never set ❌
   - merge_ready stays false ❌

3. **Downstream impact**:
   - Validation waits for merge_ready ❌
   - Pipeline stalls ❌
   - User sees "merge not ready" despite merge being complete ❌

---

## 4. Design Fix Options (NO CODE - Design Only)

### Option A: Exclude pairs_count from Expected Candidates ⭐ RECOMMENDED

**Strategy**: Don't add `pairs_count` to `expected_candidates` list. Use only `created_packs` and `packs_built`.

**Changes Required**:
- `backend/runflow/decider.py::finalize_merge_stage` lines 2430-2450
  - Remove or comment out: `if pairs_count is not None: expected_candidates.append(pairs_count)`
- `backend/runflow/decider.py::_merge_artifacts_progress` lines 3841-3844
  - Remove fallback: `if expected_total is None: expected_total = len(pairs_payload)`

**Pros**:
✅ **Simplest fix** - one-line change in each function  
✅ **Matches physical reality** - expected based on pack files, not lookup entries  
✅ **Preserves bidirectional index** - doesn't break lookup functionality  
✅ **Low risk** - only changes calculation logic, not data structures  

**Cons**:
⚠️ Loses validation that pairs array has expected length  
⚠️ Doesn't address root cause (pairs_count semantic ambiguity)  

**Expected Behavior After Fix**:
```python
# For SID bf94cced:
expected_candidates = [1, 1]  # no longer includes pairs_count
expected_total = max([1, 1]) = 1  # matches physical files
ready_counts_match = (1 == 1) and (1 == 1) = True  # passes ✅
merge_ai_applied = True  # set successfully ✅
merge_ready = True  # barrier opens ✅
```

---

### Option B: Distinguish Logical Pairs from Physical Packs

**Strategy**: Add explicit distinction between "logical pairs for lookup" and "physical packs for completion".

**Changes Required**:
- Add new field to pairs_index.json: `"physical_packs_count": 1` (half of pairs_count)
- Update index builders:
  - `scripts/build_ai_merge_packs.py` lines 195-210
  - `backend/core/logic/report_analysis/account_merge.py` lines 5223-5236
- Update expected calculation to use `physical_packs_count` instead of `pairs_count`

**Pros**:
✅ **Explicit semantics** - makes counting distinction clear in schema  
✅ **Preserves both counts** - maintains pairs_count for lookup, adds physical_packs_count for validation  
✅ **Self-documenting** - future developers understand the difference  
✅ **Backward compatible** - can fall back to created_packs if physical_packs_count missing  

**Cons**:
⚠️ **More invasive** - requires changes to index builders and schema  
⚠️ **Migration needed** - existing runs won't have new field  
⚠️ **Redundant data** - physical_packs_count should always equal created_packs  
⚠️ **More places to update** - affects multiple files  

**Expected pairs_index.json After Fix**:
```json
{
  "totals": {
    "created_packs": 1,
    "packs_built": 1,
    "scored_pairs": 3
  },
  "pairs": [
    {"pair": [7, 10], ...},
    {"pair": [10, 7], ...}
  ],
  "pairs_count": 2,                    ← Logical pairs (bidirectional)
  "physical_packs_count": 1            ← NEW: Physical pack files
}
```

---

### Option C: Remove Bidirectional Representation

**Strategy**: Change pairs array to have only one entry per physical pack, add separate lookup index.

**Changes Required**:
- Modify `scripts/build_ai_merge_packs.py` lines 182-193
  - Remove bidirectional loop: `for pair in ((a_idx, b_idx), (b_idx, a_idx))`
  - Keep only canonical direction: `pairs_payload.append({"pair": [a_idx, b_idx], ...})`
- Add separate lookup structure: `"pairs_lookup": {7: [10], 10: [7]}`
- Update all code that reads pairs array to use new lookup

**Pros**:
✅ **Eliminates ambiguity** - pairs array length matches physical files  
✅ **More efficient** - lookup dict faster than array scan  
✅ **Clearer semantics** - pairs array = physical packs  

**Cons**:
❌ **Breaking change** - existing code expects bidirectional pairs array  
❌ **High risk** - affects all merge code that reads pairs  
❌ **Migration complexity** - need to update all existing runs  
❌ **More work** - requires changes across multiple modules  

**Not Recommended** - too invasive for the benefit gained.

---

### Option D: Relax Validation Check (Use Only result_files == pack_files)

**Strategy**: Remove expected check entirely, trust that `result_files == pack_files` is sufficient.

**Changes Required**:
- `backend/runflow/decider.py::finalize_merge_stage` lines 2459-2468
  - Change: `ready_counts_match = result_files_total == pack_files_total`
  - Remove: `if expected_total is not None: ready_counts_match = ... and expected_total`
- Similar change in `_merge_artifacts_progress`

**Pros**:
✅ **Very simple** - just remove a check  
✅ **Pragmatic** - pack_files == result_files is often sufficient  
✅ **No schema changes** - works with existing data  

**Cons**:
❌ **Weakens validation** - loses check that all expected packs were processed  
❌ **Masks issues** - if pairs_index.json has wrong created_packs, we won't notice  
❌ **Silent failures** - could miss incomplete AI processing  

**Not Recommended** - validation is valuable, shouldn't weaken it.

---

### Option E: Use min() Instead of max() for Expected Calculation

**Strategy**: Change `expected_total = max(expected_candidates)` to `min(expected_candidates)`.

**Changes Required**:
- `backend/runflow/decider.py::finalize_merge_stage` line 2447
  - Change: `expected_total = min(expected_candidates) if expected_candidates else None`
- Similar change in `_merge_artifacts_progress`

**Pros**:
✅ **One-line fix** - minimal code change  
✅ **Conservative** - expects minimum files, not maximum  

**Cons**:
❌ **Doesn't fix root cause** - pairs_count still wrong semantic  
❌ **Could mask real issues** - if created_packs is wrong, min() might hide it  
❌ **Unexpected semantics** - why use min when we want "authoritative count"?  

**Not Recommended** - band-aid fix that doesn't address real problem.

---

## Recommendation Summary

### ⭐ **Best Fix: Option A**

**Exclude pairs_count from expected calculation**

**Why**:
1. **Minimal risk** - one-line change in two functions
2. **Correct semantics** - expected should be based on physical packs, not lookup entries
3. **Preserves existing design** - bidirectional pairs array still works for lookup
4. **Easy to test** - just verify expected=created_packs
5. **Quick to implement** - can be done in 10 minutes

**Implementation**:
```python
# backend/runflow/decider.py::finalize_merge_stage (lines 2430-2450)
expected_candidates: list[int] = []
if totals_created_packs is not None:
    expected_candidates.append(totals_created_packs)
packs_built_total = _maybe_int(totals.get("packs_built"))
if packs_built_total is not None:
    expected_candidates.append(packs_built_total)
# ... other candidates ...

# REMOVE THIS BLOCK:
# if pairs_count is not None:
#     expected_candidates.append(pairs_count)

expected_total = max(expected_candidates) if expected_candidates else None
```

### 🔄 **Future Enhancement: Option B**

**Add physical_packs_count field to schema**

**Why**:
- Makes the distinction explicit in documentation
- Provides clear semantics for future developers
- Can be done as a follow-up refactor

**Timeline**: After Option A is verified, consider as Phase 2 cleanup.

---

## Verification Plan

After implementing Option A, verify:

1. **Re-run finalize_merge_stage for SID bf94cced**:
   ```python
   from backend.runflow.decider import finalize_merge_stage
   result = finalize_merge_stage("bf94cced-01d4-479a-b03b-ebf92623aa03")
   assert result["merge_ai_applied"] == True
   ```

2. **Check barrier computation**:
   ```python
   from backend.runflow.decider import refresh_runflow_barriers
   refresh_runflow_barriers("bf94cced-01d4-479a-b03b-ebf92623aa03")
   # Load runflow.json and verify: merge_ready = true
   ```

3. **Verify validation can proceed**:
   - Check logs for `VALIDATION_START` (not `VALIDATION_FASTPATH_SKIP`)
   - Verify validation packs are built

4. **Test with other SIDs**:
   - Find SIDs with different pairs_count values
   - Verify expected calculation uses created_packs, not pairs_count

---

## Related Files

### Core Functions
- `backend/runflow/decider.py::finalize_merge_stage` (lines 2169-2540)
- `backend/runflow/decider.py::_merge_artifacts_progress` (lines 3773-3870)
- `backend/runflow/decider.py::_compute_umbrella_barriers` (barrier check)

### Index Builders
- `scripts/build_ai_merge_packs.py` (lines 170-220) - creates bidirectional pairs
- `backend/core/logic/report_analysis/account_merge.py::score_all_pairs_0_100` (lines 5223-5253)

### Tests to Update
- `backend/tests/test_merge_ai_applied_fix.py` - existing merge tests
- `tests/backend/runflow/test_decider.py` - finalize_merge_stage tests

---

## Conclusion

**The bug is clear**: expected calculation incorrectly uses `pairs_count` (2) which counts bidirectional lookup entries, when it should use `created_packs` (1) which counts physical pack files.

**The fix is simple**: Remove `pairs_count` from `expected_candidates` list in both `finalize_merge_stage` and `_merge_artifacts_progress`.

**The impact is contained**: One-line change per function, no schema changes, no migration needed.

**Ready for implementation** when user approves.
