# Investigation: Validation Natives Missing from Manifest
## SID: 160884fe-b510-493b-888d-dd2ec09b4bb5

**Investigation Date:** November 16, 2025  
**Status:** READ-ONLY (No code changes)  
**Problem:** Validation artifacts exist on disk but `ai.packs.validation.*` and `ai.validation.*` are null in manifest

---

## Executive Summary

**Root Cause:** The validation pack build process for this SID used the **new validation pipeline** (`backend/validation/build_packs.py` → `ValidationPackBuilder`), which **requires** `ai.packs.validation.*` paths to **already exist** in the manifest before it runs. It never writes them - it only reads them.

The code path that **should** have written these paths (`ensure_validation_section` from `backend/ai/manifest.py`) was **never called** for this SID because:

1. The new validation pipeline (`run_validation_summary_pipeline` in `backend/validation/pipeline.py`) does NOT call `ensure_validation_section` before building packs
2. It expects the manifest to be pre-populated with validation paths (legacy behavior from older init code)
3. When paths are missing, it has an **auto-repair** mechanism that catches the exception and calls `ensure_validation_section`, BUT this repair only works if:
   - The manifest can be successfully loaded
   - The SID can be inferred from the manifest
   - The runs_root can be determined
4. For this SID, something prevented the auto-repair from working, OR the packs were built via a different code path that bypassed both the init and the repair

The validation results were successfully merged into summaries using the **inline sender** (`run_validation_send_for_sid` which DOES call `ensure_validation_section`), but this happened **after** pack building, so the manifest paths were written too late for the pack builder to see them.

**Impact:**
- Validation artifacts physically exist on disk (`ai_packs/validation/{packs, results, index.json, logs.txt}`)
- Validation results were successfully applied (`merge_results_applied=true`)
- Strategy successfully used the validation data
- BUT: Manifest doesn't record where the validation artifacts are located
- This breaks any tooling that relies on manifest to find validation paths
- Auto-repair mechanisms in sender/pipeline can compensate, but it's fragile

---

## 1. Who Owns Validation Natives in the Manifest?

### 1.1 Code Locations for `ai.packs.validation.*`

**Writer: `backend/ai/manifest.py`**

**Function:** `Manifest.ensure_validation_section(sid, runs_root)` (lines 132-217)

**What it does:**
- Creates validation directories on disk if they don't exist
- Ensures `ai.packs.validation` section exists in manifest
- Writes canonical paths for:
  - `base`, `dir` → validation base directory
  - `packs`, `packs_dir` → packs directory
  - `results`, `results_dir` → results directory
  - `index` → index.json path
  - `logs` → logs.txt path
  - `last_built_at` → timestamp (if changed)
- Persists manifest to disk if any values changed
- Logs: `VALIDATION_MANIFEST_INJECTED sid={sid} packs_dir={packs_dir} results_dir={results_dir}`

**When it runs:**
```python
# Line 181-195: Core logic
canonical_values = {
    "base": str(validation_paths.base),
    "dir": str(validation_paths.base),
    "packs": str(validation_paths.packs_dir),
    "packs_dir": str(validation_paths.packs_dir),
    "results": str(validation_paths.results_dir),
    "results_dir": str(validation_paths.results_dir),
    "index": str(validation_paths.index_file),
    "logs": str(validation_paths.log_file),
}

changed = False
for key, value in canonical_values.items():
    current = validation_section.get(key)
    if not isinstance(current, str) or not current.strip():
        validation_section[key] = value
        changed = True

if changed:
    persist_manifest(manifest)
```

**Behavior:**
- **Upserts only** (sets null/empty values to canonical paths)
- **Never clears** existing non-empty values
- **Idempotent** - safe to call multiple times

---

**Alternative Writer: `backend/pipeline/runs.py`**

**Function:** `RunManifest.upsert_validation_packs_dir(...)` (lines 842-903)

**What it does:**
- Similar to `ensure_validation_section` but offers more control
- Sets both `ai.validation.*` (top-level validation section) and `ai.packs.validation.*`
- Also updates `last_built_at`, `last_prepared_at` timestamps
- Used by legacy validation builder (`backend/ai/validation_builder.py` line 2275)

**Code:**
```python
# Line 842-903
def upsert_validation_packs_dir(
    self,
    base_dir: Path,
    *,
    packs_dir: Path | None = None,
    results_dir: Path | None = None,
    index_file: Path | None = None,
    log_file: Path | None = None,
    account_dir: Path | None = None,
) -> "RunManifest":
    validation = self._ensure_validation_section()
    # ... sets ai.validation.base, dir, accounts, accounts_dir, last_prepared_at
    
    packs_validation = self._ensure_ai_validation_pack_section()
    # ... sets ai.packs.validation.base, dir, packs, packs_dir, results, results_dir, index, logs, last_built_at
```

**When it's supposed to run:**
- Legacy validation builder: `backend/ai/validation_builder.py` line 2275 (inside `_update_manifest_for_run`)
- **BUT:** Only called if `writer.last_pack_was_written()` returns true (line 2305)
- For single-account pack builds in `build_validation_pack_for_account` (line 2287-2308)

---

### 1.2 Code Locations for `ai.validation.*`

**Same writers as above:**
- `RunManifest.upsert_validation_packs_dir` also sets `ai.validation.{base, dir, accounts, accounts_dir, last_prepared_at}`
- `Manifest.ensure_validation_section` does NOT touch `ai.validation.*` - only `ai.packs.validation.*`

**Design split:**
- `ai.validation.*` = top-level validation "accounts" directory (legacy, rarely used)
- `ai.packs.validation.*` = validation pack artifacts (packs, results, index, logs)

---

### 1.3 Code Locations for `ai.status.validation.*`

**Writer: `backend/pipeline/runs.py`**

**Function:** `RunManifest.mark_validation_merge_applied(...)` (lines 1105-1199)

**What it does:**
- Sets `ai.status.validation.merge_results_applied = true`
- Sets `ai.status.validation.merge_results_applied_at = {timestamp}`
- Sets `ai.status.validation.merge_results.applied = true`
- Sets `ai.status.validation.merge_results.applied_at = {timestamp}`
- Sets `ai.status.validation.merge_results.source = {source}` (e.g., "umbrella_autofix", "validation_builder_inline")
- Also sets `ai.status.validation.sent = true` and `ai.status.validation.completed_at` if not already set
- **Does NOT touch ai.packs.validation or ai.validation paths**

**When it runs:**
```python
# Called by:
# 1. backend/runflow/umbrella.py line 532 - umbrella auto-repair
manifest.mark_validation_merge_applied(applied=True, source="umbrella_autofix")

# 2. backend/pipeline/validation_merge_helpers.py line 182 - after merge
manifest.mark_validation_merge_applied(
    applied=True,
    applied_at=now_utc,
    source=source,  # e.g., "validation_builder_inline", "auto_ai_pipeline"
)
```

---

### 1.4 All Call Sites for `ensure_validation_section`

| Call Site | File | Line | When It Runs |
|-----------|------|------|--------------|
| Legacy validation autosend | `backend/ai/validation_builder.py` | 1919 | Inside `_maybe_send_validation_packs` (legacy orchestration, DISABLED by default) |
| Inline validation sender | `backend/ai/validation_builder.py` | 2057 | Inside `run_validation_send_for_sid` (modern inline send) |
| Build packs for run (legacy) | `backend/ai/validation_builder.py` | 2320 | Inside `build_validation_packs_for_run` (legacy pack builder) |
| Auto_ai_tasks validation_build | `backend/pipeline/auto_ai_tasks.py` | 1735 | Celery task `validation_build` (modern task chain) |
| Validation pipeline auto-repair | `backend/validation/pipeline.py` | 580 | Inside `_prepare_manifest` fallback (when paths missing) |
| Devtools verify script | `devtools/verify_validation.py` | 142 | Manual verification tool |

**Key observation:**
- The **modern Celery task chain** (`validation_build` → `validation_send` → `validation_merge_ai_results_step`) calls `ensure_validation_section` in `validation_build` task (line 1735) BEFORE building packs
- The **new validation pipeline** (`run_validation_summary_pipeline` → `ValidationPackBuilder`) does NOT call it directly; relies on auto-repair fallback

---

## 2. Could Strategy or Umbrella Be Overwriting Validation Section?

### 2.1 Bulk Write Analysis

**Searched for:**
- `ai["packs"] =` (direct assignment)
- `manifest["ai"]["packs"] =` (direct assignment)
- `data["ai"]["packs"] =` (direct assignment)

**Results:**
- **Zero matches** for bulk overwrites in production code
- All writes go through `RunManifest` helper methods which upsert into existing sections
- No "compaction" or "cleanup" code that drops validation keys

**Conclusion:** No evidence of overwrite risk from strategy, umbrella, or compaction.

---

### 2.2 Umbrella Auto-Repair

**File:** `backend/runflow/umbrella.py`  
**Function:** `schedule_note_style_after_validation` (line 532)

**What it does:**
```python
# Line 520-535: Auto-repair merge flag if runflow shows applied but manifest does not
merge_applied_runflow = ... # reads from runflow.json
merge_applied_manifest = ... # reads from manifest.json

if merge_applied_runflow and not merge_applied_manifest:
    try:
        manifest.mark_validation_merge_applied(applied=True, source="umbrella_autofix")
        merge_applied_manifest = True
    except Exception:
        pass
```

**Impact:**
- **Only touches `ai.status.validation.merge_results*`**
- **Does NOT touch `ai.packs.validation` or `ai.validation` paths**
- Safe - no overwrite risk

---

### 2.3 For SID 160884fe: Was it Overwritten or Never Populated?

**Evidence:**
1. **Disk artifacts exist:**
   - `ai_packs/validation/packs/` - 3 pack files (7.4KB each, created 15:51:37Z)
   - `ai_packs/validation/results/` - 3 result files (~800 bytes each, completed 15:51:51Z - 15:52:05Z)
   - `ai_packs/validation/index.json` - 1616 bytes, last written 15:53:16Z
   - `ai_packs/validation/logs.txt` - 1486 bytes (pack build logs), 262KB (full logs)

2. **Manifest state:**
   - `ai.packs.validation.*` - ALL null
   - `ai.validation.*` - ALL null
   - `ai.status.validation.merge_results_applied` - TRUE
   - `ai.status.validation.merge_results.source` - "umbrella_autofix"

3. **Runflow state:**
   - `stages.validation.status` - "success"
   - `stages.validation.merge_results.applied` - true
   - `stages.validation.merge_results.applied_at` - "2025-11-16T15:51:59Z"
   - `stages.validation.merge_results.source` - "validation_builder_inline"
   - `stages.validation.fastpath` - false

**Timeline reconstruction:**

| Time | Event | Component | Manifest State |
|------|-------|-----------|----------------|
| 15:51:37Z | Validation packs built to disk | ValidationPackBuilder | `ai.packs.validation.*` = ??? |
| 15:51:51Z - 15:52:05Z | Validation AI results written | Validation sender | `ai.packs.validation.*` = ??? |
| 15:51:59Z | Merge applied to summaries | `validation_builder_inline` | merge_results_applied=true in runflow |
| 15:53:16Z | Validation index finalized | Validation index writer | - |
| 15:53:17Z | Strategy completed | Strategy planner | - |
| 15:53:18Z | Umbrella auto-repair | umbrella_autofix | `ai.status.validation.merge_results.source` = "umbrella_autofix" |

**Discrepancy:**
- Runflow shows `source="validation_builder_inline"` (applied at 15:51:59Z)
- Manifest shows `source="umbrella_autofix"` (applied at 15:53:18Z)
- This means: umbrella **overwrote** the merge_results source field at 15:53:18Z, ~96 seconds after validation completed

**Conclusion:**
- Validation paths were **NEVER populated** in the manifest
- Umbrella only changed the `source` field of an existing status, did not drop validation paths
- The paths were missing from the start

---

## 3. Relationship Between `merge_results_applied` and Missing Paths

### 3.1 Where `merge_results_applied` is Set

**Primary setter: `backend/pipeline/validation_merge_helpers.py`**

**Function:** `apply_validation_merge_and_update_state(sid, runs_root, source)` (line ~100-200)

**What it does:**
1. Reads validation results from disk (`validation_results_dir`)
2. Merges AI decisions into summary.json files for each account
3. Updates runflow.json validation stage metrics
4. Calls `manifest.mark_validation_merge_applied(applied=True, applied_at=now_utc, source=source)`
5. Logs: `VALIDATION_AI_MERGE_APPLIED sid={sid} accounts={count} fields={count}`

**Code inspection (inferred from usage):**
```python
# Line 182 (approximate)
manifest.mark_validation_merge_applied(
    applied=True,
    applied_at=_utc_now(),
    source=source,  # "validation_builder_inline", "auto_ai_pipeline", "umbrella_autofix"
)
```

**What it touches:**
- Reads: `ai_packs/validation/results/*.result.jsonl`
- Writes: `cases/accounts/{id}/summary.json` (merges validation decisions into field objects)
- Writes: `runflow.json` (updates validation stage metrics)
- Writes: `manifest.json` (only `ai.status.validation.merge_results*`, NOT paths)

**Does it touch `ai.packs.validation` paths?**
- **NO** - only status/metrics, not paths

---

### 3.2 Can Merge Happen Without Paths?

**Yes!** The merge helper uses **convention-based paths**, not manifest paths:

```python
# backend/pipeline/validation_merge_helpers.py (inferred)
def apply_validation_merge_and_update_state(sid, runs_root, source):
    # Uses convention: runs_root / sid / "ai_packs" / "validation" / "results"
    results_dir = validation_results_dir(sid, runs_root=runs_root, create=True)
    
    # Reads results from disk
    result_files = list(results_dir.glob("*.result.jsonl"))
    
    # Merges into summaries
    for result_file in result_files:
        # ... merge logic ...
    
    # Updates manifest status (not paths)
    manifest.mark_validation_merge_applied(applied=True, source=source)
```

**Path resolution uses helpers:**
- `backend/core/ai/paths.py` - `validation_results_dir(sid, runs_root)` computes path by convention
- Does NOT read manifest to find paths
- Assumes `{runs_root}/{sid}/ai_packs/validation/results/`

**Result:**
- `merge_results_applied=true` can be set even if `ai.packs.validation.results_dir=null` in manifest
- System relies on **convention-based paths**, not **manifest-declared paths**
- Manifest paths are intended for **documentation/tooling**, not runtime discovery

---

### 3.3 Why This Design?

**Historical context (inferred from code comments):**

1. **Early design:** Manifest was supposed to be the "source of truth" for all paths
   - Validation pipeline (`ValidationPackBuilder`) was designed to read paths from manifest
   - Expected something upstream to populate manifest before pack building

2. **Evolution:** Convention-based paths became the norm
   - Core helpers (`validation_results_dir`, `validation_index_path`, etc.) use SID + runs_root to compute paths
   - Manifest paths became **optional documentation**, not **required configuration**

3. **Current state:** Hybrid system
   - **New code** (validation_builder, merge helpers) uses convention-based paths
   - **Old code** (ValidationPackBuilder) expects manifest paths
   - **Auto-repair** mechanisms try to bridge the gap

**Result:**
- Fragile - works if auto-repair succeeds, breaks if it doesn't
- Inconsistent - some code paths populate manifest, others don't
- Hard to debug - hard to tell which path was taken just from looking at final state

---

## 4. `ai.validation` vs `ai.packs.validation` Design Split

### 4.1 Intended Roles

**`ai.validation.*` (top-level validation section):**
- `base` - base validation directory (legacy, same as ai.packs.validation.base)
- `dir` - validation directory (legacy alias for base)
- `accounts` - validation accounts directory (legacy, unused)
- `accounts_dir` - validation accounts directory (legacy alias)
- `last_prepared_at` - timestamp of last validation prep (legacy)

**Design intent (inferred):**
- **Legacy section** from older validation system
- Intended to record top-level validation "preparation" artifacts
- **Rarely used** in modern code
- Only populated by `RunManifest.upsert_validation_packs_dir` (legacy builder)
- Modern code ignores this section

---

**`ai.packs.validation.*` (validation pack artifacts):**
- `base` - base directory for validation packs
- `dir` - alias for base
- `packs` - packs directory path
- `packs_dir` - alias for packs
- `results` - results directory path
- `results_dir` - alias for results
- `index` - index.json file path
- `logs` - logs.txt file path
- `last_built_at` - timestamp of last pack build
- `status.sent` - whether packs were sent to AI
- `status.completed_at` - when AI processing completed

**Design intent:**
- **Active section** for modern validation system
- Records where validation pack artifacts are stored
- Used by `ValidationPackBuilder` to find output directories
- Populated by `ensure_validation_section` and `upsert_validation_packs_dir`

---

### 4.2 For SID 160884fe: Why Are Both Null?

**For `ai.validation.*`:**
- **Expected:** Null (modern code doesn't use this)
- **Actual:** Null ✓
- **Conclusion:** Normal - this section is deprecated/legacy

**For `ai.packs.validation.*`:**
- **Expected:** Populated with paths
- **Actual:** Null ✗
- **Conclusion:** Abnormal - indicates manifest init failed or was skipped

---

### 4.3 Usage Sites in Code

**`ai.validation.*` readers:**
```bash
git grep "ai.validation" --and --not -e "ai.packs.validation" --and --not -e "ai.status.validation"
# Results: Mostly test files, legacy code, no active production usage
```

**`ai.packs.validation.*` readers:**
- `backend/validation/build_packs.py` line 762-804 - `ValidationPackBuilder._resolve_manifest_paths`
  - **REQUIRES** these paths to exist
  - Raises ValueError if missing
- `backend/validation/pipeline.py` line 559-562 - Auto-repair check
  - Detects when paths are missing
  - Attempts to call `ensure_validation_section` to fix
- `backend/pipeline/runs.py` line 786-803 - `RunManifest._ensure_ai_validation_pack_section`
  - Helper that ensures the section exists in manifest dict (but doesn't populate paths)

**Conclusion:**
- `ai.validation.*` = **legacy, unused, safe to ignore**
- `ai.packs.validation.*` = **active, required by ValidationPackBuilder, missing for this SID**

---

## 5. Timeline for SID 160884fe: Expected vs Actual

### 5.1 Expected Timeline (Modern Celery Task Chain)

**Ideal flow (if Celery task chain was used):**

1. **Task: `validation_build`** (`auto_ai_tasks.py` line 1725-1769)
   - Calls `ensure_validation_section(sid, runs_root=runs_root)` (line 1735)
   - → Writes `ai.packs.validation.*` paths to manifest
   - → Logs: `VALIDATION_MANIFEST_INJECTED sid={sid} packs_dir={packs_dir}`
   - Calls `build_validation_packs_for_run(sid, runs_root=runs_root)` (line 1739)
   - → ValidationBuilder reads manifest paths (now populated)
   - → Writes packs to disk
   - Logs: `VALIDATION_BUILD_DONE sid={sid} packs={count}`

2. **Task: `validation_send`** (`auto_ai_tasks.py` line 1772-1861)
   - Reads index from disk
   - Sends packs to AI
   - Writes results to disk
   - Logs: `VALIDATION_SEND_DONE sid={sid} result_files={count}`

3. **Task: `validation_merge_ai_results_step`** (`auto_ai_tasks.py` line 961-1081)
   - Calls `apply_validation_merge_and_update_state(sid, runs_root, source="auto_ai_pipeline")`
   - → Merges results into summaries
   - → Calls `manifest.mark_validation_merge_applied(..., source="auto_ai_pipeline")`
   - Logs: `AUTO_AI_VALIDATION_MERGE_DONE sid={sid}`

4. **Later: Umbrella reconciliation**
   - Reads runflow and manifest
   - If merge_applied in runflow but not in manifest:
     - Calls `manifest.mark_validation_merge_applied(..., source="umbrella_autofix")`
   - (But paths already populated, so no issue)

**Result:** `ai.packs.validation.*` populated, merge_results_applied=true, source="auto_ai_pipeline"

---

### 5.2 Actual Timeline for SID 160884fe

**Reconstructed from artifacts:**

| Time | Event | Evidence | Manifest State |
|------|-------|----------|----------------|
| ~15:51:36Z | Run initialized | `runflow.json` created | `ai.packs.validation.*` = ??? |
| 15:51:37Z | Validation packs built | `val_acc_{009,010,011}.jsonl` created (disk) | `ai.packs.validation.*` = null (likely) |
| 15:51:51Z | First AI result | `acc_009.result.jsonl` created | `ai.packs.validation.*` = null |
| 15:51:55Z | Second AI result | `acc_010.result.jsonl` created | `ai.packs.validation.*` = null |
| 15:51:59Z | Merge applied (inline) | runflow.json: `merge_results.source="validation_builder_inline"` | `ai.status.validation.merge_results_applied` = ??? |
| 15:52:05Z | Third AI result | `acc_011.result.jsonl` created | - |
| 15:53:16Z | Index finalized | `index.json` last written | - |
| 15:53:17Z | Strategy completed | runflow.json: `strategy.status="success"` | - |
| 15:53:18Z | Umbrella auto-repair | manifest.json: `merge_results.source="umbrella_autofix"` | `ai.status.validation.merge_results_applied` = true |
| 16:02:59Z (inferred) | Final state | run_state=AWAITING_CUSTOMER_INPUT | `ai.packs.validation.*` = null (final) |

---

### 5.3 Why Paths Are Missing: Three Possible Scenarios

**Scenario A: ValidationPackBuilder Called Without Manifest Init**

**Flow:**
1. Something called `ValidationPackBuilder` directly (or via `run_validation_summary_pipeline`)
2. Manifest was not pre-populated with paths
3. `ValidationPackBuilder._resolve_manifest_paths` raised ValueError: "Manifest missing ai.packs.validation"
4. Auto-repair in `pipeline.py` line 559-590 **failed** or was **bypassed**
5. Packs were built via fallback mechanism or different code path

**Evidence:**
- `runflow.json` shows `fastpath=false` → didn't use fastpath
- Packs exist on disk → **something** built them
- Manifest paths are null → init/repair failed

**Likelihood:** **Medium** - auto-repair should have worked unless exception was swallowed

---

**Scenario B: Inline Sender Called Before Pack Builder**

**Flow:**
1. Validation merge task (`validation_merge_ai_results_step`) detected missing results
2. Called `run_validation_send_for_sid` as inline fallback (line 1049)
3. Inline sender calls `ensure_validation_section` (line 2057 in validation_builder.py)
4. → Writes `ai.packs.validation.*` to manifest
5. → Sends packs, writes results
6. BUT: Packs were **already built earlier** by ValidationPackBuilder (which bypassed init)
7. Umbrella later overwrites `merge_results.source` to "umbrella_autofix"

**Evidence:**
- runflow shows `merge_results.source="validation_builder_inline"` (15:51:59Z)
- manifest shows `merge_results.source="umbrella_autofix"` (15:53:18Z)
- Timestamps match: packs created 15:51:37Z, inline send 15:51:59Z, umbrella 15:53:18Z

**Likelihood:** **High** - explains the timeline and source field discrepancy

---

**Scenario C: Manifest Paths Were Written Then Cleared**

**Flow:**
1. `ensure_validation_section` called and wrote paths
2. Something later **cleared** or **reset** the `ai.packs` section
3. Paths lost, but status fields (`merge_results_applied`) preserved because they're in `ai.status.validation`

**Evidence against:**
- No code found that bulk-overwrites `ai.packs`
- All manifest updates use upsert helpers
- No compaction/cleanup code that drops keys

**Likelihood:** **Very Low** - no evidence of overwrite mechanism

---

### 5.4 Most Likely Explanation

**Scenario B is most likely:**

1. **Pack building happened first** (15:51:37Z)
   - Called via `run_validation_summary_pipeline` (modern pipeline)
   - Expected manifest to have paths pre-populated (legacy assumption)
   - Auto-repair mechanism tried to call `ensure_validation_section` but **failed silently** or **completed but manifest wasn't persisted**

2. **Inline send happened second** (15:51:51Z - 15:52:05Z)
   - `validation_merge_ai_results_step` detected missing results
   - Called `run_validation_send_for_sid` inline (line 1049 in auto_ai_tasks.py)
   - This path calls `ensure_validation_section` (line 2057 in validation_builder.py)
   - → **Should** have written paths to manifest
   - → But packs were already built, so sender just sent existing packs

3. **Merge happened third** (15:51:59Z)
   - Inline sender called `apply_validation_merge_and_update_state(..., source="validation_builder_inline")`
   - Marked `merge_results_applied=true` in **runflow.json**
   - Also marked in **manifest.json** (but this write may have been incomplete)

4. **Umbrella overwrote source field** (15:53:18Z)
   - Umbrella reconciliation detected `merge_applied` in runflow but different source in manifest
   - Called `manifest.mark_validation_merge_applied(..., source="umbrella_autofix")`
   - This **overwrote** the source field from "validation_builder_inline" to "umbrella_autofix"

**Result:**
- Packs exist on disk
- Merge applied successfully
- **But manifest paths were never written OR were written but not persisted**

**Root cause:**
- Race condition or exception swallowing in manifest write path
- OR: `ensure_validation_section` was called but `persist_manifest` failed silently
- OR: Manifest was written to disk but later process reloaded old version from disk before umbrella write

---

## 6. Root Cause & All Code Paths That Change `ai.packs.validation`

### 6.1 Summary: Root Cause

**For SID 160884fe, validation natives are missing from the manifest because:**

**Primary cause:** The validation pack build process used a code path that did NOT call `ensure_validation_section` before building packs. Specifically:

- `run_validation_summary_pipeline` in `backend/validation/pipeline.py` was likely used
- This pipeline creates a `ValidationPackBuilder` which **expects** manifest paths to already exist
- When paths are missing, it has auto-repair logic (lines 559-590) that tries to call `ensure_validation_section`
- For this SID, the auto-repair either:
  - **Failed silently** (exception caught but not logged)
  - **Succeeded but manifest wasn't persisted** (changed=False path, or persist_manifest raised exception)
  - **Was bypassed** (packs built via different mechanism)

**Contributing factors:**

1. **Fragile initialization:** Multiple code paths expect different things
   - Modern Celery tasks call `ensure_validation_section` upfront
   - New validation pipeline expects pre-populated manifest
   - Auto-repair is best-effort, can fail silently

2. **Convention-based vs manifest-based paths:**
   - Runtime code uses convention (doesn't need manifest)
   - Tooling/debugging needs manifest to find artifacts
   - Disconnect creates brittleness

3. **Source field overwrite:**
   - Umbrella overwrote `merge_results.source` from "validation_builder_inline" to "umbrella_autofix"
   - Makes it harder to trace which code path was actually used
   - Not a bug, but makes investigation harder

**Why merge_results_applied=true but paths are null:**
- Merge helpers use convention-based paths, don't read manifest
- `mark_validation_merge_applied` only touches status fields, not paths
- These are independent operations - one can succeed while other fails

---

### 6.2 All Code Paths That Change `ai.packs.validation`

| # | Component | Function | File:Line | Operation | When It Runs |
|---|-----------|----------|-----------|-----------|--------------|
| 1 | **Manifest.ensure_validation_section** | `ensure_validation_section` | `backend/ai/manifest.py:132-217` | **CREATE/UPSERT** paths | Called by validation_builder, auto_ai_tasks, pipeline auto-repair |
| 2 | **RunManifest.upsert_validation_packs_dir** | `upsert_validation_packs_dir` | `backend/pipeline/runs.py:842-903` | **UPSERT** paths + timestamps | Called by legacy validation builder after pack write |
| 3 | **Validation pipeline auto-repair** | `_prepare_manifest` | `backend/validation/pipeline.py:559-590` | **CALLS #1** when paths missing | Exception handler when ValidationPackBuilder init fails |
| 4 | **Legacy validation builder** | `_update_manifest_for_run` | `backend/ai/validation_builder.py:2263-2286` | **CALLS #2** after pack write | Single-account pack builds in legacy mode |

**Operations:**
- **CREATE/UPSERT** - sets null/empty values to canonical paths, preserves existing non-empty values
- **CALLS #1** - indirect, calls `ensure_validation_section`
- **CALLS #2** - indirect, calls `RunManifest.upsert_validation_packs_dir`

**None of these paths DELETE or CLEAR paths** - all are write-only or write-if-empty.

---

### 6.3 Safe Fix Strategy (Conceptual)

**To ensure validation paths are always populated:**

1. **Mandate init in all validation entry points:**
   - `run_validation_summary_pipeline` should call `ensure_validation_section` at START (line ~370)
   - `ValidationPackBuilder.__init__` should call it if paths are missing (defensive init)
   - Remove reliance on auto-repair exception handler

2. **Make `ensure_validation_section` idempotent and loud:**
   - Log at INFO level every time it runs
   - Log at WARNING if manifest is modified (paths were missing)
   - Raise exception if persist_manifest fails (don't swallow)

3. **Add validation path check to runflow barriers:**
   - After validation completes, check if `ai.packs.validation.packs_dir` is null
   - If null, log ERROR and call `ensure_validation_section` as repair
   - Add telemetry metric for "validation_manifest_paths_missing"

4. **Prevent umbrella source field overwrite:**
   - In `mark_validation_merge_applied`, check if source field already exists
   - Only set source if currently null or if new source is "more authoritative"
   - Preserve original source, add secondary source field if needed

5. **Add verification to CI/tests:**
   - After validation runs in tests, assert `ai.packs.validation.packs_dir is not None`
   - Catch regressions early

---

## 7. Answers to Specific Questions

### Question 1: Who is supposed to populate `ai.packs.validation.*` and `ai.validation.*`?

**Answer:**
- `ai.packs.validation.*` → `Manifest.ensure_validation_section()` (`backend/ai/manifest.py:132`)
- `ai.validation.*` → `RunManifest.upsert_validation_packs_dir()` (`backend/pipeline/runs.py:842`) - legacy only
- **When:** Before validation pack building starts
- **Expected invariant after validation completes:** `ai.packs.validation.{packs_dir, results_dir, index, logs}` should all be non-null strings pointing to validation artifacts, and should remain stable until run is archived/deleted

---

### Question 2: Could strategy or umbrella be overwriting/clearing the validation section?

**Answer:**
- **NO** - no code found that bulk-overwrites `ai.packs` or clears validation keys
- Umbrella only modifies `ai.status.validation.merge_results.source` (changes "validation_builder_inline" → "umbrella_autofix")
- This is a **metadata update**, not a path deletion
- All manifest mutations use upsert helpers that preserve existing keys

**For SID 160884fe:** Paths were **never populated**, not overwritten.

---

### Question 3: Where is `merge_results_applied=true` and `source="umbrella_autofix"` set?

**Answer:**
- Set by `RunManifest.mark_validation_merge_applied()` (`backend/pipeline/runs.py:1105`)
- Called from `backend/runflow/umbrella.py:532` (umbrella auto-repair)
- **What it touches:** Only `ai.status.validation.merge_results*` - NOT paths
- **Why umbrella sets it:** Detected `merge_results_applied=true` in runflow.json but different source in manifest.json; auto-repair to sync state
- **Path where umbrella doesn't need validation manifest paths:** Uses convention-based path resolution (`validation_results_dir(sid, runs_root)` computes path from SID, doesn't read manifest)

---

### Question 4: What is the intended split between `ai.validation` and `ai.packs.validation`?

**Answer:**
- `ai.validation.*` = **Legacy top-level validation section**, rarely used, safe to ignore
  - Intended for "validation preparation" artifacts (accounts directories)
  - Modern code doesn't use it
- `ai.packs.validation.*` = **Active validation pack artifacts section**, required by tooling
  - Records where packs, results, index, logs are stored
  - Required by `ValidationPackBuilder` to function
  - Should always be populated when validation runs

**For SID 160884fe:** Both are null because the code path that populates them was never called or failed silently.

---

### Question 5: Timeline - What should have happened vs what actually happened?

**Expected:**
1. Celery task `validation_build` calls `ensure_validation_section` → writes paths to manifest
2. Task calls `build_validation_packs_for_run` → ValidationBuilder reads paths from manifest, builds packs
3. Task `validation_send` sends packs to AI
4. Task `validation_merge_ai_results_step` merges results, marks `merge_results_applied=true`

**Actual:**
1. **Unknown init path** (possibly `run_validation_summary_pipeline`) tried to build packs
2. **Manifest paths were never written** (init skipped or failed)
3. ValidationPackBuilder **either failed or used fallback mechanism** to build packs anyway
4. **Inline sender** (`run_validation_send_for_sid`) was called as fallback, sent packs, wrote results
5. Inline sender called `ensure_validation_section` but **either too late or persist failed**
6. Merge happened successfully (convention-based paths work without manifest)
7. Umbrella overwrote source field for sync

**Critical missing step:** `ensure_validation_section` was never called **before** pack building, or was called but persist failed.

---

### Question 6: What's the most likely cause for "validation natives missing from manifest"?

**Root cause:** The validation pack build process for this SID used a code path (`run_validation_summary_pipeline` or similar) that does NOT call `ensure_validation_section` before creating a `ValidationPackBuilder`. The auto-repair mechanism failed or was bypassed, leaving paths unpopulated.

**All code paths that currently can change `ai.packs.validation`:**
1. `Manifest.ensure_validation_section` - **CREATE/UPSERT** (4 call sites)
2. `RunManifest.upsert_validation_packs_dir` - **UPSERT** (1 active call site in legacy builder)
3. Validation pipeline auto-repair - **indirect CALL #1** (exception handler)
4. Legacy builder manifest update - **indirect CALL #2** (after pack write)

**Safe fix:**
- Add explicit `ensure_validation_section` call at START of `run_validation_summary_pipeline`
- Make validation path population mandatory, not best-effort
- Add verification checks to runflow barriers
- Prevent source field overwrites that obscure actual execution path

---

## 8. Code Reference Summary

**Manifest writers:**
- `backend/ai/manifest.py:132-217` - `Manifest.ensure_validation_section` (primary)
- `backend/pipeline/runs.py:842-903` - `RunManifest.upsert_validation_packs_dir` (legacy)
- `backend/pipeline/runs.py:1105-1199` - `RunManifest.mark_validation_merge_applied` (status only)

**Manifest readers (require paths):**
- `backend/validation/build_packs.py:762-804` - `ValidationPackBuilder._resolve_manifest_paths`

**Manifest auto-repair:**
- `backend/validation/pipeline.py:559-590` - `_prepare_manifest` exception handler

**Convention-based path helpers (don't need manifest):**
- `backend/core/ai/paths.py` - `validation_results_dir`, `validation_index_path`, `validation_logs_path`
- `backend/pipeline/validation_merge_helpers.py` - `apply_validation_merge_and_update_state`

**Call sites for `ensure_validation_section`:**
1. `backend/ai/validation_builder.py:1919` - legacy autosend (DISABLED)
2. `backend/ai/validation_builder.py:2057` - inline sender (ACTIVE)
3. `backend/ai/validation_builder.py:2320` - legacy pack builder (CONDITIONAL)
4. `backend/pipeline/auto_ai_tasks.py:1735` - Celery validation_build task (ACTIVE)
5. `backend/validation/pipeline.py:580` - auto-repair fallback (BEST-EFFORT)

---

**End of Investigation**

**Recommendation:** Add explicit validation path initialization to all validation entry points, with loud logging and strict error handling. Remove reliance on best-effort auto-repair mechanisms.
