# Validation Native Paths T0 Migration Investigation

**Investigation Date:** November 18, 2025  
**Status:** Design-Only (No Code Changes)  
**Scope:** Complete mapping of validation native paths injection, persistence, reading, and safe T0 migration design

---

## Executive Summary

This investigation maps the complete lifecycle of validation native paths (`ai.packs.validation.*` and `ai.validation.*`) across the entire credit-analyzer pipeline. The goal is to design a safe migration strategy to move initial path injection from its current distributed location (inside builder/updater functions) to **T0 (pipeline entry)**, ensuring paths are set exactly once at validation start and never cleared or overwritten thereafter.

### Key Findings

1. **Current State**: Validation natives are injected at **T1 (builder phase)** by `_update_manifest_for_run`, called from `build_validation_packs_for_run`
2. **Problem**: Natives appear in logs during build but are `null` in final manifest for some SIDs
3. **Root Cause**: Multiple writers exist with varying preservation logic; no single authoritative T0 injection point
4. **Risk**: Later manifest saves can clobber natives if not using preservation-aware APIs

### Proposed Solution

Move native path injection to **T0 = VALIDATION_V2_PIPELINE_ENTRY** (in `backend/api/tasks.py` line ~903), immediately after requirements complete and before `build_validation_packs_for_run` is called. This ensures:

- Paths exist from the very start of validation processing
- Builder/sender/apply phases can assume paths are always present
- No race conditions or "first writer wins" scenarios
- Preservation logic becomes a failsafe rather than primary mechanism

---

## 1. Complete Write-Map for Validation Native Paths

### 1.1 Writers to `ai["packs"]["validation"]`

#### Writer #1: `ensure_validation_paths` (Helper Function)

**Location:** `backend/core/ai/paths.py` lines 78-115  
**Called by:** `backend/runflow/umbrella.py` (note_style context), not directly by validation code currently

**What it writes:**
```python
# Writes to manifest["ai"]["packs"]["validation"]
{
    "base": str(base_path),          # e.g., "runs/SID/ai_packs/validation"
    "dir": str(base_path),           # alias for base
    "packs": str(packs_dir),         # e.g., "runs/SID/ai_packs/validation/packs"
    "packs_dir": str(packs_dir),     # alias for packs
    "results": str(results_dir),     # e.g., "runs/SID/ai_packs/validation/results"
    "results_dir": str(results_dir), # alias for results
    "index": str(index_path),        # e.g., "runs/SID/ai_packs/validation/index.json"
    "logs": str(log_path),           # e.g., "runs/SID/ai_packs/validation/logs.txt"
}

# Also writes to manifest["artifacts"]["ai"]["packs"]["validation"]
# (mirrors same keys)

# Also writes to manifest["ai"]["validation"]
{
    "base": str(base_path),
    "dir": str(base_path)
}

# Also writes to manifest["meta"]
{
    "validation_paths_initialized": True
}
```

**Lifecycle Stage:** Can be called anytime; currently NOT called in V2 validation flow (only used in note_style)

**Writer Type:** ✅ **Initial Injection** (if called early) or **Refresh** (if called later)

**Preservation:** Does NOT persist to disk itself; modifies in-memory dict only

---

#### Writer #2: `Manifest.ensure_validation_section`

**Location:** `backend/ai/manifest.py` lines 132-217  
**Called by:** 
- `_update_manifest_for_run` in `backend/ai/validation_builder.py` (line 2314)
- `build_validation_packs_for_run` directly (line 2462)

**What it writes:**
```python
# Writes to manifest["ai"]["packs"]["validation"]
{
    "base": str(validation_paths.base),
    "dir": str(validation_paths.base),
    "packs": str(validation_paths.packs_dir),
    "packs_dir": str(validation_paths.packs_dir),
    "results": str(validation_paths.results_dir),
    "results_dir": str(validation_paths.results_dir),
    "index": str(validation_paths.index_file),
    "logs": str(validation_paths.log_file)
}
```

**Persistence:** ✅ Calls `persist_manifest(manifest)` if any value changed

**Logs:**
- `VALIDATION_MANIFEST_INJECTED` (if changes made)
- `VALIDATION_MANIFEST_ALREADY_INITIALIZED` (if already populated)

**Lifecycle Stage:** Currently called at **T1** (inside builder) via `_update_manifest_for_run`

**Writer Type:** ✅ **Initial Injection** or **Idempotent Upsert**

**Preservation:** Only writes if current value is `None` or empty string; does not overwrite existing non-empty values

**Critical Behavior:**
```python
# Line 189-195
for key, value in canonical_values.items():
    current = validation_section.get(key)
    if not isinstance(current, str) or not current.strip():
        validation_section[key] = value  # ← Only writes if missing/empty
        changed = True
```

---

#### Writer #3: `RunManifest.upsert_validation_packs_dir`

**Location:** `backend/pipeline/runs.py` lines 1075-1110  
**Called by:**
- `_update_manifest_for_run` in `backend/ai/validation_builder.py` (line 2325)
- Legacy validation builder paths (if any)

**What it writes:**
```python
# Writes to manifest["ai"]["packs"]["validation"]
{
    "base": str(base_dir),
    "dir": str(base_dir),
    "packs": str(pack_path),
    "packs_dir": str(pack_path),
    "results": str(results_path),
    "results_dir": str(results_path),
    "index": str(index_path),
    "logs": str(log_path),
    "last_built_at": timestamp  # ← Timestamp of build
}

# Also writes to manifest["ai"]["validation"]
{
    "base": str(base_dir),
    "dir": str(base_dir),
    "accounts": str(base_dir),
    "accounts_dir": str(base_dir),
    "last_prepared_at": timestamp
}

# Also updates manifest["ai"]["status"]["validation"]
{
    "built": True,
    "sent": False,
    "completed_at": None
}
```

**Persistence:** ✅ Calls `self.save()` at end (line 1110)

**Lifecycle Stage:** Currently called at **T1** (builder phase)

**Writer Type:** **UPSERT** – Unconditionally overwrites all path fields + timestamps

**Preservation:** ❌ **DANGEROUS** – Does NOT check if values already exist; always overwrites

**Critical Issue:** This is the writer that can clobber values set earlier if called after other writers

---

#### Writer #4: `RunManifest._ensure_ai_section`

**Location:** `backend/pipeline/runs.py` lines 823-940  
**Called by:** Various manifest initialization paths

**What it writes:**
```python
# Creates default structure with all None values
{
    "base": None,
    "dir": None,
    "packs": None,
    "packs_dir": None,
    "results": None,
    "results_dir": None,
    "index": None,
    "last_built_at": None,
    "logs": None
}
```

**Persistence:** Does NOT persist; only ensures in-memory structure

**Lifecycle Stage:** Runs whenever manifest is accessed and section doesn't exist

**Writer Type:** **Default Initialization**

**Preservation:** Uses `.setdefault()`, so won't overwrite existing values

**Risk Level:** ⚠️ Low – Only creates empty structure if missing

---

### 1.2 Writers to `ai["validation"]` (Non-Packs Tree)

All writers to `ai.validation.*` are the same functions as above:

1. ✅ `ensure_validation_paths` (helper) writes `base` and `dir`
2. ✅ `RunManifest.upsert_validation_packs_dir` writes all fields
3. ⚠️ `RunManifest._ensure_ai_section` creates default `None` structure

**Critical Note:** `ai.validation.*` is largely **legacy**. Modern V2 validation code primarily uses `ai.packs.validation.*`.

**Fields:**
- `base` – Validation base directory (mirrors `ai.packs.validation.base`)
- `dir` – Alias for base
- `accounts` – Legacy field (unused in V2)
- `accounts_dir` – Legacy field (unused in V2)
- `last_prepared_at` – Timestamp (only written by `upsert_validation_packs_dir`)

---

### 1.3 Manifest Save / Persistence Behavior

#### Primary Save Method: `RunManifest.save()`

**Location:** `backend/pipeline/runs.py` lines 584-648

**Preservation Logic:**

```python
# Lines 584-648: Full preservation implementation
# Before save:
# 1. Load on-disk manifest (if exists)
# 2. Extract validation natives from on-disk version
# 3. Save in-memory manifest
# 4. If in-memory lost natives, restore from on-disk snapshot
# 5. Log preservation event
```

**Protected Sections:**
- `ai.packs.validation` (all keys)
- `ai.validation` (all keys)
- `artifacts.ai.packs.validation` (all keys)
- `meta.validation_paths_initialized` (boolean flag)

**Logs:**
- `MANIFEST_SAVE_BEFORE` (before persistence, shows current state)
- `MANIFEST_VALIDATION_NATIVE_PRESERVE` (if restoration occurred)

**Critical Safeguard:**
```python
# Lines 617-641
preserved = False
if isinstance(prior_ai_packs_val, Mapping) and not isinstance(current_ai_packs_val, Mapping):
    _set_section(self.data, ("ai", "packs", "validation"), dict(prior_ai_packs_val))
    preserved = True
# ... same for ai.validation and artifacts ...
```

**Upgrade & Mirroring:**
- Line 646: `_upgrade_ai_packs_structure()` – Handles merge paths upgrade
- Line 647: `_mirror_ai_to_legacy_artifacts()` – Copies `ai.packs.*` to `artifacts.ai.packs.*`

**Safety Assessment:** ✅ **STRONG** – Preservation logic prevents accidental loss during normal saves

---

#### Helper Save Method: `persist_manifest`

**Location:** `backend/pipeline/runs.py` (used by helpers)

**Behavior:**
- Wraps `RunManifest.save()`
- Uses same preservation logic

---

#### Unsafe Writers (Direct JSON Write)

**Search Result:** None found

**Critical:** All manifest writes go through `RunManifest.save()`, which has preservation logic

**Risk Level:** ✅ **LOW** – No bypass paths detected

---

### 1.4 Summary: Writers by Risk Level

| Risk | Writer | Location | Behavior | Current Stage |
|------|--------|----------|----------|---------------|
| 🟢 **Safe** | `ensure_validation_paths` | `backend/core/ai/paths.py:78` | Only writes if missing | Not used in V2 |
| 🟢 **Safe** | `Manifest.ensure_validation_section` | `backend/ai/manifest.py:132` | Only writes if empty; persists | T1 (builder) |
| 🟠 **Clobber** | `RunManifest.upsert_validation_packs_dir` | `backend/pipeline/runs.py:1075` | Unconditional overwrite | T1 (builder) |
| 🟢 **Safe** | `RunManifest._ensure_ai_section` | `backend/pipeline/runs.py:823` | Defaults only; no persist | Initialization |
| 🟢 **Safe** | `RunManifest.save` | `backend/pipeline/runs.py:584` | Preservation on save | All phases |

---

## 2. Complete Read-Map for Validation Native Paths

### 2.1 Readers of `ai["packs"]["validation"]`

#### Reader #1: `ValidationPackBuilder._resolve_manifest_paths`

**Location:** `backend/validation/build_packs.py` lines 763-826

**What it reads:**
```python
packs_dir = validation_section.get("packs_dir") or validation_section.get("packs")
results_dir = validation_section.get("results_dir") or validation_section.get("results")
index_path = validation_section.get("index")
log_path = validation_section.get("logs") or validation_section.get("log") or validation_section.get("log_file")
```

**Lifecycle Stage:** Legacy builder (not used in V2 orchestrator flow)

**Assumptions:**
- ❌ **HARD REQUIREMENT:** Raises `ValueError` if any path is missing
- Example: `"Manifest missing ai.packs.validation.packs_dir"`

**Impact of T0 Injection:** ✅ Positive – Ensures these fields exist before builder runs

---

#### Reader #2: `run_validation_send_for_sid_v2`

**Location:** `backend/ai/validation_sender_v2.py` lines 232-248

**What it reads:**
```python
packs_dir_str = validation_section.get("packs_dir") or validation_section.get("packs")
results_dir_str = validation_section.get("results_dir") or validation_section.get("results")
index_path_str = validation_section.get("index")
```

**Lifecycle Stage:** T2 (sender phase) in V2 orchestrator

**Assumptions:**
- ❌ **HARD REQUIREMENT:** Raises `RuntimeError` if any path is missing
- Logs: `VALIDATION_V2_PATHS_MISSING`

**Impact of T0 Injection:** ✅ Positive – V2 sender expects paths to exist in manifest

---

#### Reader #3: `debug_send_validation_for_sid` / `_get_validation_paths`

**Location:** `backend/debug/send_validation_for_sid.py` lines 35-51

**What it reads:**
```python
packs_dir = validation.get("packs_dir") or validation.get("packs")
results_dir = validation.get("results_dir") or validation.get("results")
index_path = validation.get("index")
```

**Lifecycle Stage:** Manual debug/repair tool

**Assumptions:**
- ⚠️ **SOFT REQUIREMENT:** Returns empty strings if missing
- Used for debugging missing-path scenarios

**Impact of T0 Injection:** ✅ Positive – Fewer debug scenarios needed

---

#### Reader #4: `extract_stage_manifest_paths`

**Location:** `backend/ai/manifest.py` lines 54-119

**What it reads:**
```python
base_dir = stage_section.get("base") or stage_section.get("dir")
packs_dir = stage_section.get("packs_dir") or stage_section.get("packs")
results_dir = stage_section.get("results_dir") or stage_section.get("results")
index_file = stage_section.get("index")
log_file = stage_section.get("logs")
```

**Lifecycle Stage:** Generic helper used by senders and orchestrators

**Assumptions:**
- ✅ **GRACEFUL:** Returns `None` for missing fields
- Caller decides how to handle missing paths

**Impact of T0 Injection:** ✅ Positive – Fewer `None` returns

---

#### Reader #5: `ValidationPackSender` (Legacy)

**Location:** `backend/validation/send_packs.py` (multiple functions)

**What it reads:**
- Uses `extract_stage_manifest_paths` or `_resolve_validation_stage_paths`
- Falls back to convention-based paths if manifest missing

**Lifecycle Stage:** Legacy sender (V1 path)

**Assumptions:**
- ⚠️ **FALLBACK LOGIC:** Can compute paths from SID even if manifest is missing
- Logs: `VALIDATION_PACKS_DIR_MISMATCH` if manifest/index disagree

**Impact of T0 Injection:** ✅ Positive – Reduces reliance on fallback logic

---

### 2.2 Readers of `ai["validation"]` (Non-Packs Tree)

#### Reader #1: `ensure_note_style_paths`

**Location:** `backend/core/ai/paths.py` lines 497-569

**What it reads:**
- Falls back to `ai.validation.dir` if note_style paths missing
- Legacy compatibility only

**Lifecycle Stage:** Note style initialization

**Assumptions:**
- ⚠️ **FALLBACK ONLY:** Not critical for validation

**Impact of T0 Injection:** Neutral – Validation doesn't rely on this

---

#### Reader #2: `upsert_validation_packs_dir` (Self-Read)

**Location:** `backend/pipeline/runs.py` lines 1075-1110

**What it reads:**
- Reads `_ensure_validation_section()` to get mutable dict

**Lifecycle Stage:** T1 (builder phase)

**Assumptions:**
- Creates section if missing

**Impact of T0 Injection:** ✅ Positive – Section already exists, no creation needed

---

### 2.3 Summary: Reader Dependencies

| Reader | Hard Requirement? | Stage | Fallback? | Impact if Missing |
|--------|-------------------|-------|-----------|-------------------|
| `ValidationPackBuilder` | ❌ Yes (raises error) | Legacy | None | Build fails |
| `run_validation_send_for_sid_v2` | ❌ Yes (raises error) | T2 (sender) | None | Send fails |
| `debug_send_validation_for_sid` | ⚠️ Soft (empty string) | Debug | Returns `""` | Debug incomplete |
| `extract_stage_manifest_paths` | ✅ No (returns None) | Generic | Returns `None` | Caller handles |
| `ValidationPackSender` | ⚠️ Soft (computes fallback) | Legacy | Convention | Logs mismatch |

**Critical Insight:** V2 orchestrator path **requires** natives to exist in manifest. Legacy paths have fallbacks.

---

## 3. Precise Pipeline Placement Timeline (Current Implementation)

### V2 Validation Flow for SID: `2613d93c-1e67-4714-ae21-fc235dfb929b`

```
┌─────────────────────────────────────────────────────────────────┐
│  T0: VALIDATION_V2_PIPELINE_ENTRY                               │
│  backend/api/tasks.py line 903                                  │
│                                                                 │
│  Entry point after requirements complete                        │
│  - Has: sid, runs_root                                          │
│  - Calls: build_validation_packs_for_run(sid, runs_root)       │
│  - Natives: ❌ NOT YET INJECTED                                │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│  T1: PACK BUILDER ENTRY                                         │
│  backend/ai/validation_builder.py line 2460                     │
│                                                                 │
│  Function: build_validation_packs_for_run()                    │
│  - Calls: ensure_validation_section() [line 2462]              │
│     ↳ This is first injection attempt (usually succeeds)      │
│  - Natives: ⚠️ FIRST INJECTION ATTEMPT                         │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│  T1+: MANIFEST UPDATE FOR RUN                                   │
│  backend/ai/validation_builder.py line 2300                     │
│                                                                 │
│  Function: _update_manifest_for_run()                          │
│  - Calls: ensure_validation_section() [line 2314]              │
│     ↳ Second attempt (idempotent, usually no-op)              │
│  - Calls: upsert_validation_packs_dir() [line 2325]            │
│     ↳ OVERWRITES all fields + timestamps                      │
│  - Natives: ✅ DEFINITELY INJECTED + PERSISTED                 │
│  - Logs: VALIDATION_PATHS_INITIALIZED                          │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│  T2: AUTOSEND (if enabled)                                      │
│  backend/ai/validation_builder.py line 2592                     │
│                                                                 │
│  Function: run_validation_send_for_sid_v2()                    │
│  - Reads: ai.packs.validation.{packs_dir, results_dir, index}  │
│  - Expects: Natives exist (raises if missing)                  │
│  - Writes: Results to results_dir                              │
│  - Natives: ✅ READ ONLY                                       │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│  T3: APPLY RESULTS (if autosend succeeded)                      │
│  backend/validation/apply_results_v2.py line 197                │
│                                                                 │
│  Function: apply_validation_results_for_sid()                  │
│  - Reads: Results from results_dir                             │
│  - Writes: Summaries to account JSONs                          │
│  - Natives: ✅ READ ONLY (via index)                           │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│  T4: RUNFLOW PROMOTION                                          │
│  backend/runflow/decider.py (refresh_validation_stage_*)        │
│                                                                 │
│  Function: refresh_validation_stage_from_index()               │
│  - Reads: Index to count packs/results                         │
│  - Writes: runflow.json stages.validation snapshot             │
│  - Natives: ✅ READ ONLY (via index)                           │
└─────────────────────────────────────────────────────────────────┘
```

### Critical Observations

1. **Current Injection Point:** T1 (inside builder)
2. **Problem:** If builder doesn't run, natives are never injected
3. **Risk Window:** Between T0 and T1, manifest has no natives
4. **Preservation Kicks In:** If any save happens between T1 and T2, preservation logic restores natives
5. **Actual Bug:** The `upsert_validation_packs_dir` at line 2325 unconditionally overwrites, creating a "last writer wins" scenario

---

## 4. How to Move Initial Injection to T0 (Design Proposal)

### 4.1 Candidate T0 Anchors

#### Option A: Right After `VALIDATION_V2_PIPELINE_ENTRY` Log

**Location:** `backend/api/tasks.py` line 904 (immediately after log)

**Pros:**
- ✅ Runs exactly once per validation V2 flow
- ✅ Has access to `sid` and `runs_root`
- ✅ Runs before any builder code
- ✅ Guarantees natives exist for entire pipeline

**Cons:**
- ⚠️ Requires editing tasks.py (high-traffic file)
- ⚠️ Must be careful not to affect legacy flows

**Placement Code:**
```python
# backend/api/tasks.py line 904 (after VALIDATION_V2_PIPELINE_ENTRY log)
log.info("VALIDATION_V2_PIPELINE_ENTRY sid=%s runs_root=%s", sid, runs_root)

# ✅ NEW: Inject validation natives at T0
from backend.core.ai.paths import ensure_validation_paths
manifest = RunManifest.load_or_create(Path(runs_root) / sid / "manifest.json", sid)
manifest.data = ensure_validation_paths(manifest.data, sid=sid, runs_root=runs_root)
manifest.save()
log.info("VALIDATION_T0_PATHS_INJECTED sid=%s", sid)

pack_results = build_validation_packs_for_run(sid, runs_root=runs_root)
```

**Safety:** ✅ Excellent – Isolated from rest of flow

---

#### Option B: At Start of `build_validation_packs_for_run`

**Location:** `backend/ai/validation_builder.py` line 2461 (before `ensure_validation_section` call)

**Pros:**
- ✅ Runs once per build
- ✅ Less impact on tasks.py
- ✅ Already has sid and runs_root

**Cons:**
- ⚠️ Still inside builder (not truly "T0")
- ⚠️ If builder is bypassed, natives won't exist
- ⚠️ Not as clean separation of concerns

**Placement Code:**
```python
# backend/ai/validation_builder.py line 2461
runs_root_path = Path(runs_root).resolve() if runs_root else Path("runs").resolve()

# ✅ NEW: Inject natives before any other validation work
from backend.core.ai.paths import ensure_validation_paths
manifest = RunManifest.load_or_create(runs_root_path / sid / "manifest.json", sid)
manifest.data = ensure_validation_paths(manifest.data, sid=sid, runs_root=runs_root_path)
manifest.save()

ensure_validation_section(sid, runs_root=runs_root_path)  # Now becomes no-op
```

**Safety:** ✅ Good – But not as clean as Option A

---

### 4.2 Recommended T0 Anchor: **Option A** (tasks.py)

**Rationale:**
1. True pipeline entry point
2. Runs exactly once
3. No dependencies on builder internals
4. Clean separation: paths first, then build

---

### 4.3 Which Existing Writers to Keep vs Remove

#### Writers to MOVE to T0:

| Current Writer | Move to T0? | New Role at T0 |
|----------------|-------------|----------------|
| `ensure_validation_paths` helper | ✅ **YES** | Initial injection of all path fields |

**Reasoning:** This helper is currently unused in V2; we repurpose it as the T0 injector.

---

#### Writers to SIMPLIFY after T0:

| Current Writer | After T0 Injection | Reason |
|----------------|-------------------|--------|
| `Manifest.ensure_validation_section` | ⚠️ Keep as **REFRESH** (updates timestamps only) | Still needed for last_built_at updates |
| `RunManifest.upsert_validation_packs_dir` | ⚠️ Keep but change to **UPDATE TIMESTAMPS ONLY** | Currently overwrites paths; should only touch timestamps after T0 |

**Reasoning:** After T0 injection, these functions should NOT touch path fields, only metadata like `last_built_at`.

---

#### Writers to KEEP as Backup:

| Current Writer | Keep? | Reason |
|----------------|-------|--------|
| `RunManifest._ensure_ai_section` | ✅ Yes | Creates default structure; harmless |
| `RunManifest.save` preservation logic | ✅ Yes | Critical failsafe |

---

### 4.4 Persistence & "Version of Truth" After T0

#### New Invariant

> **Once validation natives are injected at T0, they become IMMUTABLE for the lifetime of the run.**

**Enforcement Mechanisms:**

1. **T0 Injection:** First and only time paths are written
2. **Preservation on Save:** `RunManifest.save()` already prevents accidental removal
3. **Timestamp-Only Updates:** Post-T0 writers only touch `last_built_at`, `completed_at`, etc.

**Rules:**

```python
# Rule 1: On-disk natives always win over in-memory None
if on_disk has non-null paths and in-memory has None:
    restore from on-disk

# Rule 2: Only explicit non-null assignments can override
if in_memory has non-null path:
    allow update (but log warning if different from on-disk)

# Rule 3: Empty/None assignments are rejected
if in_memory path is None or empty string:
    reject update, keep on-disk value
```

**Implementation:** Already exists in `RunManifest.save()` lines 617-641.

---

## 5. Long-Term Impact & Risk Analysis

### 5.1 Legacy Validation V1

**Impact:** ⚠️ **LOW RISK**

**Analysis:**
- V1 uses `ValidationPackBuilder` which expects natives in manifest
- T0 injection would help V1 by ensuring natives exist earlier
- V1 flow doesn't use tasks.py entry point (uses direct builder call)

**Mitigation:**
- T0 injection in tasks.py only affects V2 production flow
- Legacy flows unaffected
- If V1 needs T0 injection, add separate call site

---

### 5.2 Merge / Note_Style / Other AI Stages

**Impact:** ✅ **NO RISK**

**Analysis:**
- Merge has its own `ai.packs.merge.*` section
- Note_style has its own `ai.packs.note_style.*` section
- No shared paths or cross-contamination

**Observation:**
- `ensure_validation_paths` (the T0 injector) only writes to validation-specific fields
- No overlap with other AI stages

---

### 5.3 Frontend / Review / Exports

**Impact:** ✅ **NO RISK**

**Analysis:**
- Frontend reads `ai.packs` to enumerate stages
- Having validation natives present earlier is positive (frontend can show validation stage sooner)
- No code relies on "null means not initialized" for validation specifically

**Observation:**
- Some code checks for non-null to determine if a stage is active
- T0 injection means validation is "active" from the start (correct behavior)

---

### 5.4 Config / Env Overrides

**Impact:** ⚠️ **MEDIUM RISK (but manageable)**

**Environment Variables:**
- `VALIDATION_PACKS_DIR`
- `VALIDATION_RESULTS_DIR`
- `VALIDATION_INDEX_PATH`

**Current Behavior:**
- `ensure_validation_paths` helper respects these env vars (line 95-109 in `backend/core/ai/paths.py`)
- Computes overridden paths at validation start

**Risk:**
- If env vars change between T0 and later stages, paths could be inconsistent

**Mitigation:**
- ✅ T0 injection reads env vars once and persists to manifest
- Later stages read from manifest (authoritative)
- Env vars only affect T0 injection, not later reads

**Recommendation:**
- Document that env overrides must be set before validation starts
- Log env var values at T0 for debugging

---

### 5.5 Concurrency / Multiple Workers

**Impact:** ⚠️ **LOW RISK**

**Scenario:**
- Two workers try to inject natives at T0 simultaneously

**Current Protection:**
- `RunManifest.save()` uses atomic file write (`safe_replace` with `os.replace`)
- Filesystem-level atomicity prevents corruption

**Outcome:**
- Both workers compute same paths (deterministic from sid + runs_root)
- Last write wins, but both write identical values
- No corruption

**Additional Protection:**
- `ensure_validation_paths` inside `ensure_validation_section` already checks if values exist (lines 189-195)
- If first worker wins, second worker's call becomes no-op

**Recommendation:**
- Log T0 injection timestamp to detect races
- Add `VALIDATION_T0_RACE_DETECTED` warning if injection happens twice

---

### 5.6 Summary: Risk Matrix

| Risk Category | Likelihood | Severity | Mitigation | Status |
|---------------|-----------|----------|------------|--------|
| Legacy V1 confusion | Low | Low | Separate call site if needed | ✅ Safe |
| Cross-stage contamination | None | N/A | Separate sections | ✅ Safe |
| Frontend assumes null = inactive | Low | Low | Frontend already handles non-null | ✅ Safe |
| Env var changes mid-run | Medium | Medium | Document "set before start" | ⚠️ Needs docs |
| Concurrent T0 injection | Low | Low | Atomic writes + idempotency | ✅ Safe |

---

## 6. Final Wiring Plan: T0–T4 Implementation

### Phase 1: T0 – Validation Entry (NEW)

**File:** `backend/api/tasks.py`  
**Function:** Inside `validation_build_packs` task  
**Line:** ~904 (right after `VALIDATION_V2_PIPELINE_ENTRY` log)

**Changes:**

```python
# BEFORE (current):
log.info("VALIDATION_V2_PIPELINE_ENTRY sid=%s runs_root=%s", sid, runs_root)
pack_results = build_validation_packs_for_run(sid, runs_root=runs_root)

# AFTER (proposed):
log.info("VALIDATION_V2_PIPELINE_ENTRY sid=%s runs_root=%s", sid, runs_root)

# T0: Inject validation natives at pipeline entry
from backend.core.ai.paths import ensure_validation_paths
from backend.pipeline.runs import RunManifest
manifest_path = Path(runs_root) / sid / "manifest.json"
manifest = RunManifest.load_or_create(manifest_path, sid)
manifest.data = ensure_validation_paths(manifest.data, sid=sid, runs_root=runs_root)
manifest.save()
log.info("VALIDATION_T0_PATHS_INJECTED sid=%s base=%s", sid, manifest.data["ai"]["packs"]["validation"]["base"])

pack_results = build_validation_packs_for_run(sid, runs_root=runs_root)
```

**Responsibilities:**
- Compute canonical paths from sid + runs_root
- Write all path fields to `ai.packs.validation`, `ai.validation`, `artifacts.ai.packs.validation`
- Set `meta.validation_paths_initialized = True`
- Persist to disk

**Fields Written:**
- `ai.packs.validation.{base, dir, packs, packs_dir, results, results_dir, index, logs}`
- `ai.validation.{base, dir}`
- `artifacts.ai.packs.validation.*` (mirror)
- `meta.validation_paths_initialized`

**Logs:**
- `VALIDATION_T0_PATHS_INJECTED`

---

### Phase 2: T1 – Builder (SIMPLIFIED)

**File:** `backend/ai/validation_builder.py`  
**Function:** `_update_manifest_for_run` (line 2300)

**Changes:**

```python
# BEFORE (current):
manifest = RunManifest.load_or_create(manifest_path, sid)
from backend.ai.manifest import Manifest as _AIManifest
_AIManifest.ensure_validation_section(sid, runs_root=runs_root_path)  # ← May inject
manifest.upsert_validation_packs_dir(...)  # ← Overwrites

# AFTER (proposed):
manifest = RunManifest.load_or_create(manifest_path, sid)
# T0 already injected paths; this is now just a timestamp update
manifest = _update_validation_timestamps_only(sid, runs_root_path, last_built_at=_now_iso())
```

**New Helper Function:**

```python
def _update_validation_timestamps_only(sid: str, runs_root: Path, last_built_at: str) -> None:
    """Update ONLY timestamps in validation sections, never touch path fields."""
    manifest_path = runs_root / sid / "manifest.json"
    manifest = RunManifest.load_or_create(manifest_path, sid)
    
    # Update timestamp only
    validation_packs = manifest.data.get("ai", {}).get("packs", {}).get("validation", {})
    if isinstance(validation_packs, dict):
        validation_packs["last_built_at"] = last_built_at
    
    # Update status
    validation_status = manifest.ensure_ai_stage_status("validation")
    validation_status["built"] = True
    validation_status["sent"] = False
    
    manifest.save()
    log.info("VALIDATION_TIMESTAMPS_UPDATED sid=%s last_built_at=%s", sid, last_built_at)
```

**Responsibilities:**
- Update `last_built_at` timestamp
- Update `ai.status.validation.built = True`
- DO NOT touch path fields

**Fields Written:**
- `ai.packs.validation.last_built_at` (timestamp only)
- `ai.status.validation.{built, sent}`

**Logs:**
- `VALIDATION_TIMESTAMPS_UPDATED`

---

### Phase 3: T2 – Sender (NO CHANGES NEEDED)

**File:** `backend/ai/validation_sender_v2.py`  
**Function:** `run_validation_send_for_sid_v2` (line 232)

**Current Behavior:**
- Reads paths from manifest
- Raises error if missing

**After T0 Injection:**
- Paths always exist (injected at T0)
- No changes needed

**Logs:**
- `VALIDATION_V2_INDEX_LOADED`

---

### Phase 4: T3 – Apply (NO CHANGES NEEDED)

**File:** `backend/validation/apply_results_v2.py`  
**Function:** `apply_validation_results_for_sid` (line 197)

**Current Behavior:**
- Reads results from index
- Applies to account summaries

**After T0 Injection:**
- No changes needed (paths already in manifest/index)

**Logs:**
- `VALIDATION_RESULTS_APPLIED`

---

### Phase 5: T4 – Runflow Promotion (NO CHANGES NEEDED)

**File:** `backend/runflow/decider.py`  
**Function:** `refresh_validation_stage_from_index`

**Current Behavior:**
- Reads index to count packs/results
- Writes runflow.json snapshot

**After T0 Injection:**
- No changes needed (index paths in manifest)

**Logs:**
- `VALIDATION_STAGE_PROMOTED`

---

### Summary Table: Wiring Plan

| Phase | File | Function | Change | Natives Status |
|-------|------|----------|--------|----------------|
| **T0** | `backend/api/tasks.py:904` | Validation task | ✅ **NEW**: Call `ensure_validation_paths` + persist | ✅ Injected |
| **T1** | `backend/ai/validation_builder.py:2300` | `_update_manifest_for_run` | ⚠️ **SIMPLIFY**: Only update timestamps | ✅ Read-only |
| **T2** | `backend/ai/validation_sender_v2.py:232` | `run_validation_send_for_sid_v2` | ✅ **NO CHANGE** | ✅ Read-only |
| **T3** | `backend/validation/apply_results_v2.py:197` | `apply_validation_results_for_sid` | ✅ **NO CHANGE** | ✅ Read-only |
| **T4** | `backend/runflow/decider.py` | `refresh_validation_stage_from_index` | ✅ **NO CHANGE** | ✅ Read-only |

---

## 7. Implementation Steps (Phased Rollout)

### Step 1: Add T0 Injection (Non-Breaking)

1. Edit `backend/api/tasks.py` line ~904
2. Add call to `ensure_validation_paths` + `manifest.save()`
3. Test on staging with real SIDs
4. Verify `VALIDATION_T0_PATHS_INJECTED` log appears

**Risk:** ✅ **LOW** – Does not remove existing writers yet

---

### Step 2: Simplify T1 Builder (After T0 Verified)

1. Edit `backend/ai/validation_builder.py` line 2300
2. Replace `ensure_validation_section` + `upsert_validation_packs_dir` with timestamp-only update
3. Test on staging
4. Verify natives still present in final manifest

**Risk:** ⚠️ **MEDIUM** – Removes redundant injection

---

### Step 3: Monitor Production

1. Deploy to production
2. Monitor logs for `VALIDATION_T0_PATHS_INJECTED`
3. Check sample SIDs to ensure natives persist
4. Watch for `MANIFEST_VALIDATION_NATIVE_PRESERVE` (should decrease over time)

**Risk:** ✅ **LOW** – Preservation logic is failsafe

---

### Step 4: Deprecate Legacy Writers (Future)

1. Once T0 injection is stable, mark `upsert_validation_packs_dir` as deprecated for validation
2. Update documentation
3. Consider removing in future major version

**Risk:** ✅ **LOW** – Long-term cleanup

---

## 8. Conclusion

### Summary of Findings

1. **Current State:** Validation natives injected at T1 (builder phase) by multiple writers
2. **Problem:** `upsert_validation_packs_dir` unconditionally overwrites, creating "last writer wins" issue
3. **Preservation Logic:** Strong failsafe in `RunManifest.save()` prevents loss, but doesn't prevent initial injection race

### Recommended Action

✅ **Move initial injection to T0** (tasks.py line ~904) using `ensure_validation_paths` helper.

**Benefits:**
- Paths exist from start of validation
- Eliminates race conditions
- Simplifies builder logic
- Reduces reliance on preservation as primary mechanism

**Risks:**
- ⚠️ Requires careful testing of T0 injection
- ⚠️ Must document env var behavior

**Next Steps:**
1. Review this design document
2. Implement T0 injection in tasks.py
3. Test on staging SIDs
4. Simplify T1 builder after verification
5. Deploy to production with monitoring

---

**End of Investigation Report**
