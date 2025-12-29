# Validation Chain-Only Orchestration Refactor — Design Document

**Date**: November 19, 2025  
**Author**: Investigation via GitHub Copilot  
**Status**: DESIGN ONLY — No Code Changes Yet  
**Related**: `VALIDATION_V2_VS_LEGACY_INVESTIGATION.md`

---

## Table of Contents

1. [Current State Analysis](#1-current-state-analysis)
2. [Target Architecture](#2-target-architecture)
3. [Specific Code Locations to Change](#3-specific-code-locations-to-change)
4. [Idempotency Guard Design](#4-idempotency-guard-design)
5. [Test & Observability Plan](#5-test--observability-plan)
6. [Rollout & Validation Strategy](#6-rollout--validation-strategy)
7. [Risk Assessment](#7-risk-assessment)

---

## 1. Current State Analysis

### 1.1 Fastpath Inside `stage_a_task` (Current Behavior)

**File**: `backend/api/tasks.py`  
**Function**: `stage_a_task(self, sid: str)` (lines 377-1140)

#### Current Flow (Exact Locations)

**Line 886-898: Validation Requirements**
```python
if ENABLE_VALIDATION_REQUIREMENTS:
    try:
        stats = run_validation_requirements_for_all_accounts(sid)
    except Exception:
        log.error("VALIDATION_REQUIREMENTS_PIPELINE_FAILED sid=%s", sid, exc_info=True)
    else:
        # ... log stats ...
```

**Line 900-980: Inline V2 Builder (PROBLEM AREA #1)**
```python
# V2: Build validation packs after requirements (triggers autosend in orchestrator mode)
try:
    from backend.ai.validation_builder import build_validation_packs_for_run
    from backend.runflow.decider import _compute_umbrella_barriers
    
    # ... merge readiness gate check (lines 922-934) ...
    barriers = _compute_umbrella_barriers(run_dir)
    merge_ready = barriers.get("merge_ready", False)
    if not merge_ready:
        log.info("VALIDATION_PACKS_DEFERRED sid=%s reason=merge_not_ready ...")
        return summary  # EXIT EARLY
    
    # ... T0 path injection (lines 945-982) ...
    
    pack_results = build_validation_packs_for_run(sid, runs_root=runs_root)  # ← V2 BUILDER CALLED INLINE
    packs_written = sum(len(entries or []) for entries in pack_results.values())
    log.info("VALIDATION_PACKS_BUILD_DONE sid=%s packs=%d", sid, packs_written)
except Exception:
    log.error("VALIDATION_PACKS_BUILD_FAILED sid=%s", sid, exc_info=True)
```

**Line 1116-1131: Auto-AI Chain Enqueue (PROBLEM AREA #2 - Race Condition)**
```python
if os.environ.get("ENABLE_AUTO_AI_PIPELINE", "1") in ("1", "true", "True"):
    if has_ai_merge_best_tags(sid):
        if sid in _AUTO_AI_PIPELINE_ENQUEUED:
            log.info("AUTO_AI_ALREADY_ENQUEUED sid=%s", sid)
        else:
            try:
                # ... manifest handling ...
                maybe_run_ai_pipeline_task.delay(sid)  # ← CELERY TASK (ASYNC)
            except Exception:
                log.error("AUTO_AI_ENQUEUE_FAILED sid=%s", sid, exc_info=True)
            else:
                _AUTO_AI_PIPELINE_ENQUEUED.add(sid)
                log.info("AUTO_AI_ENQUEUED sid=%s", sid)
```

#### Problems with Current Fastpath

1. **Dual Orchestration**: Validation packs built at line 900, chain enqueued at line 1131
   - By the time chain starts, packs may already exist
   - Leads to "work already done" or "nothing to do" when chain tasks run
   
2. **Race Condition**: Chain enqueued **after** `stage_a_task` completes
   - `maybe_run_ai_pipeline` checks `has_ai_merge_best_tags(sid)` → may return `False` if merge already consumed tags
   - Chain aborts before reaching validation tasks
   
3. **Autosend Dependency**: Whether validation completes depends on env flag `VALIDATION_AUTOSEND_ENABLED`
   - If OFF (default): packs built, nothing else → `validation_ready=false`
   - If ON: V2 sender runs inline, but no compact/chain continuation
   
4. **No Idempotency**: If fastpath runs packs, then chain runs, both try to do the same work

---

### 1.2 Auto-AI Chain Path (Current V2 Implementation)

**File**: `backend/pipeline/auto_ai_tasks.py`  
**Function**: `enqueue_auto_ai_chain(sid, runs_root)` (lines 2091-2123)

#### Chain Tasks (Validation-Relevant)

**Lines 1698-1787: `validation_build_packs`**
```python
@shared_task(bind=True, autoretry_for=(), retry_backoff=False)
def validation_build_packs(self, prev: Mapping[str, object] | None) -> dict[str, object]:
    payload = _ensure_payload(prev)
    sid = str(payload.get("sid") or "")
    
    _populate_common_paths(payload)
    runs_root = Path(payload["runs_root"])
    
    ensure_validation_section(sid, runs_root=runs_root)
    
    # Runflow short-circuit: if validation already succeeded, skip work
    try:
        from backend.pipeline.runs import get_stage_status
        status = get_stage_status(sid, stage="validation", runs_root=runs_root)
        if isinstance(status, str) and status.strip().lower() == "success":
            log.info("VALIDATION_BUILD_SHORT_CIRCUIT sid=%s reason=runflow_success ...")
            payload["validation_packs"] = 0
            payload["validation_short_circuit"] = True
            return payload
    except Exception:
        pass
    
    # Short-circuit if validation already completed via manifest
    try:
        manifest = RunManifest.for_sid(sid, runs_root=runs_root, allow_create=False)
        validation_status = manifest.get_ai_stage_status("validation")
        state = validation_status.get("state")
        if state == "success":
            log.info("VALIDATION_BUILD_SHORT_CIRCUIT sid=%s reason=already_success ...")
            return payload
    except Exception:
        pass
    
    logger.info("VALIDATION_STAGE_STARTED sid=%s", sid)
    
    try:
        results = build_validation_packs_for_run(sid, runs_root=runs_root)  # ✅ V2 BUILDER
    except Exception as exc:
        # ... error handling, runflow recording ...
        raise
    
    packs_written = sum(len(entries or []) for entries in results.values())
    payload["validation_packs"] = packs_written
    return payload
```

**Key Observations**:
- ✅ Already has short-circuit logic for `status="success"` (runflow + manifest)
- ⚠️ Does NOT check for "packs already built but not sent" scenario
- ✅ Calls V2 builder

---

**Lines 1790-1898: `validation_send`**
```python
@shared_task(bind=True, autoretry_for=(), retry_backoff=False)
def validation_send(self, prev: Mapping[str, object] | None) -> dict[str, object]:
    payload = _ensure_payload(prev)
    sid = str(payload.get("sid") or "")
    
    if _orchestrator_mode_enabled():  # ✅ DEFAULT: True
        logger.info("VALIDATION_ORCHESTRATOR_SEND_V2 sid=%s", sid)
        
        _populate_common_paths(payload)
        runs_root = Path(payload["runs_root"])
        
        # Use new clean sender (validation_sender_v2)
        try:
            from backend.ai.validation_sender_v2 import run_validation_send_for_sid_v2
            stats = run_validation_send_for_sid_v2(sid, runs_root)  # ✅ V2 SENDER
            logger.info("VALIDATION_ORCHESTRATOR_SEND_V2_DONE sid=%s ...")
            payload["validation_sent"] = True
            payload["validation_v2_stats"] = stats
        except Exception as exc:
            logger.exception("VALIDATION_ORCHESTRATOR_SEND_V2_FAILED sid=%s", sid)
            payload["validation_sent"] = False
            raise
        
        return payload
    
    # Legacy mode (VALIDATION_ORCHESTRATOR_MODE=0) — not relevant for us
    # ...
```

**Key Observations**:
- ✅ Uses V2 sender in orchestrator mode (default)
- ⚠️ Does NOT check if results already exist before sending
- ⚠️ V2 sender (`run_validation_send_for_sid_v2`) has some idempotency inside it (checks for existing results per pack), but task doesn't precheck

---

**Lines 1913-1976: `validation_compact`**
```python
@shared_task(bind=True, autoretry_for=(), retry_backoff=False)
def validation_compact(self, prev: Mapping[str, object] | None) -> dict[str, object]:
    payload = _ensure_payload(prev)
    sid = str(payload.get("sid") or "")
    
    _populate_common_paths(payload)
    runs_root = Path(payload["runs_root"])
    index_path = validation_index_path(sid, runs_root=runs_root, create=True)
    
    if not index_path.exists():
        log.info("VALIDATION_COMPACT_SKIP sid=%s reason=index_missing ...")
        return payload
    
    logger.info("VALIDATION_COMPACT_START sid=%s", sid)
    
    try:
        rewrite_index_to_canonical_layout(index_path, runs_root=runs_root)  # ✅ V2 COMPACT
    except Exception as exc:
        # ... error handling ...
        raise
    
    payload["validation_compacted"] = True
    return payload
```

**Key Observations**:
- ✅ Already skips if index missing
- ⚠️ Does NOT check if already compacted (could be made more explicit)
- ✅ Uses V2 compact function

---

**Lines 1002-1100: `validation_merge_ai_results_step`**
```python
@shared_task(bind=True, autoretry_for=(), retry_backoff=False)
def validation_merge_ai_results_step(self, prev: Mapping[str, object] | None) -> dict[str, object]:
    payload = _ensure_payload(prev)
    
    if _orchestrator_mode_enabled():  # ⚠️ SKIPPED IN ORCHESTRATOR MODE
        logger.info("VALIDATION_ORCHESTRATOR_MODE_SKIP validation_merge sid=%s", payload.get("sid"))
        return payload  # EXIT EARLY
    
    # Legacy apply path (only when orchestrator mode OFF)
    # ... apply_validation_ai_decisions_for_all_accounts ...
```

**Key Observations**:
- ✅ Skipped in orchestrator mode (V2 sender handles inline via `refresh_runflow_validation_stage`)
- ✅ No action needed for chain-only refactor

---

#### Chain Summary: Already Mostly V2, Needs Idempotency Improvements

| Task | V2 Function | Current Idempotency | Needs Improvement? |
|------|-------------|---------------------|-------------------|
| `validation_build_packs` | `build_validation_packs_for_run()` | ✅ Short-circuits on `status=success` | ⚠️ Should also check "packs exist, not sent" |
| `validation_send` | `run_validation_send_for_sid_v2()` | ⚠️ Sender has internal pack-level checks | ✅ Should precheck runflow before calling sender |
| `validation_compact` | `rewrite_index_to_canonical_layout()` | ⚠️ Skips if index missing | ✅ Should explicitly check "already compacted" |
| `validation_merge_ai_results_step` | Skipped (V2 sender handles) | N/A | ❌ No change needed |

**Confirmation**: Chain is **100% V2** when `VALIDATION_ORCHESTRATOR_MODE=1` (default).

---

### 1.3 Legacy Validation Orchestration (For Completeness)

**File**: `backend/ai/validation_builder.py`

#### Legacy Components

**Lines 1936-2020: `_maybe_send_validation_packs`**
```python
def _maybe_send_validation_packs(
    sid: str,
    runs_root: Path,
    *,
    stage: str = "validation",
    recheck: bool = False,
) -> None:
    # Quarantine legacy autosend orchestration in production environments.
    legacy_enabled = str(os.getenv("ENABLE_LEGACY_VALIDATION_ORCHESTRATION", "")).strip().lower() in {"1", "true", "yes", "on"}
    if not legacy_enabled:
        log.info("LEGACY_VALIDATION_ORCHESTRATION_DISABLED sid=%s path=%s", sid, "VALIDATION_BUILDER_AUTOSEND")
        return  # ← EXIT IMMEDIATELY
    
    # ... rest of legacy autosend logic ...
```

**Lines 1884-1928: `_schedule_validation_recheck`**
```python
def _schedule_validation_recheck(sid: str, runs_root: Path, stage: str) -> None:
    """
    Schedule a background recheck thread for validation autosend.
    
    LOOP DRIVER: This creates a daemon thread that sleeps 2-10 seconds (random delay),
    then calls _maybe_send_validation_packs(..., recheck=True).
    """
    # ... daemon thread logic ...
```

**Lines 2020-2170: `run_validation_send_for_sid`** (Legacy inline sender)
```python
def run_validation_send_for_sid(sid: str, runs_root: Path) -> dict[str, object]:
    """Send validation packs for a single ``sid`` and return sender stats.
    
    This mirrors the legacy autosend path's sending behavior without any
    orchestration side effects (no barrier reconciliation, no merges, no
    recheck scheduling). Always writes results under
    ``ai_packs/validation/results`` for the given ``sid``.
    """
    # ... legacy sender logic ...
    # Only used when VALIDATION_ORCHESTRATOR_MODE=0
```

#### Call Sites for Legacy Functions

**`_maybe_send_validation_packs`**:
- `backend/pipeline/auto_ai.py:684` (inside `_run_auto_ai_pipeline` — but gated by `ENABLE_LEGACY_VALIDATION_ORCHESTRATION`)

**`run_validation_send_for_sid`**:
- `backend/pipeline/auto_ai_tasks.py:1869` (inside `validation_send` task — only when `VALIDATION_ORCHESTRATOR_MODE=0`)

#### Status Confirmation

✅ **All legacy components are quarantined and disabled by default**:
- `ENABLE_LEGACY_VALIDATION_ORCHESTRATION` defaults to OFF
- Logs emit `LEGACY_VALIDATION_ORCHESTRATION_DISABLED` when hit
- No legacy calls in current production path

🚫 **We will NOT re-enable legacy as part of this refactor.**

---

## 2. Target Architecture

### 2.1 Design Principles

1. **Single Source of Truth**: Only the Auto-AI chain runs validation logic
2. **Fastpath is Pure Trigger**: `stage_a_task` decides **when** to enqueue chain, not **what** to run
3. **Idempotent Chain**: Every task in the chain checks runflow/disk and skips if work already done
4. **No Hidden Paths**: No inline validation execution outside the chain

### 2.2 Target Flow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│ stage_a_task (Trigger Only)                                     │
├─────────────────────────────────────────────────────────────────┤
│ 1. Build problem cases                                          │
│ 2. Run validation requirements (analyze summaries)              │
│ 3. Check merge gate:                                            │
│    - merge_ready=true?                                          │
│    - validation_required=true?                                  │
│ 4. IF gate passes:                                              │
│    → Call enqueue_auto_ai_chain(sid, runs_root)                │
│ 5. EXIT (no inline validation work)                             │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ Auto-AI Chain (All Work Happens Here)                           │
├─────────────────────────────────────────────────────────────────┤
│ 1. ai_score_step (merge scoring)                                │
│ 2. merge_build_packs (idempotent)                               │
│ 3. merge_send (idempotent)                                      │
│ 4. merge_compact (idempotent)                                   │
│ 5. run_date_convention_detector                                 │
│ 6. ai_validation_requirements_step                              │
│ 7. validation_build_packs (idempotent) ← Check runflow first   │
│ 8. validation_send (idempotent) ← Check pending packs first    │
│ 9. validation_compact (idempotent) ← Check if already compact  │
│ 10. validation_merge_ai_results_step (skipped in orch mode)    │
│ 11. strategy_planner_step                                       │
│ 12. ai_polarity_check_step                                      │
│ 13. ai_consistency_step                                         │
│ 14. pipeline_finalize                                           │
└─────────────────────────────────────────────────────────────────┘
```

### 2.3 Key Changes from Current State

| Component | Current Behavior | Target Behavior |
|-----------|------------------|-----------------|
| `stage_a_task` | Calls `build_validation_packs_for_run` inline | Only calls `enqueue_auto_ai_chain` (no inline work) |
| `validation_build_packs` | Short-circuits on `status=success` | Also checks "packs built, not sent" and skips |
| `validation_send` | Always tries to send | Prechecks runflow for "already sent" before calling sender |
| `validation_compact` | Skips if index missing | Explicitly checks "already compacted" flag in index/runflow |
| Chain invocation | Async, may race with fastpath | Synchronous gate check → enqueue (no race) |

---

## 3. Specific Code Locations to Change

### 3.1 Remove Inline Validation from `stage_a_task`

**File**: `backend/api/tasks.py`  
**Function**: `stage_a_task` (lines 377-1140)

#### Change #1: Remove V2 Builder Call (Lines 900-982)

**Current Code** (lines 900-982):
```python
# V2: Build validation packs after requirements (triggers autosend in orchestrator mode)
try:
    from backend.ai.validation_builder import build_validation_packs_for_run
    from backend.runflow.decider import _compute_umbrella_barriers
    
    runs_root = _ensure_manifest_root()
    # ... merge readiness gate ...
    barriers = _compute_umbrella_barriers(run_dir)
    merge_ready = barriers.get("merge_ready", False)
    if not merge_ready:
        log.info("VALIDATION_PACKS_DEFERRED sid=%s reason=merge_not_ready ...")
        return summary
    
    # ... T0 path injection ...
    # ... build_validation_packs_for_run call ...
    pack_results = build_validation_packs_for_run(sid, runs_root=runs_root)
    packs_written = sum(len(entries or []) for entries in pack_results.values())
    log.info("VALIDATION_PACKS_BUILD_DONE sid=%s packs=%d", sid, packs_written)
except Exception:
    log.error("VALIDATION_PACKS_BUILD_FAILED sid=%s", sid, exc_info=True)
```

**Target Code** (NEW):
```python
# Validation now orchestrated exclusively via Auto-AI chain.
# Check gate conditions and enqueue chain if ready.
try:
    from backend.runflow.decider import _compute_umbrella_barriers
    from backend.pipeline.auto_ai import maybe_queue_auto_ai_pipeline
    
    runs_root = _ensure_manifest_root()
    if runs_root is None:
        log.warning("VALIDATION_CHAIN_TRIGGER_SKIP sid=%s reason=no_runs_root", sid)
    else:
        run_dir = Path(runs_root) / sid
        barriers = _compute_umbrella_barriers(run_dir)
        merge_ready = barriers.get("merge_ready", False)
        validation_required = barriers.get("validation_required", True)  # Or compute from summary
        
        if merge_ready and validation_required:
            log.info(
                "VALIDATION_CHAIN_TRIGGER sid=%s merge_ready=%s validation_required=%s",
                sid, merge_ready, validation_required
            )
            # Enqueue chain — it will handle all validation work
            result = maybe_queue_auto_ai_pipeline(
                sid,
                runs_root=runs_root,
                flag_env=os.environ,
                force=False,
            )
            log.info("VALIDATION_CHAIN_ENQUEUED sid=%s result=%s", sid, result)
        else:
            log.info(
                "VALIDATION_CHAIN_DEFER sid=%s merge_ready=%s validation_required=%s",
                sid, merge_ready, validation_required
            )
except Exception:
    log.error("VALIDATION_CHAIN_TRIGGER_FAILED sid=%s", sid, exc_info=True)
```

**Action**: Replace lines 900-982 with gate check + chain enqueue only.

---

#### Change #2: Remove/Consolidate Chain Enqueue at End of Task (Lines 1116-1131)

**Current Code** (lines 1116-1131):
```python
if os.environ.get("ENABLE_AUTO_AI_PIPELINE", "1") in ("1", "true", "True"):
    if has_ai_merge_best_tags(sid):
        if sid in _AUTO_AI_PIPELINE_ENQUEUED:
            log.info("AUTO_AI_ALREADY_ENQUEUED sid=%s", sid)
        else:
            # ... enqueue maybe_run_ai_pipeline_task ...
```

**Problem**: This enqueues `maybe_run_ai_pipeline_task`, which internally calls `maybe_run_ai_pipeline` → `_run_auto_ai_pipeline`. That function:
- Runs merge scoring/building/sending
- Does NOT run validation (comments say "Validation + strategy now orchestrated only via enqueue_auto_ai_chain")
- Is a legacy path that predates the full chain

**Target Code**: Remove this block entirely (lines 1116-1131). Chain is now enqueued earlier at the validation gate check.

**Action**: Delete lines 1116-1131.

---

#### Summary of `stage_a_task` Changes

| Current Lines | Action | Replacement |
|---------------|--------|-------------|
| 900-982 | Replace | Gate check + `maybe_queue_auto_ai_pipeline` call (no inline validation work) |
| 1116-1131 | Delete | Chain already enqueued earlier; this is redundant legacy path |

**Net Effect**: `stage_a_task` becomes a pure trigger that decides when to enqueue the chain, but does no validation work itself.

---

### 3.2 Add Idempotency Guards to Chain Tasks

**File**: `backend/pipeline/auto_ai_tasks.py`

#### Change #3: Improve `merge_build_packs` Idempotency (Lines ~600-800)

**Current Behavior**: Has some checks for empty/already-done scenarios, but not comprehensive.

**Target Additions** (add at start of `merge_build_packs` task, after `_populate_common_paths`):

```python
# Idempotency check: skip if merge already complete
try:
    from backend.runflow.decider import get_runflow_snapshot
    snapshot = get_runflow_snapshot(sid, runs_root=runs_root_path)
    merge_stage = snapshot.get("stages", {}).get("merge", {})
    
    # Check all completion indicators
    status = merge_stage.get("status")
    merge_applied = merge_stage.get("merge_ai_applied", False)
    result_files = merge_stage.get("result_files", 0)
    expected = merge_stage.get("expected_packs", 0)
    
    if status == "success" and merge_applied and result_files > 0 and result_files >= expected:
        log.info(
            "MERGE_BUILD_IDEMPOTENT_SKIP sid=%s reason=already_complete status=%s applied=%s results=%s/%s",
            sid, status, merge_applied, result_files, expected
        )
        payload["merge_packs"] = 0
        payload["merge_skipped"] = True
        payload["skip_reason"] = "already_complete"
        return payload
except Exception:
    log.debug("MERGE_BUILD_IDEMPOTENT_CHECK_FAILED sid=%s", sid, exc_info=True)
    # Continue with normal flow on check failure
```

**File/Lines**: Insert after line ~620 (after `_populate_common_paths`)

---

#### Change #4: Improve `merge_send` Idempotency (Lines ~900-1100)

**Current Behavior**: Always attempts to send packs.

**Target Additions** (add at start of `merge_send` task):

```python
# Idempotency check: skip if merge already sent/applied
try:
    from backend.runflow.decider import get_runflow_snapshot
    snapshot = get_runflow_snapshot(sid, runs_root=runs_root_path)
    merge_stage = snapshot.get("stages", {}).get("merge", {})
    
    merge_applied = merge_stage.get("merge_ai_applied", False)
    result_files = merge_stage.get("result_files", 0)
    
    if merge_applied and result_files > 0:
        log.info(
            "MERGE_SEND_IDEMPOTENT_SKIP sid=%s reason=already_sent applied=%s results=%s",
            sid, merge_applied, result_files
        )
        payload["merge_sent"] = True
        payload["merge_skipped"] = True
        return payload
except Exception:
    log.debug("MERGE_SEND_IDEMPOTENT_CHECK_FAILED sid=%s", sid, exc_info=True)
```

**File/Lines**: Insert after line ~920 (after `_populate_common_paths` in `merge_send`)

---

#### Change #5: Improve `validation_build_packs` Idempotency (Lines 1698-1787)

**Current State**: Already has short-circuits for `status=success` (lines 1710-1745).

**Target Addition**: Add check for "packs exist but not sent" scenario.

Insert after existing short-circuit checks (around line 1745):

```python
# Additional idempotency: check if packs already built (not just validation complete)
try:
    from backend.core.ai.paths import validation_index_path
    index_path = validation_index_path(sid, runs_root=runs_root, create=False)
    
    if index_path.exists():
        # Index exists — check if packs are already built
        import json
        try:
            index_data = json.loads(index_path.read_text(encoding="utf-8"))
            packs_list = index_data.get("packs", [])
            if len(packs_list) > 0:
                log.info(
                    "VALIDATION_BUILD_IDEMPOTENT_SKIP sid=%s reason=packs_already_built count=%d",
                    sid, len(packs_list)
                )
                payload["validation_packs"] = len(packs_list)
                payload["validation_short_circuit"] = True
                return payload
        except (json.JSONDecodeError, OSError):
            pass  # Continue if index unreadable
except Exception:
    log.debug("VALIDATION_BUILD_IDEMPOTENT_PACKS_CHECK_FAILED sid=%s", sid, exc_info=True)
```

**File/Lines**: Insert after line 1745 (after existing short-circuits, before `VALIDATION_STAGE_STARTED` log)

---

#### Change #6: Add `validation_send` Idempotency (Lines 1790-1898)

**Current State**: No precheck — always calls V2 sender.

**Target Addition**: Check runflow before calling sender.

Insert at start of orchestrator mode branch (around line 1808):

```python
if _orchestrator_mode_enabled():
    logger.info("VALIDATION_ORCHESTRATOR_SEND_V2 sid=%s", sid)
    
    _populate_common_paths(payload)
    runs_root = Path(payload["runs_root"])
    
    # Idempotency check: skip if validation already sent
    try:
        from backend.runflow.decider import get_runflow_snapshot
        snapshot = get_runflow_snapshot(sid, runs_root=runs_root)
        validation_stage = snapshot.get("stages", {}).get("validation", {})
        
        status = validation_stage.get("status")
        results = validation_stage.get("results", {})
        results_total = results.get("results_total", 0)
        missing_results = results.get("missing_results", 1)
        
        if status == "success" and results_total > 0 and missing_results == 0:
            log.info(
                "VALIDATION_SEND_IDEMPOTENT_SKIP sid=%s reason=already_sent status=%s results=%s missing=%s",
                sid, status, results_total, missing_results
            )
            payload["validation_sent"] = True
            payload["validation_skipped"] = True
            return payload
    except Exception:
        log.debug("VALIDATION_SEND_IDEMPOTENT_CHECK_FAILED sid=%s", sid, exc_info=True)
    
    # Use new clean sender (validation_sender_v2)
    try:
        from backend.ai.validation_sender_v2 import run_validation_send_for_sid_v2
        # ... rest of existing send logic ...
```

**File/Lines**: Insert after line 1808 (inside `if _orchestrator_mode_enabled():` block, before sender call)

---

#### Change #7: Add `validation_compact` Idempotency (Lines 1913-1976)

**Current State**: Skips if index missing; no explicit "already compacted" check.

**Target Addition**: Check for compaction marker in index or runflow.

Insert after index existence check (around line 1925):

```python
if not index_path.exists():
    log.info("VALIDATION_COMPACT_SKIP sid=%s reason=index_missing ...")
    return payload

# Idempotency check: skip if already compacted
try:
    import json
    index_data = json.loads(index_path.read_text(encoding="utf-8"))
    compacted_flag = index_data.get("compacted", False)
    canonical_layout = index_data.get("canonical_layout", False)
    
    if compacted_flag or canonical_layout:
        log.info(
            "VALIDATION_COMPACT_IDEMPOTENT_SKIP sid=%s reason=already_compacted flag=%s canonical=%s",
            sid, compacted_flag, canonical_layout
        )
        payload["validation_compacted"] = True
        payload["validation_compact_skipped"] = True
        return payload
except (json.JSONDecodeError, OSError, KeyError):
    pass  # Continue if check fails
```

**File/Lines**: Insert after line 1925 (after index existence check, before `VALIDATION_COMPACT_START` log)

**Note**: The `rewrite_index_to_canonical_layout` function should also set a `"canonical_layout": true` flag in the index after compaction. This is a separate small change to `backend/validation/manifest.py` (not specified in current code, but needed for this check to work).

---

### 3.3 Summary of Chain Task Changes

| Task | File | Lines | Change Type | Guard Logic |
|------|------|-------|-------------|-------------|
| `merge_build_packs` | `auto_ai_tasks.py` | ~620 | Add idempotency guard | Check `status=success`, `merge_ai_applied=true`, `result_files >= expected_packs` |
| `merge_send` | `auto_ai_tasks.py` | ~920 | Add idempotency guard | Check `merge_ai_applied=true`, `result_files > 0` |
| `merge_compact` | `auto_ai_tasks.py` | N/A (already implicit) | No change | Already skips if results missing |
| `validation_build_packs` | `auto_ai_tasks.py` | 1745 | Add idempotency guard | Check index exists with `packs.length > 0` |
| `validation_send` | `auto_ai_tasks.py` | 1808 | Add idempotency guard | Check `status=success`, `results_total > 0`, `missing_results=0` |
| `validation_compact` | `auto_ai_tasks.py` | 1925 | Add idempotency guard | Check index has `compacted=true` or `canonical_layout=true` |

---

## 4. Idempotency Guard Design

### 4.1 Design Principles

1. **Check Before Acting**: Every task inspects runflow/disk state before doing work
2. **Fail-Safe Defaults**: If guard check fails (exception), proceed with work (defensive)
3. **Clear Logging**: Every skip emits a structured log with `reason=` and relevant state
4. **Use Existing Helpers**: Leverage `get_runflow_snapshot`, manifest helpers, path helpers

### 4.2 Runflow/Disk Inspection Patterns

#### Pattern 1: Check Runflow Stage Status

**Helper**: `backend/runflow/decider.get_runflow_snapshot(sid, runs_root)`

**Example**:
```python
from backend.runflow.decider import get_runflow_snapshot

snapshot = get_runflow_snapshot(sid, runs_root=runs_root)
merge_stage = snapshot.get("stages", {}).get("merge", {})

status = merge_stage.get("status")  # "success", "error", "built", etc
merge_applied = merge_stage.get("merge_ai_applied", False)
result_files = merge_stage.get("result_files", 0)

if status == "success" and merge_applied and result_files > 0:
    # Skip work
```

**Used By**: All merge and validation tasks

---

#### Pattern 2: Check Validation Index

**Helper**: `backend/core/ai/paths.validation_index_path(sid, runs_root, create=False)`

**Example**:
```python
from backend.core.ai.paths import validation_index_path
import json

index_path = validation_index_path(sid, runs_root=runs_root, create=False)
if index_path.exists():
    index_data = json.loads(index_path.read_text(encoding="utf-8"))
    packs_list = index_data.get("packs", [])
    if len(packs_list) > 0:
        # Packs already built
```

**Used By**: `validation_build_packs`, `validation_compact`

---

#### Pattern 3: Check Manifest Stage Status

**Helper**: `backend/pipeline/runs.RunManifest.for_sid(sid, runs_root, allow_create=False)`

**Example**:
```python
from backend.pipeline.runs import RunManifest

manifest = RunManifest.for_sid(sid, runs_root=runs_root, allow_create=False)
validation_status = manifest.get_ai_stage_status("validation")
state = validation_status.get("state")  # "success", "error", "pending", etc

if state == "success":
    # Validation already complete
```

**Used By**: `validation_build_packs` (already has this check)

---

### 4.3 Logging Standards for Skipped Steps

**Format**: All skip logs should follow this pattern:
```python
log.info(
    "<TASK>_IDEMPOTENT_SKIP sid=%s reason=<reason> <key_state>=<value> ...",
    sid, ...
)
```

**Examples**:

| Task | Log Message |
|------|-------------|
| `merge_build_packs` | `MERGE_BUILD_IDEMPOTENT_SKIP sid=... reason=already_complete status=success applied=true results=1/1` |
| `merge_send` | `MERGE_SEND_IDEMPOTENT_SKIP sid=... reason=already_sent applied=true results=1` |
| `validation_build_packs` | `VALIDATION_BUILD_IDEMPOTENT_SKIP sid=... reason=packs_already_built count=3` |
| `validation_send` | `VALIDATION_SEND_IDEMPOTENT_SKIP sid=... reason=already_sent status=success results=3 missing=0` |
| `validation_compact` | `VALIDATION_COMPACT_IDEMPOTENT_SKIP sid=... reason=already_compacted flag=true` |

**Grep Pattern**: `grep "_IDEMPOTENT_SKIP" logs/` will show all tasks that skipped due to idempotency.

---

### 4.4 Defensive Error Handling

**Every guard check must be wrapped in try/except**:
```python
try:
    # Guard logic: check runflow/disk
    if <already_done>:
        log.info("<TASK>_IDEMPOTENT_SKIP ...")
        return payload  # Skip work
except Exception:
    log.debug("<TASK>_IDEMPOTENT_CHECK_FAILED sid=%s", sid, exc_info=True)
    # Continue with normal flow — better to redo work than fail
```

**Rationale**: If guard check fails (e.g., runflow corrupt, index missing), we don't want to abort the chain. Better to redo work than leave validation incomplete.

---

## 5. Test & Observability Plan

### 5.1 Unit-Level Tests

#### Test File: `tests/backend/pipeline/test_auto_ai_tasks.py`

**Existing Coverage**: Chain task ordering, basic task execution

**New Tests to Add**:

1. **`test_validation_build_packs_idempotent_skip_when_packs_exist`**
   - Setup: Mock runflow with validation packs already built (index exists with packs)
   - Call: `validation_build_packs.run(payload)`
   - Assert: Returns early with `validation_short_circuit=True`, does NOT call `build_validation_packs_for_run`

2. **`test_validation_send_idempotent_skip_when_already_sent`**
   - Setup: Mock runflow with `validation.status=success`, `results_total > 0`, `missing_results=0`
   - Call: `validation_send.run(payload)`
   - Assert: Returns early with `validation_skipped=True`, does NOT call `run_validation_send_for_sid_v2`

3. **`test_validation_compact_idempotent_skip_when_already_compacted`**
   - Setup: Mock index with `compacted=true` flag
   - Call: `validation_compact.run(payload)`
   - Assert: Returns early with `validation_compact_skipped=True`, does NOT call `rewrite_index_to_canonical_layout`

4. **`test_merge_build_idempotent_skip_when_merge_complete`**
   - Setup: Mock runflow with `merge.status=success`, `merge_ai_applied=true`, `result_files >= expected_packs`
   - Call: `merge_build_packs.run(payload)`
   - Assert: Returns early with `merge_skipped=True`, does NOT call `_build_ai_packs`

5. **`test_merge_send_idempotent_skip_when_already_sent`**
   - Setup: Mock runflow with `merge_ai_applied=true`, `result_files > 0`
   - Call: `merge_send.run(payload)`
   - Assert: Returns early with `merge_skipped=True`, does NOT call `_send_ai_packs`

**Existing Tests to Update**:
- `test_enqueue_auto_ai_chain_orders_signatures`: No change needed (still tests task ordering)
- Any tests that mock/patch `build_validation_packs_for_run`: May need to mock runflow state to bypass new guards

---

#### Test File: `tests/backend/api/test_stage_a_task.py` (or similar)

**New Test to Add**:

1. **`test_stage_a_task_enqueues_chain_when_merge_ready`**
   - Setup: Mock `_compute_umbrella_barriers` to return `merge_ready=true`, `validation_required=true`
   - Mock `maybe_queue_auto_ai_pipeline` to track calls
   - Call: `stage_a_task(sid)`
   - Assert: 
     - `maybe_queue_auto_ai_pipeline` called with `sid`
     - `build_validation_packs_for_run` NOT called (removed from fastpath)

2. **`test_stage_a_task_defers_chain_when_merge_not_ready`**
   - Setup: Mock `_compute_umbrella_barriers` to return `merge_ready=false`
   - Call: `stage_a_task(sid)`
   - Assert: `maybe_queue_auto_ai_pipeline` NOT called

---

### 5.2 Integration-Style Tests or Fixtures

#### Fixture 1: Merge Complete, No Validation Yet

**Scenario**: SID with merge complete (`merge_ready=true`, `merge_ai_applied=true`), but no validation stage.

**Setup**:
```python
def test_chain_runs_validation_after_merge_complete(temp_run_dir):
    sid = "test-sid-001"
    runs_root = temp_run_dir
    
    # Setup: Create runflow with merge complete
    runflow_data = {
        "sid": sid,
        "stages": {
            "merge": {
                "status": "success",
                "merge_ai_applied": True,
                "result_files": 1,
                "expected_packs": 1,
            }
            # No validation stage
        },
        "umbrella_barriers": {
            "merge_ready": True,
            "validation_ready": False,
        }
    }
    _write_runflow(runs_root, sid, runflow_data)
    
    # Create merge results on disk
    _create_merge_results(runs_root, sid, num_results=1)
    
    # Call chain
    task_id = auto_ai_tasks.enqueue_auto_ai_chain(sid, runs_root=runs_root)
    
    # Wait for chain to complete (in test environment, use synchronous execution)
    # ...
    
    # Assert: Validation stage now exists
    final_runflow = _load_runflow(runs_root, sid)
    assert "validation" in final_runflow["stages"]
    assert final_runflow["umbrella_barriers"]["validation_ready"] == True
```

---

#### Fixture 2: Merge + Validation Both Complete

**Scenario**: SID with both merge and validation complete. Chain re-run should skip both.

**Setup**:
```python
def test_chain_idempotent_when_validation_already_complete(temp_run_dir):
    sid = "test-sid-002"
    runs_root = temp_run_dir
    
    # Setup: Create runflow with merge + validation complete
    runflow_data = {
        "sid": sid,
        "stages": {
            "merge": {
                "status": "success",
                "merge_ai_applied": True,
                "result_files": 1,
            },
            "validation": {
                "status": "success",
                "results": {
                    "results_total": 3,
                    "missing_results": 0,
                }
            }
        },
        "umbrella_barriers": {
            "merge_ready": True,
            "validation_ready": True,
        }
    }
    _write_runflow(runs_root, sid, runflow_data)
    
    # Create merge + validation results on disk
    _create_merge_results(runs_root, sid, num_results=1)
    _create_validation_results(runs_root, sid, num_results=3)
    
    # Call chain (second time)
    with patch("backend.pipeline.auto_ai_tasks.build_validation_packs_for_run") as mock_build:
        with patch("backend.ai.validation_sender_v2.run_validation_send_for_sid_v2") as mock_send:
            task_id = auto_ai_tasks.enqueue_auto_ai_chain(sid, runs_root=runs_root)
            # Wait for chain...
            
            # Assert: Builder and sender NOT called (skipped due to idempotency)
            mock_build.assert_not_called()
            mock_send.assert_not_called()
    
    # Assert: Runflow unchanged
    final_runflow = _load_runflow(runs_root, sid)
    assert final_runflow["stages"]["validation"]["status"] == "success"
```

---

### 5.3 Logging & Observability Improvements

#### New Log Messages to Add

**1. Chain Trigger from `stage_a_task`** (in `tasks.py`):
```python
log.info(
    "VALIDATION_CHAIN_TRIGGER sid=%s merge_ready=%s validation_required=%s",
    sid, merge_ready, validation_required
)
```
**Grep**: `grep "VALIDATION_CHAIN_TRIGGER" logs/`

---

**2. Chain Enqueued Successfully** (in `tasks.py`):
```python
log.info("VALIDATION_CHAIN_ENQUEUED sid=%s result=%s", sid, result)
```
**Grep**: `grep "VALIDATION_CHAIN_ENQUEUED" logs/`

---

**3. Chain Deferred (Gate Not Passed)** (in `tasks.py`):
```python
log.info(
    "VALIDATION_CHAIN_DEFER sid=%s merge_ready=%s validation_required=%s",
    sid, merge_ready, validation_required
)
```
**Grep**: `grep "VALIDATION_CHAIN_DEFER" logs/`

---

**4. Idempotent Skip Logs** (in all chain tasks):
```python
log.info("<TASK>_IDEMPOTENT_SKIP sid=%s reason=<reason> <state>", sid, ...)
```
**Grep**: `grep "_IDEMPOTENT_SKIP" logs/`

---

**5. Chain Entry Point** (already exists in `auto_ai_tasks.py:2097`):
```python
logger.info("AUTO_AI_CHAIN_START sid=%s runs_root=%s", sid, runs_root_value)
```
**Grep**: `grep "AUTO_AI_CHAIN_START" logs/`

---

#### Observability Dashboard Queries

**Query 1: How many chains triggered for validation?**
```bash
grep "VALIDATION_CHAIN_TRIGGER" logs/ | wc -l
```

**Query 2: How many validation steps were skipped (idempotent)?**
```bash
grep "VALIDATION_.*_IDEMPOTENT_SKIP" logs/ | wc -l
```

**Query 3: Which SIDs had validation deferred?**
```bash
grep "VALIDATION_CHAIN_DEFER" logs/ | awk '{print $NF}'
```

**Query 4: Chain start vs validation completion correlation**
```bash
# Compare AUTO_AI_CHAIN_START count to VALIDATION_STAGE_PROMOTED count
grep "AUTO_AI_CHAIN_START" logs/ | wc -l
grep "VALIDATION_STAGE_PROMOTED" logs/ | wc -l
```

---

## 6. Rollout & Validation Strategy

### 6.1 Phased Rollout Plan

#### Phase 1: Add Idempotency Guards (Low Risk)
**Changes**: Add all idempotency checks to chain tasks (Changes #3-7)  
**Risk**: Low — guards are defensive (fail to normal flow)  
**Testing**: Unit tests + manual chain re-run on test SID  
**Validation**: Grep logs for `_IDEMPOTENT_SKIP` messages  
**Duration**: 1-2 days

---

#### Phase 2: Remove Fastpath Validation (Medium Risk)
**Changes**: Remove `build_validation_packs_for_run` call from `stage_a_task` (Change #1)  
**Risk**: Medium — validation now depends 100% on chain  
**Testing**: Integration tests + test SIDs in staging  
**Validation**: 
- Confirm `VALIDATION_CHAIN_TRIGGER` logs appear
- Confirm validation completes for test SIDs
- Monitor for regressions (SIDs stuck with `validation_ready=false`)
**Duration**: 2-3 days

---

#### Phase 3: Consolidate Chain Enqueue (Low Risk)
**Changes**: Remove duplicate chain enqueue at end of `stage_a_task` (Change #2)  
**Risk**: Low — redundant code removal  
**Testing**: Verify chain still enqueued once per SID  
**Validation**: Grep logs for `AUTO_AI_CHAIN_START`, confirm single entry per SID  
**Duration**: 1 day

---

### 6.2 Rollback Plan

**If validation stops working**:
1. **Immediate**: Re-enable `VALIDATION_AUTOSEND_ENABLED=1` in `.env` (restores fastpath autosend)
2. **Short-term**: Revert Changes #1-2 (restore inline `build_validation_packs_for_run` call)
3. **Root Cause**: Check logs for chain enqueue failures, gate check issues

**Rollback Triggers**:
- >10% increase in SIDs with `validation_ready=false` after merge complete
- Chain enqueue failures (`VALIDATION_CHAIN_TRIGGER_FAILED` logs)
- Validation stage never promoted (no `VALIDATION_STAGE_PROMOTED` logs)

---

### 6.3 Success Metrics

**Before Refactor** (Baseline):
- % SIDs with `validation_ready=true` after merge complete: ~X%
- Avg time from `merge_ready=true` to `validation_ready=true`: Y minutes
- Validation autosend dependency: `VALIDATION_AUTOSEND_ENABLED` flag required

**After Refactor** (Target):
- % SIDs with `validation_ready=true` after merge complete: ≥X% (no regression)
- Avg time: ≤Y minutes (no slowdown)
- No dependency on autosend flag (chain always runs)
- Idempotent re-runs: Chain can be safely re-enqueued without duplicate work

**Monitoring Dashboard**:
- Track `VALIDATION_CHAIN_TRIGGER` count (should match merge completion count)
- Track `_IDEMPOTENT_SKIP` count (shows how often chain skips redundant work)
- Alert if `validation_ready=false` persists >30min after `merge_ready=true`

---

## 7. Risk Assessment

### 7.1 High-Risk Areas

#### Risk #1: Chain Never Enqueued
**Scenario**: Gate check logic in `stage_a_task` has bug → chain never called  
**Impact**: Validation stops entirely for all new SIDs  
**Mitigation**:
- Comprehensive unit tests for gate check logic
- Staging deployment with test SIDs before production
- Fallback flag to re-enable fastpath if needed

**Detection**: Monitor for absence of `VALIDATION_CHAIN_TRIGGER` logs

---

#### Risk #2: Idempotency Guards Too Aggressive
**Scenario**: Guards skip work when it's actually needed (false positive)  
**Impact**: Validation incomplete, `validation_ready=false` despite chain running  
**Mitigation**:
- Defensive guard design (fail to "do work" if check fails)
- Explicit logging for every skip decision
- Test fixtures for edge cases (e.g., packs built but results missing)

**Detection**: SIDs stuck in limbo (merge complete, validation never completes, no error logs)

---

#### Risk #3: Runflow/Manifest State Inconsistency
**Scenario**: Runflow says "complete", but disk files missing → guards skip, work not done  
**Impact**: Validation marked complete but results unavailable  
**Mitigation**:
- Guards check BOTH runflow AND disk state (e.g., index exists + has packs)
- Fallback: If disk check fails, proceed with work (defensive)
- Separate validation step to reconcile runflow vs disk

**Detection**: Grep for `_IDEMPOTENT_CHECK_FAILED` (guard check exceptions)

---

### 7.2 Medium-Risk Areas

#### Risk #4: Chain Takes Too Long
**Scenario**: Chain runs full merge + validation, slowing down overall pipeline  
**Impact**: Longer time from merge complete to validation ready  
**Mitigation**:
- Idempotency guards ensure only missing work is done (if merge already complete, merge tasks skip)
- Monitor chain execution time before/after refactor

**Detection**: Track time from `AUTO_AI_CHAIN_START` to `VALIDATION_STAGE_PROMOTED`

---

#### Risk #5: Celery Queue Congestion
**Scenario**: Enqueuing full chain for every SID increases Celery load  
**Impact**: Slower task processing, potential queue backlog  
**Mitigation**:
- Idempotency means re-enqueued chains are fast (mostly skips)
- Monitor Celery queue depth and task latency

**Detection**: Celery monitoring dashboard (queue length, task wait time)

---

### 7.3 Low-Risk Areas

#### Risk #6: Log Volume Increase
**Scenario**: New idempotency logs increase log volume  
**Impact**: Slightly higher log storage/processing cost  
**Mitigation**: Structured logging with clear prefixes for easy filtering  
**Detection**: Monitor log ingestion rate

---

## 8. Implementation Checklist

### 8.1 Code Changes

- [ ] **Change #1**: Remove `build_validation_packs_for_run` call from `stage_a_task` (lines 900-982)
  - [ ] Replace with gate check + `maybe_queue_auto_ai_pipeline` call
  - [ ] Add `VALIDATION_CHAIN_TRIGGER` / `VALIDATION_CHAIN_DEFER` logs
- [ ] **Change #2**: Remove duplicate chain enqueue at end of `stage_a_task` (lines 1116-1131)
- [ ] **Change #3**: Add idempotency guard to `merge_build_packs` (line ~620)
- [ ] **Change #4**: Add idempotency guard to `merge_send` (line ~920)
- [ ] **Change #5**: Add idempotency guard to `validation_build_packs` (line 1745)
- [ ] **Change #6**: Add idempotency guard to `validation_send` (line 1808)
- [ ] **Change #7**: Add idempotency guard to `validation_compact` (line 1925)
- [ ] **Bonus**: Add `canonical_layout=true` flag to validation index after compaction (`backend/validation/manifest.py`)

---

### 8.2 Testing

- [ ] Unit tests for idempotency guards (5 new tests in `test_auto_ai_tasks.py`)
- [ ] Unit tests for chain trigger logic (2 new tests in `test_stage_a_task.py`)
- [ ] Integration fixture: Merge complete, no validation → chain runs validation
- [ ] Integration fixture: Merge + validation complete → chain skips both
- [ ] Manual test: Re-run chain on same SID twice, verify idempotency logs

---

### 8.3 Observability

- [ ] Add `VALIDATION_CHAIN_TRIGGER` log to `stage_a_task`
- [ ] Add `VALIDATION_CHAIN_ENQUEUED` log to `stage_a_task`
- [ ] Add `VALIDATION_CHAIN_DEFER` log to `stage_a_task`
- [ ] Add `<TASK>_IDEMPOTENT_SKIP` logs to all chain tasks
- [ ] Set up dashboard query for `_IDEMPOTENT_SKIP` count
- [ ] Set up alert for SIDs stuck with `validation_ready=false` >30min

---

### 8.4 Rollout

- [ ] Phase 1: Deploy idempotency guards to staging
- [ ] Validate Phase 1: Test chain re-run, check skip logs
- [ ] Phase 2: Deploy fastpath removal to staging
- [ ] Validate Phase 2: Test new SIDs, confirm validation completes
- [ ] Phase 3: Deploy consolidation to staging
- [ ] Production deployment (all phases together after staging validation)

---

## 9. Open Questions & Future Work

### Open Questions
1. **Q**: Should we keep `VALIDATION_AUTOSEND_ENABLED` flag as a fallback escape hatch?  
   **A**: Recommend YES for Phase 2 rollout, remove later once confident in chain-only approach.

2. **Q**: What happens if chain is enqueued twice (e.g., manual retry)?  
   **A**: Idempotency guards ensure second run is safe (mostly skips). No duplicate work.

3. **Q**: Should we add a "chain already running" check before enqueue?  
   **A**: Could check `_AUTO_AI_PIPELINE_ENQUEUED` set or Celery task state, but idempotency makes this less critical.

### Future Work
1. **Monitoring**: Build Grafana dashboard for chain execution metrics
2. **Recovery**: Add admin endpoint to manually trigger chain for stuck SIDs
3. **Optimization**: If idempotency logs show frequent skips, consider caching runflow state in Redis
4. **Documentation**: Update runflow state machine diagram with chain-only flow

---

## 10. Conclusion

This refactor moves validation orchestration to a **single source of truth** (the Auto-AI chain), eliminating the fragile fastpath logic that caused validation to stop for SID `2d125dee`.

**Key Benefits**:
- ✅ **No more race conditions**: Chain enqueued synchronously after gate check
- ✅ **Idempotent by design**: Safe to re-run chain multiple times
- ✅ **Single orchestration path**: No hidden fastpath, easier to debug
- ✅ **No autosend flag dependency**: Validation always runs when merge completes

**Implementation Effort**:
- **Code changes**: ~7 targeted edits across 2 files
- **Testing**: ~7 new unit tests + 2 integration fixtures
- **Rollout**: 3 phases over ~1 week (staging + production)

**Risk**: Medium overall — careful phasing and comprehensive testing mitigate most risks.

---

**Next Steps**: Review this design with team, then proceed to implementation (create feature branch, apply changes, run tests).

---

**End of Design Document**
