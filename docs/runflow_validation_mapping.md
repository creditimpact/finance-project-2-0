**Overview**
- Scope: Map runflow writers that update the validation stage, reconstruct the event timeline for `126426ed-1a2d-4bc6-84ba-b0a9613042e1`, and compare manifest vs runflow fields to identify why runflow shows validation incomplete while the manifest shows completed and applied.

**Writers/Plugs**
- **`backend/runflow/decider.py:record_stage`**: Primary writer that persists a stage snapshot to `runs/<sid>/runflow.json`. Inputs: explicit `status`, `counts`, optional `metrics` and `results` from the caller, plus on-disk counts via `stage_counts` helpers. Side effects: merges into existing snapshot, calls `_apply_validation_stage_promotion()` (validation, merge, frontend, note_style), writes `runflow.json`, then emits an end event via `backend.core.runflow.runflow_end_stage`. Classification: main pipeline (V2-aware through the promotion function).
- **`backend/runflow/decider.py:_apply_validation_stage_promotion`**: Internal promotion used by `record_stage`, `refresh_validation_stage_from_index`, and `reconcile_umbrella_barriers`. Inputs: on-disk validation index (`ai_packs/validation/index.json`) and manifest (`manifest.json`). Updates: `stages.validation` status/summary/metrics/results. Sets `metrics.validation_ai_required`, `metrics.validation_ai_completed`, `metrics.validation_ai_applied` (from manifest), and marks top-level `validation_ai_completed`. Classification: main pipeline (V2-aware; trusts manifest for validation completion/applied flags).
- **`backend/runflow/decider.py:refresh_validation_stage_from_index`**: Validation-specific writer that runs `_apply_validation_stage_promotion`, merges, and persists the snapshot; triggers umbrella reconciliation. Inputs: on-disk index + manifest. Classification: main pipeline (V2-aware).
- **`backend/runflow/decider.py:reconcile_umbrella_barriers`**: Recomputes umbrella readiness and persists `umbrella_barriers`. Also invokes `_apply_validation_stage_promotion` before computing barriers, thus can refresh `stages.validation`. Inputs: on-disk index + manifest. Classification: main pipeline (V2-aware).
- **`backend/runflow/decider.py:_maybe_enqueue_validation_fastpath`**: Mutates in-memory snapshot to mark `validation` sent/in_progress and seed a few metrics (e.g., `merge_zero_packs`); enqueues Celery fastpath tasks. Persisted when the caller subsequently writes the snapshot. Classification: auxiliary/compat (prepares stage state; not authoritative for completion).
- **`backend/runflow/decider.py:_watchdog_trigger_validation_fastpath`**: Similar to fastpath enqueue; updates validation stage to in-progress and enqueues when watchdog detects staleness. Persisted by the caller. Classification: auxiliary/compat.
- **`backend/runflow/decider.py:record_stage_force`**: Generic snapshot writer used by `decide_next` to persist a mutated snapshot; not validation-specific but can update `stages.validation` if present. Classification: generic writer (V2-aware only insofar as the provided snapshot contains those fields).
- Additional emissions: `backend/core/runflow.*` record step/events (`runflow_step`, `runflow_end_stage`), counters (`record_validation_build_summary`, `record_validation_results_summary`), and umbrella reconciliation events. These do not directly compute validation completion; they reflect counters already present or computed in `decider.py`.

**Writers Table (Validation-Related)**

| Plug | File:Function | Writes validation? | Legacy-only? | Runs in V2 pipeline? | Trigger/When | Overwrite risk post-apply |
| - | - | - | - | - | - | - |
| record_stage | backend/runflow/decider.py:record_stage | Yes | No | Yes | Any stage write; always runs stage promotions | Low: calls `_apply_validation_stage_promotion`; status uses priority ordering; metrics/summary merged with promotion semantics |
| validation promotion | backend/runflow/decider.py:_apply_validation_stage_promotion | Yes (authoritative) | No | Yes | Called from record_stage, refresh_validation_stage_from_index, reconcile_umbrella_barriers | None: sets status=success, results from index, V2 flags from manifest; marks `_writer`=validation_promotion to make metrics/summary authoritative |
| refresh_validation_stage_from_index | backend/runflow/decider.py:refresh_validation_stage_from_index | Yes | No | Callable and intended for V2 | Explicit refresh (post-results/apply) | None: uses promotion; persists success/results; triggers barriers refresh |
| reconcile_umbrella_barriers | backend/runflow/decider.py:reconcile_umbrella_barriers | Indirect (runs promotion first) | No | Yes | Periodic/explicit barriers reconciliation | None: promotion keeps validation at success with correct results |
| record_stage_force | backend/runflow/decider.py:record_stage_force | Yes (merges provided snapshot) | No | Yes | Used by decide_next to persist mutated snapshot | Low: does not itself recompute validation; but subsequent reconcile/promotion restores correctness |
| decide_next (mutations) | backend/runflow/decider.py:decide_next | May mark validation sent/in_progress in zero-packs fastpath | No | Yes | Early zero-packs path before send | None after apply: path gated; won’t run post-success |
| fastpath enqueue | backend/runflow/decider.py:_maybe_enqueue_validation_fastpath | Sets validation in_progress, seeds metrics | No | Yes (when zero-packs) | During merge-zero fastpath before results | None post-apply: only runs when not completed; does not set results to 0 |
| watchdog fastpath | backend/runflow/decider.py:_watchdog_trigger_validation_fastpath | Sets validation in_progress, seeds metrics | No | Yes (on staleness) | Only when stale lock/events and non-terminal | None post-apply: gated by status; won’t regress |
| events/steps | backend/core/runflow: runflow_end_stage, record_validation_* | No (events/counters only) | N/A | Yes | Telemetry/aggregates | N/A (do not write runflow.json stages) |

**SID Case Study: 126426ed-1a2d-4bc6-84ba-b0a9613042e1**
- Artifacts observed:
  - `runs/<sid>/manifest.json` shows:
    - `ai.packs.validation.*` paths populated
    - `ai.status.validation`: `built: true`, `sent: true`, `applied: true`, `results_total: 3`, `results_applied: 3`, `results_apply_ok: true`, `validation_ai_applied: true`
  - `runs/<sid>/runflow.json` shows (at 17:10:50Z):
    - `run_state: VALIDATING`
    - `stages.validation.status: built`
    - `stages.validation.metrics.validation_ai_required: true`
    - `stages.validation.metrics.validation_ai_completed: false`
    - `stages.validation.results.results_total: 0`
    - `umbrella_barriers.validation_ready: false`
    - `last_writer: record_stage`

- Timeline (from `runs/<sid>/runflow_events.jsonl`):
  - 17:10:28Z: merge scoring steps; zero packs (`merge_zero_packs: true`).
  - 17:10:28Z: validation stage start and initial end (success) with `ai_packs_built: 0` (pre-build info snapshot).
  - 17:10:28Z: validation `build_packs` summary shows `eligible_accounts: 3, packs_built: 3`.
  - 17:10:50.172Z: validation end event with `status: built`, summary metrics: `packs_total: 3`, `validation_ai_required: true`, `validation_ai_completed: false`; `results_total: 0` in results; umbrella flags show `validation_ready: false`. This aligns with `runflow.json`’s final state.
  - No subsequent `refresh_validation_stage_from_index` or `reconcile_umbrella_barriers` persisted after sender/apply completion to promote validation to `success` using manifest.

- Conclusion for this SID:
  - Validation V2 completed and applied at ~17:10:49–17:10:50Z per `manifest.json`.
  - The last runflow writer was `record_stage` at 17:10:50Z, using counts present at that moment (no results in runflow), before the promotion path ran after apply; thus runflow stayed at `status: built`, `validation_ai_completed: false`, `results_total: 0`.
  - No follow-up writer (`refresh_validation_stage_from_index` or `reconcile_umbrella_barriers`) ran afterward to recompute from disk + manifest and persist the completed/applied state.

**Model Alignment: Manifest vs Runflow**
- Manifest (source of truth now):
  - `ai.status.validation.built: true`
  - `ai.status.validation.sent: true`
  - `ai.status.validation.applied: true`
  - `ai.status.validation.results_total: 3`
  - `ai.status.validation.results_applied: 3`
  - `ai.status.validation.results_apply_done: true`
  - `ai.status.validation.validation_ai_applied: true`

- Runflow (snapshot):
  - `stages.validation.status: built`
  - `stages.validation.metrics.validation_ai_required: true`
  - `stages.validation.metrics.validation_ai_completed: false`
  - `stages.validation.results.results_total: 0`
  - `umbrella_barriers.validation_ready: false`
  - `umbrella_ready: false`

- Divergences and writers responsible:
  - `validation_ai_completed` false vs true in manifest: written by `record_stage` path before promotion; not updated later by `_apply_validation_stage_promotion` (no `refresh_*`/`reconcile_*` persisted after apply). Classification: legacy-ish moment-in-time snapshot; missing the V2 promotion step.
  - `results_total` 0 vs 3 in manifest: same cause; snapshot taken before the promotion path recomputed from the validation index and manifest.
  - `validation_ready` false vs should be true: computed in barriers based on stage snapshot; because the stage wasn’t promoted, barriers stayed unready.

**Why the mismatch exists**
- Runflow’s authoritative refresh (`_apply_validation_stage_promotion`) is only applied when one of the writers runs after results are in place: `record_stage` (when called post-results), `refresh_validation_stage_from_index`, or `reconcile_umbrella_barriers`.
- For this SID, the final write was `record_stage` at 17:10:50Z, reflecting pre-promotion values; no subsequent call to `refresh_validation_stage_from_index` or `reconcile_umbrella_barriers` persisted a post-apply promotion snapshot.

**Fix Path (for a future task; no code changes made here)**
- Update the post-apply path to persist a runflow refresh that trusts the manifest:
  - Ensure that after `validation_send_v2` completes and apply is done, we call `refresh_validation_stage_from_index(sid, runs_root)` or `reconcile_umbrella_barriers(sid, runs_root)` to run `_apply_validation_stage_promotion` and persist `stages.validation` with `status: success`, `results_total: 3`, and `validation_ai_completed: true`, `validation_ai_applied: true`.
  - This aligns runflow with the manifest and unblocks `umbrella_barriers.validation_ready`.

**Appendix: Key Code References**
- **Proof: Post-Apply Refresh Is Stable (No Regression Writers in Main Path)**
- **Status priority prevents downgrade**: `_merge_stage_snapshot` prefers higher-priority statuses (success > built). After promotion writes `status: success`, any later `record_stage` with `status: built` will not downgrade it due to `_prefer_stage_status` ordering.
- **Authoritative metrics/summary on promotion**: Promotion marks the incoming snapshot with `_writer: validation_promotion`. `_merge_stage_snapshot` treats `metrics` and `summary` as authoritative when this marker is present and replaces existing values. Subsequent non-promotion writes deep-merge but do not remove these values.
- **Results are recomputed from disk**: Promotion computes `results_total`, `completed`, `failed` via `_validation_results_progress` (reads validation index and result files). After apply, this returns the real totals; reconcile and any later `record_stage` call re-run promotion and keep these values. There is no code path in decider that, post-apply, assigns `results_total: 0` for the validation stage.
- **Fastpath writers are gated pre-apply**: `_maybe_enqueue_validation_fastpath` and `_watchdog_trigger_validation_fastpath` only run when validation is not terminal (built/in_progress) and specific staleness/lock conditions are met. After success/applied, they do not run, and thus cannot regress the stage.
- **Generic snapshot writes won’t regress counts**: `record_stage_force` (used by `decide_next`) merges snapshots and then typical flows immediately reconcile barriers; that path also invokes promotion, restoring correct validation metrics/results from disk if needed.
- Therefore, once `refresh_validation_stage_from_index(sid, runs_root)` is called post-apply, the main V2 pipeline writers either preserve or reassert the promoted state and counts; there is no later writer in the main path that will overwrite the validation stage back to `built` or reset results to zero.

- Validation writers and barriers:
  - `backend/runflow/decider.py:record_stage`
  - `backend/runflow/decider.py:_apply_validation_stage_promotion`
  - `backend/runflow/decider.py:refresh_validation_stage_from_index`
  - `backend/runflow/decider.py:reconcile_umbrella_barriers`
  - `backend/runflow/decider.py:_maybe_enqueue_validation_fastpath`
  - `backend/runflow/decider.py:_watchdog_trigger_validation_fastpath`
  - `backend/core/runflow:runflow_end_stage` (events), `record_validation_build_summary`, `record_validation_results_summary`
