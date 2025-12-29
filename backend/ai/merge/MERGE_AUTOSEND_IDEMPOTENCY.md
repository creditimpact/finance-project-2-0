Merge Autosend Idempotency

Overview
- Autosend now avoids infinite loops by consulting runflow state and skipping already-adjudicated packs.

When autosend runs
- Environment enables autosend: `MERGE_AUTOSEND=1` and `MERGE_STAGE_AUTORUN=1`.
- Runflow is readable and indicates merge is not fully applied.
- There are merge packs without corresponding result files on disk.

When autosend is skipped
- `runflow.json` temporarily unreadable (e.g., PermissionError): loader sets `runflow_unavailable=true`; scheduler logs `MERGE_AUTOSEND_SKIPPED reason=runflow_unavailable` and skips.
- Merge already fully applied per runflow (`status=success`, `merge_ai_applied=true`, and `result_files >= expected_packs` or fallback): scheduler logs `MERGE_AUTOSEND_SKIPPED reason=merge_already_applied`.

Pack filtering
- Discovery still finds all pack files, but send phase filters out any pack with an existing result file in `ai_packs/merge/results`.
- When no pending packs remain, sender logs `MERGE_AUTOSEND_NO_PENDING_PACKS` and does not invoke the external script.

Notes
- Result filename mapping uses the canonical template defined by configuration (default: `pair_{lo:03d}_{hi:03d}.result.json`).
- The external script may still be executed manually; the scheduler simply refrains from redundant enqueues.
