# Developer workflows

## Validation manifest v2 (Validation AI)

The validation sender now trusts a single manifest as the source of truth for
pack and result locations. When a run finishes, it writes
`runs/<SID>/ai_packs/validation/index.json`; both the pack builder and sender
use that manifest exclusively. Paths in schema v2 are **always POSIX-style and
relative to the manifest** so the index remains stable if the run directory is
moved (for example, when syncing `runs/` between machines or CI workers).

### Schema overview

Schema v2 documents look like the example below. Every relative path is resolved
from the directory that contains `index.json`.

```json
{
  "schema_version": 2,
  "sid": "41495514-931e-4100-90ca-4928464dcda8",
  "root": ".",
  "packs_dir": "packs",
  "results_dir": "results",
  "packs": [
    {
      "account_id": 1,
      "pack": "packs/val_acc_001.jsonl",
      "result_jsonl": "results/acc_001.result.jsonl",
      "result_json": "results/acc_001.result.json",
      "lines": 12,
      "status": "built",
      "built_at": "2024-06-18T02:14:03Z",
      "weak_fields": ["account_name"],
      "source_hash": "3a6408f2"
    }
  ]
}
```

Field reference:

| Field | Notes |
| --- | --- |
| `schema_version` | Must be `2` for relative-path manifests. |
| `sid` | Run identifier. |
| `root` | Common base directory shared by `packs_dir` and `results_dir`. Usually `"."`. |
| `packs_dir` / `results_dir` | Relative directories under `root` for pack inputs and model outputs. |
| `packs` | Array of per-account records. Each record stores relative paths for the pack payload and both result files, plus metadata such as `lines`, `status`, `built_at`, optional `weak_fields`, and any extra keys persisted by builders. |

### PowerShell quickstart (Windows)

Commands below assume the repository lives at `C:\dev\credit-analyzer` and the
virtual environment has already been created as `.venv`.

```powershell
cd C:\dev\credit-analyzer
.\.venv\Scripts\Activate.ps1
$env:PYTHONPATH = "$PWD"
$SID = "41495514-931e-4100-90ca-4928464dcda8"
python -m backend.validation.manifest --sid $SID --check
python -m backend.validation.send --sid $SID
```

### Window 1 — Celery (validation & merge queues)

```powershell
cd C:\dev\credit-analyzer
.\.venv\Scripts\Activate.ps1
$env:PYTHONPATH = "$PWD"
$env:RUNS_ROOT  = "$PWD\runs"
$env:CELERY_BROKER_URL     = "redis://localhost:6379/0"
$env:CELERY_RESULT_BACKEND = "redis://localhost:6379/1"

# OpenAI + validation
$env:OPENAI_BASE_URL   = "https://api.openai.com/v1"
$env:OPENAI_API_KEY    = "<redacted>"
$env:OPENAI_PROJECT_ID = "<proj>"
$env:VALIDATION_MODEL  = "gpt-4o-mini"

# enable sender + autosend (autosend is on by default; set to "0" to disable)
$env:ENABLE_VALIDATION_SENDER   = "1"
$env:VALIDATION_AUTOSEND_ENABLED = "1"

# manifest/index paths (relative to run root)
$env:VALIDATION_USE_MANIFEST_PATHS = "1"
$env:VALIDATION_INDEX_PATH   = "ai_packs/validation/index.json"
$env:VALIDATION_PACKS_DIR    = "ai_packs/validation/packs"
$env:VALIDATION_RESULTS_DIR  = "ai_packs/validation/results"
$env:VALIDATION_PACK_GLOB    = "val_acc_*.jsonl"
$env:VALIDATION_RESULTS_BASENAME = "acc_{account}.result"  # code appends .jsonl

# guard envelopes OFF by default
$env:VALIDATION_WRITE_JSON_ENVELOPE = "0"
$env:VALIDATION_MAX_RETRIES = "2"
$env:VALIDATION_REQUEST_GROUP_SIZE = "1"

# queues
$PY = "$PWD\.venv\Scripts\python.exe"
& $PY -m celery -A backend.api.tasks worker `
  --loglevel=INFO `
  --pool=solo `
  -Q celery,merge,validation,note_style,frontend `
  --prefetch-multiplier=1 `
  --max-tasks-per-child=50

At least one worker process must listen on the `frontend` queue so the review
packs generate promptly even while merge or validation jobs are running.
When the worker boots, confirm the banner lists `frontend` in the `queues` line
and shows `backend.api.tasks.generate_frontend_packs_task` under `[tasks]`.
Submitting a run should emit `enqueue generate_frontend_packs_task ... queue=frontend`
in the logs, indicating the job was routed correctly.
```

`backend.validation.manifest` validates that every pack referenced in the
manifest exists. `backend.validation.send` reads only the manifest, prepares the
result directories if necessary, and writes the model responses next to the
referenced result paths.

## Merge / Dedup ENV Reference

| Variable | Default | Notes |
| --- | --- | --- |
| `MERGE_ENABLED` | `false` | Enables the dedicated merge block. When omitted the legacy merge stack remains in place. |
| `MERGE_FIELDS_OVERRIDE_JSON` | `account_number`, `date_opened`, `balance_owed`, `account_type`, `account_status`, `history_2y`, `history_7y` | Controls the allowlisted merge fields. Accepts JSON or comma separated values via `MERGE_FIELDS_OVERRIDE`. |
| `MERGE_ALLOWLIST_ENFORCE` | `false` | Forces the scorer to ignore fields not in the allowlist. |
| `MERGE_USE_CUSTOM_WEIGHTS` | `false` | Toggles `MERGE_WEIGHTS_JSON` to override per-field scoring weights. |
| `MERGE_WEIGHTS_JSON` | `{}` | Optional weight overrides applied when custom weights are enabled. |
| `MERGE_THRESHOLDS_JSON` | `{}` | Optional overrides for `MERGE_SCORE_THRESHOLD` and related cutoffs. |
| `MERGE_TOLERANCES_JSON` | `{}` | Optional overrides for balance/date tolerances used when comparing fields. |
| `MERGE_OVERRIDES_JSON` | `{}` | Supplies explicit rule overrides for bespoke scenarios. |
| `MERGE_DEBUG` | `false` | Enables verbose merge scoring logs. Combine with `MERGE_LOG_EVERY` to throttle sampling. |
| `MERGE_LOG_EVERY` | `0` | Optional cadence controlling debug sampling frequency. |
| `MERGE_REQUIRE_ORIGINAL_CREDITOR_FOR_AI` | `false` | When enabled, AI merge packs are only built if either account supplies an `original_creditor` value. |
| `MERGE_POINTS_DIAGNOSTICS` | `false` | Legacy toggle retained for backwards compatibility. Diagnostics sampling is controlled via `MERGE_POINTS_PERSIST_BREAKDOWN`. |
| `MERGE_POINTS_DIAGNOSTICS_LIMIT` | `3` | Legacy limit retained for backwards compatibility. |
| `MERGE_POINTS_PERSIST_BREAKDOWN` | `false` | When `true`, persist `points_breakdown_{lo}_{hi}.json` payloads to disk and expose their paths in results. |
| `MERGE_POINTS_DIAGNOSTICS_DIR` | `ai_packs/merge/diagnostics` | Destination (absolute or relative to the run root) for persisted points-mode diagnostics files. |

The configuration loader normalizes booleans, numbers, and JSON payloads so the
runtime merge configuration can be tuned entirely via environment variables.

### Points-mode diagnostics payload

- When `MERGE_POINTS_PERSIST_BREAKDOWN=1`, each scored pair emits a
  `points_breakdown_{lo}_{hi}.json` sidecar under `MERGE_POINTS_DIAGNOSTICS_DIR`
  (relative paths are resolved within the run directory) and matching `FIELDS`
  log lines. When disabled, results omit the diagnostics path and no files are
  written.
- The persisted document includes a `pair` object with `lo`/`hi` indexes, the
  original `a`/`b` members, and the current `sid`. Thresholds are grouped under
  `thresholds.ai`, `thresholds.direct`, and `thresholds.duplicate`.
- Each field entry contains a compact `pairs` list (just bureau identifiers) and
  a verbose `pairs_verbose` list with sanitized `raw_values`, `normalized_values`,
  and the reason codes considered while matching.
- Log entries follow the shape
  `FIELDS lo-hi FIELD_NAME matched=true points=... reason=... pairs=[...]` to
  mirror the JSON payload and surface skip reasons.

## Account number validation (deterministic comparator)

The validation pipeline now routes `account_number_display` comparisons through
the same merge comparator that powers adjudication. This keeps validation and
merge aligned without inventing a separate tolerance layer. The comparator’s
`match_level` drives deterministic outcomes — masked pairs such as
`****1234` vs `1234` resolve cleanly, while two distinct full numbers that only
share the last four digits are marked mismatched.

- **No tolerance window.** Account numbers are treated as pure strings; the
  comparator alone decides whether they agree or conflict.
- **Rollback lever.** `VALIDATION_USE_LEGACY_ACCOUNT_NUMBER_COMPARE` (default
  `0`) keeps the new path enabled. Temporarily flip it to `1` if we need to
  revert to the legacy behavior during rollout.

### Expected command output

Successful manifest check:

```
Validation packs for SID 41495514-931e-4100-90ca-4928464dcda8:
ACCOUNT  PACK                      STATUS  LINES  RESULT_JSONL                 RESULT_JSON
------   ------------------------  ------  -----  ---------------------------  --------------------------
001      packs/val_acc_001.jsonl   OK      12     results/acc_001.result.jsonl results/acc_001.result.json
002      packs/val_acc_002.jsonl   OK      10     results/acc_002.result.jsonl results/acc_002.result.json

Manifest: index.json
Packs dir: packs
Results dir: results
All 2 packs present for SID 41495514-931e-4100-90ca-4928464dcda8.
```

Missing pack example (after deleting a pack):

```
Validation packs for SID 41495514-931e-4100-90ca-4928464dcda8:
ACCOUNT  PACK                      STATUS  LINES  RESULT_JSONL                 RESULT_JSON
------   ------------------------  ------  -----  ---------------------------  --------------------------
001      packs/val_acc_001.jsonl   OK      12     results/acc_001.result.jsonl results/acc_001.result.json
002      packs/val_acc_002.jsonl   MISSING 0      results/acc_002.result.jsonl results/acc_002.result.json

Manifest: index.json
Packs dir: packs
Results dir: results
Missing packs detected: 1 of 2.
```

Sender preflight summary when everything is present:

```
MANIFEST: runs/41495514-931e-4100-90ca-4928464dcda8/ai_packs/validation/index.json
PACKS: 2, missing: 0
RESULTS DIR: ok
[acc=001] pack=packs/val_acc_001.jsonl -> results/acc_001.result.jsonl, results/acc_001.result.json  (lines=12)
[acc=002] pack=packs/val_acc_002.jsonl -> results/acc_002.result.jsonl, results/acc_002.result.json  (lines=10)
```

If a pack is missing the sender reports it relative to the manifest:

```
MANIFEST: runs/41495514-931e-4100-90ca-4928464dcda8/ai_packs/validation/index.json
PACKS: 2, missing: 1
RESULTS DIR: ok
[acc=001] pack=packs/val_acc_001.jsonl -> results/acc_001.result.jsonl, results/acc_001.result.json  (lines=12)
[acc=002] pack=packs/val_acc_002.jsonl -> results/acc_002.result.jsonl, results/acc_002.result.json  (lines=0)  [MISSING: packs/val_acc_002.jsonl]
```

### Quick success checklist

1. Build or reuse a run so `runs/<SID>/ai_packs/validation/index.json` exists with `"schema_version": 2` and manifest paths that start with `packs/` and `results/`.
2. `python -m backend.validation.manifest --sid <SID> --check` should print the tabular summary and `All <N> packs present` when every file exists.
3. `python -m backend.validation.send --sid <SID>` must write new result files under the manifest-defined `results/` directory.
4. Remove or rename a pack and re-run `--check`; it should clearly report `MISSING` and exit with a non-zero status.
5. Run your usual pipeline command with `$env:ENABLE_VALIDATION_SENDER = "1"` (or set the variable in your orchestrator). Auto-send now runs by default; set `$env:VALIDATION_AUTOSEND_ENABLED = "0"` if you need to pause sending. The builder generates packs, the manifest stays relative, and the sender runs automatically using only the manifest for resolution.

