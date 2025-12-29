# Runflow artifacts

Runflow emits two complementary artifacts under `runs/<SID>/runflow/` so that
operators can debug a run in real time and replay the full story afterwards.
This note explains how to read both files, what the v2.1 schema looks like, and
which knobs control the output.

## Artifact quick reference

| File | Purpose | Format |
| --- | --- | --- |
| `runflow_events.jsonl` | Firehose of every event as it happens. Useful for streaming logs and deep forensic work. | Newline-delimited JSON. Each row is a single event. |
| `runflow_steps.json` | Deterministic story of the run with stages, steps, summaries, and counters. Treat this as the authoritative timeline. | Compact JSON document following the v2.1 schema described below. |

## Quick acceptance checks

Operators can use simple `jq` probes against `runs/<sid>/runflow.json` to confirm
each stage (and the umbrella orchestrator) has reached a healthy terminal
state. All commands return `true` once their respective conditions are met.

```bash
# Merge stage confirms success with at least one result file recorded.
jq '.stages.merge.status=="success" and ((.stages.merge.summary.result_files // .stages.merge.result_files) >= 1)' runs/<sid>/runflow.json

# Validation stage checks that all scheduled results finished processing.
jq '.stages.validation.status=="success" and (.stages.validation.results.completed == .stages.validation.results.results_total)' runs/<sid>/runflow.json

# Review (frontend) stage verifies all expected answers arrived.
jq '.stages.frontend.status=="success" and (.stages.frontend.metrics.answers_received == .stages.frontend.metrics.answers_required)' runs/<sid>/runflow.json

# Umbrella orchestrator signals the whole pipeline is ready.
jq '.umbrella_barriers.all_ready==true and .umbrella_ready==true' runs/<sid>/runflow.json
```

Both artifacts are safe to regenerate. Steps always include `schema_version` so
older tooling can continue to parse historical runs.

## `runflow_events.jsonl`

Events are written as soon as we observe them and are intentionally verbose.
Every line is a self-contained JSON object with the timestamp (`ts`), stage,
`event` type (`start`, `step`, `end`, …), and whatever context was captured at
runtime. For example:

```json
{"ts":"2024-05-01T12:00:00Z","stage":"merge","event":"step","name":"acctnum_match_level","status":"success","metrics":{"score":87}}
```

Use the firehose when you need to reconstruct debug details or investigate why a
span emitted a certain payload. Nothing in this file is deduplicated or ordered
beyond wall clock time.

## `runflow_steps.json` schema (v2.2)

The steps file provides the compact, monotonic timeline. The top-level payload
looks like this:

```json
{
  "sid": "SID-123",
  "schema_version": "2.2",
  "updated_at": "2024-05-01T12:05:00Z",
  "stages": {
    "merge": {
      "status": "success",
      "started_at": "2024-05-01T12:00:00Z",
      "ended_at": "2024-05-01T12:02:12Z",
      "summary": {"scored_pairs": 12, "matches_strong": 3},
      "empty_ok": false,
      "steps": [
        {
          "seq": 1,
          "name": "merge_scoring",
          "status": "start",
          "t": "2024-05-01T12:00:00Z",
          "metrics": {"accounts": 6},
          "span_id": "c246…"
        },
        {
          "seq": 2,
          "name": "acctnum_match_level",
          "status": "success",
          "t": "2024-05-01T12:00:07Z",
          "account": "24-31",
          "metrics": {"score": 94, "rank": 1},
          "out": {"decision": "match", "why": "digit_conflict_resolved"},
          "parent_span_id": "c246…"
        }
      ],
      "next_seq": 3,
      "substages": {
        "default": {
          "status": "success",
          "started_at": "2024-05-01T12:00:00Z",
          "steps": [
            {"seq": 2, "name": "acctnum_match_level", "status": "success", "t": "2024-05-01T12:00:07Z"}
          ]
        }
      }
    }
  }
}
```

Key fields:

* `sid` and `schema_version` – identify the run and schema revision.
* `updated_at` – reflects the newest timestamp seen in the document.
* `stages` – map keyed by stage name.
  * `status` – authoritative stage state (`running`, `success`, `empty`, `error`, …).
  * `started_at` / `ended_at` – ISO-8601 UTC timestamps. `ended_at` is omitted while the stage runs.
  * `summary` – concise counters for the stage. These are validated against disk (see below).
  * `empty_ok` – explicitly marks stages that produced no work but are considered successful.
  * `steps` – ordered list of deterministic step entries. Each entry contains:
    * `seq` – monotonically increasing integer sequence.
    * `name` – stable identifier of the step.
    * `status` – `success`, `error`, `start`, etc.
    * `t` – timestamp for the step.
    * Optional context: `account`, `metrics`, `out`, `reason`, `span_id`, `parent_span_id`, `error`.
  * `next_seq` – the next sequence number that will be assigned if we append more steps.
  * `substages` – optional grouping for legacy readers. The `default` bucket mirrors the main step list and is always present in v2.1.

Stages are added lazily. If a stage never ran there will be no entry in the map.

## Examples

### Merge stage with pack creations only

`runflow_steps.json` keeps the merge story concise by recording the successful
pack creations and omitting the skipped attempts that still appear in
`runflow_events.jsonl`. A typical snippet looks like this:

```json
"merge": {
  "status": "success",
  "started_at": "2024-05-01T12:00:00Z",
  "ended_at": "2024-05-01T12:01:12Z",
  "summary": {"pairs_scored": 5, "packs_created": 2, "empty_ok": false},
  "empty_ok": false,
  "steps": [
    {"seq": 7, "name": "pack_create", "status": "success", "t": "2024-05-01T12:00:46Z", "out": {"path": "runs/SID/ai_packs/merge/pack_000.json"}},
    {"seq": 8, "name": "pack_create", "status": "success", "t": "2024-05-01T12:01:05Z", "out": {"path": "runs/SID/ai_packs/merge/pack_001.json"}}
  ]
}
```

Only the `pack_create` entries survive here; all of the skipped attempts remain
discoverable in the events firehose.

### Merge stage with no candidates

When merge completes without creating any packs we still persist the stage, mark
it as `empty`, and set `empty_ok: true` to communicate that the result is
expected. The steps list finishes with a single `no_merge_candidates` footprint
so readers know why no packs were produced.

```json
"merge": {
  "status": "empty",
  "started_at": "2024-05-01T12:00:00Z",
  "ended_at": "2024-05-01T12:00:01Z",
  "summary": {"pairs_scored": 0, "packs_created": 0, "empty_ok": true},
  "empty_ok": true,
  "steps": [
    {
      "seq": 4,
      "name": "no_merge_candidates",
      "status": "success",
      "t": "2024-05-01T12:00:01Z",
      "metrics": {"scored_pairs": 0}
    }
  ],
  "next_seq": 5,
  "substages": {
    "default": {
      "status": "running",
      "started_at": "2024-05-01T12:00:00Z",
      "steps": [
        {"seq": 4, "name": "no_merge_candidates", "status": "success", "t": "2024-05-01T12:00:01Z"}
      ]
    }
  }
}
```

This pattern is used by validation and frontend as well when they produce no
findings or packs.

### Merge stage with Top‑N highlights

For larger runs we only keep the top `RUNFLOW_STEPS_PAIR_TOPN` candidate pairs in
line while summarising everything else in the `acctnum_pairs_summary` step. The
summary step links to the index written on disk so you can inspect the full
list.

```json
"merge": {
  "status": "success",
  "started_at": "2024-05-01T12:00:00Z",
  "ended_at": "2024-05-01T12:02:12Z",
  "summary": {
    "scored_pairs": 3,
    "matches_strong": 0,
    "matches_weak": 0,
    "conflicts": 3,
    "packs_built": 0
  },
  "steps": [
    {"seq": 1, "name": "merge_scoring", "status": "start", "t": "2024-05-01T12:00:00Z", "metrics": {"accounts": 3}, "span_id": "c246…"},
    {"seq": 2, "name": "acctnum_match_level", "status": "success", "t": "2024-05-01T12:00:07Z", "account": "0-1", "metrics": {"rank": 1, "score": 0, "allowed": 0}, "out": {"left": "0", "right": "1", "decision": "different"}, "parent_span_id": "c246…"},
    {"seq": 3, "name": "acctnum_match_level", "status": "success", "t": "2024-05-01T12:00:08Z", "account": "0-2", "metrics": {"rank": 2, "score": 0, "allowed": 0}, "out": {"left": "0", "right": "2", "decision": "different"}, "parent_span_id": "c246…"},
    {"seq": 4, "name": "acctnum_pairs_summary", "status": "success", "t": "2024-05-01T12:00:09Z", "metrics": {"scored_pairs": 3, "conflicts": 3, "skipped": 3, "packs_built": 0, "topn_limit": 2}, "out": {"pairs_index": "runs/SID-123/ai_packs/merge/pairs_index.json"}, "parent_span_id": "c246…"},
    {"seq": 5, "name": "merge_scoring", "status": "success", "t": "2024-05-01T12:02:12Z", "metrics": {"scored_pairs": 3, "conflicts": 3, "skipped": 3, "packs_built": 0, "topn_limit": 2, "normalized_accounts": 3}, "span_id": "c246…"}
  ],
  "next_seq": 6,
  "substages": {"default": {"status": "success", "started_at": "2024-05-01T12:00:00Z", "steps": [/* same steps as above for legacy readers */]}}
}
```

Downstream tooling should use the `acctnum_pairs_summary.metrics.topn_limit`
field to understand how many pairs were retained inline.

## Stage summaries and counters

Every stage summary is cross-checked against files on disk when
`RUNFLOW_STEPS_VERIFY` is enabled (default). The reconciliation helpers live in
[`backend/runflow/counters.py`](../backend/runflow/counters.py) and source their
data from:

* **Merge** – `runs/<SID>/ai_packs/merge/pairs_index.json` (`totals.scored_pairs`).
* **Validation** – `runs/<SID>/ai_packs/validation/index.json` (sum of pack line counts).
* **Frontend** – `runs/<SID>/frontend/review/packs/*.json` (count of pack files).

If the counters disagree, the stage payload receives an `error` block detailing
the mismatch. Double-check the referenced files and the upstream producer.
Setting `RUNFLOW_STEPS_VERIFY=0` suppresses the check when you intentionally
re-run instrumentation on partial artifacts.

## Environment knobs

These environment variables control instrumentation behaviour:

| Variable | Effect | Default |
| --- | --- | --- |
| `RUNS_ROOT` | Root directory for `runs/` artifacts. | `runs` |
| `RUNFLOW_VERBOSE` | Enable `runflow_steps.json` writes. | disabled |
| `RUNFLOW_EVENTS` | Enable `runflow_events.jsonl` writes. | disabled |
| `RUNFLOW_STEPS_SCHEMA_VERSION` | Override the schema version string (for forward compat testing). | `2.1` |
| `RUNFLOW_STEPS_VERIFY` | Validate stage summaries against disk counters. | `1` (enabled) |
| `RUNFLOW_STEPS_ENABLE_SPANS` | Include span identifiers in step entries. | `1` (enabled) |
| `RUNFLOW_STEP_LOG_EVERY` | Sample frequency for success steps (1 = log everything). | `1` |
| `RUNFLOW_STEPS_PAIR_TOPN` | Number of merge pairs to keep inline before aggregating into the summary. | `5` |
| `RUNFLOW_STEPS_SUPPRESS_PER_ACCOUNT` | Skip per-account step entries unless they error. | `0` |
| `RUNFLOW_STEPS_ONLY_AGGREGATES` | Emit only aggregate roll-ups when combined with suppression. | `0` |

Toggle these knobs via `monkeypatch.setenv` in tests or shell variables in local
runs.

When both `RUNFLOW_STEPS_SUPPRESS_PER_ACCOUNT=1` and
`RUNFLOW_STEPS_ONLY_AGGREGATES=1`, `runflow_steps.json` records only the compact
aggregate rows while still surfacing per-account entries if they fail. This
keeps the steps timeline readable for large runs.

## Troubleshooting mismatched counters

1. Inspect the `summary` block for the stage – it contains the counters we
   intended to write.
2. Compare the expected numbers with the source files listed above.
3. If `runflow_steps.json` shows an `error` block, copy the `hint` and `message`
   fields to your notes; they point to the first mismatch detected.
4. Use the corresponding entries in `runflow_events.jsonl` to see the raw steps
   that produced the outputs.
5. When iterating locally on partially written artifacts you can temporarily set
   `RUNFLOW_STEPS_VERIFY=0`, but make sure to re-enable it before committing.

With these guidelines a new contributor can open the doc, inspect a run’s
artifacts, and quickly understand both the firehose and the curated story.
