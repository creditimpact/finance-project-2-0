# Frontend Stage README

The frontend stage builds lightweight account packs that power customer-facing experiences. These packs mirror the case artifacts produced earlier in the run so they can be regenerated without coordinating with merge or validation.

## Pack Data Sources
- **`holder_name`** – pulled from `cases/accounts/<N>/meta.json` (`heading_guess`). If the heading is missing, fall back to parsing `cases/accounts/<N>/raw_lines.json`. Never read `bureaus.json` for this field.
- **`primary_issue`** – the first issue entry in `cases/accounts/<N>/tags.json` (`{"kind": "issue", "type": ...}`). Additional issues may also be emitted in an `issues` array for UI enrichment.
- **Other account details** – flow through directly from existing case artifacts in `cases/accounts/<N>/` so that the packs remain consumer-safe and omit tolerance/debug-only state.

## Field Guarantees
- Every account with a case has at least one issue tag, so `primary_issue` is always populated.
- `holder_name` may be `null` when neither `meta.json` nor the raw heading lines produce a usable value.
- All legacy fields in the pack schema are preserved; newly added fields only extend the payload.

## Idempotency & Runflow Signals
- The generator only reads previously materialised case artifacts, so rerunning the stage reuses the same inputs and produces identical pack files and index counts.
- Runflow records the count of `runs/<SID>/frontend/review/packs/*.json` files and any creation errors. No entries are emitted when the stage simply revalidates existing packs.
- Because the stage is decoupled from merge/validation, operators can rebuild the frontend packs independently after correcting case artifacts.

## Frontend Review Stage quick reference

### Filesystem layout

```
runs/<sid>/
  frontend/
    review/
      packs/        # one JSON file per account (e.g. acct-123.json)
      responses/    # append-only JSONL responses per account
      index.json    # manifest describing every pack
```

Each pack file is a compact document with `{"account_id", "holder_name", "primary_issue", "display"}` so the UI can render without rehydrating legacy artifacts. Responses are stored as newline-delimited JSON at `frontend/review/responses/{account_id}.jsonl`; both stage and legacy response directories are kept in sync for backward compatibility.

### Manifest schema (`frontend/review/index.json`)

The manifest exposes the full set of packs with lightweight metadata so the web UI can list accounts without opening every file:

```json
{
  "sid": "SID-123",
  "stage": "review",
  "schema_version": "1.0",
  "counts": {"packs": 2, "responses": 1},
  "packs": [
    {
      "account_id": "acct-1",
      "holder_name": "John Doe",
      "primary_issue": "wrong_account",
      "path": "frontend/review/packs/acct-1.json",
      "bytes": 512,
      "has_questions": true,
      "sha1": "…" // optional hash when available
    }
  ],
  "responses_dir": "frontend/review/responses",
  "generated_at": "2024-05-01T12:34:56Z"
}
```

- `packs[]` always includes the relative file path and size so callers can deep-link or lazily fetch a pack.
- `has_questions` flags accounts that surfaced customer prompts; `sha1` appears when the pack contents can be hashed successfully.
- `generated_at` is stable across idempotent runs to avoid unnecessary churn if nothing changed.

### API endpoints

| Method | Route | Notes |
| --- | --- | --- |
| `GET` | `/api/runs/<sid>/frontend/review/index` | Returns the manifest JSON above; `404` when the stage has not materialised. |
| `GET` | `/api/runs/<sid>/frontend/review/accounts/<account_id>` | Loads a single pack. Accepts canonical IDs or legacy numeric keys; falls back to legacy `frontend/accounts/<key>/pack.json` files when the stage copy is missing. |
| `POST` | `/api/runs/<sid>/frontend/review/accounts/<account_id>/answer` | Appends `{ "answers": {...}, "client_ts": "ISO8601?" }` payloads to `{account_id}.jsonl` and echoes `{ "ok": true }`. Payload must include an `answers` mapping; `client_ts` is optional. |

### Environment variables

The generator prefers the new review-stage environment variables but still honours the legacy names for callers that have not migrated:

| Purpose | Preferred | Legacy fallback(s) |
| --- | --- | --- |
| Stage name in the manifest/runflow | `FRONTEND_STAGE_NAME` | `FRONTEND_STAGE` |
| Base stage directory | `FRONTEND_PACKS_STAGE_DIR` | `FRONTEND_STAGE_DIR` |
| Packs output directory | `FRONTEND_PACKS_DIR` | `FRONTEND_ACCOUNTS_DIR`, `FRONTEND_PACKS_PATH` |
| Responses directory | `FRONTEND_PACKS_RESPONSES_DIR` | `FRONTEND_RESPONSES_DIR` |
| Manifest path | `FRONTEND_PACKS_INDEX` | `FRONTEND_INDEX_PATH`, `FRONTEND_INDEX` |

When unset, the defaults resolve to `frontend/review/` inside the run directory, and the system continues to mirror responses and pack metadata back into the legacy `frontend/` tree for compatibility.
