# AI Validation Pack Layout

The AI validation pack flow mirrors the merge pack structure while consolidating
all account data for a run into a single folder tree. The target layout under a
run directory is:

```
runs/<SID>/ai_packs/validation/
  packs/
    val_acc_<ACCID>.jsonl
    # Optional (per-field variant):
    # val_acc_<ACCID>__field_<FIELDKEY>.jsonl
  results/
    acc_<ACCID>.result.jsonl
    acc_<ACCID>.result.json
  index.json
  logs.txt
```

## Naming rules

* `<SID>` is the run identifier (same as other AI pack flows).
* `<ACCID>` identifies the account within the run. Use a single helper such as
  `fmt_accid(14) -> "014"` so all file names zero-pad to three digits where
  practical, keeping listings aligned. Padding is optional if a consumer requires
  the raw numeric string, but the helper should provide the consistent default.
* `<FIELDKEY>` is the snake_cased version of the field name (for example,
  `creditor_remarks`). It only appears when the optional per-field mode is
  enabled.
* Pack files use the `.jsonl` extension to hold one JSON line per weak field
  payload.
* Validation results emit two artifacts per account:
  * `acc_<ACCID>.result.jsonl` contains one JSON line per prompt response.
  * `acc_<ACCID>.result.json` is a compact summary object that embeds the
    JSONL lines and metadata.

## Pack granularity

* Default behavior writes **one pack per account**, bundling all weak
  `ai_needed` fields for that account. This keeps the pack directory concise and
  mirrors merge pack hygiene.
* A per-field mode remains supported behind a writer flag, emitting one pack per
  `(account, field)` pair using the same naming rules above.

## Index and manifest

* `index.json` catalogs every pack/result pair so downstream tooling can reason
  about validation coverage without scanning the filesystem.
* `logs.txt` captures builder activity, mirroring the merge flow.
* The run-level `manifest.json` should reference the `ai_packs.validation`
  locations so orchestration tools can discover validation artifacts the same
  way they do for merge packs.

## Data contracts

Copy these payloads verbatim when building or validating artifacts. The
business-day SLA is now canonical: `min_days` represents business days and
`duration_unit` is `business_days`. Calendar-only fields were deprecated. The
business-day calculator skips Saturday/Sunday weekends today; holiday
exclusions will be layered in as a future enhancement once regional
requirements are defined.

### 1.1 Pack line (JSONL) — one line per weak field

```
{
  "sid": "UUID",
  "account_id": 14,
  "account_key": "014",
  "id": "acc_014__account_type",
  "field": "account_type",
  "category": "open_ident",
  "min_days": 2,
  "duration_unit": "business_days",
  "documents": ["account_opening_contract","application_form","monthly_statement"],
  "strength": "weak",
  "context": {
    "consensus": "majority|split|null",
    "disagreeing_bureaus": ["equifax"],
    "missing_bureaus": ["experian","transunion"]
  },
  "bureaus": {
    "equifax":   {"raw": "…", "normalized": "…"},
    "experian":  {"raw": "…", "normalized": "…"},
    "transunion":{"raw": "…", "normalized": "…"}
  },
  "expected_output": {
    "type": "object",
    "required": ["decision", "rationale", "citations"],
    "properties": {
      "decision":  {"type": "string", "enum": ["strong","no_case"]},
      "rationale": {"type": "string"},
      "citations": {"type": "array", "items": {"type":"string"}}
    }
  },
  "prompt": {
    "system": "You are an adjudication assistant reviewing credit report discrepancies. Decide if there is a strong consumer claim.",
    "guidance": "Return JSON with keys: decision ('strong'|'no_case'), rationale, citations.",
    "user": {
      "sid": "UUID",
      "account_id": 14,
      "account_key": "014",
      "field": "account_type",
      "category": "open_ident",
      "documents": ["…"],
      "context": { "consensus": "…", "disagreeing_bureaus": ["…"], "missing_bureaus": ["…"] },
      "bureaus": {
        "equifax":   {"raw": "…", "normalized": "…"},
        "experian":  {"raw": "…", "normalized": "…"},
        "transunion":{"raw": "…", "normalized": "…"}
      }
    }
  }
}
```

### 1.2 Result line (JSONL) — one line per input line

```
{
  "id": "acc_014__account_type",
  "account_id": 14,
  "field": "account_type",
  "decision": "strong",
  "rationale": "…",
  "citations": ["…","…"]
}
```

### 1.3 Result summary (JSON)

```
{
  "sid": "UUID",
  "account_id": 14,
  "results": [ /* array of result lines */ ]
}
```
