# Problem Accounts Analyzer Investigation

Summary
- For SID 5f316c34-e65c-4336-ba5b-a2120d9fbbc4, the analyzer returns zero problematic accounts because the Stage‑A `accounts_from_full.json` does not contain the flat fields the detector expects (e.g., `past_due_amount`, `payment_status`, `account_status`, `two_year_payment_history`, `days_late_7y`). The file’s accounts are structural/triad metadata without a `fields` object or flattened field names. As a result, the detector produces no problem reasons and the builder has no candidates to write.

## Call Graph (current pipeline)
- Celery tasks → `backend/api/tasks.py:extract_problematic_accounts`
  - Calls `backend/core/logic/report_analysis/extract_problematic_accounts.extract_problematic_accounts`
    - Loads inputs via RunManifest
    - Calls `problem_extractor.detect_problem_accounts(sid)`
      - Reads `accounts_from_full.json`
      - For each account: computes `account_id`, selects `fields` mapping if present, otherwise uses the account dict itself
      - Calls `problem_detection.evaluate_account_problem(fields)` to decide
    - Calls `problem_case_builder.build_problem_cases(sid, candidates)`
      - Emits per‑account outputs if any candidates exist

## 1) Manifest Wiring (actual)
```
artifacts.traces.accounts_table:
  accounts_json:       runs/<SID>/traces/accounts_table/accounts_from_full.json
  general_json:        runs/<SID>/traces/accounts_table/general_info_from_full.json
  debug_full_tsv:      runs/<SID>/traces/accounts_table/_debug_full.tsv
  per_account_tsv_dir: runs/<SID>/traces/accounts_table/per_account_tsv

base_dirs:
  uploads_dir:         runs/<SID>/uploads
  traces_dir:          runs/<SID>/traces
  cases_dir:           runs/<SID>/cases
  traces_accounts_table: runs/<SID>/traces/accounts_table
  cases_accounts_dir:  runs/<SID>/cases/accounts

artifacts.cases:
  accounts_index:      runs/<SID>/cases/accounts/index.json
  problematic_ids:     runs/<SID>/cases/index.json
```
Evidence (PowerShell dump):
```
artifacts.traces.accounts_table:
{
  "accounts_json":  "C:\\dev\\credit-analyzer\\runs\\5f316c34-e65c-4336-ba5b-a2120d9fbbc4\\traces\\accounts_table\\accounts_from_full.json",
  "general_json":   "C:\\dev\\credit-analyzer\\runs\\5f316c34-e65c-4336-ba5b-a2120d9fbbc4\\traces\\accounts_table\\general_info_from_full.json",
  "debug_full_tsv": "C:\\dev\\credit-analyzer\\runs\\5f316c34-e65c-4336-ba5b-a2120d9fbbc4\\traces\\accounts_table\\_debug_full.tsv",
  "per_account_tsv_dir": "C:\\dev\\credit-analyzer\\runs\\5f316c34-e65c-4336-ba5b-a2120d9fbbc4\\traces\\accounts_table\\per_account_tsv"
}
base_dirs:
{
  "uploads_dir": "...\\runs\\...\\uploads",
  "traces_dir":  "...\\runs\\...\\traces",
  "cases_dir":   "...\\runs\\...\\cases",
  "traces_accounts_table": "...\\runs\\...\\traces\\accounts_table",
  "cases_accounts_dir":    "...\\runs\\...\\cases\\accounts"
}
artifacts.cases:
{
  "accounts_index":  "...\\runs\\...\\cases\\accounts\\index.json",
  "problematic_ids": "...\\runs\\...\\cases\\index.json"
}
```

## 2) Stage‑A JSON Schema vs Analyzer Expectations
Schema peek (Python):
```
ACCOUNTS_JSON_TOPLEVEL_KEYS: ['accounts', 'stop_marker_seen']
ACCOUNTS_CONTAINER_TYPE: list LEN: 16
SAMPLE_ACCOUNT_KEYS: [
  'account_index','page_start','line_start','page_end','line_end',
  'heading_guess','heading_source','section','section_prefix_seen','lines',
  'trailing_section_marker_pruned','noise_lines_skipped',
  'two_year_payment_history','seven_year_history','triad','triad_fields','triad_rows'
]
GENERAL_JSON_TOPLEVEL_KEYS: ['sections','summary_filter_applied']
```
Detector’s required/evaluated fields (from `problem_detection.py`):
- Numeric: `past_due_amount`, `balance_owed`, `credit_limit`
- Status: `payment_status`, `account_status`
- History: `two_year_payment_history`, `days_late_7y`
- Full required set: `STAGEA_REQUIRED_FIELDS = [balance_owed,payment_status,account_status,credit_limit,past_due_amount,account_rating,account_description,creditor_remarks,account_type,creditor_type,dispute_status,two_year_payment_history,days_late_7y]`

Mismatch observed:
- The sample account object contains structural and triad metadata but no `fields` object and no flat keys like `past_due_amount` or `payment_status` at the top level.
- Any bureau‑specific data might be nested under `triad_fields` (per bureau), which the current detector does not read.

## 3) Analyzer Entry Points & Logic
- `problem_extractor.detect_problem_accounts(sid)`
  - Loads `accounts_from_full.json` via RunManifest.
  - For each account:
    - `fields = account['fields']` if present and mapping; else `fields = account`.
    - Calls `evaluate_account_problem(fields)`.
  - A problematic account requires `problem_reasons` or normalized `signals`.
- `problem_detection.evaluate_account_problem(acct)`
  - Builds `problem_reasons` from the presence/values of:
    - `past_due_amount` (numeric > 0)
    - any of `payment_status` / `account_status`
    - late counts from `two_year_payment_history` or `days_late_7y`
  - Returns neutral decision if none of the above are present.
- `problem_case_builder.build_problem_cases(sid, candidates)`
  - Creates per‑account folders only for provided candidate IDs.
  - Writes `runs/<SID>/cases/accounts/index.json` with the list of IDs written.

## 4) Runtime (analyzer‑only)
Smoke script result:
```
python .\scripts\smoke_problem_cases.py --sid <SID>
{"sid": "5f316c34-e65c-4336-ba5b-a2120d9fbbc4", "problematic": 0, "out": "...\\runs\\...\\cases\\accounts\\index.json", "sample": []}
```
Filesystem state:
```
runs/<SID>/cases/accounts/index.json:
{
  "sid": "5f316c34-e65c-4336-ba5b-a2120d9fbbc4",
  "count": 0,
  "ids": [],
  "items": []
}
(no account subfolders present)
```
Detector direct:
```
COUNT 0
[]
```

## 5) Detector Diagnosis
Why COUNT == 0 despite known issues:
- The detector only considers fields in the provided `fields` mapping (if present) or the account dict itself.
- The Stage‑A accounts here have neither a `fields` mapping nor the flat field names the detector checks, so:
  - `past_due_amount` not found → no numeric reason
  - `payment_status` / `account_status` not found → no status reason
  - `two_year_payment_history` present, but its companion `days_late_7y` is absent and the value format may not match the detector’s parser; also, without other signals, this often yields no reasons.
- Result: `problem_reasons` empty for all accounts → zero candidates.

## 6) Toggles / Thresholds
- AI adjudication is gated by `config.ENABLE_AI_ADJUDICATOR` with `AI_MIN_CONFIDENCE`. In this run, the detector uses rules only; no evidence suggests AI contributed.
- No environment toggles found that would suppress results given valid fields. The outcome is driven by schema mismatch, not toggles.

## 7) Builder Invocation (observed)
- The builder is invoked but receives an empty candidate list.
- It still writes the session index (`runs/<SID>/cases/index.json`) and the accounts index (`runs/<SID>/cases/accounts/index.json`) with `count: 0` and no items.
- No per‑account folders are created.

## 8) Conclusion & Root Cause
- Root cause: Schema mismatch. The detector expects flattened Stage‑A account fields (`past_due_amount`, `payment_status`, `account_status`, `days_late_7y`, etc.) or a `fields` mapping. The current `accounts_from_full.json` contains triad/structural metadata but no such keys, so the rules‑based detector finds no evidence and returns neutral decisions for all accounts.
- Secondary observation: If relevant bureau data exists under `triad_fields`, the current detector does not traverse it.

## Appendix (Evidence)
- Manifest snippets: see section 1 dumps.
- Schema peek: see section 2 output.
- Smoke/analyzer outputs: section 4 JSON and file listings.
- Detector logic references:
  - `backend/core/logic/report_analysis/problem_extractor.py`
  - `backend/core/logic/report_analysis/problem_detection.py` (EVIDENCE fields and evaluate_account_problem)

