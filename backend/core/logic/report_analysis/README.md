# Report Analysis

## Purpose
Extract structured sections from SmartCredit reports and categorize accounts for downstream strategy and letter generation.

## Pipeline position
Ingests the uploaded PDF and produces bureau-specific sections (`disputes`, `goodwill`, `inquiries`, etc.) consumed by strategy modules.

## Parsing flow (deterministic)
PyMuPDF text → Selective OCR (flag-gated) → Normalization → Deterministic Extractors → Case Store

## Files
- `__init__.py`: package marker.
- `analyze_report.py`: orchestrates deterministic parsing and post-processing; integrates Stage A adjudication results when enabled (separate stage).
  - Key function: `analyze_credit_report()` runs deterministic parsing and prepares data for downstream consumers.
  - Internal deps: `.report_parsing`, `.report_prompting`, `.report_postprocessing`, `backend.core.logic.utils.text_parsing`, `backend.core.logic.utils.inquiries`.
- `extract_info.py`: pull identity columns from the report.
  - Key functions: `extract_clean_name()`, `normalize_name_order()`, `extract_bureau_info_column_refined()`.
  - Internal deps: `backend.core.logic.utils.names_normalization`.
- `process_accounts.py`: convert analysis output into bureau payloads.
  - Key items: `Account` dataclass; functions `process_analyzed_report()` and `save_bureau_outputs()`; helpers `infer_hardship_reason()`, `infer_personal_impact()`, `infer_recovery_summary()`.
  - Internal deps: `backend.core.logic.utils.names_normalization`, `backend.core.logic.strategy.fallback_manager`, `backend.audit.audit`.
- `report_parsing.py`: read PDF text and helper for converting dicts.
  - Key functions: `bureau_data_from_dict()`.
  - Internal deps: none.
- `report_postprocessing.py`: clean and augment parsed data; may merge Stage A adjudication outputs if enabled.
  - Key functions: `_merge_parser_inquiries()`, `_sanitize_late_counts()`, `_cleanup_unverified_late_text()`, `_inject_missing_late_accounts()`, `validate_analysis_sanity()`.
  - Internal deps: `backend.core.logic.utils` modules.
- `report_prompting.py`: Stage A AI adjudication (separate from parsing); builds prompts and calls the AI client.
  - Key function: `call_ai_analysis()`.
  - Internal deps: `backend.core.services.ai_client`.

## Entry points
- `analyze_report.analyze_credit_report`
- `process_accounts.process_analyzed_report`

## Guardrails / constraints
- Sanitization helpers ensure PII is normalized and late-payment data is validated.
- Summaries should remain factual and neutral.

## Output fields
The JSON produced by this stage may include informational fields that are not
yet consumed by downstream modules:

- `confidence`: heuristic confidence score from AI adjudication (if enabled).
- `needs_human_review`: flag indicating analysis uncertainty.
- `missing_bureaus`: list of bureaus absent from the source report.

These fields are optional and safely ignored by letter generation and
instructions pipelines.

## Local development

Set the environment variable `ANALYSIS_DISABLE_CACHE=1` to bypass the
in-memory analysis cache and force a fresh analysis on each run.

## Manual trace cleanup

After Stage A export you can clear intermediate files for a session ID (SID)
while keeping the final artifacts. Run the helper script:

```bash
python scripts/cleanup_trace.py --sid <SID> --root .
```

Example PowerShell usage:

```powershell
# venv + project root
& .\.venv\Scripts\Activate.ps1
$env:PYTHONPATH = (Get-Location).Path

# Set SID
$SID = 'PUT-YOUR-SID-HERE'

# Run cleanup
python .\scripts\cleanup_trace.py --sid $SID --root .

# Verify
$blocks = ".\traces\blocks\$SID"
$acct   = "$blocks\accounts_table"
$texts  = ".\traces\texts\$SID"

"Remaining in accounts_table:"
Get-ChildItem $acct -Force | Format-Table Name,Length -Auto
"Texts dir exists? " + (Test-Path $texts)
```

Only `_debug_full.tsv`, `accounts_from_full.json`, and
`general_info_from_full.json` remain in the `accounts_table` folder and the
corresponding `texts/<SID>` directory is removed.

### Preserving per-account TSVs

By default the cleanup routine removes the per-account TSVs under
`accounts_table/per_account_tsv`. Set `KEEP_PER_ACCOUNT_TSV=1` in the
environment to whitelist this directory and retain the TSVs for debugging
purposes.
