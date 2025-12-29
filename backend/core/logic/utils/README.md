# Utils

## Purpose
Shared helpers for parsing, normalization, JSON handling, and file operations used across logic modules.

## Pipeline position
Utilities are invoked by analysis, strategy, letter generation, and rendering steps for common tasks like PDF parsing and name normalization.

## Files
- `__init__.py`: package marker.
- `bootstrap.py`: environment bootstrap helpers.
  - Key functions: `get_current_month()`, `extract_all_accounts()`.
  - Internal deps: standard library.
- `file_paths.py`: safe filename utilities.
  - Key function: `safe_filename()`.
  - Internal deps: `re`.
- `inquiries.py`: parse inquiry blocks from raw text.
  - Key function: `extract_inquiries()`.
  - Internal deps: `re`.
- `json_utils.py`: robust JSON parsing.
  - Key functions: `_basic_clean()`, `_repair_json()`, `parse_json()`.
  - Internal deps: `json`, `logging`.
- `names_normalization.py`: normalize creditor and bureau names.
  - Key functions: `normalize_creditor_name()`, `normalize_bureau_name()`.
  - Internal deps: standard library; defines `BUREAUS` constant.
- `note_handling.py`: build client address lines.
  - Key function: `get_client_address_lines()`.
  - Internal deps: none.
- `pdf_ops.py`: PDF utilities and document text aggregation.
  - Key functions: `convert_txts_to_pdfs()`, `gather_supporting_docs()`, `gather_supporting_docs_text()`.
  - Internal deps: `pdfminer`/`pdfkit` style libraries.
- `report_sections.py`: helpers for filtering and summarizing report sections.
  - Key functions: `filter_sections_by_bureau()`, `extract_summary_from_sections()`.
  - Internal deps: `.names_normalization`, `.text_parsing`.
- `text_parsing.py`: generic text parsing and late-payment extraction.
  - Key functions: `extract_account_blocks()`, `parse_late_history_from_block()`, `extract_late_history_blocks()`, `has_late_indicator()`, `enforce_collection_status()`.
  - Internal deps: `re`.

## Entry points
- `json_utils.parse_json`
- `names_normalization.normalize_creditor_name`
- `pdf_ops.convert_txts_to_pdfs`

## Guardrails / constraints
- Utilities should remain side-effect free and avoid retaining PII beyond processing needs.
