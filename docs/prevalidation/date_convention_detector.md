# Date Convention Detector

## What it does
- Scans each account's `raw_lines.json` before validation begins.
- Detects the language used for Two-Year Payment History month labels (Hebrew vs. English).
- Persists a single decision per run under `traces/general_info_from_full.json` so downstream validation can parse dates consistently.

## Inputs/outputs
- **Input:** Read-only access to each account's `raw_lines.json`.
- **Output:** Appends a JSON block to `traces/general_info_from_full.json` describing the detected date convention, including scope, convention, month language, confidence, evidence counts, and detector version.

## ENV flag
- Controlled by the `PREVALIDATION_DATE_CONVENTION_DETECTOR` environment variable (truthy to enable).

## Limitations
- Version 1 only distinguishes between Hebrew (`he`) and English (`en`) month labels.
- Detection relies solely on the Two-Year Payment History block; other sections are ignored.
- Does not modify existing validation logic or tolerances and must complete quickly.

## Future hooks
- Extend detection to disambiguate numeric-only month labels (e.g., `01/02`).
- Support bureau-specific conventions if required (per-bureau split).
- Broaden language coverage beyond Hebrew and English.
