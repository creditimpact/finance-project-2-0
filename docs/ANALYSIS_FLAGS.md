# Analysis Flags

Environment variables that tweak the credit report analysis flow. Defaults are shown in parentheses.

- `ANALYSIS_CHUNK_BY_BUREAU` (`1`): split report text by bureau before sending to the model. Set to `0` to analyze the full report at once.
- `ANALYSIS_INJECT_HEADINGS` (`1`): if enabled, detected account headings are prepended to the prompt.
- `ANALYSIS_CACHE_ENABLED` (`1`): enables caching of analysis results.
- `ANALYSIS_DEBUG_STORE_RAW` (`0`): when true, store the raw model response and debugging excerpts alongside the JSON output.
- `ANALYSIS_MAX_REMEDIATION_PASSES` (`2`): number of remediation attempts after schema validation errors.

Flags are defined in `backend/core/logic/report_analysis/flags.py` and can be adjusted via environment variables or by modifying `FLAGS` at runtime.
