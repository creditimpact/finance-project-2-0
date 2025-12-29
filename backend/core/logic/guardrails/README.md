# Guardrails

## Purpose
Validate structured summaries and generated content to enforce neutral tone and prevent unsafe output.

## Pipeline position
Runs after client explanations are structured and before strategy or letters are finalized, ensuring only sanitized summaries flow downstream.

## Files
- `__init__.py`: package marker.
- `summary_validator.py`: validate structured summaries for required fields and safe tone.
  - Key function: `validate_structured_summaries()`.
  - Internal deps: `typing` and standard library.

## Entry points
- `summary_validator.validate_structured_summaries`

## Guardrails / constraints
- Designed to prevent PII leakage and disallowed phrasing before interacting with models or rendering letters.
