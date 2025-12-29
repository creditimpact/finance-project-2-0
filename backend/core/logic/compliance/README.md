# Compliance

## Purpose
Apply regulatory rules, sanitize model outputs, and validate uploaded documents.

## Pipeline position
Runs alongside strategy and letter generation to ensure recommendations and documents meet legal requirements and avoid PII leaks.

## Files
- `__init__.py`: package marker.
- `compliance_adapter.py`: normalize strategist output and client data.
  - Key functions: `sanitize_disputes()`, `sanitize_client_info()`, `adapt_gpt_output()`.
  - Internal deps: `backend.core.logic.compliance.constants`.
- `compliance_pipeline.py`: orchestrate compliance checks.
  - Key function: `run_compliance_pipeline()`.
  - Internal deps: `backend.core.logic.compliance.rule_checker`, `backend.core.logic.compliance.rules_loader`.
- `constants.py`: enumerations and helpers for action tags.
  - Key items: `FallbackReason`, `StrategistFailureReason`, `normalize_action_tag()`.
  - Internal deps: none beyond standard library.
- `rule_checker.py`: evaluate text against rulebook and append state clauses.
  - Key items: `RuleViolation` typed dict; function `check_letter()`.
  - Internal deps: `backend.core.logic.compliance.rules_loader`, `backend.core.models.letter`.
- `rules_loader.py`: load YAML rule definitions and neutral phrases.
  - Key functions: `_load_yaml()`, `load_rules()`, `load_neutral_phrases()`, `get_neutral_phrase()`, `load_state_rules()`.
  - Internal deps: `yaml`.
- `upload_validator.py`: ensure uploaded PDFs are safe and stored under session folders.
  - Key functions: `is_valid_filename()`, `contains_suspicious_pdf_elements()`, `is_safe_pdf()`, `move_uploaded_file()`.
  - Internal deps: standard library (`pathlib`).

## Entry points
- `compliance_pipeline.run_compliance_pipeline`
- `rule_checker.check_letter`
- `rules_loader.load_rules`
- `upload_validator.is_safe_pdf`

## Guardrails / constraints
- Enforces PII masking and tone neutrality.
- Rules must remain current with regulation updates; PDFs are discarded after processing.
