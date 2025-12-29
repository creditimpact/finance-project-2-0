# Letters

## Purpose
Prepare dispute and goodwill letters using client data, strategy outputs, and LLM prompting.

## Pipeline position
Takes bureau payloads and strategy recommendations to produce letter drafts, apply guardrails and compliance checks, and render final documents.

## Files
- `__init__.py`: package marker.
- `dispute_preparation.py`: deduplicate accounts and match inquiries.
  - Key functions: `dedupe_disputes()`, `prepare_disputes_and_inquiries()`.
  - Internal deps: `backend.core.logic.strategy.fallback_manager`, `backend.core.logic.utils.names_normalization`, `backend.core.models`.
- `explanations_normalizer.py`: sanitize and structure user explanations.
  - Key functions: `_redact()`, `sanitize()`, `extract_structured()`.
  - Internal deps: `backend.core.logic.utils.json_utils`.
- `generate_custom_letters.py`: create custom dispute letters and save PDFs.
  - Key functions: `_pdf_config()`, `call_gpt_for_custom_letter()`, `generate_custom_letter()`, `generate_custom_letters()`.
  - Internal deps: `backend.core.logic.guardrails`, `backend.core.logic.utils.pdf_ops`, `backend.api.session_manager`, `backend.core.logic.strategy.summary_classifier`, `backend.core.logic.compliance.rules_loader`, `backend.audit.audit`.
- `generate_debt_validation_letters.py`: craft debt validation letters for collectors.
  - Key functions: `generate_debt_validation_letter()`, `generate_debt_validation_letters()`.
  - Internal deps: `backend.core.letters.router`, `backend.core.letters.sanitizer`, `backend.core.letters.validators`.
- `generate_goodwill_letters.py`: orchestrate goodwill letter generation.
  - Key functions: `generate_goodwill_letter_with_ai()`, `generate_goodwill_letters()`.
  - Internal deps: `backend.core.logic.letters.goodwill_preparation`, `backend.core.logic.letters.goodwill_prompting`, `backend.core.logic.letters.goodwill_rendering`, `backend.core.logic.rendering.pdf_renderer`, `backend.core.logic.utils.pdf_ops`, `backend.core.logic.compliance.compliance_pipeline`.
- `goodwill_preparation.py`: select goodwill candidates and summarize accounts.
  - Key functions: `select_goodwill_candidates()`, `prepare_account_summaries()`.
  - Internal deps: `backend.core.logic.utils.names_normalization`.
- `goodwill_prompting.py`: craft LLM prompts for goodwill letters.
  - Key function: `generate_goodwill_letter_draft()`.
  - Internal deps: `backend.core.logic.utils.pdf_ops`, `backend.core.logic.utils.json_utils`.
- `goodwill_rendering.py`: render goodwill letters and apply compliance checks.
  - Key functions: `load_creditor_address_map()`, `render_goodwill_letter()`.
  - Internal deps: `backend.assets.paths`, `backend.core.logic.rendering.pdf_renderer`, `backend.core.logic.utils.file_paths`, `backend.core.logic.utils.note_handling`, `backend.core.logic.compliance.compliance_pipeline`.
- `gpt_prompting.py`: shared GPT helpers for dispute letters.
  - Key function: `call_gpt_dispute_letter()`.
  - Internal deps: `backend.core.services.ai_client`.
- `letter_generator.py`: high-level dispute letter orchestration.
  - Key functions: `call_gpt_dispute_letter()` (proxy), `generate_all_dispute_letters_with_ai()`.
  - Internal deps: `backend.core.logic.strategy.strategy_engine`, `backend.core.logic.rendering`, `backend.core.logic.compliance.compliance_pipeline`, `backend.core.logic.guardrails.summary_validator`.
- `outcomes_store.py`: record and retrieve letter outcomes.
  - Key functions: `record_outcome()`, `get_outcomes()`.
  - Internal deps: none significant.

## Entry points
- `generate_custom_letters.generate_custom_letters`
- `generate_goodwill_letters.generate_goodwill_letters`
- `generate_debt_validation_letters.generate_debt_validation_letters`
- `letter_generator.generate_all_dispute_letters_with_ai`
- `dispute_preparation.prepare_disputes_and_inquiries`

## Guardrails / constraints
- Generated letters must maintain a neutral tone and avoid admissions of fault.
- Compliance checks (`compliance_pipeline`) sanitize client info and disputes before rendering.
