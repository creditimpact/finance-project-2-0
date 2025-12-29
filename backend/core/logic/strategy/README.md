# Strategy

## Purpose
Derive action plans and recommendations from analyzed report data and client summaries.

## Pipeline position
Consumes structured report sections and sanitized summaries to classify accounts, generate strategic plans, and merge strategist output before letter generation.

## Files
- `__init__.py`: package marker.
- `fallback_manager.py`: choose fallback actions when strategist recommendations are missing or unrecognized.
  - Key functions: `_get()`, `determine_fallback_action()`.
  - Internal deps: `backend.core.logic.compliance.constants`.
- `generate_strategy_report.py`: wrapper around the AI strategist.
  - Key class: `StrategyGenerator` with methods `generate()` and `save_report()`.
  - Internal deps: `backend.core.services.ai_client`, `backend.core.logic.guardrails`, `backend.core.logic.utils.json_utils`.
- `strategy_engine.py`: build dispute items and assemble strategy documents.
  - Key functions: `_lookup_account()`, `_build_dispute_items()`, `generate_strategy()`.
  - Internal deps: `backend.api.session_manager`, `backend.core.logic.compliance.rules_loader`, `backend.core.logic.letters.outcomes_store`, `backend.core.logic.guardrails.summary_validator`.
- `strategy_merger.py`: align strategist output with parsed bureau data and handle fallbacks.
  - Key functions: `merge_strategy_outputs()`, `handle_strategy_fallbacks()`, `merge_strategy_data()`.
  - Internal deps: `backend.core.logic.compliance.constants`, `backend.core.logic.strategy.fallback_manager`, `backend.core.logic.utils.names_normalization`, `backend.core.models`.
- `summary_classifier.py`: categorize summaries to drive strategy decisions.
  - Key functions: `_heuristic_category()`, `classify_client_summary()`.
  - Internal deps: `backend.core.services.ai_client`, `backend.core.logic.utils.json_utils`.

## Entry points
- `generate_strategy_report.StrategyGenerator.generate`
- `strategy_engine.generate_strategy`
- `strategy_merger.merge_strategy_data`
- `summary_classifier.classify_client_summary`
- `fallback_manager.determine_fallback_action`

## Classification cache
`summary_classifier` caches classification results using a key of session id,
account id, summary hash, client state, and rules version. Entries expire when
evicted by TTL or size limits. See
[`tests/test_classification_cache.py`](../../../../tests/test_classification_cache.py)
for coverage of the cache behavior.

## Guardrails / constraints
- Recommendations must comply with `compliance.constants.VALID_ACTION_TAGS` and neutral phrasing rules.
