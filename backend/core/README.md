# Core

## Purpose
Central domain layer coordinating analysis, strategy, letter generation, and final delivery.

## Pipeline position
Receives sanitized inputs from the API and drives the end‑to‑end credit repair workflow before handing outputs to rendering and email.

## Files
- `__init__.py`: package marker.
- `email_sender.py`: helpers for emailing generated PDFs.
  - Key functions: `collect_all_files()` gathers PDFs; `send_email_with_attachment()` sends messages via SMTP.
  - Internal deps: standard library only.
- `orchestrators.py`: high-level pipeline controller.
  - Key functions: `run_credit_repair_process()` executes intake→analysis→letters; `process_client_intake()`, `analyze_credit_report()`, `generate_strategy_plan()`, `generate_letters()`, `finalize_outputs()`, `extract_problematic_accounts_from_report()` expose individual stages.
  - Internal deps: `backend.core.logic.*`, `backend.audit.audit`, `backend.analytics.analytics_tracker`.

## Subfolders
- `logic/` – processing steps for [report analysis](logic/report_analysis/README.md), [strategy](logic/strategy/README.md), [letters](logic/letters/README.md), [rendering](logic/rendering/README.md), [compliance](logic/compliance/README.md), [utils](logic/utils/README.md), and [guardrails](logic/guardrails/README.md).
- `models/` – dataclasses representing clients, accounts, and strategy objects.
- `services/` – connectors such as the OpenAI client.
- `rules/` – YAML rule definitions loaded by compliance components.

## Entry points
- `orchestrators.run_credit_repair_process`
- `orchestrators.extract_problematic_accounts_from_report`
- `email_sender.send_email_with_attachment`

## Guardrails / constraints
- Handles PII; ensure secure storage and respect rule definitions in `rules/`.
