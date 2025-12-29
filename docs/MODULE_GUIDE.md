# Module Guide

| Module | Role | Public API | Key Dependencies |
| ------ | ---- | ---------- | ---------------- |
| `backend/core/orchestrators.py` | Coordinates the end-to-end credit repair pipeline. | `run_credit_repair_process`, `extract_problematic_accounts_from_report` → `BureauPayload` | `backend.core.logic.*`, `session_manager`, `analytics_tracker` |
| `backend/core/models/` | Dataclasses representing accounts, bureaus, letters, client metadata and strategy plans. | `Account`, `ProblemAccount`, `BureauAccount`, `BureauSection`, `BureauPayload`, `ClientInfo`, `ProofDocuments`, `LetterAccount`, `LetterContext`, `LetterArtifact`, `Recommendation`, `StrategyItem`, `StrategyPlan` | `dataclasses`, `typing` |
| `backend/core/logic/` | Business logic: report parsing, strategy generation, compliance checks and PDF rendering. | `analyze_credit_report`, `StrategyGenerator`, `run_compliance_pipeline`, `pdf_ops.convert_txts_to_pdfs` | OpenAI API, PDF utilities |
| `backend/core/services/` | Lightweight wrappers for external integrations. | `AIClient`, email utilities | `requests`, `smtplib` |
| `backend/assets/templates/` | Jinja2 letter templates rendered into HTML/PDF. | N/A – consumed by letter generation code | `Jinja2`, `backend.core.logic.utils.pdf_ops` |

All GPT-enabled functions now require an explicit `AIClient` instance; no global
fallback is provided. The orchestrators build a single client and inject it into
downstream calls.
