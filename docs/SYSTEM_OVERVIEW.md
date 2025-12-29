# System Overview

This document outlines the high-level architecture of the credit repair pipeline.

The orchestrators module (`backend/core/orchestrators.py`) serves as the single entry point for the backend.  It coordinates the following phases:

1. **Intake** – `process_client_intake` collects client information and initial session state.
2. **Analysis** – `analyze_credit_report` ingests the uploaded report and extracts bureau data.
2.5 **Tri-merge** – [`normalize_and_match` and `compute_mismatches`](tri_merge/README.md) מנרמלים ומשווים נתוני לשכות אשראי לאיתור פערים.
3. **Strategy** – `generate_strategy_plan` merges classification results and bureau findings into a plan.
4. **Letters** – dedicated generators create dispute or goodwill letters.  Draft HTML passes through the compliance pipeline and the PDF renderer (`backend.core.logic.compliance`, `backend.core.logic.utils.pdf_ops`) before returning artifacts.
5. **Finalization** – orchestrators save artifacts, record analytics, and send notifications to finish the workflow.

The orchestrators construct a single `AIClient` at startup and pass it to all
LLM-powered functions, eliminating hidden global dependencies.

Compliance checks and PDF rendering live inside `backend/core/logic` and are invoked during letter generation to ensure all outbound documents meet guardrails and are rendered to PDF.
