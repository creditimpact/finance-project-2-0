# Audit

## Purpose
Record structured logs and export trace files for each processing run.

## Pipeline position
Receives step and account events from orchestrators and letters modules, then writes audit JSON and optional trace breakdowns.

## Files
- `__init__.py`: package marker.
- `audit.py`: structured audit logger.
  - Key items: `AuditLevel` enum; class `AuditLogger` with methods `log_step()`, `log_account()`, `log_error()`, `save()`; helper `create_audit_logger()`.
  - Internal deps: standard library only.
- `trace_exporter.py`: export detailed strategist traces and per-account breakdowns.
  - Key functions: `export_trace_file()`, `export_trace_breakdown()`.
  - Internal deps: `backend.core.models.strategy`.

## Entry points
- `audit.create_audit_logger`
- `trace_exporter.export_trace_file`

## Guardrails / constraints
- Logs may include account details; store outputs securely and avoid sharing externally.
