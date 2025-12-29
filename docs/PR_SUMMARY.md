# Title
Feature-based backend reorg: api/core/analytics/audit/assets + logic split into subpackages

## Overview
One short paragraph describing: PDF+email → analysis → explanations → strategy → letters (guardrails) → HTML/PDF + logs. Why the reorg (clarity, maintainability, imports stability).

## What changed
- Moved API to `backend/api/*` (app, admin, tasks, session_manager, config).
- Consolidated domain into `backend/core/*` (logic, models, services, rules, orchestrators, email_sender).
- Split `logic/` into subpackages:
  - report_analysis/, strategy/, letters/, rendering/, compliance/, utils/, guardrails/
- Separated analytics/audit under `backend/analytics/*`, `backend/audit/*`.
- Centralized assets under `backend/assets/{templates,static,fonts,data}` + `backend/assets/paths.py`.
- Moved examples to `examples/*`, legacy to `archive/*`.
- Updated imports to `backend.*` paths; added compatibility shims where needed.
- Centralized rules loading via `importlib.resources` from `backend.core.rules`.
- Repo hygiene: `.gitignore`, `pytest.ini`, `.pre-commit-config.yaml`, CI workflow.

## Breaking changes / migration notes
- Old imports like `logic.*`, `models.*`, `app`, `audit`, etc. must now use `backend.core.*`, `backend.api.*`, etc.
- Templates/data/fonts paths must go through `backend.assets.paths`.
- Tests now expect `DISABLE_PDF_RENDER=true` by default (see `pytest.ini`).

## How to run locally (quick start)
- Backend (PowerShell):
  ```
  .\.venv\Scripts\Activate.ps1
  set FLASK_DEBUG=1
  set DISABLE_PDF_RENDER=true
  python -m backend.api.app
  ```
- Frontend:
  ```
  cd frontend
  npm install
  npm run dev
  ```
- Smoke:
  ```
  python tools/import_sanity_check.py
  DISABLE_PDF_RENDER=true python -m pytest -q
  ```

## Test & CI
- Local: `DISABLE_PDF_RENDER=true pytest -q`
- CI: `.github/workflows/ci.yml` runs import smoke + tests on Python 3.11.

## Risks & mitigations
- Import drift → mitigated by mechanical rewrite + `tools/import_sanity_check.py`.
- Asset path regressions → `backend/assets/paths.py`.
- Rules file location → `importlib.resources` from `backend.core.rules`.

## Rollback plan
- Revert this PR; folders are moved via `git mv` preserving history.

## Follow-ups (nice-to-have)
- CRA alignment enhancements, soft-rule telemetry, clean demo assets, optional Celery path.

## Pre-Outcome Ingestion sign-off
- Staging canary observed for one cycle; no elevated `router.error`, `planner.error_count`, or `planner.sla_violations_total` metrics.
- Audit logs for planner transitions reviewed; events are deterministic and replay-safe.
