# Changelog

## 2025-08-12
### Summary
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

### Upgrade notes
- Old imports like `logic.*`, `models.*`, `app`, `audit`, etc. must now use `backend.core.*`, `backend.api.*`, etc.
- Templates/data/fonts paths must go through `backend.assets.paths`.


## Unreleased
### Removed
- Remove deprecated shims and aliases in audit, letter rendering, goodwill letters, instructions, and report analysis modules.
- Remove unused `logic.copy_documents` module.
