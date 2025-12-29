# Finance Platform

This project ingests a PDF credit report and optional email, analyzes the report, highlights issues with explanations, builds a strategy, and generates guardrail-compliant letters as HTML/PDF alongside audit logs.
Legacy materializer has been removed; use `/api/account/<session>/<account_id>` for per-account case views.

## Data Layer

See [docs/case_store.md](docs/case_store.md) for per-account case shape, per-bureau fields, Stage-A artifacts, and the Account View API.

## Project Structure

```
backend/
  api/
  core/
    logic/
      report_analysis/
      strategy/
      letters/
      rendering/
      compliance/
      utils/
      guardrails/
  analytics/
  audit/
  assets/
    templates/
    static/
    fonts/
    data/
frontend/
tools/
scripts/
tests/
docs/
archive/
examples/
```

## System Map (Directory Guide)

### Table of Contents
- [archive](#archive)
- [backend](#backend)
- [docs](#docs)
- [frontend](#frontend)
- [scripts](#scripts)
- [services](#services)
- [tests](#tests)
- [tools](#tools)

### How pieces connect
- Intake/API: [`backend/api`](backend/api/README.md) accepts the PDF report and optional email, storing session data.
- Core processing: [`backend/core`](backend/core/README.md) orchestrates analysis via `logic/report_analysis`, builds strategies in `logic/strategy`, generates letters through `logic/letters`, and enforces rules from `logic/compliance` and `logic/guardrails`.
- Rendering & assets: [`backend/core/logic/rendering`](backend/core/logic/rendering/README.md) converts text to HTML/PDF using templates and fonts from [`backend/assets`](backend/assets/README.md).
- Audit & analytics: [`backend/audit`](backend/audit/README.md) captures structured logs while [`backend/analytics`](backend/analytics/README.md) records run metrics.
- Frontend & tooling: [`frontend`](frontend/README.md) provides the React UI; [`scripts`](scripts) and [`tools`](tools/README.md) host developer utilities; reference docs live in [`docs`](docs) and legacy materials in [`archive`](archive/README.md); service stubs sit under [`services`](services); tests reside in [`tests`](tests).

### archive
Legacy documentation and sample artifacts retained for historical reference. See [archive/README.md](archive/README.md) for file-level details.

### backend
Python backend powering ingestion, analysis, strategy generation, letter creation, analytics, and auditing. Key subpackages include [api](backend/api/README.md), [core](backend/core/README.md), [analytics](backend/analytics/README.md), [audit](backend/audit/README.md), and [assets](backend/assets/README.md).

### docs
Reference materials such as data models, stage guides, and contributing
instructions. Includes pipeline docs like [Stage 2.5](docs/STAGE_2_5.md),
[Stage 3 Hardening](docs/stage3_hardening.md), and the
[Case-First Debug Runbook](docs/casefirst-debug.md). Deterministic escalation
policy details live in
[docs/validation_reasoning.md](docs/validation_reasoning.md). The
mismatch-to-tag rulebook lives in
[`docs/rulebook`](docs/rulebook/README.md). No dedicated README; browse files
within [`docs`](docs) for deeper documentation.

### frontend
React client that interacts with the API to upload reports and display results. See [frontend/README.md](frontend/README.md) for build and run instructions.

### scripts
Standalone Python utilities for maintenance tasks (encoding fixes, data exports). No README; inspect individual scripts in [`scripts`](scripts).

### services
Lightweight connectors to external services (e.g., OpenAI client). This folder currently lacks a README; see [`services`](services) for modules.

### tests
Automated tests exercising API endpoints and core logic. The [`tests`](tests) folder has no README; individual test files provide context.

### tools
Developer-facing command-line helpers and checks. See [tools/README.md](tools/README.md) for usage examples.

## Getting Started (Local)

### Backend (PowerShell)

```powershell
.\.venv\Scripts\Activate.ps1
set FLASK_DEBUG=1
set DISABLE_PDF_RENDER=true
python -m backend.api.app
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

## Manual Scenario

1. Open http://localhost:5173.
2. Upload a credit report PDF (email optional).
3. Review the analysis and strategy.
4. Letters are written to the `examples/` folder or a configured output path.

## Environment Variables

Provide these in a `.env` file:

- `OPENAI_API_KEY` – required for LLM calls.
- `ADMIN_PASSWORD` – password for admin endpoints.
- `OPENAI_BASE_URL` – optional override for OpenAI's URL.
- `CLASSIFY_CACHE_ENABLED` – set to `0` to disable summary classification caching (default `1`).
- `CLASSIFY_CACHE_MAXSIZE` – maximum entries for the summary classification cache (default `5000`).
- `CLASSIFY_CACHE_TTL_SEC` – expiration in seconds for cached classifications; `0` disables TTL.
- `ENABLE_PLANNER` – set to `0` to bypass the planner and execute the tactical
  strategy directly (default `0`).
- `PLANNER_CANARY_PERCENT` – percentage of sessions that invoke the planner
  when enabled (default `100`). Lower this value or disable `ENABLE_PLANNER` to
  roll back to the pre-planner pipeline.
- `ENABLE_PLANNER_PIPELINE` – gate the planner between router candidate and
  finalize stages (default `0`). Set to `0` to keep the legacy router order.
- `PLANNER_PIPELINE_CANARY_PERCENT` – percentage of accounts that follow the
  planner pipeline when enabled (default `100`).
- `ENABLE_FIELD_POPULATION` – controls automatic field filling during
  finalization (default `1`). Set to `0` to bypass population.
- `FIELD_POPULATION_CANARY_PERCENT` – percentage of accounts that run field
  fillers when enabled (default `100`). Lower this or disable
  `ENABLE_FIELD_POPULATION` to roll back.
- `ENABLE_OBSERVABILITY_H` – toggles analytics metrics and dashboards
  (default `1`). Set to `0` to revert to baseline metrics.
- `ENABLE_BATCH_RUNNER` – allows the batch analytics job runner to accept
  jobs via the API (default `1`). Set to `0` to disable the endpoint.

Secrets are never committed to the repository.

## Classification Cache

The stage 2 summary classifier caches each account's classification to avoid
repeated LLM calls. Keys combine `session_id`, `account_id`, the structured
summary hash, client `state`, and the active rules version. Cache entries are
evicted on manual invalidation, TTL expiry, or when the least recently used
record exceeds `CLASSIFY_CACHE_MAXSIZE`.

**Environment flags**
- `CLASSIFY_CACHE_ENABLED` – disable caching when set to `0` (default `1`).
- `CLASSIFY_CACHE_MAXSIZE` – maximum number of cached classifications.
- `CLASSIFY_CACHE_TTL_SEC` – seconds before a cache entry expires; `0` keeps entries indefinitely.

**Production note:** This is an in-memory, per-process cache. Deployments with
multiple worker processes keep separate caches, so hit rates decrease as worker
counts grow.

## PDF Rendering

PDF generation is disabled by default with `DISABLE_PDF_RENDER=true`. To enable PDFs, install [wkhtmltopdf](https://wkhtmltopdf.org/) and unset the flag.

## Folder READMEs

See the README files inside `backend/api`, `backend/core`, `backend/analytics`, `backend/audit`, `backend/assets`, and other folders for more details.

## Tests (optional for now)

```bash
DISABLE_PDF_RENDER=true python -m pytest -q
# or
python tools/import_sanity_check.py
```

## Contributing / Changelog

- [Contributing Guide](docs/CONTRIBUTING.md)
- [Changelog](CHANGELOG.md)

## Pre-commit & Tests

- Install hooks: `pip install pre-commit && pre-commit install`
- Run import smoke: `python tools/import_sanity_check.py`
- Run tests: `DISABLE_PDF_RENDER=true python -m pytest -q`
