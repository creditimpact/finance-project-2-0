# API

## Purpose
Expose Flask endpoints for uploading credit reports, collecting client explanations, and dispatching Celery tasks.

## Pipeline position
Receives the PDF report and optional email, persists session data, and triggers core orchestration. Returns JSON summaries and task results.

## Files
- `__init__.py`: package marker.
- `admin.py`: admin dashboard and download routes.
  - Key functions: `login()` authenticates admin users; `index()` lists client folders and analytics; `download_client()` and `download_analytics()` stream archives.
  - Internal deps: `backend.api.config`, `backend.api.telegram_alert`.
- `app.py`: main API blueprint and route handlers.
  - Key functions: `start_process()` saves uploads and launches analysis; `explanations_endpoint()` stores sanitized summaries; `get_summaries()` returns structured data; `create_app()` assembles the Flask app.
  - Internal deps: `backend.api.tasks`, `backend.api.session_manager`, `backend.core.orchestrators`, `backend.core.models`.
- `config.py`: environment-driven settings.
  - Key items: `AppConfig` dataclass; helpers `get_app_config()` and `get_ai_config()`.
  - Internal deps: `backend.core.services.ai_client`.
- `session_manager.py`: JSON-backed session and intake stores.
  - Key functions: `set_session()`, `get_session()`, `update_session()`, `update_intake()`, `get_intake()`.
  - Internal deps: `backend.assets.paths`.
- `tasks.py`: Celery task definitions for background processing.
  - Key functions: `extract_problematic_accounts()` parses reports; `process_report()` runs the full pipeline.
  - Internal deps: `backend.core.orchestrators`, `backend.core.models`.
- `telegram_alert.py`: logs admin login events.
  - Key function: `send_admin_login_alert()`.

## Entry points
- `app.create_app`
- `app.start_process`
- `app.explanations_endpoint`
- `app.get_summaries`
- `admin.login`
- `tasks.extract_problematic_accounts`
- `tasks.process_report`

## Guardrails / constraints
- Session files store sanitized data; raw explanations remain in intake-only storage.
- Secrets and API keys must come from environment variables.

## Authentication & throttling
- Set `API_AUTH_TOKENS` to a comma-separated list of bearer tokens. When provided, requests must include `Authorization: Bearer <token>`.
- Configure `API_RATE_LIMIT_PER_MINUTE` to control how many requests a token or IP may make per minute. Exceeding the limit returns HTTP `429`.
