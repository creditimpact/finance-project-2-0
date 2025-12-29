# Assets

## Purpose
Provide templates, static files, fonts, and data fixtures used during rendering and letter generation.

## Pipeline position
Referenced by rendering modules to locate HTML templates, fonts for PDF conversion, and helper data such as creditor addresses.

## Files
- `paths.py`: resolve asset locations.
  - Key functions: `templates_path()`, `data_path()`, `fonts_path()`, `static_path()`.
  - Internal deps: standard library `pathlib`.

## Subfolders
- `templates/` – HTML templates for letters and instructions.
- `static/` – CSS, images, and other assets bundled into PDFs.
- `fonts/` – font files required for PDF rendering.
- `data/` – JSON fixtures (e.g., creditor address maps); contains no secrets.

## Entry points
- `paths.templates_path`
- `paths.data_path`

## Guardrails / constraints
- Keep `data/` free of credentials; set `DISABLE_PDF_RENDER=1` when fonts are unavailable.
