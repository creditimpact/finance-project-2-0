# Rendering

## Purpose
Transform prepared letter and instruction content into final HTML and PDF artifacts.

## Pipeline position
Consumes text from letter generators and strategy modules, produces client-facing documents, and writes them to disk for emailing.

## Files
- `__init__.py`: package marker.
- `instruction_data_preparation.py`: assemble context for instruction packets.
  - Key functions: `extract_clean_name()`, `generate_account_action()`, `prepare_instruction_data()`.
  - Internal deps: `backend.core.logic.utils.file_paths`, `backend.core.logic.utils.names_normalization`.
- `instruction_renderer.py`: turn instruction context into HTML.
  - Key functions: `render_instruction_html()`, `build_instruction_html()`.
  - Internal deps: `backend.core.models.letter`.
- `instructions_generator.py`: orchestrate instruction generation and PDF conversion.
  - Key functions: `get_logo_base64()`, `generate_instruction_file()`, `render_pdf_from_html()`, `save_json_output()`.
  - Internal deps: `backend.core.logic.rendering.pdf_renderer`, `backend.core.logic.utils.pdf_ops`.
- `letter_rendering.py`: create dispute letter HTML artifacts.
  - Key function: `render_dispute_letter_html()`.
  - Internal deps: `backend.core.models.letter`, `backend.core.logic.utils.note_handling`.
- `pdf_renderer.py`: render HTML strings to PDF.
  - Key functions: `ensure_template_env()`, `normalize_output_path()`, `render_html_to_pdf()`.
  - Internal deps: `jinja2` templates, `pdfkit` or similar libraries.

## Entry points
- `instructions_generator.generate_instruction_file`
- `letter_rendering.render_dispute_letter_html`
- `pdf_renderer.render_html_to_pdf`

## Guardrails / constraints
- Output files should exclude PII beyond what is required and respect compliance checks from upstream modules.
