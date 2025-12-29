"""HTML rendering utilities for instruction generation."""
from __future__ import annotations

from typing import Any

from backend.assets.paths import templates_path
from backend.core.logic.rendering import pdf_renderer
from backend.core.models.letter import LetterContext
from backend.telemetry.metrics import emit_counter


def render_instruction_html(
    context: LetterContext | dict[str, Any], template_path: str
) -> str:
    """Render the Jinja2 template with the provided context."""
    if not template_path:
        emit_counter("rendering.missing_template_path")
        raise ValueError("template_path is required")
    env = pdf_renderer.ensure_template_env(templates_path(""))
    template = env.get_template(template_path)
    return template.render(**context)


def build_instruction_html(
    context: LetterContext | dict[str, Any], template_path: str
) -> str:
    """Return the rendered instruction HTML."""
    return render_instruction_html(context, template_path)
