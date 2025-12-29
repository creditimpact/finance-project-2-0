"""Rendering utilities for dispute letters."""

from __future__ import annotations

from backend.assets.paths import templates_path
from backend.core.logic.rendering import pdf_renderer
from backend.core.models.letter import LetterArtifact, LetterContext
from backend.telemetry.metrics import emit_counter
from backend.core.letters import validators
from backend.core.letters.sanitizer import sanitize_rendered_html


def render_dispute_letter_html(
    context: LetterContext | object, template_path: str
) -> LetterArtifact:
    """Render the dispute letter HTML using the Jinja template."""

    if not template_path:
        emit_counter("rendering.missing_template_path")
        raise ValueError("template_path is required")

    ctx = context.to_dict() if hasattr(context, "to_dict") else context
    missing = validators.validate_substance(template_path, ctx)
    if missing:
        for field in missing:
            emit_counter(f"validation.failed.{template_path}.{field}")
        raise ValueError("substance checklist failed")

    env = pdf_renderer.ensure_template_env(templates_path(""))
    template = env.get_template(template_path)
    html = template.render(**ctx)
    emit_counter(f"letter_template_selected.{template_path}")
    html, _ = sanitize_rendered_html(html, template_path, ctx)
    return LetterArtifact(html=html)


__all__ = ["render_dispute_letter_html"]
