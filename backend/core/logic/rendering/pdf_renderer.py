from __future__ import annotations

from pathlib import Path
from typing import Optional

import pdfkit
import time
from jinja2 import Environment, FileSystemLoader

from backend.api.config import get_app_config
from backend.assets.paths import templates_path
from backend.analytics.analytics_tracker import log_ai_stage, set_metric

_template_env: Environment | None = None


def ensure_template_env(base_template_dir: Optional[str] = None) -> Environment:
    """Return a Jinja2 environment rooted at ``base_template_dir``.

    The environment is cached so multiple calls reuse the same loader.
    """
    global _template_env
    base_dir = base_template_dir or templates_path("")
    if (
        _template_env is None
        or getattr(_template_env.loader, "searchpath", [None])[0] != base_dir
    ):
        _template_env = Environment(
            loader=FileSystemLoader(base_dir),
            trim_blocks=True,
            lstrip_blocks=True,
        )
    return _template_env


def normalize_output_path(path: str) -> str:
    """Normalize an output path for PDF generation.

    Ensures the path is absolute, ends with ``.pdf`` and that the parent
    directory exists.  The returned path is formatted using POSIX-style
    forward slashes so tests do not depend on the host platform.
    """
    p = Path(path).expanduser()
    if not p.is_absolute():
        p = Path.cwd() / p
    if p.suffix.lower() != ".pdf":
        p = p.with_suffix(".pdf")
    p = p.resolve()
    p.parent.mkdir(parents=True, exist_ok=True)
    return p.as_posix()


def render_html_to_pdf(
    html: str,
    output_path: str,
    *,
    wkhtmltopdf_path: Optional[str] = None,
    template_name: Optional[str] = None,
) -> None:
    """Render ``html`` to a PDF at ``output_path``.

    Parameters
    ----------
    html:
        The HTML string to convert.
    output_path:
        Desired file path for the resulting PDF. The path is normalized using
        :func:`normalize_output_path`.
    wkhtmltopdf_path:
        Optional path to the ``wkhtmltopdf`` executable. Defaults to the
        repository-wide configuration.
    """
    output_path = normalize_output_path(output_path)
    wkhtmltopdf = wkhtmltopdf_path or get_app_config().wkhtmltopdf_path
    start = time.perf_counter()
    try:
        config = pdfkit.configuration(wkhtmltopdf=wkhtmltopdf)
        options = {"quiet": ""}
        pdfkit.from_string(html, output_path, configuration=config, options=options)
        # Emit a simple ASCII message so the source file remains UTF-8 clean
        # and portable across operating systems.
        print(f"[INFO] PDF rendered: {output_path}")
    except Exception as e:  # pragma: no cover - rendering failures are logged
        print(f"[ERROR] Failed to render PDF: {e}")
    finally:
        elapsed = (time.perf_counter() - start) * 1000
        if template_name:
            set_metric(f"letter.render_ms.{template_name}", elapsed)
        log_ai_stage("render", 0, 0)
