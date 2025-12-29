"""Routes for serving run asset files."""
from __future__ import annotations

import os
from pathlib import Path

from flask import Blueprint, abort, current_app, send_file
from werkzeug.utils import safe_join

bp = Blueprint("run_assets", __name__)


def _ensure_under_runs_root(full_path: str | os.PathLike[str], runs_root: str | os.PathLike[str]) -> None:
    """Abort if ``full_path`` escapes the configured runs root."""
    root_real = Path(runs_root).resolve()
    full_real = Path(full_path).resolve()

    try:
        full_real.relative_to(root_real)
    except ValueError:
        abort(403)


@bp.get("/runs/<sid>/<path:relpath>")
def get_run_asset(sid: str, relpath: str):
    runs_root = os.environ.get("RUNS_ROOT") or current_app.config.get("RUNS_ROOT")
    if not runs_root:
        abort(500, "RUNS_ROOT not configured")

    relpath = relpath.replace("\\", "/")

    base_dir = safe_join(runs_root, sid)
    if base_dir is None:
        abort(404)

    full_path = safe_join(base_dir, relpath)
    if full_path is None:
        abort(404)

    _ensure_under_runs_root(full_path, runs_root)

    if not os.path.exists(full_path):
        abort(404)

    return send_file(full_path, conditional=True)
