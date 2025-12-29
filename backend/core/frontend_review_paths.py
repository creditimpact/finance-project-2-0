"""Canonical paths for frontend review artifacts."""

from __future__ import annotations

import os
from typing import Dict


__all__ = [
    "get_frontend_review_paths",
    "ensure_frontend_review_dirs",
]


def get_frontend_review_paths(run_dir: str) -> Dict[str, str]:
    """Return canonical frontend review locations for ``run_dir``."""

    frontend_base = os.path.join(run_dir, "frontend")
    review_dir = os.path.join(frontend_base, "review")
    packs_dir = os.path.join(review_dir, "packs")
    responses_dir = os.path.join(review_dir, "responses")
    uploads_dir = os.path.join(review_dir, "uploads")
    index_path = os.path.join(review_dir, "index.json")
    legacy_index = os.path.join(frontend_base, "index.json")
    return {
        "frontend_base": frontend_base,
        "review_dir": review_dir,
        "packs_dir": packs_dir,
        "responses_dir": responses_dir,
        "uploads_dir": uploads_dir,
        "legacy_index": legacy_index,
        "index": index_path,
    }


def ensure_frontend_review_dirs(run_dir: str) -> Dict[str, str]:
    """Ensure canonical frontend review directories exist for ``run_dir``."""

    paths = get_frontend_review_paths(run_dir)
    os.makedirs(paths["review_dir"], exist_ok=True)
    os.makedirs(paths["packs_dir"], exist_ok=True)
    os.makedirs(paths["responses_dir"], exist_ok=True)
    os.makedirs(paths["uploads_dir"], exist_ok=True)
    return paths
