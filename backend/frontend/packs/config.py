"""Environment-driven configuration for frontend review pack generation."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from backend.core.paths.frontend_review import ensure_frontend_review_dirs


@dataclass(frozen=True, slots=True)
class FrontendStageConfig:
    """Resolved filesystem locations for the frontend review stage."""

    stage_dir: Path
    packs_dir: Path
    responses_dir: Path
    uploads_dir: Path
    index_path: Path


def _is_within(child: Path, parent: Path) -> bool:
    """Return ``True`` when ``child`` is located under ``parent``."""

    try:
        child.relative_to(parent)
    except ValueError:
        return False
    return True


def _resolve_review_path(
    run_dir: Path,
    env_name: str,
    *,
    canonical: Path,
    review_dir: Path,
    require_descendant: bool = False,
) -> Path:
    """Resolve an environment override constrained to the review tree.

    The canonical review directories live under ``runs/<sid>/frontend/review``. When an
    override points inside ``frontend/`` but outside the ``review`` subtree, the canonical
    location is used instead. Absolute paths outside of the run directory are preserved so
    operators can redirect to custom targets if needed.
    """

    value = os.getenv(env_name)
    if not value:
        return canonical

    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = run_dir / candidate

    frontend_base = review_dir.parent
    if _is_within(candidate, frontend_base) and not _is_within(candidate, review_dir):
        return canonical

    if require_descendant and _is_within(candidate, review_dir) and candidate == review_dir:
        return canonical

    return candidate


def load_frontend_stage_config(run_dir: Path | str) -> FrontendStageConfig:
    """Load the configured frontend review stage paths."""

    base_dir = Path(run_dir)
    canonical = ensure_frontend_review_dirs(str(base_dir))

    review_dir = Path(canonical["review_dir"])
    packs_default = Path(canonical["packs_dir"])
    responses_default = Path(canonical["responses_dir"])
    uploads_default = Path(canonical["uploads_dir"])
    index_default = review_dir / "index.json"

    stage_dir = _resolve_review_path(
        base_dir,
        "FRONTEND_PACKS_STAGE_DIR",
        canonical=review_dir,
        review_dir=review_dir,
    )
    packs_dir = _resolve_review_path(
        base_dir,
        "FRONTEND_PACKS_DIR",
        canonical=packs_default,
        review_dir=review_dir,
        require_descendant=True,
    )
    responses_dir = _resolve_review_path(
        base_dir,
        "FRONTEND_PACKS_RESPONSES_DIR",
        canonical=responses_default,
        review_dir=review_dir,
        require_descendant=True,
    )
    uploads_dir = _resolve_review_path(
        base_dir,
        "FRONTEND_PACKS_UPLOADS_DIR",
        canonical=uploads_default,
        review_dir=review_dir,
        require_descendant=True,
    )
    index_path = _resolve_review_path(
        base_dir,
        "FRONTEND_PACKS_INDEX",
        canonical=index_default,
        review_dir=review_dir,
    )

    return FrontendStageConfig(
        stage_dir=stage_dir,
        packs_dir=packs_dir,
        responses_dir=responses_dir,
        uploads_dir=uploads_dir,
        index_path=index_path,
    )

