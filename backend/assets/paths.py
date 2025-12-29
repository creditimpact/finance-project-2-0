"""Helper functions for resolving asset paths.

This module centralizes path construction for assets bundled with the
application.  Using these helpers avoids hard-coding relative paths
throughout the codebase.
"""

from pathlib import Path

ASSETS_ROOT = Path(__file__).parent


def templates_path(name: str) -> str:
    """Return the absolute path to a template asset."""
    return str(ASSETS_ROOT / "templates" / name)


def data_path(name: str) -> str:
    """Return the absolute path to a data asset."""
    return str(ASSETS_ROOT / "data" / name)


def fonts_path(name: str) -> str:
    """Return the absolute path to a font asset."""
    return str(ASSETS_ROOT / "fonts" / name)


def static_path(name: str) -> str:
    """Return the absolute path to a static asset."""
    return str(ASSETS_ROOT / "static" / name)


# --- Helpers for session artifacts ------------------------------------------


def sessions_path(session_id: str) -> str:
    """Return path to the compact session summary JSON."""
    return str(Path("sessions") / f"{session_id}.json")
