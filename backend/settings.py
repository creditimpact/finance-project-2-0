from pathlib import Path

# Absolute path to the repository root.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Location where trace artifacts are stored.
TRACES_DIR = PROJECT_ROOT / "traces"

__all__ = ["PROJECT_ROOT", "TRACES_DIR"]
