import os, sys
from pathlib import Path

_here = Path(__file__).resolve()
_repo_root = _here.parent.parent  # <repo>
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

# Optional: load .env
try:
    from dotenv import load_dotenv  # type: ignore
except Exception:
    load_dotenv = None
if load_dotenv:
    env_path = _repo_root / ".env"
    if env_path.exists():
        load_dotenv(dotenv_path=str(env_path))
