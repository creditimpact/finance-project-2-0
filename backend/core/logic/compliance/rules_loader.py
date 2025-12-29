import hashlib
from importlib.resources import files
from typing import Any, Mapping

import yaml

RULES_PKG = "backend.core.rules"
_RULES_VERSION: str | None = None


def _load_yaml(filename: str):
    try:
        path = files(RULES_PKG) / filename
        with path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Missing required rules file: {RULES_PKG}/{filename}"
        ) from exc
    except ModuleNotFoundError as exc:
        raise FileNotFoundError(f"Missing rules package: {RULES_PKG}") from exc
    except yaml.YAMLError as exc:
        raise RuntimeError(f"Invalid YAML in {filename}: {exc}") from exc


def load_rules() -> list:
    """Load and return dispute rules from ``dispute_rules.yaml``.

    Returns a list of rule dictionaries.
    """
    data = _load_yaml("dispute_rules.yaml")
    if not isinstance(data, list):
        raise ValueError("dispute_rules.yaml must define a list of rules")
    return data


def load_neutral_phrases() -> Mapping[str, Any]:
    """Load and return neutral phrases mapping from ``neutral_phrases.yaml``."""
    data = _load_yaml("neutral_phrases.yaml")
    if not isinstance(data, dict):
        raise ValueError("neutral_phrases.yaml must define a mapping")
    return data


_NEUTRAL_CACHE: dict | None = None


def get_neutral_phrase(
    category: str, structured_summary: dict | None = None
) -> tuple[str | None, dict]:
    """Return a neutral phrase and selection metadata for ``category``.

    The phrase list is loaded from ``neutral_phrases.yaml`` only once. A very
    lightweight heuristic picks the phrase sharing the most word overlap with
    the client's structured summary. If no summary is provided, the first phrase
    for the category is returned. The second element of the tuple describes why
    the phrase was chosen.
    """

    global _NEUTRAL_CACHE
    if _NEUTRAL_CACHE is None:
        _NEUTRAL_CACHE = load_neutral_phrases()

    phrases = _NEUTRAL_CACHE.get(category, [])
    if not phrases:
        return None, {}

    if structured_summary:
        text = " ".join(
            str(v).lower() for v in structured_summary.values() if isinstance(v, str)
        )
        words = set(text.split())
        if words:
            chosen = max(phrases, key=lambda p: len(words & set(p.lower().split())))
            matched = sorted(words & set(chosen.lower().split()))
            return chosen, {"method": "word_overlap", "matched_words": matched}

    return phrases[0], {"method": "default", "matched_words": []}


def load_state_rules() -> Mapping[str, Any]:
    """Load and return state compliance rules from ``state_rules.yaml``."""
    data = _load_yaml("state_rules.yaml")
    if not isinstance(data, dict):
        raise ValueError("state_rules.yaml must define a mapping")
    return data


def recompute_rules_version() -> str:
    """Recompute and return the current rules version hash."""

    global _RULES_VERSION
    sha = hashlib.sha256()
    pkg = files(RULES_PKG)
    for path in sorted(p for p in pkg.iterdir() if p.name.endswith((".yml", ".yaml"))):
        sha.update(path.read_bytes())
    _RULES_VERSION = sha.hexdigest()
    return _RULES_VERSION
