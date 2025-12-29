# NOTE: keep docstring to describe module responsibilities
"""Path helpers for AI adjudication artifacts."""

from __future__ import annotations

import json
import functools
import os
import re
import string

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from backend.core.paths import coerce_stage_path, normalize_worker_path


def _normalize_stage_override(
    value: object,
    *,
    run_base: Path,
    fallback: Path,
) -> Path:
    """Return a normalized path for manifest or environment overrides."""

    return coerce_stage_path(run_base, value, fallback=fallback)

from backend import config


@dataclass(frozen=True)
class ValidationAccountPaths:
    """Resolved filesystem locations for a single validation AI pack."""

    account_id: int
    pack_file: Path
    prompt_file: Path
    result_jsonl_file: Path
    result_summary_file: Path


@dataclass(frozen=True)
class ValidationPaths:
    """Resolved filesystem locations for validation AI packs."""

    base: Path
    packs_dir: Path
    results_dir: Path
    index_file: Path
    log_file: Path


@dataclass(frozen=True)
class NoteStyleAccountPaths:
    """Resolved filesystem locations for a single note_style AI pack."""

    account_id: str
    pack_file: Path
    result_file: Path
    result_raw_file: Path
    debug_file: Path


@dataclass(frozen=True)
class NoteStylePaths:
    """Resolved filesystem locations for note_style AI packs."""

    base: Path
    packs_dir: Path
    results_dir: Path
    results_raw_dir: Path
    debug_dir: Path
    index_file: Path
    log_file: Path


def inject_validation_paths_to_manifest(
    manifest_data: dict, sid: str, runs_root: Path | str, *, create: bool = True
) -> dict:
    """Inject validation native paths into manifest dict for T0 initialization.
    
    This function populates all validation path fields in the manifest:
    - ai.packs.validation.{base, dir, packs, packs_dir, results, results_dir, index, logs}
    - ai.validation.{base, dir}
    - artifacts.ai.packs.validation.* (mirror)
    - meta.validation_paths_initialized = True
    
    The function is idempotent: if paths already exist and match canonical values,
    no changes are made. If paths differ, existing values are preserved and a
    warning is logged.
    
    Parameters
    ----------
    manifest_data:
        The manifest dict to modify in-place
    sid:
        Session ID
    runs_root:
        Runs root directory path
    create:
        Whether to create directories on disk (default True)
    
    Returns
    -------
    dict:
        The modified manifest_data (same object, returned for convenience)
    """
    import logging
    log = logging.getLogger(__name__)
    
    runs_root_path = Path(runs_root) if not isinstance(runs_root, Path) else runs_root
    validation_paths = ensure_validation_paths(runs_root_path, sid, create=create)
    
    # Convert paths to strings
    base_str = str(validation_paths.base)
    packs_str = str(validation_paths.packs_dir)
    results_str = str(validation_paths.results_dir)
    index_str = str(validation_paths.index_file)
    logs_str = str(validation_paths.log_file)
    
    canonical_values = {
        "base": base_str,
        "dir": base_str,
        "packs": packs_str,
        "packs_dir": packs_str,
        "results": results_str,
        "results_dir": results_str,
        "index": index_str,
        "logs": logs_str,
    }
    
    # Ensure structure exists
    ai = manifest_data.setdefault("ai", {})
    ai_packs = ai.setdefault("packs", {})
    validation_packs = ai_packs.setdefault("validation", {})
    
    # Check if paths already exist and differ from canonical
    paths_differ = False
    for key, canonical_value in canonical_values.items():
        existing = validation_packs.get(key)
        if existing and isinstance(existing, str) and existing.strip():
            # Path exists - check if it matches canonical
            if existing != canonical_value:
                log.warning(
                    "VALIDATION_MANIFEST_PATHS_DIFFER sid=%s key=%s existing=%s canonical=%s",
                    sid, key, existing, canonical_value
                )
                paths_differ = True
            # Keep existing value (don't overwrite)
        else:
            # Path missing or empty - inject canonical value
            validation_packs[key] = canonical_value
    
    # Ensure last_built_at exists (but don't overwrite)
    if "last_built_at" not in validation_packs:
        validation_packs["last_built_at"] = None
    
    # Write to artifacts.ai.packs.validation (mirror)
    artifacts = manifest_data.setdefault("artifacts", {})
    artifacts_ai = artifacts.setdefault("ai", {})
    artifacts_ai_packs = artifacts_ai.setdefault("packs", {})
    validation_artifacts = artifacts_ai_packs.setdefault("validation", {})
    
    for key, value in canonical_values.items():
        if key not in validation_artifacts or not validation_artifacts.get(key):
            validation_artifacts[key] = value
    
    # Write to ai.validation (legacy mirror)
    ai_validation = ai.setdefault("validation", {})
    if "base" not in ai_validation or not ai_validation.get("base"):
        ai_validation["base"] = base_str
    if "dir" not in ai_validation or not ai_validation.get("dir"):
        ai_validation["dir"] = base_str
    
    # Set metadata flag
    meta = manifest_data.setdefault("meta", {})
    meta["validation_paths_initialized"] = True
    
    return manifest_data


def ensure_validation_paths(
    runs_root: Path, sid: str, create: bool = True
) -> ValidationPaths:
    """Return the canonical validation AI pack paths for ``sid``."""

    runs_root_path = Path(runs_root).resolve()
    run_base = (runs_root_path / sid).resolve()
    base_path = (run_base / "ai_packs" / "validation").resolve()

    def _resolve_override(env_name: str, default: Path) -> Path:
        raw = os.getenv(env_name)
        if not raw:
            return default

        return _normalize_stage_override(raw, run_base=run_base, fallback=default)

    packs_dir = _resolve_override("VALIDATION_PACKS_DIR", base_path / "packs")
    results_dir = _resolve_override("VALIDATION_RESULTS_DIR", base_path / "results")
    index_file = base_path / "index.json"
    log_file = base_path / "logs.txt"

    if create:
        base_path.mkdir(parents=True, exist_ok=True)
        packs_dir.mkdir(parents=True, exist_ok=True)
        results_dir.mkdir(parents=True, exist_ok=True)

    return ValidationPaths(
        base=base_path,
        packs_dir=packs_dir.resolve(),
        results_dir=results_dir.resolve(),
        index_file=index_file.resolve(strict=False),
        log_file=log_file.resolve(strict=False),
    )


def ensure_validation_account_paths(
    paths: ValidationPaths, account_idx: int | str, *, create: bool = True
) -> ValidationAccountPaths:
    """Return filesystem locations for ``account_idx`` under ``paths``."""

    try:
        normalized_idx = int(str(account_idx))
    except (TypeError, ValueError):
        raise ValueError("account_idx must be coercible to an integer") from None

    pack_filename = validation_pack_filename_for_account(normalized_idx)
    pack_file = paths.packs_dir / pack_filename
    prompt_file = paths.packs_dir / f"{pack_filename}.prompt.txt"
    result_jsonl_file = (
        paths.results_dir / validation_result_jsonl_filename_for_account(normalized_idx)
    )
    result_summary_file = (
        paths.results_dir / validation_result_filename_for_account(normalized_idx)
    )

    if create:
        pack_file.parent.mkdir(parents=True, exist_ok=True)
        prompt_file.parent.mkdir(parents=True, exist_ok=True)
        result_jsonl_file.parent.mkdir(parents=True, exist_ok=True)
        result_summary_file.parent.mkdir(parents=True, exist_ok=True)

    return ValidationAccountPaths(
        account_id=normalized_idx,
        pack_file=pack_file,
        prompt_file=prompt_file,
        result_jsonl_file=result_jsonl_file,
        result_summary_file=result_summary_file,
    )


@dataclass(frozen=True)
class MergePaths:
    """Resolved filesystem locations for merge AI packs."""

    base: Path
    packs_dir: Path
    results_dir: Path
    log_file: Path
    index_file: Path


_DEFAULT_MERGE_RESULTS_TEMPLATE = "pair_{lo:03d}_{hi:03d}.result.json"


def _merge_results_template() -> str:
    template = getattr(config, "MERGE_RESULTS_BASENAME", "") or ""
    if "{lo" not in template or "{hi" not in template:
        return _DEFAULT_MERGE_RESULTS_TEMPLATE
    return template


def _scoped_merge_path(run_base: Path, raw_value: str) -> Path:
    value_path = Path(raw_value)
    if value_path.is_absolute():
        return value_path.resolve()
    return (run_base / value_path).resolve()


def _merge_paths_from_run_base(run_base: Path, *, create: bool) -> MergePaths:
    stage_dir = _scoped_merge_path(run_base, config.MERGE_STAGE_DIR)
    packs_dir = _scoped_merge_path(run_base, config.MERGE_PACKS_DIR)
    results_dir = _scoped_merge_path(run_base, config.MERGE_RESULTS_DIR)
    index_file = _scoped_merge_path(run_base, config.MERGE_INDEX_PATH)

    if create:
        stage_dir.mkdir(parents=True, exist_ok=True)
        packs_dir.mkdir(parents=True, exist_ok=True)
        results_dir.mkdir(parents=True, exist_ok=True)
        index_file.parent.mkdir(parents=True, exist_ok=True)

    return MergePaths(
        base=stage_dir,
        packs_dir=packs_dir,
        results_dir=results_dir,
        log_file=stage_dir / "logs.txt",
        index_file=index_file,
    )


def ensure_merge_paths(runs_root: Path, sid: str, create: bool = True) -> MergePaths:
    """Return the canonical merge AI pack paths for ``sid``.

    When ``create`` is ``True`` (the default) the base directory along with the
    ``packs`` and ``results`` subdirectories are created if they do not already
    exist. When ``create`` is ``False`` the paths are computed without touching
    the filesystem.
    """

    runs_root_path = Path(runs_root).resolve()
    run_base = (runs_root_path / sid).resolve()
    return _merge_paths_from_run_base(run_base, create=create)


def merge_paths_from_any(path: Path | str, *, create: bool = False) -> MergePaths:
    """Return :class:`MergePaths` using ``path`` rooted at the merge base.

    ``path`` may point at the merge stage itself or any of the configured pack
    or result directories. The caller controls directory creation via
    ``create``; by default this function is read-only.
    """

    resolved = Path(path).resolve()
    suffixes = (
        Path(config.MERGE_PACKS_DIR),
        Path(config.MERGE_RESULTS_DIR),
        Path(config.MERGE_STAGE_DIR),
        Path(config.MERGE_INDEX_PATH),
    )

    candidates = (resolved,) + tuple(resolved.parents)
    for candidate in candidates:
        for suffix in suffixes:
            suffix_parts = tuple(part for part in suffix.parts if part not in {"", "."})
            if not suffix_parts:
                continue
            candidate_parts = candidate.parts
            if len(candidate_parts) < len(suffix_parts):
                continue
            if candidate_parts[-len(suffix_parts) :] != suffix_parts:
                continue
            run_base_parts = candidate_parts[: -len(suffix_parts)]
            run_base = Path(*run_base_parts)
            return _merge_paths_from_run_base(run_base, create=create)

    raise ValueError(f"Path does not identify merge layout: {resolved}")


def pair_pack_filename(a_idx: int, b_idx: int) -> str:
    """Return the canonical filename for a pair pack."""

    lo, hi = sorted((a_idx, b_idx))
    return f"pair_{lo:03d}_{hi:03d}.jsonl"


def pair_result_filename(a_idx: int, b_idx: int) -> str:
    """Return the canonical filename for a pair result."""

    lo, hi = sorted((a_idx, b_idx))
    template = _merge_results_template()
    try:
        return template.format(lo=lo, hi=hi)
    except Exception:
        return _DEFAULT_MERGE_RESULTS_TEMPLATE.format(lo=lo, hi=hi)


def pair_pack_path(paths: MergePaths, a_idx: int, b_idx: int) -> Path:
    """Return the resolved filesystem path for a pair pack."""

    return paths.packs_dir / pair_pack_filename(a_idx, b_idx)


def pair_result_path(paths: MergePaths, a_idx: int, b_idx: int) -> Path:
    """Return the resolved filesystem path for a pair result."""

    return paths.results_dir / pair_result_filename(a_idx, b_idx)


@functools.lru_cache(maxsize=None)
def _merge_result_pattern(template: str) -> re.Pattern[str]:
    formatter = string.Formatter()
    regex_parts: list[str] = []
    for literal_text, field_name, format_spec, _ in formatter.parse(template):
        if literal_text:
            regex_parts.append(re.escape(literal_text))
        if field_name is None:
            continue
        key = field_name.strip()
        if key not in {"lo", "hi"}:
            return _merge_result_pattern(_DEFAULT_MERGE_RESULTS_TEMPLATE)
        width: Optional[int] = None
        if format_spec:
            width_match = re.search(r"(\d+)", format_spec)
            if width_match:
                try:
                    width = int(width_match.group(1))
                except (TypeError, ValueError):
                    width = None
        if width:
            regex_parts.append(rf"(?P<{key}>\d{{{width}}})")
        else:
            regex_parts.append(rf"(?P<{key}>\d+)")

    pattern_text = "".join(regex_parts)
    if not pattern_text:
        return _merge_result_pattern(_DEFAULT_MERGE_RESULTS_TEMPLATE)
    return re.compile(f"^{pattern_text}$")


def merge_result_glob_pattern() -> str:
    template = _merge_results_template()
    formatter = string.Formatter()
    parts: list[str] = []
    for literal_text, field_name, _, _ in formatter.parse(template):
        if literal_text:
            parts.append(literal_text)
        if field_name is not None:
            parts.append("*")
    return "".join(parts) or "*.result.json"


def parse_pair_result_filename(name: str) -> Optional[tuple[int, int]]:
    template = _merge_results_template()
    pattern = _merge_result_pattern(template)
    match = pattern.match(name)
    if not match:
        return None
    try:
        lo = int(match.group("lo"))
        hi = int(match.group("hi"))
    except (TypeError, ValueError):
        return None
    return (lo, hi)


def get_merge_paths(runs_root: Path, sid: str, *, create: bool = True) -> MergePaths:
    """Return the resolved merge AI pack paths for ``sid``."""

    return ensure_merge_paths(runs_root, sid, create=create)


def ensure_note_style_paths(
    runs_root: Path | str, sid: str, *, create: bool = True
) -> NoteStylePaths:
    """Return the canonical note_style AI pack paths for ``sid``."""

    runs_root_path = Path(runs_root).resolve()
    run_base = (runs_root_path / sid).resolve()
    default_base = (run_base / config.NOTE_STYLE_STAGE_DIR).resolve()
    default_packs = (run_base / config.NOTE_STYLE_PACKS_DIR).resolve()
    default_results = (run_base / config.NOTE_STYLE_RESULTS_DIR).resolve()

    base_path: Path | None = None
    packs_dir: Path | None = None
    results_dir: Path | None = None
    index_file: Path | None = None
    log_file: Path | None = None

    if config.NOTE_STYLE_USE_MANIFEST_PATHS:
        manifest_path = (run_base / "manifest.json").resolve()
        try:
            raw_manifest = manifest_path.read_text(encoding="utf-8")
        except FileNotFoundError:
            raw_manifest = ""
        except OSError:
            raw_manifest = ""

        if raw_manifest:
            try:
                manifest_payload = json.loads(raw_manifest)
            except json.JSONDecodeError:
                manifest_payload = None
            if isinstance(manifest_payload, dict):
                ai_section = manifest_payload.get("ai")
                if isinstance(ai_section, dict):
                    packs_section = ai_section.get("packs")
                    if isinstance(packs_section, dict):
                        note_style_section = packs_section.get("note_style")
                        if isinstance(note_style_section, dict):
                            base_fallback = default_base

                            base_value = note_style_section.get("base") or note_style_section.get("dir")
                            if base_value is not None:
                                base_candidate = _normalize_stage_override(
                                    base_value,
                                    run_base=run_base,
                                    fallback=default_base,
                                )
                                base_path = base_candidate
                                base_fallback = base_candidate

                            packs_value = note_style_section.get("packs_dir") or note_style_section.get("packs")
                            if packs_value is not None:
                                packs_dir = _normalize_stage_override(
                                    packs_value,
                                    run_base=run_base,
                                    fallback=(base_fallback / "packs").resolve(),
                                )
                            elif base_path is not None:
                                packs_dir = (base_path / "packs").resolve()

                            results_value = note_style_section.get("results_dir") or note_style_section.get("results")
                            if results_value is not None:
                                results_dir = _normalize_stage_override(
                                    results_value,
                                    run_base=run_base,
                                    fallback=(base_fallback / "results").resolve(),
                                )
                            elif base_path is not None:
                                results_dir = (base_path / "results").resolve()

                            index_value = note_style_section.get("index")
                            if index_value is not None:
                                index_file = _normalize_stage_override(
                                    index_value,
                                    run_base=run_base,
                                    fallback=(base_fallback / "index.json").resolve(),
                                )

                            log_value = note_style_section.get("logs")
                            if log_value is not None:
                                log_file = _normalize_stage_override(
                                    log_value,
                                    run_base=run_base,
                                    fallback=(base_fallback / "logs.txt").resolve(),
                                )

    if base_path is None:
        base_path = default_base

    try:
        base_path = normalize_worker_path(run_base, os.fspath(base_path))
    except ValueError:
        base_path = default_base

    if packs_dir is None:
        packs_dir = (
            default_packs
            if base_path == default_base
            else (base_path / "packs").resolve()
        )
    if results_dir is None:
        results_dir = (
            default_results
            if base_path == default_base
            else (base_path / "results").resolve()
        )

    try:
        packs_dir = normalize_worker_path(run_base, os.fspath(packs_dir))
    except ValueError:
        packs_dir = (
            default_packs
            if base_path == default_base
            else (base_path / "packs").resolve()
        )

    try:
        results_dir = normalize_worker_path(run_base, os.fspath(results_dir))
    except ValueError:
        results_dir = (
            default_results
            if base_path == default_base
            else (base_path / "results").resolve()
        )

    results_raw_dir = base_path / "results_raw"
    debug_dir = base_path / "debug"

    if create:
        base_path.mkdir(parents=True, exist_ok=True)
        packs_dir.mkdir(parents=True, exist_ok=True)
        results_dir.mkdir(parents=True, exist_ok=True)
        results_raw_dir.mkdir(parents=True, exist_ok=True)
        debug_dir.mkdir(parents=True, exist_ok=True)

    if index_file is None:
        index_file = base_path / "index.json"
    else:
        try:
            index_file = normalize_worker_path(run_base, os.fspath(index_file))
        except ValueError:
            index_file = base_path / "index.json"
    if log_file is None:
        log_file = base_path / "logs.txt"
    else:
        try:
            log_file = normalize_worker_path(run_base, os.fspath(log_file))
        except ValueError:
            log_file = base_path / "logs.txt"

    return NoteStylePaths(
        base=base_path.resolve(),
        packs_dir=packs_dir.resolve(),
        results_dir=results_dir.resolve(),
        results_raw_dir=results_raw_dir.resolve(),
        debug_dir=debug_dir.resolve(),
        index_file=index_file.resolve(strict=False),
        log_file=log_file.resolve(strict=False),
    )


_NOTE_STYLE_ACCOUNT_PATTERN = re.compile(r"[^A-Za-z0-9_.-]")


def _normalize_note_style_account_id(account_id: object) -> str:
    text = str(account_id).strip() if account_id is not None else ""
    if not text:
        return "account"
    sanitized = _NOTE_STYLE_ACCOUNT_PATTERN.sub("_", text)
    return sanitized or "account"


def normalize_note_style_account_id(account_id: object) -> str:
    """Return the sanitized note_style account identifier for ``account_id``."""

    return _normalize_note_style_account_id(account_id)


def note_style_pack_filename(account_id: object) -> str:
    """Return the canonical note_style pack filename for ``account_id``."""

    normalized = normalize_note_style_account_id(account_id)
    return f"acc_{normalized}.jsonl"


def note_style_result_filename(account_id: object) -> str:
    """Return the canonical note_style result filename for ``account_id``."""

    normalized = normalize_note_style_account_id(account_id)
    template = config.NOTE_STYLE_RESULTS_BASENAME or "acc_{account}.result.jsonl"
    try:
        filename = template.format(account=normalized)
    except KeyError:
        filename = template
    return filename


def ensure_note_style_account_paths(
    paths: NoteStylePaths, account_id: object, *, create: bool = True
) -> NoteStyleAccountPaths:
    """Return filesystem locations for ``account_id`` under ``paths``."""

    normalized = normalize_note_style_account_id(account_id)
    pack_path = paths.packs_dir / note_style_pack_filename(normalized)
    result_path = paths.results_dir / note_style_result_filename(normalized)
    raw_result_path = paths.results_raw_dir / f"acc_{normalized}.raw.txt"
    debug_path = paths.debug_dir / f"{normalized}.context.json"

    if create:
        pack_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.parent.mkdir(parents=True, exist_ok=True)
        raw_result_path.parent.mkdir(parents=True, exist_ok=True)
        debug_path.parent.mkdir(parents=True, exist_ok=True)

    return NoteStyleAccountPaths(
        account_id=normalized,
        pack_file=pack_path,
        result_file=result_path,
        result_raw_file=raw_result_path,
        debug_file=debug_path,
    )


def probe_legacy_ai_packs(runs_root: Path, sid: str) -> Optional[Path]:
    """Return the legacy ``ai_packs`` directory if it contains any pair packs."""

    legacy_dir = Path(runs_root) / sid / "ai_packs"
    if not legacy_dir.is_dir():
        return None

    pattern = config.MERGE_PACK_GLOB or "pair_*.jsonl"
    if any(legacy_dir.glob(pattern)):
        return legacy_dir

    return None


def _resolve_runs_root(runs_root: Path | str | None) -> Path:
    """Return the effective runs root using ``RUNS_ROOT`` env fallback."""

    if runs_root is None:
        env_root = os.getenv("RUNS_ROOT")
        return Path(env_root) if env_root else Path("runs")
    return Path(runs_root)


def validation_base_dir(
    sid: str, runs_root: Path | str | None = None, *, create: bool = True
) -> Path:
    """Return the canonical validation base directory for ``sid``.

    When ``create`` is ``True`` the directory is created if it does not exist.
    """

    base = _resolve_runs_root(runs_root) / sid / "ai_packs" / "validation"
    if create:
        base.mkdir(parents=True, exist_ok=True)
    return base.resolve()


def validation_packs_dir(
    sid: str, runs_root: Path | str | None = None, *, create: bool = True
) -> Path:
    """Return the directory holding validation pack payloads for ``sid``."""

    packs_dir = validation_base_dir(sid, runs_root=runs_root, create=create) / "packs"
    if create:
        packs_dir.mkdir(parents=True, exist_ok=True)
    return packs_dir.resolve()


def validation_results_dir(
    sid: str, runs_root: Path | str | None = None, *, create: bool = True
) -> Path:
    """Return the directory holding validation model results for ``sid``."""

    results_dir = (
        validation_base_dir(sid, runs_root=runs_root, create=create) / "results"
    )
    if create:
        results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir.resolve()


def validation_index_path(
    sid: str, runs_root: Path | str | None = None, *, create: bool = True
) -> Path:
    """Return the manifest index path for validation packs."""

    runs_root_path = _resolve_runs_root(runs_root).resolve()
    validation_paths = ensure_validation_paths(runs_root_path, sid, create=create)
    return validation_paths.index_file


def validation_logs_path(
    sid: str, runs_root: Path | str | None = None, *, create: bool = True
) -> Path:
    """Return the log file path for validation pack activity."""

    base = validation_base_dir(sid, runs_root=runs_root, create=create)
    if create:
        base.mkdir(parents=True, exist_ok=True)
    return (base / "logs.txt").resolve()


def _normalize_account_id(account_id: int | str) -> int:
    try:
        return int(str(account_id).strip())
    except (TypeError, ValueError):  # pragma: no cover - defensive
        raise ValueError("account_id must be coercible to an integer") from None


_RESULTS_BASENAME_ENV = "VALIDATION_RESULTS_BASENAME"
_DEFAULT_RESULTS_BASENAME = "acc_{account:03d}.result"
_WRITE_JSON_ENV = "VALIDATION_WRITE_JSON"


def _env_flag(name: str) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return False
    text = str(raw).strip().lower()
    return text in {"1", "true", "yes", "y", "on"}


def validation_write_json_enabled() -> bool:
    """Return ``True`` when optional JSON result envelopes should be written."""

    return _env_flag(_WRITE_JSON_ENV)


def _validation_results_basename(account_id: int | str) -> str:
    normalized = _normalize_account_id(account_id)
    template = os.getenv(_RESULTS_BASENAME_ENV) or _DEFAULT_RESULTS_BASENAME
    try:
        formatted = template.format(account=normalized, account_id=normalized)
    except Exception:
        formatted = _DEFAULT_RESULTS_BASENAME.format(account=normalized)
    text = str(formatted).strip()
    if not text:
        return _DEFAULT_RESULTS_BASENAME.format(account=normalized)
    return text


def validation_pack_filename_for_account(account_id: int | str) -> str:
    """Return the canonical validation pack filename for ``account_id``."""

    normalized = _normalize_account_id(account_id)
    return f"val_acc_{normalized:03d}.jsonl"


def validation_result_jsonl_filename_for_account(account_id: int | str) -> str:
    """Return the canonical validation result JSONL filename for ``account_id``."""

    return f"{_validation_results_basename(account_id)}.jsonl"


def validation_result_json_filename_for_account(account_id: int | str) -> str:
    """Return the optional validation JSON envelope filename for ``account_id``."""

    return f"{_validation_results_basename(account_id)}.json"


def validation_result_summary_filename_for_account(account_id: int | str) -> str:
    """Backward-compatible alias for the canonical validation result filename."""

    if validation_write_json_enabled():
        return validation_result_json_filename_for_account(account_id)
    return validation_result_jsonl_filename_for_account(account_id)


def validation_result_filename_for_account(account_id: int | str) -> str:
    """Backward-compatible alias for the summary filename."""

    return validation_result_summary_filename_for_account(account_id)


def validation_result_error_filename_for_account(account_id: int | str) -> str:
    """Return the canonical validation error sidecar filename for ``account_id``."""

    return f"{_validation_results_basename(account_id)}.error.json"

