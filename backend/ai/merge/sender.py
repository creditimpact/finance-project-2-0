from __future__ import annotations

import contextlib
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Mapping, Optional

from backend import config
from backend.core.ai.paths import get_merge_paths, ensure_merge_paths, pair_result_filename
from backend.runflow import decider

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MergeStagePaths:
    packs_dir: Path
    index_path: Path


@dataclass(frozen=True)
class MergeAutosendResult:
    sid: str
    packs_dir: Path
    index_path: Path
    glob: str
    discovered: int
    sent: int
    task_id: Optional[str]
    reason: str


def _resolve_runs_root(candidate: Optional[str | os.PathLike[str]]) -> Path:
    if candidate is None:
        env_root = os.getenv("RUNS_ROOT")
        return Path(env_root).resolve() if env_root else Path("runs").resolve()
    return Path(candidate).resolve()


def _resolve_stage_path(
    *,
    runs_root: Path,
    sid: str,
    env_value: Optional[str],
    default: Path,
) -> Path:
    if not env_value:
        return default
    candidate = Path(env_value).expanduser()
    if candidate.is_absolute():
        try:
            return candidate.resolve()
        except OSError:
            return candidate
    return (runs_root / sid / candidate).resolve()


def resolve_merge_stage_paths(
    sid: str,
    runs_root: Optional[str | os.PathLike[str]] = None,
) -> MergeStagePaths:
    sid_text = str(sid)
    runs_root_path = _resolve_runs_root(runs_root)
    merge_paths = get_merge_paths(runs_root_path, sid_text, create=True)
    packs_dir = _resolve_stage_path(
        runs_root=runs_root_path,
        sid=sid_text,
        env_value=os.getenv("MERGE_PACKS_DIR", config.MERGE_PACKS_DIR),
        default=merge_paths.packs_dir,
    )
    index_path = _resolve_stage_path(
        runs_root=runs_root_path,
        sid=sid_text,
        env_value=os.getenv("MERGE_INDEX_PATH", config.MERGE_INDEX_PATH),
        default=merge_paths.index_file,
    )
    return MergeStagePaths(packs_dir=packs_dir, index_path=index_path)


def _glob_matches(directory: Path, pattern: str) -> List[Path]:
    if not directory.exists():
        return []
    return sorted(directory.glob(pattern))


def discover_merge_packs(
    sid: str,
    *,
    runs_root: Optional[str | os.PathLike[str]] = None,
    glob_pattern: Optional[str] = None,
) -> tuple[MergeStagePaths, List[Path]]:
    paths = resolve_merge_stage_paths(sid, runs_root)
    pattern = glob_pattern or os.getenv("MERGE_PACK_GLOB", config.MERGE_PACK_GLOB)
    matches = _glob_matches(paths.packs_dir, pattern)
    return paths, matches


# ---- Idempotency helpers ----------------------------------------------------

def _coerce_int(value: object) -> Optional[int]:
    try:
        if value is None:
            return None
        return int(value)  # type: ignore[return-value]
    except (TypeError, ValueError):
        return None


def _read_runflow(run_dir: Path, sid: str) -> dict:
    """Read runflow using the same logic as decider.

    Returns a dict snapshot. When the file was unreadable (e.g., PermissionError),
    the snapshot contains marker key 'runflow_unavailable' = True.
    """
    path = run_dir / "runflow.json"
    return decider._load_runflow(path, str(sid))  # type: ignore[attr-defined]


def _merge_stage(snapshot: Mapping[str, object]) -> Optional[Mapping[str, object]]:
    stages = snapshot.get("stages")
    if isinstance(stages, Mapping):
        candidate = stages.get("merge")
        if isinstance(candidate, Mapping):
            return candidate
    return None


def is_merge_fully_applied(run_dir: Path, sid: str) -> bool:
    """Return True when merge is fully applied for the given SID.

    Read-only check against runflow.json:
    - merge status is 'success'
    - merge_ai_applied is True (when present)
    - and results are complete according to either:
        * summary.expected_packs (preferred), or
        * fallback: summary.pack_files, when expected is missing.

    Notes:
    - If runflow cannot be read and decider returns the default skeleton, this
      function returns False (state unknown is treated as not confirmed).
    """
    snapshot = _read_runflow(run_dir, sid)

    stage = _merge_stage(snapshot)
    if not isinstance(stage, Mapping):
        return False

    status = stage.get("status")
    if not isinstance(status, str) or status.strip().lower() != "success":
        return False

    # merge_ai_applied gate: if present it must be True
    applied_flag = stage.get("merge_ai_applied")
    if applied_flag is not None and not bool(applied_flag):
        return False

    summary = stage.get("summary") if isinstance(stage, Mapping) else None
    summary_map = summary if isinstance(summary, Mapping) else {}

    result_files = _coerce_int(summary_map.get("result_files"))
    if result_files is None:
        result_files = _coerce_int(stage.get("result_files"))
    if result_files is None:
        # Cannot confirm completeness without a result count
        return False

    expected = _coerce_int(summary_map.get("expected_packs"))
    if expected is None:
        # Fallback: use physical pack count from summary.stage if available
        expected = _coerce_int(summary_map.get("pack_files"))
        if expected is None:
            expected = _coerce_int(stage.get("pack_files"))

    if expected is not None:
        return result_files >= expected

    # As a last resort, if expected is unknown but results exist, assume complete
    # only when there are zero packs outstanding on disk. We cannot reliably read
    # that here without I/O; prefer conservative False.
    return False


@contextlib.contextmanager
def _temporary_environ(overrides: Mapping[str, Optional[str]]):
    sentinel = object()
    previous: dict[str, object] = {}
    try:
        for key, value in overrides.items():
            previous[key] = os.environ.get(key, sentinel)
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = str(value)
        yield
    finally:
        for key, value in previous.items():
            if value is sentinel:
                os.environ.pop(key, None)
            else:
                os.environ[key] = str(value)


def _merge_sender_env() -> dict[str, Optional[str]]:
    overrides: dict[str, Optional[str]] = {}
    model = os.getenv("MERGE_MODEL", config.MERGE_MODEL)
    if model:
        overrides["AI_MODEL"] = model
    timeout = os.getenv("MERGE_REQUEST_TIMEOUT")
    if timeout:
        overrides["AI_REQUEST_TIMEOUT"] = timeout
    else:
        overrides["AI_REQUEST_TIMEOUT"] = str(config.MERGE_REQUEST_TIMEOUT)
    max_retries = os.getenv("MERGE_MAX_RETRIES")
    if max_retries:
        overrides["AI_MAX_RETRIES"] = max_retries
    else:
        overrides["AI_MAX_RETRIES"] = str(config.MERGE_MAX_RETRIES)
    backoff = os.getenv("MERGE_BACKOFF_SCHEDULE", config.MERGE_BACKOFF_SCHEDULE)
    if backoff:
        overrides["AI_BACKOFF_SCHEDULE"] = backoff
    return overrides


def _script_arguments(
    sid: str,
    *,
    runs_root: Path,
    paths: MergeStagePaths,
) -> List[str]:
    args: List[str] = ["--sid", sid]
    if runs_root:
        args.extend(["--runs-root", str(runs_root)])
    args.extend(["--packs-dir", str(paths.packs_dir)])
    max_retries = os.getenv("MERGE_MAX_RETRIES")
    if max_retries:
        args.extend(["--max-retries", max_retries])
    backoff = os.getenv("MERGE_BACKOFF_SCHEDULE")
    if backoff:
        args.extend(["--backoff", backoff])
    return args


def send_merge_packs(
    sid: str,
    *,
    runs_root: Optional[str | os.PathLike[str]] = None,
    glob_pattern: Optional[str] = None,
    task_id: Optional[str] = None,
    reason: str = "manual",
) -> MergeAutosendResult:
    sid_text = str(sid)
    runs_root_path = _resolve_runs_root(runs_root)
    paths, matches = discover_merge_packs(
        sid_text, runs_root=runs_root_path, glob_pattern=glob_pattern
    )
    glob_value = glob_pattern or os.getenv("MERGE_PACK_GLOB", config.MERGE_PACK_GLOB)
    discovered = len(matches)
    logger.info(
        "MERGE_AUTOSEND_DISCOVERY sid=%s reason=%s packs_dir=%s glob=%s count=%d index=%s task_id=%s",
        sid_text,
        reason,
        paths.packs_dir,
        glob_value,
        discovered,
        paths.index_path,
        task_id,
    )
    if discovered == 0:
        return MergeAutosendResult(
            sid=sid_text,
            packs_dir=paths.packs_dir,
            index_path=paths.index_path,
            glob=glob_value,
            discovered=0,
            sent=0,
            task_id=task_id,
            reason=reason,
        )

    # Build pending set by skipping packs that already have result files
    merge_paths = ensure_merge_paths(runs_root_path, sid_text, create=True)
    pending: list[Path] = []
    for path in matches:
        # parse pair indices from filename: pair_XXX_YYY.jsonl
        stem = path.stem
        parts = stem.split("_")
        if len(parts) == 3 and parts[0] == "pair":
            try:
                lo = int(parts[1])
                hi = int(parts[2])
            except (TypeError, ValueError):
                lo = hi = -1
        else:
            lo = hi = -1

        result_exists = False
        if lo >= 0 and hi >= 0:
            result_name = pair_result_filename(lo, hi)
            result_path = (merge_paths.results_dir / result_name)
            try:
                result_exists = result_path.is_file()
            except OSError:
                result_exists = False

        if not result_exists:
            pending.append(path)

    pending_count = len(pending)
    logger.info(
        "MERGE_AUTOSEND_DISCOVERY sid=%s reason=%s packs_dir=%s glob=%s count=%d index=%s task_id=%s pending=%d",
        sid_text,
        reason,
        paths.packs_dir,
        glob_value,
        discovered,
        paths.index_path,
        task_id,
        pending_count,
    )

    if pending_count == 0:
        logger.info(
            "MERGE_AUTOSEND_NO_PENDING_PACKS sid=%s discovered=%d pending=0 task_id=%s",
            sid_text,
            discovered,
            task_id,
        )
        return MergeAutosendResult(
            sid=sid_text,
            packs_dir=paths.packs_dir,
            index_path=paths.index_path,
            glob=glob_value,
            discovered=discovered,
            sent=0,
            task_id=task_id,
            reason=reason,
        )

    overrides = _merge_sender_env()
    from scripts import send_ai_merge_packs

    # Execute script once per pending pack by narrowing MERGE_PACK_GLOB
    sent = 0
    for pack_path in pending:
        args = _script_arguments(sid_text, runs_root=runs_root_path, paths=paths)
        narrowed_env = dict(overrides)
        narrowed_env["MERGE_PACK_GLOB"] = pack_path.name
        logger.info(
            "MERGE_AUTOSEND_EXECUTE sid=%s packs=1 task_id=%s args=%s pack=%s",
            sid_text,
            task_id,
            args,
            pack_path.name,
        )
        try:
            with _temporary_environ(narrowed_env):
                send_ai_merge_packs.main(args)
            sent += 1
        except SystemExit as exc:  # pragma: no cover - script conformity
            logger.warning(
                "MERGE_AUTOSEND_EXECUTE_FAILED sid=%s pack=%s code=%s",
                sid_text,
                pack_path.name,
                getattr(exc, "code", None),
                exc_info=False,
            )
        except Exception:  # pragma: no cover - defensive
            logger.warning(
                "MERGE_AUTOSEND_EXECUTE_FAILED sid=%s pack=%s",
                sid_text,
                pack_path.name,
                exc_info=True,
            )

    logger.info(
        "MERGE_AUTOSEND_COMPLETED sid=%s discovered=%d sent=%d task_id=%s",
        sid_text,
        discovered,
        sent,
        task_id,
    )

    return MergeAutosendResult(
        sid=sid_text,
        packs_dir=paths.packs_dir,
        index_path=paths.index_path,
        glob=glob_value,
        discovered=discovered,
        sent=sent,
        task_id=task_id,
        reason=reason,
    )


def _should_autosend() -> bool:
    return bool(config.MERGE_AUTOSEND)


def _should_autosend_on_build() -> bool:
    return _should_autosend() and bool(config.MERGE_SEND_ON_BUILD)


def trigger_autosend_after_build(
    sid: str,
    *,
    runs_root: Optional[str | os.PathLike[str]] = None,
    created: int = 0,
) -> None:
    if not _should_autosend_on_build():
        logger.debug(
            "MERGE_AUTOSEND_BUILD_SKIP sid=%s created=%d reason=env_disabled",
            sid,
            created,
        )
        return
    runs_root_path = _resolve_runs_root(runs_root)
    try:
        from backend.api.tasks import app as celery_app
    except Exception:  # pragma: no cover - defensive logging
        logger.warning(
            "MERGE_AUTOSEND_CELERY_IMPORT_FAILED sid=%s", sid, exc_info=True
        )
        return

    try:
        result = celery_app.send_task(
            "backend.ai.merge.tasks.send_merge_packs",
            args=[str(sid)],
            kwargs={
                "runs_root": str(runs_root_path),
                "reason": "build",
            },
        )
    except Exception:  # pragma: no cover - defensive logging
        logger.warning("MERGE_AUTOSEND_ENQUEUE_FAILED sid=%s", sid, exc_info=True)
        return

    logger.info(
        "MERGE_AUTOSEND_ENQUEUED sid=%s created=%d task_id=%s reason=build",
        sid,
        created,
        getattr(result, "id", None),
    )


def schedule_stage_autosend(
    sid: str,
    *,
    run_dir: Path,
) -> None:
    if not (_should_autosend() and bool(config.MERGE_STAGE_AUTORUN)):
        logger.debug(
            "MERGE_AUTOSEND_STAGE_SKIP sid=%s reason=env_disabled", sid
        )
        return

    # Check runflow state first to avoid redundant or unsafe scheduling
    try:
        snapshot = _read_runflow(run_dir, sid)
    except Exception:  # pragma: no cover - defensive
        snapshot = {"stages": {}, "run_state": "INIT"}

    if bool(snapshot.get("runflow_unavailable")):
        logger.info(
            "MERGE_AUTOSEND_SKIPPED sid=%s reason=runflow_unavailable", sid
        )
        return

    if is_merge_fully_applied(run_dir, sid):
        logger.info(
            "MERGE_AUTOSEND_SKIPPED sid=%s reason=merge_already_applied", sid
        )
        return

    runs_root_path = run_dir.parent.resolve()
    paths, matches = discover_merge_packs(
        str(sid), runs_root=runs_root_path, glob_pattern=None
    )
    if not matches:
        logger.info(
            "MERGE_AUTOSEND_STAGE_SKIP sid=%s reason=no_packs dir=%s",
            sid,
            paths.packs_dir,
        )
        return

    try:
        from backend.api.tasks import app as celery_app
    except Exception:  # pragma: no cover - defensive logging
        logger.warning(
            "MERGE_AUTOSEND_STAGE_CELERY_IMPORT_FAILED sid=%s", sid, exc_info=True
        )
        return

    try:
        result = celery_app.send_task(
            "backend.ai.merge.tasks.send_merge_packs",
            args=[str(sid)],
            kwargs={
                "runs_root": str(runs_root_path),
                "reason": "autorun",
            },
        )
    except Exception:  # pragma: no cover - defensive logging
        logger.warning(
            "MERGE_AUTOSEND_STAGE_ENQUEUE_FAILED sid=%s", sid, exc_info=True
        )
        return

    logger.info(
        "MERGE_AUTOSEND_STAGE_ENQUEUED sid=%s packs=%d task_id=%s",
        sid,
        len(matches),
        getattr(result, "id", None),
    )


__all__ = [
    "MergeAutosendResult",
    "discover_merge_packs",
    "resolve_merge_stage_paths",
    "schedule_stage_autosend",
    "send_merge_packs",
    "trigger_autosend_after_build",
]
