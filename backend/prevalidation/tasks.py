from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Mapping

from celery import shared_task

from backend.config import (
    DATE_CONVENTION_PATH,
    DATE_CONVENTION_SCOPE,
    PREVALIDATION_DETECT_DATES,
)
from backend.core.manifest_utils import register_date_convention_in_manifest
from backend.core.runflow import runflow_step
from backend.prevalidation.date_convention_detector import detect_month_language_for_run

logger = logging.getLogger(__name__)


def _ensure_payload(prev: Mapping[str, object] | None) -> dict[str, object]:
    if isinstance(prev, Mapping):
        return dict(prev)
    return {}


def _resolve_runs_root(payload: Mapping[str, object], sid: str) -> Path:
    runs_root_value = payload.get("runs_root")
    if isinstance(runs_root_value, (str, os.PathLike)):
        return Path(runs_root_value)

    env_root = os.getenv("RUNS_ROOT")
    if env_root:
        return Path(env_root)

    return Path("runs")


def _fsync_directory(directory: Path) -> None:
    try:
        dir_fd = os.open(str(directory), os.O_RDONLY)
    except (AttributeError, OSError):
        return
    try:
        os.fsync(dir_fd)
    except OSError:
        pass
    finally:
        try:
            os.close(dir_fd)
        except OSError:
            pass


def _atomic_write_json_if_changed(path: Path, document: dict[str, object]) -> bool:
    payload = json.dumps(document, ensure_ascii=False, indent=2) + "\n"

    try:
        existing = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        existing = None
    except OSError:
        existing = None

    if existing == payload:
        return False

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    try:
        tmp_path.unlink()
    except FileNotFoundError:
        pass

    with tmp_path.open("w", encoding="utf-8") as handle:
        handle.write(payload)
        handle.flush()
        try:
            os.fsync(handle.fileno())
        except OSError:
            pass

    os.replace(tmp_path, path)
    _fsync_directory(path.parent)
    return True


def detect_and_persist_date_convention(
    sid: str,
    *,
    runs_root: Path | str | None = None,
) -> dict[str, object] | None:
    """Run the month language detector for ``sid`` and persist the result."""

    if not PREVALIDATION_DETECT_DATES:
        logger.info("DATE_CONVENTION_SKIP sid=%s reason=disabled", sid)
        return None

    if not sid:
        return None

    base_root = Path(runs_root) if runs_root is not None else Path(os.getenv("RUNS_ROOT", "runs"))
    run_root = base_root / sid
    if not run_root.exists():
        logger.info("DATE_CONVENTION_SKIP sid=%s reason=run_missing path=%s", sid, run_root)
        return None

    detection = detect_month_language_for_run(str(run_root))
    block = detection.get("date_convention") if isinstance(detection, dict) else None
    if not isinstance(block, dict):
        logger.info("DATE_CONVENTION_SKIP sid=%s reason=no_detection", sid)
        return None

    block = dict(block)
    block.setdefault("scope", DATE_CONVENTION_SCOPE)

    evidence = block.get("evidence_counts") if isinstance(block.get("evidence_counts"), dict) else {}
    out_path_config = DATE_CONVENTION_PATH
    out_path_obj = Path(out_path_config)
    if not out_path_obj.is_absolute():
        target_path = run_root / out_path_obj
        log_out_path = out_path_config
    else:
        target_path = out_path_obj
        log_out_path = str(out_path_obj)

    legacy_general_info_relative = Path("traces") / "accounts_table" / "general_info_from_full.json"
    try:
        relative_target = target_path.relative_to(run_root)
    except ValueError:
        relative_target = None

    is_legacy_general_info = False
    if relative_target is not None:
        is_legacy_general_info = relative_target == legacy_general_info_relative
    else:
        suffix = legacy_general_info_relative.as_posix()
        is_legacy_general_info = str(target_path).endswith(suffix)

    if is_legacy_general_info:
        logger.error(
            "DATE_CONVENTION_SKIP sid=%s reason=legacy_general_info_path path=%s",
            sid,
            target_path,
        )
        return None

    document = {"date_convention": block}

    try:
        _atomic_write_json_if_changed(target_path, document)
    except Exception:
        logger.error(
            "DATE_CONVENTION_WRITE_FAILED sid=%s path=%s", sid, target_path, exc_info=True
        )
        return None

    metrics: dict[str, object] = {}
    confidence_value = block.get("confidence")
    try:
        if confidence_value is not None:
            metrics["confidence"] = float(confidence_value)
    except (TypeError, ValueError):
        pass

    accounts_scanned = evidence.get("accounts_scanned")
    try:
        metrics["accounts_scanned"] = int(accounts_scanned)
    except (TypeError, ValueError):
        pass

    try:
        relative_out = target_path.relative_to(run_root)
        out_path = relative_out.as_posix()
    except ValueError:
        out_path = str(target_path)

    runflow_step(
        sid,
        "validation",
        "detect_date_convention",
        metrics=metrics or None,
        out={"path": out_path},
    )

    try:
        run_root_abs = run_root.resolve()
    except Exception:
        run_root_abs = run_root

    env_out_rel = os.getenv("PREVALIDATION_OUT_PATH_REL")
    if env_out_rel:
        out_path_rel_value = env_out_rel
    elif relative_target is not None:
        out_path_rel_value = relative_target.as_posix()
    elif not out_path_obj.is_absolute():
        out_path_rel_value = out_path_config
    else:
        out_path_rel_value = target_path.name

    try:
        register_date_convention_in_manifest(
            sid=sid,
            run_root=str(run_root_abs),
            out_path_rel=out_path_rel_value,
            detector_meta={
                "convention": block.get("convention"),
                "month_language": block.get("month_language"),
                "confidence": block.get("confidence"),
                "detector_version": block.get("detector_version"),
            },
        )
    except Exception:  # pragma: no cover - defensive logging
        manifest_path = run_root_abs / "manifest.json"
        logger.error(
            "DATE_CONVENTION_MANIFEST_UPDATE_FAILED sid=%s path=%s", sid, manifest_path, exc_info=True
        )

    summary_path = run_root / "logs.txt"
    convention_value = block.get("convention") or "unknown"
    language_value = block.get("month_language") or "unknown"
    confidence_value = block.get("confidence")
    confidence_repr = confidence_value if confidence_value is not None else "unknown"
    summary_line = (
        "DATE_DETECT_SUMMARY: "
        f"path={log_out_path} "
        f"conv={convention_value} "
        f"lang={language_value} "
        f"conf={confidence_repr}\n"
    )

    try:
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with summary_path.open("a", encoding="utf-8") as handle:
            handle.write(summary_line)
    except Exception:
        logger.error(
            "DATE_CONVENTION_LOG_WRITE_FAILED sid=%s path=%s", sid, summary_path, exc_info=True
        )

    logger.info(
        "DATE_DETECT: scope=%s conv=%s lang=%s conf=%s he=%s en=%s scanned=%s out=%s",
        block.get("scope"),
        block.get("convention") or "unknown",
        block.get("month_language"),
        block.get("confidence"),
        evidence.get("he_hits"),
        evidence.get("en_hits"),
        evidence.get("accounts_scanned"),
        log_out_path,
    )

    return block


@shared_task(bind=True, autoretry_for=(), retry_backoff=False)
def run_date_convention_detector(
    self, prev: Mapping[str, object] | None
) -> dict[str, object]:
    """Celery task wrapper that runs the date convention detector."""

    payload = _ensure_payload(prev)
    sid = str(payload.get("sid") or "")
    if not sid:
        logger.info("DATE_CONVENTION_PAYLOAD_SKIP payload=%s", payload)
        return payload

    runs_root = _resolve_runs_root(payload, sid)
    try:
        block = detect_and_persist_date_convention(sid, runs_root=runs_root)
    except Exception:
        logger.error("DATE_CONVENTION_TASK_FAILED sid=%s", sid, exc_info=True)
        return payload

    if isinstance(block, dict):
        payload["date_convention"] = dict(block)
    return payload


__all__ = [
    "detect_and_persist_date_convention",
    "run_date_convention_detector",
]
