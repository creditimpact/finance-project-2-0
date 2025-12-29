from __future__ import annotations

import logging
from pathlib import Path
from backend.pipeline.runs import RunManifest

from backend.core.config import CLEANUP_AFTER_EXPORT
from backend.core.logic.report_analysis import block_exporter
from backend.core.logic.report_analysis.trace_cleanup import purge_after_export

log = logging.getLogger(__name__)


def run_stage_a(session_id: str, project_root: Path = Path(".")) -> dict:
    """Run Stage-A export and optionally clean up trace artifacts.

    Parameters
    ----------
    session_id:
        Identifier for the trace directory under ``traces/blocks/<sid>``.
    project_root:
        Base directory of the repository.  Defaults to the current working
        directory.

    Returns
    -------
    dict
        Metadata including artifact paths and cleanup details.
    """

    # Prefer canonical runs/<SID> paths via manifest
    m = RunManifest.for_sid(session_id)
    traces_dir = m.ensure_run_subdir("traces_dir", "traces")
    accounts_dir = (traces_dir / "accounts_table").resolve()
    accounts_dir.mkdir(parents=True, exist_ok=True)
    stage_a = block_exporter.export_stage_a(session_id, accounts_out_dir=accounts_dir)
    if not stage_a.get("ok"):
        return {"sid": session_id, "ok": False, "where": "stage_a"}

    meta: dict = {"sid": session_id, "ok": True, "artifacts": stage_a["artifacts"]}

    artifacts = stage_a.get("artifacts", {})
    full_tsv = Path(artifacts.get("full_tsv", ""))
    accounts_json = Path(artifacts.get("accounts_json", ""))
    general_json = Path(artifacts.get("general_info_json", ""))

    blocks_dir = m.ensure_run_subdir("traces_blocks_dir", "traces/blocks")
    assert blocks_dir.exists(), f"blocks dir not found: {blocks_dir}"

    if not (full_tsv.exists() and accounts_json.exists() and general_json.exists()):
        meta["cleanup"] = {"performed": False, "reason": "artifacts_missing"}
        return meta

    if CLEANUP_AFTER_EXPORT:
        log.warning(
            "CLEANUP_AFTER_EXPORT enabled; duplicate cleanup may occur but is idempotent",
            extra={"sid": session_id},
        )
        cleanup_summary = purge_after_export(sid=session_id, project_root=project_root)
        log.info("purge_after_export", extra={"sid": session_id, **cleanup_summary})
        meta["cleanup"] = {"performed": True, "summary": cleanup_summary}
    else:
        log.debug(
            "run_stage_a: cleanup delegated to chain", extra={"sid": session_id}
        )
        meta["cleanup"] = {"performed": False, "reason": "delegated_to_chain"}

    return meta
