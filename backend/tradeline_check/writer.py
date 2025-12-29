"""Writers for tradeline_check per-bureau outputs."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Mapping

from backend.core.io.json_io import _atomic_write_json
from backend.tradeline_check.schema import bureau_output_template

log = logging.getLogger(__name__)


def write_bureau_findings(
    output_dir: Path,
    account_key: str,
    bureau: str,
    payload: Mapping,
) -> Path:
    """Atomically write per-bureau findings to JSON.

    Parameters
    ----------
    output_dir
        Parent directory for tradeline_check outputs (cases/accounts/<id>/tradeline_check)
    account_key
        Account identifier (directory name, e.g. "idx-007")
    bureau
        Bureau name ("equifax", "experian", "transunion")
    payload
        Bureau findings payload

    Returns
    -------
    Path
        Path to written file
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    filename = f"{bureau}.json"
    output_path = output_dir / filename

    payload_to_write = dict(payload) if isinstance(payload, Mapping) else payload

    try:
        _atomic_write_json(output_path, payload_to_write)
    except Exception as exc:
        log.error(
            "TRADELINE_CHECK_WRITE_FAILED path=%s bureau=%s error=%s",
            output_path,
            bureau,
            exc,
            exc_info=True,
        )
        raise

    return output_path
