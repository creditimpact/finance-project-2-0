from __future__ import annotations

from typing import Any, Dict

from backend.core.case_store.api import get_account_case
from backend.core.config.flags import FLAGS
from backend.core.compat.legacy_shim import build_by_bureau_shim


def build_account_view(session_id: str, account_id: str) -> dict:
    case = get_account_case(session_id, account_id)
    fields_dict: Dict[str, Any] = case.fields.model_dump()
    by_bureau = fields_dict.get("by_bureau") or {}
    if FLAGS.one_case_per_account_enabled and not by_bureau:
        by_bureau = build_by_bureau_shim(session_id, account_id)

    view_fields: Dict[str, Any] = {"by_bureau": by_bureau}
    if "normalized" in fields_dict:
        view_fields["normalized"] = fields_dict["normalized"]
    # Legacy cases without by_bureau: include raw fields
    if not by_bureau and not FLAGS.one_case_per_account_enabled:
        legacy_fields = {
            k: v
            for k, v in fields_dict.items()
            if k not in ("by_bureau", "normalized")
        }
        view_fields.update(legacy_fields)

    artifacts_dict = {
        k: (v.model_dump() if v is not None else None)
        for k, v in (case.artifacts or {}).items()
    }
    stagea = {
        k: v for k, v in artifacts_dict.items() if k.startswith("stageA_detection.")
    }
    out_artifacts: Dict[str, Any] = {}
    if stagea:
        out_artifacts = stagea
    elif "stageA_detection" in artifacts_dict:
        out_artifacts["stageA_detection"] = artifacts_dict["stageA_detection"]

    view = {
        "account_id": account_id,
        "fields": view_fields,
        "artifacts": out_artifacts,
        "meta": {
            "flags": {
                "one_case_per_account_enabled": FLAGS.one_case_per_account_enabled,
                "normalized_overlay_enabled": FLAGS.normalized_overlay_enabled,
            },
            "present_bureaus": sorted(list(by_bureau.keys())),
        },
    }
    return view
