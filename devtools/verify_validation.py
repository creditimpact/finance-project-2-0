"""Developer helper to exercise the validation pipeline end-to-end."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping
from unittest.mock import patch

from backend.ai.manifest import ensure_validation_section
from backend.core.ai.paths import ensure_validation_paths
from backend.ai.validation_builder import ValidationPackWriter
from backend.pipeline.auto_ai_tasks import validation_send
from backend.pipeline.runs import RunManifest
from backend.validation.send_packs import ValidationPackSender


class _ValidationStubClient:
    """Simple OpenAI replacement for validation pack verification."""

    def __init__(self, response_payload: Mapping[str, Any]) -> None:
        self._payload = dict(response_payload)
        self.calls: list[Mapping[str, Any]] = []

    def create(
        self,
        *,
        model: str,
        messages: list[Mapping[str, Any]],
        response_format: Mapping[str, Any],
        **kwargs: Any,
    ) -> Mapping[str, Any]:
        self.calls.append(
            {
                "model": model,
                "messages": messages,
                "response_format": response_format,
                "extra": dict(kwargs),
            }
        )
        return {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(self._payload, ensure_ascii=False),
                    }
                }
            ]
        }


def _seed_manifest(run_dir: Path, account_id: int) -> None:
    manifest = RunManifest.load_or_create(run_dir / "manifest.json", run_dir.name)
    artifacts = manifest.data.setdefault("artifacts", {})
    cases = artifacts.setdefault("cases", {})
    accounts = cases.setdefault("accounts", {})
    accounts[str(account_id)] = {
        "dir": f"cases/accounts/{account_id}",
        "summary": "summary.json",
    }
    manifest.save()


def _write_summary(run_dir: Path, account_id: int) -> None:
    account_dir = run_dir / "cases" / "accounts" / str(account_id)
    account_dir.mkdir(parents=True, exist_ok=True)
    summary_payload = {
        "validation_requirements": {
            "findings": [
                {
                    "field": "account_type",
                    "category": "identity",
                    "min_days": 30,
                    "documents": ["statement"],
                    "ai_needed": True,
                    "is_mismatch": True,
                    "send_to_ai": True,
                }
            ],
            "send_to_ai": {"account_type": True},
            "field_consistency": {
                "account_type": {
                    "consensus": "split",
                    "disagreeing_bureaus": ["experian"],
                    "missing_bureaus": ["equifax"],
                    "raw": {"transunion": "credit_card", "experian": "revolving", "equifax": None},
                    "normalized": {
                        "transunion": "credit_card",
                        "experian": "revolving",
                        "equifax": None,
                    },
                }
            },
        }
    }
    (account_dir / "summary.json").write_text(
        json.dumps(summary_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _write_local_config(base_dir: Path) -> None:
    base_dir.mkdir(parents=True, exist_ok=True)
    config_path = base_dir / "ai_packs_config.yml"
    config_payload = {
        "validation_packs": {
            "enable_write": True,
            "enable_infer": False,
            "model": "stub-validation",
        }
    }
    config_path.write_text(json.dumps(config_payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _list_pack_files(paths) -> list[str]:
    return sorted(path.name for path in paths.packs_dir.glob("*.jsonl"))


def _count_results(paths) -> tuple[int, list[str]]:
    total_lines = 0
    files: list[str] = []
    for file in sorted(paths.results_dir.glob("*.jsonl")):
        lines = [line for line in file.read_text(encoding="utf-8").splitlines() if line.strip()]
        if lines:
            files.append(file.name)
            total_lines += len(lines)
    return total_lines, files


def verify_validation(sid: str) -> None:
    runs_root = Path(tempfile.mkdtemp(prefix="validation-verify-"))
    os.environ["RUNS_ROOT"] = str(runs_root)

    run_dir = runs_root / sid
    account_id = 101

    _seed_manifest(run_dir, account_id)
    _write_summary(run_dir, account_id)

    ensure_validation_section(sid, runs_root=runs_root)
    validation_paths = ensure_validation_paths(runs_root, sid, create=True)
    _write_local_config(validation_paths.base)

    writer = ValidationPackWriter(sid, runs_root=runs_root)
    pack_lines = writer.write_pack_for_account(account_id)
    if not pack_lines:
        raise SystemExit("validation pack build produced no lines; verification failed")

    pack_files = _list_pack_files(validation_paths)

    stub_response = {
        "sid": sid,
        "account_id": account_id,
        "id": "account_type",
        "field": "account_type",
        "decision": "no_case",
        "rationale": "Deterministic outcome (FIELD_MISMATCH).",
        "citations": ["transunion: value"],
        "checks": {
            "materiality": False,
            "supports_consumer": False,
            "doc_requirements_met": False,
            "mismatch_code": "FIELD_MISMATCH",
        },
        "reason_code": "FIELD_MISMATCH",
        "reason_label": "Field 1 mismatch",
        "modifiers": {
            "material_mismatch": True,
            "time_anchor": False,
            "doc_dependency": False,
        },
        "confidence": 0.75,
    }
    client = _ValidationStubClient(stub_response)

    payload = {"sid": sid, "runs_root": str(runs_root)}
    with patch.object(ValidationPackSender, "_build_client", lambda self: client):
        validation_send.run(payload)

    result_lines, result_files = _count_results(validation_paths)

    manifest = RunManifest(run_dir / "manifest.json").load()
    stage_status = manifest.get_ai_stage_status("validation")

    print(f"Runs root: {runs_root}")
    print(f"Found packs: {len(pack_files)} {pack_files}")
    print(f"Requests sent: {len(client.calls)}")
    print(f"Results written: {result_lines} {result_files}")
    print(
        "Manifest sent: {sent} completed_at: {completed}".format(
            sent=stage_status.get("sent"),
            completed=stage_status.get("completed_at"),
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify the validation AI pipeline")
    parser.add_argument("--sid", default="VERIFY-VALIDATION", help="Run identifier to use")
    args = parser.parse_args()
    verify_validation(args.sid)


if __name__ == "__main__":  # pragma: no cover - manual tool
    main()
