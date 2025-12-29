"""Developer helper to exercise the note_style pipeline end-to-end."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping
from unittest.mock import patch

from backend.ai.manifest import ensure_note_style_section
from backend.ai.note_style.tasks import note_style_send_sid_task
from backend.ai.note_style_stage import build_note_style_pack_for_account
from backend.core.ai.paths import (
    ensure_note_style_account_paths,
    ensure_note_style_paths,
)
from backend.pipeline.runs import RunManifest


class _NoteStyleStubClient:
    """Minimal OpenAI replacement that records requests and returns fixture data."""

    def __init__(self) -> None:
        self.calls: list[Mapping[str, Any]] = []

    def chat_completion(
        self,
        *,
        model: str,
        messages: list[Mapping[str, Any]],
        temperature: float,
        **_: Any,
    ) -> Mapping[str, Any]:
        self.calls.append({"model": model, "messages": messages, "temperature": temperature})
        payload = {
            "tone": "Empathetic",
            "context_hints": {
                "timeframe": {"month": "April", "relative": "Last month"},
                "topic": "Payment_Dispute",
                "entities": {"creditor": "Capital One", "amount": "$123.45"},
            },
            "emphasis": ["paid_already", "custom", "support_request"],
            "confidence": 0.91,
            "risk_flags": [
                "Follow_Up",
                "duplicate",
                "FOLLOW_UP",
                "Mixed Language",
                "ALL CAPS",
                "possible-template copy",
            ],
        }
        return {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(payload, ensure_ascii=False),
                    }
                }
            ]
        }


def _seed_manifest(run_dir: Path, account_id: str) -> None:
    manifest = RunManifest.load_or_create(run_dir / "manifest.json", run_dir.name)
    data = manifest.data.setdefault("artifacts", {})
    cases = data.setdefault("cases", {})
    accounts = cases.setdefault("accounts", {})
    accounts[account_id] = {
        "dir": f"cases/accounts/{account_id}",
        "meta": "meta.json",
        "bureaus": "bureaus.json",
        "tags": "tags.json",
    }
    manifest.save()


def _write_account_context(run_dir: Path, account_id: str) -> None:
    account_dir = run_dir / "cases" / "accounts" / account_id
    account_dir.mkdir(parents=True, exist_ok=True)

    (account_dir / "meta.json").write_text(
        json.dumps({"consumer_name": "Avery Example"}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (account_dir / "bureaus.json").write_text(
        json.dumps(
            {
                "transunion": {"account_number": "12345", "balance": 123.45},
                "experian": {"account_number": "12345", "balance": 123.45},
                "equifax": {"account_number": "12345", "balance": 123.45},
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (account_dir / "tags.json").write_text(
        json.dumps(
            {
                "primary_issue_tag": "Billing_Dispute",
                "tags": ["billing", "dispute"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def _write_response(run_dir: Path, account_id: str, sid: str) -> None:
    response_dir = run_dir / "frontend" / "review" / "responses"
    response_dir.mkdir(parents=True, exist_ok=True)
    response_payload = {
        "sid": sid,
        "account_id": account_id,
        "answers": {"explanation": "Please help, this was already paid last month."},
    }
    (response_dir / f"{account_id}.result.json").write_text(
        json.dumps(response_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _discover_pack_files(paths) -> list[str]:
    return sorted(path.name for path in paths.packs_dir.glob("*.jsonl"))


def _count_result_files(paths) -> tuple[int, list[str]]:
    result_files: list[str] = []
    total_lines = 0
    for file in sorted(paths.results_dir.glob("*.jsonl")):
        lines = [line for line in file.read_text(encoding="utf-8").splitlines() if line.strip()]
        if lines:
            result_files.append(file.name)
            total_lines += len(lines)
    return total_lines, result_files


def verify_note_style(sid: str) -> None:
    runs_root = Path(tempfile.mkdtemp(prefix="note-style-verify-"))
    os.environ["RUNS_ROOT"] = str(runs_root)

    run_dir = runs_root / sid
    account_id = "acct-001"

    _seed_manifest(run_dir, account_id)
    _write_account_context(run_dir, account_id)
    _write_response(run_dir, account_id, sid)

    ensure_note_style_section(sid, runs_root=runs_root)
    build_note_style_pack_for_account(sid, account_id, runs_root=runs_root)

    paths = ensure_note_style_paths(runs_root, sid, create=False)
    pack_files = _discover_pack_files(paths)

    client = _NoteStyleStubClient()
    with patch("backend.ai.note_style_sender.get_ai_client", lambda: client):
        note_style_send_sid_task.run(sid, runs_root=str(runs_root))

    account_paths = ensure_note_style_account_paths(paths, account_id, create=False)
    if not account_paths.result_file.exists():
        raise SystemExit("note_style result file missing; verification failed")

    result_lines, result_files = _count_result_files(paths)

    manifest = RunManifest(run_dir / "manifest.json").load()
    stage_status = manifest.get_ai_stage_status("note_style")

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
    parser = argparse.ArgumentParser(description="Verify the note_style AI pipeline")
    parser.add_argument("--sid", default="VERIFY-NOTE-STYLE", help="Run identifier to use")
    args = parser.parse_args()
    verify_note_style(args.sid)


if __name__ == "__main__":  # pragma: no cover - manual tool
    main()
