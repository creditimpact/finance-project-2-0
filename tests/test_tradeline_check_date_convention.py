import json
from pathlib import Path

import pytest

from backend.tradeline_check.runner import run_for_account
from backend.validation.pipeline import AccountContext

SID = "sid_date_convention"


def _write_manifest(run_dir: Path, trace_payload: dict | None) -> None:
    traces_dir = run_dir / "traces"
    traces_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "sid": SID,
        "base_dirs": {
            "traces_dir": str(traces_dir),
            "cases_accounts_dir": str(run_dir / "cases" / "accounts"),
        },
        "artifacts": {
            "traces": {
                "date_convention": str(traces_dir / "date_convention.json"),
                "date_convention_rel": "traces/date_convention.json",
            }
        },
    }

    if trace_payload is not None:
        (traces_dir / "date_convention.json").write_text(json.dumps(trace_payload), encoding="utf-8")

    (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def _make_acc_ctx(tmp_path: Path, account_key: str, bureaus_payload: dict, trace_payload: dict | None) -> AccountContext:
    runs_root = tmp_path / "runs"
    run_dir = runs_root / SID
    _write_manifest(run_dir, trace_payload)

    acc_dir = run_dir / "cases" / "accounts" / str(account_key)
    acc_dir.mkdir(parents=True, exist_ok=True)

    bureaus_path = acc_dir / "bureaus.json"
    bureaus_path.write_text(json.dumps(bureaus_payload), encoding="utf-8")

    summary_path = acc_dir / "summary.json"
    summary_path.write_text(json.dumps({"validation_requirements": {"schema_version": 3, "findings": []}}))

    return AccountContext(
        sid=SID,
        runs_root=runs_root,
        index=str(account_key),
        account_key=str(account_key),
        account_id=f"idx-{int(account_key):03d}",
        account_dir=acc_dir,
        summary_path=summary_path,
        bureaus_path=bureaus_path,
    )


def _read_bureau_file(tmp_path: Path, account_key: str, bureau: str) -> dict:
    path = tmp_path / "runs" / SID / "cases" / "accounts" / str(account_key) / "tradeline_check" / f"{bureau}.json"
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(autouse=True)
def clear_env(monkeypatch):
    monkeypatch.setenv("TRADELINE_CHECK_ENABLED", "1")
    monkeypatch.setenv("TRADELINE_CHECK_WRITE_DEBUG", "1")
    monkeypatch.setenv("TRADELINE_CHECK_PLACEHOLDER_TOKENS", "--,n/a,unknown")
    monkeypatch.setenv("TRADELINE_CHECK_WRITE_EMPTY_RESULTS", "0")
    yield


def test_date_convention_attached_from_manifest(tmp_path: Path):
    trace_payload = {
        "created_at": "2024-02-02T00:00:00Z",
        "date_convention": {
            "version": "date_convention_v1",
            "scope": "global",
            "convention": "DMY",
            "month_language": "unknown",
            "confidence": 0.9,
            "evidence_counts": {"samples": 3},
            "detector_version": "v0",
        },
    }
    acc_ctx = _make_acc_ctx(
        tmp_path,
        account_key="1",
        bureaus_payload={"equifax": {"date_opened": "01/02/2024", "date_reported": "02/02/2024"}},
        trace_payload=trace_payload,
    )

    run_for_account(acc_ctx)

    data = _read_bureau_file(tmp_path, "1", "equifax")
    dc = data["date_convention"]
    expected_path = str((tmp_path / "runs" / SID / "traces" / "date_convention.json").resolve())

    assert dc["convention"] == "DMY"
    assert dc["scope"] == "global"
    assert dc["confidence"] == 0.9
    assert dc["source"]["file_abs"] == expected_path
    assert dc["source"]["file_rel"] == "traces/date_convention.json"


def test_date_convention_missing_trace_falls_back(tmp_path: Path):
    acc_ctx = _make_acc_ctx(
        tmp_path,
        account_key="2",
        bureaus_payload={"equifax": {"date_opened": "2024-01-01", "date_reported": "2024-02-01"}},
        trace_payload=None,
    )

    run_for_account(acc_ctx)

    data = _read_bureau_file(tmp_path, "2", "equifax")
    dc = data["date_convention"]

    assert dc["convention"] == "unknown"
    assert dc["confidence"] == 0.0
    assert dc["source"]["file_rel"] == "traces/date_convention.json"
    assert dc["source"]["file_abs"] is None or dc["source"]["file_abs"].endswith("traces\\date_convention.json")
