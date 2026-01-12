import json
from copy import deepcopy
from pathlib import Path

import pytest

from backend.config import merge_config
from backend.core.ai.paths import get_merge_paths
from backend.core.logic.report_analysis import account_merge
from backend.core.logic.report_analysis import ai_pack


FIXTURE_DIR = Path(__file__).parent / "fixtures" / "account_merge"


def _load_fixture(name: str) -> dict:
    with (FIXTURE_DIR / name).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _strip_original_creditor(payload: dict) -> dict:
    data = deepcopy(payload)
    for branch in data.values():
        if isinstance(branch, dict):
            branch.pop("original_creditor", None)
    return data


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_trimmed_pair_scores_above_threshold(monkeypatch):
    monkeypatch.setenv("MERGE_REQUIRE_ORIGINAL_CREDITOR_FOR_AI", "0")
    merge_config.reset_merge_config_cache()
    try:
        cfg = account_merge.get_merge_cfg()

        left = _load_fixture("trimmed_9.json")
        right = _load_fixture("trimmed_10.json")

        result = account_merge.score_pair_0_100(left, right, cfg)

        assert result.get("total", 0.0) >= 6.0
        assert result.get("score_points", 0.0) >= 6.0
    finally:
        merge_config.reset_merge_config_cache()


def test_ai_pack_skipped_when_original_creditor_required(monkeypatch, tmp_path):
    monkeypatch.setenv("MERGE_REQUIRE_ORIGINAL_CREDITOR_FOR_AI", "1")
    merge_config.reset_merge_config_cache()
    try:
        left = _strip_original_creditor(_load_fixture("trimmed_9.json"))
        right = _strip_original_creditor(_load_fixture("trimmed_10.json"))

        sid = "trimmed-sid"
        for index, payload in enumerate((left, right), start=1):
            account_dir = tmp_path / sid / "cases" / "accounts" / str(index)
            account_dir.mkdir(parents=True, exist_ok=True)
            (account_dir / "bureaus.json").write_text(
                json.dumps(payload),
                encoding="utf-8",
            )
            (account_dir / "raw_lines.json").write_text("[]\n", encoding="utf-8")

        result = ai_pack.build_ai_pack_for_pair(sid, tmp_path, 1, 2, highlights=None)

        assert result.get("skipped") is True
        assert result.get("reason") == "missing_original_creditor"

        pack_path = (
            tmp_path
            / sid
            / "ai_packs"
            / "merge"
            / "packs"
            / "pair_001_002.json"
        )
        assert not pack_path.exists()
    finally:
        merge_config.reset_merge_config_cache()


def test_points_breakdown_contains_redacted_values(monkeypatch):
    monkeypatch.setenv("MERGE_POINTS_DIAGNOSTICS", "1")
    monkeypatch.setenv("MERGE_POINTS_DIAGNOSTICS_LIMIT", "5")
    monkeypatch.setenv("MERGE_REQUIRE_ORIGINAL_CREDITOR_FOR_AI", "0")
    merge_config.reset_merge_config_cache()
    try:
        cfg = account_merge.get_merge_cfg()

        left = _load_fixture("trimmed_9.json")
        right = _load_fixture("trimmed_10.json")

        result = account_merge.score_pair_0_100(left, right, cfg)

        breakdown = result.get("points_breakdown")
        assert isinstance(breakdown, dict)

        serialized = json.dumps(breakdown)
        assert serialized  # ensure JSON serializable

        fields_map = breakdown.get("fields", {})
        assert isinstance(fields_map, dict)
        weights = breakdown.get("weights", {})
        assert weights
        assert pytest.approx(weights.get("account_number", 0.0)) == 1.0

        account_number_entry = fields_map.get("account_number")
        assert account_number_entry is not None
        assert account_number_entry.get("matched") is True
        assert account_number_entry.get("pairs")
        for pair_entry in account_number_entry.get("pairs_verbose", []):
            assert "9000001111" not in str(pair_entry.get("raw_values", {}).get("a", ""))
            assert "9000001111" not in str(pair_entry.get("raw_values", {}).get("b", ""))

        account_label_entry = fields_map.get("account_label")
        assert account_label_entry is not None
        assert isinstance(account_label_entry.get("reason"), str)
        assert account_label_entry.get("reason")
    finally:
        merge_config.reset_merge_config_cache()


def test_points_diagnostics_persisted_when_gate_blocks(monkeypatch, tmp_path):
    monkeypatch.setenv("MERGE_POINTS_DIAGNOSTICS", "1")
    monkeypatch.setenv("MERGE_POINTS_DIAGNOSTICS_LIMIT", "5")
    monkeypatch.setenv("MERGE_REQUIRE_ORIGINAL_CREDITOR_FOR_AI", "1")
    monkeypatch.setenv("MERGE_POINTS_PERSIST_BREAKDOWN", "1")
    monkeypatch.setenv("MERGE_POINTS_MODE", "1")
    merge_config.reset_merge_config_cache()

    sid = "diag-sid"
    left = _strip_original_creditor(_load_fixture("trimmed_9.json"))
    right = _strip_original_creditor(_load_fixture("trimmed_10.json"))

    for index, payload in enumerate((left, right), start=1):
        account_dir = tmp_path / sid / "cases" / "accounts" / str(index)
        account_dir.mkdir(parents=True, exist_ok=True)
        (account_dir / "bureaus.json").write_text(json.dumps(payload), encoding="utf-8")
        (account_dir / "raw_lines.json").write_text("[]\n", encoding="utf-8")

    monkeypatch.setattr(account_merge, "start_span", lambda *args, **kwargs: None)
    monkeypatch.setattr(account_merge, "span_step", lambda *args, **kwargs: None)
    monkeypatch.setattr(account_merge, "end_span", lambda *args, **kwargs: None)
    monkeypatch.setattr(account_merge, "runflow_event", lambda *args, **kwargs: None)
    monkeypatch.setattr(account_merge, "steps_pair_topn", lambda: 0)
    monkeypatch.setattr(account_merge, "build_ai_pack_for_pair", lambda *args, **kwargs: {})

    import backend.ai.merge.sender as merge_sender

    monkeypatch.setattr(merge_sender, "trigger_autosend_after_build", lambda *args, **kwargs: None)

    try:
        scores = account_merge.score_all_pairs_0_100(sid, [1, 2], runs_root=tmp_path)
        pair_result = scores[1][2]
        breakdown = pair_result.get("points_breakdown")
        assert isinstance(breakdown, dict)
        oc_gate = breakdown.get("oc_gate")
        assert isinstance(oc_gate, dict)
        assert oc_gate.get("required") is True
        assert oc_gate.get("present_left") is False
        assert oc_gate.get("present_right") is False
        assert oc_gate.get("score_allowed") is True
        assert oc_gate.get("final_allowed") is False
        assert oc_gate.get("action") == "skip"
        assert oc_gate.get("reason") == "missing_original_creditor"

        path_value = pair_result.get("points_breakdown_path")
        assert path_value is not None
        diagnostics_path = Path(path_value)
        assert diagnostics_path.exists()
        persisted = json.loads(diagnostics_path.read_text(encoding="utf-8"))
        assert persisted.get("oc_gate", {}).get("reason") == "missing_original_creditor"
        assert persisted.get("pair", {}).get("left") == 1
        assert persisted.get("pair", {}).get("right") == 2
        assert persisted.get("pair", {}).get("a") == 1
        assert persisted.get("pair", {}).get("b") == 2
        assert persisted.get("pair", {}).get("lo") == 1
        assert persisted.get("pair", {}).get("hi") == 2
        assert persisted.get("pair", {}).get("sid") == sid

        fields_map = persisted.get("fields", {})
        assert isinstance(fields_map, dict)
        acct_entry = fields_map.get("account_number")
        assert acct_entry
        for pair_entry in acct_entry.get("pairs_verbose", []):
            raw_values = pair_entry.get("raw_values", {}) if isinstance(pair_entry, dict) else {}
            assert "9000001111" not in str(raw_values.get("a", ""))
            assert "9000001111" not in str(raw_values.get("b", ""))

        merge_paths = get_merge_paths(tmp_path, sid, create=False)
        assert merge_paths.log_file.exists()
        log_text = merge_paths.log_file.read_text(encoding="utf-8")
        assert "FIELDS 1-2" in log_text
        assert "FIELDS 1-2 ACCOUNT_NUMBER matched" in log_text
        assert "OC_GATE 1-2 required=" in log_text
        assert "missing_original_creditor" in log_text
        assert "9000001111" not in log_text
        run_root = tmp_path / sid
        assert diagnostics_path.is_relative_to(run_root)
        assert "ai_packs" in diagnostics_path.parts
    finally:
        merge_config.reset_merge_config_cache()


def test_points_diagnostics_path_absent_when_persistence_disabled(monkeypatch, tmp_path):
    monkeypatch.setenv("MERGE_POINTS_DIAGNOSTICS", "1")
    monkeypatch.setenv("MERGE_POINTS_DIAGNOSTICS_LIMIT", "5")
    monkeypatch.setenv("MERGE_REQUIRE_ORIGINAL_CREDITOR_FOR_AI", "1")
    monkeypatch.setenv("MERGE_POINTS_PERSIST_BREAKDOWN", "0")
    monkeypatch.setenv("MERGE_POINTS_MODE", "1")
    merge_config.reset_merge_config_cache()

    sid = "no-persist-sid"
    left = _strip_original_creditor(_load_fixture("trimmed_9.json"))
    right = _strip_original_creditor(_load_fixture("trimmed_10.json"))

    for index, payload in enumerate((left, right), start=1):
        account_dir = tmp_path / sid / "cases" / "accounts" / str(index)
        account_dir.mkdir(parents=True, exist_ok=True)
        (account_dir / "bureaus.json").write_text(json.dumps(payload), encoding="utf-8")
        (account_dir / "raw_lines.json").write_text("[]\n", encoding="utf-8")

    monkeypatch.setattr(account_merge, "start_span", lambda *args, **kwargs: None)
    monkeypatch.setattr(account_merge, "span_step", lambda *args, **kwargs: None)
    monkeypatch.setattr(account_merge, "end_span", lambda *args, **kwargs: None)
    monkeypatch.setattr(account_merge, "runflow_event", lambda *args, **kwargs: None)
    monkeypatch.setattr(account_merge, "steps_pair_topn", lambda: 0)
    monkeypatch.setattr(account_merge, "build_ai_pack_for_pair", lambda *args, **kwargs: {})

    import backend.ai.merge.sender as merge_sender

    monkeypatch.setattr(merge_sender, "trigger_autosend_after_build", lambda *args, **kwargs: None)

    try:
        scores = account_merge.score_all_pairs_0_100(sid, [1, 2], runs_root=tmp_path)
        pair_result = scores[1][2]
        assert pair_result.get("points_breakdown")
        assert pair_result.get("points_breakdown_path") is None
        diagnostics_root = tmp_path / sid / "ai_packs" / "merge" / "diagnostics"
        assert not diagnostics_root.exists()
    finally:
        merge_config.reset_merge_config_cache()


def test_history_2y_prefers_inline_over_monthly_and_legacy():
    payload = {
        "history_2y": {"transunion": ["OK", "LATE"]},
        "two_year_payment_history_monthly_tsv_v2": {
            "transunion": [
                {"month": "2024-01", "status": "IGNORED"},
            ]
        },
        "two_year_payment_history": {"transunion": ["LEGACY"]},
    }

    context = account_merge._build_inline_points_mode_context(payload)

    assert context["history_2y"]["transunion"] == ["OK", "LATE"]


def test_history_2y_uses_monthly_tsv_v2_when_inline_missing():
    payload = {
        "two_year_payment_history_monthly_tsv_v2": {
            "equifax": [
                {"month": "2024-01", "status": "OK"},
                {"month": "2024-02", "status": None},
                {"month": "2024-03", "status": ""},
            ]
        }
    }

    context = account_merge._build_inline_points_mode_context(payload)

    assert context["history_2y"]["equifax"] == ["OK", "--", "--"]


def test_history_2y_falls_back_to_legacy_when_monthly_absent():
    payload = {
        "two_year_payment_history": {
            "experian": ["30", "60"],
        }
    }

    context = account_merge._build_inline_points_mode_context(payload)

    assert context["history_2y"]["experian"] == ["30", "60"]


def test_history_2y_monthly_context_is_comparable():
    payload = {
        "two_year_payment_history_monthly_tsv_v2": {
            "transunion": [
                {"month": "2024-01", "status": "OK"},
                {"month": "2024-02", "status": "--"},
            ],
            "experian": [
                {"month": "2024-01", "status": "OK"},
                {"month": "2024-02", "status": "--"},
            ],
        }
    }

    left_ctx = account_merge._build_inline_points_mode_context(payload)
    right_ctx = account_merge._build_inline_points_mode_context(payload)

    matched, aux = account_merge._points_mode_compare_history_2y(
        left_ctx,
        right_ctx,
        threshold=0.9,
    )

    assert matched is True
    assert aux.get("match_score") == 1.0
    assert aux.get("compared") == 2
