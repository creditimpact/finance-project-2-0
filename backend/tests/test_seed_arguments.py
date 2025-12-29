import pytest

from backend.validation import seed_arguments

from ..core.logic.validation_requirements import (
    _build_finding,
    _collect_seed_arguments,
)


@pytest.fixture
def normalized_map_c4():
    return {
        "experian": "open",
        "equifax": "open",
        "transunion": "closed",
    }


@pytest.fixture
def details_from_map():
    def _factory(normalized_map):
        return {"normalized": normalized_map}

    return _factory


def test_strong_finding_populates_seed_and_aggregates(normalized_map_c4, details_from_map):
    entry = {"field": "account_status"}
    finding = _build_finding(
        entry,
        {},
        details=details_from_map(normalized_map_c4),
        normalized_map=normalized_map_c4,
    )

    assert finding["decision"] == "strong_actionable"

    argument = finding.get("argument")
    assert argument is not None
    seed = argument.get("seed")
    assert seed == {
        "id": "account_status__C4_TWO_MATCH_ONE_DIFF",
        "tone": "firm_courteous",
        "text": (
            "Two bureaus agree the account is <X>, while one shows <Y>. "
            "Please verify the source records and align the discrepant bureau."
        ),
    }

    seeds = _collect_seed_arguments([finding])
    assert seeds == [seed]


def test_supportive_finding_does_not_receive_seed(details_from_map):
    normalized_map = {
        "experian": "open",
        "equifax": None,
        "transunion": None,
    }

    finding = _build_finding(
        {"field": "account_status"},
        {},
        details=details_from_map(normalized_map),
        normalized_map=normalized_map,
    )

    assert finding["decision"] == "supportive_needs_companion"
    assert "argument" not in finding

    seeds = _collect_seed_arguments([finding])
    assert seeds == []


def test_missing_template_does_not_raise_and_skips_seed(details_from_map):
    normalized_map = {
        "experian": "A",
        "equifax": "B",
        "transunion": "C",
    }

    finding = _build_finding(
        {"field": "account_rating"},
        {},
        details=details_from_map(normalized_map),
        normalized_map=normalized_map,
    )

    decision = finding.get("decision")
    if decision is not None:
        assert decision == "strong_actionable"
    assert "argument" not in finding

    seeds = _collect_seed_arguments([finding])
    assert seeds == []


def test_missing_template_file_logs_warning(monkeypatch, caplog):
    caplog.set_level("WARNING")
    seed_arguments.load_seed_templates.cache_clear()
    monkeypatch.setattr(
        seed_arguments, "TEMPLATE_PATH", seed_arguments.HERE / "does-not-exist.json"
    )

    templates = seed_arguments.load_seed_templates()

    assert templates == {}
    assert any(
        "Seed argument templates file missing" in record.message
        for record in caplog.records
    )

    seed_arguments.load_seed_templates.cache_clear()


def test_invalid_template_json_raises(monkeypatch, tmp_path):
    broken = tmp_path / "seed_argument_templates.json"
    broken.write_text("{\n  'oops':\n", encoding="utf-8")

    seed_arguments.load_seed_templates.cache_clear()
    monkeypatch.setattr(seed_arguments, "TEMPLATE_PATH", broken)

    with pytest.raises(ValueError, match="Failed to parse seed argument templates JSON"):
        seed_arguments.load_seed_templates()

    seed_arguments.load_seed_templates.cache_clear()


def test_duplicate_seed_entries_are_deduplicated(normalized_map_c4, details_from_map):
    entry = {"field": "account_status"}
    finding_one = _build_finding(
        entry,
        {},
        details=details_from_map(normalized_map_c4),
        normalized_map=normalized_map_c4,
    )
    finding_two = _build_finding(
        entry,
        {},
        details=details_from_map(normalized_map_c4),
        normalized_map=normalized_map_c4,
    )

    seeds = _collect_seed_arguments([finding_one, finding_two])
    assert seeds == [finding_one["argument"]["seed"]]


def test_seed_generation_respects_feature_flag(
    normalized_map_c4, details_from_map, monkeypatch
):
    monkeypatch.setattr(
        "backend.core.logic.validation_requirements.backend_config.SEED_ARGUMENTS_ENABLE",
        False,
    )

    finding = _build_finding(
        {"field": "account_status"},
        {},
        details=details_from_map(normalized_map_c4),
        normalized_map=normalized_map_c4,
    )

    assert finding["decision"] == "strong_actionable"
    assert "argument" not in finding

    seeds = _collect_seed_arguments([finding])
    assert seeds == []
