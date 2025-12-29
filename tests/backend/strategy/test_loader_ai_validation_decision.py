import json
from pathlib import Path

from backend.strategy.io import load_findings_from_summary
from backend.strategy.order_rules import rank_findings


def test_loader_ai_validation_decision(tmp_path: Path):
    # Build minimal summary.json with only ai_validation_decision present
    summary_payload = {
        "validation_requirements": {
            "findings": [
                {
                    "field": "creditor_type",
                    "min_days": 6,
                    "duration_unit": "business_days",
                    # AI-only decision path (was previously ignored by loader)
                    "ai_validation_decision": "supportive_needs_companion",
                }
            ]
        }
    }
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(json.dumps(summary_payload), encoding="utf-8")

    findings = load_findings_from_summary(summary_path)
    assert len(findings) == 1, "Expected one finding to be loaded"
    finding = findings[0]

    # The loader should now map ai_validation_decision into both ai_decision and default_decision
    assert getattr(finding, "default_decision") == "supportive_needs_companion"
    assert getattr(finding, "ai_decision") == "supportive_needs_companion"

    # Rank findings should classify this as supportive (not unsupported_decision)
    openers, middle, closers, skipped = rank_findings([finding], include_supporters=True)
    assert not skipped, f"Finding incorrectly skipped: {skipped}"
    assert any(f.field == "creditor_type" for f in middle), "Supportive finding not placed in middle bucket"
