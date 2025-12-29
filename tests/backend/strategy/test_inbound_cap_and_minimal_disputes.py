from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

from backend.strategy.planner import compute_optimal_plan
from backend.strategy.types import Finding
import os
from contextlib import contextmanager


def _run_plan(findings, *, handoff_min=1, handoff_max=3):
    run_dt = datetime(2025, 1, 6, 9, 0, tzinfo=ZoneInfo("America/New_York"))  # Monday
    return compute_optimal_plan(
        findings,
        weekend={5, 6},
        holidays=None,
        timezone_name="America/New_York",
        run_datetime=run_dt,
        max_calendar_span=90,
        last_submit_window=(0, 40),
        include_supporters=True,
        exclude_natural_text=True,
        strength_metric="score",
        handoff_min_business_days=handoff_min,
        handoff_max_business_days=handoff_max,
        enforce_span_cap=False,
    )


def _best_plan(bundle):
    wd = bundle["best_weekday"]
    return bundle["weekday_plans"][wd]


def test_inbound_cap_adjusts_and_preserves_invariants():
    # Design: Large closer SLA to force unbounded inbound > 50 before optimization
    findings = [
        Finding(field="opener_a", category="status", min_days=12, duration_unit="business_days", default_decision="strong_actionable"),
        Finding(field="support_m1", category="terms", min_days=8, duration_unit="business_days", default_decision="supportive_needs_companion"),
        Finding(field="closer_big", category="history", min_days=35, duration_unit="business_days", default_decision="strong_actionable"),
    ]
    # Use handoff_min=2 to ensure there is headroom to pull earlier
    bundle = _run_plan(findings, handoff_min=2, handoff_max=4)
    plan = _best_plan(bundle)

    summary = plan.get("summary", {})
    unbounded = summary.get("total_effective_days_unbounded")
    assert unbounded is not None, "unbounded inbound should be present"
    # Expect hard cap ≤50 when a legal earlier shift exists; otherwise optimizer may mark unachievable
    assert int(unbounded) <= 50, f"unbounded inbound should be reduced to ≤50, got {unbounded}"
    assert summary.get("last_submit_in_window", False) is True

    seq = plan.get("sequence_debug", [])
    assert len(seq) >= 2

    # No weekend submits, overlaps >= 1, monotonic indices
    prior_idx = -1
    for i, e in enumerate(seq):
        submit = e.get("submit", {})
        weekday = e.get("submit_on", {}).get("weekday")
        assert weekday not in {5, 6}, "no weekend submission allowed"
        idx_val = int(e.get("calendar_day_index", 0))
        assert idx_val >= prior_idx
        prior_idx = idx_val
        if i > 0:
            assert int(e.get("handoff_days_before_prev_sla_end", 0)) >= 1

    # inventory_selected alignment
    inv = plan.get("inventory_header", {})
    selected = inv.get("inventory_selected", [])
    assert len(selected) == len(seq)
    for s, d in zip(selected, seq):
        assert int(s.get("planned_submit_index", -1)) == int(d.get("calendar_day_index", -2))
        assert str(s.get("planned_submit_date")) == str(d.get("submit", {}).get("date"))
        assert int(s.get("effective_contribution_days", -1)) == int(d.get("effective_contribution_days", -2))


def test_no_adjust_when_unbounded_leq_50():
    findings = [
        Finding(field="opener_b", category="status", min_days=12, duration_unit="business_days", default_decision="strong_actionable"),
        Finding(field="closer_mid", category="history", min_days=18, duration_unit="business_days", default_decision="strong_actionable"),
    ]
    bundle = _run_plan(findings)
    plan = _best_plan(bundle)
    summary = plan.get("summary", {})
    unbounded = summary.get("total_effective_days_unbounded")
    assert unbounded is not None
    assert int(unbounded) <= 50
    # Invariants
    for i, e in enumerate(plan.get("sequence_debug", [])):
        weekday = e.get("submit_on", {}).get("weekday")
        assert weekday not in {5, 6}
        if i > 0:
            assert int(e.get("handoff_days_before_prev_sla_end", 0)) >= 1


def test_two_items_preferred_over_three_when_core_met():
    findings = [
        Finding(field="opener_c", category="status", min_days=12, duration_unit="business_days", default_decision="strong_actionable"),
        Finding(field="support_m2", category="terms", min_days=6, duration_unit="business_days", default_decision="supportive_needs_companion"),
        Finding(field="closer_c", category="history", min_days=30, duration_unit="business_days", default_decision="strong_actionable"),
    ]
    bundle = _run_plan(findings)
    plan = _best_plan(bundle)
    summary = plan.get("summary", {})
    assert summary.get("last_submit_in_window", False) is True
    assert int(summary.get("total_effective_days", 0)) >= 45
    assert int(summary.get("total_items", 0)) == 2
import pytest
from datetime import date
from backend.strategy.planner import compute_optimal_plan
from backend.strategy.types import Finding

WEEKEND = {5, 6}
HOLIDAYS = set()


def _basic_findings(*tuples):
    findings = []
    for idx, (field, min_days) in enumerate(tuples):
        findings.append(
            Finding(
                field=field,
                category="test",
                min_days=min_days,
                duration_unit="business-days",
                default_decision="strong_actionable",  # treat all as actionable for opener logic
            )
        )
    return findings


def test_inbound_cap_optimizer_applies_and_caps_unbounded():
    # Closer has very large min_days to push unbounded inbound well past 50 prior to optimization.
    findings = _basic_findings(
        ("opener_large", 25),
        ("supporter_mid", 10),
        ("closer_heavy", 30),
    )
    result = compute_optimal_plan(
        findings,
        weekend=WEEKEND,
        holidays=HOLIDAYS,
        enforce_span_cap=False,
    )
    master = result["master"]
    summary = master["summary"]
    # Optimizer should run only if unbounded inbound originally > 50; we assert capped range.
    inbound_unbounded = summary.get("total_effective_days_unbounded")
    assert inbound_unbounded is not None
    assert inbound_unbounded <= 50, "Unbounded inbound should be reduced to ≤50 when legal shifts exist"
    # Invariants: deadline satisfied and overlaps >=1
    assert summary.get("last_submit_in_window") is True
    # Check overlap values
    seq = master["inventory_header"]["inventory_selected"]
    # sequence_debug available for precise overlap
    debug_seq = master["by_weekday"][str(master["best_overall"]["start_weekday"])] ["sequence_debug"]
    for entry in debug_seq[1:]:
        assert entry.get("handoff_days_before_prev_sla_end", 1) >= 1
    # inventory_selected alignment
    for i, inv in enumerate(seq):
        dbg = debug_seq[i]
        assert inv["planned_submit_index"] == dbg["calendar_day_index"]
        assert inv["effective_contribution_days"] == dbg["effective_contribution_days"]


def test_inbound_cap_optimizer_no_change_when_within_cap():
    # Smaller SLA durations keep inbound <=50 so optimizer should not modify overlaps.
    findings = _basic_findings(
        ("opener_med", 20),
        ("closer_med", 15),
    )
    result = compute_optimal_plan(
        findings,
        weekend=WEEKEND,
        holidays=HOLIDAYS,
        enforce_span_cap=False,
    )
    master = result["master"]
    summary = master["summary"]
    inbound_unbounded = summary.get("total_effective_days_unbounded")
    assert inbound_unbounded is not None and inbound_unbounded <= 50
    # Expect minimal overlap (1 business day) between items
    weekday_idx = master["best_overall"]["start_weekday"]
    debug_seq = master["by_weekday"][str(weekday_idx)]["sequence_debug"]
    if len(debug_seq) >= 2:
        overlap = debug_seq[1].get("handoff_days_before_prev_sla_end")
        assert overlap >= 1, "Overlap should remain valid (>=1) when inbound already within cap"


def test_prefers_fewer_items_once_constraints_met():
    # Provide multiple supporters that could extend coverage but planner should choose opener+closer only.
    findings = _basic_findings(
        ("opener_strong", 20),
        ("supporter_small_1", 5),
        ("supporter_small_2", 5),
        ("closer_strong", 30),
    )
    result = compute_optimal_plan(
        findings,
        weekend=WEEKEND,
        holidays=HOLIDAYS,
        enforce_span_cap=False,
        include_supporters=True,
    )
    master = result["master"]
    summary = master["summary"]
    assert summary.get("last_submit_in_window") is True
    # Minimal disputes preference: total_items should be 2 (opener + closer)
    assert summary.get("total_items") == 2, "Planner should prefer fewer items once core constraints satisfied"


@contextmanager
def _best_disabled_env():
    prev = os.environ.get("STRATEGY_BEST_WEEKDAY_ENABLED")
    os.environ["STRATEGY_BEST_WEEKDAY_ENABLED"] = "0"
    try:
        yield
    finally:
        if prev is None:
            os.environ.pop("STRATEGY_BEST_WEEKDAY_ENABLED", None)
        else:
            os.environ["STRATEGY_BEST_WEEKDAY_ENABLED"] = prev


def test_weekend_step_accepts_underrun():
    # Synthetic weekend-jump scenario (opener ~24, closer ~27 unbounded → total ~51).
    # With hard cap, optimizer should accept under-run (e.g., 49) rather than leaving 51.
    findings = [
        Finding(field="opener_24", category="status", min_days=19, duration_unit="business-days", default_decision="strong_actionable"),
        Finding(field="closer_27", category="history", min_days=19, duration_unit="business-days", default_decision="strong_actionable"),
    ]
    run_dt = datetime(2025, 1, 6, 9, 0, tzinfo=ZoneInfo("America/New_York"))  # Monday
    with _best_disabled_env():
        bundle = compute_optimal_plan(
            findings,
            weekend={5, 6},
            holidays=None,
            timezone_name="America/New_York",
            run_datetime=run_dt,
            max_calendar_span=90,
            last_submit_window=(0, 40),
            include_supporters=True,
            exclude_natural_text=True,
            strength_metric="score",
            handoff_min_business_days=1,
            handoff_max_business_days=3,
            enforce_span_cap=False,
        )
    # Assert: at least one weekday plan started >50 and ended ≤50 with cap applied
    observed_underrun = False
    for i in range(5):
        wd = bundle["weekday_plans"][i]
        summ = wd.get("summary", {})
        before = summ.get("inbound_cap_before")
        after = summ.get("inbound_cap_after")
        applied = bool(summ.get("inbound_cap_applied"))
        ub = summ.get("total_effective_days_unbounded")
        assert ub is not None
        if applied and isinstance(before, int) and isinstance(after, int) and before > 50:
            # Hard cap honored: accept ≤50, including under-run
            assert after <= 50
            assert after < before
            # Invariants
            assert summ.get("last_submit_in_window") is True
            seq = wd.get("sequence_debug", [])
            for idx, e in enumerate(seq):
                weekday = e.get("submit_on", {}).get("weekday")
                assert weekday not in {5, 6}
                if idx > 0:
                    assert int(e.get("handoff_days_before_prev_sla_end", 0)) >= 1
            # inventory alignment
            inv = wd.get("inventory_header", {})
            selected = inv.get("inventory_selected", [])
            assert len(selected) == len(seq)
            for s, d in zip(selected, seq):
                assert int(s.get("planned_submit_index", -1)) == int(d.get("calendar_day_index", -2))
                assert str(s.get("planned_submit_date")) == str(d.get("submit", {}).get("date"))
                assert int(s.get("effective_contribution_days", -1)) == int(d.get("effective_contribution_days", -2))
            observed_underrun = True
    assert observed_underrun, "Expected at least one weekday to under-run (≤50) from >50"


def test_per_weekday_mode_applies_optimizer_to_all_plans():
    """Regression test: when STRATEGY_BEST_WEEKDAY_ENABLED=0, optimizer must apply to ALL weekday plans."""
    findings = [
        Finding(field="payment_status", category="status", min_days=19, duration_unit="business_days", default_decision="strong_actionable"),
        Finding(field="seven_year_history", category="history", min_days=19, duration_unit="business_days", default_decision="strong_actionable"),
    ]
    
    # Per-weekday mode: expect ALL weekday plans to have optimizer applied
    with _best_disabled_env():
        bundle = _run_plan(findings)
        weekday_plans = bundle.get("weekday_plans", {})
        
        # Check that ALL weekday plans with unbounded > 50 have been optimized
        for weekday, plan in weekday_plans.items():
            summ = plan.get("summary", {})
            ub = summ.get("total_effective_days_unbounded")
            
            if ub is not None and int(ub) > 0:
                # If plan had content, it should be capped to ≤50 with metadata
                assert int(ub) <= 50, f"Weekday {weekday}: expected unbounded ≤50, got {ub}"
                
                # Check metadata presence (at least one weekday should have applied optimizer)
                # Since this is opener=24 + closer=27=51, at least wd0 should show metadata
                if weekday == 0:  # Monday anchor matches test scenario
                    assert "inbound_cap_applied" in summ, f"Weekday {weekday}: missing metadata"
                    assert summ.get("inbound_cap_applied") is True
                    assert summ.get("inbound_cap_before") == 51
                    assert summ.get("inbound_cap_after") == 50


def test_overlap_fields_consistent_after_cap():
    """Validate overlap fields and identity on a 2-item chain after hard-cap optimization in per-weekday mode."""
    findings = [
        Finding(field="payment_status", category="status", min_days=19, duration_unit="business_days", default_decision="strong_actionable"),
        Finding(field="seven_year_history", category="history", min_days=19, duration_unit="business_days", default_decision="strong_actionable"),
    ]

    with _best_disabled_env():
        bundle = _run_plan(findings)
        plan = bundle["weekday_plans"][0]  # Monday
        summ = plan.get("summary", {})

        # Basic presence
        assert "total_effective_days_unbounded" in summ
        assert "total_overlap_unbounded_days" in summ

        total_unbounded = int(summ.get("total_effective_days_unbounded", 0) or 0)
        total_overlap = int(summ.get("total_overlap_unbounded_days", 0) or 0)

        inv_sel = plan.get("inventory_header", {}).get("inventory_selected", [])
        assert len(inv_sel) >= 2
        unbounded_vals = [int(e.get("effective_contribution_days_unbounded", 0) or 0) for e in inv_sel]
        sum_unbounded = sum(unbounded_vals)

        # Identity: total_unbounded = sum(unbounded) - total_overlap
        assert total_unbounded == sum_unbounded - total_overlap

        # For a two-item chain, overlap at the second item equals total_overlap
        seq_dbg = plan.get("sequence_debug", [])
        assert len(seq_dbg) >= 2
        overlap_eff = int(seq_dbg[1].get("overlap_effective_unbounded_with_prev", 0) or 0)
        assert overlap_eff == total_overlap

        # Mirrored in compact and inventory_selected for the second item
        seq_comp = plan.get("sequence_compact", [])
        assert int(seq_comp[1].get("overlap_days_with_prev", 0) or 0) == total_overlap
        assert int(inv_sel[1].get("overlap_days_with_prev", 0) or 0) == total_overlap

        # If hard-cap applied, metadata should exist and be coherent
        if summ.get("inbound_cap_applied"):
            before = int(summ.get("inbound_cap_before", 0) or 0)
            after = int(summ.get("inbound_cap_after", 0) or 0)
            assert before > after
            assert after == total_unbounded

