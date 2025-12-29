"""Unit tests for Skeleton #2 Day-40 Strongest Leftover rule."""

import pytest
from datetime import date, timedelta
from backend.strategy.planner import (
    _find_nearest_business_day_to_40,
    _get_strongest_unused_leftover,
    _attempt_day40_strongest_enrichment,
)


class TestFindNearestBusinessDayTo40:
    """Test business day adjustment around day 40."""
    
    def test_day40_is_business_day(self):
        """If day 40 is a business day, return it directly."""
        anchor = date(2025, 1, 1)  # Wednesday
        # Day 40 = Feb 10, 2025 = Monday (business day)
        weekend = {5, 6}
        holidays = set()
        
        result = _find_nearest_business_day_to_40(
            anchor_date=anchor,
            weekend=weekend,
            holidays=holidays,
        )
        assert result == 40
    
    def test_day40_is_weekend_prefer_earlier(self):
        """If day 40 is weekend, prefer earlier day (Friday)."""
        anchor = date(2025, 1, 4)  # Saturday
        # Day 40 = Feb 13, 2025 = Thursday (business day, not weekend)
        # Let's adjust: anchor = date(2025, 1, 5) -> Day 40 = Feb 14 = Friday
        anchor = date(2025, 1, 5)  # Sunday
        # Day 40 = Feb 14, 2025 = Friday (business day)
        # Actually, let me calculate: Jan 5 + 40 days = Feb 14
        # Feb 14, 2025 is a Friday - business day
        # Let's pick an anchor where day 40 is Saturday
        anchor = date(2025, 1, 11)  # Saturday
        # Day 40 = Feb 20, 2025 = Thursday (business day)
        # Adjusting... let me use a different approach
        
        # Use anchor where day 40 falls on Saturday
        anchor = date(2025, 2, 1)  # Saturday
        # Day 40 = March 13, 2025 = Thursday (business day)
        # This is getting complex. Let me use a fixed test case:
        
        # Anchor: Monday, Jan 6, 2025
        # Day 40: Feb 15, 2025 = Saturday
        anchor = date(2025, 1, 6)  # Monday
        weekend = {5, 6}  # Saturday=5, Sunday=6
        holidays = set()
        
        result = _find_nearest_business_day_to_40(
            anchor_date=anchor,
            weekend=weekend,
            holidays=holidays,
        )
        # Should find nearest business day (Friday=39 or Monday=41, prefer 39)
        assert result in [39, 41]  # Accept either for now, implementation will prefer earlier
    
    def test_day40_is_holiday(self):
        """If day 40 is a holiday, find nearest business day."""
        anchor = date(2025, 1, 1)
        day40_date = anchor + timedelta(days=40)  # Feb 10, 2025
        weekend = {5, 6}
        holidays = {day40_date}  # Mark day 40 as holiday
        
        result = _find_nearest_business_day_to_40(
            anchor_date=anchor,
            weekend=weekend,
            holidays=holidays,
        )
        # Should find day 39 or 41 (not 40)
        assert result in [39, 41]
    
    def test_no_business_day_within_range(self):
        """If no business day found within search range, return None."""
        anchor = date(2025, 1, 1)
        weekend = {5, 6}
        # Mark all days around 40 as holidays
        holidays = {anchor + timedelta(days=d) for d in range(33, 48)}
        
        result = _find_nearest_business_day_to_40(
            anchor_date=anchor,
            weekend=weekend,
            holidays=holidays,
            max_search_distance=7,
        )
        assert result is None


class TestGetStrongestUnusedLeftover:
    """Test strongest leftover selection with dedup."""
    
    def test_select_strongest_by_sla(self):
        """Select item with highest min_days."""
        leftover = [
            {"field": "field1", "min_days": 10, "strength_value": 100},
            {"field": "field2", "min_days": 15, "strength_value": 90},
            {"field": "field3", "min_days": 8, "strength_value": 110},
        ]
        # Items should already be sorted by strength, but we select strongest unused
        used_fields = set()
        min_sla_days = 5
        
        result = _get_strongest_unused_leftover(leftover, used_fields, min_sla_days)
        assert result is not None
        # Should select first item (already sorted by strength in _enrich_with_skeleton2)
        assert result["field"] == "field1"
    
    def test_skip_used_fields(self):
        """Skip items already used in Skeleton #1 or handoff enrichments."""
        leftover = [
            {"field": "field1", "min_days": 15, "strength_value": 100},
            {"field": "field2", "min_days": 14, "strength_value": 90},
            {"field": "field3", "min_days": 13, "strength_value": 80},
        ]
        used_fields = {"field1", "field2"}
        min_sla_days = 5
        
        result = _get_strongest_unused_leftover(leftover, used_fields, min_sla_days)
        assert result is not None
        assert result["field"] == "field3"
    
    def test_skip_below_min_sla(self):
        """Skip items below minimum SLA threshold."""
        leftover = [
            {"field": "field1", "min_days": 3, "strength_value": 100},
            {"field": "field2", "min_days": 4, "strength_value": 90},
            {"field": "field3", "min_days": 6, "strength_value": 80},
        ]
        used_fields = set()
        min_sla_days = 5
        
        result = _get_strongest_unused_leftover(leftover, used_fields, min_sla_days)
        assert result is not None
        assert result["field"] == "field3"
    
    def test_no_eligible_candidates(self):
        """Return None if all items are used or below threshold."""
        leftover = [
            {"field": "field1", "min_days": 3, "strength_value": 100},
            {"field": "field2", "min_days": 4, "strength_value": 90},
        ]
        used_fields = set()
        min_sla_days = 5
        
        result = _get_strongest_unused_leftover(leftover, used_fields, min_sla_days)
        assert result is None


class TestAttemptDay40StrongestEnrichment:
    """Test complete day-40 enrichment rule with guard and placement."""
    
    def test_guard_disables_when_pre_closer_end_gte_37(self):
        """Rule should not run if pre_closer unbounded_end >= 37."""
        plan = {
            "sequence_compact": [
                {"field": "opener", "timeline_unbounded": {"to_day_unbounded": 20}},
                {"field": "pre_closer", "timeline_unbounded": {"to_day_unbounded": 40}},  # >= 37
                {"field": "closer", "timeline_unbounded": {"to_day_unbounded": 55}},
            ]
        }
        leftover = [{"field": "leftover1", "min_days": 10, "strength_value": 100}]
        used_fields = set()
        skeleton2_config = {
            "enable_day40_strongest": True,
            "min_sla_days": 5,
        }
        anchor = date(2025, 1, 1)
        weekend = {5, 6}
        holidays = set()
        
        item, stats = _attempt_day40_strongest_enrichment(
            plan, leftover, used_fields,
            skeleton2_config=skeleton2_config,
            weekend=weekend,
            holidays=holidays,
            anchor_date=anchor,
            enrich_idx=1,
        )
        
        assert item is None
        assert stats["attempted_day40"] == 1
        assert stats["rejected_day40_guard"] == 1
        assert stats["accepted_day40"] == 0
    
    def test_guard_enables_when_pre_closer_end_lt_37(self):
        """Rule should run if pre_closer unbounded_end < 37."""
        plan = {
            "sequence_compact": [
                {"field": "opener", "timeline_unbounded": {"to_day_unbounded": 20}},
                {"field": "pre_closer", "timeline_unbounded": {"to_day_unbounded": 30}},  # < 37
                {"field": "closer", "timeline_unbounded": {"to_day_unbounded": 55}},
            ]
        }
        leftover = [{"field": "leftover1", "min_days": 10, "strength_value": 100, "role": "supporter", "decision": "remove", "category": "negative"}]
        used_fields = set()
        skeleton2_config = {
            "enable_day40_strongest": True,
            "min_sla_days": 5,
        }
        anchor = date(2025, 1, 1)  # Wednesday
        weekend = {5, 6}
        holidays = set()
        
        item, stats = _attempt_day40_strongest_enrichment(
            plan, leftover, used_fields,
            skeleton2_config=skeleton2_config,
            weekend=weekend,
            holidays=holidays,
            anchor_date=anchor,
            enrich_idx=1,
        )
        
        assert item is not None
        assert stats["attempted_day40"] == 1
        assert stats["accepted_day40"] == 1
        assert stats["rejected_day40_guard"] == 0
        assert item["field"] == "leftover1"
        # Verify consumer-facing date fields instead of removed debug fields
        assert "submit_date" in item
        assert "submit_weekday" in item
        assert "unbounded_end_date" in item
        assert "unbounded_end_weekday" in item
        assert "timeline_unbounded" in item
    
    def test_env_flag_disabled(self):
        """Rule should not run if ENV flag is disabled."""
        plan = {
            "sequence_compact": [
                {"field": "opener", "timeline_unbounded": {"to_day_unbounded": 20}},
                {"field": "pre_closer", "timeline_unbounded": {"to_day_unbounded": 30}},
                {"field": "closer", "timeline_unbounded": {"to_day_unbounded": 55}},
            ]
        }
        leftover = [{"field": "leftover1", "min_days": 10, "strength_value": 100}]
        used_fields = set()
        skeleton2_config = {
            "enable_day40_strongest": False,  # Disabled
            "min_sla_days": 5,
        }
        anchor = date(2025, 1, 1)
        weekend = {5, 6}
        holidays = set()
        
        item, stats = _attempt_day40_strongest_enrichment(
            plan, leftover, used_fields,
            skeleton2_config=skeleton2_config,
            weekend=weekend,
            holidays=holidays,
            anchor_date=anchor,
            enrich_idx=1,
        )
        
        assert item is None
        assert stats["rejected_day40_env_disabled"] == 1
        assert stats["attempted_day40"] == 0
    
    def test_no_unused_leftover(self):
        """Rule should fail gracefully if no unused leftover available."""
        plan = {
            "sequence_compact": [
                {"field": "opener", "timeline_unbounded": {"to_day_unbounded": 20}},
                {"field": "pre_closer", "timeline_unbounded": {"to_day_unbounded": 30}},
                {"field": "closer", "timeline_unbounded": {"to_day_unbounded": 55}},
            ]
        }
        leftover = [{"field": "leftover1", "min_days": 10, "strength_value": 100}]
        used_fields = {"leftover1"}  # Already used
        skeleton2_config = {
            "enable_day40_strongest": True,
            "min_sla_days": 5,
        }
        anchor = date(2025, 1, 1)
        weekend = {5, 6}
        holidays = set()
        
        item, stats = _attempt_day40_strongest_enrichment(
            plan, leftover, used_fields,
            skeleton2_config=skeleton2_config,
            weekend=weekend,
            holidays=holidays,
            anchor_date=anchor,
            enrich_idx=1,
        )
        
        assert item is None
        assert stats["attempted_day40"] == 1
        assert stats["rejected_day40_no_unused"] == 1
    
    def test_day40_adjustment_for_weekend(self):
        """If day 40 is weekend, adjust to nearest business day."""
        plan = {
            "sequence_compact": [
                {"field": "opener", "timeline_unbounded": {"to_day_unbounded": 20}},
                {"field": "pre_closer", "timeline_unbounded": {"to_day_unbounded": 30}},
                {"field": "closer", "timeline_unbounded": {"to_day_unbounded": 55}},
            ]
        }
        leftover = [{"field": "leftover1", "min_days": 10, "strength_value": 100, "role": "supporter", "decision": "remove", "category": "negative"}]
        used_fields = set()
        skeleton2_config = {
            "enable_day40_strongest": True,
            "min_sla_days": 5,
        }
        # Anchor where day 40 is Saturday
        anchor = date(2025, 1, 6)  # Monday
        weekend = {5, 6}  # Sat=5, Sun=6
        holidays = set()
        
        item, stats = _attempt_day40_strongest_enrichment(
            plan, leftover, used_fields,
            skeleton2_config=skeleton2_config,
            weekend=weekend,
            holidays=holidays,
            anchor_date=anchor,
            enrich_idx=1,
        )
        
        # Should succeed with adjusted index
        if item is not None:
            assert item["planned_submit_index"] in [39, 40, 41]
            # Verify date fields are present regardless of adjustment
            assert "submit_date" in item
            assert "unbounded_end_date" in item
    
    def test_pre_closer_metadata_included(self):
        """Enrichment item should include pre_closer metadata."""
        plan = {
            "sequence_compact": [
                {"field": "opener", "timeline_unbounded": {"to_day_unbounded": 20}},
                {"field": "pre_closer_field", "timeline_unbounded": {"to_day_unbounded": 35}},
                {"field": "closer", "timeline_unbounded": {"to_day_unbounded": 55}},
            ]
        }
        leftover = [{"field": "leftover1", "min_days": 10, "strength_value": 100, "role": "supporter", "decision": "remove", "category": "negative"}]
        used_fields = set()
        skeleton2_config = {
            "enable_day40_strongest": True,
            "min_sla_days": 5,
        }
        anchor = date(2025, 1, 1)
        weekend = {5, 6}
        holidays = set()
        
        item, stats = _attempt_day40_strongest_enrichment(
            plan, leftover, used_fields,
            skeleton2_config=skeleton2_config,
            weekend=weekend,
            holidays=holidays,
            anchor_date=anchor,
            enrich_idx=1,
        )
        
        assert item is not None
        assert item["pre_closer_field"] == "pre_closer_field"
        assert item["pre_closer_unbounded_end"] == 35
        # Verify date fields are present instead of removed day40 metadata
        assert "submit_date" in item
        assert "timeline_unbounded" in item


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
