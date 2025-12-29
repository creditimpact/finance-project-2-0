from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional
from uuid import NAMESPACE_URL, uuid5


def stable_uuid(*components: str) -> str:
    """Generate a deterministic UUID5 from the provided components."""
    name = ":".join(components)
    return str(uuid5(NAMESPACE_URL, name))


@dataclass
class Step:
    """Single actionable step within a cycle."""

    allowed_tags: List[str]
    sla_days: int
    dependent_on: Optional[str] = None
    step_id: str = ""

    def __post_init__(self) -> None:
        if not self.step_id:
            comp = ",".join(sorted(self.allowed_tags))
            comp = f"{comp}|{self.sla_days}|{self.dependent_on or ''}"
            self.step_id = stable_uuid(comp)


@dataclass
class Cycle:
    """A cycle groups a collection of steps."""

    steps: List[Step] = field(default_factory=list)
    cycle_id: str = ""

    def __post_init__(self) -> None:
        if not self.cycle_id:
            step_ids = ",".join(step.step_id for step in self.steps)
            self.cycle_id = stable_uuid(step_ids)


@dataclass
class StrategyPlan:
    """A full strategy plan composed of cycles and steps."""

    version: int
    cycles: List[Cycle] = field(default_factory=list)
    plan_id: str = ""

    def __post_init__(self) -> None:
        if not self.plan_id:
            cycle_ids = ",".join(cycle.cycle_id for cycle in self.cycles)
            self.plan_id = stable_uuid(str(self.version), cycle_ids)
