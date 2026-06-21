from __future__ import annotations

from dataclasses import dataclass, field
import json
from typing import Any


def _as_cell(value: Any) -> tuple[int, int]:
    if isinstance(value, tuple) and len(value) == 2:
        return (int(value[0]), int(value[1]))
    if isinstance(value, list) and len(value) == 2:
        return (int(value[0]), int(value[1]))
    raise ValueError("cell must be a two-item row/col sequence")


@dataclass(frozen=True)
class AssignmentAction:
    car_id: str
    person_id: str

    @classmethod
    def from_any(cls, raw: Any) -> "AssignmentAction":
        if not isinstance(raw, dict):
            raise ValueError("assignment must be an object")
        return cls(car_id=str(raw["car_id"]), person_id=str(raw["person_id"]))

    def to_dict(self) -> dict[str, str]:
        return {"car_id": self.car_id, "person_id": self.person_id}


@dataclass(frozen=True)
class RepositionAction:
    car_id: str
    target: tuple[int, int]

    @classmethod
    def from_any(cls, raw: Any) -> "RepositionAction":
        if not isinstance(raw, dict):
            raise ValueError("reposition must be an object")
        return cls(car_id=str(raw["car_id"]), target=_as_cell(raw["target"]))

    def to_dict(self) -> dict[str, Any]:
        return {"car_id": self.car_id, "target": list(self.target)}


@dataclass
class ActionPlan:
    assignments: list[AssignmentAction] = field(default_factory=list)
    repositions: list[RepositionAction] = field(default_factory=list)
    holds: list[str] = field(default_factory=list)
    rationale: str = ""

    @classmethod
    def from_any(cls, raw: Any) -> "ActionPlan":
        if isinstance(raw, ActionPlan):
            return raw
        if raw is None:
            return cls()
        if isinstance(raw, str):
            raw = raw.strip()
            if not raw:
                return cls()
            raw = json.loads(raw)
        if not isinstance(raw, dict):
            raise ValueError("action plan must be a JSON object")
        return cls(
            assignments=[
                AssignmentAction.from_any(item)
                for item in raw.get("assignments", [])
            ],
            repositions=[
                RepositionAction.from_any(item)
                for item in raw.get("repositions", [])
            ],
            holds=[str(item) for item in raw.get("holds", [])],
            rationale=str(raw.get("rationale", "")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "assignments": [item.to_dict() for item in self.assignments],
            "repositions": [item.to_dict() for item in self.repositions],
            "holds": self.holds,
            "rationale": self.rationale,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), separators=(",", ":"), allow_nan=False)
