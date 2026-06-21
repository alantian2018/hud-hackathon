"""HUD mobility orchestrator package.

This package turns the demo mobility simulator into a HUD-trainable
orchestration environment. The training reward is computed from absolute
episode metrics and does not compare against the existing greedy baseline.
"""

from .schemas import ActionPlan, AssignmentAction, RepositionAction
from .world import MobilityWorld

__all__ = [
    "ActionPlan",
    "AssignmentAction",
    "MobilityWorld",
    "RepositionAction",
]

