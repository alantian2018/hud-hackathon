from __future__ import annotations

from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hud import Taskset

from hud_mobility.env import optimize_mobility


TASK_SPECS = [
    {"seed": 11, "start_minute": 7 * 60, "horizon_steps": 8, "event_id": None, "demand_scale": 0.90},
    {"seed": 12, "start_minute": 8 * 60, "horizon_steps": 8, "event_id": None, "demand_scale": 1.00},
    {"seed": 13, "start_minute": 17 * 60, "horizon_steps": 8, "event_id": None, "demand_scale": 1.15},
    {"seed": 21, "start_minute": 19 * 60, "horizon_steps": 8, "event_id": "chase_center_exit", "demand_scale": 1.15},
    {"seed": 22, "start_minute": 12 * 60, "horizon_steps": 8, "event_id": "market_st_surge", "demand_scale": 1.10},
    {"seed": 23, "start_minute": 16 * 60, "horizon_steps": 8, "event_id": "fidi_conference", "demand_scale": 1.12},
]


def build_taskset() -> Taskset:
    tasks = []
    for idx, spec in enumerate(TASK_SPECS):
        task = optimize_mobility(
            fleet_size=20,
            rows=14,
            cols=14,
            **spec,
        )
        task.slug = f"mobility-orchestrator-{idx:02d}-{spec['event_id'] or 'base'}"
        task.agent_config = {"max_steps": 12}
        tasks.append(task)
    return Taskset("mobility-orchestrator-v1", tasks)


taskset = build_taskset()
