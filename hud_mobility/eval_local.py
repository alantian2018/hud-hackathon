from __future__ import annotations

import argparse
import json
import statistics

from .planners import build_value_aware_plan
from .world import MobilityWorld


def run_episode(seed: int, event_id: str | None, horizon_steps: int, fleet_size: int, demand_scale: float) -> dict:
    world = MobilityWorld(
        (14, 14),
        seed=seed,
        fleet_size=fleet_size,
        horizon_steps=horizon_steps,
        event_id=event_id,
        demand_scale=demand_scale,
    )
    while not world.done:
        world.step(build_value_aware_plan(world))
    return world.reward()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run local non-greedy mobility orchestrator smoke evals.")
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--horizon-steps", type=int, default=8)
    parser.add_argument("--fleet-size", type=int, default=20)
    parser.add_argument("--demand-scale", type=float, default=1.0)
    args = parser.parse_args()

    events = [None, "chase_center_exit", "market_st_surge", "fidi_conference"]
    results = []
    for idx in range(args.episodes):
        event_id = events[idx % len(events)]
        result = run_episode(
            seed=100 + idx,
            event_id=event_id,
            horizon_steps=args.horizon_steps,
            fleet_size=args.fleet_size,
            demand_scale=args.demand_scale,
        )
        results.append(result)
        print(json.dumps({"episode": idx, "event_id": event_id, **result}, sort_keys=True))

    rewards = [item["reward"] for item in results]
    print(
        json.dumps(
            {
                "episodes": len(results),
                "mean_reward": round(statistics.mean(rewards), 6),
                "reward_std": round(statistics.pstdev(rewards), 6) if len(rewards) > 1 else 0.0,
                "min_reward": min(rewards),
                "max_reward": max(rewards),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

