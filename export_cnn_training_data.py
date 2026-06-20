#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from mobility_sim import CnnFeatureConfig, WorldGenerators


def load_grid(path: Path) -> dict | tuple[int, int]:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    fallback = Path("dist/data/population_density_grid.json")
    if fallback.exists():
        return json.loads(fallback.read_text(encoding="utf-8"))
    return (50, 50)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export per-car CNN training examples from the backend mobility generators."
    )
    parser.add_argument("--grid", default="public/data/population_density_grid.json")
    parser.add_argument("--out", default="cnn_training_examples.jsonl")
    parser.add_argument("--seed", default=7, type=int)
    parser.add_argument("--fleet-size", default=24, type=int)
    parser.add_argument("--step-minutes", default=15, type=int)
    parser.add_argument("--patch-radius", default=3, type=int)
    parser.add_argument("--max-steps", default=0, type=int, help="0 means run a full simulated day.")
    args = parser.parse_args()

    grid = load_grid(Path(args.grid))
    world = WorldGenerators(
        grid=grid,
        seed=args.seed,
        fleet_size=args.fleet_size,
        cnn_config=CnnFeatureConfig(patch_radius=args.patch_radius),
    )

    step = max(1, args.step_minutes)
    timesteps = list(range(0, 24 * 60, step))
    if args.max_steps > 0:
        timesteps = timesteps[: args.max_steps]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with out_path.open("w", encoding="utf-8") as handle:
        for timestep in timesteps:
            payload = world.step(timestep)
            agent_state = payload["agent_state"]
            global_state = agent_state["global_state"]
            for example in agent_state["cnn"]["training_examples"]:
                row = {
                    "seed": args.seed,
                    "timestep": timestep,
                    "grid": global_state["grid"],
                    "business_metrics": global_state["business_metrics"],
                    "global_summary": {
                        "demand": global_state["demand"],
                        "traffic": global_state["traffic"],
                        "num_active_requests": len(global_state["active_requests"]),
                        "num_cars": len(global_state["fleet_distribution"]["cars"]),
                    },
                    "example": example,
                }
                handle.write(json.dumps(row, separators=(",", ":"), allow_nan=False))
                handle.write("\n")
                count += 1

    print(f"Wrote {count} CNN training examples to {out_path}")


if __name__ == "__main__":
    main()

