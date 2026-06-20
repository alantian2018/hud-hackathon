#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from mobility_sim import WorldGenerators


def load_grid(path: Path) -> dict | tuple[int, int]:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return (50, 50)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export greedy mobility snapshots for the map UI.")
    parser.add_argument("--grid", default="public/data/population_density_grid.json")
    parser.add_argument("--out", default="public/data/mobility_world.json")
    parser.add_argument("--seed", default=7, type=int)
    parser.add_argument("--fleet-size", default=16, type=int)
    parser.add_argument("--step-minutes", default=60, type=int)
    args = parser.parse_args()

    grid = load_grid(Path(args.grid))
    snapshots = []
    for timestep in range(0, 24 * 60, max(1, args.step_minutes)):
        world = WorldGenerators(grid=grid, seed=args.seed + timestep, fleet_size=args.fleet_size)
        snapshots.append(world.step(timestep))

    payload = {
        "seed": args.seed,
        "fleet_size": args.fleet_size,
        "step_minutes": args.step_minutes,
        "snapshots": snapshots,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {len(snapshots)} greedy mobility snapshots to {out_path}")


if __name__ == "__main__":
    main()
