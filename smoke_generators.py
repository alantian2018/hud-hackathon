#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from mobility_sim import WorldGenerators


def load_grid(path: Path) -> dict | tuple[int, int]:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return (10, 10)


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke test backend mobility generators.")
    parser.add_argument("--grid", default="dist/data/population_density_grid.json")
    parser.add_argument("--seed", default=7, type=int)
    parser.add_argument("--timestep", default=8 * 60, type=int)
    parser.add_argument("--fleet-size", default=16, type=int)
    args = parser.parse_args()

    grid = load_grid(Path(args.grid))
    world = WorldGenerators(grid=grid, seed=args.seed, fleet_size=args.fleet_size)
    payload = world.step(args.timestep)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
