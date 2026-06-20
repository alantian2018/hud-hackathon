# HUD Hackathon Mobility Map

Interactive San Francisco mobility demo with Deck.gl, MapTiler, OSMnx road data, grid overlays, demand generation, and a stateful greedy ride-dispatch simulation.

## Prerequisites

- Node.js 18+
- npm
- Python 3.10+
- Optional, only for rebuilding the OSMnx road dataset: `osmnx` and `networkx`

The generated demo data is checked into `public/data`, so the app can run without rebuilding the map dataset.

## Setup

```bash
git clone https://github.com/alantian2018/hud-hackathon.git
cd hud-hackathon
npm install
```

## Run the Map

```bash
npm run dev
```

Open the local URL printed by Vite. The npm script starts at `http://127.0.0.1:5173`; if that port is already busy, Vite may use the next available port, such as `5174`.

The MapTiler key currently lives at the top of `main.jsx` as `MAPTILER_KEY`. Replace it there if the basemap ever stops loading.

## Main Demo Controls

- `Cars Grid`: shows vehicle occupancy by grid cell.
- `People Grid`: shows pickup cells in blue and destination cells in red.
- `Greedy`: enables the stateful greedy dispatch simulation.
- `Fast Forward`: changes simulated playback speed.
- Time presets jump the simulation to morning, midday, evening rush, or night.

## Regenerate Simulation Data

The greedy dispatch, demand, traffic, and people snapshots are exported to `public/data/mobility_world.json`.

```bash
python3 export_mobility_world.py
```

Useful options:

```bash
python3 export_mobility_world.py --fleet-size 40 --step-minutes 15 --seed 7
```

The export is intentionally scaled for the demo so route overlays stay readable. The default generated file is large because it includes full Dijkstra route coordinates for playback.

The map export now keeps one stable demand generator, one stable traffic generator, and one stable people generator alive across the simulated day. Demand hotspots evolve with time/noise instead of being re-randomized every snapshot, which makes the data better for comparing greedy dispatch against future AI agents.

## CNN / Agent Data

The backend can also expose MobilitySim-style state for the CNN and LLM orchestrator:

- Global state summary: top demand cells, traffic bottlenecks, active requests, fleet distribution, and greedy business metrics.
- CNN feature channels: demand, traffic, car occupancy, idle-car occupancy, active pickups, active dropoffs, and reserved targets.
- Per-car local patches around the car.
- Candidate next-cell actions: wait, north, east, south, west, clipped to map bounds.
- Greedy baseline labels so the CNN can start by imitating the current dispatcher.

Export JSONL examples for CNN work:

```bash
python3 export_cnn_training_data.py --fleet-size 24 --step-minutes 15 --patch-radius 3
```

Each JSONL row contains one car example with `local_patch`, `candidate_moves`, and a `label` derived from the greedy baseline. This is the handoff point for replacing greedy with CNN proposals and then comparing an AI agent/orchestrator against greedy.

## Rebuild OSMnx Road Data

Only do this if you need to regenerate the base map/network artifacts.

```bash
python3 -m pip install osmnx networkx
npm run build:osmnx
python3 export_mobility_world.py
```

`npm run build:osmnx` writes road, node, population-grid, and sample-trip artifacts into `public/data`.

## Tests and Checks

```bash
python3 -B -m unittest test_generators.py
./node_modules/.bin/vite build --outDir /private/tmp/hud-hackathon-build
```

The Vite build currently emits a large-chunk warning because the app bundle and checked-in demo data are sizeable; that warning does not block the build.

## Important Files

- `main.jsx`: React/Deck.gl map UI and overlays.
- `map.py`: OSMnx network and base data generator.
- `export_mobility_world.py`: stateful greedy dispatch export for the frontend.
- `export_cnn_training_data.py`: JSONL export for CNN local-patch training examples.
- `mobility_sim/generators.py`: demand, traffic, people, and world generator logic.
- `public/data/`: generated JSON artifacts used by the map.
