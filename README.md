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

## HUD Manual Dispatch Environment

Install the Python package into your active venv:

```bash
python -m pip install -e .
```

Set your HUD key once:

```bash
hud set HUD_API_KEY=...
```

The first command to run is an evaluation. It starts the HUD environment, exposes the dispatch tools, lets the trainable Qwen model route cars, and reports a HUD reward:

```bash
hud eval tasks.py openai_compatible \
  --gateway \
  --model Qwen/Qwen3.5-4B \
  --task-ids manual-dispatch-balanced \
  --max-steps 40 \
  --yes
```

Fork the base Qwen model once so HUD's RL service has a team-owned trainable
target for checkpoints:

```bash
hud models fork Qwen/Qwen3.5-4B --name manual-dispatch-qwen35-4b --json
```

Use the returned `model_name` for training. For example, run one small Tinker
training update from HUD rollouts. The training script keeps simulator state in
`runs/hud/{slug}_state.pkl`, so later rollouts continue the same fleet world
instead of resetting:

```bash
python scripts/train_manual_dispatch.py \
  --model manual-dispatch-qwen35-4b \
  --steps 1 \
  --group-size 2 \
  --learning-rate 1e-5 \
  --reset-persistent-state
```

Continue training from the saved simulator state:

```bash
python scripts/train_manual_dispatch.py \
  --model manual-dispatch-qwen35-4b \
  --steps 3 \
  --group-size 2
```

Use `--no-persistent-state` when you intentionally want standard independent
HUD episodes.

Use the surge task as a second persistent scenario:

```bash
python scripts/train_manual_dispatch.py \
  --model manual-dispatch-qwen35-4b \
  --task-id manual-dispatch-surge \
  --steps 3 \
  --group-size 2
```

## Important Files

- `main.jsx`: React/Deck.gl map UI and overlays.
- `map.py`: OSMnx network and base data generator.
- `export_mobility_world.py`: stateful greedy dispatch export for the frontend.
- `mobility_sim/generators.py`: demand, traffic, people, and world generator logic.
- `public/data/`: generated JSON artifacts used by the map.
- `env.py`: HUD environment exposing the manual passenger-dispatch tools.
- `tasks.py`: HUD task rows for balanced and surge dispatch episodes.
- `scripts/train_manual_dispatch.py`: HUD/Tinker training loop for the LLM dispatcher.
