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

Pages:

- `/`: existing greedy dispatch map.
- `/rl.html`: Agentic Fleet map using the precomputed policy feed.
- `/compare.html`: greedy and Agentic Fleet maps side by side with one shared Start/Pause/Reset control.

Copy `.env.example` to `.env.local` and set `VITE_MAPTILER_KEY` to your MapTiler key if the basemap stops loading.

## Main Demo Controls

- `Cars Grid`: shows vehicle occupancy by grid cell.
- `People Grid`: shows pickup cells in blue and destination cells in red.
- `Greedy`: enables the stateful greedy dispatch simulation.
- `Speed`: starts at `0.5x` and cycles through slower/faster demo speeds.
- `Events`: toggles one of three preplanned demand surges: Chase Center exit, Market St surge, or FiDi conference surge. Each event is marked with a red surge circle and generates more people near that location, changing traffic pressure, routes, and stats.
- Time presets jump the simulation to morning, midday, evening rush, or night.

## Regenerate Simulation Data

The greedy dispatch snapshots are exported to `public/data/mobility_world.json`. The export uses 5-minute ticks by default so requests and assignments flow continuously during the demo, plus three lighter event timelines for the scenario toggles.

```bash
python3 export_mobility_world.py
```

Useful options:

```bash
python3 export_mobility_world.py --fleet-size 40 --step-minutes 5 --seed 7
```

The export is intentionally scaled for the demo so route overlays stay readable. Dijkstra route geometry is deduplicated into a shared route table to keep the checked-in demo file manageable.

Precompute the Agentic Fleet comparison feed, including the same three event timelines used by the greedy page:

```bash
python3 precompute_orchestrator_world.py --include-events
```

This writes `public/data/mobility_orchestrator_world.json`, using the existing greedy export as the shared request/timeline source. Omit `--include-events` when you only want to rebuild the base 24-hour comparison feed quickly.

Current base comparison feed:

- greedy: `240` completed trips, `$8,622.26` profit, `73.19%` served, `5.51m` average wait;
- Agentic Fleet: `291` completed trips, `$10,699.16` profit, `88.55%` served, `4.57m` average wait;
- delta: `+51` completed trips, `+$2,076.90` profit, `+15.36pp` served demand, `-0.94m` average wait.

Current event comparison deltas:

- Chase Center exit: `+127` completed trips, `+$6,903.50` profit, `+12.50pp` served demand, `-3.21m` average wait.
- Market St surge: `+185` completed trips, `+$8,640.02` profit, `+22.48pp` served demand, `-7.75m` average wait.
- FiDi conference: `+171` completed trips, `+$8,165.09` profit, `+20.09pp` served demand, `-5.72m` average wait.

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
python3 -B -m unittest test_hud_mobility.py
python3 -m hud_mobility.eval_local --episodes 8 --horizon-steps 8 --fleet-size 20
python3 precompute_orchestrator_world.py --include-events
./node_modules/.bin/vite build --outDir /private/tmp/hud-hackathon-build
```

The Vite build currently emits a large-chunk warning because the app bundle and checked-in demo data are sizeable; that warning does not block the build.

## HUD LLM Orchestrator

This branch adds a HUD-trainable fleet orchestrator in `hud_mobility/`. It does not use the demo greedy dispatcher for training reward. Instead, the HUD task exposes MCP tools for:

- observing fleet, traffic, demand, and pending requests;
- forecasting future demand hotspots;
- proposing global value/urgency-aware matches;
- proposing idle-car repositioning;
- running one full episode through non-greedy matching/repositioning specialists;
- stepping the simulator with an LLM-produced JSON action plan;
- submitting an episode for an absolute reward.

Install the HUD runtime dependencies:

```bash
python3 -m pip install -r requirements-hud.txt
hud set HUD_API_KEY=...
```

Run a local heuristic smoke test of the same non-greedy action API:

```bash
python3 -m hud_mobility.eval_local --episodes 8 --horizon-steps 8 --fleet-size 20
```

Compare the specialist planner against the nearest-car baseline on the six HUD seeds:

```bash
python3 benchmark_nearest_baseline.py --jsonl
```

Run the HUD taskset against a gateway model:

```bash
hud eval hud_mobility/tasks.py claude --full --max-steps 12 --gateway --auto-respond --yes
```

Train a forked trainable model with HUD GRPO:

```bash
hud models list
hud models fork <trainable-base-model> --name mobility-orchestrator-rl
python3 -m hud_mobility.train --model mobility-orchestrator-rl --steps 5 --group 8
```

For a tiny end-to-end smoke run before spending a full batch:

```bash
python3 -m hud_mobility.train --model mobility-orchestrator-rl --steps 1 --group 2 --limit-tasks 1 --max-concurrent 1
```

Latest measured run:

- nearest-car baseline mean reward: `0.268809`;
- non-greedy specialist planner mean reward: `0.320795`;
- trained HUD model `mobility-orchestrator-rl-codex-01` checkpoint `fff84abe-f839-47f9-9d2a-8304e35963b8`: `0.321 +/- 0.059` over all six tasks;
- confirmation job: `https://hud.ai/jobs/3b9089b03b7e4590aa0d80a9dedf77d6`.

The training objective is an absolute normalized metric blend: profit capture, demand served, wait score, productive utilization, future supply alignment, cancellation penalty, deadhead penalty, and invalid-action penalty. Greedy metrics are intentionally excluded from the reward path.

## Important Files

- `main.jsx`: React/Deck.gl map UI and overlays.
- `map.py`: OSMnx network and base data generator.
- `export_mobility_world.py`: stateful greedy dispatch export for the frontend.
- `mobility_sim/generators.py`: demand, traffic, people, and world generator logic.
- `public/data/`: generated JSON artifacts used by the map.
