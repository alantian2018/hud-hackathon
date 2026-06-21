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
- `Speed`: cycles simulated playback speed through `0.1x`, `0.5x`, and faster demo speeds.
- `Events`: toggles one of three preplanned demand surges: Chase Center exit, Market St surge, or FiDi conference surge. Each event generates more people near that location, which then changes traffic pressure, greedy routes, and stats.
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
- `mobility_sim/generators.py`: demand, traffic, people, and world generator logic.
- `public/data/`: generated JSON artifacts used by the map.
