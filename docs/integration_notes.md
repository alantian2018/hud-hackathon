# Integration Notes

These notes capture the current repository boundary for the JAX fleet backend.
The first backend implementation is headless Python; it does not replace the
React/Deck.gl demo or add WebSockets.

## Current Frontend Data Flow

- The frontend entrypoint is `main.jsx`.
- It fetches static files from `/data/*.json`, including
  `osmnx_edges.geojson`, `sample_trips.json`, `population_density_grid.json`,
  `ppo_nodes.json`, and optionally `mobility_world.json`.
- Roads are rendered from `GeoJsonLayer` features in
  `public/data/osmnx_edges.geojson`. Feature properties include `u`, `v`,
  `key`, `length_m`, `base_speed_kph`, `free_flow_time_s`,
  `hourly_speed_kph`, `hourly_travel_time_s`, and `hourly_congestion`.
- Trips are rendered from `sample_trips.json` as `{path, timestamps}` records,
  where `path` is an array of `[lon, lat]` coordinates and `timestamps` are
  minute values consumed by `TripsLayer`.
- The optional greedy simulator export is `public/data/mobility_world.json`.
  It contains fixed minute snapshots with `map_dispatch.cars`, `map_people`,
  assignment route coordinates, and summary stats.
- The JAX live/training backend now treats those fixed `new_people` snapshots
  as a legacy `js-visual` source. The default SF passenger source is density
  top-up: cars still seed from the visual node set, while requests are sampled
  continuously from time-varying node density.

## Existing Python Boundary

- `mobility_sim/generators.py` is a mutable Python grid/tick simulator used to
  create demo demand, traffic, people, and greedy dispatch snapshots.
- `export_mobility_world.py` adapts the map artifacts into grid-simulator
  snapshots and mirrors directed edges in its `MapGraph` to keep the visual demo
  connected.
- `map.py` is the OSMnx export pipeline. OSMnx is optional for this backend
  because it is not installed locally; the checked-in `public/data` artifacts
  are treated as canonical runtime inputs.

## Canonical Graph Artifacts

- `public/data/ppo_nodes.json` has 10,022 nodes.
- `public/data/ppo_edges.json` has 23,505 directed edge records, including
  parallel records.
- The simple directed graph formed from unique `(u, v)` pairs has 23,394 edges.
- The largest strongly connected component has 8,783 nodes and 22,016 edges,
  with maximum out-degree 6. The JAX backend uses this SCC by default so every
  controllable route has a directed path inside the training graph.

## Backend Integration Scope

- `jax_fleet` loads existing map artifacts directly and compacts node ids for
  array-based JAX execution.
- Routing tables are generated outside JIT and cached under `cache/jax_fleet/`.
- The environment is continuous-time and event-based. A single `step` consumes
  exactly one repositioning action for the current decision car, then advances
  to the next decision, truncation, or overflow.
- Scene export is JSON-compatible and intentionally non-jitted. It is suitable
  for Matplotlib debug visualization and future frontend bridges.
