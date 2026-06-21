# Integration Notes

## Relevant Entry Points

- `main.jsx` is the React/Deck.gl visualization entry point. It creates the root app directly and fetches static JSON from `/data/...`.
- `map.py` generates the OSMnx-backed browser artifacts: `osmnx_edges.geojson`, `ppo_nodes.json`, `ppo_edges.json`, `population_density_grid.json`, `sample_trips.json`, and `pipeline_meta.json`.
- `export_mobility_world.py` builds `mobility_world.json`, a fixed-minute stateful greedy-dispatch snapshot file for the React map.
- `mobility_sim/generators.py` contains the existing Python grid demand, traffic, person, router, and greedy dispatcher logic used by smoke tests and the mobility-world export.
- `public/data/mobility_world.json` is generated output consumed by the browser. `dist/data/*` contains the built static graph/data artifacts used by the current RL adapter in this checkout.

## Frontend Data Flow

`main.jsx` does not use WebSockets or event messages. It loads static JSON with `fetch`:

- `/data/osmnx_edges.geojson`
- `/data/sample_trips.json`
- `/data/population_density_grid.json`
- `/data/ppo_nodes.json`
- `/data/mobility_world.json`

The app maintains a browser clock in minutes over a 24-hour day. It selects the latest `mobility_world.snapshots[]` item whose `timestep` is less than or equal to the browser clock. The generated greedy snapshots are fixed time slices, defaulting to 15-minute spacing.

The frontend animates within a fixed snapshot by computing snapshot progress from:

- `snapshot.timestep`
- browser `clockMinute`
- `mobilityWorld.step_minutes`

It then interpolates assigned cars along a stateful combined route using `route_elapsed`, `pickup_route`, `dropoff_route`, and snapshot elapsed time. It is still a fixed-snapshot playback contract rather than an event-stream contract.

## Frontend Vehicle, Request, Route, and Time Assumptions

Cars in `mobility_world.json` are represented as objects under `snapshot.map_dispatch.cars`:

- `id`, usually `"car-<idx>"`
- `node_id`
- `position` as `[lon, lat]`
- `grid_cell` as `[row, col]`
- `status`, currently `"idle"` or `"to_pickup"` in generated snapshots
- `assigned_person_id`
- `stall_ticks`

Assignments are under `snapshot.map_dispatch.assignments`:

- `car_id`
- `person_id`
- `pickup_node_id`
- `dropoff_node_id`
- `pickup_position` and `dropoff_position` as `[lon, lat]`
- `pickup_route` and `dropoff_route`
- `total_cost`

Routes are dictionaries with:

- `node_path`, a list of OSM node ids
- `coordinates`, a polyline as `[lon, lat]`
- `cost`
- `fallback`

Requests/people are under `snapshot.map_people`:

- `id`
- `origin` and `destination` grid cells
- `created_at`, `patience`, `value`, `party_size`
- `pickup_node_id`, `dropoff_node_id`
- `pickup_position`, `dropoff_position`
- `pickup_grid_cell`, `dropoff_grid_cell`
- `request_origin`, `request_destination`

The visualizer also supports older grid-only fallback shapes through `snapshot.dispatch`, `snapshot.people_grid`, and route `path` arrays of grid cells. The current graph-backed path is coordinate-based.

## Existing Graph Representation

`map.py` loads an OSMnx `networkx.MultiDiGraph` using `ox.graph_from_place(..., network_type="drive")`, then enriches it with OSMnx speeds/travel times and synthetic hourly traffic profiles.

Request generation visible in the JS app is produced offline by `export_mobility_world.py`: each snapshot builds a `DemandGenerator` heatmap, a `TrafficGenerator` heatmap, and `PeopleGenerator` Poisson people using `base_arrival_rate`, demand mean/max, traffic mean/max, weighted pickup cells, and distance-biased dropoff cells. The JAX environment now uses a continuous-time equivalent of that rate profile rather than copying the 15-minute batch spawning.

`ppo_nodes.json` stores stable OSM node ids directly:

- `node_id`
- `lon`, `lat`
- `grid_row`, `grid_col`
- population-density features

`ppo_edges.json` stores directed graph edges:

- `edge_id`, formatted as `"{u}-{v}-{key}"`
- `u`, `v`, `key`
- `features.length_m`
- `features.base_speed_kph`
- `features.highway`
- source/destination density features
- `dynamic_weights_travel_time_s`, a 24-hour travel-time profile
- `dynamic_volume_index`

`osmnx_edges.geojson` stores the same directed edge keys in feature properties, plus geometry coordinates and hourly speed/congestion profiles for display.

Important distinction: `map.py` exports directed edges from OSMnx. `export_mobility_world.py::MapGraph` adds reverse adjacency edges for the old greedy browser demo so grid requests stay connected. A JAX RL loader should not copy that behavior unless explicitly configured.

## Existing Routing Utilities

- `map.py::sample_trips` uses NetworkX shortest paths with an hourly travel-time weight, outside the training path.
- `export_mobility_world.py::MapGraph.route` uses a Python `heapq` Dijkstra over its export adjacency, outside the training path.
- `mobility_sim/generators.py::GridRouter` is a Python grid Dijkstra for the earlier synthetic grid simulator.

None of these are JAX-native. The RL simulator should preprocess static routing tables in Python, then use table lookups in `jit`/`vmap` code.

## Coordinate and Raster Assumptions

Frontend geographic coordinates are `[lon, lat]`. Grid cells are `[row, col]`. The population grid stores bounds and resolution in lon/lat.

For the JAX backend, the core graph should store normalized coordinates for observations plus original lon/lat for export. Raster observations can project node coordinates to a fixed grid using graph bounds. Future JS integration can consume exported lon/lat and route node ids/edge ids without needing the training raster projection.

## Future Integration Path

Do not wire the RL backend into `main.jsx` in this milestone. The safest path is:

1. Keep the JAX simulator headless and event-driven.
2. Export non-jitted scene dictionaries from simulator state using `jax.device_get`.
3. Include `sim_time_seconds`, cars, active/queued requests, congestion, and recent events.
4. Add a later converter that samples event-driven scenes into either:
   - the current fixed-snapshot `mobility_world.json` shape, or
   - a new event-stream/WebSocket shape once the frontend is changed.
5. Preserve OSM node ids and `u-v-key` edge ids in export metadata so Deck.gl can map scenes back to existing graph features.

## Risks and Open Edges

- The exported SF graph has about 10k nodes and 23k directed edges. The actual training loader keeps the largest directed strongly connected component, currently 8,783 nodes and 22,110 edges, so sampled pickup/dropoff routing remains reachable. Dense all-pairs next-hop/travel-time tables are avoided for this graph; the training path uses compact landmark routing and keeps dense exact routing only for small synthetic/debug graphs.
- The current browser animation interpolates only within fixed-minute snapshots and mostly along dropoff routes. It is not a faithful event-driven playback contract yet.
- The existing greedy export adds reverse edges, which can hide directed-routing issues. RL tests should explicitly cover directed asymmetry.
- Browser data files are generated artifacts. New backend tests should not depend on large checked-in JSON unless they are adapter-specific and optional.
- Continuous-time request spawning must be implemented with JAX random keys and bounded loops; overflow flags are needed when many events occur between policy decisions.
