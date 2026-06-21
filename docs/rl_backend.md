# RL Backend

`jax_fleet` is a headless Python package for directed fleet repositioning on the
existing map artifacts. It is separate from the React/Deck.gl frontend.

## Graphs

- `build_synthetic_graph(...)` creates small directed graphs for tests and PPO
  smoke runs.
- `load_public_data_graph("public/data")` loads the canonical San Francisco
  artifacts and keeps the largest directed strongly connected component by
  default.
- The SF graph compacts original OSM node ids into dense ids for array indexing.
  Original ids are retained for scene export.
- Outgoing edges are padded to `graph.max_degree`. Actions are
  `Discrete(max_degree)` slots. There is no stay action.
- Full routing tables are generated outside JIT in chunks and cached under
  `cache/jax_fleet/`, which is already ignored by git. Each cache has a JSON
  manifest with graph shape and fingerprint metadata; stale or mismatched caches
  are rebuilt.

## Environment

`reset(rng, params)` and `step(state, action, params)` are JAX PyTree APIs.

- `step` consumes one policy action for `state.current_car_id`.
- If multiple cars need actions at the same simulation time, they are exposed in
  ascending car id order with `dt_seconds = 0`.
- Once a car is assigned to a request, pickup and dropoff routing proceed
  automatically. The policy is not queried during `TO_PICKUP` or `TO_DROPOFF`.
- Travel time is sampled from the current hourly profile when a car enters an
  edge and is not mutated mid-edge.
- Requests use fixed-size arrays. Preplanned requests are useful for tests.
  Uniform random demand can still use exponential interarrival times, but SF
  defaults to density top-up demand.
- Density top-up demand keeps `target_active_requests` passengers in play. In
  SF mode that target defaults to half the car count, capped by
  `max_requests`. Each replacement request samples origin and destination nodes
  from a time-varying density table derived from node population density and the
  frontend's hourly population profile. This is the fixed-count form of a
  spatial Poisson process: counts are clamped to the active-passenger target
  while locations follow the density heatmap.
- Queued requests are assigned deterministically to the eligible car with the
  lowest directed ETA, but only after that car's directed route to the request
  origin is within `assignment_max_route_edges` hops. The default is 15 route
  edges.
- Invalid action slots fall back to the first valid outgoing edge and increment
  `metrics.invalid_actions`.
- Default reward is sparse pickup wait penalty:
  `-wait_time_scale * (pickup_time - spawn_time)`, where
  `wait_time_scale = 1/60`.
- Per-transition discount is
  `gamma ** (dt_seconds / discount_time_unit_seconds)`.

## Observations

Each timestep includes:

- `raster`: global `50x50x5` map with car counts, queued request origins,
  congestion, request-origin density likelihood, and a one-hot focus-car block.
- `local_raster`: centered `50x50x5` intersection-level view around the
  decision car with the same channels projected from lon/lat coordinates.
- `structured`: compact scalar environment features.
- `candidate_edges`: per-action edge features.
- `action_mask`: valid outgoing-edge slots for the current car.

## Scene Export And Debugging

`export_scene(state, timestep, params)` returns JSON-compatible cars, requests,
congestion, recent events, edge progress, and route previews. `debug_viz.py`
builds a Matplotlib figure from that same schema, so visualization stays outside
JIT and does not affect training code.

## PPO

`jax_fleet.ppo` contains a CleanRL-style Flax actor-critic and smoke trainer.
The model combines global and local CNN raster encoders, a structured-feature
MLP, and a candidate-edge encoder, then masks logits over `graph.max_degree`.

The trainer uses vectorized envs, variable-time GAE with per-transition
discounts, shuffled minibatches over multiple update epochs, clipped policy
loss, clipped value loss, entropy bonus, gradient clipping, JSONL metric
logging, and Orbax checkpoints. It is intentionally close to CleanRL's JAX PPO
loop, adapted to the event-driven fleet environment. It defaults to synthetic
graphs; full SF training loads the public-data graph with routing enabled and
uses the cached dense routing tables.

## Commands

Prepare or validate the full SF routing cache:

```bash
python3 -m jax_fleet.cli prepare-routing \
  --data-dir public/data \
  --cache-dir cache/jax_fleet \
  --chunk-size 512
```

Run a checkpointed synthetic smoke train:

```bash
python3 -m jax_fleet.cli train \
  --graph synthetic \
  --num-envs 4 \
  --num-steps 16 \
  --num-updates 2 \
  --update-epochs 4 \
  --num-minibatches 4 \
  --checkpoint-dir runs/jax_fleet/synthetic/checkpoints \
  --metrics-path runs/jax_fleet/synthetic/metrics.jsonl
```

Run full SF training after the routing cache exists:

```bash
python3 -m jax_fleet.cli train \
  --graph sf \
  --data-dir public/data \
  --routing-cache-dir cache/jax_fleet \
  --num-envs 8 \
  --num-steps 64 \
  --num-updates 1000 \
  --update-epochs 4 \
  --num-minibatches 4 \
  --max-cars 16 \
  --max-requests 256 \
  --assignment-max-route-edges 15 \
  --spawn-source density \
  --checkpoint-dir runs/jax_fleet/sf/checkpoints \
  --metrics-path runs/jax_fleet/sf/metrics.jsonl
```

Resume a run:

```bash
python3 -m jax_fleet.cli train \
  --graph sf \
  --resume \
  --num-updates 1500 \
  --checkpoint-dir runs/jax_fleet/sf/checkpoints \
  --metrics-path runs/jax_fleet/sf/metrics.jsonl
```

## Verification

Run:

```bash
python3 -m pytest tests/test_jax_fleet_graph.py tests/test_jax_fleet_env.py tests/test_jax_fleet_ppo_debug.py -q
python3 -m pytest tests/test_jax_fleet_training_stability.py -q
python3 -m unittest -v
```
