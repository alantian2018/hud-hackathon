# RL Backend

`jax_fleet` is a headless Python package for directed fleet repositioning on the
existing map artifacts. It is separate from the React/Deck.gl frontend.

## Graphs

- `build_synthetic_graph(...)` creates small directed graphs for tests and PPO
  smoke runs.
- `load_public_data_graph("dist/data")` loads the canonical San Francisco
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
  origin is within `assignment_max_route_edges` hops. The default is 10000 route
  edges, which effectively lets the directed shortest-path table decide
  reachability rather than imposing a local assignment radius.
- Invalid action slots fall back to the first valid outgoing edge and increment
  `metrics.invalid_actions`.
- Default reward mode is `dense_wait`:
  `-wait_time_scale * waiting_request_seconds - drop_penalty * newly_dropped + pickup_bonus * newly_picked_up`.
  `wait_time_scale` defaults to `1/60`, `drop_penalty` to `10.0`, and
  `pickup_bonus` to `0.0`. Waiting seconds are accumulated across internal
  event intervals as `(queued + assigned-not-yet-picked-up requests) * dt`.
- Legacy sparse pickup reward remains available with
  `reward_mode="legacy_pickup_wait"` or CLI `--reward-mode legacy_pickup_wait`.
  It emits `-wait_time_scale * (pickup_time - spawn_time)` only at pickup.
- Per-transition discount is time-aware: nonterminal transitions use
  `gamma ** (dt_seconds / time_discount_reference_seconds)`, where
  `time_discount_reference_seconds` defaults to `60.0`; terminal transitions
  use `0.0`.

## Observations

Each timestep includes:

- `raster`: global `50x50x8` map. Channels are available cars, queued request
  origins, queued wait-sum in minutes normalized by 30, expected demand next
  10 minutes, future supply 0-5 minutes, future supply 5-15 minutes, active
  request destinations, and focus car.
- `local_raster`: centered `50x50x8` view around the decision car with the same
  channel order.
- `structured`: 11 scalar features: elapsed episode fraction, sin/cos time of
  day, queued fraction, available car fraction, busy car fraction, spawn rate
  normalized by 10/minute, mean/max queued wait normalized by 30 minutes, and
  current car normalized `x/y`. Arbitrary car id is not part of the default
  policy input.
- `candidate_edges`: 22 per-action features. The first eight are target
  `dx/dy`, target `x/y`, edge length km, edge travel time minutes, congestion,
  and validity. The remaining features are queued reachable counts within
  5/10 minutes, wait-weighted reachable demand, max reachable wait, min ETA to
  queue, ETA-improvement aggregates, best-car/marginal-advantage aggregates,
  expected demand near target at 10/30 minutes, future supply at 0-5/5-15
  minutes, and target supply-demand imbalance. Invalid rows are zeroed.
- `action_mask`: valid outgoing-edge slots for the current car.

Set `observation_mode="legacy"` or CLI `--observation-mode legacy` to recover
the old `6/10/12` raster, structured, and candidate-edge shapes.

## Scene Export And Debugging

`export_scene(state, timestep, params)` returns JSON-compatible cars, requests,
congestion, recent events, edge progress, and route previews. `debug_viz.py`
builds a Matplotlib figure from that same schema, so visualization stays outside
JIT and does not affect training code.

## PPO

`jax_fleet.ppo` contains a CleanRL-style Flax actor-critic and smoke trainer.
The model combines global and local CNN raster encoders, a structured-feature
MLP, and a shared candidate-edge scorer. Every edge slot uses the same edge
encoder and action scorer weights, then invalid logits are masked to `-1e9`.

The trainer uses vectorized envs, fixed-discount GAE, shuffled minibatches over
multiple update epochs, clipped policy loss, clipped value loss, entropy bonus,
gradient clipping, CleanRL-style `charts/*`, `losses/*`, `rollout/*`, and
`env/*` metrics, JSONL metric logging, optional W&B tracking, and Orbax
checkpoints. It is intentionally close to CleanRL's JAX PPO loop, adapted to
the event-driven fleet environment. It defaults to full SF training with the
dist-data graph, cached dense routing tables, 40 cars, and 32 request slots.
Synthetic smoke training must opt in with `--graph synthetic`.

## Commands

Prepare or validate the full SF routing cache:

W&B is optional. Install it with `pip install wandb` or
`pip install .[tracking]`, then pass `--track` to training commands.
Periodic W&B videos are disabled by default. Enable them with
`--wandb-video-every N`; each diagnostic rollout uses the current greedy policy
and stops at `--wandb-video-max-steps` or `--wandb-video-max-pickups`, whichever
comes first.
NVIDIA GPU support uses JAX's CUDA 13 wheels. Install it with
`python3 -m pip install -e ".[gpu]"`, then verify CUDA visibility with
`jax-fleet check-gpu --require-gpu`. Pass `--require-gpu` to train or
benchmark commands to fail fast instead of silently falling back to CPU.
The train and benchmark commands default to the full SF graph. Synthetic runs
must opt in with `--graph synthetic`. The default SF demand source maintains
roughly half as many active passengers as cars, capped by `max_requests`.

```bash
python3 -m jax_fleet.cli prepare-routing \
  --data-dir dist/data \
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
  --track \
  --wandb-project-name jax_fleet \
  --checkpoint-dir runs/jax_fleet/synthetic/checkpoints \
  --metrics-path runs/jax_fleet/synthetic/metrics.jsonl
```

Run full SF training after the routing cache exists:

```bash
python3 -m jax_fleet.cli train \
  --graph sf \
  --data-dir dist/data \
  --routing-cache-dir cache/jax_fleet \
  --require-gpu \
  --num-envs 8 \
  --num-steps 64 \
  --num-updates 1000 \
  --update-epochs 4 \
  --num-minibatches 4 \
  --max-cars 40 \
  --max-requests 32 \
  --assignment-max-route-edges 10000 \
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

Render the latest trained policy:

```bash
jax-fleet-live \
  --graph sf \
  --data-dir dist/data \
  --cache-dir cache/jax_fleet \
  --policy checkpoint \
  --policy-checkpoint latest \
  --policy-checkpoint-dir runs/jax_fleet/sf/checkpoints \
  --max-cars 40 \
  --max-requests 32 \
  --spawn-source density \
  --jit
```

## Verification

Run:

```bash
python3 -m pytest tests/test_jax_fleet_graph.py tests/test_jax_fleet_env.py tests/test_jax_fleet_ppo_debug.py -q
python3 -m pytest tests/test_jax_fleet_training_stability.py -q
python3 -m unittest -v
```
