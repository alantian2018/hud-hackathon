# Fleet RL Backend

## Module Layout

- `fleet_rl.graph`: synthetic graph builders, PPO JSON graph adapter, Python route oracle, and precomputed directed routing tables.
- `fleet_rl.routing`: `TableRouter`, the current swappable routing abstraction over `next_hop`, `next_edge`, and `travel_time_estimate`.
- `fleet_rl.env`: JAX-native continuous-time event-driven fleet environment.
- `fleet_rl.export`: non-jitted scene export adapter for debugging and future JS integration.
- `fleet_rl.debug_viz`: standalone matplotlib visualizer that consumes scene exports.
- `fleet_rl.ppo`: Flax/Optax PPO model, rollout collection, variable-time GAE, update, and checkpoint helpers.

## State And Time

`FleetEnv` uses fixed-size `flax.struct.dataclass` PyTrees. The core state stores absolute `sim_time_seconds`, per-car edge entry/exit times, fixed request slots, request statuses, edge congestion, pending decision cars, recent event ring buffers, and cumulative metrics.

This is not a fixed-timestep Gym loop. `env.step(state, action, params)` consumes one action for the currently exposed car, then advances internal continuous-time events until the next policy decision. If several cars need decisions at the same simulated time, they are exposed one at a time in ascending car id order with `dt_seconds == 0`.

## Actions

The action space is `Discrete(max_degree)`. Each action selects one outgoing edge slot from the current node. There is no stay/park action. `timestep.action_mask` marks valid slots; invalid actions are replaced with the first valid edge and counted in `metrics.invalid_actions`.

## Routing

The training path does not call NetworkX, OSMnx, `heapq`, JSON, or Python graph code. `fleet_rl.graph` precomputes directed all-pairs baseline travel-time, next-hop, and next-edge tables outside jit. The env uses those tables for assignment ETA and automatic pickup/dropoff routing. `TableRouter` documents the swappable interface for future dynamic JAX routing.

## Traffic

Traffic is edge-based. `graph.edge_traffic_profile[edge_id, hour]` and `state.edge_congestion[edge_id]` determine the multiplier when a car enters an edge. The resulting `car_edge_end_time` is fixed for that edge traversal; the env does not mutate remaining travel time mid-edge. Graph preprocessing also stores hourly traffic pressure summaries so request-rate evaluation does not scan every edge during spawn sampling.

## Requests And Assignment

Requests use fixed slots with status masks. Spawns are continuous-time events sampled by integrating a nonhomogeneous Poisson hazard across fixed time bins. The default demand profile mirrors `mobility_sim.generators.DEFAULT_TIME_OF_DAY_PROFILE`, and the rate uses the same broad factors as the JS-visible offline exporter: demand mean/max and traffic mean/max pressure. Pickup nodes are sampled from graph demand weights; dropoff nodes are conditioned on pickup with demand and distance bias.

If no eligible car exists, the request remains queued. When a car completes a dropoff or reaches an empty-policy node, queued requests are assigned deterministically before exposing another policy decision. Full request arrays cause explicit drops and metric increments.

## Observation

Each timestep observes only the current decision car:

- Raster tensor `(50, 50, 9)` by default: policy car counts, to-pickup counts, to-dropoff counts, decision car location, active pickup targets, active dropoff targets, queued pickups, demand probability, and edge congestion projection.
- Global features: time of day, current car id, current node coordinates/degree, queue length, fleet utilization, local supply/demand, and simulation time.
- Per-action candidate features for padded outgoing edges: valid flag, next-node id/coordinates, edge travel time, length, congestion, nearby demand, and nearby available supply.

## Reward And Discount

The default reward is sparse wait-time penalty at pickup:

`reward -= wait_time_scale * (pickup_time_seconds - spawn_time_seconds)`

No dense shaping is enabled by default. Pickup wait metrics include running average plus p50/p90/p95 over the fixed wait sample buffer. Each timestep returns a variable-time discount:

`discount = gamma ** (dt_seconds / discount_time_unit_seconds)`

PPO GAE uses this per-transition discount.

## PPO

`fleet_rl.ppo` uses a feed-forward Flax actor-critic:

- CNN raster encoder
- MLP global-feature encoder
- per-action edge encoder
- masked categorical policy over `max_degree`
- scalar value head

Rollout collection uses `jax.vmap` over independent envs and `jax.lax.scan` over rollout length. The PPO update includes clipped policy loss, value loss, entropy bonus, advantage normalization, gradient clipping, linear learning-rate schedule, and checkpoint helpers.

Run a small actual-graph trainer job:

```bash
python -m fleet_rl.train_ppo --num-cars 16 --num-envs 8 --num-steps 64 --updates 10
```

Run a synthetic debug job explicitly:

```bash
python -m fleet_rl.train_ppo --graph grid3 --num-cars 4 --num-envs 2 --num-steps 32 --updates 10
```

The actual-graph path uses the largest directed strongly connected component from the exported OSMnx graph, then compact landmark routing instead of dense all-pairs routing. In the checked-in SF data this is 8,783 nodes and 22,110 directed edges. `--route-landmarks 64` is the default; increase it for better route estimates at the cost of preprocessing time and memory. Use `--node-limit N` only for debugging.

## Python Debug Visualization

Run a synthetic debug graph:

```bash
python -m fleet_rl.debug_viz --graph grid3 --num-cars 4 --steps 40 --pause
```

Run a small SF subgraph:

```bash
python -m fleet_rl.debug_viz --nodes dist/data/ppo_nodes.json --edges dist/data/ppo_edges.json --node-limit 256 --steps 40
```

The visualizer is outside jit and outside the training path. It consumes the same scene export intended for future JS integration.

## Future JS Integration

Use `fleet_rl.export.export_scene` as the boundary. It returns JSON-compatible dictionaries with `sim_time_seconds`, interpolated moving car positions, active/queued requests, edge congestion, and recent named events. A later frontend milestone can convert these event scenes into the current `mobility_world.json` snapshot style or a new event/WebSocket stream.
