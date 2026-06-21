# FleetForge

<p align="center">
  <img src="assets/fleetforge-hero.png" alt="FleetForge - training AI agents to operate real-world fleets" />
</p>

**Training AI agents to operate real-world fleets.**

FleetForge is an RL environment for training, evaluating, and benchmarking AI fleet orchestrators. It compares a reactive greedy dispatcher against an Agentic RL Fleet on the same San Francisco road network, vehicle supply, demand stream, and event stress tests.

The core question is simple: can an agentic fleet controller make better city-scale decisions than a nearest-car style greedy policy, while improving completed trips, profit, demand served, and passenger wait time?

## Why FleetForge

Fleet operations are inherently complex and constantly changing:

- demand fluctuates by the minute;
- traffic and events create uncertainty;
- revenue, coverage, utilization, and wait time compete with each other;
- local decisions can hurt global outcomes through overconcentration, long waits, and missed demand.

The next wave of mobility, delivery, logistics, robotics, and autonomous operations needs a safe way to train and evaluate decision-making before deployment. FleetForge provides that evaluation and training layer: agents observe real-world signals, take high-level actions, receive verified rewards, and get benchmarked across diverse scenarios.

## What Judges Should Look For

- **Same world, two policies:** Greedy and Agentic Fleet run side by side on identical map data, traffic, vehicles, requests, and events.
- **Business metrics:** the comparison focuses on completed trips, profit, demand served, and average wait time.
- **Agentic behavior:** the HUD tool trace shows the observation, hotspot forecasting, matching, repositioning, critique, action, and submission loop behind the Agentic Fleet.
- **Stress tests:** Base demand, Chase Center Exit, Market St Surge, and FiDi Conference scenarios expose where reactive dispatch breaks down under spatial demand shocks.
- **HUD training loop:** the fleet controller is framed as a verifiable HUD environment with an absolute reward, MCP tools, taskset seeds, rollout evaluation, and RL training through HUD.

FleetForge turns AI decisions into measurable impact across the physical economy: measurable ROI, safer scenario testing, extensible tools and constraints, and a path beyond cars into delivery, logistics, robots, and more.

## Live Demo

```bash
npm install
npm run dev
```

Open the local URL printed by Vite. The app is configured around:

- `/` -> FleetForge comparison home
- `/compare.html` -> synchronized Greedy vs Agentic Fleet comparison
- `/greedy.html` -> Greedy policy view
- `/rl.html` -> Agentic Fleet policy view

Optional basemap configuration:

```bash
cp .env.example .env.local
# set VITE_MAPTILER_KEY if using your own MapTiler project key
```

<p align="center">
  <img src="assets/fleetforge.gif" alt="FleetForge live comparison demo" />
</p>

## Demo Flow

1. Open `/compare.html`.
2. The page loads both policies on the same map and initializes all metrics at zero.
3. The synchronized run starts automatically after the interface is ready, or can be started manually with **Start Both**.
4. Select one of the event scenarios to compare how each policy responds to localized demand pressure.
5. At the end of the hour, FleetForge freezes the final frame and summarizes the business impact of Agentic Fleet over Greedy.

## Greedy Policy

The Greedy policy is the baseline fleet dispatcher. It assigns available cars to currently visible requests with a local, immediate objective: serve the nearest or cheapest feasible pickup now.

This works in stable demand, but it has predictable failure modes:

- cars overconcentrate around the most obvious demand spike;
- vehicles chase current requests without preserving future coverage;
- nearby pickup corridors become undersupplied;
- wait time rises when traffic and demand peak in the same area.

Greedy is useful because it is intuitive, fast, and hard to beat without a better global policy. It is also exactly the kind of reactive baseline an RL fleet controller should outperform.

## Agentic RL Fleet

Agentic Fleet is the learned orchestration approach. Instead of only matching the closest car to the current request queue, it reasons over:

- current fleet state and active rides;
- pending and forecast demand;
- road traffic and bottlenecks;
- request value and urgency;
- future supply alignment;
- proactive repositioning opportunities.

The Agentic Fleet can assign trips, hold vehicles, and reposition idle cars before the obvious greedy action becomes expensive. In the live comparison, trip routes remain blue for both policies, while green routes show Agentic Fleet repositioning.

## Product Loop

FleetForge follows a train, evaluate, deploy workflow:

- **Observe real-world signals:** demand, traffic, vehicles, events, requests, and business metrics.
- **Take high-level actions:** assign trips, reposition vehicles, hold supply, or reroute.
- **Receive verified rewards:** score decisions against simulator outcomes and business objectives.
- **Train, test, and benchmark:** compare agents across baseline and event-driven scenarios.

## HUD Environment

The HUD environment lives in `hud_mobility/`. It exposes a verifiable fleet-control task where an LLM orchestrator controls the simulator through MCP tools and receives an absolute normalized reward.

The architecture combines an LLM orchestrator, specialist planning tools, and a simulator verifier. World inputs flow into the agent, the agent calls matching, routing, repositioning, pricing, or charging tools, and the simulator executes actions, advances the world, computes metrics, and returns reward feedback.

The agent tool surface includes:

- `observe_state`: inspect fleet, requests, traffic, demand, and running metrics.
- `forecast_hotspots`: identify high-demand cells over the next rollout window.
- `propose_matching`: generate value- and urgency-aware assignment candidates.
- `propose_repositioning`: generate proactive idle-car reposition targets.
- `propose_full_plan`: assemble a complete candidate action plan.
- `critique_action_plan`: validate the plan and surface concentration, coverage, and validity risks.
- `step_world`: apply one structured action plan and advance the simulator.
- `submit_episode`: return the final reward and metrics for the episode.

The reward is absolute, not a greedy-relative score. It combines profit capture, demand served, lower wait time, productive fleet utilization, future supply alignment, cancellation avoidance, deadhead control, and action validity.

## HUD Training

The HUD taskset evaluates the orchestrator across baseline and event-driven mobility scenarios:

- morning and evening base demand;
- Chase Center exit wave;
- Market St downtown surge;
- FiDi conference exit wave.

The training loop uses HUD rollouts to collect trajectories, score them with the simulator reward, and update a trainable model through GRPO-style RL. The model is trained to use the available tools, submit complete episodes, and maximize the environment reward without using Greedy metrics inside the reward path.

Install HUD dependencies:

```bash
python3 -m pip install -r requirements-hud.txt
hud set HUD_API_KEY=...
```

Run local environment checks:

```bash
python3 -B -m unittest test_generators.py
python3 -B -m unittest test_hud_mobility.py
python3 -m hud_mobility.eval_local --episodes 8 --horizon-steps 8 --fleet-size 20
```

Evaluate the HUD taskset:

```bash
hud eval hud_mobility/tasks.py claude --full --max-steps 12 --gateway --auto-respond --yes
```

Train a HUD model:

```bash
hud models list
hud models fork <trainable-base-model> --name mobility-orchestrator-rl
python3 -m hud_mobility.train --model mobility-orchestrator-rl --steps 5 --group 8
```

## Current Comparison Results

The current scenario suite shows Agentic Fleet outperforming Greedy on the primary business metrics:

| Scenario | Additional Trips | Profit Lift | Demand Served Lift | Avg Wait Reduction |
| --- | ---: | ---: | ---: | ---: |
| Base | +51 | +$2,076.90 | +15.36 pp | -0.96 min |
| Chase Center Exit | +127 | +$6,903.50 | +12.50 pp | -3.00 min |
| Market St Surge | +185 | +$8,640.02 | +22.48 pp | -7.77 min |
| FiDi Conference | +171 | +$8,165.09 | +20.09 pp | -5.76 min |

HUD reward validation:

- nearest-car baseline mean reward: `0.268809`
- Agentic Fleet planner mean reward: `0.320795`
- trained HUD model `mobility-orchestrator-rl-codex-01`: `0.321 +/- 0.059` over six tasks

## Regenerating Comparison Data

Rebuild the Greedy world:

```bash
python3 export_mobility_world.py
```

Rebuild the Agentic Fleet comparison world:

```bash
python3 precompute_orchestrator_world.py --include-events
```

Run the baseline comparison check:

```bash
python3 benchmark_nearest_baseline.py --jsonl
```

Run the frontend build:

```bash
npm run build
```

## Project Structure

- `comparison.jsx`: FleetForge comparison UI, synchronized controls, map layers, legends, metrics, and HUD tool trace.
- `hud_mobility/`: HUD environment, taskset, world simulator, tools, reward, planners, and training entrypoint.
- `mobility_sim/`: demand, traffic, request, and fleet simulation primitives used by the map exports.
- `export_mobility_world.py`: Greedy policy world export.
- `precompute_orchestrator_world.py`: Agentic Fleet world export.
- `public/data/`: generated scenario data consumed by the frontend.
- `map.py`: San Francisco road/grid data generation.
