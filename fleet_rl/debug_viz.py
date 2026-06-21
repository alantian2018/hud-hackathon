from __future__ import annotations

import argparse

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from .env import EnvParams, FleetEnv
from .export import export_scene
from .graph import build_synthetic_debug_graph, load_ppo_json_graph


STATUS_COLORS = {
    "POLICY_CONTROLLED": "#64748b",
    "TO_PICKUP": "#2563eb",
    "TO_DROPOFF": "#16a34a",
}


def _route_nodes(graph, start: int, target: int, max_hops: int = 128) -> list[int]:
    nodes = [int(start)]
    current = int(start)
    for _ in range(max_hops):
        if current == int(target):
            break
        edge = _route_edge(graph, current, int(target))
        if edge < 0:
            break
        next_node = int(graph.edge_to[edge])
        if next_node == current:
            break
        nodes.append(next_node)
        current = next_node
    return nodes


def _route_edge(graph, source: int, target: int) -> int:
    if graph.route_mode == "dense":
        return int(graph.next_edge_table[source, target])

    best_idx = 0
    best_cost = float("inf")
    for idx in range(int(graph.num_landmarks)):
        cost = float(graph.node_to_landmark_time[source, idx] + graph.landmark_to_node_time[idx, target])
        if cost < best_cost:
            best_cost = cost
            best_idx = idx

    landmark = int(graph.landmark_nodes[best_idx])
    edge = int(graph.node_to_landmark_next_edge[source, best_idx]) if source != landmark else int(graph.landmark_to_node_next_edge[best_idx, target])
    if edge >= 0 and int(graph.edge_from[edge]) == source:
        return edge

    # Landmark tables can be approximate for compact routing; use the same one-step greedy fallback
    # as the JAX env for debug drawing.
    best_edge = -1
    best_score = float("inf")
    for edge_id, valid in zip(graph.out_edges[source].tolist(), graph.out_edge_mask[source].tolist()):
        if not valid:
            continue
        next_node = int(graph.edge_to[int(edge_id)])
        estimate = float(graph.node_to_landmark_time[next_node, best_idx] + graph.landmark_to_node_time[best_idx, target])
        score = float(graph.edge_base_travel_time_s[int(edge_id)]) + estimate
        if score < best_score:
            best_score = score
            best_edge = int(edge_id)
    return best_edge


def plot_scene(scene: dict, graph, ax=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 8))
    ax.clear()
    ax.set_title(f"Fleet RL debug | t={scene['sim_time_seconds']:.1f}s | decision car={scene['current_car_id']}")
    ax.set_aspect("equal", adjustable="box")

    for edge in scene["edges"]:
        u = edge["from_node"]
        v = edge["to_node"]
        x0, y0 = float(graph.node_lon[u]), float(graph.node_lat[u])
        x1, y1 = float(graph.node_lon[v]), float(graph.node_lat[v])
        congestion = edge["congestion"]
        color = plt.cm.inferno(min(1.0, max(0.0, (congestion - 1.0) / 3.0)))
        ax.annotate(
            "",
            xy=(x1, y1),
            xytext=(x0, y0),
            arrowprops={"arrowstyle": "->", "color": color, "lw": 1.0, "alpha": 0.55},
        )

    ax.scatter(graph.node_lon[: int(graph.num_nodes)], graph.node_lat[: int(graph.num_nodes)], s=16, c="#111827", zorder=2)
    for node_id in range(int(graph.num_nodes)):
        ax.text(float(graph.node_lon[node_id]), float(graph.node_lat[node_id]), str(node_id), fontsize=7, color="#374151")

    for req in scene["active_requests"] + scene["queued_requests"]:
        pickup = req["pickup"]
        dropoff = req["dropoff"]
        ax.scatter(pickup["lon"], pickup["lat"], marker="^", s=90, c="#3b82f6", edgecolors="white", zorder=4)
        ax.scatter(dropoff["lon"], dropoff["lat"], marker="v", s=90, c="#ef4444", edgecolors="white", zorder=4)
        ax.plot([pickup["lon"], dropoff["lon"]], [pickup["lat"], dropoff["lat"]], color="#94a3b8", ls=":", lw=1)

    for car in scene["cars"]:
        if car["status"] in {"TO_PICKUP", "TO_DROPOFF"}:
            start = car["to_node"] if car["edge_id"] is not None and car["edge_id"] >= 0 else car["current_node"]
            route = _route_nodes(graph, start, car["target_node"])
            if len(route) >= 2:
                xs = [float(graph.node_lon[n]) for n in route]
                ys = [float(graph.node_lat[n]) for n in route]
                ax.plot(xs, ys, color=STATUS_COLORS[car["status"]], lw=2.3, alpha=0.55, zorder=3)

    for car in scene["cars"]:
        color = STATUS_COLORS.get(car["status"], "#111827")
        size = 170 if car["id"] == scene["current_car_id"] else 110
        ax.scatter(car["lon"], car["lat"], s=size, c=color, edgecolors="white", linewidths=1.5, zorder=5)
        ax.text(car["lon"], car["lat"], str(car["id"]), color="white", ha="center", va="center", fontsize=8, zorder=6)

    ax.text(
        0.01,
        0.01,
        (
            f"queued={scene['metrics']['queue_length']} "
            f"assigned={scene['metrics']['requests_assigned']} "
            f"picked={scene['metrics']['requests_picked_up']} "
            f"done={scene['metrics']['requests_completed']}"
        ),
        transform=ax.transAxes,
        fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.78, "edgecolor": "#cbd5e1"},
    )
    ax.grid(True, alpha=0.18)
    return ax


def run_manual_debugger(args) -> None:
    if args.nodes and args.edges:
        graph = load_ppo_json_graph(
            args.nodes,
            args.edges,
            node_limit=None if args.node_limit <= 0 else args.node_limit,
            route_mode="landmark",
            num_landmarks=args.route_landmarks,
        )
    else:
        graph = build_synthetic_debug_graph(args.graph)
    env = FleetEnv(graph)
    params = EnvParams.for_graph(
        graph,
        num_cars=args.num_cars,
        demand_rate_per_second=args.demand_rate,
        max_active_requests=32,
        randomize_start_time=False,
    )
    key = jax.random.PRNGKey(args.seed)
    state, timestep = env.reset(key, params)

    plt.ion()
    fig, ax = plt.subplots(figsize=(8, 8))
    action_key = jax.random.PRNGKey(args.seed + 1)
    for step in range(args.steps):
        scene = export_scene(state, timestep, params)
        plot_scene(scene, graph, ax=ax)
        fig.canvas.draw()
        fig.canvas.flush_events()
        if args.pause:
            input(f"step {step}, t={scene['sim_time_seconds']:.1f}s, car={scene['current_car_id']} | Enter for next")
        else:
            plt.pause(args.interval)
        action_key, subkey = jax.random.split(action_key)
        logits = jnp.where(timestep.action_mask, 0.0, -1e9)
        action = jax.random.categorical(subkey, logits).astype(jnp.int32)
        state, timestep = env.step(state, action, params)
    plt.ioff()
    plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description="Standalone Python 2D debugger for the JAX fleet environment.")
    parser.add_argument("--graph", default="grid3", choices=["line", "asymmetric", "directed_assignment", "variable_degree", "grid3"])
    parser.add_argument("--nodes", default=None, help="Optional ppo_nodes.json path for an SF subgraph.")
    parser.add_argument("--edges", default=None, help="Optional ppo_edges.json path for an SF subgraph.")
    parser.add_argument("--node-limit", default=128, type=int, help="Limit actual graph nodes for visualization; 0 means full graph.")
    parser.add_argument("--route-landmarks", default=16, type=int)
    parser.add_argument("--num-cars", default=4, type=int)
    parser.add_argument("--demand-rate", default=1.0 / 450.0, type=float)
    parser.add_argument("--steps", default=40, type=int)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--interval", default=0.35, type=float)
    parser.add_argument("--pause", action="store_true", help="Wait for Enter between steps.")
    run_manual_debugger(parser.parse_args())


if __name__ == "__main__":
    main()
