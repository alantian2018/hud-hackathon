from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from PIL import Image

from jax_fleet.env import make_env_params, reset, step
from jax_fleet.graph import build_synthetic_graph, load_public_data_graph
from jax_fleet.heuristics import choose_marginal_value_action
from jax_fleet.scene_export import export_scene


def make_debug_figure(scene: dict[str, Any]):
    fig, ax = plt.subplots(figsize=(8, 8))
    draw_scene(ax, scene)
    fig.tight_layout()
    return fig, ax


def draw_scene(ax, scene: dict[str, Any]) -> None:
    ax.clear()
    congestion = scene.get("congestion", [])
    ax.set_title(f"JAX Fleet t={scene.get('time_seconds', 0):.1f}s")
    _draw_edges(ax, congestion)

    for request in scene.get("requests", []):
        origin = request.get("origin")
        destination = request.get("destination")
        if origin:
            ax.scatter(origin[0], origin[1], marker="o", s=55, c="#f59e0b", zorder=4)
        if destination:
            ax.scatter(destination[0], destination[1], marker="x", s=55, c="#7c3aed", zorder=4)

    for car in scene.get("cars", []):
        lon, lat = car["position"]
        color = "#1d4ed8" if car["status"] == "decision" else "#16a34a"
        ax.scatter(lon, lat, marker="s", s=70, c=color, edgecolors="white", linewidths=0.5, zorder=5)
        ax.text(lon, lat, str(car["id"]), fontsize=8, ha="center", va="center", color="white", zorder=6)

    if congestion:
        ax.text(
            0.01,
            0.01,
            f"{len(congestion)} edges, {len(scene.get('requests', []))} requests",
            transform=ax.transAxes,
            fontsize=8,
        )
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    ax.grid(True, alpha=0.25)


def show_scene(scene: dict[str, Any]) -> None:
    fig, _ = make_debug_figure(scene)
    fig.show()


def run_debug_demo(
    *,
    graph,
    out_path: str | Path | None = None,
    steps: int = 20,
    seed: int = 0,
    max_cars: int = 8,
    max_requests: int = 64,
    spawn_rate_per_minute: float = 1.0,
    episode_seconds: float = 1800.0,
    policy: str = "first",
    show: bool = False,
    pause_seconds: float = 0.15,
    hold_seconds: float = 0.0,
    fps: int = 4,
) -> list[dict[str, Any]]:
    initial_nodes = _initial_car_nodes(graph.num_nodes, max_cars, seed)
    params = make_env_params(
        graph,
        max_cars=max_cars,
        max_requests=max_requests,
        initial_car_nodes=initial_nodes,
        spawn_rate_per_minute=spawn_rate_per_minute,
        episode_seconds=episode_seconds,
    )
    state, timestep = reset(jax.random.PRNGKey(seed), params)
    scenes = [export_scene(state, timestep, params)]
    action_rng = np.random.default_rng(seed + 404)
    live_fig = live_ax = None
    if show:
        plt.ion()
        live_fig, live_ax = plt.subplots(figsize=(8, 8))
        draw_scene(live_ax, scenes[-1])
        live_fig.tight_layout()
        live_fig.canvas.draw_idle()
        plt.pause(pause_seconds)

    for _ in range(steps):
        action = _choose_action(timestep.observation, policy, action_rng)
        state, timestep = step(state, jnp.asarray(action, dtype=jnp.int32), params)
        scenes.append(export_scene(state, timestep, params))
        if show and live_ax is not None and live_fig is not None:
            draw_scene(live_ax, scenes[-1])
            live_fig.tight_layout()
            live_fig.canvas.draw_idle()
            plt.pause(pause_seconds)
        if bool(np.asarray(timestep.done)):
            break

    if out_path is not None:
        save_demo(scenes, out_path, fps=fps)
    if show and hold_seconds > 0:
        plt.pause(hold_seconds)
    return scenes


def save_demo(scenes: list[dict[str, Any]], out_path: str | Path, *, fps: int = 4) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.suffix.lower() == ".gif":
        frames = [_render_scene_image(scene) for scene in scenes]
        duration_ms = int(1000 / max(1, fps))
        frames[0].save(
            out_path,
            save_all=True,
            append_images=frames[1:],
            duration=duration_ms,
            loop=0,
            optimize=True,
        )
    else:
        fig, _ = make_debug_figure(scenes[-1])
        fig.savefig(out_path, dpi=160)
        plt.close(fig)
    return out_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the Python 2D JAX fleet debug visualizer.")
    parser.add_argument("--graph", choices=["synthetic", "sf"], default="sf")
    parser.add_argument("--data-dir", default="public/data")
    parser.add_argument("--cache-dir", default="cache/jax_fleet")
    parser.add_argument("--out", default="runs/jax_fleet/debug_viz/sf_demo.gif")
    parser.add_argument("--steps", default=20, type=int)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--max-cars", default=8, type=int)
    parser.add_argument("--max-requests", default=64, type=int)
    parser.add_argument("--spawn-rate-per-minute", default=1.0, type=float)
    parser.add_argument("--episode-seconds", default=1800.0, type=float)
    parser.add_argument("--policy", choices=["first", "random", "heuristic"], default="first")
    parser.add_argument("--fps", default=4, type=int)
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--pause-seconds", default=0.15, type=float)
    parser.add_argument("--hold-seconds", default=0.0, type=float)
    args = parser.parse_args(argv)

    graph = _load_demo_graph(args.graph, args.data_dir, args.cache_dir)
    out_path = None if args.no_save or args.out.lower() in {"", "none", "null"} else args.out
    scenes = run_debug_demo(
        graph=graph,
        out_path=out_path,
        steps=args.steps,
        seed=args.seed,
        max_cars=args.max_cars,
        max_requests=args.max_requests,
        spawn_rate_per_minute=args.spawn_rate_per_minute,
        episode_seconds=args.episode_seconds,
        policy=args.policy,
        show=args.show,
        pause_seconds=args.pause_seconds,
        hold_seconds=args.hold_seconds,
        fps=args.fps,
    )
    if out_path is None:
        print(f"Rendered {len(scenes)} live 2D Python demo frames without saving")
    else:
        print(f"Wrote {len(scenes)} 2D Python demo frames to {out_path}")
    return 0


def _draw_edges(ax, congestion: list[dict[str, Any]]) -> None:
    segments = [
        [edge["source"], edge["target"]]
        for edge in congestion
        if edge.get("source") is not None and edge.get("target") is not None
    ]
    if not segments:
        return
    values = np.asarray([float(edge.get("congestion", 1.0)) for edge in congestion[: len(segments)]])
    collection = LineCollection(
        segments,
        cmap="RdYlGn_r",
        linewidths=0.35,
        alpha=0.55,
        zorder=1,
    )
    collection.set_array(values)
    collection.set_clim(1.0, max(2.5, float(np.nanpercentile(values, 95))))
    ax.add_collection(collection)
    ax.autoscale()


def _render_scene_image(scene: dict[str, Any]) -> Image.Image:
    fig, _ = make_debug_figure(scene)
    fig.canvas.draw()
    buffer = np.asarray(fig.canvas.buffer_rgba())
    image = Image.fromarray(buffer[:, :, :3].copy())
    plt.close(fig)
    return image


def _choose_action(observation, policy: str, rng: np.random.Generator) -> int:
    if policy == "heuristic":
        return int(np.asarray(choose_marginal_value_action(observation)))
    mask = np.asarray(observation.action_mask, dtype=bool)
    valid = np.flatnonzero(mask)
    if len(valid) == 0:
        return 0
    if policy == "random":
        return int(rng.choice(valid))
    return int(valid[0])


def _show_scenes(scenes: list[dict[str, Any]]) -> None:
    for scene in scenes:
        fig, _ = make_debug_figure(scene)
        fig.show()
        plt.pause(0.25)
        plt.close(fig)


def _load_demo_graph(kind: str, data_dir: str, cache_dir: str):
    if kind == "synthetic":
        return build_synthetic_graph(
            node_lonlat=[(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)],
            edges=[
                {"source": 0, "target": 1, "travel_time_s": 2.0},
                {"source": 1, "target": 2, "travel_time_s": 2.0},
                {"source": 2, "target": 3, "travel_time_s": 2.0},
                {"source": 3, "target": 0, "travel_time_s": 2.0},
                {"source": 0, "target": 2, "travel_time_s": 3.0},
            ],
        )
    return load_public_data_graph(data_dir, include_routing=True, cache_dir=cache_dir)


def _initial_car_nodes(num_nodes: int, max_cars: int, seed: int) -> list[int]:
    rng = np.random.default_rng(seed + 2026)
    return [int(node) for node in rng.integers(0, num_nodes, size=max_cars)]


if __name__ == "__main__":
    raise SystemExit(main())
