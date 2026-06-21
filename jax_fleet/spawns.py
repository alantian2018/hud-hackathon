from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import random
from typing import Any

import numpy as np

from jax_fleet.env import make_env_params
from jax_fleet.types import EnvParams, GraphArrays


@dataclass(frozen=True)
class RequestSchedule:
    spawn_times_s: np.ndarray
    origin_nodes: np.ndarray
    destination_nodes: np.ndarray
    patience_seconds: np.ndarray


def seed_initial_car_nodes_like_js(
    data_dir: str | Path,
    graph: GraphArrays,
    *,
    seed: int,
    fleet_size: int,
) -> list[int]:
    nodes_payload = _load_json(Path(data_dir) / "ppo_nodes.json")
    exported_node_ids = sorted(int(node["node_id"]) for node in nodes_payload.get("nodes", []))
    if not exported_node_ids:
        raise ValueError("ppo_nodes.json does not contain exported nodes")

    compact_by_original = {
        int(original_id): compact_id
        for compact_id, original_id in enumerate(np.asarray(graph.original_node_ids))
    }
    rng = random.Random(int(seed) + 991)
    selected: list[int] = []
    attempts = 0
    max_attempts = max(len(exported_node_ids) * 4, max(1, int(fleet_size)) * 100)
    while len(selected) < int(fleet_size) and attempts < max_attempts:
        original_id = exported_node_ids[rng.randrange(len(exported_node_ids))]
        compact_id = compact_by_original.get(original_id)
        if compact_id is not None:
            selected.append(int(compact_id))
        attempts += 1
    if len(selected) < int(fleet_size):
        raise ValueError("could not seed enough cars inside the JAX graph SCC")
    return selected


def build_js_visual_request_schedule(
    data_dir: str | Path,
    graph: GraphArrays,
    *,
    start_time_seconds: float,
    episode_seconds: float,
) -> RequestSchedule:
    world = _load_json(Path(data_dir) / "mobility_world.json")
    start = float(start_time_seconds)
    day_seconds = 24.0 * 60.0 * 60.0
    requested_duration = float(episode_seconds)
    end = start + (requested_duration if np.isfinite(requested_duration) else day_seconds)
    rows = np.asarray(graph.node_grid_rows)
    cols = np.asarray(graph.node_grid_cols)

    records: list[tuple[float, int, int, float]] = []
    first_day = int(np.floor(start / day_seconds)) - 1
    last_day = int(np.ceil(end / day_seconds)) + 1
    for day in range(first_day, last_day + 1):
        offset = day * day_seconds
        for snapshot in world.get("snapshots", []):
            spawn_time = offset + float(snapshot.get("timestep", 0)) * 60.0
            if spawn_time < start or spawn_time >= end:
                continue
            for person in snapshot.get("new_people", []):
                origin_cell = person.get("origin", [0, 0])
                dest_cell = person.get("destination", [0, 0])
                origin = nearest_compact_node_for_grid_cell(
                    graph,
                    int(origin_cell[0]),
                    int(origin_cell[1]),
                    rows=rows,
                    cols=cols,
                )
                destination = nearest_compact_node_for_grid_cell(
                    graph,
                    int(dest_cell[0]),
                    int(dest_cell[1]),
                    rows=rows,
                    cols=cols,
                )
                records.append(
                    (
                        float(spawn_time),
                        int(origin),
                        int(destination),
                        float(person.get("patience", np.inf)) * 60.0,
                    )
                )

    records.sort(key=lambda item: item[0])
    if not records:
        return RequestSchedule(
            spawn_times_s=np.zeros((0,), dtype=np.float32),
            origin_nodes=np.zeros((0,), dtype=np.int32),
            destination_nodes=np.zeros((0,), dtype=np.int32),
            patience_seconds=np.zeros((0,), dtype=np.float32),
        )
    spawn_times, origins, destinations, patience = zip(*records, strict=True)
    return RequestSchedule(
        spawn_times_s=np.asarray(spawn_times, dtype=np.float32),
        origin_nodes=np.asarray(origins, dtype=np.int32),
        destination_nodes=np.asarray(destinations, dtype=np.int32),
        patience_seconds=np.asarray(patience, dtype=np.float32),
    )


def nearest_compact_node_for_grid_cell(
    graph: GraphArrays,
    row: int,
    col: int,
    *,
    rows: np.ndarray | None = None,
    cols: np.ndarray | None = None,
) -> int:
    rows = np.asarray(graph.node_grid_rows) if rows is None else rows
    cols = np.asarray(graph.node_grid_cols) if cols is None else cols
    candidates = np.flatnonzero((rows == int(row)) & (cols == int(col)))
    if len(candidates) == 0:
        for radius in range(1, 8):
            nearby = (
                (rows >= int(row) - radius)
                & (rows <= int(row) + radius)
                & (cols >= int(col) - radius)
                & (cols <= int(col) + radius)
            )
            candidates = np.flatnonzero(nearby)
            if len(candidates):
                break
    if len(candidates) == 0:
        return 0
    center_row = float(row) + 0.5
    center_col = float(col) + 0.5
    scores = (rows[candidates].astype(float) - center_row) ** 2 + (
        cols[candidates].astype(float) - center_col
    ) ** 2
    return int(candidates[int(np.argmin(scores))])


def make_spawned_env_params(
    graph: GraphArrays,
    *,
    graph_name: str = "synthetic",
    data_dir: str | Path = "public/data",
    spawn_source: str | None = None,
    seed: int = 0,
    max_cars: int = 16,
    max_requests: int = 128,
    initial_car_nodes: list[int] | np.ndarray | None = None,
    start_time_seconds: float = 0.0,
    episode_seconds: float = 3600.0,
    spawn_rate_per_minute: float = 0.0,
    **env_kwargs: Any,
) -> EnvParams:
    source = spawn_source or ("density" if graph_name == "sf" else "uniform")
    preplanned_requests: list[dict[str, float | int]] | None = None
    if source == "density":
        if initial_car_nodes is None and graph_name == "sf":
            initial_car_nodes = seed_initial_car_nodes_like_js(
                data_dir,
                graph,
                seed=seed,
                fleet_size=max_cars,
            )
        spawn_rate_per_minute = 0.0
        env_kwargs.setdefault("target_active_request_fraction", 0.5)
    if source == "js-visual":
        if initial_car_nodes is None:
            initial_car_nodes = seed_initial_car_nodes_like_js(
                data_dir,
                graph,
                seed=seed,
                fleet_size=max_cars,
            )
        schedule = build_js_visual_request_schedule(
            data_dir,
            graph,
            start_time_seconds=start_time_seconds,
            episode_seconds=episode_seconds,
        )
        preplanned_requests = [
            {
                "spawn_time_s": float(spawn),
                "origin": int(origin),
                "destination": int(destination),
                "patience_s": float(patience),
            }
            for spawn, origin, destination, patience in zip(
                schedule.spawn_times_s,
                schedule.origin_nodes,
                schedule.destination_nodes,
                schedule.patience_seconds,
                strict=True,
            )
        ]
        spawn_rate_per_minute = 0.0
        env_kwargs.setdefault("target_active_requests", 0)
    elif source not in {"uniform", "density"}:
        raise ValueError(f"unknown spawn_source: {source}")

    return make_env_params(
        graph,
        max_cars=max_cars,
        max_requests=max_requests,
        initial_car_nodes=initial_car_nodes,
        preplanned_requests=preplanned_requests,
        start_time_seconds=start_time_seconds,
        episode_seconds=episode_seconds,
        spawn_rate_per_minute=spawn_rate_per_minute,
        **env_kwargs,
    )


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))
