from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np

from jax_fleet.graph import load_public_data_graph
from jax_fleet.spawns import (
    build_js_visual_request_schedule,
    make_spawned_env_params,
    seed_initial_car_nodes_like_js,
)


DATA_DIR = Path("public/data")


def _expected_js_seeded_original_ids(graph, *, seed: int, fleet_size: int) -> list[int]:
    node_payload = json.loads((DATA_DIR / "ppo_nodes.json").read_text(encoding="utf-8"))
    exported_ids = sorted(int(node["node_id"]) for node in node_payload["nodes"])
    scc_ids = {int(node_id) for node_id in np.asarray(graph.original_node_ids)}
    rng = random.Random(seed + 991)
    selected: list[int] = []
    while len(selected) < fleet_size:
        original_id = exported_ids[rng.randrange(len(exported_ids))]
        if original_id in scc_ids:
            selected.append(original_id)
    return selected


def _expected_nearest_compact_node_for_cell(graph, row: int, col: int) -> int:
    rows = np.asarray(graph.node_grid_rows)
    cols = np.asarray(graph.node_grid_cols)
    candidates = np.flatnonzero((rows == row) & (cols == col))
    if len(candidates) == 0:
        for radius in range(1, 8):
            nearby = (
                (rows >= row - radius)
                & (rows <= row + radius)
                & (cols >= col - radius)
                & (cols <= col + radius)
            )
            candidates = np.flatnonzero(nearby)
            if len(candidates):
                break
    if len(candidates) == 0:
        return 0
    center_row = row + 0.5
    center_col = col + 0.5
    scores = (rows[candidates].astype(float) - center_row) ** 2 + (
        cols[candidates].astype(float) - center_col
    ) ** 2
    return int(candidates[int(np.argmin(scores))])


def test_js_visual_car_seeding_matches_export_order_inside_scc() -> None:
    graph = load_public_data_graph(DATA_DIR, include_routing=False)

    compact_nodes = seed_initial_car_nodes_like_js(DATA_DIR, graph, seed=7, fleet_size=4)

    original_ids = [int(np.asarray(graph.original_node_ids)[node]) for node in compact_nodes]
    assert original_ids == _expected_js_seeded_original_ids(graph, seed=7, fleet_size=4)


def test_js_visual_request_schedule_uses_mobility_world_minutes_and_grid_snapping() -> None:
    graph = load_public_data_graph(DATA_DIR, include_routing=False)
    world = json.loads((DATA_DIR / "mobility_world.json").read_text(encoding="utf-8"))
    first_person = world["snapshots"][0]["new_people"][0]

    schedule = build_js_visual_request_schedule(
        DATA_DIR,
        graph,
        start_time_seconds=0.0,
        episode_seconds=16 * 60.0,
    )

    assert schedule.spawn_times_s[:4].tolist() == [0.0, 0.0, 0.0, 900.0]
    assert int(schedule.origin_nodes[0]) == _expected_nearest_compact_node_for_cell(
        graph,
        first_person["origin"][0],
        first_person["origin"][1],
    )
    assert int(schedule.destination_nodes[0]) == _expected_nearest_compact_node_for_cell(
        graph,
        first_person["destination"][0],
        first_person["destination"][1],
    )
    assert float(schedule.patience_seconds[0]) == float(first_person["patience"] * 60)


def test_make_spawned_env_params_defaults_sf_to_density_top_up() -> None:
    graph = load_public_data_graph(DATA_DIR, include_routing=False)

    params = make_spawned_env_params(
        graph,
        data_dir=DATA_DIR,
        graph_name="sf",
        seed=7,
        max_cars=3,
        max_requests=4,
        episode_seconds=16 * 60.0,
    )

    original_ids = [int(np.asarray(graph.original_node_ids)[node]) for node in np.asarray(params.initial_car_nodes)]
    assert original_ids == _expected_js_seeded_original_ids(graph, seed=7, fleet_size=3)
    assert params.target_active_requests == 1
    assert params.preplanned_spawn_times.shape[0] == 0


def test_make_spawned_env_params_keeps_explicit_js_visual_schedule() -> None:
    graph = load_public_data_graph(DATA_DIR, include_routing=False)

    params = make_spawned_env_params(
        graph,
        data_dir=DATA_DIR,
        graph_name="sf",
        spawn_source="js-visual",
        seed=7,
        max_cars=3,
        max_requests=4,
        episode_seconds=16 * 60.0,
    )

    assert params.preplanned_spawn_times.shape[0] > params.max_requests
    assert params.target_active_requests == 0
