#!/usr/bin/env python3
"""
Build an OSMnx-based network dataset for the Deck.gl demo and PPO experiments.

Pipeline:
OSMnx graph -> engineered edge features -> hourly traffic profile ->
dynamic edge weights -> sample trajectories.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any

import networkx as nx
import osmnx as ox


# Relative speed multipliers per hour (0-23).
# 1.0 means free-flow. Values <1 indicate congestion.
HOURLY_SPEED_MULTIPLIER = [
    0.95,  # 00
    1.00,  # 01
    1.00,  # 02
    1.00,  # 03
    0.95,  # 04
    0.90,  # 05
    0.75,  # 06
    0.60,  # 07
    0.55,  # 08
    0.65,  # 09
    0.78,  # 10
    0.84,  # 11
    0.88,  # 12
    0.85,  # 13
    0.82,  # 14
    0.72,  # 15
    0.60,  # 16
    0.52,  # 17
    0.58,  # 18
    0.70,  # 19
    0.80,  # 20
    0.88,  # 21
    0.92,  # 22
    0.95,  # 23
]


BASE_SPEED_BY_HIGHWAY_KPH = {
    "motorway": 90.0,
    "trunk": 75.0,
    "primary": 55.0,
    "secondary": 42.0,
    "tertiary": 35.0,
    "residential": 28.0,
    "service": 22.0,
    "living_street": 15.0,
}


GRID_SIZE = 50
NODE_DENSITY_NOISE_STD_FRAC = 0.18
ROAD_CONGESTION_SENSITIVITY = {
    "motorway": 0.7,
    "trunk": 0.8,
    "primary": 0.95,
    "secondary": 1.0,
    "tertiary": 1.08,
    "residential": 1.15,
    "service": 1.2,
    "living_street": 1.25,
}


def normalize_highway(highway: Any) -> str:
    if isinstance(highway, list) and highway:
        return str(highway[0])
    if highway is None:
        return "residential"
    return str(highway)


def parse_speed_kph(maxspeed: Any, fallback_kph: float) -> float:
    if maxspeed is None:
        return fallback_kph

    if isinstance(maxspeed, list) and maxspeed:
        raw = str(maxspeed[0])
    else:
        raw = str(maxspeed)

    numeric = "".join(ch for ch in raw if ch.isdigit() or ch == ".")
    if not numeric:
        return fallback_kph

    value = float(numeric)
    if "mph" in raw.lower():
        return value * 1.60934
    return value


def default_speed_for_highway(highway: str) -> float:
    for key, speed in BASE_SPEED_BY_HIGHWAY_KPH.items():
        if key in highway:
            return speed
    return 30.0


def congestion_sensitivity_for_highway(highway: str) -> float:
    for key, value in ROAD_CONGESTION_SENSITIVITY.items():
        if key in highway:
            return value
    return 1.0


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def synthetic_population_density(
    row: int,
    col: int,
    rows: int,
    cols: int,
    rng: random.Random,
) -> float:
    # Radial urban core + secondary center + low-amplitude random texture.
    cx = (cols - 1) / 2.0
    cy = (rows - 1) / 2.0
    x = (col - cx) / max(1.0, cols / 2.0)
    y = (row - cy) / max(1.0, rows / 2.0)

    core = math.exp(-2.6 * (x * x + y * y))
    secondary = 0.45 * math.exp(-10.0 * ((x - 0.45) ** 2 + (y + 0.2) ** 2))
    corridor = 0.15 * (math.sin((col / max(1.0, cols - 1)) * math.pi * 3.0) + 1.0) / 2.0
    texture = rng.uniform(-0.06, 0.06)

    density = 1200.0 + 12500.0 * (core + secondary) + 3200.0 * corridor + 900.0 * texture
    return round(clamp(density, 300.0, 24000.0), 3)


def build_population_grid(
    graph: nx.MultiDiGraph,
    rows: int,
    cols: int,
    rng: random.Random,
) -> dict[str, Any]:
    xs = [float(data["x"]) for _, data in graph.nodes(data=True)]
    ys = [float(data["y"]) for _, data in graph.nodes(data=True)]
    min_lon, max_lon = min(xs), max(xs)
    min_lat, max_lat = min(ys), max(ys)
    lon_span = max(1e-9, max_lon - min_lon)
    lat_span = max(1e-9, max_lat - min_lat)

    values = []
    for r in range(rows):
        row_vals = []
        for c in range(cols):
            row_vals.append(synthetic_population_density(r, c, rows, cols, rng))
        values.append(row_vals)

    return {
        "rows": rows,
        "cols": cols,
        "bounds": {
            "min_lon": min_lon,
            "max_lon": max_lon,
            "min_lat": min_lat,
            "max_lat": max_lat,
        },
        "resolution": {
            "delta_lon": lon_span / cols,
            "delta_lat": lat_span / rows,
        },
        "values": values,
    }


def node_to_grid_index(
    lon: float,
    lat: float,
    grid: dict[str, Any],
) -> tuple[int, int]:
    bounds = grid["bounds"]
    rows = int(grid["rows"])
    cols = int(grid["cols"])
    lon_span = max(1e-9, bounds["max_lon"] - bounds["min_lon"])
    lat_span = max(1e-9, bounds["max_lat"] - bounds["min_lat"])

    col = int((lon - bounds["min_lon"]) / lon_span * cols)
    row = int((lat - bounds["min_lat"]) / lat_span * rows)
    row = int(clamp(row, 0, rows - 1))
    col = int(clamp(col, 0, cols - 1))
    return row, col


def assign_node_population_density(
    graph: nx.MultiDiGraph,
    grid: dict[str, Any],
    rng: random.Random,
) -> list[dict[str, Any]]:
    nodes_payload = []
    values = grid["values"]

    for node_id, data in graph.nodes(data=True):
        lon = float(data["x"])
        lat = float(data["y"])
        row, col = node_to_grid_index(lon, lat, grid)
        base_density = float(values[row][col])
        sigma = max(15.0, base_density * NODE_DENSITY_NOISE_STD_FRAC)
        noisy_density = clamp(rng.gauss(base_density, sigma), 0.0, 30000.0)
        noisy_density = round(noisy_density, 3)

        graph.nodes[node_id]["grid_row"] = row
        graph.nodes[node_id]["grid_col"] = col
        graph.nodes[node_id]["population_density_base"] = base_density
        graph.nodes[node_id]["population_density_noisy"] = noisy_density

        nodes_payload.append(
            {
                "node_id": int(node_id),
                "lon": lon,
                "lat": lat,
                "grid_row": row,
                "grid_col": col,
                "population_density_base": base_density,
                "population_density_noisy": noisy_density,
            }
        )

    return nodes_payload


def build_network(place: str, seed: int) -> dict[str, Any]:
    random.seed(seed)
    rng = random.Random(seed)

    graph = ox.graph_from_place(place, network_type="drive")
    graph = ox.add_edge_speeds(graph)
    graph = ox.add_edge_travel_times(graph)

    population_grid = build_population_grid(graph, GRID_SIZE, GRID_SIZE, rng)
    ppo_nodes = assign_node_population_density(graph, population_grid, rng)

    edges_geojson = {
        "type": "FeatureCollection",
        "features": [],
    }
    ppo_edges = []

    # Build edge features and attach travel time profiles for routing.
    for u, v, key, data in graph.edges(keys=True, data=True):
        if "geometry" not in data:
            continue

        coords = [[float(x), float(y)] for x, y in data["geometry"].coords]
        highway = normalize_highway(data.get("highway"))
        fallback = default_speed_for_highway(highway)
        base_speed_kph = parse_speed_kph(data.get("maxspeed"), fallback)
        length_m = float(data.get("length", 0.0))
        if length_m <= 0:
            continue

        src_density = float(graph.nodes[u].get("population_density_noisy", 0.0))
        dst_density = float(graph.nodes[v].get("population_density_noisy", 0.0))
        avg_density = (src_density + dst_density) / 2.0
        density_scale = clamp(avg_density / 12000.0, 0.25, 2.0)
        road_sensitivity = congestion_sensitivity_for_highway(highway)
        edge_bias = rng.uniform(0.88, 1.12)
        edge_phase = rng.uniform(0.0, 2.0 * math.pi)

        free_flow_time_s = length_m / max(1.0, (base_speed_kph * 1000.0 / 3600.0))
        hourly_speed_kph = []
        for hour, mult in enumerate(HOURLY_SPEED_MULTIPLIER):
            peak_strength = 1.0 - mult
            localized_penalty = (
                peak_strength
                * 0.34
                * road_sensitivity
                * (0.65 + 0.55 * density_scale)
                * edge_bias
            )
            wave = 1.0 + 0.08 * math.sin((hour / 24.0) * 2.0 * math.pi + edge_phase)
            eff_mult = clamp((mult - localized_penalty) * wave, 0.18, 1.08)
            hourly_speed_kph.append(round(base_speed_kph * eff_mult, 3))

        hourly_travel_time_s = [
            round(length_m / max(1.0, (speed * 1000.0 / 3600.0)), 3)
            for speed in hourly_speed_kph
        ]
        hourly_congestion = [
            round(base_speed_kph / max(1.0, speed), 4) for speed in hourly_speed_kph
        ]
        hourly_volume = [
            round(
                (max(0.0, 1.0 - (speed / max(1.0, base_speed_kph))) * 140.0)
                + density_scale * 18.0
                + rng.uniform(0.0, 10.0),
                2,
            )
            for speed in hourly_speed_kph
        ]

        data["hourly_travel_time_s"] = hourly_travel_time_s

        props = {
            "u": int(u),
            "v": int(v),
            "key": int(key),
            "osmid": str(data.get("osmid", "")),
            "name": str(data.get("name", "")),
            "highway": highway,
            "length_m": round(length_m, 3),
            "lanes": str(data.get("lanes", "")),
            "base_speed_kph": round(base_speed_kph, 3),
            "free_flow_time_s": round(free_flow_time_s, 3),
            "hourly_speed_kph": hourly_speed_kph,
            "hourly_travel_time_s": hourly_travel_time_s,
            "hourly_congestion_factor": hourly_congestion,
            "hourly_volume_index": hourly_volume,
        }

        edges_geojson["features"].append(
            {
                "type": "Feature",
                "geometry": {"type": "LineString", "coordinates": coords},
                "properties": props,
            }
        )

        # Minimal PPO-friendly edge payload: features + dynamic weights.
        ppo_edges.append(
            {
                "edge_id": f"{u}-{v}-{key}",
                "u": int(u),
                "v": int(v),
                "key": int(key),
                "features": {
                    "length_m": round(length_m, 3),
                    "base_speed_kph": round(base_speed_kph, 3),
                    "highway": highway,
                    "src_population_density_noisy": round(
                        src_density, 3
                    ),
                    "dst_population_density_noisy": round(
                        dst_density, 3
                    ),
                },
                "dynamic_weights_travel_time_s": hourly_travel_time_s,
                "dynamic_volume_index": hourly_volume,
            }
        )

    trips = sample_trips(graph, trip_count=16)

    return {
        "edges_geojson": edges_geojson,
        "population_grid": population_grid,
        "ppo_nodes": ppo_nodes,
        "ppo_edges": ppo_edges,
        "sample_trips": trips,
        "meta": {
            "place": place,
            "hourly_speed_multiplier": HOURLY_SPEED_MULTIPLIER,
            "population_grid_rows": GRID_SIZE,
            "population_grid_cols": GRID_SIZE,
            "node_count": len(ppo_nodes),
            "edge_count": len(edges_geojson["features"]),
            "trip_count": len(trips),
        },
    }


def route_weight_by_hour(data: dict[str, Any], hour: int) -> float:
    profile = data.get("hourly_travel_time_s")
    if profile and len(profile) == 24:
        return float(profile[hour])
    return float(data.get("travel_time", 1.0))


def sample_trips(graph: nx.MultiDiGraph, trip_count: int = 12) -> list[dict[str, Any]]:
    nodes = list(graph.nodes())
    if len(nodes) < 2:
        return []

    trips = []
    attempts = 0
    max_attempts = trip_count * 16

    while len(trips) < trip_count and attempts < max_attempts:
        attempts += 1
        src, dst = random.sample(nodes, 2)
        start_hour = random.choice([6, 8, 9, 12, 15, 17, 18, 21])
        start_minute = start_hour * 60 + random.randint(0, 59)

        try:
            path = nx.shortest_path(
                graph,
                source=src,
                target=dst,
                weight=lambda u, v, data: route_weight_by_hour(data, start_hour),
            )
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            continue

        coords = []
        timestamps = []
        running_t = float(start_minute)

        for idx in range(len(path) - 1):
            u, v = path[idx], path[idx + 1]
            edge_options = graph.get_edge_data(u, v)
            if not edge_options:
                continue

            best_key = min(
                edge_options.keys(),
                key=lambda k: route_weight_by_hour(edge_options[k], start_hour),
            )
            edge = edge_options[best_key]

            geometry = edge.get("geometry")
            if geometry is None:
                continue

            segment = [[float(x), float(y)] for x, y in geometry.coords]
            segment_time_minutes = route_weight_by_hour(edge, start_hour) / 60.0
            per_vertex = segment_time_minutes / max(1, len(segment) - 1)

            if not coords:
                coords.extend(segment)
                timestamps.extend(
                    [round(running_t + i * per_vertex, 3) for i in range(len(segment))]
                )
            else:
                coords.extend(segment[1:])
                timestamps.extend(
                    [
                        round(running_t + (i + 1) * per_vertex, 3)
                        for i in range(len(segment) - 1)
                    ]
                )
            running_t += segment_time_minutes

        if len(coords) >= 2 and len(coords) == len(timestamps):
            trips.append(
                {
                    "path": coords,
                    "timestamps": timestamps,
                    "start_hour": start_hour,
                }
            )

    return trips


def dump_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate OSMnx traffic dataset.")
    parser.add_argument(
        "--place",
        default="San Francisco, California, USA",
        help="Place name for OSMnx graph_from_place.",
    )
    parser.add_argument(
        "--out-dir",
        default="public/data",
        help="Output directory for generated JSON artifacts.",
    )
    parser.add_argument("--seed", default=7, type=int, help="Random seed.")
    args = parser.parse_args()

    payload = build_network(place=args.place, seed=args.seed)
    out_dir = Path(args.out_dir)

    dump_json(out_dir / "osmnx_edges.geojson", payload["edges_geojson"])
    dump_json(out_dir / "sample_trips.json", payload["sample_trips"])
    dump_json(out_dir / "population_density_grid.json", payload["population_grid"])
    dump_json(
        out_dir / "ppo_nodes.json",
        {
            "meta": payload["meta"],
            "nodes": payload["ppo_nodes"],
        },
    )
    dump_json(
        out_dir / "ppo_edges.json",
        {
            "meta": payload["meta"],
            "edges": payload["ppo_edges"],
        },
    )
    dump_json(out_dir / "pipeline_meta.json", payload["meta"])

    print(
        f"Generated {payload['meta']['edge_count']} edges and "
        f"{payload['meta']['trip_count']} trips in {out_dir}"
    )


if __name__ == "__main__":
    main()
