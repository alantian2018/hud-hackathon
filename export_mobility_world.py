#!/usr/bin/env python3
from __future__ import annotations

import argparse
import heapq
import json
import math
from pathlib import Path
import random

from mobility_sim import DemandGenerator, PeopleGenerator, TrafficGenerator, WorldGenerators


def load_grid(path: Path) -> dict | tuple[int, int]:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return (50, 50)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


class MapGraph:
    def __init__(self, nodes_payload: dict, edges_payload: dict, edges_geojson: dict) -> None:
        self.nodes = {
            int(node["node_id"]): node
            for node in nodes_payload.get("nodes", [])
        }
        self.nodes_by_grid: dict[tuple[int, int], list[dict]] = {}
        for node in self.nodes.values():
            self.nodes_by_grid.setdefault((int(node["grid_row"]), int(node["grid_col"])), []).append(node)

        geometry_by_edge = {}
        for feature in edges_geojson.get("features", []):
            props = feature.get("properties", {})
            key = (int(props["u"]), int(props["v"]), int(props["key"]))
            geometry_by_edge[key] = feature.get("geometry", {}).get("coordinates", [])

        self.adjacency: dict[int, list[dict]] = {}
        for edge in edges_payload.get("edges", []):
            u = int(edge["u"])
            v = int(edge["v"])
            key = int(edge["key"])
            weights = edge.get("dynamic_weights_travel_time_s") or [1.0] * 24
            coords = geometry_by_edge.get((u, v, key)) or [
                self.node_position(u),
                self.node_position(v),
            ]
            self.adjacency.setdefault(u, []).append(
                {"to": v, "weights": weights, "coords": coords}
            )
            # The grid simulator is not modeling one-way constraints yet, so keep routing connected.
            self.adjacency.setdefault(v, []).append(
                {"to": u, "weights": weights, "coords": list(reversed(coords))}
            )

    def node_position(self, node_id: int) -> list[float]:
        node = self.nodes[int(node_id)]
        return [float(node["lon"]), float(node["lat"])]

    def nearest_node_for_cell(self, cell: list[int] | tuple[int, int]) -> dict:
        row, col = int(cell[0]), int(cell[1])
        candidates = self.nodes_by_grid.get((row, col))
        if not candidates:
            candidates = self._nearby_grid_candidates(row, col)
        if not candidates:
            return next(iter(self.nodes.values()))

        center_row = row + 0.5
        center_col = col + 0.5
        return min(
            candidates,
            key=lambda node: (float(node["grid_row"]) - center_row) ** 2
            + (float(node["grid_col"]) - center_col) ** 2,
        )

    def _nearby_grid_candidates(self, row: int, col: int) -> list[dict]:
        for radius in range(1, 8):
            candidates = []
            for rr in range(row - radius, row + radius + 1):
                for cc in range(col - radius, col + radius + 1):
                    candidates.extend(self.nodes_by_grid.get((rr, cc), []))
            if candidates:
                return candidates
        return []

    def route(self, start_node: int, goal_node: int, hour: int) -> dict:
        start_node = int(start_node)
        goal_node = int(goal_node)
        if start_node == goal_node:
            point = self.node_position(start_node)
            return {"node_path": [start_node], "coordinates": [point], "cost": 0.0}

        frontier = [(0.0, start_node)]
        best = {start_node: 0.0}
        came_from: dict[int, tuple[int, dict]] = {}

        while frontier:
            cost, node = heapq.heappop(frontier)
            if node == goal_node:
                break
            if cost > best.get(node, math.inf):
                continue
            for edge in self.adjacency.get(node, []):
                weight = float(edge["weights"][hour % len(edge["weights"])])
                next_cost = cost + max(0.001, weight)
                to_node = int(edge["to"])
                if next_cost < best.get(to_node, math.inf):
                    best[to_node] = next_cost
                    came_from[to_node] = (node, edge)
                    heapq.heappush(frontier, (next_cost, to_node))

        if goal_node not in best:
            return {
                "node_path": [start_node, goal_node],
                "coordinates": [self.node_position(start_node), self.node_position(goal_node)],
                "cost": math.inf,
            }

        node_path = [goal_node]
        edge_path = []
        while node_path[-1] != start_node:
            previous, edge = came_from[node_path[-1]]
            edge_path.append(edge)
            node_path.append(previous)
        node_path.reverse()
        edge_path.reverse()

        coordinates = []
        for edge in edge_path:
            coords = edge["coords"]
            if not coordinates:
                coordinates.extend(coords)
            else:
                coordinates.extend(coords[1:])
        return {
            "node_path": node_path,
            "coordinates": coordinates,
            "cost": round(best[goal_node], 6),
        }


def seed_car_nodes(graph: MapGraph, seed: int, fleet_size: int) -> list[int]:
    rng = random.Random(seed + 991)
    node_ids = sorted(graph.nodes.keys())
    if not node_ids:
        return []
    return [node_ids[rng.randrange(len(node_ids))] for _ in range(fleet_size)]


def build_map_dispatch(graph: MapGraph, people: list, timestep: int, seed: int, fleet_size: int) -> dict:
    hour = (timestep // 60) % 24
    car_nodes = seed_car_nodes(graph, seed + timestep, fleet_size)
    cars = [
        {
            "id": f"car-{idx}",
            "node_id": node_id,
            "position": graph.node_position(node_id),
            "status": "idle",
            "assigned_person_id": None,
            "stall_ticks": 1,
        }
        for idx, node_id in enumerate(car_nodes)
    ]

    map_people = []
    assignments = []
    idle_car_indexes = list(range(len(cars)))
    assigned_person_ids = set()

    for person in people:
        pickup_node = graph.nearest_node_for_cell(person.origin)
        dropoff_node = graph.nearest_node_for_cell(person.destination)
        person_payload = person.to_dict()
        person_payload.update(
            {
                "pickup_node_id": int(pickup_node["node_id"]),
                "dropoff_node_id": int(dropoff_node["node_id"]),
                "pickup_position": [float(pickup_node["lon"]), float(pickup_node["lat"])],
                "dropoff_position": [float(dropoff_node["lon"]), float(dropoff_node["lat"])],
            }
        )
        map_people.append(person_payload)

        best = None
        for car_idx in idle_car_indexes:
            route_to_pickup = graph.route(cars[car_idx]["node_id"], pickup_node["node_id"], hour)
            cost = route_to_pickup["cost"]
            if best is None or cost < best[0]:
                best = (cost, car_idx, route_to_pickup)
        if best is None:
            continue

        _, car_idx, pickup_route = best
        idle_car_indexes.remove(car_idx)
        dropoff_route = graph.route(pickup_node["node_id"], dropoff_node["node_id"], hour)
        cars[car_idx].update(
            {
                "status": "to_pickup",
                "assigned_person_id": person.id,
                "pickup_node_id": int(pickup_node["node_id"]),
                "dropoff_node_id": int(dropoff_node["node_id"]),
                "stall_ticks": 0,
            }
        )
        if len(pickup_route["coordinates"]) > 1:
            cars[car_idx]["position"] = pickup_route["coordinates"][1]
        assignments.append(
            {
                "car_id": cars[car_idx]["id"],
                "person_id": person.id,
                "pickup_node_id": int(pickup_node["node_id"]),
                "dropoff_node_id": int(dropoff_node["node_id"]),
                "pickup_position": [float(pickup_node["lon"]), float(pickup_node["lat"])],
                "dropoff_position": [float(dropoff_node["lon"]), float(dropoff_node["lat"])],
                "pickup_route": pickup_route,
                "dropoff_route": dropoff_route,
                "total_cost": round(pickup_route["cost"] + dropoff_route["cost"], 6),
            }
        )
        assigned_person_ids.add(person.id)

    active_cars = len(assignments)
    stalled_cars = len(cars) - active_cars
    assigned_people = [person for person in people if person.id in assigned_person_ids]
    stats = {
        "completed_trips": len(assigned_people),
        "revenue": round(sum(person.value for person in assigned_people), 2),
        "demand_served_pct": round(len(assigned_people) / len(people) * 100.0, 2) if people else 100.0,
        "fleet_utilization_pct": round(active_cars / max(1, len(cars)) * 100.0, 2),
        "active_cars": active_cars,
        "stalled_cars": stalled_cars,
        "unassigned_people": len(people) - len(assigned_people),
    }

    return {
        "map_people": map_people,
        "map_dispatch": {
            "assignments": assignments,
            "cars": cars,
            "summary": {
                "num_assignments": len(assignments),
                "num_unassigned_people": stats["unassigned_people"],
                "num_stalled_cars": stalled_cars,
                "num_active_cars": active_cars,
            },
        },
        "map_greedy_stats": stats,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Export greedy mobility snapshots for the map UI.")
    parser.add_argument("--grid", default="public/data/population_density_grid.json")
    parser.add_argument("--nodes", default="public/data/ppo_nodes.json")
    parser.add_argument("--edges", default="public/data/ppo_edges.json")
    parser.add_argument("--edges-geojson", default="public/data/osmnx_edges.geojson")
    parser.add_argument("--out", default="public/data/mobility_world.json")
    parser.add_argument("--seed", default=7, type=int)
    parser.add_argument("--fleet-size", default=16, type=int)
    parser.add_argument("--step-minutes", default=60, type=int)
    args = parser.parse_args()

    grid = load_grid(Path(args.grid))
    graph = MapGraph(
        load_json(Path(args.nodes)),
        load_json(Path(args.edges)),
        load_json(Path(args.edges_geojson)),
    )
    snapshots = []
    for timestep in range(0, 24 * 60, max(1, args.step_minutes)):
        demand = DemandGenerator(grid=grid, seed=args.seed + timestep + 1)
        traffic = TrafficGenerator(grid=grid, seed=args.seed + timestep + 2)
        people_generator = PeopleGenerator(grid=grid, seed=args.seed + timestep + 3)
        demand_heatmap = demand.get_heatmap(timestep)
        traffic_heatmap = traffic.get_heatmap(timestep, demand_heatmap)
        people = people_generator.generate(timestep, demand_heatmap, traffic_heatmap)
        world = WorldGenerators(grid=grid, seed=args.seed + timestep, fleet_size=args.fleet_size)
        snapshot = world.step(timestep)
        map_payload = build_map_dispatch(graph, people, timestep, args.seed, args.fleet_size)
        snapshot["new_people"] = [person.to_dict() for person in people]
        snapshot["map_people"] = map_payload["map_people"]
        snapshot["map_dispatch"] = map_payload["map_dispatch"]
        snapshot["map_greedy_stats"] = map_payload["map_greedy_stats"]
        snapshot["greedy_stats"] = map_payload["map_greedy_stats"]
        snapshot["summary"]["greedy_stats"] = map_payload["map_greedy_stats"]
        snapshots.append(snapshot)

    payload = {
        "seed": args.seed,
        "fleet_size": args.fleet_size,
        "step_minutes": args.step_minutes,
        "snapshots": snapshots,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    print(f"Wrote {len(snapshots)} greedy mobility snapshots to {out_path}")


if __name__ == "__main__":
    main()
