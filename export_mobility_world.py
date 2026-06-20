#!/usr/bin/env python3
from __future__ import annotations

import argparse
import heapq
import json
import math
from pathlib import Path
import random

from mobility_sim import DemandGenerator, PeopleGenerator, TrafficGenerator, build_people_grid


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
        self._route_cache: dict[tuple[int, int, int], dict] = {}
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
        cache_key = (start_node, goal_node, int(hour) % 24)
        if cache_key in self._route_cache:
            return self._route_cache[cache_key]

        if start_node == goal_node:
            point = self.node_position(start_node)
            route = {"node_path": [start_node], "coordinates": [point], "cost": 0.0, "fallback": False}
            self._route_cache[cache_key] = route
            return route

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
            route = self._direct_fallback_route(start_node, goal_node)
            self._route_cache[cache_key] = route
            return route

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
        route = {
            "node_path": node_path,
            "coordinates": coordinates,
            "cost": round(best[goal_node], 6),
            "fallback": False,
        }
        self._route_cache[cache_key] = route
        return route

    def _direct_fallback_route(self, start_node: int, goal_node: int) -> dict:
        start = self.node_position(start_node)
        goal = self.node_position(goal_node)
        mean_lat = math.radians((start[1] + goal[1]) / 2)
        dx = (goal[0] - start[0]) * math.cos(mean_lat) * 111_320
        dy = (goal[1] - start[1]) * 111_320
        distance_m = math.hypot(dx, dy)
        # Keep browser JSON valid even when OSMnx graph components are disconnected.
        return {
            "node_path": [start_node, goal_node],
            "coordinates": [start, goal],
            "cost": round(distance_m / 7.0 + 45.0, 6),
            "fallback": True,
        }


def seed_car_nodes(graph: MapGraph, seed: int, fleet_size: int) -> list[int]:
    rng = random.Random(seed + 991)
    node_ids = sorted(graph.nodes.keys())
    if not node_ids:
        return []
    return [node_ids[rng.randrange(len(node_ids))] for _ in range(fleet_size)]


def grid_cell_for_position(position: list[float], grid: dict) -> list[int] | None:
    bounds = grid.get("bounds") or {}
    rows = int(grid.get("rows", 0))
    cols = int(grid.get("cols", 0))
    min_lon = bounds.get("min_lon")
    max_lon = bounds.get("max_lon")
    min_lat = bounds.get("min_lat")
    max_lat = bounds.get("max_lat")
    if not rows or not cols or min_lon is None or max_lon is None or min_lat is None or max_lat is None:
        return None
    lon, lat = float(position[0]), float(position[1])
    col = int((lon - float(min_lon)) / max(1e-9, float(max_lon) - float(min_lon)) * cols)
    row = int((lat - float(min_lat)) / max(1e-9, float(max_lat) - float(min_lat)) * rows)
    return [max(0, min(rows - 1, row)), max(0, min(cols - 1, col))]


def distance_m(a: list[float], b: list[float]) -> float:
    mean_lat = math.radians((float(a[1]) + float(b[1])) / 2.0)
    dx = (float(b[0]) - float(a[0])) * math.cos(mean_lat) * 111_320
    dy = (float(b[1]) - float(a[1])) * 111_320
    return math.hypot(dx, dy)


def interpolate_position(coordinates: list[list[float]], progress: float) -> list[float]:
    if not coordinates:
        return [0.0, 0.0]
    if len(coordinates) == 1:
        return coordinates[0]

    target = max(0.0, min(1.0, progress))
    segments = []
    total = 0.0
    for idx in range(len(coordinates) - 1):
        length = distance_m(coordinates[idx], coordinates[idx + 1])
        segments.append(length)
        total += length
    if total <= 0.0:
        return coordinates[0]

    remaining = total * target
    for idx, length in enumerate(segments):
        if remaining > length:
            remaining -= length
            continue
        t = 0.0 if length <= 0 else remaining / length
        a = coordinates[idx]
        b = coordinates[idx + 1]
        return [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t]
    return coordinates[-1]


def merge_route_coordinates(*routes: dict) -> list[list[float]]:
    coordinates: list[list[float]] = []
    for route in routes:
        route_coords = route.get("coordinates") or []
        if not route_coords:
            continue
        if not coordinates:
            coordinates.extend(route_coords)
        else:
            coordinates.extend(route_coords[1:])
    return coordinates


class StatefulMapDispatch:
    def __init__(
        self,
        graph: MapGraph,
        grid: dict,
        seed: int,
        fleet_size: int,
        candidate_limit: int = 8,
    ) -> None:
        self.graph = graph
        self.grid = grid
        self.seed = int(seed)
        self.candidate_limit = int(candidate_limit)
        self.cars = []
        for idx, node_id in enumerate(seed_car_nodes(graph, seed, fleet_size)):
            node = graph.nodes[node_id]
            self.cars.append(
                {
                    "id": f"car-{idx}",
                    "node_id": int(node_id),
                    "position": graph.node_position(node_id),
                    "grid_cell": [int(node["grid_row"]), int(node["grid_col"])],
                    "status": "idle",
                    "assigned_person_id": None,
                    "stall_ticks": 0,
                    "assignment": None,
                    "route_elapsed": 0.0,
                }
            )
        self.pending: list[dict] = []
        self.total_requests = 0
        self.completed_trips = 0
        self.canceled_requests = 0
        self.revenue = 0.0
        self.total_wait_minutes = 0.0

    def add_requests(self, people: list, timestep: int) -> None:
        for person in people:
            self.pending.append(
                {
                    "person": person,
                    "payload": self._map_person(person, "waiting"),
                    "created_at": timestep,
                }
            )
            self.total_requests += 1

    def step(self, timestep: int, people: list) -> dict:
        self._expire_requests(timestep)
        self.add_requests(people, timestep)
        new_assignments = self._assign_waiting(timestep)
        active_assignments = []
        for car in self.cars:
            assignment = car.get("assignment")
            if assignment is None:
                continue
            active_assignments.append(
                {
                    **assignment,
                    "current_position": car["position"],
                    "current_grid_cell": car.get("grid_cell"),
                    "route_elapsed": round(float(car.get("route_elapsed", 0.0)), 6),
                }
            )
        active_people_ids = {assignment["person_id"] for assignment in active_assignments}
        map_people = [
            {**item["payload"], "status": "assigned" if item["payload"]["id"] in active_people_ids else "waiting"}
            for item in self.pending
        ]
        active_cars = sum(1 for car in self.cars if car["status"] != "idle")
        stalled_cars = len(self.cars) - active_cars
        matched_requests = self.completed_trips + len(active_people_ids)
        demand_served = matched_requests / max(1, self.total_requests) * 100.0
        wait_time = self.total_wait_minutes + self._queued_customer_wait(timestep)

        return {
            "map_people": map_people,
            "map_dispatch": {
                "assignments": active_assignments,
                "new_assignments": new_assignments,
                "cars": [self._car_payload(car) for car in self.cars],
                "summary": {
                    "num_assignments": len(active_assignments),
                    "num_new_assignments": len(new_assignments),
                    "num_unassigned_people": len(map_people) - len(active_people_ids),
                    "num_stalled_cars": stalled_cars,
                    "num_active_cars": active_cars,
                },
            },
            "map_greedy_stats": {
                "completed_trips": self.completed_trips,
                "revenue": round(self.revenue, 2),
                "demand_served_pct": round(demand_served, 2),
                "wait_time_min": round(wait_time, 2),
                "fleet_utilization_pct": round(active_cars / max(1, len(self.cars)) * 100.0, 2),
                "active_cars": active_cars,
                "stalled_cars": stalled_cars,
                "unassigned_people": len(map_people) - len(active_people_ids),
                "canceled_requests": self.canceled_requests,
                "total_requests": self.total_requests,
            },
        }

    def advance(self, seconds: float) -> None:
        for car in self.cars:
            assignment = car.get("assignment")
            if assignment is None:
                car["stall_ticks"] += 1
                continue

            route = assignment["route"]
            total_cost = max(1.0, float(route["cost"]))
            car["route_elapsed"] = min(total_cost, float(car["route_elapsed"]) + float(seconds))
            progress = car["route_elapsed"] / total_cost
            car["position"] = interpolate_position(route["coordinates"], progress)
            car["grid_cell"] = grid_cell_for_position(car["position"], self.grid)
            car["status"] = "to_pickup" if car["route_elapsed"] < assignment["pickup_route"]["cost"] else "to_dropoff"
            if car["route_elapsed"] >= total_cost:
                self._complete_trip(car)

    def _map_person(self, person, status: str) -> dict:
        pickup_node = self.graph.nearest_node_for_cell(person.origin)
        dropoff_node = self.graph.nearest_node_for_cell(person.destination)
        payload = person.to_dict()
        payload.update(
            {
                "status": status,
                "pickup_node_id": int(pickup_node["node_id"]),
                "dropoff_node_id": int(dropoff_node["node_id"]),
                "pickup_position": [float(pickup_node["lon"]), float(pickup_node["lat"])],
                "dropoff_position": [float(dropoff_node["lon"]), float(dropoff_node["lat"])],
                "pickup_grid_cell": [int(pickup_node["grid_row"]), int(pickup_node["grid_col"])],
                "dropoff_grid_cell": [int(dropoff_node["grid_row"]), int(dropoff_node["grid_col"])],
                "request_origin": list(person.origin),
                "request_destination": list(person.destination),
            }
        )
        return payload

    def _expire_requests(self, timestep: int) -> None:
        retained = []
        for item in self.pending:
            payload = item["payload"]
            if payload.get("status") == "assigned":
                retained.append(item)
                continue
            if timestep - int(item["created_at"]) > int(payload.get("patience", 15)):
                self.canceled_requests += 1
            else:
                retained.append(item)
        self.pending = retained

    def _assign_waiting(self, timestep: int) -> list[dict]:
        hour = (timestep // 60) % 24
        waiting = [item for item in self.pending if item["payload"].get("status") != "assigned"]
        idle_cars = [car for car in self.cars if car["status"] == "idle" and car.get("assignment") is None]
        assignments = []

        for item in waiting:
            if not idle_cars:
                break
            payload = item["payload"]
            pickup_node_id = int(payload["pickup_node_id"])
            dropoff_node_id = int(payload["dropoff_node_id"])
            pickup_position = payload["pickup_position"]
            candidates = sorted(
                enumerate(idle_cars),
                key=lambda pair: distance_m(pair[1]["position"], pickup_position),
            )[: self.candidate_limit]

            best = None
            for idle_idx, car in candidates:
                route_to_pickup = self.graph.route(int(car["node_id"]), pickup_node_id, hour)
                cost = float(route_to_pickup["cost"])
                if best is None or cost < best[0]:
                    best = (cost, idle_idx, car, route_to_pickup)
            if best is None:
                continue

            _, idle_idx, car, pickup_route = best
            idle_cars.pop(idle_idx)
            dropoff_route = self.graph.route(pickup_node_id, dropoff_node_id, hour)
            full_coords = merge_route_coordinates(pickup_route, dropoff_route)
            full_cost = float(pickup_route["cost"]) + float(dropoff_route["cost"])
            wait_minutes = max(0.0, float(timestep - int(item["created_at"]))) + float(pickup_route["cost"]) / 60.0
            assignment = {
                "car_id": car["id"],
                "person_id": payload["id"],
                "assigned_at": timestep,
                "request_created_at": int(item["created_at"]),
                "wait_time_min": round(wait_minutes, 2),
                "pickup_node_id": pickup_node_id,
                "dropoff_node_id": dropoff_node_id,
                "pickup_position": payload["pickup_position"],
                "dropoff_position": payload["dropoff_position"],
                "pickup_grid_cell": payload["pickup_grid_cell"],
                "dropoff_grid_cell": payload["dropoff_grid_cell"],
                "pickup_route": pickup_route,
                "dropoff_route": dropoff_route,
                "route": {
                    "coordinates": full_coords,
                    "cost": round(full_cost, 6),
                    "fallback": bool(pickup_route.get("fallback")) or bool(dropoff_route.get("fallback")),
                },
                "total_cost": round(full_cost, 6),
            }
            payload["status"] = "assigned"
            car.update(
                {
                    "status": "to_pickup",
                    "assigned_person_id": payload["id"],
                    "pickup_node_id": pickup_node_id,
                    "dropoff_node_id": dropoff_node_id,
                    "stall_ticks": 0,
                    "assignment": assignment,
                    "route_elapsed": 0.0,
                }
            )
            self.total_wait_minutes += wait_minutes
            assignments.append(assignment)

        return assignments

    def _queued_customer_wait(self, timestep: int) -> float:
        hour = (timestep // 60) % 24
        waiting = [item for item in self.pending if item["payload"].get("status") != "assigned"]
        if not waiting:
            return 0.0

        total_wait = 0.0
        car_eta_cache: dict[tuple[str, int], float] = {}
        for item in waiting:
            payload = item["payload"]
            pickup_node_id = int(payload["pickup_node_id"])
            request_age = max(0.0, float(timestep - int(item["created_at"])))
            best_eta = None
            for car in self.cars:
                cache_key = (car["id"], pickup_node_id)
                if cache_key in car_eta_cache:
                    eta = car_eta_cache[cache_key]
                else:
                    assignment = car.get("assignment")
                    if assignment is None:
                        eta = float(self.graph.route(int(car["node_id"]), pickup_node_id, hour)["cost"]) / 60.0
                    else:
                        route = assignment["route"]
                        remaining_current = max(
                            0.0,
                            float(route["cost"]) - float(car.get("route_elapsed", 0.0)),
                        ) / 60.0
                        reposition = float(
                            self.graph.route(int(assignment["dropoff_node_id"]), pickup_node_id, hour)["cost"]
                        ) / 60.0
                        eta = remaining_current + reposition
                    car_eta_cache[cache_key] = eta
                if best_eta is None or eta < best_eta:
                    best_eta = eta
            total_wait += request_age + (best_eta if best_eta is not None else 15.0)
        return total_wait

    def _complete_trip(self, car: dict) -> None:
        assignment = car["assignment"]
        car["position"] = assignment["dropoff_position"]
        car["grid_cell"] = assignment["dropoff_grid_cell"]
        car["node_id"] = assignment["dropoff_node_id"]
        car["status"] = "idle"
        car["assigned_person_id"] = None
        car["assignment"] = None
        car["route_elapsed"] = 0.0
        car["stall_ticks"] = 0
        self.completed_trips += 1
        person_id = assignment["person_id"]
        for item in list(self.pending):
            if item["payload"]["id"] == person_id:
                self.revenue += float(item["payload"].get("value", 0.0))
                self.pending.remove(item)
                break

    def _car_payload(self, car: dict) -> dict:
        return {
            "id": car["id"],
            "node_id": car["node_id"],
            "position": car["position"],
            "grid_cell": car.get("grid_cell"),
            "status": car["status"],
            "assigned_person_id": car.get("assigned_person_id"),
            "pickup_node_id": car.get("pickup_node_id"),
            "dropoff_node_id": car.get("dropoff_node_id"),
            "stall_ticks": car["stall_ticks"],
            "route_elapsed": round(float(car.get("route_elapsed", 0.0)), 6),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Export greedy mobility snapshots for the map UI.")
    parser.add_argument("--grid", default="public/data/population_density_grid.json")
    parser.add_argument("--nodes", default="public/data/ppo_nodes.json")
    parser.add_argument("--edges", default="public/data/ppo_edges.json")
    parser.add_argument("--edges-geojson", default="public/data/osmnx_edges.geojson")
    parser.add_argument("--out", default="public/data/mobility_world.json")
    parser.add_argument("--seed", default=7, type=int)
    parser.add_argument("--fleet-size", default=40, type=int)
    parser.add_argument("--step-minutes", default=15, type=int)
    args = parser.parse_args()

    grid = load_grid(Path(args.grid))
    graph = MapGraph(
        load_json(Path(args.nodes)),
        load_json(Path(args.edges)),
        load_json(Path(args.edges_geojson)),
    )
    dispatcher = StatefulMapDispatch(
        graph,
        grid,
        seed=args.seed,
        fleet_size=args.fleet_size,
        candidate_limit=6,
    )
    snapshots = []
    step_seconds = max(1, args.step_minutes) * 60
    for timestep in range(0, 24 * 60, max(1, args.step_minutes)):
        demand = DemandGenerator(grid=grid, seed=args.seed + timestep + 1)
        traffic = TrafficGenerator(grid=grid, seed=args.seed + timestep + 2)
        people_generator = PeopleGenerator(
            grid=grid,
            seed=args.seed + timestep + 3,
            base_arrival_rate=2.8,
            max_new_people_per_tick=6,
        )
        demand_heatmap = demand.get_heatmap(timestep)
        traffic_heatmap = traffic.get_heatmap(timestep, demand_heatmap)
        people = people_generator.generate(timestep, demand_heatmap, traffic_heatmap)
        map_payload = dispatcher.step(timestep, people)
        greedy_stats = map_payload["map_greedy_stats"]
        snapshot = {
            "timestep": timestep,
            "demand_heatmap": demand_heatmap,
            "traffic_heatmap": traffic_heatmap,
            "new_people": [person.to_dict() for person in people],
            "people_grid": build_people_grid(grid, people),
            "dispatch": map_payload["map_dispatch"],
            "greedy_stats": greedy_stats,
            "summary": {
                "num_new_people": len(people),
                "top_demand_cells": demand.top_demand_cells(5, timestep),
                "traffic_bottlenecks": traffic.top_bottlenecks(5, timestep, demand_heatmap),
                "dispatch": map_payload["map_dispatch"]["summary"],
                "greedy_stats": greedy_stats,
            },
        }
        snapshot["map_people"] = map_payload["map_people"]
        snapshot["map_dispatch"] = map_payload["map_dispatch"]
        snapshot["map_greedy_stats"] = greedy_stats
        snapshots.append(snapshot)
        dispatcher.advance(step_seconds)

    payload = {
        "seed": args.seed,
        "fleet_size": args.fleet_size,
        "step_minutes": args.step_minutes,
        "snapshots": snapshots,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, separators=(",", ":"), allow_nan=False), encoding="utf-8")
    print(f"Wrote {len(snapshots)} greedy mobility snapshots to {out_path}")


if __name__ == "__main__":
    main()
