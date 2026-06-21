#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from mobility_sim import PersonRequest

from export_mobility_world import (
    EVENT_PRESETS,
    MAX_ROUTE_SEGMENT_METERS,
    MapGraph,
    StatefulMapDispatch,
    assignment_for_export,
    clamp,
    dispatch_for_export,
    distance_m,
    grid_cell_for_position,
    interpolate_position,
    load_json,
    merge_route_coordinates,
    route_is_usable,
    route_reference,
    seed_car_nodes,
    with_route_metrics,
)


ORCHESTRATOR_MAX_ROUTE_SEGMENT_METERS = 650.0


def first_existing_path(*paths: str) -> Path:
    for raw in paths:
        path = Path(raw)
        if path.exists():
            return path
    return Path(paths[0])


def person_from_payload(raw: dict[str, Any]) -> PersonRequest:
    return PersonRequest(
        id=str(raw["id"]),
        origin=tuple(raw.get("request_origin") or raw["origin"]),
        destination=tuple(raw.get("request_destination") or raw["destination"]),
        created_at=int(raw.get("created_at", raw.get("timestep", 0))),
        patience=int(raw.get("patience", 15)),
        value=float(raw.get("value", 0.0)),
        party_size=int(raw.get("party_size", 1)),
    )


def demand_lookup(summary: dict[str, Any]) -> dict[tuple[int, int], float]:
    cells = {}
    for item in summary.get("top_demand_cells", []):
        cells[(int(item["row"]), int(item["col"]))] = float(item["value"])
    return cells


def orchestrator_route_is_usable(route: dict) -> bool:
    if route.get("fallback"):
        return False
    coords = route.get("coordinates") or []
    if len(coords) < 2:
        return float(route.get("cost", 0.0)) <= 0.0
    return float(with_route_metrics(route)["_max_segment_m"]) <= ORCHESTRATOR_MAX_ROUTE_SEGMENT_METERS


class OrchestratorMapDispatch(StatefulMapDispatch):
    def __init__(
        self,
        *args,
        beam_width: int = 128,
        value_weight: float = 1.35,
        urgency_weight: float = 12.0,
        pressure_weight: float = 10.0,
        stall_weight: float = 0.18,
        pickup_weight: float = 2.35,
        trip_weight: float = 0.05,
        assignment_bonus: float = 120.0,
        max_repositions_per_tick: int = 1,
        min_reposition_stall: int = 6,
        min_reposition_distance_m: float = 1200.0,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.beam_width = int(beam_width)
        self.value_weight = float(value_weight)
        self.urgency_weight = float(urgency_weight)
        self.pressure_weight = float(pressure_weight)
        self.stall_weight = float(stall_weight)
        self.pickup_weight = float(pickup_weight)
        self.trip_weight = float(trip_weight)
        self.assignment_bonus = float(assignment_bonus)
        self.max_repositions_per_tick = int(max_repositions_per_tick)
        self.min_reposition_stall = int(min_reposition_stall)
        self.min_reposition_distance_m = float(min_reposition_distance_m)
        self.reposition_cost = 0.0
        for car in self.cars:
            car["reposition"] = None
            car["queued_assignment"] = None

    def step(self, timestep: int, people: list[PersonRequest], event: dict | None = None, source_summary: dict[str, Any] | None = None) -> dict:
        self._expire_requests(timestep)
        self.add_requests(people, timestep)
        new_assignments = self._assign_waiting(timestep, event, source_summary or {})
        new_queued_assignments = self._queue_next_assignments(timestep, event, source_summary or {})
        new_repositions = self._reposition_idle(timestep, event, source_summary or {})

        active_assignments = []
        queued_assignments = []
        active_repositions = []
        for car in self.cars:
            assignment = car.get("assignment")
            if assignment is not None:
                active_assignments.append(
                    {
                        **assignment,
                        "current_position": car["position"],
                        "current_grid_cell": car.get("grid_cell"),
                        "route_elapsed": round(float(car.get("route_elapsed", 0.0)), 6),
                    }
                )
            queued = car.get("queued_assignment")
            if queued is not None:
                queued_assignments.append(queued)
            reposition = car.get("reposition")
            if reposition is not None:
                active_repositions.append(
                    {
                        **reposition,
                        "current_position": car["position"],
                        "current_grid_cell": car.get("grid_cell"),
                        "route_elapsed": round(float(car.get("route_elapsed", 0.0)), 6),
                    }
                )

        active_people_ids = {assignment["person_id"] for assignment in active_assignments}
        accepted_people_ids = active_people_ids | {assignment["person_id"] for assignment in queued_assignments}
        map_people = [
            {**item["payload"], "status": "assigned" if item["payload"]["id"] in accepted_people_ids else "waiting"}
            for item in self.pending
        ]
        active_cars = sum(1 for car in self.cars if car["status"] != "idle")
        repositioning_cars = sum(1 for car in self.cars if car["status"] == "repositioning")
        stalled_cars = len(self.cars) - active_cars
        matched_requests = self.completed_trips + len(accepted_people_ids)
        demand_served = matched_requests / max(1, self.total_requests) * 100.0
        wait_time = self.total_wait_minutes + self._queued_customer_wait(timestep, event)
        current_utilization = active_cars / max(1, len(self.cars)) * 100.0
        self.utilization_pct_total += current_utilization
        self.utilization_samples += 1
        avg_fleet_utilization = self.utilization_pct_total / max(1, self.utilization_samples)

        stats = {
            "completed_trips": self.completed_trips,
            "revenue": round(self.revenue, 2),
            "demand_served_pct": round(demand_served, 2),
            "wait_time_min": round(wait_time, 2),
            "fleet_utilization_pct": round(current_utilization, 2),
            "avg_fleet_utilization_pct": round(avg_fleet_utilization, 2),
            "active_cars": active_cars,
            "repositioning_cars": repositioning_cars,
            "stalled_cars": stalled_cars,
            "unassigned_people": len(map_people) - len(accepted_people_ids),
            "canceled_requests": self.canceled_requests,
            "total_requests": self.total_requests,
            "reposition_cost": round(self.reposition_cost / 60.0, 2),
        }
        return {
            "map_people": map_people,
            "map_dispatch": {
                "assignments": active_assignments,
                "new_assignments": new_assignments,
                "queued_assignments": queued_assignments,
                "new_queued_assignments": new_queued_assignments,
                "repositions": active_repositions,
                "new_repositions": new_repositions,
                "cars": [self._car_payload(car) for car in self.cars],
                "summary": {
                    "num_assignments": len(active_assignments),
                    "num_new_assignments": len(new_assignments),
                    "num_queued_assignments": len(queued_assignments),
                    "num_new_queued_assignments": len(new_queued_assignments),
                    "num_repositions": len(active_repositions),
                    "num_new_repositions": len(new_repositions),
                    "num_unassigned_people": len(map_people) - len(accepted_people_ids),
                    "num_stalled_cars": stalled_cars,
                    "num_active_cars": active_cars,
                },
            },
            "map_orchestrator_stats": stats,
        }

    def advance(self, seconds: float) -> None:
        for car in self.cars:
            assignment = car.get("assignment")
            reposition = car.get("reposition")
            route = assignment["route"] if assignment is not None else reposition["route"] if reposition is not None else None
            if route is None:
                car["stall_ticks"] += 1
                continue

            total_cost = max(1.0, float(route["cost"]))
            car["route_elapsed"] = min(total_cost, float(car.get("route_elapsed", 0.0)) + float(seconds))
            progress = car["route_elapsed"] / total_cost
            car["position"] = interpolate_position(
                route["coordinates"],
                progress,
                route.get("_segment_lengths"),
                route.get("_geometry_length_m"),
            )
            car["grid_cell"] = grid_cell_for_position(car["position"], self.grid)

            if assignment is not None:
                car["status"] = "to_pickup" if car["route_elapsed"] < assignment["pickup_route"]["cost"] else "to_dropoff"
                if car["route_elapsed"] >= total_cost:
                    self._complete_trip(car)
            elif reposition is not None:
                car["status"] = "repositioning"
                if car["route_elapsed"] >= total_cost:
                    car["position"] = reposition["target_position"]
                    car["grid_cell"] = reposition["target_grid_cell"]
                    car["node_id"] = reposition["target_node_id"]
                    car["status"] = "idle"
                    car["reposition"] = None
                    car["route_elapsed"] = 0.0
                    car["stall_ticks"] = 0

    def _assign_waiting(self, timestep: int, event: dict | None = None, source_summary: dict[str, Any] | None = None) -> list[dict]:
        hour = (timestep // 60) % 24
        waiting = [item for item in self.pending if item["payload"].get("status") != "assigned"]
        idle_cars = [car for car in self.cars if car["status"] == "idle" and car.get("assignment") is None and car.get("reposition") is None]
        if not waiting or not idle_cars:
            return []

        demand_by_cell = demand_lookup(source_summary or {})
        candidate_pairs: dict[str, list[tuple[float, dict, dict, dict, dict, dict]]] = {}
        for item in waiting:
            payload = item["payload"]
            pickup_node_id = int(payload["pickup_node_id"])
            dropoff_node_id = int(payload["dropoff_node_id"])
            pickup_position = payload["pickup_position"]
            pickup_cell = tuple(payload.get("pickup_grid_cell") or payload.get("origin") or (0, 0))
            age = max(0.0, float(timestep - int(item["created_at"])))
            patience = max(1.0, float(payload.get("patience", 15)))
            urgency = clamp(age / patience)
            value = float(payload.get("value", 0.0))
            pressure = demand_by_cell.get((int(pickup_cell[0]), int(pickup_cell[1])), 0.0)
            candidates = sorted(
                idle_cars,
                key=lambda car: distance_m(car["position"], pickup_position),
            )[: self.candidate_limit]

            scored = []
            for car in candidates:
                pickup_route = self.graph.route(int(car["node_id"]), pickup_node_id, hour, event)
                if not orchestrator_route_is_usable(pickup_route):
                    continue
                dropoff_route = self.graph.route(pickup_node_id, dropoff_node_id, hour, event)
                if not orchestrator_route_is_usable(dropoff_route):
                    continue
                full_coords = merge_route_coordinates(pickup_route, dropoff_route)
                full_cost = float(pickup_route["cost"]) + float(dropoff_route["cost"])
                full_route = with_route_metrics({"coordinates": full_coords, "cost": full_cost, "fallback": False})
                if not orchestrator_route_is_usable(full_route):
                    continue
                pickup_minutes = float(pickup_route["cost"]) / 60.0
                trip_minutes = float(dropoff_route["cost"]) / 60.0
                score = (
                    self.assignment_bonus
                    + self.value_weight * value
                    + self.urgency_weight * urgency
                    + self.pressure_weight * pressure
                    + self.stall_weight * float(car.get("stall_ticks", 0))
                    - self.pickup_weight * pickup_minutes
                    - self.trip_weight * trip_minutes
                )
                scored.append((score, item, car, pickup_route, dropoff_route, full_route))
            scored.sort(key=lambda row: row[0], reverse=True)
            candidate_pairs[payload["id"]] = scored[: min(len(scored), 10)]

        ordered_waiting = sorted(
            waiting,
            key=lambda item: (
                max(0.0, float(timestep - int(item["created_at"]))) / max(1.0, float(item["payload"].get("patience", 15))),
                float(item["payload"].get("value", 0.0)),
            ),
            reverse=True,
        )
        beams: list[tuple[float, tuple, frozenset[str]]] = [(0.0, tuple(), frozenset())]
        for item in ordered_waiting:
            next_beams = list(beams)
            for total, selected, used_cars in beams:
                for candidate in candidate_pairs.get(item["payload"]["id"], []):
                    score, _item, car, _pickup_route, _dropoff_route, _full_route = candidate
                    if car["id"] in used_cars:
                        continue
                    next_beams.append((total + score, selected + (candidate,), used_cars | {car["id"]}))
            next_beams.sort(key=lambda row: (row[0], len(row[1])), reverse=True)
            beams = next_beams[: self.beam_width]

        selected = beams[0][1] if beams else tuple()
        assignments = []
        for _score, item, car, pickup_route, dropoff_route, full_route in selected:
            payload = item["payload"]
            full_coords = full_route["coordinates"]
            wait_minutes = max(0.0, float(timestep - int(item["created_at"]))) + float(pickup_route["cost"]) / 60.0
            assignment = {
                "car_id": car["id"],
                "person_id": payload["id"],
                "assigned_at": timestep,
                "request_created_at": int(item["created_at"]),
                "wait_time_min": round(wait_minutes, 2),
                "pickup_node_id": int(payload["pickup_node_id"]),
                "dropoff_node_id": int(payload["dropoff_node_id"]),
                "pickup_position": payload["pickup_position"],
                "dropoff_position": payload["dropoff_position"],
                "pickup_grid_cell": payload["pickup_grid_cell"],
                "dropoff_grid_cell": payload["dropoff_grid_cell"],
                "pickup_route": pickup_route,
                "dropoff_route": dropoff_route,
                "route": {
                    "coordinates": full_coords,
                    "cost": round(float(full_route["cost"]), 6),
                    "fallback": bool(pickup_route.get("fallback")) or bool(dropoff_route.get("fallback")),
                    "_segment_lengths": full_route["_segment_lengths"],
                    "_geometry_length_m": full_route["_geometry_length_m"],
                    "_max_segment_m": full_route["_max_segment_m"],
                },
                "total_cost": round(float(full_route["cost"]), 6),
            }
            payload["status"] = "assigned"
            car.update(
                {
                    "status": "to_pickup",
                    "assigned_person_id": payload["id"],
                    "pickup_node_id": int(payload["pickup_node_id"]),
                    "dropoff_node_id": int(payload["dropoff_node_id"]),
                    "stall_ticks": 0,
                    "assignment": assignment,
                    "reposition": None,
                    "route_elapsed": 0.0,
                }
            )
            self.total_wait_minutes += wait_minutes
            assignments.append(assignment)
        return assignments

    def _queue_next_assignments(self, timestep: int, event: dict | None, source_summary: dict[str, Any]) -> list[dict]:
        hour = (timestep // 60) % 24
        waiting = [item for item in self.pending if item["payload"].get("status") != "assigned"]
        candidate_cars = [
            car
            for car in self.cars
            if car.get("assignment") is not None
            and car.get("queued_assignment") is None
            and car.get("reposition") is None
        ]
        if not waiting or not candidate_cars:
            return []

        demand_by_cell = demand_lookup(source_summary or {})
        queued = []
        max_new = min(len(waiting), max(1, len(self.cars) // 4))
        used_cars: set[str] = set()
        ordered_waiting = sorted(
            waiting,
            key=lambda item: (
                max(0.0, float(timestep - int(item["created_at"]))) / max(1.0, float(item["payload"].get("patience", 15))),
                float(item["payload"].get("value", 0.0)),
            ),
            reverse=True,
        )

        for item in ordered_waiting:
            if len(queued) >= max_new:
                break
            payload = item["payload"]
            pickup_node_id = int(payload["pickup_node_id"])
            dropoff_node_id = int(payload["dropoff_node_id"])
            pickup_cell = tuple(payload.get("pickup_grid_cell") or payload.get("origin") or (0, 0))
            age = max(0.0, float(timestep - int(item["created_at"])))
            patience = max(1.0, float(payload.get("patience", 15)))
            pressure = demand_by_cell.get((int(pickup_cell[0]), int(pickup_cell[1])), 0.0)
            best = None

            for car in candidate_cars:
                if car["id"] in used_cars:
                    continue
                current = car.get("assignment")
                if current is None:
                    continue
                remaining_current = max(
                    0.0,
                    float(current["route"]["cost"]) - float(car.get("route_elapsed", 0.0)),
                ) / 60.0
                pickup_route = self.graph.route(int(current["dropoff_node_id"]), pickup_node_id, hour, event)
                if not orchestrator_route_is_usable(pickup_route):
                    continue
                dropoff_route = self.graph.route(pickup_node_id, dropoff_node_id, hour, event)
                if not orchestrator_route_is_usable(dropoff_route):
                    continue
                full_coords = merge_route_coordinates(pickup_route, dropoff_route)
                full_cost = float(pickup_route["cost"]) + float(dropoff_route["cost"])
                full_route = with_route_metrics({"coordinates": full_coords, "cost": full_cost, "fallback": False})
                if not orchestrator_route_is_usable(full_route):
                    continue
                pickup_minutes = float(pickup_route["cost"]) / 60.0
                trip_minutes = float(dropoff_route["cost"]) / 60.0
                predicted_wait = age + remaining_current + pickup_minutes
                if predicted_wait > patience + 120.0:
                    continue
                score = (
                    180.0
                    + self.value_weight * float(payload.get("value", 0.0))
                    + self.urgency_weight * clamp(age / patience)
                    + self.pressure_weight * pressure
                    - 1.8 * predicted_wait
                    - self.trip_weight * trip_minutes
                )
                if best is None or score > best[0]:
                    best = (score, car, pickup_route, dropoff_route, full_route, predicted_wait)

            if best is None:
                continue

            _score, car, pickup_route, dropoff_route, full_route, predicted_wait = best
            assignment = {
                "car_id": car["id"],
                "person_id": payload["id"],
                "assigned_at": timestep,
                "request_created_at": int(item["created_at"]),
                "wait_time_min": round(float(predicted_wait), 2),
                "pickup_node_id": pickup_node_id,
                "dropoff_node_id": dropoff_node_id,
                "pickup_position": payload["pickup_position"],
                "dropoff_position": payload["dropoff_position"],
                "pickup_grid_cell": payload["pickup_grid_cell"],
                "dropoff_grid_cell": payload["dropoff_grid_cell"],
                "pickup_route": pickup_route,
                "dropoff_route": dropoff_route,
                "route": {
                    "coordinates": full_route["coordinates"],
                    "cost": round(float(full_route["cost"]), 6),
                    "fallback": bool(pickup_route.get("fallback")) or bool(dropoff_route.get("fallback")),
                    "_segment_lengths": full_route["_segment_lengths"],
                    "_geometry_length_m": full_route["_geometry_length_m"],
                    "_max_segment_m": full_route["_max_segment_m"],
                },
                "total_cost": round(float(full_route["cost"]), 6),
                "queued_after_person_id": car.get("assigned_person_id"),
            }
            payload["status"] = "assigned"
            car["queued_assignment"] = assignment
            self.total_wait_minutes += float(predicted_wait)
            used_cars.add(car["id"])
            queued.append(assignment)
        return queued

    def _reposition_idle(self, timestep: int, event: dict | None, source_summary: dict[str, Any]) -> list[dict]:
        hour = (timestep // 60) % 24
        idle = [car for car in self.cars if car["status"] == "idle" and car.get("assignment") is None and car.get("reposition") is None]
        if not idle:
            return []

        hot_cells = [
            (int(item["row"]), int(item["col"]), float(item["value"]))
            for item in source_summary.get("top_demand_cells", [])
        ]
        if not hot_cells:
            return []

        idle = [car for car in idle if int(car.get("stall_ticks", 0)) >= self.min_reposition_stall]
        if not idle:
            return []

        max_repositions = min(len(idle), max(0, self.max_repositions_per_tick))
        if max_repositions <= 0:
            return []
        target_load: dict[tuple[int, int], int] = {}
        repositions = []
        for car in sorted(idle, key=lambda item: item.get("stall_ticks", 0), reverse=True):
            if len(repositions) >= max_repositions:
                break
            best = None
            for row, col, pressure in hot_cells[:12]:
                target = (row, col)
                load = target_load.get(target, 0)
                if load >= 3:
                    continue
                target_node = self.graph.nearest_node_for_cell(target)
                target_position = [float(target_node["lon"]), float(target_node["lat"])]
                distance = distance_m(car["position"], target_position)
                if distance < self.min_reposition_distance_m:
                    continue
                score = pressure / (1.0 + distance / 1800.0 + 0.6 * load)
                if best is None or score > best[0]:
                    best = (score, target, target_node, target_position)
            if best is None:
                continue
            _score, target, target_node, target_position = best
            route = self.graph.route(int(car["node_id"]), int(target_node["node_id"]), hour, event)
            if not orchestrator_route_is_usable(route) or float(route.get("cost", 0.0)) <= 0:
                continue
            reposition = {
                "car_id": car["id"],
                "assigned_at": timestep,
                "target_node_id": int(target_node["node_id"]),
                "target_grid_cell": [int(target_node["grid_row"]), int(target_node["grid_col"])],
                "target_position": target_position,
                "route": route,
                "total_cost": round(float(route["cost"]), 6),
            }
            car.update(
                {
                    "status": "repositioning",
                    "assigned_person_id": None,
                    "pickup_node_id": None,
                    "dropoff_node_id": None,
                    "stall_ticks": 0,
                    "assignment": None,
                    "reposition": reposition,
                    "route_elapsed": 0.0,
                }
            )
            self.reposition_cost += float(route["cost"])
            target_load[target] = target_load.get(target, 0) + 1
            repositions.append(reposition)
        return repositions

    def _complete_trip(self, car: dict) -> None:
        super()._complete_trip(car)
        queued = car.get("queued_assignment")
        car["reposition"] = None
        if queued is None:
            return
        car.update(
            {
                "status": "to_pickup",
                "assigned_person_id": queued["person_id"],
                "pickup_node_id": queued["pickup_node_id"],
                "dropoff_node_id": queued["dropoff_node_id"],
                "stall_ticks": 0,
                "assignment": queued,
                "queued_assignment": None,
                "route_elapsed": 0.0,
            }
        )


def reposition_for_export(reposition: dict[str, Any], route_registry: dict[str, dict]) -> dict[str, Any]:
    exported = {key: value for key, value in reposition.items() if key != "route"}
    if reposition.get("route"):
        exported["route"] = route_reference(reposition["route"], route_registry)
    return exported


def dispatch_for_orchestrator_export(dispatch: dict[str, Any], route_registry: dict[str, dict]) -> dict[str, Any]:
    exported = dispatch_for_export(dispatch, route_registry)
    exported["repositions"] = [
        reposition_for_export(item, route_registry)
        for item in dispatch.get("repositions", [])
    ]
    exported["new_repositions"] = [
        reposition_for_export(item, route_registry)
        for item in dispatch.get("new_repositions", [])
    ]
    exported["queued_assignments"] = [
        assignment_for_export(item, route_registry)
        for item in dispatch.get("queued_assignments", [])
    ]
    exported["new_queued_assignments"] = [
        assignment_for_export(item, route_registry)
        for item in dispatch.get("new_queued_assignments", [])
    ]
    return exported


def build_orchestrator_snapshots(
    source_snapshots: list[dict[str, Any]],
    graph: MapGraph,
    grid: dict,
    seed: int,
    fleet_size: int,
    step_minutes: int,
    route_registry: dict[str, dict],
    event: dict | None,
    candidate_limit: int,
    beam_width: int,
    policy_args: argparse.Namespace,
) -> list[dict[str, Any]]:
    dispatcher = OrchestratorMapDispatch(
        graph,
        grid,
        seed=seed,
        fleet_size=fleet_size,
        candidate_limit=candidate_limit,
        beam_width=beam_width,
        value_weight=policy_args.value_weight,
        urgency_weight=policy_args.urgency_weight,
        pressure_weight=policy_args.pressure_weight,
        stall_weight=policy_args.stall_weight,
        pickup_weight=policy_args.pickup_weight,
        trip_weight=policy_args.trip_weight,
        assignment_bonus=policy_args.assignment_bonus,
        max_repositions_per_tick=policy_args.max_repositions_per_tick,
        min_reposition_stall=policy_args.min_reposition_stall,
        min_reposition_distance_m=policy_args.min_reposition_distance_m,
    )
    snapshots = []
    step_seconds = step_minutes * 60
    for source in source_snapshots:
        timestep = int(source["timestep"])
        people = [person_from_payload(item) for item in source.get("new_people", [])]
        payload = dispatcher.step(timestep, people, event=event, source_summary=source.get("summary", {}))
        dispatch = dispatch_for_orchestrator_export(payload["map_dispatch"], route_registry)
        stats = payload["map_orchestrator_stats"]
        snapshots.append(
            {
                "timestep": timestep,
                "new_people": source.get("new_people", []),
                "summary": {
                    "num_new_people": len(people),
                    "top_demand_cells": source.get("summary", {}).get("top_demand_cells", []),
                    "traffic_bottlenecks": source.get("summary", {}).get("traffic_bottlenecks", []),
                    "dispatch": dispatch["summary"],
                    "orchestrator_stats": stats,
                },
                "map_dispatch": dispatch,
                "map_people": payload["map_people"],
                "map_orchestrator_stats": stats,
            }
        )
        dispatcher.advance(step_seconds)
    return snapshots


def build_world(args: argparse.Namespace) -> dict[str, Any]:
    source = load_json(Path(args.greedy_world))
    graph = MapGraph(
        load_json(first_existing_path(args.nodes, "dist/data/ppo_nodes.json")),
        load_json(first_existing_path(args.edges, "dist/data/ppo_edges.json")),
        load_json(first_existing_path(args.edges_geojson, "dist/data/osmnx_edges.geojson")),
    )
    grid = load_json(first_existing_path(args.grid, "dist/data/population_density_grid.json"))
    route_registry: dict[str, dict] = {}
    seed = int(source.get("seed", args.seed))
    fleet_size = int(source.get("fleet_size", args.fleet_size))
    step_minutes = int(source.get("step_minutes", args.step_minutes))

    print("Building base orchestrator comparison snapshots...")
    snapshots = build_orchestrator_snapshots(
        source.get("snapshots", []),
        graph,
        grid,
        seed,
        fleet_size,
        step_minutes,
        route_registry,
        event=None,
        candidate_limit=args.candidate_limit,
        beam_width=args.beam_width,
        policy_args=args,
    )
    event_by_id = {event["id"]: event for event in EVENT_PRESETS}
    event_scenarios = {}
    if args.include_events:
        for event_id, scenario in source.get("event_scenarios", {}).items():
            event = event_by_id.get(event_id)
            scenario_step = int(scenario.get("step_minutes", step_minutes))
            print(f"Building event orchestrator comparison snapshots: {event_id}...")
            event_scenarios[event_id] = {
                "step_minutes": scenario_step,
                "snapshots": build_orchestrator_snapshots(
                    scenario.get("snapshots", []),
                    graph,
                    grid,
                    seed,
                    fleet_size,
                    scenario_step,
                    route_registry,
                    event=event,
                    candidate_limit=args.candidate_limit,
                    beam_width=args.beam_width,
                    policy_args=args,
                ),
            }

    return {
        "seed": seed,
        "fleet_size": fleet_size,
        "step_minutes": step_minutes,
        "route_segment_max_meters": MAX_ROUTE_SEGMENT_METERS,
        "policy": "value_aware_orchestrator",
        "events": source.get("events", EVENT_PRESETS),
        "routes": route_registry,
        "snapshots": snapshots,
        "event_scenarios": event_scenarios,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Precompute value-aware orchestrator snapshots for the comparison page.")
    parser.add_argument("--greedy-world", default="public/data/mobility_world.json")
    parser.add_argument("--grid", default="public/data/population_density_grid.json")
    parser.add_argument("--nodes", default="public/data/ppo_nodes.json")
    parser.add_argument("--edges", default="public/data/ppo_edges.json")
    parser.add_argument("--edges-geojson", default="public/data/osmnx_edges.geojson")
    parser.add_argument("--out", default="public/data/mobility_orchestrator_world.json")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--fleet-size", type=int, default=40)
    parser.add_argument("--step-minutes", type=int, default=5)
    parser.add_argument("--candidate-limit", type=int, default=5)
    parser.add_argument("--beam-width", type=int, default=96)
    parser.add_argument("--value-weight", type=float, default=1.35)
    parser.add_argument("--urgency-weight", type=float, default=12.0)
    parser.add_argument("--pressure-weight", type=float, default=10.0)
    parser.add_argument("--stall-weight", type=float, default=0.18)
    parser.add_argument("--pickup-weight", type=float, default=2.35)
    parser.add_argument("--trip-weight", type=float, default=0.05)
    parser.add_argument("--assignment-bonus", type=float, default=120.0)
    parser.add_argument("--max-repositions-per-tick", type=int, default=1)
    parser.add_argument("--min-reposition-stall", type=int, default=6)
    parser.add_argument("--min-reposition-distance-m", type=float, default=1200.0)
    parser.add_argument("--include-events", action="store_true")
    args = parser.parse_args()

    world = build_world(args)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(world, separators=(",", ":"), allow_nan=False), encoding="utf-8")
    print(f"Wrote {out} with {len(world.get('snapshots', []))} base snapshots and {len(world.get('routes', {}))} routes.")


if __name__ == "__main__":
    main()
