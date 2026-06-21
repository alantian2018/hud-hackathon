from __future__ import annotations

from dataclasses import dataclass, field
import math
import random
from typing import Any

from mobility_sim import DemandGenerator, GridRouter, GridSpec, PeopleGenerator, PersonRequest, TrafficGenerator

from .schemas import ActionPlan


EVENT_PRESETS: dict[str, dict[str, float | tuple[float, float]]] = {
    "chase_center_exit": {
        "center": (0.36, 0.76),
        "sigma": 0.13,
        "demand_boost": 0.95,
        "arrival_boost": 1.05,
        "fare_boost": 0.18,
    },
    "market_st_surge": {
        "center": (0.55, 0.54),
        "sigma": 0.11,
        "demand_boost": 0.72,
        "arrival_boost": 0.62,
        "fare_boost": 0.08,
    },
    "fidi_conference": {
        "center": (0.68, 0.63),
        "sigma": 0.10,
        "demand_boost": 0.85,
        "arrival_boost": 0.75,
        "fare_boost": 0.14,
    },
}


@dataclass
class FleetCar:
    id: str
    position: tuple[int, int]
    status: str = "idle"
    route: list[tuple[int, int]] = field(default_factory=list)
    assigned_person_id: str | None = None
    pickup: tuple[int, int] | None = None
    dropoff: tuple[int, int] | None = None
    stall_ticks: int = 0
    route_started_at: int | None = None
    pickup_cost: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "position": list(self.position),
            "status": self.status,
            "route_remaining": len(self.route),
            "assigned_person_id": self.assigned_person_id,
            "pickup": list(self.pickup) if self.pickup else None,
            "dropoff": list(self.dropoff) if self.dropoff else None,
            "stall_ticks": self.stall_ticks,
        }


@dataclass
class PendingRequest:
    person: PersonRequest
    status: str = "waiting"
    assigned_car_id: str | None = None
    assigned_at: int | None = None
    pickup_cost: float = 0.0

    def to_dict(self, now: int) -> dict[str, Any]:
        age = max(0, now - self.person.created_at)
        return {
            **self.person.to_dict(),
            "status": self.status,
            "assigned_car_id": self.assigned_car_id,
            "assigned_at": self.assigned_at,
            "pickup_cost": round(self.pickup_cost, 4),
            "age": age,
            "patience_remaining": max(0, self.person.patience - age),
        }


def _clip(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _manhattan(a: tuple[int, int], b: tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _mean(values: list[float], default: float = 0.0) -> float:
    return sum(values) / len(values) if values else default


class MobilityWorld:
    """A deterministic fleet-control world for HUD rollouts.

    The world exposes enough state for an LLM orchestrator to choose global
    assignments and proactive repositioning actions. It intentionally does not
    import or call the baseline dispatcher used by the demo UI.
    """

    def __init__(
        self,
        grid: Any = (14, 14),
        *,
        seed: int = 7,
        fleet_size: int = 20,
        start_minute: int = 8 * 60,
        step_minutes: int = 5,
        horizon_steps: int = 12,
        event_id: str | None = None,
        demand_scale: float = 1.0,
        traffic_weight: float = 4.0,
    ) -> None:
        self.grid = GridSpec.from_grid(grid)
        self.seed = int(seed)
        self.fleet_size = int(fleet_size)
        self.start_minute = int(start_minute)
        self.timestep = int(start_minute)
        self.step_minutes = int(step_minutes)
        self.horizon_steps = int(horizon_steps)
        self.current_step = 0
        self.event_id = event_id if event_id in EVENT_PRESETS else None
        self.demand_scale = float(demand_scale)
        self.router = GridRouter(self.grid, traffic_weight=traffic_weight)
        event = EVENT_PRESETS.get(self.event_id or "")
        arrival_boost = float(event.get("arrival_boost", 0.0)) if event else 0.0
        self.demand_generator = DemandGenerator(self.grid, seed=self.seed + 11)
        self.traffic_generator = TrafficGenerator(self.grid, seed=self.seed + 19, demand_coupling=0.2 if event else 0.12)
        self.people_generator = PeopleGenerator(
            self.grid,
            seed=self.seed + 29,
            base_arrival_rate=max(1.0, 4.0 * self.demand_scale * (1.0 + arrival_boost)),
            max_new_people_per_tick=max(3, int(9 * self.demand_scale * (1.0 + arrival_boost))),
            min_trip_distance=max(3, min(20, (self.grid.rows + self.grid.cols) // 2)),
        )
        self.cars = self._seed_cars()
        self.pending: list[PendingRequest] = []
        self.last_demand_heatmap: list[list[float]] = []
        self.last_traffic_heatmap: list[list[float]] = []
        self._prepared_timestep: int | None = None
        self.history: list[dict[str, Any]] = []
        self.done = False

        self.total_requests = 0
        self.total_possible_value = 0.0
        self.total_direct_trip_cost = 0.0
        self.completed_trips = 0
        self.canceled_requests = 0
        self.revenue = 0.0
        self.total_wait_minutes = 0.0
        self.total_pickup_cost = 0.0
        self.deadhead_cost = 0.0
        self.invalid_actions = 0
        self.utilization_samples: list[float] = []
        self.alignment_samples: list[float] = []

    def _seed_cars(self) -> list[FleetCar]:
        rng = random.Random(self.seed + 503)
        cars = []
        used: set[tuple[int, int]] = set()
        for idx in range(max(0, self.fleet_size)):
            for _ in range(30):
                cell = (rng.randrange(self.grid.rows), rng.randrange(self.grid.cols))
                if cell not in used or len(used) >= self.grid.rows * self.grid.cols:
                    break
            used.add(cell)
            cars.append(FleetCar(id=f"car-{idx}", position=cell))
        return cars

    def _event_pressure(self, row: int, col: int) -> float:
        event = EVENT_PRESETS.get(self.event_id or "")
        if not event:
            return 0.0
        center = event["center"]
        sigma = max(1e-6, float(event["sigma"]))
        row_frac = row / max(1, self.grid.rows - 1)
        col_frac = col / max(1, self.grid.cols - 1)
        dist2 = (row_frac - float(center[0])) ** 2 + (col_frac - float(center[1])) ** 2
        return math.exp(-dist2 / (2 * sigma * sigma))

    def demand_heatmap(self, timestep: int | None = None) -> list[list[float]]:
        ts = self.timestep if timestep is None else int(timestep)
        raw = self.demand_generator.get_heatmap(ts)
        event = EVENT_PRESETS.get(self.event_id or "")
        boost = float(event.get("demand_boost", 0.0)) if event else 0.0
        scaled: list[list[float]] = []
        for row in range(self.grid.rows):
            out_row = []
            for col in range(self.grid.cols):
                value = raw[row][col] * self.demand_scale + boost * self._event_pressure(row, col)
                out_row.append(round(_clip(value), 6))
            scaled.append(out_row)
        return scaled

    def forecast_hotspots(self, lookahead_steps: int = 3, k: int = 8) -> list[dict[str, Any]]:
        aggregate: dict[tuple[int, int], float] = {}
        for offset in range(max(1, lookahead_steps)):
            ts = self.timestep + (offset + 1) * self.step_minutes
            heatmap = self.demand_heatmap(ts)
            discount = 1.0 / (1.0 + 0.35 * offset)
            for row in range(self.grid.rows):
                for col in range(self.grid.cols):
                    aggregate[(row, col)] = aggregate.get((row, col), 0.0) + heatmap[row][col] * discount
        ranked = sorted(aggregate.items(), key=lambda item: item[1], reverse=True)
        return [
            {"cell": [row, col], "score": round(score, 6)}
            for (row, col), score in ranked[: max(0, k)]
        ]

    def step(self, plan: ActionPlan | dict[str, Any] | str | None = None) -> dict[str, Any]:
        if self.done:
            return self.observe()

        action_plan = ActionPlan.from_any(plan)
        self.prepare_current_step()
        self._apply_action_plan(action_plan)
        self._sample_episode_health()
        self.history.append({"timestep": self.timestep, "plan": action_plan.to_dict(), "metrics": self.metrics()})
        self._advance_cars()
        self.current_step += 1
        self.timestep += self.step_minutes
        self.done = self.current_step >= self.horizon_steps
        self._prepared_timestep = None
        return self.observe()

    def run_policy(self, policy) -> dict[str, Any]:
        while not self.done:
            self.step(policy(self))
        return self.reward()

    def _add_new_requests(self) -> None:
        people = self.people_generator.generate(self.timestep, self.last_demand_heatmap, self.last_traffic_heatmap)
        event = EVENT_PRESETS.get(self.event_id or "")
        fare_boost = float(event.get("fare_boost", 0.0)) if event else 0.0
        for person in people:
            adjusted = person
            if fare_boost:
                adjusted = PersonRequest(
                    id=person.id,
                    origin=person.origin,
                    destination=person.destination,
                    created_at=person.created_at,
                    patience=max(5, int(person.patience * 0.9)),
                    value=round(person.value * (1.0 + fare_boost), 2),
                    party_size=person.party_size,
                )
            direct = self.router.route(adjusted.origin, adjusted.destination, self.last_traffic_heatmap)
            self.pending.append(PendingRequest(person=adjusted))
            self.total_requests += 1
            self.total_possible_value += float(adjusted.value)
            if math.isfinite(direct.cost):
                self.total_direct_trip_cost += float(direct.cost)

    def prepare_current_step(self) -> None:
        if self.done or self._prepared_timestep == self.timestep:
            return
        self.last_demand_heatmap = self.demand_heatmap()
        self.last_traffic_heatmap = self.traffic_generator.get_heatmap(self.timestep, self.last_demand_heatmap)
        self._add_new_requests()
        self._expire_requests()
        self._prepared_timestep = self.timestep

    def _expire_requests(self) -> None:
        retained = []
        for item in self.pending:
            if item.status == "assigned":
                retained.append(item)
                continue
            age = max(0, self.timestep - item.person.created_at)
            if age > item.person.patience:
                self.canceled_requests += 1
            else:
                retained.append(item)
        self.pending = retained

    def _advance_cars(self) -> None:
        for car in self.cars:
            if not car.route:
                if car.status == "repositioning":
                    car.status = "idle"
                    car.route_started_at = None
                if car.status == "idle":
                    car.stall_ticks += 1
                continue

            car.position = car.route.pop(0)
            car.stall_ticks = 0
            if car.status == "to_pickup" and car.pickup and car.position == car.pickup:
                car.status = "to_dropoff"
            if not car.route:
                if car.status == "to_dropoff":
                    self._complete_trip(car)
                elif car.status == "repositioning":
                    car.status = "idle"
                    car.route_started_at = None

    def _complete_trip(self, car: FleetCar) -> None:
        person_id = car.assigned_person_id
        if person_id:
            for item in list(self.pending):
                if item.person.id == person_id:
                    self.completed_trips += 1
                    self.revenue += float(item.person.value)
                    if item.assigned_at is not None:
                        wait = max(0, item.assigned_at - item.person.created_at) + item.pickup_cost
                        self.total_wait_minutes += wait
                    self.pending.remove(item)
                    break
        car.status = "idle"
        car.assigned_person_id = None
        car.pickup = None
        car.dropoff = None
        car.route_started_at = None
        car.pickup_cost = 0.0

    def _apply_action_plan(self, plan: ActionPlan) -> None:
        used_cars: set[str] = set()
        used_people: set[str] = set()

        for action in plan.assignments:
            if action.car_id in used_cars or action.person_id in used_people:
                self.invalid_actions += 1
                continue
            car = self.car_by_id(action.car_id)
            request = self.request_by_id(action.person_id)
            if car is None or request is None or car.status != "idle" or request.status != "waiting":
                self.invalid_actions += 1
                continue
            pickup_route = self.router.route(car.position, request.person.origin, self.last_traffic_heatmap)
            dropoff_route = self.router.route(request.person.origin, request.person.destination, self.last_traffic_heatmap)
            if not math.isfinite(pickup_route.cost) or not math.isfinite(dropoff_route.cost):
                self.invalid_actions += 1
                continue
            car.assigned_person_id = request.person.id
            car.pickup = request.person.origin
            car.dropoff = request.person.destination
            car.pickup_cost = float(pickup_route.cost)
            car.route_started_at = self.timestep
            car.route = pickup_route.path[1:] + dropoff_route.path[1:]
            car.status = "to_pickup" if pickup_route.path[1:] else "to_dropoff"
            car.stall_ticks = 0
            request.status = "assigned"
            request.assigned_car_id = car.id
            request.assigned_at = self.timestep
            request.pickup_cost = float(pickup_route.cost)
            self.total_pickup_cost += float(pickup_route.cost)
            self.deadhead_cost += float(pickup_route.cost)
            used_cars.add(car.id)
            used_people.add(request.person.id)

        for action in plan.repositions:
            if action.car_id in used_cars:
                continue
            car = self.car_by_id(action.car_id)
            if car is None or car.status != "idle" or not self.grid.contains(action.target):
                self.invalid_actions += 1
                continue
            route = self.router.route(car.position, action.target, self.last_traffic_heatmap)
            if not math.isfinite(route.cost):
                self.invalid_actions += 1
                continue
            if len(route.path) <= 1:
                continue
            car.route = route.path[1:]
            car.status = "repositioning"
            car.route_started_at = self.timestep
            car.stall_ticks = 0
            self.deadhead_cost += float(route.cost)
            used_cars.add(car.id)

    def _sample_episode_health(self) -> None:
        active = sum(1 for car in self.cars if car.status != "idle")
        self.utilization_samples.append(active / max(1, len(self.cars)))
        self.alignment_samples.append(self.supply_alignment_score())

    def supply_alignment_score(self) -> float:
        hot = self.forecast_hotspots(lookahead_steps=3, k=8)
        idle_positions = [
            car.position
            for car in self.cars
            if car.status == "idle" or car.status == "repositioning"
        ]
        if not hot or not idle_positions:
            return 0.0
        diag = max(1.0, math.hypot(self.grid.rows - 1, self.grid.cols - 1))
        weighted = 0.0
        total = 0.0
        for item in hot:
            cell = (int(item["cell"][0]), int(item["cell"][1]))
            weight = max(0.001, float(item["score"]))
            distance = min(_manhattan(cell, pos) for pos in idle_positions)
            weighted += weight * _clip(1.0 - distance / diag)
            total += weight
        return _clip(weighted / max(1e-9, total))

    def reward(self) -> dict[str, Any]:
        active_items = [item for item in self.pending if item.status == "assigned"]
        waiting_items = [item for item in self.pending if item.status == "waiting"]
        active_value = sum(float(item.person.value) for item in active_items)
        captured_value = self.revenue + 0.65 * active_value
        revenue_capture = _clip(captured_value / max(1.0, self.total_possible_value))
        accepted = self.completed_trips + len(active_items)
        demand_served = _clip(accepted / max(1, self.total_requests))
        active_wait = [
            max(0, (item.assigned_at or self.timestep) - item.person.created_at) + item.pickup_cost
            for item in active_items
        ]
        queued_wait = [
            max(0, self.timestep - item.person.created_at) + 10.0
            for item in waiting_items
        ]
        mean_wait = (
            self.total_wait_minutes + sum(active_wait) + sum(queued_wait)
        ) / max(1, self.completed_trips + len(active_items) + len(waiting_items))
        wait_score = _clip(1.0 - mean_wait / 30.0)
        avg_utilization = _mean(self.utilization_samples)
        productive_utilization = _clip(avg_utilization / 0.55)
        supply_alignment = _mean(self.alignment_samples)
        deadhead_penalty = _clip(
            self.deadhead_cost / max(1.0, self.total_direct_trip_cost + self.total_pickup_cost),
            0.0,
            1.0,
        )
        invalid_penalty = min(0.25, 0.035 * self.invalid_actions)
        cancellation_penalty = _clip(self.canceled_requests / max(1, self.total_requests))

        reward = (
            0.35 * revenue_capture
            + 0.23 * demand_served
            + 0.18 * wait_score
            + 0.12 * productive_utilization
            + 0.12 * supply_alignment
            - 0.08 * deadhead_penalty
            - 0.10 * cancellation_penalty
            - invalid_penalty
        )
        reward = _clip(reward)
        return {
            "reward": round(reward, 6),
            "components": {
                "revenue_capture": round(revenue_capture, 6),
                "demand_served": round(demand_served, 6),
                "wait_score": round(wait_score, 6),
                "productive_utilization": round(productive_utilization, 6),
                "supply_alignment": round(supply_alignment, 6),
                "deadhead_penalty": round(deadhead_penalty, 6),
                "cancellation_penalty": round(cancellation_penalty, 6),
                "invalid_penalty": round(invalid_penalty, 6),
            },
            "metrics": self.metrics(),
        }

    def metrics(self) -> dict[str, Any]:
        active_cars = sum(1 for car in self.cars if car.status != "idle")
        waiting = sum(1 for item in self.pending if item.status == "waiting")
        active_requests = sum(1 for item in self.pending if item.status == "assigned")
        return {
            "timestep": self.timestep,
            "step": self.current_step,
            "horizon_steps": self.horizon_steps,
            "total_requests": self.total_requests,
            "completed_trips": self.completed_trips,
            "active_requests": active_requests,
            "waiting_requests": waiting,
            "canceled_requests": self.canceled_requests,
            "revenue": round(self.revenue, 2),
            "total_possible_value": round(self.total_possible_value, 2),
            "avg_fleet_utilization_pct": round(_mean(self.utilization_samples) * 100.0, 2),
            "active_cars": active_cars,
            "idle_cars": len(self.cars) - active_cars,
            "deadhead_cost": round(self.deadhead_cost, 4),
            "invalid_actions": self.invalid_actions,
        }

    def observe(self) -> dict[str, Any]:
        self.prepare_current_step()
        demand = self.last_demand_heatmap or self.demand_heatmap()
        traffic = self.last_traffic_heatmap or self.traffic_generator.get_heatmap(self.timestep, demand)
        waiting = [
            item.to_dict(self.timestep)
            for item in self.pending
            if item.status == "waiting"
        ]
        waiting.sort(key=lambda item: (item["patience_remaining"], -float(item["value"])))
        assigned = [
            item.to_dict(self.timestep)
            for item in self.pending
            if item.status == "assigned"
        ]
        return {
            "scenario": {
                "seed": self.seed,
                "event_id": self.event_id,
                "grid": {"rows": self.grid.rows, "cols": self.grid.cols},
                "fleet_size": self.fleet_size,
                "step_minutes": self.step_minutes,
            },
            "done": self.done,
            "timestep": self.timestep,
            "step": self.current_step,
            "remaining_steps": max(0, self.horizon_steps - self.current_step),
            "metrics": self.metrics(),
            "top_demand_cells": self.top_heatmap_cells(demand, 8),
            "traffic_bottlenecks": self.top_heatmap_cells(traffic, 8),
            "cars": [self._compact_car(car) for car in self.cars],
            "waiting_requests": [self._compact_request(item) for item in waiting[:24]],
            "assigned_requests": [self._compact_assigned(item) for item in assigned[:16]],
            "truncated": {
                "waiting_requests": max(0, len(waiting) - 24),
                "assigned_requests": max(0, len(assigned) - 16),
            },
            "reward_so_far": self.reward()["reward"] if self.total_requests else 0.0,
        }

    def _compact_car(self, car: FleetCar) -> dict[str, Any]:
        return {
            "id": car.id,
            "pos": list(car.position),
            "status": car.status,
            "route": len(car.route),
            "stall": car.stall_ticks,
            "person": car.assigned_person_id,
        }

    def _compact_request(self, item: dict[str, Any]) -> dict[str, Any]:
        return {
            "id": item["id"],
            "origin": item["origin"],
            "destination": item["destination"],
            "age": item["age"],
            "patience_remaining": item["patience_remaining"],
            "value": item["value"],
            "party_size": item["party_size"],
        }

    def _compact_assigned(self, item: dict[str, Any]) -> dict[str, Any]:
        return {
            "id": item["id"],
            "car": item["assigned_car_id"],
            "age": item["age"],
            "pickup_cost": item["pickup_cost"],
            "value": item["value"],
        }

    def top_heatmap_cells(self, heatmap: list[list[float]], k: int) -> list[dict[str, Any]]:
        cells = [
            {"cell": [row, col], "value": float(heatmap[row][col])}
            for row in range(len(heatmap))
            for col in range(len(heatmap[row]))
        ]
        cells.sort(key=lambda item: item["value"], reverse=True)
        return cells[: max(0, k)]

    def car_by_id(self, car_id: str) -> FleetCar | None:
        return next((car for car in self.cars if car.id == car_id), None)

    def request_by_id(self, person_id: str) -> PendingRequest | None:
        return next((item for item in self.pending if item.person.id == person_id), None)

    def idle_cars(self) -> list[FleetCar]:
        return [car for car in self.cars if car.status == "idle" and not car.route]

    def waiting_requests(self) -> list[PendingRequest]:
        return [item for item in self.pending if item.status == "waiting"]
