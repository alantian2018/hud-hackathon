from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

from .schemas import ActionPlan, AssignmentAction, RepositionAction
from .world import FleetCar, MobilityWorld, PendingRequest, _clip, _manhattan


@dataclass(frozen=True)
class PairScore:
    car_id: str
    person_id: str
    score: float
    pickup_cost: float
    trip_cost: float
    value: float
    urgency: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "car_id": self.car_id,
            "person_id": self.person_id,
            "score": round(self.score, 6),
            "pickup_cost": round(self.pickup_cost, 6),
            "trip_cost": round(self.trip_cost, 6),
            "value": round(self.value, 2),
            "urgency": round(self.urgency, 6),
        }


def score_pair(world: MobilityWorld, car: FleetCar, request: PendingRequest) -> PairScore | None:
    pickup_route = world.router.route(car.position, request.person.origin, world.last_traffic_heatmap)
    trip_route = world.router.route(request.person.origin, request.person.destination, world.last_traffic_heatmap)
    if not math.isfinite(pickup_route.cost) or not math.isfinite(trip_route.cost):
        return None
    age = max(0, world.timestep - request.person.created_at)
    urgency = _clip(age / max(1, request.person.patience))
    value = float(request.person.value)
    demand_bonus = 0.0
    if world.last_demand_heatmap:
        row, col = request.person.origin
        demand_bonus = 6.0 * float(world.last_demand_heatmap[row][col])
    score = (
        2.4 * value
        + 14.0 * urgency
        + demand_bonus
        - 1.15 * float(pickup_route.cost)
        - 0.18 * float(trip_route.cost)
    )
    return PairScore(
        car_id=car.id,
        person_id=request.person.id,
        score=score,
        pickup_cost=float(pickup_route.cost),
        trip_cost=float(trip_route.cost),
        value=value,
        urgency=urgency,
    )


def global_batch_matching(world: MobilityWorld, *, beam_width: int = 96) -> list[PairScore]:
    world.prepare_current_step()
    idle_cars = world.idle_cars()
    waiting = world.waiting_requests()
    if not idle_cars or not waiting:
        return []

    pair_by_request: dict[str, list[PairScore]] = {}
    for request in waiting:
        pairs = []
        for car in idle_cars:
            pair = score_pair(world, car, request)
            if pair is not None:
                pairs.append(pair)
        pairs.sort(key=lambda item: item.score, reverse=True)
        pair_by_request[request.person.id] = pairs[: min(len(pairs), 10)]

    ordered_requests = sorted(
        waiting,
        key=lambda item: (
            max(0, world.timestep - item.person.created_at) / max(1, item.person.patience),
            item.person.value,
        ),
        reverse=True,
    )

    beams: list[tuple[float, tuple[PairScore, ...], frozenset[str]]] = [(0.0, tuple(), frozenset())]
    for request in ordered_requests:
        next_beams = list(beams)
        for total, selected, used_cars in beams:
            for pair in pair_by_request.get(request.person.id, []):
                if pair.car_id in used_cars or pair.score <= -8.0:
                    continue
                next_beams.append((total + pair.score, selected + (pair,), used_cars | {pair.car_id}))
        next_beams.sort(key=lambda item: (item[0], len(item[1])), reverse=True)
        beams = next_beams[:beam_width]

    best = beams[0][1] if beams else tuple()
    return list(best)


def reposition_targets(world: MobilityWorld, assigned_car_ids: set[str] | None = None, *, max_targets: int | None = None) -> list[RepositionAction]:
    world.prepare_current_step()
    assigned_car_ids = assigned_car_ids or set()
    idle = [car for car in world.idle_cars() if car.id not in assigned_car_ids]
    if not idle:
        return []
    waiting_origins = {item.person.origin for item in world.waiting_requests()}
    hot = world.forecast_hotspots(lookahead_steps=4, k=max(8, len(idle)))
    hot_cells = [(int(item["cell"][0]), int(item["cell"][1]), float(item["score"])) for item in hot]
    targets: list[RepositionAction] = []
    target_load: dict[tuple[int, int], int] = {}
    limit = max_targets if max_targets is not None else max(1, len(idle) // 2)

    for car in sorted(idle, key=lambda item: item.stall_ticks, reverse=True):
        if len(targets) >= limit:
            break
        best = None
        for row, col, score in hot_cells:
            cell = (row, col)
            if cell in waiting_origins:
                continue
            load = target_load.get(cell, 0)
            if load >= 2:
                continue
            distance = _manhattan(car.position, cell)
            value = score / (1.0 + 0.18 * distance + 0.6 * load)
            if best is None or value > best[0]:
                best = (value, cell)
        if best is None:
            continue
        if _manhattan(car.position, best[1]) < 2 and car.stall_ticks < 2:
            continue
        target_load[best[1]] = target_load.get(best[1], 0) + 1
        targets.append(RepositionAction(car_id=car.id, target=best[1]))
    return targets


def build_value_aware_plan(world: MobilityWorld) -> ActionPlan:
    world.prepare_current_step()
    pairs = global_batch_matching(world)
    assigned_car_ids = {pair.car_id for pair in pairs}
    assignments = [
        AssignmentAction(car_id=pair.car_id, person_id=pair.person_id)
        for pair in pairs
    ]
    repositions = reposition_targets(world, assigned_car_ids=assigned_car_ids)
    return ActionPlan(
        assignments=assignments,
        repositions=repositions,
        rationale=(
            "Batch-selected assignments by value, urgency, demand pressure, and pickup cost; "
            "repositioned idle supply toward forecast demand."
        ),
    )


def critique_plan(world: MobilityWorld, plan: ActionPlan) -> dict[str, Any]:
    world.prepare_current_step()
    car_ids = [item.car_id for item in plan.assignments]
    duplicate_cars = len(car_ids) - len(set(car_ids))
    person_ids = [item.person_id for item in plan.assignments]
    duplicate_people = len(person_ids) - len(set(person_ids))
    unknown_cars = [
        car_id
        for car_id in set(car_ids + [item.car_id for item in plan.repositions])
        if world.car_by_id(car_id) is None
    ]
    unknown_people = [
        person_id
        for person_id in set(person_ids)
        if world.request_by_id(person_id) is None
    ]
    value = 0.0
    pickup_cost = 0.0
    for action in plan.assignments:
        car = world.car_by_id(action.car_id)
        request = world.request_by_id(action.person_id)
        if car is None or request is None or not world.last_traffic_heatmap:
            continue
        pair = score_pair(world, car, request)
        if pair:
            value += pair.value
            pickup_cost += pair.pickup_cost
    return {
        "duplicate_cars": duplicate_cars,
        "duplicate_people": duplicate_people,
        "unknown_cars": unknown_cars,
        "unknown_people": unknown_people,
        "planned_assignment_value": round(value, 2),
        "planned_pickup_cost": round(pickup_cost, 4),
        "num_assignments": len(plan.assignments),
        "num_repositions": len(plan.repositions),
    }
