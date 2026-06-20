from __future__ import annotations

from dataclasses import dataclass, field
import heapq
import math
import random
from typing import Any

try:
    import numpy as _np
except ImportError:  # pragma: no cover - exercised only when numpy is absent
    _np = None


DEFAULT_TIME_OF_DAY_PROFILE = [
    0.35,
    0.28,
    0.22,
    0.20,
    0.25,
    0.42,
    0.72,
    0.98,
    1.00,
    0.82,
    0.70,
    0.76,
    0.84,
    0.82,
    0.78,
    0.86,
    1.06,
    1.18,
    1.10,
    0.88,
    0.72,
    0.60,
    0.50,
    0.42,
]

# Mirrors the existing map.py hourly profile. Lower speeds imply higher traffic.
DEFAULT_HOURLY_SPEED_MULTIPLIER = [
    0.95,
    1.00,
    1.00,
    1.00,
    0.95,
    0.90,
    0.75,
    0.60,
    0.55,
    0.65,
    0.78,
    0.84,
    0.88,
    0.85,
    0.82,
    0.72,
    0.60,
    0.52,
    0.58,
    0.70,
    0.80,
    0.88,
    0.92,
    0.95,
]


@dataclass(frozen=True)
class GridSpec:
    rows: int
    cols: int
    values: list[list[float]] | None = None

    @classmethod
    def from_grid(cls, grid: Any) -> "GridSpec":
        if isinstance(grid, GridSpec):
            return grid
        if isinstance(grid, tuple) and len(grid) == 2:
            return cls(rows=int(grid[0]), cols=int(grid[1]))
        if isinstance(grid, dict):
            return cls(
                rows=int(grid.get("rows", 0)),
                cols=int(grid.get("cols", 0)),
                values=grid.get("values"),
            )

        rows = getattr(grid, "rows", None)
        cols = getattr(grid, "cols", None)
        if rows is None or cols is None:
            raise ValueError("grid must be a GridSpec, (rows, cols), dict, or object with rows/cols")
        return cls(rows=int(rows), cols=int(cols), values=getattr(grid, "values", None))

    def contains(self, cell: tuple[int, int]) -> bool:
        row, col = cell
        return 0 <= row < self.rows and 0 <= col < self.cols


@dataclass(frozen=True)
class PersonRequest:
    id: str
    origin: tuple[int, int]
    destination: tuple[int, int]
    created_at: int
    patience: int
    value: float
    party_size: int = 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "origin": list(self.origin),
            "destination": list(self.destination),
            "created_at": self.created_at,
            "patience": self.patience,
            "value": self.value,
            "party_size": self.party_size,
        }


@dataclass(frozen=True)
class RouteResult:
    path: list[tuple[int, int]]
    cost: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": [list(cell) for cell in self.path],
            "cost": round(self.cost, 6),
        }


@dataclass
class CarState:
    id: str
    position: tuple[int, int]
    status: str = "idle"
    assigned_person_id: str | None = None
    pickup: tuple[int, int] | None = None
    dropoff: tuple[int, int] | None = None
    route: list[tuple[int, int]] = field(default_factory=list)
    stall_ticks: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "position": list(self.position),
            "status": self.status,
            "assigned_person_id": self.assigned_person_id,
            "pickup": list(self.pickup) if self.pickup else None,
            "dropoff": list(self.dropoff) if self.dropoff else None,
            "route": [list(cell) for cell in self.route],
            "stall_ticks": self.stall_ticks,
        }


@dataclass(frozen=True)
class DispatchAssignment:
    car_id: str
    person_id: str
    pickup: tuple[int, int]
    dropoff: tuple[int, int]
    assigned_at: int
    pickup_route: RouteResult
    dropoff_route: RouteResult

    @property
    def total_cost(self) -> float:
        return self.pickup_route.cost + self.dropoff_route.cost

    def to_dict(self) -> dict[str, Any]:
        return {
            "car_id": self.car_id,
            "person_id": self.person_id,
            "pickup": list(self.pickup),
            "dropoff": list(self.dropoff),
            "assigned_at": self.assigned_at,
            "pickup_route": self.pickup_route.to_dict(),
            "dropoff_route": self.dropoff_route.to_dict(),
            "total_cost": round(self.total_cost, 6),
        }


def _clip(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _rng_for(seed: int, timestep: int, salt: int) -> random.Random:
    mixed = (seed * 1_000_003 + timestep * 97_531 + salt * 10_007) & ((1 << 63) - 1)
    return random.Random(mixed)


def _profile_value(profile: list[float], timestep: int) -> float:
    if not profile:
        return 1.0

    if len(profile) == 24:
        phase = (timestep % (24 * 60)) / 60.0
    else:
        phase = timestep % len(profile)

    i0 = int(math.floor(phase)) % len(profile)
    i1 = (i0 + 1) % len(profile)
    frac = phase - math.floor(phase)
    return float(profile[i0]) * (1.0 - frac) + float(profile[i1]) * frac


def _normalize(values: list[list[float]]) -> list[list[float]]:
    flat = [float(v) for row in values for v in row]
    if not flat:
        return []
    low = min(flat)
    high = max(flat)
    if math.isclose(low, high):
        return [[0.0 for _ in row] for row in values]
    span = high - low
    return [[_clip((float(v) - low) / span) for v in row] for row in values]


def _shape_like(grid: GridSpec, value: float = 0.0) -> list[list[float]]:
    return [[value for _ in range(grid.cols)] for _ in range(grid.rows)]


def _manhattan(a: tuple[int, int], b: tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _ensure_heatmap(heatmap: list[list[float]], grid: GridSpec, name: str) -> None:
    if len(heatmap) != grid.rows or any(len(row) != grid.cols for row in heatmap):
        raise ValueError(f"{name} must have shape {grid.rows}x{grid.cols}")


def _smooth_heatmap(values: list[list[float]], iterations: int = 1) -> list[list[float]]:
    if iterations <= 0 or not values:
        return values

    rows = len(values)
    cols = len(values[0])
    current = [[float(v) for v in row] for row in values]
    for _ in range(iterations):
        next_values = _shape_like(GridSpec(rows, cols))
        for row in range(rows):
            for col in range(cols):
                total = 0.0
                count = 0
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        rr = row + dr
                        cc = col + dc
                        if 0 <= rr < rows and 0 <= cc < cols:
                            total += current[rr][cc]
                            count += 1
                next_values[row][col] = total / max(1, count)
        current = next_values
    return current


def _weighted_cell(
    heatmap: list[list[float]],
    rng: random.Random,
    excluded: tuple[int, int] | None = None,
    destination_bias: float = 1.0,
    min_distance: int = 0,
) -> tuple[int, int]:
    rows = len(heatmap)
    cols = len(heatmap[0]) if rows else 0
    weights: list[tuple[tuple[int, int], float]] = []
    total = 0.0
    diag = math.hypot(max(1, rows - 1), max(1, cols - 1))
    eligible_cells: list[tuple[int, int]] = []

    for row in range(rows):
        for col in range(cols):
            cell = (row, col)
            if excluded == cell:
                continue
            if excluded is not None and min_distance > 0 and _manhattan(excluded, cell) < min_distance:
                continue
            eligible_cells.append(cell)

    if not eligible_cells and excluded is not None and min_distance > 0:
        return _weighted_cell(heatmap, rng, excluded, destination_bias, min_distance=0)

    for row, col in eligible_cells:
        demand = _clip(float(heatmap[row][col]))
        weight = 0.001 + demand
        if excluded is not None:
            distance = math.hypot(row - excluded[0], col - excluded[1]) / max(1e-9, diag)
            weight *= 0.2 + destination_bias * demand + 0.9 * distance
        weights.append(((row, col), weight))
        total += weight

    if not weights:
        return excluded if excluded is not None else (0, 0)

    pick = rng.random() * total
    running = 0.0
    for cell, weight in weights:
        running += weight
        if running >= pick:
            return cell
    return weights[-1][0]


def _poisson(lam: float, rng: random.Random, seed: int) -> int:
    lam = max(0.0, float(lam))
    if lam == 0:
        return 0

    if _np is not None:
        return int(_np.random.default_rng(seed).poisson(lam))

    # Knuth is exact but slow for large lambda; normal approximation is fine for high demo rates.
    if lam >= 30.0:
        return max(0, int(round(rng.gauss(lam, math.sqrt(lam)))))

    threshold = math.exp(-lam)
    product = 1.0
    count = 0
    while product > threshold:
        count += 1
        product *= rng.random()
    return count - 1


def _default_hotspots(grid: GridSpec, seed: int) -> list[dict[str, float]]:
    rng = random.Random(seed)
    anchors = [
        (0.50, 0.56, 1.00),
        (0.68, 0.62, 0.72),
        (0.36, 0.42, 0.64),
        (0.55, 0.78, 0.52),
    ]
    hotspots = []
    for row_frac, col_frac, intensity in anchors:
        hotspots.append(
            {
                "row": _clip(row_frac + rng.uniform(-0.06, 0.06)) * max(0, grid.rows - 1),
                "col": _clip(col_frac + rng.uniform(-0.06, 0.06)) * max(0, grid.cols - 1),
                "intensity": intensity * rng.uniform(0.88, 1.12),
                "sigma": max(2.0, min(grid.rows, grid.cols) * rng.uniform(0.08, 0.16)),
            }
        )
    return hotspots


def _coerce_hotspot(raw: Any, grid: GridSpec) -> dict[str, float]:
    if isinstance(raw, dict):
        row = raw.get("row", raw.get("r", raw.get("y", 0)))
        col = raw.get("col", raw.get("c", raw.get("x", 0)))
        return {
            "row": float(row),
            "col": float(col),
            "intensity": float(raw.get("intensity", raw.get("weight", 1.0))),
            "sigma": float(raw.get("sigma", max(2.0, min(grid.rows, grid.cols) * 0.12))),
        }

    row, col, *rest = raw
    return {
        "row": float(row),
        "col": float(col),
        "intensity": float(rest[0]) if rest else 1.0,
        "sigma": float(rest[1]) if len(rest) > 1 else max(2.0, min(grid.rows, grid.cols) * 0.12),
    }


class DemandGenerator:
    def __init__(
        self,
        grid: Any,
        seed: int = 7,
        base_hotspots: list[Any] | None = None,
        time_of_day_profile: list[float] | None = None,
        event_multiplier: float = 1.0,
        noise_level: float = 0.035,
        smooth_iterations: int = 1,
    ) -> None:
        self.grid = GridSpec.from_grid(grid)
        self.seed = int(seed)
        self.time_of_day_profile = time_of_day_profile or DEFAULT_TIME_OF_DAY_PROFILE
        self.event_multiplier = float(event_multiplier)
        self.noise_level = float(noise_level)
        self.smooth_iterations = int(smooth_iterations)
        raw_hotspots = base_hotspots if base_hotspots is not None else _default_hotspots(self.grid, seed)
        self.base_hotspots = [_coerce_hotspot(h, self.grid) for h in raw_hotspots]
        self._spatial_base = self._build_spatial_base()

    def _build_spatial_base(self) -> list[list[float]]:
        values = _shape_like(self.grid)
        for row in range(self.grid.rows):
            for col in range(self.grid.cols):
                demand = 0.0
                for hotspot in self.base_hotspots:
                    sigma = max(1e-6, hotspot["sigma"])
                    dist2 = (row - hotspot["row"]) ** 2 + (col - hotspot["col"]) ** 2
                    demand += hotspot["intensity"] * math.exp(-dist2 / (2.0 * sigma * sigma))
                values[row][col] = demand

        if self.grid.values:
            population = _normalize(self.grid.values)
            for row in range(self.grid.rows):
                for col in range(self.grid.cols):
                    values[row][col] += 0.28 * population[row][col]

        return _normalize(values)

    def get_heatmap(self, timestep: int) -> list[list[float]]:
        rng = _rng_for(self.seed, timestep, 11)
        time_multiplier = _profile_value(self.time_of_day_profile, timestep)
        demand = _shape_like(self.grid)

        for row in range(self.grid.rows):
            for col in range(self.grid.cols):
                noise = rng.uniform(-self.noise_level, self.noise_level)
                demand[row][col] = _clip(
                    self._spatial_base[row][col] * time_multiplier * self.event_multiplier + noise
                )

        if self.smooth_iterations:
            demand = _smooth_heatmap(demand, self.smooth_iterations)
        return [[round(_clip(v), 6) for v in row] for row in demand]

    def top_demand_cells(self, k: int, timestep: int = 0) -> list[dict[str, float | int]]:
        heatmap = self.get_heatmap(timestep)
        cells = [
            {"row": row, "col": col, "value": heatmap[row][col]}
            for row in range(self.grid.rows)
            for col in range(self.grid.cols)
        ]
        cells.sort(key=lambda item: float(item["value"]), reverse=True)
        return cells[: max(0, k)]


class PeopleGenerator:
    def __init__(
        self,
        grid: Any,
        seed: int = 7,
        base_arrival_rate: float = 4.0,
        max_new_people_per_tick: int = 12,
        destination_bias: float = 0.75,
        min_trip_distance: int = 20,
    ) -> None:
        self.grid = GridSpec.from_grid(grid)
        self.seed = int(seed)
        self.base_arrival_rate = float(base_arrival_rate)
        self.max_new_people_per_tick = int(max_new_people_per_tick)
        self.destination_bias = float(destination_bias)
        self.min_trip_distance = int(min_trip_distance)

    def generate(
        self,
        timestep: int,
        demand_heatmap: list[list[float]],
        traffic_heatmap: list[list[float]] | None = None,
    ) -> list[PersonRequest]:
        _ensure_heatmap(demand_heatmap, self.grid, "demand_heatmap")
        if traffic_heatmap is not None:
            _ensure_heatmap(traffic_heatmap, self.grid, "traffic_heatmap")

        flat_demand = [float(v) for row in demand_heatmap for v in row]
        mean_demand = sum(flat_demand) / max(1, len(flat_demand))
        max_demand = max(flat_demand, default=0.0)
        traffic_multiplier = 1.0
        if traffic_heatmap is not None:
            flat_traffic = [float(v) for row in traffic_heatmap for v in row]
            mean_traffic = sum(flat_traffic) / max(1, len(flat_traffic))
            max_traffic = max(flat_traffic, default=0.0)
            traffic_multiplier += 0.75 * mean_traffic + 0.55 * max_traffic
        lam = self.base_arrival_rate * (
            0.25 + 0.50 * mean_demand + 0.95 * max_demand
        ) * traffic_multiplier
        rng = _rng_for(self.seed, timestep, 29)
        poisson_seed = (self.seed * 1_000_003 + timestep * 65_537 + 29) & ((1 << 63) - 1)
        count = min(self.max_new_people_per_tick, _poisson(lam, rng, poisson_seed))

        people = []
        for idx in range(count):
            origin = _weighted_cell(demand_heatmap, rng)
            destination = _weighted_cell(
                demand_heatmap,
                rng,
                excluded=origin,
                destination_bias=self.destination_bias,
                min_distance=self.min_trip_distance,
            )
            traffic_at_origin = traffic_heatmap[origin[0]][origin[1]] if traffic_heatmap else 0.0
            distance = math.hypot(destination[0] - origin[0], destination[1] - origin[1])
            party_size = 1 + (1 if rng.random() < 0.18 else 0) + (1 if rng.random() < 0.05 else 0)
            patience = int(_clip(10 + rng.randint(0, 12) + demand_heatmap[origin[0]][origin[1]] * 8 - traffic_at_origin * 4, 5, 32))
            value = round(7.5 + distance * 1.15 + party_size * 2.25 + traffic_at_origin * 3.0, 2)
            people.append(
                PersonRequest(
                    id=f"person-{timestep}-{idx}-{rng.randrange(1_000_000):06d}",
                    origin=origin,
                    destination=destination,
                    created_at=timestep,
                    patience=patience,
                    value=value,
                    party_size=party_size,
                )
            )

        return people


class TrafficGenerator:
    def __init__(
        self,
        grid: Any,
        seed: int = 7,
        base_heatmap: list[list[float]] | None = None,
        hourly_speed_multiplier: list[float] | None = None,
        noise_level: float = 0.04,
        demand_coupling: float = 0.12,
        smooth_iterations: int = 1,
    ) -> None:
        self.grid = GridSpec.from_grid(grid)
        self.seed = int(seed)
        self.hourly_speed_multiplier = hourly_speed_multiplier or DEFAULT_HOURLY_SPEED_MULTIPLIER
        self.noise_level = float(noise_level)
        self.demand_coupling = float(demand_coupling)
        self.smooth_iterations = int(smooth_iterations)

        if base_heatmap is not None:
            _ensure_heatmap(base_heatmap, self.grid, "base_heatmap")
            self._base = [[_clip(float(v)) for v in row] for row in base_heatmap]
        else:
            self._base = self._build_base_heatmap()

    def _build_base_heatmap(self) -> list[list[float]]:
        rng = random.Random(self.seed + 101)
        population = _normalize(self.grid.values) if self.grid.values else _shape_like(self.grid, 0.35)
        values = _shape_like(self.grid)
        row_mid = (self.grid.rows - 1) / 2.0
        col_mid = (self.grid.cols - 1) / 2.0

        for row in range(self.grid.rows):
            for col in range(self.grid.cols):
                row_core = 1.0 - abs(row - row_mid) / max(1.0, self.grid.rows / 2.0)
                col_core = 1.0 - abs(col - col_mid) / max(1.0, self.grid.cols / 2.0)
                corridor = max(0.0, 0.5 * row_core + 0.5 * col_core)
                texture = rng.uniform(-0.035, 0.035)
                values[row][col] = _clip(0.12 + 0.42 * population[row][col] + 0.22 * corridor + texture)

        return _smooth_heatmap(values, 1)

    def get_heatmap(
        self,
        timestep: int,
        demand_heatmap: list[list[float]] | None = None,
    ) -> list[list[float]]:
        if demand_heatmap is not None:
            _ensure_heatmap(demand_heatmap, self.grid, "demand_heatmap")

        rng = _rng_for(self.seed, timestep, 47)
        speed_multiplier = _profile_value(self.hourly_speed_multiplier, timestep)
        min_speed = min(self.hourly_speed_multiplier) if self.hourly_speed_multiplier else 0.52
        peak = _clip((1.0 - speed_multiplier) / max(1e-9, 1.0 - min_speed))
        time_multiplier = 0.62 + peak * 0.72
        heatmap = _shape_like(self.grid)

        for row in range(self.grid.rows):
            for col in range(self.grid.cols):
                demand_pressure = demand_heatmap[row][col] * self.demand_coupling if demand_heatmap else 0.0
                noise = rng.uniform(-self.noise_level, self.noise_level)
                heatmap[row][col] = _clip(self._base[row][col] * time_multiplier + demand_pressure + noise)

        if self.smooth_iterations:
            heatmap = _smooth_heatmap(heatmap, self.smooth_iterations)
        return [[round(_clip(v), 6) for v in row] for row in heatmap]

    def top_bottlenecks(
        self,
        k: int,
        timestep: int,
        demand_heatmap: list[list[float]] | None = None,
    ) -> list[dict[str, float | int]]:
        heatmap = self.get_heatmap(timestep, demand_heatmap)
        cells = [
            {"row": row, "col": col, "value": heatmap[row][col]}
            for row in range(self.grid.rows)
            for col in range(self.grid.cols)
        ]
        cells.sort(key=lambda item: float(item["value"]), reverse=True)
        return cells[: max(0, k)]


class GridRouter:
    def __init__(self, grid: Any, traffic_weight: float = 4.0) -> None:
        self.grid = GridSpec.from_grid(grid)
        self.traffic_weight = float(traffic_weight)

    def route(
        self,
        start: tuple[int, int],
        goal: tuple[int, int],
        traffic_heatmap: list[list[float]],
    ) -> RouteResult:
        _ensure_heatmap(traffic_heatmap, self.grid, "traffic_heatmap")
        if not self.grid.contains(start) or not self.grid.contains(goal):
            raise ValueError("route start and goal must be inside the grid")
        if start == goal:
            return RouteResult(path=[start], cost=0.0)

        frontier: list[tuple[float, tuple[int, int]]] = [(0.0, start)]
        best_cost = {start: 0.0}
        came_from: dict[tuple[int, int], tuple[int, int]] = {}

        while frontier:
            cost, cell = heapq.heappop(frontier)
            if cell == goal:
                break
            if cost > best_cost.get(cell, math.inf):
                continue

            for neighbor in self._neighbors(cell):
                row, col = neighbor
                move_cost = 1.0 + self.traffic_weight * _clip(float(traffic_heatmap[row][col]))
                next_cost = cost + move_cost
                if next_cost < best_cost.get(neighbor, math.inf):
                    best_cost[neighbor] = next_cost
                    came_from[neighbor] = cell
                    heapq.heappush(frontier, (next_cost, neighbor))

        if goal not in best_cost:
            return RouteResult(path=[start], cost=math.inf)

        path = [goal]
        while path[-1] != start:
            path.append(came_from[path[-1]])
        path.reverse()
        return RouteResult(path=path, cost=best_cost[goal])

    def _neighbors(self, cell: tuple[int, int]) -> list[tuple[int, int]]:
        row, col = cell
        candidates = [
            (row - 1, col),
            (row, col + 1),
            (row + 1, col),
            (row, col - 1),
        ]
        return [candidate for candidate in candidates if self.grid.contains(candidate)]


class GreedyDispatcher:
    def __init__(
        self,
        grid: Any,
        seed: int = 7,
        fleet_size: int = 16,
        initial_car_positions: list[tuple[int, int]] | None = None,
        router: GridRouter | None = None,
    ) -> None:
        self.grid = GridSpec.from_grid(grid)
        self.seed = int(seed)
        self.router = router or GridRouter(self.grid)
        positions = initial_car_positions or self._seed_car_positions(fleet_size)
        self.cars = [
            CarState(id=f"car-{idx}", position=(int(position[0]), int(position[1])))
            for idx, position in enumerate(positions)
        ]

    def _seed_car_positions(self, fleet_size: int) -> list[tuple[int, int]]:
        rng = random.Random(self.seed + 503)
        positions = []
        used: set[tuple[int, int]] = set()
        max_unique = self.grid.rows * self.grid.cols

        for _ in range(max(0, fleet_size)):
            for _attempt in range(20):
                cell = (rng.randrange(self.grid.rows), rng.randrange(self.grid.cols))
                if len(used) >= max_unique or cell not in used:
                    break
            positions.append(cell)
            used.add(cell)
        return positions

    def step(
        self,
        timestep: int,
        people: list[PersonRequest],
        traffic_heatmap: list[list[float]],
    ) -> dict[str, Any]:
        _ensure_heatmap(traffic_heatmap, self.grid, "traffic_heatmap")
        completed_person_ids = self._advance_cars(mark_idle_stall=False)
        assignments, unassigned_people = self._assign_people(timestep, people, traffic_heatmap)
        completed_person_ids.extend(self._advance_cars(mark_idle_stall=True))

        stalled_cars = [
            car.to_dict()
            for car in self.cars
            if car.status == "idle" and not car.route
        ]

        return {
            "assignments": [assignment.to_dict() for assignment in assignments],
            "unassigned_people": [person.to_dict() for person in unassigned_people],
            "completed_person_ids": completed_person_ids,
            "stalled_cars": stalled_cars,
            "cars": [car.to_dict() for car in self.cars],
            "car_grid": build_car_grid(self.grid, self.cars),
            "summary": {
                "num_assignments": len(assignments),
                "num_unassigned_people": len(unassigned_people),
                "num_stalled_cars": len(stalled_cars),
                "num_active_cars": sum(1 for car in self.cars if car.status != "idle"),
            },
        }

    def _assign_people(
        self,
        timestep: int,
        people: list[PersonRequest],
        traffic_heatmap: list[list[float]],
    ) -> tuple[list[DispatchAssignment], list[PersonRequest]]:
        idle_cars = [car for car in self.cars if car.status == "idle" and not car.route]
        assignments: list[DispatchAssignment] = []
        unassigned_people: list[PersonRequest] = []

        for person in people:
            if not idle_cars:
                unassigned_people.append(person)
                continue

            dropoff_route = self.router.route(person.origin, person.destination, traffic_heatmap)
            best: tuple[float, int, CarState, RouteResult] | None = None
            for idx, car in enumerate(idle_cars):
                pickup_route = self.router.route(car.position, person.origin, traffic_heatmap)
                cost = pickup_route.cost
                if best is None or cost < best[0]:
                    best = (cost, idx, car, pickup_route)

            if best is None:
                unassigned_people.append(person)
                continue

            _, idle_idx, car, pickup_route = best
            idle_cars.pop(idle_idx)
            assignment = DispatchAssignment(
                car_id=car.id,
                person_id=person.id,
                pickup=person.origin,
                dropoff=person.destination,
                assigned_at=timestep,
                pickup_route=pickup_route,
                dropoff_route=dropoff_route,
            )
            self._apply_assignment(car, assignment)
            assignments.append(assignment)

        return assignments, unassigned_people

    def _apply_assignment(self, car: CarState, assignment: DispatchAssignment) -> None:
        car.assigned_person_id = assignment.person_id
        car.pickup = assignment.pickup
        car.dropoff = assignment.dropoff
        car.stall_ticks = 0

        to_pickup = assignment.pickup_route.path[1:]
        to_dropoff = assignment.dropoff_route.path[1:]
        car.route = to_pickup + to_dropoff
        car.status = "to_pickup" if to_pickup else "to_dropoff"

    def _advance_cars(self, mark_idle_stall: bool) -> list[str]:
        completed_person_ids = []

        for car in self.cars:
            if car.route:
                car.position = car.route.pop(0)
                car.stall_ticks = 0
                if car.pickup and car.position == car.pickup and car.status == "to_pickup":
                    car.status = "to_dropoff"
                if not car.route and car.status == "to_dropoff":
                    if car.assigned_person_id:
                        completed_person_ids.append(car.assigned_person_id)
                    car.status = "idle"
                    car.assigned_person_id = None
                    car.pickup = None
                    car.dropoff = None
            else:
                car.status = "idle"
                if mark_idle_stall:
                    car.stall_ticks += 1

        return completed_person_ids


def build_people_grid(grid: Any, people: list[PersonRequest]) -> dict[str, Any]:
    spec = GridSpec.from_grid(grid)
    pickup_grid = _shape_like(spec, 0.0)
    dropoff_grid = _shape_like(spec, 0.0)
    markers = []

    for person in people:
        if spec.contains(person.origin):
            pickup_grid[person.origin[0]][person.origin[1]] += 1.0
        if spec.contains(person.destination):
            dropoff_grid[person.destination[0]][person.destination[1]] += 1.0
        markers.append(
            {
                "person_id": person.id,
                "pickup": list(person.origin),
                "dropoff": list(person.destination),
                "pickup_color": "#a855f7",
                "dropoff_color": "#f97316",
            }
        )

    max_pickup = max([value for row in pickup_grid for value in row], default=1.0)
    max_dropoff = max([value for row in dropoff_grid for value in row], default=1.0)
    return {
        "pickup_grid": [[round(value / max(1.0, max_pickup), 6) for value in row] for row in pickup_grid],
        "dropoff_grid": [[round(value / max(1.0, max_dropoff), 6) for value in row] for row in dropoff_grid],
        "markers": markers,
        "legend": {
            "pickup": "#a855f7",
            "dropoff": "#f97316",
        },
    }


def build_car_grid(grid: Any, cars: list[CarState]) -> dict[str, Any]:
    spec = GridSpec.from_grid(grid)
    car_grid = _shape_like(spec, 0.0)
    markers = []

    for car in cars:
        if spec.contains(car.position):
            car_grid[car.position[0]][car.position[1]] += 1.0
        markers.append(
            {
                "car_id": car.id,
                "position": list(car.position),
                "status": car.status,
                "color": "#38bdf8" if car.status == "idle" else "#00d2a5",
            }
        )

    max_count = max([value for row in car_grid for value in row], default=1.0)
    return {
        "grid": [[round(value / max(1.0, max_count), 6) for value in row] for row in car_grid],
        "markers": markers,
        "legend": {
            "idle": "#38bdf8",
            "assigned": "#00d2a5",
        },
    }


class WorldGenerators:
    def __init__(
        self,
        grid: Any,
        seed: int = 7,
        demand_generator: DemandGenerator | None = None,
        traffic_generator: TrafficGenerator | None = None,
        people_generator: PeopleGenerator | None = None,
        dispatcher: GreedyDispatcher | None = None,
        fleet_size: int = 16,
    ) -> None:
        self.grid = GridSpec.from_grid(grid)
        self.seed = int(seed)
        self.demand = demand_generator or DemandGenerator(self.grid, seed=self.seed + 1)
        self.traffic = traffic_generator or TrafficGenerator(self.grid, seed=self.seed + 2)
        self.people = people_generator or PeopleGenerator(self.grid, seed=self.seed + 3)
        self.dispatcher = dispatcher or GreedyDispatcher(
            self.grid,
            seed=self.seed + 4,
            fleet_size=fleet_size,
        )

    def step(self, timestep: int) -> dict[str, Any]:
        demand_heatmap = self.demand.get_heatmap(timestep)
        traffic_heatmap = self.traffic.get_heatmap(timestep, demand_heatmap)
        new_people = self.people.generate(timestep, demand_heatmap, traffic_heatmap)
        dispatch = self.dispatcher.step(timestep, new_people, traffic_heatmap)
        assigned_ids = {
            assignment["person_id"]
            for assignment in dispatch["assignments"]
        }
        assigned_people = [person for person in new_people if person.id in assigned_ids]
        served_pct = (
            len(assigned_people) / len(new_people) * 100.0
            if new_people
            else 100.0
        )
        wait_times = []
        people_by_id = {person.id: person for person in new_people}
        for assignment in dispatch["assignments"]:
            person = people_by_id.get(assignment["person_id"])
            if person is None:
                continue
            pickup_cost = float(assignment.get("pickup_route", {}).get("cost", 0.0))
            queue_wait = max(0, timestep - int(person.created_at))
            wait_times.append(queue_wait + pickup_cost)
        wait_time = sum(wait_times)
        fleet_size = max(1, len(dispatch["cars"]))
        fleet_utilization = dispatch["summary"]["num_active_cars"] / fleet_size * 100.0
        greedy_stats = {
            "completed_trips": len(assigned_people),
            "revenue": round(sum(person.value for person in assigned_people), 2),
            "demand_served_pct": round(served_pct, 2),
            "wait_time_min": round(wait_time, 2),
            "fleet_utilization_pct": round(fleet_utilization, 2),
            "avg_fleet_utilization_pct": round(fleet_utilization, 2),
            "active_cars": dispatch["summary"]["num_active_cars"],
            "stalled_cars": dispatch["summary"]["num_stalled_cars"],
            "unassigned_people": dispatch["summary"]["num_unassigned_people"],
        }

        return {
            "timestep": timestep,
            "demand_heatmap": demand_heatmap,
            "traffic_heatmap": traffic_heatmap,
            "new_people": [person.to_dict() for person in new_people],
            "people_grid": build_people_grid(self.grid, new_people),
            "dispatch": dispatch,
            "greedy_stats": greedy_stats,
            "summary": {
                "num_new_people": len(new_people),
                "top_demand_cells": self.demand.top_demand_cells(5, timestep),
                "traffic_bottlenecks": self.traffic.top_bottlenecks(5, timestep, demand_heatmap),
                "dispatch": dispatch["summary"],
                "greedy_stats": greedy_stats,
            },
        }
