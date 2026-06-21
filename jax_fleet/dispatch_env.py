from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import pickle
import tempfile
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from jax_fleet.env import (
    CAR_DECISION,
    CAR_REPOSITION,
    REQUEST_ASSIGNED,
    REQUEST_ONBOARD,
    REQUEST_QUEUED,
    _add_transition_reward,
    _assign_request_to_car,
    _make_timestep,
    _next_event_time,
    _process_events_at_time,
    _start_auto_edge,
    make_env_params,
    reset as functional_reset,
)
from jax_fleet.scene_export import export_scene
from jax_fleet.spawns import nearest_compact_node_for_grid_cell
from jax_fleet.types import EnvParams, EnvState, GraphArrays, Timestep


@dataclass(frozen=True)
class DispatchResult:
    timestep: Timestep
    action_results: list[dict[str, Any]]
    scene: dict[str, Any]


class ManualDispatchEnv:
    """High-level dispatcher environment for LLM control.

    The base JAX environment asks a policy to choose every outgoing road edge.
    This wrapper instead asks for macro dispatch decisions only when cars are
    idle: assign a queued request, reposition toward a node/grid cell, or wait.
    """

    def __init__(
        self,
        *,
        graph: GraphArrays,
        params: EnvParams | None = None,
        seed: int = 0,
        max_cars: int = 8,
        max_requests: int = 64,
        initial_car_nodes: list[int] | np.ndarray | None = None,
        spawn_rate_per_minute: float = 0.0,
        episode_seconds: float = 1800.0,
        start_time_seconds: float = 0.0,
        default_wait_seconds: float = 30.0,
    ) -> None:
        self.params = params or make_env_params(
            graph,
            max_cars=max_cars,
            max_requests=max_requests,
            initial_car_nodes=initial_car_nodes,
            spawn_rate_per_minute=spawn_rate_per_minute,
            episode_seconds=episode_seconds,
            start_time_seconds=start_time_seconds,
            manual_dispatch=True,
        )
        if not self.params.manual_dispatch:
            self.params = self.params.replace(manual_dispatch=True)
        self.seed = int(seed)
        self.default_wait_seconds = float(default_wait_seconds)
        self._rng = jax.random.PRNGKey(self.seed)
        self._state: EnvState | None = None
        self._timestep: Timestep | None = None
        self._wait_until = np.full((self.params.max_cars,), -math.inf, dtype=np.float32)

    @property
    def state(self) -> EnvState:
        if self._state is None:
            raise RuntimeError("reset must be called before reading state")
        return self._state

    @property
    def timestep(self) -> Timestep:
        if self._timestep is None:
            raise RuntimeError("reset must be called before reading timestep")
        return self._timestep

    @property
    def done(self) -> bool:
        return bool(np.asarray(self.state.done)) if self._state is not None else False

    def reset(self, *, seed: int | None = None) -> dict[str, Any]:
        if seed is not None:
            self.seed = int(seed)
            self._rng = jax.random.PRNGKey(self.seed)
        self._rng, reset_key = jax.random.split(self._rng)
        self._wait_until = np.full((self.params.max_cars,), -math.inf, dtype=np.float32)
        self._state, self._timestep = functional_reset(reset_key, self.params)
        self._state = self._refresh_dispatch_decision(self._state)
        self._timestep = _make_timestep(
            self._state,
            self.params,
            self._timestep.reward,
            self._timestep.dt_seconds,
        )
        return self.get_dispatch_state()

    def save_snapshot(self, path: str | Path) -> dict[str, Any]:
        """Persist the live simulator state so another process can resume it."""
        if self._state is None or self._timestep is None:
            raise RuntimeError("reset must be called before saving state")
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "seed": self.seed,
            "default_wait_seconds": self.default_wait_seconds,
            "rng": self._rng,
            "state": self._state,
            "timestep": self._timestep,
            "wait_until": self._wait_until,
            "max_cars": self.params.max_cars,
            "max_requests": self.params.max_requests,
        }
        with tempfile.NamedTemporaryFile("wb", delete=False, dir=target.parent) as tmp:
            pickle.dump(payload, tmp)
            tmp_path = Path(tmp.name)
        tmp_path.replace(target)
        return {
            "path": str(target),
            "time_seconds": self._time_seconds(self.state),
            "done": self.done,
        }

    def load_snapshot(self, path: str | Path) -> dict[str, Any]:
        """Load a simulator snapshot written by save_snapshot."""
        source = Path(path)
        with source.open("rb") as fh:
            payload = pickle.load(fh)
        if int(payload.get("max_cars", -1)) != self.params.max_cars:
            raise ValueError("Persistent dispatch state max_cars does not match this task")
        if int(payload.get("max_requests", -1)) != self.params.max_requests:
            raise ValueError("Persistent dispatch state max_requests does not match this task")
        self.seed = int(payload.get("seed", self.seed))
        self.default_wait_seconds = float(
            payload.get("default_wait_seconds", self.default_wait_seconds)
        )
        self._rng = payload["rng"]
        self._state = payload["state"]
        self._timestep = payload["timestep"]
        self._wait_until = np.asarray(payload["wait_until"], dtype=np.float32)
        return {
            "path": str(source),
            "time_seconds": self._time_seconds(self.state),
            "done": self.done,
        }

    def get_dispatch_state(self) -> dict[str, Any]:
        scene = self.scene(include_static=False, include_route_previews=True)
        return {
            "time_seconds": scene["time_seconds"],
            "done": scene["done"],
            "current_car_id": scene["current_car_id"],
            "decision_required": bool(np.asarray(self.state.decision_required)),
            "idle_cars": self.get_idle_cars(ready_only=True),
            "sleeping_idle_cars": self.get_idle_cars(ready_only=False, sleeping_only=True),
            "open_requests": self.get_open_requests(),
            "metrics": scene["metrics"],
        }

    def get_idle_cars(
        self,
        *,
        ready_only: bool = False,
        sleeping_only: bool = False,
    ) -> list[dict[str, Any]]:
        state = self.state
        graph = self.params.graph
        time_seconds = self._time_seconds(state)
        statuses = np.asarray(state.car_status)
        nodes = np.asarray(state.car_nodes)
        cars = []
        for car_id in range(self.params.max_cars):
            if int(statuses[car_id]) != CAR_DECISION:
                continue
            ready = self._wait_until[car_id] <= time_seconds + 1e-6
            if ready_only and not ready:
                continue
            if sleeping_only and ready:
                continue
            node = int(nodes[car_id])
            cars.append(
                {
                    "car_id": car_id,
                    "compact_node_id": node,
                    "original_node_id": int(np.asarray(graph.original_node_ids)[node]),
                    "lonlat": np.asarray(graph.node_lonlat)[node].astype(float).tolist(),
                    "grid_cell": [
                        int(np.asarray(graph.node_grid_rows)[node]),
                        int(np.asarray(graph.node_grid_cols)[node]),
                    ],
                    "dispatch_ready": ready,
                    "sleep_until_seconds": None if ready else float(self._wait_until[car_id]),
                }
            )
        return cars

    def get_open_requests(self, *, include_assigned: bool = False) -> list[dict[str, Any]]:
        state = self.state
        graph = self.params.graph
        statuses = np.asarray(state.request_status)
        origins = np.asarray(state.request_origin_nodes)
        destinations = np.asarray(state.request_dest_nodes)
        spawn_times = np.asarray(state.request_spawn_times)
        assigned_cars = np.asarray(state.request_assigned_car_ids)
        time_seconds = self._time_seconds(state)
        allowed = {REQUEST_QUEUED}
        if include_assigned:
            allowed |= {REQUEST_ASSIGNED, REQUEST_ONBOARD}

        requests = []
        for request_id in range(self.params.max_requests):
            status = int(statuses[request_id])
            if status not in allowed:
                continue
            origin = int(origins[request_id])
            destination = int(destinations[request_id])
            requests.append(
                {
                    "request_id": request_id,
                    "status": self._request_status_label(status),
                    "origin_compact_node_id": origin,
                    "destination_compact_node_id": destination,
                    "origin_original_node_id": int(np.asarray(graph.original_node_ids)[origin]),
                    "destination_original_node_id": int(np.asarray(graph.original_node_ids)[destination]),
                    "origin_lonlat": np.asarray(graph.node_lonlat)[origin].astype(float).tolist(),
                    "destination_lonlat": np.asarray(graph.node_lonlat)[destination].astype(float).tolist(),
                    "origin_grid_cell": [
                        int(np.asarray(graph.node_grid_rows)[origin]),
                        int(np.asarray(graph.node_grid_cols)[origin]),
                    ],
                    "destination_grid_cell": [
                        int(np.asarray(graph.node_grid_rows)[destination]),
                        int(np.asarray(graph.node_grid_cols)[destination]),
                    ],
                    "wait_seconds": max(0.0, time_seconds - float(spawn_times[request_id])),
                    "assigned_car_id": int(assigned_cars[request_id]),
                }
            )
        return requests

    def get_eta_matrix(
        self,
        car_ids: list[int] | None = None,
        request_ids: list[int] | None = None,
    ) -> dict[str, Any]:
        idle_car_ids = [car["car_id"] for car in self.get_idle_cars(ready_only=True)]
        open_request_ids = [request["request_id"] for request in self.get_open_requests()]
        selected_cars = idle_car_ids if car_ids is None else [int(car_id) for car_id in car_ids]
        selected_requests = open_request_ids if request_ids is None else [int(request_id) for request_id in request_ids]

        car_nodes = np.asarray(self.state.car_nodes)
        request_origins = np.asarray(self.state.request_origin_nodes)
        routing = np.asarray(self.params.graph.routing_travel_time_s)
        entries = []
        for car_id in selected_cars:
            if car_id < 0 or car_id >= self.params.max_cars:
                continue
            car_node = int(car_nodes[car_id])
            for request_id in selected_requests:
                if request_id < 0 or request_id >= self.params.max_requests:
                    continue
                origin = int(request_origins[request_id])
                eta = float(routing[car_node, origin]) if origin >= 0 else math.inf
                entries.append(
                    {
                        "car_id": car_id,
                        "request_id": request_id,
                        "pickup_eta_seconds": eta,
                        "reachable": math.isfinite(eta),
                    }
                )
        return {"entries": entries}

    def get_reposition_targets(self, *, count: int = 8) -> list[dict[str, Any]]:
        graph = self.params.graph
        hour = int((self._time_seconds(self.state) // 3600) % 24)
        node_weights = np.asarray(self.params.node_density_by_hour)[hour]
        rows = np.asarray(graph.node_grid_rows)
        cols = np.asarray(graph.node_grid_cols)
        lonlat = np.asarray(graph.node_lonlat)
        by_cell: dict[tuple[int, int], dict[str, Any]] = {}
        for node_id, weight in enumerate(node_weights):
            row = int(rows[node_id])
            col = int(cols[node_id])
            if row < 0 or col < 0:
                continue
            key = (row, col)
            current = by_cell.get(key)
            if current is None:
                by_cell[key] = {
                    "grid_cell": [row, col],
                    "score": float(weight),
                    "_best_node_weight": float(weight),
                    "representative_compact_node_id": int(node_id),
                    "representative_original_node_id": int(np.asarray(graph.original_node_ids)[node_id]),
                    "lonlat": lonlat[node_id].astype(float).tolist(),
                }
            else:
                current["score"] += float(weight)
                if float(weight) > float(current.get("_best_node_weight", -math.inf)):
                    current["_best_node_weight"] = float(weight)
                    current["representative_compact_node_id"] = int(node_id)
                    current["representative_original_node_id"] = int(np.asarray(graph.original_node_ids)[node_id])
                    current["lonlat"] = lonlat[node_id].astype(float).tolist()

        targets = sorted(by_cell.values(), key=lambda item: item["score"], reverse=True)[: int(count)]
        for target in targets:
            target.pop("_best_node_weight", None)
        return targets

    def simulate_dispatch_plan(self, actions: list[dict[str, Any]]) -> dict[str, Any]:
        saved_state = self._state
        saved_timestep = self._timestep
        saved_wait_until = self._wait_until.copy()
        try:
            result = self.submit_dispatch_plan(actions, mutate=False)
            return {
                "reward": float(np.asarray(result.timestep.reward)),
                "dt_seconds": float(np.asarray(result.timestep.dt_seconds)),
                "action_results": result.action_results,
                "metrics": result.scene["metrics"],
            }
        finally:
            self._state = saved_state
            self._timestep = saved_timestep
            self._wait_until = saved_wait_until

    def submit_dispatch_plan(
        self,
        actions: list[dict[str, Any]],
        *,
        mutate: bool = True,
    ) -> DispatchResult:
        if self._state is None or self._timestep is None:
            self.reset()

        state = self.state
        reward = jnp.asarray(0.0, dtype=jnp.float32)
        previous_time = self._time_seconds(state)
        controllable_at_start = set(self._dispatch_ready_car_ids(state))
        acted: set[int] = set()
        action_results: list[dict[str, Any]] = []

        for action in actions:
            state, reward, result = self._apply_dispatch_action(state, reward, action)
            if result.get("car_id") is not None:
                acted.add(int(result["car_id"]))
            action_results.append(result)

        for car_id in sorted(controllable_at_start - acted):
            self._wait_until[car_id] = previous_time + self.default_wait_seconds
            action_results.append(
                {
                    "car_id": car_id,
                    "action": "wait",
                    "valid": True,
                    "reason": "default wait for omitted idle car",
                    "sleep_until_seconds": float(self._wait_until[car_id]),
                }
            )

        state = self._refresh_dispatch_decision(state)
        state, reward = self._advance_until_dispatch_decision(state, reward)
        state = _add_transition_reward(state, reward)
        timestep = _make_timestep(
            state,
            self.params,
            reward,
            self._time_seconds(state) - previous_time,
        )
        scene = export_scene(
            state,
            timestep,
            self.params,
            include_static=False,
            include_route_previews=True,
        )

        if mutate:
            self._state = state
            self._timestep = timestep
        return DispatchResult(timestep=timestep, action_results=action_results, scene=scene)

    def scene(
        self,
        *,
        include_static: bool = True,
        include_route_previews: bool = True,
    ) -> dict[str, Any]:
        return export_scene(
            self.state,
            self.timestep,
            self.params,
            include_static=include_static,
            include_route_previews=include_route_previews,
        )

    def _apply_dispatch_action(
        self,
        state: EnvState,
        reward,
        action: dict[str, Any],
    ) -> tuple[EnvState, jnp.ndarray, dict[str, Any]]:
        car_id = int(action.get("car_id", -1))
        kind = str(action.get("action", action.get("type", "wait"))).lower()
        result = {"car_id": car_id if car_id >= 0 else None, "action": kind, "valid": False}
        if car_id < 0 or car_id >= self.params.max_cars:
            result["reason"] = "unknown car_id"
            return state, reward, result
        if not self._car_dispatch_ready(state, car_id):
            result["reason"] = "car is not idle or is waiting for its next dispatch window"
            return state, reward, result

        if kind in {"assign", "assign_request", "pickup"}:
            return self._apply_assign_request(state, reward, car_id, action, result)
        if kind in {"reposition", "rebalance"}:
            return self._apply_reposition(state, reward, car_id, action, result)
        if kind == "wait":
            seconds = float(action.get("duration_seconds", self.default_wait_seconds))
            self._wait_until[car_id] = self._time_seconds(state) + max(0.0, seconds)
            result.update(
                {
                    "valid": True,
                    "reason": action.get("reason", "wait"),
                    "sleep_until_seconds": float(self._wait_until[car_id]),
                }
            )
            return state, reward, result

        result["reason"] = f"unsupported action {kind!r}"
        return state, reward, result

    def _apply_assign_request(
        self,
        state: EnvState,
        reward,
        car_id: int,
        action: dict[str, Any],
        result: dict[str, Any],
    ) -> tuple[EnvState, jnp.ndarray, dict[str, Any]]:
        request_id = int(action.get("request_id", -1))
        if request_id < 0 or request_id >= self.params.max_requests:
            result["reason"] = "unknown request_id"
            return state, reward, result
        if int(np.asarray(state.request_status)[request_id]) != REQUEST_QUEUED:
            result["reason"] = "request is not queued"
            return state, reward, result

        next_state, next_reward = _assign_request_to_car(
            state,
            self.params,
            jnp.asarray(request_id, dtype=jnp.int32),
            jnp.asarray(car_id, dtype=jnp.int32),
            reward,
        )
        self._wait_until[car_id] = -math.inf
        result.update(
            {
                "valid": True,
                "request_id": request_id,
                "reason": action.get("reason", "manual assignment"),
            }
        )
        return next_state, next_reward, result

    def _apply_reposition(
        self,
        state: EnvState,
        reward,
        car_id: int,
        action: dict[str, Any],
        result: dict[str, Any],
    ) -> tuple[EnvState, jnp.ndarray, dict[str, Any]]:
        try:
            target = self._target_node_from_action(action)
        except ValueError as exc:
            result["reason"] = str(exc)
            return state, reward, result

        current = int(np.asarray(state.car_nodes)[car_id])
        if target == current:
            self._wait_until[car_id] = self._time_seconds(state) + self.default_wait_seconds
            result.update(
                {
                    "valid": True,
                    "target_compact_node_id": target,
                    "reason": "car already at reposition target",
                    "sleep_until_seconds": float(self._wait_until[car_id]),
                }
            )
            return state, reward, result

        edge_id = int(np.asarray(self.params.graph.routing_next_edge)[current, target])
        if edge_id < 0:
            result.update(
                {
                    "target_compact_node_id": target,
                    "reason": "target is not reachable from current car node",
                }
            )
            return state, reward, result

        next_state = _start_auto_edge(
            state,
            self.params,
            jnp.asarray(car_id, dtype=jnp.int32),
            jnp.asarray(target, dtype=jnp.int32),
            jnp.asarray(-1, dtype=jnp.int32),
            CAR_REPOSITION,
        )
        self._wait_until[car_id] = -math.inf
        result.update(
            {
                "valid": True,
                "target_compact_node_id": target,
                "reason": action.get("reason", "manual reposition"),
            }
        )
        return next_state, reward, result

    def _advance_until_dispatch_decision(
        self,
        state: EnvState,
        reward,
    ) -> tuple[EnvState, jnp.ndarray]:
        loops = 0
        while (
            not bool(np.asarray(state.done))
            and not self._has_dispatch_decision(state)
            and loops < self.params.max_event_steps
        ):
            sim_next = float(np.asarray(_next_event_time(state, self.params)))
            sleep_next = self._next_sleep_ready_time(state)
            episode_end = float(np.asarray(self.params.start_time_seconds + self.params.episode_seconds))
            event_time = min(sim_next, sleep_next, episode_end)
            if not math.isfinite(event_time):
                event_time = episode_end

            state = state.replace(time_seconds=jnp.asarray(event_time, dtype=jnp.float32))
            if event_time >= episode_end:
                state = self._refresh_dispatch_decision(state.replace(done=jnp.asarray(True)))
                break

            state, reward = _process_events_at_time(state, self.params, reward)
            state = self._refresh_dispatch_decision(state)
            loops += 1

        if loops >= self.params.max_event_steps and not self._has_dispatch_decision(state):
            state = self._refresh_dispatch_decision(state.replace(done=jnp.asarray(True)))
        return self._refresh_dispatch_decision(state), reward

    def _target_node_from_action(self, action: dict[str, Any]) -> int:
        if "target_compact_node_id" in action:
            return self._validate_node_id(int(action["target_compact_node_id"]))
        if "target_node" in action:
            return self._validate_node_id(int(action["target_node"]))
        if "target_original_node_id" in action:
            original_ids = np.asarray(self.params.graph.original_node_ids)
            matches = np.flatnonzero(original_ids == int(action["target_original_node_id"]))
            if len(matches) == 0:
                raise ValueError("target_original_node_id is not in this graph")
            return int(matches[0])

        target = action.get("target")
        grid_cell = action.get("grid_cell")
        if isinstance(target, int):
            return self._validate_node_id(int(target))
        if isinstance(target, dict) and target.get("type") == "grid_cell":
            grid_cell = target
        if isinstance(grid_cell, dict):
            row = int(grid_cell["row"])
            col = int(grid_cell["col"])
            return nearest_compact_node_for_grid_cell(self.params.graph, row, col)
        if isinstance(grid_cell, (list, tuple)) and len(grid_cell) == 2:
            return nearest_compact_node_for_grid_cell(
                self.params.graph,
                int(grid_cell[0]),
                int(grid_cell[1]),
            )

        raise ValueError("reposition requires target node or grid_cell")

    def _validate_node_id(self, node_id: int) -> int:
        if node_id < 0 or node_id >= self.params.graph.num_nodes:
            raise ValueError("target node is outside the graph")
        return int(node_id)

    def _refresh_dispatch_decision(self, state: EnvState) -> EnvState:
        ids = self._dispatch_ready_car_ids(state)
        current = min(ids) if ids else -1
        return state.replace(
            current_car_id=jnp.asarray(current, dtype=jnp.int32),
            decision_required=jnp.asarray(current >= 0),
        )

    def _dispatch_ready_car_ids(self, state: EnvState) -> list[int]:
        mask = self._dispatch_decision_mask(state)
        return [int(car_id) for car_id in np.flatnonzero(mask)]

    def _dispatch_decision_mask(self, state: EnvState) -> np.ndarray:
        if bool(np.asarray(state.done)):
            return np.zeros((self.params.max_cars,), dtype=bool)
        time_seconds = self._time_seconds(state)
        statuses = np.asarray(state.car_status)
        return (statuses == CAR_DECISION) & (self._wait_until <= time_seconds + 1e-6)

    def _has_dispatch_decision(self, state: EnvState) -> bool:
        return bool(np.any(self._dispatch_decision_mask(state)))

    def _car_dispatch_ready(self, state: EnvState, car_id: int) -> bool:
        return bool(self._dispatch_decision_mask(state)[car_id])

    def _next_sleep_ready_time(self, state: EnvState) -> float:
        if bool(np.asarray(state.done)):
            return math.inf
        time_seconds = self._time_seconds(state)
        statuses = np.asarray(state.car_status)
        sleeping = (statuses == CAR_DECISION) & (self._wait_until > time_seconds + 1e-6)
        if not bool(np.any(sleeping)):
            return math.inf
        return float(np.min(self._wait_until[sleeping]))

    @staticmethod
    def _time_seconds(state: EnvState) -> float:
        return float(np.asarray(state.time_seconds))

    @staticmethod
    def _request_status_label(status: int) -> str:
        return {
            REQUEST_QUEUED: "queued",
            REQUEST_ASSIGNED: "assigned",
            REQUEST_ONBOARD: "onboard",
        }.get(status, "unknown")
