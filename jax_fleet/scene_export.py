from __future__ import annotations

from typing import Any

import numpy as np

from jax_fleet.env import (
    CAR_DECISION,
    CAR_REPOSITION,
    CAR_TO_DROPOFF,
    CAR_TO_PICKUP,
    REQUEST_ASSIGNED,
    REQUEST_COMPLETED,
    REQUEST_DROPPED,
    REQUEST_EMPTY,
    REQUEST_ONBOARD,
    REQUEST_QUEUED,
)
from jax_fleet.routing import shortest_path_edges
from jax_fleet.types import EnvParams, EnvState, Timestep


CAR_STATUS_LABELS = {
    CAR_DECISION: "decision",
    CAR_REPOSITION: "repositioning",
    CAR_TO_PICKUP: "to_pickup",
    CAR_TO_DROPOFF: "to_dropoff",
}

REQUEST_STATUS_LABELS = {
    REQUEST_EMPTY: "empty",
    REQUEST_QUEUED: "queued",
    REQUEST_ASSIGNED: "assigned",
    REQUEST_ONBOARD: "onboard",
    REQUEST_COMPLETED: "completed",
    REQUEST_DROPPED: "dropped",
}


def export_scene(
    state: EnvState,
    timestep: Timestep,
    params: EnvParams,
    *,
    include_static: bool = True,
    include_route_previews: bool = True,
) -> dict[str, Any]:
    graph = params.graph
    node_lonlat = np.asarray(graph.node_lonlat)
    original_ids = np.asarray(graph.original_node_ids)
    car_nodes = np.asarray(state.car_nodes)
    car_status = np.asarray(state.car_status)
    time_seconds = float(np.asarray(state.time_seconds))
    hour = int((time_seconds // 3600) % 24)

    cars = []
    edge_progress = []
    route_previews = []
    for car_id in range(params.max_cars):
        node = int(car_nodes[car_id])
        status = int(car_status[car_id])
        position = _car_position(state, params, car_id)
        car_payload = {
            "id": int(car_id),
            "node_id": int(original_ids[node]),
            "compact_node_id": node,
            "position": position,
            "status": CAR_STATUS_LABELS.get(status, "unknown"),
            "request_id": int(np.asarray(state.car_request_ids)[car_id]),
            "edge_id": int(np.asarray(state.car_edge_ids)[car_id]),
            "target_node_id": _node_id_or_none(original_ids, np.asarray(state.car_target_nodes)[car_id]),
            "goal_node_id": _node_id_or_none(original_ids, np.asarray(state.car_goal_nodes)[car_id]),
            "ready_time_seconds": float(np.asarray(state.car_ready_times)[car_id]),
        }
        cars.append(car_payload)
        if status != CAR_DECISION:
            progress = _edge_progress(state, car_id)
            edge_progress.append(
                {
                    "car_id": int(car_id),
                    "edge_id": int(np.asarray(state.car_edge_ids)[car_id]),
                    "status": CAR_STATUS_LABELS.get(status, "unknown"),
                    "progress": progress,
                    "from": node_lonlat[node].tolist(),
                    "to": node_lonlat[int(np.asarray(state.car_target_nodes)[car_id])].tolist(),
                }
            )
            goal = int(np.asarray(state.car_goal_nodes)[car_id])
            if include_route_previews and status in {CAR_TO_PICKUP, CAR_TO_DROPOFF} and 0 <= goal < graph.num_nodes:
                edge_ids = shortest_path_edges(graph, node, goal)
                route_previews.append(
                    {
                        "car_id": int(car_id),
                        "status": CAR_STATUS_LABELS.get(status, "unknown"),
                        "request_id": int(np.asarray(state.car_request_ids)[car_id]),
                        "goal_node_id": int(original_ids[goal]),
                        "goal_compact_node_id": goal,
                        "edge_ids": edge_ids,
                        "points": _route_preview_points(
                            node_lonlat=node_lonlat,
                            edge_targets=np.asarray(graph.edge_targets),
                            edge_ids=edge_ids,
                            position=position,
                            goal=goal,
                        ),
                    }
                )

    request_status = np.asarray(state.request_status)
    requests = []
    for request_id in range(params.max_requests):
        status = int(request_status[request_id])
        if status == REQUEST_EMPTY:
            continue
        origin = int(np.asarray(state.request_origin_nodes)[request_id])
        dest = int(np.asarray(state.request_dest_nodes)[request_id])
        requests.append(
            {
                "id": int(request_id),
                "status": REQUEST_STATUS_LABELS.get(status, "unknown"),
                "origin_node_id": int(original_ids[origin]) if origin >= 0 else None,
                "destination_node_id": int(original_ids[dest]) if dest >= 0 else None,
                "origin": node_lonlat[origin].tolist() if origin >= 0 else None,
                "destination": node_lonlat[dest].tolist() if dest >= 0 else None,
                "spawn_time_seconds": float(np.asarray(state.request_spawn_times)[request_id]),
                "pickup_time_seconds": _nullable_float(np.asarray(state.request_pickup_times)[request_id]),
                "assigned_car_id": int(np.asarray(state.request_assigned_car_ids)[request_id]),
            }
        )

    if include_static:
        congestion = [
            {
                "edge_id": int(edge_id),
                "source_node_id": int(original_ids[int(np.asarray(graph.edge_sources)[edge_id])]),
                "target_node_id": int(original_ids[int(np.asarray(graph.edge_targets)[edge_id])]),
                "source": node_lonlat[int(np.asarray(graph.edge_sources)[edge_id])].tolist(),
                "target": node_lonlat[int(np.asarray(graph.edge_targets)[edge_id])].tolist(),
                "congestion": float(np.asarray(graph.edge_congestion)[edge_id, hour]),
            }
            for edge_id in range(graph.num_edges)
        ]
    else:
        congestion = []

    action_mask = np.asarray(timestep.observation.action_mask).astype(bool)
    status_counts = {
        label: int((car_status == status).sum())
        for status, label in CAR_STATUS_LABELS.items()
    }
    request_counts = {
        label: int((request_status == status).sum())
        for status, label in REQUEST_STATUS_LABELS.items()
        if status != REQUEST_EMPTY
    }
    active_requests = int(
        np.isin(request_status, [REQUEST_QUEUED, REQUEST_ASSIGNED, REQUEST_ONBOARD]).sum()
    )
    recent_pickup_count = int(np.asarray(state.metrics.recent_pickup_wait_count))
    recent_pickup_total = float(np.asarray(state.metrics.recent_pickup_wait_seconds).sum())
    avg_pickup_wait_last_10 = recent_pickup_total / max(1, recent_pickup_count)
    metrics = {
        "completed_requests": int(np.asarray(state.metrics.completed_requests)),
        "dropped_requests": int(np.asarray(state.metrics.dropped_requests)),
        "queued_requests": int(np.asarray(state.metrics.queued_requests)),
        "active_requests": active_requests,
        "target_active_requests": int(params.target_active_requests),
        "invalid_actions": int(np.asarray(state.metrics.invalid_actions)),
        "pickup_wait_seconds": float(np.asarray(state.metrics.pickup_wait_seconds)),
        "avg_pickup_wait_last_10_seconds": avg_pickup_wait_last_10,
        "recent_pickup_wait_count": recent_pickup_count,
        "aggregate_reward": float(np.asarray(state.metrics.aggregate_reward)),
    }

    return {
        "time_seconds": time_seconds,
        "current_car_id": int(np.asarray(state.current_car_id)),
        "decision_required": bool(np.asarray(state.decision_required)),
        "done": bool(np.asarray(state.done)),
        "step_count": int(np.asarray(state.step_count)),
        "discount": float(np.asarray(timestep.discount)),
        "action_mask": action_mask.tolist(),
        "graph": {
            "num_nodes": int(graph.num_nodes),
            "num_edges": int(graph.num_edges),
            "max_degree": int(graph.max_degree),
            "bounds": np.asarray(graph.bounds).astype(float).tolist(),
        },
        "cars": cars,
        "requests": requests,
        "congestion": congestion,
        "status_counts": {
            "cars": status_counts,
            "requests": request_counts,
        },
        "metrics": metrics,
        "recent_events": {
            "reward": float(np.asarray(timestep.reward)),
            "dt_seconds": float(np.asarray(timestep.dt_seconds)),
            **metrics,
        },
        "edge_progress": edge_progress,
        "route_previews": route_previews,
    }


def _car_position(state: EnvState, params: EnvParams, car_id: int) -> list[float]:
    graph = params.graph
    node_lonlat = np.asarray(graph.node_lonlat)
    status = int(np.asarray(state.car_status)[car_id])
    node = int(np.asarray(state.car_nodes)[car_id])
    if status == CAR_DECISION:
        return node_lonlat[node].tolist()

    progress = _edge_progress(state, car_id)
    target = int(np.asarray(state.car_target_nodes)[car_id])
    start = node_lonlat[node]
    end = node_lonlat[target]
    return (start * (1.0 - progress) + end * progress).tolist()


def _edge_progress(state: EnvState, car_id: int) -> float:
    now = float(np.asarray(state.time_seconds))
    depart = float(np.asarray(state.car_departure_times)[car_id])
    duration = max(1e-6, float(np.asarray(state.car_edge_durations)[car_id]))
    return float(np.clip((now - depart) / duration, 0.0, 1.0))


def _nullable_float(value) -> float | None:
    value = float(value)
    if np.isnan(value):
        return None
    return value


def _route_preview_points(
    *,
    node_lonlat: np.ndarray,
    edge_targets: np.ndarray,
    edge_ids: list[int],
    position: list[float],
    goal: int,
) -> list[list[float]]:
    points = [list(position)]
    for edge_id in edge_ids:
        target = int(edge_targets[int(edge_id)])
        next_point = node_lonlat[target].tolist()
        if next_point != points[-1]:
            points.append(next_point)
    if len(points) == 1 and 0 <= goal < len(node_lonlat):
        goal_point = node_lonlat[goal].tolist()
        if goal_point != points[-1]:
            points.append(goal_point)
    return points


def _node_id_or_none(original_ids: np.ndarray, value) -> int | None:
    value = int(value)
    if value < 0 or value >= len(original_ids):
        return None
    return int(original_ids[value])
