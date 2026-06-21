from __future__ import annotations

import jax
import numpy as np

from .env import EnvParams
from .types import (
    EnvState,
    NO_EDGE,
    NO_REQUEST,
    POLICY_CONTROLLED,
    REQ_ASSIGNED,
    REQ_PICKED_UP,
    REQ_QUEUED,
    Timestep,
    TO_DROPOFF,
    TO_PICKUP,
)


EVENT_NAMES = {
    1: "REQUEST_SPAWNED",
    2: "REQUEST_ASSIGNED",
    3: "REQUEST_PICKED_UP",
    4: "REQUEST_COMPLETED",
    5: "REQUEST_DROPPED",
    6: "OVERFLOW",
}

CAR_STATUS_NAMES = {
    POLICY_CONTROLLED: "POLICY_CONTROLLED",
    TO_PICKUP: "TO_PICKUP",
    TO_DROPOFF: "TO_DROPOFF",
}

REQUEST_STATUS_NAMES = {
    REQ_QUEUED: "QUEUED",
    REQ_ASSIGNED: "ASSIGNED_TO_PICKUP",
    REQ_PICKED_UP: "PICKED_UP",
}


def _scalar(value):
    if isinstance(value, np.ndarray):
        return value.item()
    if hasattr(value, "item"):
        return value.item()
    return value


def _node_payload(graph, node: int) -> dict:
    node = int(node)
    original = int(_scalar(graph.original_node_ids[node]))
    return {
        "id": node,
        "original_id": original,
        "lon": float(_scalar(graph.node_lon[node])),
        "lat": float(_scalar(graph.node_lat[node])),
        "x": float(_scalar(graph.node_x[node])),
        "y": float(_scalar(graph.node_y[node])),
    }


def _interpolate_node_position(graph, from_node: int, to_node: int, progress: float) -> tuple[float, float]:
    lon0 = float(_scalar(graph.node_lon[from_node]))
    lat0 = float(_scalar(graph.node_lat[from_node]))
    lon1 = float(_scalar(graph.node_lon[to_node]))
    lat1 = float(_scalar(graph.node_lat[to_node]))
    return lon0 + (lon1 - lon0) * progress, lat0 + (lat1 - lat0) * progress


def export_scene(state: EnvState, timestep: Timestep, params: EnvParams) -> dict:
    state = jax.device_get(state)
    timestep = jax.device_get(timestep)
    graph = jax.device_get(params.graph)
    sim_time = float(_scalar(state.sim_time_seconds))

    cars = []
    for car_id in range(params.max_cars):
        if not bool(_scalar(state.car_active[car_id])):
            continue
        edge_id = int(_scalar(state.car_edge_id[car_id]))
        start = float(_scalar(state.car_edge_start_time[car_id]))
        end = float(_scalar(state.car_edge_end_time[car_id]))
        progress = 0.0
        if edge_id != NO_EDGE and end > start:
            progress = max(0.0, min(1.0, (sim_time - start) / (end - start)))
        from_node = int(_scalar(state.car_from_node[car_id]))
        to_node = int(_scalar(state.car_to_node[car_id]))
        current_node = int(_scalar(state.car_node[car_id]))
        assigned = int(_scalar(state.car_assigned_request[car_id]))
        if edge_id != NO_EDGE:
            lon, lat = _interpolate_node_position(graph, from_node, to_node, progress)
        else:
            lon = float(_scalar(graph.node_lon[current_node]))
            lat = float(_scalar(graph.node_lat[current_node]))
        cars.append(
            {
                "id": car_id,
                "status": CAR_STATUS_NAMES.get(int(_scalar(state.car_status[car_id])), "UNKNOWN"),
                "current_node": current_node,
                "current_node_original_id": int(_scalar(graph.original_node_ids[current_node])),
                "from_node": from_node,
                "to_node": to_node,
                "edge_id": edge_id,
                "edge_original_id": int(_scalar(graph.edge_original_ids[edge_id])) if edge_id != NO_EDGE else None,
                "edge_progress": progress,
                "assigned_request_id": assigned if assigned != NO_REQUEST else None,
                "target_node": int(_scalar(state.car_target_node[car_id])),
                "lon": lon,
                "lat": lat,
            }
        )

    active_requests = []
    queued_requests = []
    for slot in range(params.max_active_requests):
        status = int(_scalar(state.request_status[slot]))
        if status not in REQUEST_STATUS_NAMES:
            continue
        pickup = int(_scalar(state.request_pickup_node[slot]))
        dropoff = int(_scalar(state.request_dropoff_node[slot]))
        req = {
            "slot": slot,
            "id": int(_scalar(state.request_ids[slot])),
            "status": REQUEST_STATUS_NAMES[status],
            "pickup_node": pickup,
            "dropoff_node": dropoff,
            "pickup": _node_payload(graph, pickup),
            "dropoff": _node_payload(graph, dropoff),
            "spawn_time_seconds": float(_scalar(state.request_spawn_time[slot])),
            "assigned_car_id": int(_scalar(state.request_assigned_car_id[slot])),
            "pickup_time_seconds": float(_scalar(state.request_pickup_time[slot])),
            "dropoff_time_seconds": float(_scalar(state.request_dropoff_time[slot])),
        }
        if status == REQ_QUEUED:
            queued_requests.append(req)
        else:
            active_requests.append(req)

    edges = []
    for edge_id in range(int(_scalar(graph.num_edges))):
        congestion = float(_scalar(state.edge_congestion[edge_id]))
        base_time = float(_scalar(graph.edge_base_travel_time_s[edge_id]))
        edges.append(
            {
                "edge_id": edge_id,
                "edge_original_id": int(_scalar(graph.edge_original_ids[edge_id])),
                "from_node": int(_scalar(graph.edge_from[edge_id])),
                "to_node": int(_scalar(graph.edge_to[edge_id])),
                "length_m": float(_scalar(graph.edge_length_m[edge_id])),
                "base_travel_time_seconds": base_time,
                "current_travel_time_seconds": base_time * congestion,
                "congestion": congestion,
            }
        )

    recent_events = []
    for idx in range(params.max_recent_events):
        code = int(_scalar(state.recent_event_codes[idx]))
        if code == 0:
            continue
        recent_events.append(
            {
                "code": code,
                "name": EVENT_NAMES.get(code, "UNKNOWN"),
                "car_id": int(_scalar(state.recent_event_car_ids[idx])),
                "request_id": int(_scalar(state.recent_event_request_ids[idx])),
                "time_seconds": float(_scalar(state.recent_event_times[idx])),
            }
        )

    return {
        "sim_time_seconds": sim_time,
        "dt_seconds": float(_scalar(timestep.dt_seconds)),
        "current_car_id": int(_scalar(timestep.current_car_id)),
        "current_node_id": int(_scalar(timestep.current_node_id)),
        "cars": cars,
        "active_requests": active_requests,
        "queued_requests": queued_requests,
        "edges": edges,
        "recent_events": recent_events,
        "metrics": {
            "requests_spawned": int(_scalar(state.metrics.requests_spawned)),
            "requests_queued": int(_scalar(state.metrics.requests_queued)),
            "requests_assigned": int(_scalar(state.metrics.requests_assigned)),
            "requests_picked_up": int(_scalar(state.metrics.requests_picked_up)),
            "requests_completed": int(_scalar(state.metrics.requests_completed)),
            "dropped_requests": int(_scalar(state.metrics.dropped_requests)),
            "queue_length": int(_scalar(state.metrics.queue_length)),
            "fleet_utilization": float(_scalar(state.metrics.fleet_utilization)),
            "empty_driving_time": float(_scalar(state.metrics.empty_driving_time)),
            "empty_driving_distance": float(_scalar(state.metrics.empty_driving_distance)),
        },
    }
