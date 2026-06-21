from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from jax_fleet.observations import build_observation
from jax_fleet.types import EnvMetrics, EnvParams, EnvState, GraphArrays, Timestep


CAR_DECISION = 0
CAR_REPOSITION = 1
CAR_TO_PICKUP = 2
CAR_TO_DROPOFF = 3

REQUEST_EMPTY = 0
REQUEST_QUEUED = 1
REQUEST_ASSIGNED = 2
REQUEST_ONBOARD = 3
REQUEST_COMPLETED = 4
REQUEST_DROPPED = 5

_INF = 1.0e20
_RECENT_PICKUP_WAIT_WINDOW = 10
_POPULATION_HOURLY_MULTIPLIER = np.asarray(
    [
        0.82,
        0.78,
        0.74,
        0.72,
        0.74,
        0.82,
        0.96,
        1.10,
        1.18,
        1.20,
        1.16,
        1.10,
        1.06,
        1.02,
        1.00,
        1.04,
        1.12,
        1.20,
        1.14,
        1.04,
        0.96,
        0.90,
        0.86,
        0.84,
    ],
    dtype=np.float32,
)


def make_env_params(
    graph: GraphArrays,
    *,
    max_cars: int = 16,
    max_requests: int = 128,
    initial_car_nodes: list[int] | np.ndarray | None = None,
    preplanned_requests: list[dict[str, Any]] | None = None,
    start_time_seconds: float = 0.0,
    episode_seconds: float = 3600.0,
    spawn_rate_per_minute: float = 0.0,
    target_active_requests: int | None = None,
    target_active_request_fraction: float = 0.0,
    density_spawn_patience_seconds: float = np.inf,
    density_destination_time_shift_seconds: float = 2.0 * 3600.0,
    wait_time_scale: float = 1.0 / 60.0,
    gamma: float = 0.99,
    discount_time_unit_seconds: float = 60.0,
    raster_size: int = 50,
    max_event_steps: int = 512,
    assignment_max_route_edges: int = 15,
) -> EnvParams:
    if initial_car_nodes is None:
        initial = np.zeros((max_cars,), dtype=np.int32)
    else:
        initial = np.asarray(initial_car_nodes, dtype=np.int32)
        if initial.shape[0] != max_cars:
            raise ValueError("initial_car_nodes length must match max_cars")
    if np.any(initial < 0) or np.any(initial >= graph.num_nodes):
        raise ValueError("initial_car_nodes contains ids outside the graph")
    if target_active_requests is None:
        target_active = int(np.floor(max_cars * float(target_active_request_fraction)))
    else:
        target_active = int(target_active_requests)
    target_active = int(np.clip(target_active, 0, max_requests))

    scheduled = list(preplanned_requests or [])
    spawn_times = np.full((len(scheduled),), np.inf, dtype=np.float32)
    origin_nodes = np.full((len(scheduled),), -1, dtype=np.int32)
    dest_nodes = np.full((len(scheduled),), -1, dtype=np.int32)
    deadline_times = np.full((len(scheduled),), np.inf, dtype=np.float32)
    for idx, request in enumerate(scheduled):
        spawn_times[idx] = float(request["spawn_time_s"])
        origin_nodes[idx] = int(request["origin"])
        dest_nodes[idx] = int(request["destination"])
        patience = request.get("patience_s", request.get("patience_seconds"))
        deadline_times[idx] = (
            float(request.get("deadline_s", request.get("deadline_seconds")))
            if request.get("deadline_s", request.get("deadline_seconds")) is not None
            else spawn_times[idx] + float(patience)
            if patience is not None
            else np.inf
        )

    return EnvParams(
        graph=graph,
        max_cars=max_cars,
        max_requests=max_requests,
        raster_size=raster_size,
        max_event_steps=max_event_steps,
        target_active_requests=target_active,
        assignment_max_route_edges=max(0, int(assignment_max_route_edges)),
        initial_car_nodes=jnp.asarray(initial, dtype=jnp.int32),
        start_time_seconds=jnp.asarray(start_time_seconds, dtype=jnp.float32),
        episode_seconds=jnp.asarray(episode_seconds, dtype=jnp.float32),
        spawn_rate_per_minute=jnp.asarray(spawn_rate_per_minute, dtype=jnp.float32),
        density_spawn_patience_seconds=jnp.asarray(density_spawn_patience_seconds, dtype=jnp.float32),
        density_destination_time_shift_seconds=jnp.asarray(
            density_destination_time_shift_seconds,
            dtype=jnp.float32,
        ),
        wait_time_scale=jnp.asarray(wait_time_scale, dtype=jnp.float32),
        gamma=jnp.asarray(gamma, dtype=jnp.float32),
        discount_time_unit_seconds=jnp.asarray(discount_time_unit_seconds, dtype=jnp.float32),
        preplanned_spawn_times=jnp.asarray(spawn_times, dtype=jnp.float32),
        preplanned_origin_nodes=jnp.asarray(origin_nodes, dtype=jnp.int32),
        preplanned_dest_nodes=jnp.asarray(dest_nodes, dtype=jnp.int32),
        preplanned_deadline_times=jnp.asarray(deadline_times, dtype=jnp.float32),
        node_density_by_hour=jnp.asarray(_precompute_node_density_by_hour(graph), dtype=jnp.float32),
        edge_raster_by_hour=jnp.asarray(_precompute_edge_raster_by_hour(graph, raster_size), dtype=jnp.float32),
    )


def reset(rng, params: EnvParams) -> tuple[EnvState, Timestep]:
    rng, spawn_rng = jax.random.split(rng)
    start = params.start_time_seconds
    state = EnvState(
        rng=rng,
        time_seconds=start,
        car_nodes=params.initial_car_nodes,
        car_status=jnp.full((params.max_cars,), CAR_DECISION, dtype=jnp.int32),
        car_edge_ids=jnp.full((params.max_cars,), -1, dtype=jnp.int32),
        car_target_nodes=params.initial_car_nodes,
        car_goal_nodes=jnp.full((params.max_cars,), -1, dtype=jnp.int32),
        car_request_ids=jnp.full((params.max_cars,), -1, dtype=jnp.int32),
        car_ready_times=jnp.full((params.max_cars,), start, dtype=jnp.float32),
        car_departure_times=jnp.full((params.max_cars,), start, dtype=jnp.float32),
        car_edge_durations=jnp.zeros((params.max_cars,), dtype=jnp.float32),
        request_status=jnp.full((params.max_requests,), REQUEST_EMPTY, dtype=jnp.int32),
        request_origin_nodes=jnp.full((params.max_requests,), -1, dtype=jnp.int32),
        request_dest_nodes=jnp.full((params.max_requests,), -1, dtype=jnp.int32),
        request_spawn_times=jnp.full((params.max_requests,), jnp.inf, dtype=jnp.float32),
        request_deadline_times=jnp.full((params.max_requests,), jnp.inf, dtype=jnp.float32),
        request_assigned_car_ids=jnp.full((params.max_requests,), -1, dtype=jnp.int32),
        request_pickup_times=jnp.full((params.max_requests,), jnp.nan, dtype=jnp.float32),
        current_car_id=jnp.asarray(0, dtype=jnp.int32),
        decision_required=jnp.asarray(True),
        done=jnp.asarray(False),
        next_random_spawn_time=_sample_next_spawn_time(spawn_rng, start, params.spawn_rate_per_minute),
        next_scheduled_request_index=jnp.asarray(0, dtype=jnp.int32),
        step_count=jnp.asarray(0, dtype=jnp.int32),
        metrics=EnvMetrics(
            invalid_actions=jnp.asarray(0, dtype=jnp.int32),
            dropped_requests=jnp.asarray(0, dtype=jnp.int32),
            completed_requests=jnp.asarray(0, dtype=jnp.int32),
            queued_requests=jnp.asarray(0, dtype=jnp.int32),
            pickup_wait_seconds=jnp.asarray(0.0, dtype=jnp.float32),
            aggregate_reward=jnp.asarray(0.0, dtype=jnp.float32),
            recent_pickup_wait_seconds=jnp.zeros((_RECENT_PICKUP_WAIT_WINDOW,), dtype=jnp.float32),
            recent_pickup_wait_count=jnp.asarray(0, dtype=jnp.int32),
            recent_pickup_wait_index=jnp.asarray(0, dtype=jnp.int32),
        ),
    )
    state, reward = _process_events_at_time(state, params, jnp.asarray(0.0, dtype=jnp.float32))
    state, reward = _advance_until_decision(state, params, reward)
    state = _add_transition_reward(state, reward)
    return state, _make_timestep(state, params, reward, state.time_seconds - start)


def step(state: EnvState, action, params: EnvParams) -> tuple[EnvState, Timestep]:
    previous_time = state.time_seconds

    def do_step(s: EnvState):
        applied = _apply_policy_action(s, action, params)
        next_state, reward = _process_events_at_time(applied, params, jnp.asarray(0.0, dtype=jnp.float32))
        next_state, reward = _advance_until_decision(next_state, params, reward)
        next_state = next_state.replace(step_count=next_state.step_count + 1)
        next_state = _add_transition_reward(next_state, reward)
        return next_state, reward

    def noop(s: EnvState):
        return s, jnp.asarray(0.0, dtype=jnp.float32)

    state, reward = jax.lax.cond(
        state.done | (~state.decision_required),
        noop,
        do_step,
        state,
    )
    return state, _make_timestep(state, params, reward, state.time_seconds - previous_time)


def nearest_eligible_car_by_eta(state: EnvState, request_id, params: EnvParams):
    request_id = jnp.asarray(request_id, dtype=jnp.int32)
    origin = state.request_origin_nodes[request_id]
    car_nodes = jnp.clip(state.car_nodes, 0, params.graph.num_nodes - 1)
    etas = params.graph.routing_travel_time_s[car_nodes, origin]
    within_range = _route_within_edge_range(
        params.graph,
        car_nodes,
        origin,
        params.assignment_max_route_edges,
    )
    eligible = (state.car_status == CAR_DECISION) & within_range
    scored = jnp.where(eligible, etas, _INF)
    best = jnp.argmin(scored).astype(jnp.int32)
    return jnp.where(jnp.isfinite(scored[best]) & (scored[best] < _INF / 2.0), best, -1)


def _route_within_edge_range(
    graph: GraphArrays,
    source_nodes,
    target_node,
    max_route_edges: int,
):
    target = jnp.clip(jnp.asarray(target_node, dtype=jnp.int32), 0, graph.num_nodes - 1)
    current = jnp.clip(jnp.asarray(source_nodes, dtype=jnp.int32), 0, graph.num_nodes - 1)
    reached = current == target

    def body(_, carry):
        node, done = carry
        edge_id = graph.routing_next_edge[node, target]
        can_step = (~done) & (edge_id >= 0)
        next_node = graph.edge_targets[jnp.clip(edge_id, 0, graph.num_edges - 1)]
        node = jnp.where(can_step, next_node, node)
        done = done | (can_step & (node == target))
        return node, done

    _, reached = jax.lax.fori_loop(
        0,
        max(0, int(max_route_edges)),
        body,
        (current, reached),
    )
    return reached


def _apply_policy_action(state: EnvState, action, params: EnvParams) -> EnvState:
    graph = params.graph
    car = state.current_car_id
    node = jnp.clip(state.car_nodes[car], 0, graph.num_nodes - 1)
    mask = graph.outgoing_mask[node]
    clipped_action = jnp.clip(jnp.asarray(action, dtype=jnp.int32), 0, graph.max_degree - 1)
    fallback_slot = jnp.argmax(mask.astype(jnp.int32)).astype(jnp.int32)
    action_valid = mask[clipped_action]
    slot = jnp.where(action_valid, clipped_action, fallback_slot)
    edge_id = graph.outgoing_edge_ids[node, slot]
    edge_valid = edge_id >= 0
    safe_edge = jnp.clip(edge_id, 0, graph.num_edges - 1)
    target = graph.edge_targets[safe_edge]
    duration = _edge_travel_time_at(safe_edge, state.time_seconds, graph)
    ready = state.time_seconds + duration
    status = jnp.where(edge_valid, CAR_REPOSITION, CAR_DECISION)

    metrics = state.metrics.replace(
        invalid_actions=state.metrics.invalid_actions + (~action_valid | ~edge_valid).astype(jnp.int32)
    )
    return state.replace(
        car_status=state.car_status.at[car].set(status),
        car_edge_ids=state.car_edge_ids.at[car].set(jnp.where(edge_valid, edge_id, -1)),
        car_target_nodes=state.car_target_nodes.at[car].set(jnp.where(edge_valid, target, node)),
        car_goal_nodes=state.car_goal_nodes.at[car].set(-1),
        car_request_ids=state.car_request_ids.at[car].set(-1),
        car_ready_times=state.car_ready_times.at[car].set(jnp.where(edge_valid, ready, state.time_seconds)),
        car_departure_times=state.car_departure_times.at[car].set(state.time_seconds),
        car_edge_durations=state.car_edge_durations.at[car].set(jnp.where(edge_valid, duration, 0.0)),
        decision_required=jnp.asarray(False),
        metrics=metrics,
    )


def _process_events_at_time(
    state: EnvState,
    params: EnvParams,
    reward,
) -> tuple[EnvState, jnp.ndarray]:
    state = _expire_queued_requests(state, params)
    state = _spawn_preplanned_requests(state, params)
    state = _spawn_random_request_if_due(state, params)
    state, reward = _process_car_arrivals(state, params, reward)
    state = _spawn_density_top_up_requests(state, params)
    state, reward = _assign_queued_requests(state, params, reward)
    state = _refresh_decision(state, params)
    return state, reward


def _advance_until_decision(
    state: EnvState,
    params: EnvParams,
    reward,
) -> tuple[EnvState, jnp.ndarray]:
    def keep_going(carry):
        loop_state, _, loops = carry
        return (
            (~loop_state.decision_required)
            & (~loop_state.done)
            & (loops < params.max_event_steps)
        )

    def body(carry):
        loop_state, loop_reward, loops = carry
        next_time = _next_event_time(loop_state, params)
        episode_end = params.start_time_seconds + params.episode_seconds
        event_time = jnp.minimum(next_time, episode_end)
        loop_state = loop_state.replace(time_seconds=event_time)

        def finish_episode(s):
            return _refresh_decision(s.replace(done=jnp.asarray(True)), params), loop_reward

        def process_event(s):
            return _process_events_at_time(s, params, loop_reward)

        loop_state, loop_reward = jax.lax.cond(
            event_time >= episode_end,
            finish_episode,
            process_event,
            loop_state,
        )
        return loop_state, loop_reward, loops + 1

    state, reward, loops = jax.lax.while_loop(
        keep_going,
        body,
        (state, reward, jnp.asarray(0, dtype=jnp.int32)),
    )
    overflow = loops >= params.max_event_steps
    state = jax.lax.cond(
        overflow & (~state.decision_required),
        lambda s: _refresh_decision(s.replace(done=jnp.asarray(True)), params),
        lambda s: s,
        state,
    )
    return state, reward


def _next_event_time(state: EnvState, params: EnvParams):
    active = state.car_status != CAR_DECISION
    car_time = jnp.min(jnp.where(active, state.car_ready_times, _INF))
    preplanned_time = _next_scheduled_request_time(state, params)
    queued_deadline = (state.request_status == REQUEST_QUEUED) & jnp.isfinite(state.request_deadline_times)
    deadline_time = jnp.min(
        jnp.where(
            queued_deadline & (state.request_deadline_times > state.time_seconds),
            state.request_deadline_times,
            _INF,
        )
    )
    random_time = jnp.where(params.spawn_rate_per_minute > 0.0, state.next_random_spawn_time, _INF)
    episode_end = params.start_time_seconds + params.episode_seconds
    return jnp.minimum(
        jnp.minimum(jnp.minimum(car_time, preplanned_time), deadline_time),
        jnp.minimum(random_time, episode_end),
    )


def _next_scheduled_request_time(state: EnvState, params: EnvParams):
    schedule_len = params.preplanned_spawn_times.shape[0]
    if schedule_len == 0:
        return jnp.asarray(_INF, dtype=jnp.float32)
    idx = state.next_scheduled_request_index
    safe_idx = jnp.minimum(idx, schedule_len - 1)
    return jnp.where(idx < schedule_len, params.preplanned_spawn_times[safe_idx], _INF)


def _expire_queued_requests(state: EnvState, params: EnvParams) -> EnvState:
    del params
    expired = (
        (state.request_status == REQUEST_QUEUED)
        & jnp.isfinite(state.request_deadline_times)
        & (state.time_seconds >= state.request_deadline_times)
    )
    dropped = expired.sum().astype(jnp.int32)
    status = jnp.where(expired, REQUEST_DROPPED, state.request_status)
    metrics = state.metrics.replace(
        dropped_requests=state.metrics.dropped_requests + dropped,
        queued_requests=(status == REQUEST_QUEUED).sum().astype(jnp.int32),
    )
    return state.replace(
        request_status=status,
        request_assigned_car_ids=jnp.where(expired, -1, state.request_assigned_car_ids),
        metrics=metrics,
    )


def _spawn_preplanned_requests(state: EnvState, params: EnvParams) -> EnvState:
    schedule_len = params.preplanned_spawn_times.shape[0]
    if schedule_len == 0:
        return state.replace(
            metrics=state.metrics.replace(
                queued_requests=(state.request_status == REQUEST_QUEUED).sum().astype(jnp.int32)
            )
        )

    def due(s: EnvState):
        idx = s.next_scheduled_request_index
        safe_idx = jnp.minimum(idx, schedule_len - 1)
        return (idx < schedule_len) & (params.preplanned_spawn_times[safe_idx] <= s.time_seconds)

    def body(s: EnvState):
        idx = s.next_scheduled_request_index
        safe_idx = jnp.minimum(idx, schedule_len - 1)
        available = _request_slot_available(s.request_status)
        has_slot = jnp.any(available)
        slot = jnp.argmax(available.astype(jnp.int32)).astype(jnp.int32)

        def write(inner: EnvState):
            return inner.replace(
                request_status=inner.request_status.at[slot].set(REQUEST_QUEUED),
                request_origin_nodes=inner.request_origin_nodes.at[slot].set(
                    params.preplanned_origin_nodes[safe_idx]
                ),
                request_dest_nodes=inner.request_dest_nodes.at[slot].set(params.preplanned_dest_nodes[safe_idx]),
                request_spawn_times=inner.request_spawn_times.at[slot].set(
                    params.preplanned_spawn_times[safe_idx]
                ),
                request_deadline_times=inner.request_deadline_times.at[slot].set(
                    params.preplanned_deadline_times[safe_idx]
                ),
                request_assigned_car_ids=inner.request_assigned_car_ids.at[slot].set(-1),
                request_pickup_times=inner.request_pickup_times.at[slot].set(jnp.nan),
            )

        def drop(inner: EnvState):
            return inner.replace(
                metrics=inner.metrics.replace(
                    dropped_requests=inner.metrics.dropped_requests + jnp.asarray(1, dtype=jnp.int32)
                )
            )

        updated = jax.lax.cond(has_slot, write, drop, s)
        return updated.replace(next_scheduled_request_index=idx + jnp.asarray(1, dtype=jnp.int32))

    state = jax.lax.while_loop(due, body, state)
    return state.replace(
        metrics=state.metrics.replace(
            queued_requests=(state.request_status == REQUEST_QUEUED).sum().astype(jnp.int32)
        )
    )


def _spawn_random_request_if_due(state: EnvState, params: EnvParams) -> EnvState:
    should_spawn = (params.spawn_rate_per_minute > 0.0) & (state.next_random_spawn_time <= state.time_seconds)

    def spawn(s: EnvState):
        empty = _request_slot_available(s.request_status)
        has_empty = jnp.any(empty)
        slot = jnp.argmax(empty.astype(jnp.int32)).astype(jnp.int32)
        rng, origin_rng, dest_rng, spawn_rng = jax.random.split(s.rng, 4)
        origin = jax.random.randint(origin_rng, (), 0, params.graph.num_nodes, dtype=jnp.int32)
        offset = jax.random.randint(dest_rng, (), 1, params.graph.num_nodes, dtype=jnp.int32)
        dest = (origin + offset) % params.graph.num_nodes
        next_spawn = _sample_next_spawn_time(spawn_rng, s.time_seconds, params.spawn_rate_per_minute)

        def write_request(inner: EnvState):
            return inner.replace(
                request_status=inner.request_status.at[slot].set(REQUEST_QUEUED),
                request_origin_nodes=inner.request_origin_nodes.at[slot].set(origin),
                request_dest_nodes=inner.request_dest_nodes.at[slot].set(dest),
                request_spawn_times=inner.request_spawn_times.at[slot].set(inner.time_seconds),
                request_deadline_times=inner.request_deadline_times.at[slot].set(jnp.inf),
                request_assigned_car_ids=inner.request_assigned_car_ids.at[slot].set(-1),
                request_pickup_times=inner.request_pickup_times.at[slot].set(jnp.nan),
                rng=rng,
                next_random_spawn_time=next_spawn,
            )

        def drop_request(inner: EnvState):
            return inner.replace(
                rng=rng,
                next_random_spawn_time=next_spawn,
                metrics=inner.metrics.replace(
                    dropped_requests=inner.metrics.dropped_requests + jnp.asarray(1, dtype=jnp.int32)
                ),
            )

        updated = jax.lax.cond(has_empty, write_request, drop_request, s)
        return updated.replace(
            metrics=updated.metrics.replace(
                queued_requests=(updated.request_status == REQUEST_QUEUED).sum().astype(jnp.int32)
            )
        )

    return jax.lax.cond(should_spawn, spawn, lambda s: s, state)


def _spawn_density_top_up_requests(state: EnvState, params: EnvParams) -> EnvState:
    if params.target_active_requests <= 0:
        return state

    active_count = _active_request_mask(state.request_status).sum().astype(jnp.int32)
    deficit = jnp.maximum(
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(params.target_active_requests, dtype=jnp.int32) - active_count,
    )

    def body(_, carry):
        s, spawned = carry
        available = _request_slot_available(s.request_status)
        has_slot = jnp.any(available)
        slot = jnp.argmax(available.astype(jnp.int32)).astype(jnp.int32)
        should_spawn = (spawned < deficit) & has_slot

        def write(inner_carry):
            inner, count = inner_carry
            rng, origin_rng, dest_rng, offset_rng = jax.random.split(inner.rng, 4)
            origin = _sample_node_from_density(origin_rng, inner.time_seconds, params)
            raw_dest = _sample_node_from_density(
                dest_rng,
                inner.time_seconds + params.density_destination_time_shift_seconds,
                params,
            )
            if params.graph.num_nodes > 1:
                offset = jax.random.randint(
                    offset_rng,
                    (),
                    1,
                    params.graph.num_nodes,
                    dtype=jnp.int32,
                )
                dest = jnp.where(raw_dest == origin, (origin + offset) % params.graph.num_nodes, raw_dest)
            else:
                dest = raw_dest
            deadline = jnp.where(
                jnp.isfinite(params.density_spawn_patience_seconds),
                inner.time_seconds + params.density_spawn_patience_seconds,
                jnp.asarray(jnp.inf, dtype=jnp.float32),
            )
            updated = inner.replace(
                request_status=inner.request_status.at[slot].set(REQUEST_QUEUED),
                request_origin_nodes=inner.request_origin_nodes.at[slot].set(origin),
                request_dest_nodes=inner.request_dest_nodes.at[slot].set(dest),
                request_spawn_times=inner.request_spawn_times.at[slot].set(inner.time_seconds),
                request_deadline_times=inner.request_deadline_times.at[slot].set(deadline),
                request_assigned_car_ids=inner.request_assigned_car_ids.at[slot].set(-1),
                request_pickup_times=inner.request_pickup_times.at[slot].set(jnp.nan),
                rng=rng,
            )
            return updated, count + jnp.asarray(1, dtype=jnp.int32)

        return jax.lax.cond(should_spawn, write, lambda c: c, (s, spawned))

    state, _ = jax.lax.fori_loop(
        0,
        params.max_requests,
        body,
        (state, jnp.asarray(0, dtype=jnp.int32)),
    )
    return state.replace(
        metrics=state.metrics.replace(
            queued_requests=(state.request_status == REQUEST_QUEUED).sum().astype(jnp.int32)
        )
    )


def _sample_node_from_density(rng, time_seconds, params: EnvParams):
    weights = _node_density_at(time_seconds, params)
    logits = jnp.log(jnp.maximum(weights, 1.0e-12))
    return jax.random.categorical(rng, logits).astype(jnp.int32)


def _node_density_at(time_seconds, params: EnvParams):
    hour_float = (time_seconds / 3600.0) % 24.0
    h0 = jnp.floor(hour_float).astype(jnp.int32)
    h1 = (h0 + 1) % 24
    frac = hour_float - jnp.floor(hour_float)
    weights = params.node_density_by_hour[h0] * (1.0 - frac) + params.node_density_by_hour[h1] * frac
    return jnp.maximum(weights, 1.0e-6)


def _request_slot_available(status):
    return (status == REQUEST_EMPTY) | (status == REQUEST_COMPLETED) | (status == REQUEST_DROPPED)


def _active_request_mask(status):
    return (status == REQUEST_QUEUED) | (status == REQUEST_ASSIGNED) | (status == REQUEST_ONBOARD)


def _process_car_arrivals(
    state: EnvState,
    params: EnvParams,
    reward,
) -> tuple[EnvState, jnp.ndarray]:
    def body(car, carry):
        loop_state, loop_reward = carry
        arrived = (loop_state.car_status[car] != CAR_DECISION) & (
            loop_state.car_ready_times[car] <= loop_state.time_seconds
        )
        return jax.lax.cond(
            arrived,
            lambda c: _handle_car_arrival(c[0], params, c[1], car),
            lambda c: c,
            (loop_state, loop_reward),
        )

    return jax.lax.fori_loop(0, params.max_cars, body, (state, reward))


def _handle_car_arrival(
    state: EnvState,
    params: EnvParams,
    reward,
    car,
) -> tuple[EnvState, jnp.ndarray]:
    status = state.car_status[car]
    arrived_node = state.car_target_nodes[car]
    state = state.replace(car_nodes=state.car_nodes.at[car].set(arrived_node))

    def finish_reposition(carry):
        s, r = carry
        return (
            s.replace(
                car_status=s.car_status.at[car].set(CAR_DECISION),
                car_edge_ids=s.car_edge_ids.at[car].set(-1),
                car_goal_nodes=s.car_goal_nodes.at[car].set(-1),
                car_request_ids=s.car_request_ids.at[car].set(-1),
            ),
            r,
        )

    def continue_or_finish_pickup(carry):
        s, r = carry
        at_goal = s.car_nodes[car] == s.car_goal_nodes[car]
        return jax.lax.cond(
            at_goal,
            lambda c: _finish_pickup(c[0], params, c[1], car),
            lambda c: (_start_auto_edge(c[0], params, car, c[0].car_goal_nodes[car], c[0].car_request_ids[car], CAR_TO_PICKUP), c[1]),
            (s, r),
        )

    def continue_or_finish_dropoff(carry):
        s, r = carry
        at_goal = s.car_nodes[car] == s.car_goal_nodes[car]
        return jax.lax.cond(
            at_goal,
            lambda c: _finish_dropoff(c[0], params, c[1], car),
            lambda c: (_start_auto_edge(c[0], params, car, c[0].car_goal_nodes[car], c[0].car_request_ids[car], CAR_TO_DROPOFF), c[1]),
            (s, r),
        )

    state, reward = jax.lax.switch(
        jnp.clip(status, 0, 3),
        [
            lambda c: c,
            finish_reposition,
            continue_or_finish_pickup,
            continue_or_finish_dropoff,
        ],
        (state, reward),
    )
    return state, reward


def _assign_queued_requests(
    state: EnvState,
    params: EnvParams,
    reward,
) -> tuple[EnvState, jnp.ndarray]:
    def body(request_id, carry):
        loop_state, loop_reward = carry
        car_id = nearest_eligible_car_by_eta(loop_state, request_id, params)
        should_assign = (loop_state.request_status[request_id] == REQUEST_QUEUED) & (car_id >= 0)
        return jax.lax.cond(
            should_assign,
            lambda c: _assign_request_to_car(c[0], params, request_id, car_id, c[1]),
            lambda c: c,
            (loop_state, loop_reward),
        )

    state, reward = jax.lax.fori_loop(0, params.max_requests, body, (state, reward))
    queued = (state.request_status == REQUEST_QUEUED).sum().astype(jnp.int32)
    return state.replace(metrics=state.metrics.replace(queued_requests=queued)), reward


def _assign_request_to_car(
    state: EnvState,
    params: EnvParams,
    request_id,
    car,
    reward,
) -> tuple[EnvState, jnp.ndarray]:
    origin = state.request_origin_nodes[request_id]
    dest = state.request_dest_nodes[request_id]
    at_origin = state.car_nodes[car] == origin
    state = state.replace(request_assigned_car_ids=state.request_assigned_car_ids.at[request_id].set(car))

    def pickup_now(carry):
        s, r = carry
        s, r = _record_pickup_wait_reward(s, request_id, params, r)
        s = s.replace(
            request_status=s.request_status.at[request_id].set(REQUEST_ONBOARD),
            request_pickup_times=s.request_pickup_times.at[request_id].set(s.time_seconds),
            car_request_ids=s.car_request_ids.at[car].set(request_id),
            car_goal_nodes=s.car_goal_nodes.at[car].set(dest),
        )
        return _start_auto_edge(s, params, car, dest, request_id, CAR_TO_DROPOFF), r

    def drive_to_pickup(carry):
        s, r = carry
        s = s.replace(
            request_status=s.request_status.at[request_id].set(REQUEST_ASSIGNED),
            car_request_ids=s.car_request_ids.at[car].set(request_id),
            car_goal_nodes=s.car_goal_nodes.at[car].set(origin),
        )
        return _start_auto_edge(s, params, car, origin, request_id, CAR_TO_PICKUP), r

    return jax.lax.cond(at_origin, pickup_now, drive_to_pickup, (state, reward))


def _finish_pickup(
    state: EnvState,
    params: EnvParams,
    reward,
    car,
) -> tuple[EnvState, jnp.ndarray]:
    request_id = jnp.clip(state.car_request_ids[car], 0, params.max_requests - 1)
    dest = state.request_dest_nodes[request_id]
    state, reward = _record_pickup_wait_reward(state, request_id, params, reward)
    state = state.replace(
        request_status=state.request_status.at[request_id].set(REQUEST_ONBOARD),
        request_pickup_times=state.request_pickup_times.at[request_id].set(state.time_seconds),
        car_goal_nodes=state.car_goal_nodes.at[car].set(dest),
    )
    state = _start_auto_edge(state, params, car, dest, request_id, CAR_TO_DROPOFF)
    return state, reward


def _finish_dropoff(
    state: EnvState,
    params: EnvParams,
    reward,
    car,
) -> tuple[EnvState, jnp.ndarray]:
    request_id = jnp.clip(state.car_request_ids[car], 0, params.max_requests - 1)
    metrics = state.metrics.replace(
        completed_requests=state.metrics.completed_requests + jnp.asarray(1, dtype=jnp.int32),
    )
    state = state.replace(
        request_status=state.request_status.at[request_id].set(REQUEST_COMPLETED),
        car_status=state.car_status.at[car].set(CAR_DECISION),
        car_edge_ids=state.car_edge_ids.at[car].set(-1),
        car_goal_nodes=state.car_goal_nodes.at[car].set(-1),
        car_request_ids=state.car_request_ids.at[car].set(-1),
        metrics=metrics,
    )
    return state, reward


def _pickup_wait_seconds(state: EnvState, request_id):
    request_id = jnp.clip(
        jnp.asarray(request_id, dtype=jnp.int32),
        0,
        state.request_spawn_times.shape[0] - 1,
    )
    return jnp.maximum(0.0, state.time_seconds - state.request_spawn_times[request_id])


def _record_pickup_wait_reward(
    state: EnvState,
    request_id,
    params: EnvParams,
    reward,
) -> tuple[EnvState, jnp.ndarray]:
    wait = _pickup_wait_seconds(state, request_id)
    state = _record_pickup_wait(state, request_id)
    return state, reward - params.wait_time_scale * wait


def _record_pickup_wait(state: EnvState, request_id) -> EnvState:
    request_id = jnp.clip(jnp.asarray(request_id, dtype=jnp.int32), 0, state.request_spawn_times.shape[0] - 1)
    wait = _pickup_wait_seconds(state, request_id)
    slot = jnp.mod(state.metrics.recent_pickup_wait_index, _RECENT_PICKUP_WAIT_WINDOW)
    metrics = state.metrics.replace(
        pickup_wait_seconds=state.metrics.pickup_wait_seconds + wait,
        recent_pickup_wait_seconds=state.metrics.recent_pickup_wait_seconds.at[slot].set(wait),
        recent_pickup_wait_count=jnp.minimum(
            state.metrics.recent_pickup_wait_count + jnp.asarray(1, dtype=jnp.int32),
            jnp.asarray(_RECENT_PICKUP_WAIT_WINDOW, dtype=jnp.int32),
        ),
        recent_pickup_wait_index=jnp.mod(
            state.metrics.recent_pickup_wait_index + jnp.asarray(1, dtype=jnp.int32),
            jnp.asarray(_RECENT_PICKUP_WAIT_WINDOW, dtype=jnp.int32),
        ),
    )
    return state.replace(metrics=metrics)


def _add_transition_reward(state: EnvState, reward) -> EnvState:
    metrics = state.metrics.replace(
        aggregate_reward=state.metrics.aggregate_reward + jnp.asarray(reward, dtype=jnp.float32)
    )
    return state.replace(metrics=metrics)


def _start_auto_edge(
    state: EnvState,
    params: EnvParams,
    car,
    goal,
    request_id,
    status,
) -> EnvState:
    node = state.car_nodes[car]
    at_goal = node == goal
    edge_id = params.graph.routing_next_edge[node, goal]
    has_edge = edge_id >= 0
    safe_edge = jnp.clip(edge_id, 0, params.graph.num_edges - 1)
    target = params.graph.edge_targets[safe_edge]
    duration = _edge_travel_time_at(safe_edge, state.time_seconds, params.graph)
    next_status = jnp.where(has_edge & (~at_goal), status, CAR_DECISION)
    return state.replace(
        car_status=state.car_status.at[car].set(next_status),
        car_edge_ids=state.car_edge_ids.at[car].set(jnp.where(next_status == CAR_DECISION, -1, edge_id)),
        car_target_nodes=state.car_target_nodes.at[car].set(jnp.where(next_status == CAR_DECISION, node, target)),
        car_goal_nodes=state.car_goal_nodes.at[car].set(jnp.where(next_status == CAR_DECISION, -1, goal)),
        car_request_ids=state.car_request_ids.at[car].set(
            jnp.where(next_status == CAR_DECISION, -1, request_id)
        ),
        car_ready_times=state.car_ready_times.at[car].set(
            jnp.where(next_status == CAR_DECISION, state.time_seconds, state.time_seconds + duration)
        ),
        car_departure_times=state.car_departure_times.at[car].set(state.time_seconds),
        car_edge_durations=state.car_edge_durations.at[car].set(
            jnp.where(next_status == CAR_DECISION, 0.0, duration)
        ),
    )


def _refresh_decision(state: EnvState, params: EnvParams) -> EnvState:
    decision_mask = (state.car_status == CAR_DECISION) & (~state.done)
    any_decision = jnp.any(decision_mask)
    car_ids = jnp.arange(params.max_cars, dtype=jnp.int32)
    current = jnp.min(jnp.where(decision_mask, car_ids, params.max_cars)).astype(jnp.int32)
    current = jnp.where(any_decision, current, -1)
    return state.replace(current_car_id=current, decision_required=any_decision)


def _make_timestep(state: EnvState, params: EnvParams, reward, dt_seconds) -> Timestep:
    discount = params.gamma ** (dt_seconds / jnp.maximum(params.discount_time_unit_seconds, 1e-6))
    return Timestep(
        observation=build_observation(state, params),
        reward=jnp.asarray(reward, dtype=jnp.float32),
        discount=jnp.asarray(discount, dtype=jnp.float32),
        done=state.done,
        dt_seconds=jnp.asarray(dt_seconds, dtype=jnp.float32),
        metrics=state.metrics,
    )


def _edge_travel_time_at(edge_id, time_seconds, graph: GraphArrays):
    hour_float = (time_seconds / 3600.0) % 24.0
    h0 = jnp.floor(hour_float).astype(jnp.int32)
    h1 = (h0 + 1) % 24
    frac = hour_float - jnp.floor(hour_float)
    return graph.edge_travel_time_s[edge_id, h0] * (1.0 - frac) + graph.edge_travel_time_s[edge_id, h1] * frac


def _sample_next_spawn_time(rng, time_seconds, spawn_rate_per_minute):
    rate_per_second = spawn_rate_per_minute / 60.0
    u = jnp.maximum(jax.random.uniform(rng, (), dtype=jnp.float32), 1e-6)
    interval = -jnp.log(u) / jnp.maximum(rate_per_second, 1e-6)
    return jnp.where(spawn_rate_per_minute > 0.0, time_seconds + interval, _INF)


def _precompute_node_density_by_hour(graph: GraphArrays) -> np.ndarray:
    density = np.asarray(graph.node_population_density, dtype=np.float32)
    if density.shape != (graph.num_nodes,):
        density = np.ones((graph.num_nodes,), dtype=np.float32)
    density = np.nan_to_num(density, nan=0.0, posinf=0.0, neginf=0.0)
    density = np.maximum(density, 0.0)
    if float(density.sum()) <= 0.0:
        density = np.ones((graph.num_nodes,), dtype=np.float32)

    rows = np.asarray(graph.node_grid_rows, dtype=np.float32)
    cols = np.asarray(graph.node_grid_cols, dtype=np.float32)
    if rows.shape != (graph.num_nodes,) or cols.shape != (graph.num_nodes,):
        rows = np.zeros((graph.num_nodes,), dtype=np.float32)
        cols = np.arange(graph.num_nodes, dtype=np.float32)
    rows = np.where(rows >= 0, rows, 0.0)
    cols = np.where(cols >= 0, cols, 0.0)
    row_count = max(1.0, float(rows.max(initial=0.0) + 1.0))
    col_count = max(1.0, float(cols.max(initial=0.0) + 1.0))
    cx = (col_count - 1.0) / 2.0
    cy = (row_count - 1.0) / 2.0
    nx = (cols - cx) / max(1.0, col_count / 2.0)
    ny = (rows - cy) / max(1.0, row_count / 2.0)
    center_weight = np.clip(1.0 - np.sqrt(nx * nx + ny * ny), 0.0, 1.0).astype(np.float32)

    by_hour = np.zeros((24, graph.num_nodes), dtype=np.float32)
    for hour in range(24):
        midday_boost = _time_wave(float(hour), peak_hour=13.0, spread_hours=3.4)
        evening_residential = _time_wave(float(hour), peak_hour=21.0, spread_hours=2.8)
        spatial_shift = 1.0 + 0.35 * center_weight * midday_boost - 0.18 * center_weight * evening_residential
        factor = np.clip(_POPULATION_HOURLY_MULTIPLIER[hour] * spatial_shift, 0.4, 1.75)
        weights = np.maximum(density * factor.astype(np.float32), 1.0e-6)
        by_hour[hour] = weights
    return by_hour


def _time_wave(hour: float, *, peak_hour: float, spread_hours: float) -> float:
    delta = min(abs(hour - peak_hour), 24.0 - abs(hour - peak_hour))
    return float(np.exp(-(delta * delta) / (2.0 * spread_hours * spread_hours)))


def _precompute_edge_raster_by_hour(graph: GraphArrays, raster_size: int) -> np.ndarray:
    size = int(raster_size)
    raster = np.zeros((24, size, size), dtype=np.float32)
    if graph.num_edges == 0:
        return raster
    node_lonlat = np.asarray(graph.node_lonlat)
    edge_sources = np.asarray(graph.edge_sources)
    edge_targets = np.asarray(graph.edge_targets)
    edge_congestion = np.asarray(graph.edge_congestion)
    bounds = np.asarray(graph.bounds)
    min_lon, min_lat, max_lon, max_lat = bounds
    span_lon = max(float(max_lon - min_lon), 1e-6)
    span_lat = max(float(max_lat - min_lat), 1e-6)
    edge_midpoints = (node_lonlat[edge_sources] + node_lonlat[edge_targets]) / 2.0
    cols = np.floor((edge_midpoints[:, 0] - min_lon) / span_lon * size).astype(np.int32)
    rows = np.floor((edge_midpoints[:, 1] - min_lat) / span_lat * size).astype(np.int32)
    rows = np.clip(rows, 0, size - 1)
    cols = np.clip(cols, 0, size - 1)
    for hour in range(24):
        np.maximum.at(raster[hour], (rows, cols), edge_congestion[:, hour])
    return raster
