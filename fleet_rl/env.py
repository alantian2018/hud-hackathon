from __future__ import annotations

from flax import struct
import jax
import jax.numpy as jnp
from jax import lax

from .types import (
    EnvState,
    EVENT_OVERFLOW,
    EVENT_REQUEST_ASSIGNED,
    EVENT_REQUEST_COMPLETED,
    EVENT_REQUEST_DROPPED,
    EVENT_REQUEST_PICKED_UP,
    EVENT_REQUEST_SPAWNED,
    GraphData,
    Metrics,
    NO_CAR,
    NO_EDGE,
    NO_REQUEST,
    Observation,
    POLICY_CONTROLLED,
    REQ_ASSIGNED,
    REQ_EMPTY,
    REQ_PICKED_UP,
    REQ_QUEUED,
    Timestep,
    TO_DROPOFF,
    TO_PICKUP,
)

INF = 1.0e20
DAY_SECONDS = 86_400.0
DEFAULT_DEMAND_TIME_PROFILE = (
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
)


@struct.dataclass
class EnvParams:
    graph: GraphData
    num_cars: jnp.ndarray
    demand_rate_per_second: jnp.ndarray
    demand_scale: jnp.ndarray
    demand_time_profile: jnp.ndarray
    wait_time_scale: jnp.ndarray
    gamma: jnp.ndarray
    discount_time_unit_seconds: jnp.ndarray
    demand_time_bin_seconds: jnp.ndarray
    max_sim_time_seconds_value: jnp.ndarray
    randomize_start_time_value: jnp.ndarray
    max_cars: int = struct.field(pytree_node=False)
    max_active_requests: int = struct.field(pytree_node=False)
    max_recent_events: int = struct.field(pytree_node=False)
    max_internal_events: int = struct.field(pytree_node=False)
    max_request_time_bins: int = struct.field(pytree_node=False)
    max_degree: int = struct.field(pytree_node=False)
    raster_size: int = struct.field(pytree_node=False)

    @classmethod
    def for_graph(
        cls,
        graph: GraphData,
        *,
        num_cars: int = 8,
        max_cars: int = 64,
        demand_rate_per_second: float = 1.0 / 900.0,
        demand_scale: float = 1.0,
        demand_time_profile: tuple[float, ...] | list[float] = DEFAULT_DEMAND_TIME_PROFILE,
        wait_time_scale: float = 1.0,
        gamma: float = 0.99,
        discount_time_unit_seconds: float = 60.0,
        demand_time_bin_seconds: float = 3600.0,
        max_active_requests: int = 128,
        max_recent_events: int = 64,
        max_internal_events: int = 256,
        max_request_time_bins: int = 168,
        max_sim_time_seconds: float = 24.0 * 3600.0,
        randomize_start_time: bool = True,
    ) -> "EnvParams":
        profile = jnp.asarray(demand_time_profile, dtype=jnp.float32)
        profile = jnp.resize(profile, (24,))
        return cls(
            graph=graph,
            num_cars=jnp.array(num_cars, dtype=jnp.int32),
            demand_rate_per_second=jnp.array(demand_rate_per_second, dtype=jnp.float32),
            demand_scale=jnp.array(demand_scale, dtype=jnp.float32),
            demand_time_profile=profile,
            wait_time_scale=jnp.array(wait_time_scale, dtype=jnp.float32),
            gamma=jnp.array(gamma, dtype=jnp.float32),
            discount_time_unit_seconds=jnp.array(discount_time_unit_seconds, dtype=jnp.float32),
            demand_time_bin_seconds=jnp.array(demand_time_bin_seconds, dtype=jnp.float32),
            max_sim_time_seconds_value=jnp.array(max_sim_time_seconds, dtype=jnp.float32),
            randomize_start_time_value=jnp.array(1 if randomize_start_time else 0, dtype=jnp.int32),
            max_cars=max_cars,
            max_active_requests=max_active_requests,
            max_recent_events=max_recent_events,
            max_internal_events=max_internal_events,
            max_request_time_bins=max_request_time_bins,
            max_degree=graph.max_degree,
            raster_size=graph.raster_size,
        )


def _zeros_metrics(params: EnvParams) -> Metrics:
    return Metrics(
        requests_spawned=jnp.array(0, dtype=jnp.int32),
        requests_queued=jnp.array(0, dtype=jnp.int32),
        requests_assigned=jnp.array(0, dtype=jnp.int32),
        requests_picked_up=jnp.array(0, dtype=jnp.int32),
        requests_completed=jnp.array(0, dtype=jnp.int32),
        dropped_requests=jnp.array(0, dtype=jnp.int32),
        queue_length=jnp.array(0, dtype=jnp.int32),
        total_pickup_wait_time=jnp.array(0.0, dtype=jnp.float32),
        avg_pickup_wait_time=jnp.array(0.0, dtype=jnp.float32),
        p50_pickup_wait_time=jnp.array(0.0, dtype=jnp.float32),
        p90_pickup_wait_time=jnp.array(0.0, dtype=jnp.float32),
        p95_pickup_wait_time=jnp.array(0.0, dtype=jnp.float32),
        pickup_wait_samples=jnp.zeros((params.max_active_requests,), dtype=jnp.float32),
        pickup_wait_sample_times=jnp.zeros((params.max_active_requests,), dtype=jnp.float32),
        pickup_wait_count=jnp.array(0, dtype=jnp.int32),
        fleet_utilization=jnp.array(0.0, dtype=jnp.float32),
        empty_driving_time=jnp.array(0.0, dtype=jnp.float32),
        empty_driving_distance=jnp.array(0.0, dtype=jnp.float32),
        invalid_actions=jnp.array(0, dtype=jnp.int32),
        overflow=jnp.array(False),
    )


class FleetEnv:
    def __init__(self, graph: GraphData):
        self.graph = graph

    def reset(self, rng: jnp.ndarray, params: EnvParams) -> tuple[EnvState, Timestep]:
        graph = params.graph
        key_time, key_cars, key_request = jax.random.split(rng, 3)
        random_start = jax.random.uniform(key_time, (), minval=0.0, maxval=DAY_SECONDS)
        sim_time = jnp.where(params.randomize_start_time_value > 0, random_start, 0.0).astype(jnp.float32)

        car_ids = jnp.arange(params.max_cars, dtype=jnp.int32)
        car_active = car_ids < params.num_cars
        initial_nodes = jax.random.randint(key_cars, (params.max_cars,), minval=0, maxval=graph.num_nodes, dtype=jnp.int32)
        initial_nodes = jnp.where(car_active, initial_nodes, 0)

        request_status = jnp.zeros((params.max_active_requests,), dtype=jnp.int32)
        state = EnvState(
            rng=key_request,
            sim_time_seconds=sim_time,
            next_request_time_seconds=jnp.array(INF, dtype=jnp.float32),
            request_id_counter=jnp.array(0, dtype=jnp.int32),
            current_car_id=jnp.array(NO_CAR, dtype=jnp.int32),
            current_node_id=jnp.array(-1, dtype=jnp.int32),
            decision_pending=car_active,
            car_active=car_active,
            car_status=jnp.full((params.max_cars,), POLICY_CONTROLLED, dtype=jnp.int32),
            car_node=initial_nodes,
            car_from_node=initial_nodes,
            car_to_node=initial_nodes,
            car_edge_id=jnp.full((params.max_cars,), NO_EDGE, dtype=jnp.int32),
            car_edge_start_time=jnp.full((params.max_cars,), sim_time, dtype=jnp.float32),
            car_edge_end_time=jnp.full((params.max_cars,), sim_time, dtype=jnp.float32),
            car_assigned_request=jnp.full((params.max_cars,), NO_REQUEST, dtype=jnp.int32),
            car_target_node=initial_nodes,
            request_ids=jnp.full((params.max_active_requests,), -1, dtype=jnp.int32),
            request_status=request_status,
            request_pickup_node=jnp.zeros((params.max_active_requests,), dtype=jnp.int32),
            request_dropoff_node=jnp.zeros((params.max_active_requests,), dtype=jnp.int32),
            request_spawn_time=jnp.zeros((params.max_active_requests,), dtype=jnp.float32),
            request_assigned_car_id=jnp.full((params.max_active_requests,), NO_CAR, dtype=jnp.int32),
            request_pickup_time=jnp.full((params.max_active_requests,), -1.0, dtype=jnp.float32),
            request_dropoff_time=jnp.full((params.max_active_requests,), -1.0, dtype=jnp.float32),
            edge_congestion=self._edge_congestion_vector(sim_time, params),
            recent_event_codes=jnp.zeros((params.max_recent_events,), dtype=jnp.int32),
            recent_event_car_ids=jnp.full((params.max_recent_events,), NO_CAR, dtype=jnp.int32),
            recent_event_request_ids=jnp.full((params.max_recent_events,), -1, dtype=jnp.int32),
            recent_event_times=jnp.zeros((params.max_recent_events,), dtype=jnp.float32),
            recent_event_cursor=jnp.array(0, dtype=jnp.int32),
            metrics=_zeros_metrics(params),
        )
        next_request_time, next_rng = self._sample_next_request_time(state, params)
        state = state.replace(next_request_time_seconds=next_request_time, rng=next_rng)
        state = self._select_next_decision(state, params)
        timestep = self._make_timestep(
            state,
            params,
            reward=jnp.array(0.0, dtype=jnp.float32),
            dt=jnp.array(0.0, dtype=jnp.float32),
        )
        return state, timestep

    def step(self, state: EnvState, action: jnp.ndarray, params: EnvParams) -> tuple[EnvState, Timestep]:
        prev_time = state.sim_time_seconds
        state, invalid = self._apply_policy_action(state, action.astype(jnp.int32), params)
        state = state.replace(
            metrics=state.metrics.replace(
                invalid_actions=state.metrics.invalid_actions + invalid.astype(jnp.int32)
            )
        )

        has_pending = jnp.any(state.decision_pending & state.car_active)

        def same_time(s: EnvState):
            return self._select_next_decision(s, params), jnp.array(0.0, dtype=jnp.float32)

        def advance(s: EnvState):
            return self._advance_until_decision(s, params)

        state, reward = lax.cond(has_pending, same_time, advance, state)
        dt = jnp.maximum(0.0, state.sim_time_seconds - prev_time)
        timestep = self._make_timestep(state, params, reward=reward, dt=dt)
        return state, timestep

    def debug_insert_request(
        self,
        state: EnvState,
        *,
        pickup_node: int,
        dropoff_node: int,
        spawn_time: float,
        assign: bool,
        params: EnvParams,
    ) -> EnvState:
        slot = jnp.argmax(state.request_status == REQ_EMPTY).astype(jnp.int32)
        state = self._write_request_slot(
            state,
            slot,
            jnp.array(pickup_node, dtype=jnp.int32),
            jnp.array(dropoff_node, dtype=jnp.int32),
            jnp.array(spawn_time, dtype=jnp.float32),
            params,
        )
        return lax.cond(
            jnp.array(assign),
            lambda s: self._assign_request_or_queue(s, slot, params),
            lambda s: s,
            state,
        )

    def debug_assign_queued_to_available_cars(self, state: EnvState, params: EnvParams) -> EnvState:
        def body(car_id, s):
            eligible = (
                s.car_active[car_id]
                & (s.car_status[car_id] == POLICY_CONTROLLED)
                & (s.car_edge_id[car_id] == NO_EDGE)
                & (s.car_assigned_request[car_id] == NO_REQUEST)
            )
            return lax.cond(eligible, lambda x: self._assign_queued_to_car_or_mark_pending(x, car_id, params), lambda x: x, s)

        state = lax.fori_loop(0, params.max_cars, body, state)
        return self._refresh_metrics(state, params)

    def debug_advance_to_next_decision(self, state: EnvState, params: EnvParams) -> tuple[EnvState, Timestep]:
        prev_time = state.sim_time_seconds
        state = state.replace(current_car_id=jnp.array(NO_CAR, dtype=jnp.int32), current_node_id=jnp.array(-1, dtype=jnp.int32))
        state, reward = self._advance_until_decision(state, params)
        dt = jnp.maximum(0.0, state.sim_time_seconds - prev_time)
        return state, self._make_timestep(state, params, reward=reward, dt=dt)

    def _apply_policy_action(self, state: EnvState, action: jnp.ndarray, params: EnvParams) -> tuple[EnvState, jnp.ndarray]:
        graph = params.graph
        car = state.current_car_id
        valid_car = (car >= 0) & (car < params.max_cars)
        safe_car = jnp.clip(car, 0, params.max_cars - 1)
        node = state.car_node[safe_car]
        action_mask = graph.out_edge_mask[node]
        fallback = jnp.argmax(action_mask).astype(jnp.int32)
        in_range = (action >= 0) & (action < params.max_degree)
        safe_action = jnp.where(in_range, action, fallback)
        selected_valid = valid_car & action_mask[safe_action]
        selected_action = jnp.where(selected_valid, safe_action, fallback)
        invalid = valid_car & (~selected_valid)
        edge = graph.out_edges[node, selected_action]

        can_apply = valid_car & (edge != NO_EDGE) & (state.car_status[safe_car] == POLICY_CONTROLLED)

        def enter(s: EnvState) -> EnvState:
            s = self._enter_edge(s, safe_car, edge, params)
            return s.replace(current_car_id=jnp.array(NO_CAR, dtype=jnp.int32), current_node_id=jnp.array(-1, dtype=jnp.int32))

        state = lax.cond(can_apply, enter, lambda s: s, state)
        return state, invalid

    def _enter_edge(self, state: EnvState, car: jnp.ndarray, edge: jnp.ndarray, params: EnvParams) -> EnvState:
        graph = params.graph
        edge = edge.astype(jnp.int32)
        from_node = graph.edge_from[edge]
        to_node = graph.edge_to[edge]
        congestion = jnp.maximum(state.edge_congestion[edge], self._edge_congestion_at(edge, state.sim_time_seconds, params))
        travel_time = graph.edge_base_travel_time_s[edge] * congestion
        end_time = state.sim_time_seconds + travel_time
        empty = state.car_status[car] == POLICY_CONTROLLED
        metrics = state.metrics.replace(
            empty_driving_time=state.metrics.empty_driving_time + jnp.where(empty, travel_time, 0.0),
            empty_driving_distance=state.metrics.empty_driving_distance + jnp.where(empty, graph.edge_length_m[edge], 0.0),
        )
        return state.replace(
            car_from_node=state.car_from_node.at[car].set(from_node),
            car_to_node=state.car_to_node.at[car].set(to_node),
            car_edge_id=state.car_edge_id.at[car].set(edge),
            car_edge_start_time=state.car_edge_start_time.at[car].set(state.sim_time_seconds),
            car_edge_end_time=state.car_edge_end_time.at[car].set(end_time),
            edge_congestion=state.edge_congestion.at[edge].set(congestion),
            metrics=metrics,
        )

    def _profile_value(self, profile: jnp.ndarray, sim_time: jnp.ndarray) -> jnp.ndarray:
        hour = (sim_time % DAY_SECONDS) / 3600.0
        h0 = jnp.floor(hour).astype(jnp.int32) % 24
        h1 = (h0 + 1) % 24
        frac = hour - jnp.floor(hour)
        return profile[h0] * (1.0 - frac) + profile[h1] * frac

    def _edge_profile_vector(self, sim_time: jnp.ndarray, params: EnvParams) -> jnp.ndarray:
        graph = params.graph
        hour = (sim_time % DAY_SECONDS) / 3600.0
        h0 = jnp.floor(hour).astype(jnp.int32) % 24
        h1 = (h0 + 1) % 24
        frac = hour - jnp.floor(hour)
        return graph.edge_traffic_profile[:, h0] * (1.0 - frac) + graph.edge_traffic_profile[:, h1] * frac

    def _edge_congestion_at(self, edge: jnp.ndarray, sim_time: jnp.ndarray, params: EnvParams) -> jnp.ndarray:
        graph = params.graph
        profile = self._edge_profile_vector(sim_time, params)
        safe_edge = jnp.clip(edge, 0, graph.max_edges - 1)
        return graph.edge_congestion_base[safe_edge] * profile[safe_edge]

    def _edge_congestion_vector(self, sim_time: jnp.ndarray, params: EnvParams) -> jnp.ndarray:
        return params.graph.edge_congestion_base * self._edge_profile_vector(sim_time, params)

    def _log_event(
        self,
        state: EnvState,
        params: EnvParams,
        code: jnp.ndarray,
        car_id: jnp.ndarray,
        request_slot: jnp.ndarray,
    ) -> EnvState:
        idx = state.recent_event_cursor % params.max_recent_events
        code = jnp.asarray(code, dtype=jnp.int32)
        car_id = jnp.asarray(car_id, dtype=jnp.int32)
        request_slot = jnp.asarray(request_slot, dtype=jnp.int32)
        valid_slot = (request_slot >= 0) & (request_slot < params.max_active_requests)
        safe_slot = jnp.clip(request_slot, 0, params.max_active_requests - 1)
        request_id = jnp.where(valid_slot, state.request_ids[safe_slot], -1)
        return state.replace(
            recent_event_codes=state.recent_event_codes.at[idx].set(code),
            recent_event_car_ids=state.recent_event_car_ids.at[idx].set(car_id),
            recent_event_request_ids=state.recent_event_request_ids.at[idx].set(request_id.astype(jnp.int32)),
            recent_event_times=state.recent_event_times.at[idx].set(state.sim_time_seconds),
            recent_event_cursor=state.recent_event_cursor + 1,
        )

    def _maybe_start_auto_edge(self, state: EnvState, car: jnp.ndarray, params: EnvParams) -> EnvState:
        node = state.car_node[car]
        target = state.car_target_node[car]
        edge = self._next_route_edge(state, node, target, params)
        should_enter = (node != target) & (edge != NO_EDGE) & (state.car_edge_id[car] == NO_EDGE)
        return lax.cond(should_enter, lambda s: self._enter_edge(s, car, edge, params), lambda s: s, state)

    def _advance_until_decision(self, state: EnvState, params: EnvParams) -> tuple[EnvState, jnp.ndarray]:
        initial_time = state.sim_time_seconds
        carry = (state, jnp.array(0.0, dtype=jnp.float32), jnp.array(0, dtype=jnp.int32))

        def cond(carry):
            s, _reward, count = carry
            has_decision = jnp.any(s.decision_pending & s.car_active)
            return (
                (~has_decision)
                & (~s.metrics.overflow)
                & (count < params.max_internal_events)
                & (s.sim_time_seconds < params.max_sim_time_seconds_value)
            )

        def body(carry):
            s, reward, count = carry
            s, delta_reward = self._process_next_internal_event(s, params)
            return s, reward + delta_reward, count + 1

        state, reward, count = lax.while_loop(cond, body, carry)
        overflow = (count >= params.max_internal_events) & (~jnp.any(state.decision_pending & state.car_active))
        state = state.replace(metrics=state.metrics.replace(overflow=state.metrics.overflow | overflow))
        state = lax.cond(
            overflow,
            lambda s: self._log_event(s, params, EVENT_OVERFLOW, jnp.array(NO_CAR, dtype=jnp.int32), jnp.array(NO_REQUEST, dtype=jnp.int32)),
            lambda s: s,
            state,
        )
        state = self._select_next_decision(state, params)
        state = lax.cond(
            jnp.isfinite(state.sim_time_seconds),
            lambda s: s,
            lambda s: s.replace(sim_time_seconds=initial_time, metrics=s.metrics.replace(overflow=jnp.array(True))),
            state,
        )
        return state, reward

    def _process_next_internal_event(self, state: EnvState, params: EnvParams) -> tuple[EnvState, jnp.ndarray]:
        moving = state.car_active & (state.car_edge_id != NO_EDGE)
        next_arrival = jnp.min(jnp.where(moving, state.car_edge_end_time, INF))
        next_spawn = state.next_request_time_seconds
        next_time = jnp.minimum(next_arrival, next_spawn)
        no_event = next_time >= INF / 2

        def overflow(s: EnvState) -> tuple[EnvState, jnp.ndarray]:
            s = s.replace(metrics=s.metrics.replace(overflow=jnp.array(True)))
            s = self._log_event(s, params, EVENT_OVERFLOW, jnp.array(NO_CAR, dtype=jnp.int32), jnp.array(NO_REQUEST, dtype=jnp.int32))
            return s, jnp.array(0.0, dtype=jnp.float32)

        def process(s: EnvState) -> tuple[EnvState, jnp.ndarray]:
            s = s.replace(sim_time_seconds=next_time)
            s = s.replace(edge_congestion=self._edge_congestion_vector(next_time, params))
            spawn_first = next_spawn <= next_arrival
            return lax.cond(
                spawn_first,
                lambda x: (self._process_request_spawn(x, params), jnp.array(0.0, dtype=jnp.float32)),
                lambda x: self._process_arrivals(x, params),
                s,
            )

        return lax.cond(no_event, overflow, process, state)

    def _process_request_spawn(self, state: EnvState, params: EnvParams) -> EnvState:
        key_pickup, key_dropoff, key_next = jax.random.split(state.rng, 3)
        pickup, dropoff = self._sample_request_nodes(key_pickup, key_dropoff, params)
        empty_mask = state.request_status == REQ_EMPTY
        has_slot = jnp.any(empty_mask)
        slot = jnp.argmax(empty_mask).astype(jnp.int32)

        def add_request(s: EnvState) -> EnvState:
            s = s.replace(rng=key_next, metrics=s.metrics.replace(requests_spawned=s.metrics.requests_spawned + 1))
            s = self._write_request_slot(s, slot, pickup, dropoff, s.sim_time_seconds, params)
            s = self._log_event(s, params, EVENT_REQUEST_SPAWNED, jnp.array(NO_CAR, dtype=jnp.int32), slot)
            return self._assign_request_or_queue(s, slot, params)

        def drop_request(s: EnvState) -> EnvState:
            s = s.replace(
                rng=key_next,
                metrics=s.metrics.replace(
                    requests_spawned=s.metrics.requests_spawned + 1,
                    dropped_requests=s.metrics.dropped_requests + 1,
                ),
            )
            return self._log_event(s, params, EVENT_REQUEST_DROPPED, jnp.array(NO_CAR, dtype=jnp.int32), jnp.array(NO_REQUEST, dtype=jnp.int32))

        state = lax.cond(has_slot, add_request, drop_request, state)
        next_request_time, next_rng = self._sample_next_request_time(state, params)
        state = state.replace(next_request_time_seconds=next_request_time, rng=next_rng)
        return self._refresh_metrics(state, params)

    def _sample_request_nodes(
        self,
        key_pickup: jnp.ndarray,
        key_dropoff: jnp.ndarray,
        params: EnvParams,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        graph = params.graph
        pickup = jax.random.categorical(key_pickup, jnp.log(jnp.maximum(graph.pickup_prob, 1e-12))).astype(jnp.int32)
        dx = graph.node_x - graph.node_x[pickup]
        dy = graph.node_y - graph.node_y[pickup]
        distance = jnp.sqrt(dx * dx + dy * dy) / jnp.sqrt(jnp.array(2.0, dtype=jnp.float32))
        demand_bias = graph.node_demand_weight
        weights = graph.dropoff_prob * (0.2 + 0.75 * demand_bias + 0.9 * distance)
        valid = jnp.arange(graph.max_nodes, dtype=jnp.int32) < graph.num_nodes
        weights = jnp.where(valid & (jnp.arange(graph.max_nodes, dtype=jnp.int32) != pickup), weights, 0.0)
        weights = weights / jnp.maximum(1e-9, jnp.sum(weights))
        raw_dropoff = jax.random.categorical(key_dropoff, jnp.log(jnp.maximum(weights, 1e-12))).astype(jnp.int32)
        dropoff = jnp.where(raw_dropoff == pickup, (raw_dropoff + 1) % graph.num_nodes, raw_dropoff).astype(jnp.int32)
        return pickup, dropoff

    def _write_request_slot(
        self,
        state: EnvState,
        slot: jnp.ndarray,
        pickup: jnp.ndarray,
        dropoff: jnp.ndarray,
        spawn_time: jnp.ndarray,
        params: EnvParams,
    ) -> EnvState:
        request_id = state.request_id_counter
        return state.replace(
            request_id_counter=state.request_id_counter + 1,
            request_ids=state.request_ids.at[slot].set(request_id),
            request_status=state.request_status.at[slot].set(REQ_QUEUED),
            request_pickup_node=state.request_pickup_node.at[slot].set(pickup),
            request_dropoff_node=state.request_dropoff_node.at[slot].set(dropoff),
            request_spawn_time=state.request_spawn_time.at[slot].set(spawn_time),
            request_assigned_car_id=state.request_assigned_car_id.at[slot].set(NO_CAR),
            request_pickup_time=state.request_pickup_time.at[slot].set(-1.0),
            request_dropoff_time=state.request_dropoff_time.at[slot].set(-1.0),
        )

    def _assign_request_or_queue(self, state: EnvState, slot: jnp.ndarray, params: EnvParams) -> EnvState:
        graph = params.graph
        pickup = state.request_pickup_node[slot]
        eligible = (
            state.car_active
            & (state.car_status == POLICY_CONTROLLED)
            & (state.car_assigned_request == NO_REQUEST)
        )
        moving = state.car_edge_id != NO_EDGE
        start_node = jnp.where(moving, state.car_to_node, state.car_node)
        remaining = jnp.maximum(0.0, state.car_edge_end_time - state.sim_time_seconds)
        eta = jnp.where(moving, remaining, 0.0) + self._travel_time_estimate(start_node, pickup, params)
        eta = jnp.where(eligible & jnp.isfinite(eta), eta, INF)
        has_car = jnp.min(eta) < INF / 2
        car = jnp.argmin(eta).astype(jnp.int32)

        def assign(s: EnvState) -> EnvState:
            s = s.replace(
                request_status=s.request_status.at[slot].set(REQ_ASSIGNED),
                request_assigned_car_id=s.request_assigned_car_id.at[slot].set(car),
                car_status=s.car_status.at[car].set(TO_PICKUP),
                car_assigned_request=s.car_assigned_request.at[car].set(slot),
                car_target_node=s.car_target_node.at[car].set(pickup),
                decision_pending=s.decision_pending.at[car].set(False),
                metrics=s.metrics.replace(requests_assigned=s.metrics.requests_assigned + 1),
            )
            s = self._log_event(s, params, EVENT_REQUEST_ASSIGNED, car, slot)
            return self._maybe_start_auto_edge(s, car, params)

        def queue(s: EnvState) -> EnvState:
            already_queued = s.request_status[slot] == REQ_QUEUED
            return s.replace(metrics=s.metrics.replace(requests_queued=s.metrics.requests_queued + already_queued.astype(jnp.int32)))

        state = lax.cond(has_car, assign, queue, state)
        return self._refresh_metrics(state, params)

    def _process_arrivals(self, state: EnvState, params: EnvParams) -> tuple[EnvState, jnp.ndarray]:
        def body(car_id, carry):
            s, reward = carry
            s, delta = self._process_car_arrival(s, car_id, params)
            return s, reward + delta

        state, reward = lax.fori_loop(
            0,
            params.max_cars,
            body,
            (state, jnp.array(0.0, dtype=jnp.float32)),
        )
        return self._refresh_metrics(state, params), reward

    def _process_car_arrival(self, state: EnvState, car: jnp.ndarray, params: EnvParams) -> tuple[EnvState, jnp.ndarray]:
        arrived = (
            state.car_active[car]
            & (state.car_edge_id[car] != NO_EDGE)
            & (state.car_edge_end_time[car] <= state.sim_time_seconds + 1e-5)
        )

        def process(s: EnvState) -> tuple[EnvState, jnp.ndarray]:
            status = s.car_status[car]
            node = s.car_to_node[car]
            s = s.replace(
                car_node=s.car_node.at[car].set(node),
                car_from_node=s.car_from_node.at[car].set(node),
                car_edge_id=s.car_edge_id.at[car].set(NO_EDGE),
                car_edge_start_time=s.car_edge_start_time.at[car].set(s.sim_time_seconds),
                car_edge_end_time=s.car_edge_end_time.at[car].set(s.sim_time_seconds),
            )

            def controlled(x):
                return self._assign_queued_to_car_or_mark_pending(x, car, params), jnp.array(0.0, dtype=jnp.float32)

            def to_pickup(x):
                at_target = x.car_node[car] == x.car_target_node[car]
                return lax.cond(
                    at_target,
                    lambda y: self._pickup_request(y, car, params),
                    lambda y: (self._maybe_start_auto_edge(y, car, params), jnp.array(0.0, dtype=jnp.float32)),
                    x,
                )

            def to_dropoff(x):
                at_target = x.car_node[car] == x.car_target_node[car]
                return lax.cond(
                    at_target,
                    lambda y: self._complete_request(y, car, params),
                    lambda y: (self._maybe_start_auto_edge(y, car, params), jnp.array(0.0, dtype=jnp.float32)),
                    x,
                )

            s, reward = lax.switch(status, (controlled, to_pickup, to_dropoff), s)
            return s, reward

        return lax.cond(arrived, process, lambda s: (s, jnp.array(0.0, dtype=jnp.float32)), state)

    def _pickup_request(self, state: EnvState, car: jnp.ndarray, params: EnvParams) -> tuple[EnvState, jnp.ndarray]:
        slot = state.car_assigned_request[car]
        wait_time = jnp.maximum(0.0, state.sim_time_seconds - state.request_spawn_time[slot])
        metrics = self._record_pickup_wait(state.metrics, wait_time, state.sim_time_seconds, params)
        state = state.replace(
            request_status=state.request_status.at[slot].set(REQ_PICKED_UP),
            request_pickup_time=state.request_pickup_time.at[slot].set(state.sim_time_seconds),
            car_status=state.car_status.at[car].set(TO_DROPOFF),
            car_target_node=state.car_target_node.at[car].set(state.request_dropoff_node[slot]),
            metrics=metrics,
        )
        state = self._log_event(state, params, EVENT_REQUEST_PICKED_UP, car, slot)
        state = self._maybe_start_auto_edge(state, car, params)
        reward = -params.wait_time_scale * wait_time
        return state, reward.astype(jnp.float32)

    def _record_pickup_wait(
        self,
        metrics: Metrics,
        wait_time: jnp.ndarray,
        sim_time: jnp.ndarray,
        params: EnvParams,
    ) -> Metrics:
        sample_idx = metrics.pickup_wait_count % params.max_active_requests
        new_count = metrics.pickup_wait_count + 1
        new_total = metrics.total_pickup_wait_time + wait_time
        samples = metrics.pickup_wait_samples.at[sample_idx].set(wait_time)
        sample_times = metrics.pickup_wait_sample_times.at[sample_idx].set(sim_time)
        valid_count = jnp.minimum(new_count, params.max_active_requests)
        sample_indices = jnp.arange(params.max_active_requests, dtype=jnp.int32)
        valid = sample_indices < valid_count
        sorted_samples = jnp.sort(jnp.where(valid, samples, INF))

        def percentile(q: float) -> jnp.ndarray:
            idx = jnp.floor((valid_count.astype(jnp.float32) - 1.0) * q).astype(jnp.int32)
            idx = jnp.clip(idx, 0, params.max_active_requests - 1)
            return jnp.where(valid_count > 0, sorted_samples[idx], 0.0)

        return metrics.replace(
            requests_picked_up=metrics.requests_picked_up + 1,
            total_pickup_wait_time=new_total,
            avg_pickup_wait_time=new_total / jnp.maximum(1.0, new_count.astype(jnp.float32)),
            p50_pickup_wait_time=percentile(0.50),
            p90_pickup_wait_time=percentile(0.90),
            p95_pickup_wait_time=percentile(0.95),
            pickup_wait_samples=samples,
            pickup_wait_sample_times=sample_times,
            pickup_wait_count=new_count,
        )

    def _complete_request(self, state: EnvState, car: jnp.ndarray, params: EnvParams) -> tuple[EnvState, jnp.ndarray]:
        slot = state.car_assigned_request[car]
        metrics = state.metrics.replace(requests_completed=state.metrics.requests_completed + 1)
        state = state.replace(
            request_status=state.request_status.at[slot].set(REQ_EMPTY),
            request_dropoff_time=state.request_dropoff_time.at[slot].set(state.sim_time_seconds),
            request_assigned_car_id=state.request_assigned_car_id.at[slot].set(NO_CAR),
            car_status=state.car_status.at[car].set(POLICY_CONTROLLED),
            car_assigned_request=state.car_assigned_request.at[car].set(NO_REQUEST),
            car_target_node=state.car_target_node.at[car].set(state.car_node[car]),
            metrics=metrics,
        )
        state = self._log_event(state, params, EVENT_REQUEST_COMPLETED, car, slot)
        state = self._assign_queued_to_car_or_mark_pending(state, car, params)
        return state, jnp.array(0.0, dtype=jnp.float32)

    def _assign_queued_to_car_or_mark_pending(self, state: EnvState, car: jnp.ndarray, params: EnvParams) -> EnvState:
        queued = state.request_status == REQ_QUEUED
        has_queue = jnp.any(queued)
        spawn_rank = jnp.where(queued, state.request_spawn_time, INF)
        slot = jnp.argmin(spawn_rank).astype(jnp.int32)

        def assign(s: EnvState) -> EnvState:
            pickup = s.request_pickup_node[slot]
            s = s.replace(
                request_status=s.request_status.at[slot].set(REQ_ASSIGNED),
                request_assigned_car_id=s.request_assigned_car_id.at[slot].set(car),
                car_status=s.car_status.at[car].set(TO_PICKUP),
                car_assigned_request=s.car_assigned_request.at[car].set(slot),
                car_target_node=s.car_target_node.at[car].set(pickup),
                decision_pending=s.decision_pending.at[car].set(False),
                metrics=s.metrics.replace(requests_assigned=s.metrics.requests_assigned + 1),
            )
            s = self._log_event(s, params, EVENT_REQUEST_ASSIGNED, car, slot)
            return self._maybe_start_auto_edge(s, car, params)

        def pending(s: EnvState) -> EnvState:
            return s.replace(decision_pending=s.decision_pending.at[car].set(True))

        state = lax.cond(has_queue, assign, pending, state)
        return self._refresh_metrics(state, params)

    def _refresh_metrics(self, state: EnvState, params: EnvParams) -> EnvState:
        serving = state.car_active & ((state.car_status == TO_PICKUP) | (state.car_status == TO_DROPOFF))
        utilization = jnp.sum(serving).astype(jnp.float32) / jnp.maximum(1.0, params.num_cars.astype(jnp.float32))
        queue_length = jnp.sum(state.request_status == REQ_QUEUED).astype(jnp.int32)
        return state.replace(metrics=state.metrics.replace(queue_length=queue_length, fleet_utilization=utilization))

    def _select_next_decision(self, state: EnvState, params: EnvParams) -> EnvState:
        pending = state.decision_pending & state.car_active
        car_ids = jnp.arange(params.max_cars, dtype=jnp.int32)
        car = jnp.min(jnp.where(pending, car_ids, params.max_cars)).astype(jnp.int32)
        has_car = car < params.max_cars
        safe_car = jnp.clip(car, 0, params.max_cars - 1)
        node = jnp.where(has_car, state.car_node[safe_car], -1).astype(jnp.int32)
        new_pending_value = jnp.where(has_car, False, state.decision_pending[safe_car])
        return state.replace(
            current_car_id=jnp.where(has_car, car, NO_CAR).astype(jnp.int32),
            current_node_id=node,
            decision_pending=state.decision_pending.at[safe_car].set(new_pending_value),
        )

    def _travel_time_estimate(self, source_node: jnp.ndarray, target_node: jnp.ndarray, params: EnvParams) -> jnp.ndarray:
        graph = params.graph
        if graph.route_mode == "dense":
            return graph.travel_time_table[source_node, target_node]

        via_landmarks = graph.node_to_landmark_time[source_node, :] + jnp.moveaxis(
            graph.landmark_to_node_time[:, target_node],
            0,
            -1,
        )
        best = jnp.min(via_landmarks, axis=-1)
        fallback = self._straight_line_time_estimate(source_node, target_node, params)
        same_node = source_node == target_node
        return jnp.where(same_node, 0.0, jnp.where(jnp.isfinite(best), best, fallback))

    def _next_route_edge(self, state: EnvState, source_node: jnp.ndarray, target_node: jnp.ndarray, params: EnvParams) -> jnp.ndarray:
        graph = params.graph
        if graph.route_mode == "dense":
            return graph.next_edge_table[source_node, target_node]

        via_landmarks = graph.node_to_landmark_time[source_node, :] + graph.landmark_to_node_time[:, target_node]
        landmark_idx = jnp.argmin(via_landmarks).astype(jnp.int32)
        landmark_node = graph.landmark_nodes[landmark_idx]
        edge_to_landmark = graph.node_to_landmark_next_edge[source_node, landmark_idx]
        edge_from_landmark = graph.landmark_to_node_next_edge[landmark_idx, target_node]
        edge = jnp.where(source_node != landmark_node, edge_to_landmark, edge_from_landmark)
        greedy_edge = self._greedy_next_edge(state, source_node, target_node, params)
        safe_edge = jnp.clip(edge, 0, graph.max_edges - 1)
        edge_starts_at_source = (edge != NO_EDGE) & (graph.edge_from[safe_edge] == source_node)
        edge = jnp.where(edge_starts_at_source, edge, greedy_edge)
        return jnp.where(source_node == target_node, NO_EDGE, edge).astype(jnp.int32)

    def _greedy_next_edge(self, state: EnvState, source_node: jnp.ndarray, target_node: jnp.ndarray, params: EnvParams) -> jnp.ndarray:
        graph = params.graph
        edge_ids = graph.out_edges[source_node]
        valid = graph.out_edge_mask[source_node]
        safe_edges = jnp.clip(edge_ids, 0, graph.max_edges - 1)
        next_nodes = graph.edge_to[safe_edges]
        edge_cost = graph.edge_base_travel_time_s[safe_edges] * state.edge_congestion[safe_edges]
        estimate = self._travel_time_estimate(next_nodes, target_node, params)
        score = jnp.where(valid, edge_cost + estimate, INF)
        slot = jnp.argmin(score).astype(jnp.int32)
        return jnp.where(jnp.any(valid), edge_ids[slot], NO_EDGE).astype(jnp.int32)

    def _straight_line_time_estimate(self, source_node: jnp.ndarray, target_node: jnp.ndarray, params: EnvParams) -> jnp.ndarray:
        graph = params.graph
        lat0 = graph.node_lat[source_node] * jnp.pi / 180.0
        lat1 = graph.node_lat[target_node] * jnp.pi / 180.0
        lon0 = graph.node_lon[source_node] * jnp.pi / 180.0
        lon1 = graph.node_lon[target_node] * jnp.pi / 180.0
        mean_lat = (lat0 + lat1) * 0.5
        dx = (lon1 - lon0) * jnp.cos(mean_lat) * 6_371_000.0
        dy = (lat1 - lat0) * 6_371_000.0
        distance_m = jnp.sqrt(dx * dx + dy * dy)
        return distance_m / 7.0 + 30.0

    def _sample_next_request_time(self, state: EnvState, params: EnvParams) -> tuple[jnp.ndarray, jnp.ndarray]:
        key, next_key = jax.random.split(state.rng)
        disabled = (params.demand_rate_per_second * params.demand_scale <= 0.0) | (jnp.max(params.demand_time_profile) <= 0.0)

        def no_spawn(_):
            return jnp.array(INF, dtype=jnp.float32), next_key

        def sample(_):
            return self._sample_enabled_next_request_time(state, params, key), next_key

        return lax.cond(disabled, no_spawn, sample, operand=None)

    def _sample_enabled_next_request_time(
        self,
        state: EnvState,
        params: EnvParams,
        key: jnp.ndarray,
    ) -> jnp.ndarray:
        u = jax.random.uniform(key, (), minval=1e-6, maxval=1.0)
        hazard = -jnp.log(u)
        start = state.sim_time_seconds
        bin_seconds = jnp.maximum(1.0, params.demand_time_bin_seconds)
        carry = (
            start,
            hazard,
            jnp.array(0, dtype=jnp.int32),
            jnp.array(False),
        )

        def cond(carry):
            _time, remaining, count, found = carry
            return (~found) & (remaining > 0.0) & (count < params.max_request_time_bins)

        def body(carry):
            time, remaining, count, _found = carry
            next_boundary = (jnp.floor(time / bin_seconds) + 1.0) * bin_seconds
            segment = jnp.maximum(1e-6, next_boundary - time)
            mid_time = time + 0.5 * segment
            rate = self._request_rate_at(mid_time, params)
            segment_hazard = rate * segment
            found = (rate > 0.0) & (remaining <= segment_hazard)
            spawn_time = time + remaining / jnp.maximum(rate, 1e-9)
            next_time = jnp.where(found, spawn_time, next_boundary)
            next_remaining = jnp.where(found, 0.0, remaining - segment_hazard)
            return next_time, next_remaining, count + 1, found

        next_time, _remaining, _count, found = lax.while_loop(cond, body, carry)
        next_time = jnp.where(found, next_time, jnp.array(INF, dtype=jnp.float32))
        return next_time.astype(jnp.float32)

    def _request_rate_at(self, sim_time: jnp.ndarray, params: EnvParams) -> jnp.ndarray:
        demand_factor = jnp.maximum(0.0, self._profile_value(params.demand_time_profile, sim_time))
        demand_pressure = 0.25 + 0.50 * params.graph.demand_mean + 0.95 * params.graph.demand_max
        mean_traffic = self._profile_value(params.graph.traffic_mean_profile, sim_time)
        max_traffic = self._profile_value(params.graph.traffic_max_profile, sim_time)
        traffic_multiplier = 1.0 + 0.75 * mean_traffic + 0.55 * max_traffic
        return params.demand_rate_per_second * params.demand_scale * demand_factor * demand_pressure * traffic_multiplier

    def _make_timestep(self, state: EnvState, params: EnvParams, *, reward: jnp.ndarray, dt: jnp.ndarray) -> Timestep:
        discount = params.gamma ** (dt / jnp.maximum(1e-6, params.discount_time_unit_seconds))
        truncated = state.metrics.overflow | (state.sim_time_seconds >= params.max_sim_time_seconds_value)
        done = truncated
        obs = self._make_observation(state, params)
        safe_node = jnp.clip(state.current_node_id, 0, params.graph.max_nodes - 1)
        action_mask = jnp.where(state.current_car_id >= 0, params.graph.out_edge_mask[safe_node], jnp.zeros((params.max_degree,), dtype=bool))
        return Timestep(
            observation=obs,
            reward=reward.astype(jnp.float32),
            discount=jnp.where(done, 0.0, discount).astype(jnp.float32),
            dt_seconds=dt.astype(jnp.float32),
            done=done,
            truncated=truncated,
            metrics=state.metrics,
            action_mask=action_mask,
            current_car_id=state.current_car_id,
            current_node_id=state.current_node_id,
            sim_time_seconds=state.sim_time_seconds,
        )

    def _make_observation(self, state: EnvState, params: EnvParams) -> Observation:
        graph = params.graph
        size = params.raster_size
        raster = jnp.zeros((size, size, 9), dtype=jnp.float32)
        car_nodes = jnp.where(state.car_edge_id == NO_EDGE, state.car_node, state.car_to_node)
        rows = graph.node_raster_row[car_nodes]
        cols = graph.node_raster_col[car_nodes]
        active = state.car_active
        raster = raster.at[rows, cols, 0].add(jnp.where(active & (state.car_status == POLICY_CONTROLLED), 1.0, 0.0))
        raster = raster.at[rows, cols, 1].add(jnp.where(active & (state.car_status == TO_PICKUP), 1.0, 0.0))
        raster = raster.at[rows, cols, 2].add(jnp.where(active & (state.car_status == TO_DROPOFF), 1.0, 0.0))

        safe_current_node = jnp.clip(state.current_node_id, 0, graph.max_nodes - 1)
        cur_row = graph.node_raster_row[safe_current_node]
        cur_col = graph.node_raster_col[safe_current_node]
        raster = raster.at[cur_row, cur_col, 3].add(jnp.where(state.current_car_id >= 0, 1.0, 0.0))

        req_active = (state.request_status == REQ_ASSIGNED) | (state.request_status == REQ_PICKED_UP) | (state.request_status == REQ_QUEUED)
        pickup_rows = graph.node_raster_row[state.request_pickup_node]
        pickup_cols = graph.node_raster_col[state.request_pickup_node]
        drop_rows = graph.node_raster_row[state.request_dropoff_node]
        drop_cols = graph.node_raster_col[state.request_dropoff_node]
        raster = raster.at[pickup_rows, pickup_cols, 4].add(jnp.where(req_active, 1.0, 0.0))
        raster = raster.at[drop_rows, drop_cols, 5].add(jnp.where(req_active, 1.0, 0.0))
        raster = raster.at[pickup_rows, pickup_cols, 6].add(jnp.where(state.request_status == REQ_QUEUED, 1.0, 0.0))

        raster = raster.at[graph.node_raster_row, graph.node_raster_col, 7].add(graph.demand_prob)
        valid_edge_values = jnp.where(
            jnp.arange(graph.max_edges, dtype=jnp.int32) < graph.num_edges,
            state.edge_congestion,
            0.0,
        )
        raster = raster.at[graph.edge_raster_row, graph.edge_raster_col, 8].add(valid_edge_values / jnp.maximum(1.0, graph.num_edges.astype(jnp.float32)))
        raster = jnp.clip(raster, 0.0, 10.0)

        current_node = safe_current_node
        hour = (state.sim_time_seconds % DAY_SECONDS) / DAY_SECONDS * 2.0 * jnp.pi
        local_demand = graph.demand_prob[current_node]
        local_supply = jnp.sum((car_nodes == current_node) & active & (state.car_status == POLICY_CONTROLLED)).astype(jnp.float32)
        global_features = jnp.array(
            [
                jnp.sin(hour),
                jnp.cos(hour),
                jnp.maximum(0, state.current_car_id).astype(jnp.float32) / jnp.maximum(1.0, params.num_cars.astype(jnp.float32)),
                graph.node_x[current_node],
                graph.node_y[current_node],
                graph.out_degree[current_node].astype(jnp.float32),
                state.metrics.queue_length.astype(jnp.float32),
                state.metrics.fleet_utilization,
                local_supply,
                local_demand,
                state.sim_time_seconds / DAY_SECONDS,
                params.num_cars.astype(jnp.float32),
            ],
            dtype=jnp.float32,
        )

        edge_ids = graph.out_edges[current_node]
        valid = graph.out_edge_mask[current_node]
        safe_edges = jnp.clip(edge_ids, 0, graph.max_edges - 1)
        next_nodes = graph.edge_to[safe_edges]
        available_at_node = (car_nodes[None, :] == next_nodes[:, None]) & active[None, :] & (state.car_status[None, :] == POLICY_CONTROLLED)
        nearby_supply = jnp.sum(available_at_node, axis=-1).astype(jnp.float32)
        edge_time = graph.edge_base_travel_time_s[safe_edges] * state.edge_congestion[safe_edges]
        action_features = jnp.stack(
            [
                valid.astype(jnp.float32),
                next_nodes.astype(jnp.float32) / jnp.maximum(1.0, graph.num_nodes.astype(jnp.float32) - 1.0),
                graph.node_x[next_nodes],
                graph.node_y[next_nodes],
                edge_time,
                graph.edge_length_m[safe_edges],
                state.edge_congestion[safe_edges],
                graph.demand_prob[next_nodes],
                nearby_supply,
            ],
            axis=-1,
        )
        action_features = jnp.where(valid[:, None], action_features, 0.0)
        return Observation(raster=raster, global_features=global_features, action_features=action_features)
