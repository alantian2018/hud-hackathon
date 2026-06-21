from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from jax_fleet.env import make_env_params, reset as functional_reset, step as functional_step
from jax_fleet.graph import build_synthetic_graph, load_public_data_graph
from jax_fleet.heuristics import choose_marginal_value_action
from jax_fleet.ppo.policy import load_checkpoint_policy, resolve_policy_checkpoint_path
from jax_fleet.rich_renderer import PygletWindowRenderer, RichRenderer
from jax_fleet.scene_export import export_scene
from jax_fleet.spawns import make_spawned_env_params
from jax_fleet.types import EnvParams, EnvState, GraphArrays, Observation, Timestep


class JaxFleetEnv:
    """Small Gym-like wrapper around the functional JAX fleet environment."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 12}

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
        assignment_max_route_edges: int = 10000,
        episode_seconds: float = 1800.0,
        start_time_seconds: float = 0.0,
        reward_mode: str = "dense_wait",
        observation_mode: str = "learning_v1",
        drop_penalty: float = 10.0,
        pickup_bonus: float = 0.0,
        time_discount_reference_seconds: float = 60.0,
        render_width: int = 1280,
        render_height: int = 800,
        render_scale: int = 1,
        render_fullscreen: bool = False,
        use_jit: bool = False,
    ) -> None:
        self.params = params or make_env_params(
            graph,
            max_cars=max_cars,
            max_requests=max_requests,
            initial_car_nodes=initial_car_nodes,
            spawn_rate_per_minute=spawn_rate_per_minute,
            assignment_max_route_edges=assignment_max_route_edges,
            episode_seconds=episode_seconds,
            start_time_seconds=start_time_seconds,
            reward_mode=reward_mode,
            observation_mode=observation_mode,
            drop_penalty=drop_penalty,
            pickup_bonus=pickup_bonus,
            time_discount_reference_seconds=time_discount_reference_seconds,
        )
        self.seed = int(seed)
        self._rng = jax.random.PRNGKey(self.seed)
        self._action_rng = np.random.default_rng(self.seed + 77)
        self._state: EnvState | None = None
        self._timestep: Timestep | None = None
        self._reset_fn = jax.jit(functional_reset) if use_jit else functional_reset
        self._step_fn = jax.jit(functional_step) if use_jit else functional_step
        self._renderer = RichRenderer(width=render_width, height=render_height, scale=render_scale)
        self._human_renderer: PygletWindowRenderer | None = None
        self._render_fullscreen = bool(render_fullscreen)

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
        return bool(np.asarray(self.timestep.done)) if self._timestep is not None else False

    @property
    def human_window_open(self) -> bool:
        return self._human_renderer is None or not self._human_renderer.closed

    def reset(self, *, seed: int | None = None) -> Observation:
        if seed is not None:
            self.seed = int(seed)
            self._rng = jax.random.PRNGKey(self.seed)
            self._action_rng = np.random.default_rng(self.seed + 77)
        self._rng, reset_key = jax.random.split(self._rng)
        self._state, self._timestep = self._reset_fn(reset_key, self.params)
        return self._timestep.observation

    def step(self, action: int) -> tuple[Observation, float, bool, dict[str, Any]]:
        if self._state is None or self._timestep is None:
            self.reset()
        self._state, self._timestep = self._step_fn(
            self.state,
            jnp.asarray(int(action), dtype=jnp.int32),
            self.params,
        )
        info = {
            "scene": self.scene(include_static=False, include_route_previews=False),
            "dt_seconds": float(np.asarray(self._timestep.dt_seconds)),
            "discount": float(np.asarray(self._timestep.discount)),
            "metrics": self._timestep.metrics,
        }
        return (
            self._timestep.observation,
            float(np.asarray(self._timestep.reward)),
            bool(np.asarray(self._timestep.done)),
            info,
        )

    def sample_random_action(self) -> int:
        mask = np.asarray(self.timestep.observation.action_mask, dtype=bool)
        valid = np.flatnonzero(mask)
        if len(valid) == 0:
            return 0
        return int(self._action_rng.choice(valid))

    def render(self, mode: str = "human") -> np.ndarray | None:
        if self._state is None or self._timestep is None:
            self.reset()
        time_seconds = float(np.asarray(self.state.time_seconds))
        frame = self._renderer.render(
            self.scene(
                include_static=self._renderer.needs_static_scene(time_seconds),
                include_route_previews=True,
            )
        )
        if mode == "rgb_array":
            return frame
        if mode != "human":
            raise ValueError(f"unsupported render mode {mode!r}")
        if self._human_renderer is None:
            height, width = frame.shape[:2]
            self._human_renderer = PygletWindowRenderer(
                width=width,
                height=height,
                fullscreen=self._render_fullscreen,
            )
        self._human_renderer.render(frame)
        return None

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

    def close(self) -> None:
        if self._human_renderer is not None:
            self._human_renderer.close()
            self._human_renderer = None


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)

    if not args.jit:
        jax.config.update("jax_disable_jit", True)

    graph = _load_graph(args.graph, data_dir=args.data_dir, cache_dir=args.cache_dir)
    render_width, render_height = _render_size(args.width, args.height, fullscreen=args.fullscreen)
    spawn_source = args.spawn_source or ("density" if args.graph == "sf" else "uniform")
    initial_nodes = (
        None
        if spawn_source in {"js-visual", "density"}
        else _initial_car_nodes(graph.num_nodes, args.max_cars, args.seed)
    )
    params = make_spawned_env_params(
        graph,
        graph_name=args.graph,
        data_dir=args.data_dir,
        spawn_source=spawn_source,
        seed=args.seed,
        max_cars=args.max_cars,
        max_requests=args.max_requests,
        initial_car_nodes=initial_nodes,
        spawn_rate_per_minute=args.spawn_rate_per_minute,
        assignment_max_route_edges=args.assignment_max_route_edges,
        episode_seconds=args.episode_seconds,
        start_time_seconds=args.start_time_seconds,
        reward_mode=args.reward_mode,
        observation_mode=args.observation_mode,
        drop_penalty=args.drop_penalty,
        pickup_bonus=args.pickup_bonus,
        time_discount_reference_seconds=args.time_discount_reference_seconds,
    )
    env = JaxFleetEnv(
        graph=graph,
        params=params,
        seed=args.seed,
        render_width=render_width,
        render_height=render_height,
        render_scale=args.render_scale,
        render_fullscreen=args.fullscreen,
        use_jit=args.jit,
    )
    env.reset()
    checkpoint_policy = None
    if args.policy == "checkpoint" or args.policy_checkpoint is not None:
        checkpoint_path = resolve_policy_checkpoint_path(
            args.policy_checkpoint,
            checkpoint_dir=args.policy_checkpoint_dir,
        )
        checkpoint_policy = load_checkpoint_policy(
            checkpoint_path,
            env.timestep.observation,
            max_degree=graph.max_degree,
            use_jit=args.jit,
        )
        print(
            "loaded policy checkpoint path={path} update={update}".format(
                path=checkpoint_policy.path,
                update=checkpoint_policy.update if checkpoint_policy.update is not None else "unknown",
            ),
            flush=True,
        )
    env.render(mode="human")

    steps = 0
    try:
        while (
            not env.done
            and env.human_window_open
            and _within_step_limit(steps, args.max_steps)
            and _within_pickup_limit(env, args.max_pickups)
        ):
            started = time.perf_counter()
            frame_steps = 0
            while (
                frame_steps < max(1, args.sim_steps_per_render)
                and not env.done
                and env.human_window_open
                and _within_step_limit(steps, args.max_steps)
                and _within_pickup_limit(env, args.max_pickups)
            ):
                action = _choose_live_action(env, args.policy, checkpoint_policy)
                _, reward, done, info = env.step(action)
                steps += 1
                frame_steps += 1
                if steps == 1 or steps % 25 == 0 or done:
                    print(
                        "step={step} action={action} reward={reward:.3f} time={time:.1f}s done={done}".format(
                            step=steps,
                            action=action,
                            reward=reward,
                            time=info["scene"]["time_seconds"],
                            done=done,
                        )
                    )
            env.render(mode="human")
            delay = max(0.0, (1.0 / max(args.fps, 1.0e-6)) - (time.perf_counter() - started))
            time.sleep(delay)
        if args.hold_seconds > 0.0 and env.human_window_open:
            end = time.perf_counter() + args.hold_seconds
            while time.perf_counter() < end and env.human_window_open:
                env.render(mode="human")
                time.sleep(1.0 / max(args.fps, 1.0))
    finally:
        env.close()

    print(f"live render loop exited after {steps} {args.policy} policy steps")
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a live Gym-style JAX Fleet render loop.")
    parser.add_argument("--graph", choices=["synthetic", "sf"], default="sf")
    parser.add_argument("--data-dir", default="dist/data")
    parser.add_argument("--cache-dir", default="cache/jax_fleet")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-cars", type=int, default=40)
    parser.add_argument("--max-requests", type=int, default=32)
    parser.add_argument("--assignment-max-route-edges", type=int, default=10000)
    parser.add_argument("--spawn-rate-per-minute", type=float, default=0.0)
    parser.add_argument("--spawn-source", choices=["uniform", "density", "js-visual"], default=None)
    parser.add_argument("--reward-mode", choices=["dense_wait", "legacy_pickup_wait"], default="dense_wait")
    parser.add_argument("--observation-mode", choices=["learning_v1", "legacy"], default="learning_v1")
    parser.add_argument("--drop-penalty", type=float, default=10.0)
    parser.add_argument("--pickup-bonus", type=float, default=0.0)
    parser.add_argument("--time-discount-reference-seconds", type=float, default=60.0)
    parser.add_argument("--policy", choices=["random", "first", "heuristic", "checkpoint"], default="random")
    parser.add_argument(
        "--policy-checkpoint",
        default=None,
        help='Checkpoint path to load, or "latest" to resolve from --policy-checkpoint-dir.',
    )
    parser.add_argument("--policy-checkpoint-dir", default="runs/jax_fleet/sf/checkpoints")
    parser.add_argument("--episode-seconds", type=float, default=float("inf"))
    parser.add_argument("--start-time-seconds", type=float, default=7.0 * 3600.0)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Optional policy-step cap. Omit or pass 0 for an endless live loop.",
    )
    parser.add_argument(
        "--max-pickups",
        type=int,
        default=None,
        help="Optional pickup cap. Omit or pass 0 to disable.",
    )
    parser.add_argument("--fps", type=float, default=12.0)
    parser.add_argument("--sim-steps-per-render", type=int, default=1)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=800)
    parser.add_argument("--render-scale", type=int, default=1)
    parser.add_argument(
        "--fullscreen",
        "--maximize",
        dest="fullscreen",
        action="store_true",
        help="Maximize a normal DPI-safe window instead of using native fullscreen.",
    )
    parser.add_argument("--hold-seconds", type=float, default=0.0)
    execution = parser.add_mutually_exclusive_group()
    execution.add_argument(
        "--jit",
        dest="jit",
        action="store_true",
        default=True,
        help="Compile reset/step with JAX JIT. This is the default for the live loop.",
    )
    execution.add_argument(
        "--eager",
        dest="jit",
        action="store_false",
        help="Disable JAX JIT for easier debugging and stack traces.",
    )
    return parser


def _within_step_limit(steps: int, max_steps: int | None) -> bool:
    return max_steps is None or int(max_steps) <= 0 or int(steps) < int(max_steps)


def _within_pickup_limit(env: JaxFleetEnv, max_pickups: int | None) -> bool:
    if max_pickups is None or int(max_pickups) <= 0:
        return True
    return int(np.asarray(env.state.metrics.picked_up_requests)) < int(max_pickups)


def _choose_live_action(env: JaxFleetEnv, policy: str, checkpoint_policy) -> int:
    if checkpoint_policy is not None:
        return checkpoint_policy.action(env.timestep.observation)
    if policy == "heuristic":
        return int(np.asarray(choose_marginal_value_action(env.timestep.observation)))
    if policy == "first":
        mask = np.asarray(env.timestep.observation.action_mask, dtype=bool)
        valid = np.flatnonzero(mask)
        return int(valid[0]) if len(valid) else 0
    return env.sample_random_action()


def _render_size(width: int, height: int, *, fullscreen: bool) -> tuple[int, int]:
    if not fullscreen:
        return int(width), int(height)
    try:
        import pyglet

        PygletWindowRenderer.configure_pyglet(pyglet)
        screen = pyglet.display.get_display().get_default_screen()
        return int(screen.width), int(screen.height)
    except Exception:
        return int(width), int(height)


def _load_graph(kind: str, *, data_dir: str | Path, cache_dir: str | Path) -> GraphArrays:
    if kind == "synthetic":
        return build_synthetic_graph(
            node_lonlat=[
                (0.0, 0.0),
                (1.0, 0.0),
                (1.0, 1.0),
                (0.0, 1.0),
                (0.5, 1.45),
                (1.45, 0.5),
            ],
            edges=[
                {"source": 0, "target": 1, "travel_time_s": 2.0},
                {"source": 1, "target": 2, "travel_time_s": 2.0, "hourly_multiplier": {17: 1.8}},
                {"source": 2, "target": 3, "travel_time_s": 2.0},
                {"source": 3, "target": 0, "travel_time_s": 2.0},
                {"source": 0, "target": 2, "travel_time_s": 3.0},
                {"source": 2, "target": 0, "travel_time_s": 3.0},
                {"source": 1, "target": 5, "travel_time_s": 1.8},
                {"source": 5, "target": 2, "travel_time_s": 1.8},
                {"source": 2, "target": 4, "travel_time_s": 1.6},
                {"source": 4, "target": 3, "travel_time_s": 1.6},
                {"source": 4, "target": 5, "travel_time_s": 2.4},
                {"source": 5, "target": 4, "travel_time_s": 2.4},
            ],
        )
    return load_public_data_graph(data_dir, include_routing=True, cache_dir=cache_dir)


def _initial_car_nodes(num_nodes: int, max_cars: int, seed: int) -> list[int]:
    rng = np.random.default_rng(seed + 2026)
    return [int(node) for node in rng.integers(0, num_nodes, size=max_cars)]


if __name__ == "__main__":
    raise SystemExit(main())
