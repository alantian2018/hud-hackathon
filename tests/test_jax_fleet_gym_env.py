from __future__ import annotations

import numpy as np

import jax

from jax_fleet.env import make_env_params, reset
from jax_fleet.graph import build_synthetic_graph
from jax_fleet.gym_env import JaxFleetEnv, build_arg_parser
from jax_fleet.rich_renderer import PygletWindowRenderer, RichRenderer, render_scene_to_array


def small_loop_graph():
    return build_synthetic_graph(
        node_lonlat=[
            (0.0, 0.0),
            (1.0, 0.0),
            (1.0, 1.0),
            (0.0, 1.0),
        ],
        edges=[
            {"source": 0, "target": 1, "travel_time_s": 2.0},
            {"source": 1, "target": 2, "travel_time_s": 2.0},
            {"source": 2, "target": 3, "travel_time_s": 2.0},
            {"source": 3, "target": 0, "travel_time_s": 2.0},
            {"source": 0, "target": 2, "travel_time_s": 3.0},
            {"source": 2, "target": 0, "travel_time_s": 3.0},
        ],
    )


def test_gym_env_supports_random_step_render_loop() -> None:
    env = JaxFleetEnv(
        graph=small_loop_graph(),
        seed=123,
        max_cars=2,
        max_requests=4,
        initial_car_nodes=[0, 2],
        spawn_rate_per_minute=0.0,
        episode_seconds=12.0,
        render_width=640,
        render_height=420,
    )
    observation = env.reset()

    assert observation.action_mask.shape == (env.params.graph.max_degree,)
    frames = []
    steps = 0
    while not env.done and steps < 6:
        action = env.sample_random_action()
        observation, reward, done, info = env.step(action)
        frame = env.render(mode="rgb_array")
        frames.append(frame)
        assert isinstance(action, int)
        assert np.isfinite(float(reward))
        assert done == env.done
        assert info["scene"]["current_car_id"] == int(env.state.current_car_id)
        assert info["scene"]["congestion"] == []
        steps += 1

    env.close()

    assert steps > 0
    assert frames[-1].shape == (420, 640, 3)
    assert frames[-1].dtype == np.uint8
    assert frames[-1].std() > 0.0


def test_rich_renderer_draws_info_hud_without_matplotlib() -> None:
    env = JaxFleetEnv(
        graph=small_loop_graph(),
        seed=5,
        max_cars=1,
        max_requests=2,
        initial_car_nodes=[0],
        spawn_rate_per_minute=0.0,
        render_width=800,
        render_height=480,
    )
    env.reset()
    env.step(env.sample_random_action())
    scene = env.scene()

    frame, metadata = render_scene_to_array(scene, width=800, height=480, return_metadata=True)
    hud_text = "\n".join(metadata["hud_lines"])

    env.close()

    assert frame.shape == (480, 800, 3)
    assert frame.dtype == np.uint8
    assert "time" in hud_text
    assert "current car" in hud_text
    assert "reward" in hud_text
    assert "action mask" in hud_text


def test_rich_renderer_reuses_static_road_layer() -> None:
    env = JaxFleetEnv(
        graph=small_loop_graph(),
        seed=11,
        max_cars=1,
        max_requests=2,
        initial_car_nodes=[0],
        spawn_rate_per_minute=0.0,
        render_width=640,
        render_height=420,
    )
    renderer = RichRenderer(width=640, height=420)
    env.reset()
    scene = env.scene()

    renderer.render(scene)
    renderer.render(scene)

    env.close()

    assert renderer.base_cache_hits == 1


def test_rich_renderer_reuses_static_layer_for_light_scene() -> None:
    env = JaxFleetEnv(
        graph=small_loop_graph(),
        seed=14,
        max_cars=1,
        max_requests=2,
        initial_car_nodes=[0],
        spawn_rate_per_minute=0.0,
        render_width=640,
        render_height=420,
        render_scale=1,
    )
    renderer = RichRenderer(width=640, height=420, scale=1)
    env.reset()

    renderer.render(env.scene())
    frame = renderer.render(env.scene(include_static=False, include_route_previews=False))

    env.close()

    assert renderer.base_cache_hits == 1
    assert frame.shape == (420, 640, 3)


def test_scene_export_can_skip_static_congestion_for_live_loop() -> None:
    env = JaxFleetEnv(
        graph=small_loop_graph(),
        seed=17,
        max_cars=1,
        max_requests=2,
        initial_car_nodes=[0],
        spawn_rate_per_minute=0.0,
    )
    env.reset()

    full_scene = env.scene()
    light_scene = env.scene(include_static=False, include_route_previews=False)

    env.close()

    assert len(full_scene["congestion"]) == full_scene["graph"]["num_edges"]
    assert light_scene["congestion"] == []
    assert light_scene["route_previews"] == []
    assert light_scene["graph"] == full_scene["graph"]


def test_live_cli_defaults_to_sf_graph() -> None:
    args = build_arg_parser().parse_args([])

    assert args.graph == "sf"
    assert args.start_time_seconds == 7 * 3600.0
    assert args.max_cars == 40
    assert args.render_scale == 1
    assert args.sim_steps_per_render == 1
    assert args.jit is True


def test_live_sf_defaults_leave_policy_decision_after_immediate_js_assignments() -> None:
    args = build_arg_parser().parse_args([])
    graph = build_synthetic_graph(
        node_lonlat=[(0.0, 0.0), (1.0, 0.0), (2.0, 0.0)],
        edges=[
            {"source": 0, "target": 1, "travel_time_s": 1000.0},
            {"source": 1, "target": 2, "travel_time_s": 1000.0},
            {"source": 2, "target": 0, "travel_time_s": 1000.0},
        ],
    )
    params = make_env_params(
        graph,
        max_cars=args.max_cars,
        max_requests=args.max_requests,
        initial_car_nodes=[0] * args.max_cars,
        preplanned_requests=[
            {"spawn_time_s": 0.0, "origin": 1, "destination": 2}
            for _ in range(4)
        ],
        episode_seconds=args.episode_seconds,
    )

    state, timestep = reset(jax.random.PRNGKey(args.seed), params)

    assert not bool(timestep.done)
    assert bool(state.decision_required)
    assert np.asarray(timestep.observation.action_mask).any()


def test_live_cli_accepts_fullscreen() -> None:
    args = build_arg_parser().parse_args(["--fullscreen"])

    assert args.fullscreen is True


def test_live_cli_accepts_render_speed_controls() -> None:
    args = build_arg_parser().parse_args(
        ["--render-scale", "2", "--sim-steps-per-render", "5", "--eager"]
    )

    assert args.render_scale == 2
    assert args.sim_steps_per_render == 5
    assert args.jit is False


def test_pyglet_renderer_configures_stretch_dpi(monkeypatch) -> None:
    import pyglet

    monkeypatch.setitem(pyglet.options, "dpi_scaling", "platform")

    PygletWindowRenderer.configure_pyglet(pyglet)

    assert pyglet.options["dpi_scaling"] == "stretch"
