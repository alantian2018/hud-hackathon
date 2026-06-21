from __future__ import annotations

import matplotlib
import numpy as np

import jax
import jax.numpy as jnp

from jax_fleet.debug_viz import make_debug_figure, run_debug_demo
from jax_fleet.env import make_env_params, reset, step
from jax_fleet.graph import build_synthetic_graph
from jax_fleet.ppo.train import train_smoke
from jax_fleet.scene_export import export_scene


matplotlib.use("Agg")


def small_graph():
    return build_synthetic_graph(
        node_lonlat=[(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)],
        edges=[
            {"source": 0, "target": 1, "travel_time_s": 2.0},
            {"source": 1, "target": 2, "travel_time_s": 2.0},
            {"source": 2, "target": 0, "travel_time_s": 2.0},
        ],
    )


def test_debug_visualizer_consumes_scene_export_schema() -> None:
    params = make_env_params(small_graph(), max_cars=1, max_requests=2, initial_car_nodes=[0])
    state, timestep = reset(jax.random.PRNGKey(0), params)
    state, timestep = step(state, jnp.int32(0), params)
    scene = export_scene(state, timestep, params)

    fig, ax = make_debug_figure(scene)

    assert fig is not None
    assert ax.get_title()


def test_debug_demo_saves_python_2d_artifact(tmp_path) -> None:
    out_path = tmp_path / "demo.png"
    gif_path = tmp_path / "demo.gif"

    scenes = run_debug_demo(
        graph=small_graph(),
        out_path=out_path,
        steps=3,
        seed=0,
        policy="random",
        max_cars=1,
        max_requests=2,
        spawn_rate_per_minute=0.0,
        show=False,
    )

    assert len(scenes) == 4
    assert out_path.exists()
    assert out_path.stat().st_size > 0

    run_debug_demo(
        graph=small_graph(),
        out_path=gif_path,
        steps=2,
        seed=1,
        policy="random",
        max_cars=1,
        max_requests=2,
        spawn_rate_per_minute=0.0,
        show=False,
    )
    assert gif_path.exists()
    assert gif_path.stat().st_size > 0


def test_ppo_smoke_training_on_synthetic_graph_updates_parameters() -> None:
    metrics = train_smoke(
        graph=small_graph(),
        seed=0,
        num_envs=2,
        num_steps=4,
        num_updates=1,
        learning_rate=3e-4,
    )

    assert metrics["updates"] == 1
    assert np.isfinite(metrics["last_loss"])
    assert np.isfinite(metrics["last_mean_reward"])
