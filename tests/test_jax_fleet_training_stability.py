from __future__ import annotations

import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

from jax_fleet.graph import (
    build_synthetic_graph,
    ensure_routing_cache,
    load_routing_cache,
)
from jax_fleet.ppo.train import TrainingConfig, compute_gae, train
from jax_fleet.cli import build_parser


def stable_graph():
    return build_synthetic_graph(
        node_lonlat=[(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)],
        edges=[
            {"source": 0, "target": 1, "travel_time_s": 2.0},
            {"source": 1, "target": 2, "travel_time_s": 3.0},
            {"source": 2, "target": 3, "travel_time_s": 4.0},
            {"source": 3, "target": 0, "travel_time_s": 5.0},
            {"source": 0, "target": 2, "travel_time_s": 9.0},
        ],
    )


def test_chunked_routing_cache_writes_manifest_and_reuses_hit(tmp_path: Path) -> None:
    graph = stable_graph()
    cache_dir = tmp_path / "routing"

    info = ensure_routing_cache(
        num_nodes=graph.num_nodes,
        sources=np.asarray(graph.edge_sources),
        targets=np.asarray(graph.edge_targets),
        travel_time_s=np.asarray(graph.edge_travel_time_s).mean(axis=1),
        cache_dir=cache_dir,
        graph_key="unit",
        chunk_size=1,
    )

    assert info.status == "built"
    assert info.table_path.exists()
    assert info.manifest_path.exists()
    manifest = json.loads(info.manifest_path.read_text(encoding="utf-8"))
    assert manifest["num_nodes"] == 4
    assert manifest["num_edges"] == 5
    assert manifest["chunk_size"] == 1

    next_edges, travel_times = load_routing_cache(info.table_path)
    assert next_edges.shape == (4, 4)
    assert travel_times.shape == (4, 4)
    assert int(next_edges[0, 3]) == 0

    reused = ensure_routing_cache(
        num_nodes=graph.num_nodes,
        sources=np.asarray(graph.edge_sources),
        targets=np.asarray(graph.edge_targets),
        travel_time_s=np.asarray(graph.edge_travel_time_s).mean(axis=1),
        cache_dir=cache_dir,
        graph_key="unit",
        chunk_size=2,
    )

    assert reused.status == "hit"
    assert reused.fingerprint == info.fingerprint


def test_gae_uses_fixed_transition_discount_inputs() -> None:
    rewards = np.asarray([[1.0], [2.0]], dtype=np.float32)
    values = np.asarray([[0.5], [0.25]], dtype=np.float32)
    bootstrap = np.asarray([0.75], dtype=np.float32)
    discounts = np.asarray([[0.9], [0.9]], dtype=np.float32)
    dones = np.asarray([[False], [False]])

    advantages, returns = compute_gae(
        rewards=rewards,
        values=values,
        bootstrap_value=bootstrap,
        discounts=discounts,
        dones=dones,
        gae_lambda=0.8,
    )

    expected_last = 2.0 + 0.9 * 0.75 - 0.25
    expected_first = 1.0 + 0.9 * 0.25 - 0.5 + 0.9 * 0.8 * expected_last
    np.testing.assert_allclose(np.asarray(advantages).ravel(), [expected_first, expected_last], rtol=1e-6)
    np.testing.assert_allclose(np.asarray(returns), np.asarray(advantages) + values, rtol=1e-6)


def test_train_writes_metrics_checkpoint_and_resumes(tmp_path: Path) -> None:
    graph = stable_graph()
    metrics_path = tmp_path / "metrics.jsonl"
    checkpoint_dir = tmp_path / "checkpoints"
    config = TrainingConfig(
        graph_name="synthetic",
        seed=3,
        num_envs=2,
        num_steps=3,
        num_updates=1,
        max_cars=1,
        max_requests=4,
        learning_rate=3e-4,
        checkpoint_dir=checkpoint_dir,
        checkpoint_every=1,
        metrics_path=metrics_path,
    )

    first = train(config, graph=graph)

    assert first["updates"] == 1
    assert np.isfinite(first["last_approx_kl"])
    assert np.isfinite(first["last_clipfrac"])
    assert first["latest_checkpoint"] is not None
    assert Path(first["latest_checkpoint"]).exists()
    lines = metrics_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    first_metrics = json.loads(lines[0])
    assert first_metrics["update"] == 1
    assert first_metrics["latest_checkpoint"] == first["latest_checkpoint"]
    assert np.isfinite(first_metrics["losses/explained_variance"]) or np.isnan(
        first_metrics["losses/explained_variance"]
    )
    assert first_metrics["charts/global_step"] == config.num_envs * config.num_steps
    assert 0.0 <= first_metrics["rollout/mean_discount"] <= 1.0

    resumed = train(config.replace(num_updates=2, resume=True), graph=graph)

    assert resumed["updates"] == 2
    lines = metrics_path.read_text(encoding="utf-8").strip().splitlines()
    assert [json.loads(line)["update"] for line in lines] == [1, 2]


def test_train_logs_to_wandb_when_tracking_enabled(tmp_path: Path, monkeypatch) -> None:
    graph = stable_graph()
    logged: list[tuple[dict, int | None]] = []
    finished = {"value": False}
    init_kwargs = {}

    class FakeRun:
        def log(self, metrics, step=None):
            logged.append((dict(metrics), step))

        def finish(self):
            finished["value"] = True

    def fake_init(**kwargs):
        init_kwargs.update(kwargs)
        return FakeRun()

    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(init=fake_init))
    config = TrainingConfig(
        graph_name="synthetic",
        seed=5,
        num_envs=1,
        num_steps=2,
        num_updates=1,
        max_cars=1,
        max_requests=4,
        checkpoint_dir=None,
        metrics_path=tmp_path / "metrics.jsonl",
        track=True,
        wandb_project_name="unit-project",
        wandb_run_name="unit-run",
        wandb_mode="disabled",
    )

    train(config, graph=graph)

    assert init_kwargs["project"] == "unit-project"
    assert init_kwargs["name"] == "unit-run"
    assert init_kwargs["mode"] == "disabled"
    assert init_kwargs["config"]["track"] is True
    assert len(logged) == 1
    assert logged[0][1] == config.num_envs * config.num_steps
    assert "charts/SPS" in logged[0][0]
    assert "losses/approx_kl" in logged[0][0]
    assert finished["value"] is True


def test_train_resolves_relative_artifact_paths(tmp_path: Path, monkeypatch) -> None:
    graph = stable_graph()
    monkeypatch.chdir(tmp_path)
    config = TrainingConfig(
        graph_name="synthetic",
        seed=4,
        num_envs=1,
        num_steps=1,
        num_updates=1,
        max_cars=1,
        max_requests=4,
        checkpoint_dir=Path("relative_checkpoints"),
        metrics_path=Path("relative_metrics/metrics.jsonl"),
    )

    result = train(config, graph=graph)

    assert Path(result["latest_checkpoint"]).is_absolute()
    assert Path(result["latest_checkpoint"]).exists()
    assert Path("relative_metrics/metrics.jsonl").exists()


def test_cli_parser_exposes_train_and_prepare_routing_commands(tmp_path: Path) -> None:
    parser = build_parser()
    default_train_args = parser.parse_args(["train"])
    default_benchmark_args = parser.parse_args(["benchmark-env"])

    assert default_train_args.graph == "sf"
    assert default_benchmark_args.graph == "sf"
    assert default_train_args.data_dir == "dist/data"
    assert default_benchmark_args.data_dir == "dist/data"
    assert default_train_args.max_cars == 40
    assert default_train_args.max_requests == 32
    assert default_benchmark_args.max_cars == 40
    assert default_benchmark_args.max_requests == 32
    assert default_train_args.require_gpu is False
    assert default_benchmark_args.require_gpu is False

    gpu_args = parser.parse_args(["check-gpu", "--require-gpu"])
    assert gpu_args.command == "check-gpu"
    assert gpu_args.require_gpu is True

    train_args = parser.parse_args(
        [
            "train",
            "--graph",
            "synthetic",
            "--spawn-source",
            "uniform",
            "--num-updates",
            "2",
            "--checkpoint-dir",
            str(tmp_path / "ckpt"),
        ]
    )
    assert train_args.command == "train"
    assert train_args.graph == "synthetic"
    assert train_args.spawn_source == "uniform"
    assert train_args.num_updates == 2
    assert train_args.assignment_max_route_edges == 10000
    assert train_args.reward_mode == "dense_wait"
    assert train_args.observation_mode == "learning_v1"
    assert train_args.drop_penalty == 10.0
    assert train_args.pickup_bonus == 0.0
    assert train_args.time_discount_reference_seconds == 60.0
    assert train_args.update_epochs == 4
    assert train_args.num_minibatches == 4
    assert train_args.require_gpu is False
    assert train_args.track is False
    assert train_args.wandb_project_name == "jax_fleet"
    assert train_args.wandb_video_every == 0
    assert train_args.wandb_video_max_steps == 50000
    assert train_args.wandb_video_max_pickups == 20
    assert train_args.wandb_video_max_frames == 240

    routing_args = parser.parse_args(
        [
            "prepare-routing",
            "--data-dir",
            "public/data",
            "--cache-dir",
            str(tmp_path / "routing"),
            "--chunk-size",
            "128",
        ]
    )
    assert routing_args.command == "prepare-routing"
    assert routing_args.chunk_size == 128

    benchmark_args = parser.parse_args(
        [
            "benchmark-env",
            "--graph",
            "synthetic",
            "--steps",
            "8",
            "--num-envs",
            "2",
        ]
    )
    assert benchmark_args.command == "benchmark-env"
    assert benchmark_args.steps == 8
    assert benchmark_args.num_envs == 2
    assert benchmark_args.assignment_max_route_edges == 10000
    assert benchmark_args.reward_mode == "dense_wait"
    assert benchmark_args.observation_mode == "learning_v1"
    assert benchmark_args.require_gpu is False
