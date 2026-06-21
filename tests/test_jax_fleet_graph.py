from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from jax_fleet.graph import build_synthetic_graph, load_public_data_graph
from jax_fleet.routing import next_edge, shortest_path_edges, travel_time_estimate


def test_synthetic_graph_has_variable_degree_action_mask_without_stay_action() -> None:
    graph = build_synthetic_graph(
        node_lonlat=[(0.0, 0.0), (1.0, 0.0), (2.0, 0.0)],
        edges=[
            {"source": 0, "target": 1, "travel_time_s": 5.0},
            {"source": 0, "target": 2, "travel_time_s": 9.0},
            {"source": 1, "target": 2, "travel_time_s": 4.0},
            {"source": 2, "target": 0, "travel_time_s": 7.0},
        ],
    )

    assert graph.max_degree == 2
    np.testing.assert_array_equal(np.asarray(graph.outgoing_mask[0]), [True, True])
    np.testing.assert_array_equal(np.asarray(graph.outgoing_mask[1]), [True, False])
    assert np.all(np.asarray(graph.outgoing_target_nodes[0]) != 0)


def test_directed_routing_is_asymmetric_and_matches_oracle_path() -> None:
    graph = build_synthetic_graph(
        node_lonlat=[(0.0, 0.0), (1.0, 0.0), (2.0, 0.0)],
        edges=[
            {"source": 0, "target": 1, "travel_time_s": 5.0},
            {"source": 1, "target": 0, "travel_time_s": 20.0},
            {"source": 1, "target": 2, "travel_time_s": 6.0},
            {"source": 2, "target": 1, "travel_time_s": 2.0},
        ],
    )

    assert math.isclose(float(travel_time_estimate(graph, 0, 2)), 11.0)
    assert math.isclose(float(travel_time_estimate(graph, 2, 0)), 22.0)
    assert int(next_edge(graph, 0, 2)) == 0
    assert shortest_path_edges(graph, 0, 2) == [0, 2]
    assert shortest_path_edges(graph, 2, 0) == [3, 1]


def test_public_data_loader_uses_largest_directed_scc_without_routing_cache() -> None:
    graph = load_public_data_graph(Path("public/data"), include_routing=False)

    assert graph.num_nodes == 8783
    assert graph.num_edges == 22016
    assert graph.max_degree == 6
    assert graph.routing_next_edge.shape == (0, 0)
    assert graph.original_node_ids.shape == (8783,)
