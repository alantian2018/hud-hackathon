from __future__ import annotations

import unittest

from mobility_sim import (
    CnnFeatureConfig,
    GreedyDispatcher,
    DemandGenerator,
    GridRouter,
    PeopleGenerator,
    PersonRequest,
    TrafficGenerator,
    WorldGenerators,
    build_cnn_feature_channels,
    build_cnn_training_examples,
    build_mobility_agent_state,
    candidate_next_cells,
)


def assert_heatmap_shape_and_range(testcase: unittest.TestCase, heatmap, rows: int, cols: int) -> None:
    testcase.assertEqual(len(heatmap), rows)
    for row in heatmap:
        testcase.assertEqual(len(row), cols)
        for value in row:
            testcase.assertGreaterEqual(value, 0.0)
            testcase.assertLessEqual(value, 1.0)


class GeneratorTests(unittest.TestCase):
    def test_world_generators_are_deterministic_by_seed(self) -> None:
        grid = (8, 8)
        first = WorldGenerators(grid, seed=42).step(8 * 60)
        second = WorldGenerators(grid, seed=42).step(8 * 60)
        different = WorldGenerators(grid, seed=43).step(8 * 60)

        self.assertEqual(first, second)
        self.assertNotEqual(first["demand_heatmap"], different["demand_heatmap"])

    def test_demand_heatmap_shape_and_range(self) -> None:
        generator = DemandGenerator((7, 9), seed=3)
        heatmap = generator.get_heatmap(12 * 60)

        assert_heatmap_shape_and_range(self, heatmap, 7, 9)
        top = generator.top_demand_cells(3, timestep=12 * 60)
        self.assertEqual(len(top), 3)
        self.assertGreaterEqual(top[0]["value"], top[-1]["value"])

    def test_poisson_generation_is_bounded(self) -> None:
        demand = [[1.0 for _ in range(5)] for _ in range(5)]
        generator = PeopleGenerator((5, 5), seed=9, base_arrival_rate=80, max_new_people_per_tick=6)
        people = generator.generate(5, demand)

        self.assertLessEqual(len(people), 6)

    def test_people_origins_and_destinations_are_valid(self) -> None:
        demand = [[0.0 for _ in range(6)] for _ in range(6)]
        demand[2][3] = 1.0
        traffic = [[0.1 for _ in range(6)] for _ in range(6)]
        generator = PeopleGenerator((6, 6), seed=12, base_arrival_rate=60, max_new_people_per_tick=10)
        people = generator.generate(17, demand, traffic)

        self.assertGreater(len(people), 0)
        for person in people:
            self.assertTrue(0 <= person.origin[0] < 6)
            self.assertTrue(0 <= person.origin[1] < 6)
            self.assertTrue(0 <= person.destination[0] < 6)
            self.assertTrue(0 <= person.destination[1] < 6)
            self.assertNotEqual(person.origin, person.destination)

    def test_people_dropoffs_are_twenty_plus_grids_away_when_possible(self) -> None:
        demand = [[0.8 for _ in range(50)] for _ in range(50)]
        generator = PeopleGenerator(
            (50, 50),
            seed=21,
            base_arrival_rate=80,
            max_new_people_per_tick=8,
            min_trip_distance=20,
        )
        people = generator.generate(31, demand)

        self.assertGreater(len(people), 0)
        for person in people:
            distance = abs(person.origin[0] - person.destination[0]) + abs(
                person.origin[1] - person.destination[1]
            )
            self.assertGreaterEqual(distance, 20)

    def test_people_origins_follow_demand_distribution(self) -> None:
        demand = [[0.0 for _ in range(5)] for _ in range(5)]
        demand[0][0] = 1.0
        generator = PeopleGenerator((5, 5), seed=30, base_arrival_rate=90, max_new_people_per_tick=12)
        people = generator.generate(99, demand)

        self.assertGreater(len(people), 0)
        high_demand_origins = sum(1 for person in people if person.origin == (0, 0))
        self.assertGreaterEqual(high_demand_origins, max(1, int(len(people) * 0.75)))

    def test_people_arrivals_increase_with_traffic_pressure(self) -> None:
        demand = [[1.0 for _ in range(8)] for _ in range(8)]
        low_traffic = [[0.0 for _ in range(8)] for _ in range(8)]
        high_traffic = [[1.0 for _ in range(8)] for _ in range(8)]
        generator = PeopleGenerator((8, 8), seed=11, base_arrival_rate=8, max_new_people_per_tick=20)

        low = generator.generate(44, demand, low_traffic)
        high = generator.generate(44, demand, high_traffic)

        self.assertGreater(len(high), len(low))

    def test_traffic_modulation_and_noise_are_clipped(self) -> None:
        generator = TrafficGenerator((8, 8), seed=5, noise_level=0.2, demand_coupling=0.35)
        demand = [[1.0 if row == col else 0.2 for col in range(8)] for row in range(8)]
        night = generator.get_heatmap(3 * 60, demand)
        rush = generator.get_heatmap(8 * 60, demand)

        assert_heatmap_shape_and_range(self, night, 8, 8)
        assert_heatmap_shape_and_range(self, rush, 8, 8)
        night_mean = sum(sum(row) for row in night) / 64
        rush_mean = sum(sum(row) for row in rush) / 64
        self.assertGreater(rush_mean, night_mean)

    def test_dijkstra_routes_around_high_traffic_cells(self) -> None:
        traffic = [[0.0 for _ in range(5)] for _ in range(5)]
        for col in (1, 2, 3):
            traffic[2][col] = 1.0
        router = GridRouter((5, 5), traffic_weight=20)
        route = router.route((2, 0), (2, 4), traffic)

        self.assertEqual(route.path[0], (2, 0))
        self.assertEqual(route.path[-1], (2, 4))
        self.assertTrue(all(cell not in route.path for cell in [(2, 1), (2, 2), (2, 3)]))

    def test_greedy_dispatch_assigns_nearest_car_and_stalls_idle_cars(self) -> None:
        dispatcher = GreedyDispatcher(
            (10, 10),
            seed=4,
            initial_car_positions=[(0, 0), (9, 9)],
        )
        traffic = [[0.0 for _ in range(10)] for _ in range(10)]
        people = [
            PersonRequest(
                id="person-a",
                origin=(1, 1),
                destination=(9, 1),
                created_at=1,
                patience=20,
                value=20.0,
            )
        ]
        result = dispatcher.step(1, people, traffic)

        self.assertEqual(result["assignments"][0]["car_id"], "car-0")
        self.assertEqual(result["assignments"][0]["person_id"], "person-a")
        self.assertEqual(result["stalled_cars"][0]["id"], "car-1")
        self.assertEqual(result["stalled_cars"][0]["stall_ticks"], 1)

    def test_world_step_exposes_people_and_dispatch_grids(self) -> None:
        payload = WorldGenerators((12, 12), seed=18, fleet_size=4).step(8 * 60)

        self.assertIn("people_grid", payload)
        self.assertIn("dispatch", payload)
        self.assertIn("agent_state", payload)
        self.assertIn("pickup_grid", payload["people_grid"])
        self.assertIn("dropoff_grid", payload["people_grid"])
        self.assertIn("car_grid", payload["dispatch"])
        self.assertEqual(
            payload["summary"]["cnn_training_examples"],
            len(payload["dispatch"]["cars"]),
        )

    def test_cnn_feature_channels_have_expected_shape(self) -> None:
        grid = (6, 6)
        demand = [[0.2 for _ in range(6)] for _ in range(6)]
        traffic = [[0.1 for _ in range(6)] for _ in range(6)]
        people = [
            PersonRequest(
                id="person-a",
                origin=(2, 2),
                destination=(5, 5),
                created_at=1,
                patience=20,
                value=18.0,
            )
        ]
        dispatcher = GreedyDispatcher(grid, initial_car_positions=[(1, 1)], seed=1)
        dispatch = dispatcher.step(1, people, traffic)

        features = build_cnn_feature_channels(grid, demand, traffic, people, dispatch["cars"])

        self.assertEqual(features["shape"], [7, 6, 6])
        self.assertEqual(features["channel_names"][0], "demand")
        self.assertEqual(features["channels"][4][2][2], 1.0)
        self.assertEqual(features["channels"][5][5][5], 1.0)

    def test_cnn_training_examples_include_local_patch_and_greedy_label(self) -> None:
        grid = (8, 8)
        demand = [[0.4 for _ in range(8)] for _ in range(8)]
        traffic = [[0.0 for _ in range(8)] for _ in range(8)]
        people = [
            PersonRequest(
                id="person-a",
                origin=(1, 2),
                destination=(6, 6),
                created_at=1,
                patience=20,
                value=22.0,
            )
        ]
        dispatcher = GreedyDispatcher(grid, initial_car_positions=[(1, 1)], seed=1)
        dispatch = dispatcher.step(1, people, traffic)

        examples = build_cnn_training_examples(
            grid,
            demand,
            traffic,
            people,
            dispatch["cars"],
            config=CnnFeatureConfig(patch_radius=2),
        )

        self.assertEqual(len(examples), 1)
        example = examples[0]
        self.assertEqual(example["local_patch"]["shape"], [7, 5, 5])
        self.assertIn(example["label"]["action"], {"WAIT", "N", "E", "S", "W"})
        self.assertEqual(
            tuple(example["candidate_moves"][example["label"]["action_index"]]["next_cell"]),
            tuple(example["label"]["next_cell"]),
        )

    def test_candidate_next_cells_respect_grid_bounds(self) -> None:
        candidates = candidate_next_cells((3, 3), (0, 0))
        actions = {candidate["action"] for candidate in candidates}

        self.assertEqual(actions, {"WAIT", "E", "S"})

    def test_agent_state_wraps_global_state_and_cnn_examples(self) -> None:
        payload = WorldGenerators((10, 10), seed=5, fleet_size=3).step(8 * 60)
        agent_state = build_mobility_agent_state(
            (10, 10),
            payload["timestep"],
            payload["demand_heatmap"],
            payload["traffic_heatmap"],
            payload["new_people"],
            payload["dispatch"],
            greedy_stats=payload["greedy_stats"],
        )

        self.assertIn("global_state", agent_state)
        self.assertIn("cnn", agent_state)
        self.assertEqual(len(agent_state["cnn"]["training_examples"]), 3)
        self.assertIn("top_cells", agent_state["global_state"]["demand"])


if __name__ == "__main__":
    unittest.main()
