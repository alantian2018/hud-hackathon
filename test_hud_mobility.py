from __future__ import annotations

import unittest

from hud_mobility.planners import build_value_aware_plan, global_batch_matching
from hud_mobility.schemas import ActionPlan
from hud_mobility.world import MobilityWorld


class MobilityOrchestratorTests(unittest.TestCase):
    def test_value_aware_policy_runs_episode_without_baseline(self) -> None:
        world = MobilityWorld((12, 12), seed=41, fleet_size=10, horizon_steps=6, demand_scale=1.0)
        while not world.done:
            world.step(build_value_aware_plan(world))
        result = world.reward()
        self.assertGreaterEqual(result["reward"], 0.0)
        self.assertLessEqual(result["reward"], 1.0)
        self.assertGreater(world.total_requests, 0)
        self.assertEqual(world.invalid_actions, 0)

    def test_reward_spread_across_seeds(self) -> None:
        rewards = []
        for seed in (51, 52, 53, 54):
            world = MobilityWorld((12, 12), seed=seed, fleet_size=8, horizon_steps=5, demand_scale=1.1)
            while not world.done:
                world.step(build_value_aware_plan(world))
            rewards.append(world.reward()["reward"])
        self.assertGreater(max(rewards) - min(rewards), 0.01)

    def test_invalid_actions_are_penalized(self) -> None:
        world = MobilityWorld((10, 10), seed=61, fleet_size=4, horizon_steps=2)
        world.step(ActionPlan.from_any({"assignments": [{"car_id": "missing", "person_id": "missing"}]}))
        result = world.reward()
        self.assertGreater(world.invalid_actions, 0)
        self.assertGreater(result["components"]["invalid_penalty"], 0.0)

    def test_global_matching_returns_unique_cars_and_requests(self) -> None:
        world = MobilityWorld((12, 12), seed=71, fleet_size=8, horizon_steps=3, demand_scale=1.4)
        world.step(ActionPlan())
        pairs = global_batch_matching(world)
        car_ids = [pair.car_id for pair in pairs]
        person_ids = [pair.person_id for pair in pairs]
        self.assertEqual(len(car_ids), len(set(car_ids)))
        self.assertEqual(len(person_ids), len(set(person_ids)))


if __name__ == "__main__":
    unittest.main()

