from env import manual_dispatch_episode


tasks = []

_balanced = manual_dispatch_episode(
    seed=1,
    scenario="balanced",
    max_cars=3,
    max_requests=8,
    episode_seconds=240.0,
    max_dispatch_rounds=12,
)
_balanced.slug = "manual-dispatch-balanced"
_balanced.columns = {"scenario": "balanced", "difficulty": "easy", "control": "llm-dispatch"}
tasks.append(_balanced)

_surge = manual_dispatch_episode(
    seed=7,
    scenario="surge",
    max_cars=3,
    max_requests=10,
    episode_seconds=300.0,
    max_dispatch_rounds=16,
)
_surge.slug = "manual-dispatch-surge"
_surge.columns = {"scenario": "surge", "difficulty": "medium", "control": "llm-dispatch"}
tasks.append(_surge)
