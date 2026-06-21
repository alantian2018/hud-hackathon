from __future__ import annotations

import asyncio
from pathlib import Path
import socket
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hud import Environment
from hud.capabilities import Capability

from hud_mobility.tools import close_episode, create_episode, get_episode, build_mcp_server


env = Environment(name="mobility-orchestrator", version="0.1.0")

_mcp_task: asyncio.Task | None = None
_mcp_port: int | None = None


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


async def _wait_until_listening(port: int, timeout: float = 10.0) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    while True:
        try:
            reader, writer = await asyncio.open_connection("127.0.0.1", port)
            writer.close()
            await writer.wait_closed()
            return
        except OSError:
            if asyncio.get_running_loop().time() >= deadline:
                raise TimeoutError(f"MCP server did not bind on port {port}")
            await asyncio.sleep(0.1)


@env.initialize
async def _start_tools() -> None:
    global _mcp_task, _mcp_port
    if _mcp_task is None:
        _mcp_port = _free_port()
        server = build_mcp_server()
        _mcp_task = asyncio.create_task(
            server.run_async(transport="http", host="127.0.0.1", port=_mcp_port)
        )
        await _wait_until_listening(_mcp_port)
    env.add_capability(Capability.mcp(name="mobility_tools", url=f"http://127.0.0.1:{_mcp_port}/mcp"))


@env.shutdown
async def _stop_tools() -> None:
    global _mcp_task, _mcp_port
    if _mcp_task is not None:
        _mcp_task.cancel()
        try:
            await _mcp_task
        except asyncio.CancelledError:
            pass
        _mcp_task = None
        _mcp_port = None


@env.template(description="Control a mobility fleet with an LLM orchestrator using MCP tools.")
async def optimize_mobility(
    seed: int = 7,
    fleet_size: int = 20,
    start_minute: int = 8 * 60,
    horizon_steps: int = 8,
    event_id: str | None = None,
    demand_scale: float = 1.0,
    rows: int = 14,
    cols: int = 14,
):
    episode_id = create_episode(
        seed=seed,
        fleet_size=fleet_size,
        start_minute=start_minute,
        horizon_steps=horizon_steps,
        event_id=event_id,
        demand_scale=demand_scale,
        rows=rows,
        cols=cols,
    )
    prompt = f"""
You are the central LLM orchestrator for a fleet of AI dispatch specialists.

Your goal is to maximize the absolute simulator reward for episode `{episode_id}`.
The reward combines revenue capture, served demand, low wait time, productive fleet
utilization, future supply alignment, low cancellations, low deadhead movement, and
valid actions. Do not compare against or ask for any greedy baseline.

Use the `mobility_tools` MCP capability. The most reliable path is:
1. Call `run_recommended_episode(episode_id)` once. It delegates the full horizon
   to non-greedy matching and repositioning specialist planners.
2. Call `submit_episode(episode_id)` and finish with a concise JSON summary.

If you want step-level control, call `execute_recommended_step(episode_id)` once
per simulator step, repeat until the tool says `done` is true, then submit.

If you need to inspect or edit a specialist plan, call `observe_state`,
`forecast_hotspots`, `propose_matching`, `propose_repositioning`, `propose_full_plan`,
and `critique_action_plan`; then call `step_world(episode_id, plan_json=...)` with
a structured JSON object (or a JSON string only if your tool interface requires strings):
   {{
     "assignments": [{{"car_id": "car-0", "person_id": "person-..."}}],
     "repositions": [{{"car_id": "car-1", "target": [row, col]}}],
     "holds": ["car-2"],
     "rationale": "short tactical reason"
   }}
Episode settings: seed={seed}, fleet_size={fleet_size}, start_minute={start_minute},
horizon_steps={horizon_steps}, event_id={event_id}, demand_scale={demand_scale},
grid={rows}x{cols}.
"""
    _answer = yield prompt
    try:
        result = get_episode(episode_id).reward()
        reward = float(result["reward"])
    finally:
        close_episode(episode_id)
    yield reward
