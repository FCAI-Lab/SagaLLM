"""
coop_navi_adapter.py — Coop Navi environment adapter for SafeSagaLLM.

Wraps mpe2's simple_spread_v3 (PettingZoo parallel API) and produces
structured episode traces compatible with convert_to_safesagallm.py.

Observation layout for simple_spread_v3 (N=3 agents, 3 landmarks):
  [0:2]               — own velocity (vx, vy)
  [2:4]               — own absolute position (px, py)
  [4:4+2*n_landmarks] — landmark relative positions (lk_dx, lk_dy per landmark)
  [4+2*n_lm:]         — other agents' relative positions (2*(N-1) values)
  [...]               — inter-agent communications (2*(N-1) values, may be zeros)

Episode trace schema:
{
  "seed": int,
  "n_agents": int,
  "n_landmarks": int,
  "max_cycles": int,
  "agent_names": ["agent_0", "agent_1", "agent_2"],
  "initial_state": {
    "agent_0": {"position": [x, y], "velocity": [vx, vy]},
    "agent_1": {...},
    "agent_2": {...},
    "landmark_0": {"position": [x, y]},
    "landmark_1": {...},
    "landmark_2": {...},
  },
  "steps": [
    {
      "timestep": int,
      "positions": {"agent_0": [x, y], ...},
      "rewards":   {"agent_0": float, ...},
      "collisions": [["agent_0", "agent_1"], ...],
    },
    ...
  ],
  "total_collision_count": int,
  "total_reward": float,
}
"""

from __future__ import annotations

import numpy as np

# Matches the agent size defined in simple_spread scenario (simple_spread.py line ~17)
AGENT_SIZE = 0.15
COLLISION_THRESHOLD = 2 * AGENT_SIZE   # two agents collide when closer than this


def _parse_observation(
    obs: np.ndarray,
    n_agents: int,
    n_landmarks: int,
) -> dict:
    """Extract structured state from a raw simple_spread observation vector.

    Returns:
        velocity        — own (vx, vy)
        position        — own absolute (px, py)
        landmark_positions — absolute positions of all landmarks
        other_positions    — absolute positions of other agents
    """
    own_vel = [round(float(obs[0]), 3), round(float(obs[1]), 3)]
    own_pos = [round(float(obs[2]), 3), round(float(obs[3]), 3)]

    # Landmark positions are stored as relative offsets; convert to absolute
    landmark_abs: list[list[float]] = []
    for i in range(n_landmarks):
        dx = float(obs[4 + 2 * i])
        dy = float(obs[4 + 2 * i + 1])
        landmark_abs.append([
            round(own_pos[0] + dx, 3),
            round(own_pos[1] + dy, 3),
        ])

    # Other agent positions are also relative offsets
    offset = 4 + 2 * n_landmarks
    n_others = n_agents - 1
    other_abs: list[list[float]] = []
    for i in range(n_others):
        dx = float(obs[offset + 2 * i])
        dy = float(obs[offset + 2 * i + 1])
        other_abs.append([
            round(own_pos[0] + dx, 3),
            round(own_pos[1] + dy, 3),
        ])

    return {
        "velocity": own_vel,
        "position": own_pos,
        "landmark_positions": landmark_abs,
        "other_positions": other_abs,
    }


def _detect_collisions(
    parsed: dict[str, dict],
    agent_names: list[str],
) -> list[list[str]]:
    """Return list of colliding agent pairs (distance < COLLISION_THRESHOLD)."""
    collisions: list[list[str]] = []
    for i, a1 in enumerate(agent_names):
        for j, a2 in enumerate(agent_names):
            if j <= i:
                continue
            p1 = parsed[a1]["position"]
            p2 = parsed[a2]["position"]
            dist = float(np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2))
            if dist < COLLISION_THRESHOLD:
                collisions.append([a1, a2])
    return collisions


def run_episode(
    seed: int = 42,
    n_agents: int = 3,
    n_landmarks: int = 3,
    max_cycles: int = 50,
    policy: str = "random",
) -> dict:
    """Run a single Coop Navi episode and return a structured trace.

    Args:
        seed       — environment reset seed (controls landmark/agent init positions)
        n_agents   — number of cooperative agents
        n_landmarks — number of landmarks to cover (should equal n_agents)
        max_cycles — episode length
        policy     — "random" (default) or a callable dict[agent_name → action]

    Returns:
        Structured episode trace dict (see module docstring).
    """
    from mpe2 import simple_spread_v3

    env = simple_spread_v3.parallel_env(
        N=n_agents,
        max_cycles=max_cycles,
        continuous_actions=False,
        render_mode=None,
        local_ratio=0.5,
    )

    observations, _ = env.reset(seed=seed)
    agent_names: list[str] = list(observations.keys())

    # ── Initial state ──────────────────────────────────────────────────────────
    initial_parsed: dict[str, dict] = {
        name: _parse_observation(obs, n_agents, n_landmarks)
        for name, obs in observations.items()
    }

    # Agent positions + velocities
    initial_state: dict = {
        name: {
            "position": initial_parsed[name]["position"],
            "velocity": initial_parsed[name]["velocity"],
        }
        for name in agent_names
    }

    # Landmark absolute positions (inferred from agent_0's observation)
    ref = initial_parsed[agent_names[0]]
    for i, lpos in enumerate(ref["landmark_positions"]):
        initial_state[f"landmark_{i}"] = {"position": lpos}

    # ── Step loop ─────────────────────────────────────────────────────────────
    steps: list[dict] = []
    total_collisions = 0
    total_reward = 0.0

    for t in range(max_cycles):
        if not env.agents:
            break

        # Parse positions BEFORE stepping (so collision check is at current state)
        parsed_now: dict[str, dict] = {
            name: _parse_observation(observations[name], n_agents, n_landmarks)
            for name in env.agents
        }
        collisions = _detect_collisions(parsed_now, list(env.agents))
        total_collisions += len(collisions)

        # Sample random actions (policy=random) or use provided callable
        if policy == "random":
            actions = {agent: env.action_space(agent).sample() for agent in env.agents}
        else:
            actions = policy(env.agents, observations)

        observations, rewards, terminations, truncations, _ = env.step(actions)
        step_reward = sum(rewards.values())
        total_reward += step_reward

        steps.append({
            "timestep": t,
            "positions": {name: parsed_now[name]["position"] for name in parsed_now},
            "rewards": {k: round(float(v), 4) for k, v in rewards.items()},
            "collisions": collisions,
        })

    env.close()

    return {
        "seed": seed,
        "n_agents": n_agents,
        "n_landmarks": n_landmarks,
        "max_cycles": max_cycles,
        "agent_names": agent_names,
        "initial_state": initial_state,
        "steps": steps,
        "total_collision_count": total_collisions,
        "total_reward": round(total_reward, 4),
    }


def load_issta_seeds(seed_file: str, n_seeds: int | None = None) -> list[dict]:
    """Load Coop Navi seeds from ISSTA24_MAT init_state text files.

    ISSTA24_MAT saves episode state sequences as Python literal strings.
    Each line is one episode: list[timestep][agent_id] = [name, pos, vel].

    Returns a list of seed dicts: {"seed_index": i, "p_pos": [...], "initial_positions": [...]}.
    NOTE: ISSTA24_MAT uses simple_adv format; for pure Coop Navi (simple_spread)
    seeds are generated fresh via run_episode(seed=int).
    """
    from ast import literal_eval

    seeds: list[dict] = []
    with open(seed_file, "r") as f:
        for idx, line in enumerate(f):
            if n_seeds and idx >= n_seeds:
                break
            episode = literal_eval(line.strip())
            # Extract first timestep positions (t=0)
            t0 = episode[0]
            initial_positions = [[float(t0[j][1][0]), float(t0[j][1][1])] for j in range(len(t0))]
            seeds.append({
                "seed_index": idx,
                "initial_positions": initial_positions,
                "episode_length": len(episode),
            })
    return seeds
