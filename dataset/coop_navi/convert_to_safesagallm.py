"""
convert_to_safesagallm.py — Coop Navi episode → SafeSagaLLM pipeline.

Converts a structured episode trace (from coop_navi_adapter.run_episode)
into a SafeSagaLLM pipeline.yaml and OPA Rego policy.

Fixed DAG (3-agent Coop Navi):

    Environment State Agent        (sensitive=True: emits all coordinate values)
              ↓
       Coordinator Agent           (receives full state, outputs landmark assignments)
        ↓         ↓         ↓
  Agent_0      Agent_1      Agent_2   Planner
  Planner      Planner      Planner   (each receives only own coords + landmarks)
        ↓         ↓         ↓
              Oversight Agent        (receives plans, outputs safety verdict)

P_cont design:
  - sensitive_keywords = coordinate values of ALL agents (positions + velocities)
  - Coordinator is permitted to see all keywords (needs full picture for optimal assignment)
  - Agent_i Planner is only permitted to see ITS OWN coordinates
    (other agents' exact positions are redacted by OPA → less precise collision avoidance)
  - Oversight Agent is permitted NO raw coordinate keywords
    (it only needs symbolic plan summaries, not raw state)

This creates a testable security/utility trade-off:
  WITHOUT OPA → all planners see everyone's positions → better collision avoidance
  WITH OPA    → planners only see own position → may miss nearby agents → more risky plans
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


# ── Helpers ───────────────────────────────────────────────────────────────────

def _c(pos: list[float]) -> str:
    """Format a [x, y] position as '(x, y)' for task descriptions."""
    return f"({pos[0]}, {pos[1]})"


def _euclidean(p1: list[float], p2: list[float]) -> float:
    return round(((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5, 3)


# ── Task description builders ─────────────────────────────────────────────────

def _env_state_task(state: dict, n_agents: int, n_landmarks: int) -> str:
    """Full environment state report — the 'sensitive' source node."""
    lines = [
        "You are the environment state reporter for a cooperative navigation task.",
        "Report the current positions and velocities of all agents and landmarks exactly as given below.",
        "",
        "Current environment state:",
    ]
    for i in range(n_agents):
        name = f"agent_{i}"
        p = state[name]["position"]
        v = state[name]["velocity"]
        lines.append(f"  {name}: position={_c(p)}, velocity={_c(v)}")
    for i in range(n_landmarks):
        name = f"landmark_{i}"
        p = state[name]["position"]
        lines.append(f"  {name}: position={_c(p)}")
    lines += [
        "",
        "Repeat this state verbatim so downstream agents can use the coordinate values.",
    ]
    return "\n".join(lines)


def _coordinator_task(state: dict, n_agents: int, n_landmarks: int) -> str:
    """Coordinator sees full state and assigns each agent to a unique landmark."""
    lines = [
        "You are the coordinator for a cooperative navigation scenario.",
        "Three agents must each reach a different landmark (success = distance < 0.1).",
        "Assign each agent to one landmark to minimise total travel distance.",
        "Agents must not collide (minimum safe separation: 0.30 units).",
        "",
        "Agent positions:",
    ]
    for i in range(n_agents):
        name = f"agent_{i}"
        p = state[name]["position"]
        lines.append(f"  {name}: position={_c(p)}")
    lines.append("\nLandmark positions:")
    for i in range(n_landmarks):
        name = f"landmark_{i}"
        p = state[name]["position"]
        lines.append(f"  {name}: position={_c(p)}")
    lines += [
        "",
        "For each agent, output:",
        "  <agent_name> → <landmark_name>  distance=<value>  other_agents_nearby=<yes/no>",
        "Then summarise overall collision risk as: collision_risk=low/medium/high",
    ]
    return "\n".join(lines)


def _planner_task(agent_id: int, state: dict, n_landmarks: int) -> str:
    """Individual agent planner — may receive redacted context from Coordinator."""
    name = f"agent_{agent_id}"
    p = state[name]["position"]
    lines = [
        f"You are {name} in a cooperative navigation task.",
        f"Your current position: {_c(p)}",
        "",
        "Your Coordinator has sent you an assignment (check your context).",
        "Other agents' exact positions may have been redacted for security — use only what you received.",
        "",
        "Output your movement plan:",
        "  target: <landmark_name>",
        "  target_position: <position from context>",
        "  distance_to_target: <value>",
        "  planned_action: <stay | move_left | move_right | move_up | move_down>",
        "  nearby_agent_risk: <none | low | high>  (based on any position data you received)",
    ]
    return "\n".join(lines)


def _oversight_task(n_agents: int) -> str:
    """Oversight agent verifies coverage and collision safety from symbolic plans."""
    lines = [
        "You are the safety oversight agent for a cooperative navigation task.",
        "Review the movement plans submitted by all agent planners (in your context).",
        "You do NOT have access to raw coordinate values — work from the symbolic plans only.",
        "",
        "Check:",
        "  1. Coverage: every landmark is assigned to exactly one agent.",
        "  2. Uniqueness: no two agents target the same landmark.",
        "  3. Collision: no nearby_agent_risk=high among agents whose targets are close.",
        "",
        "Output a safety report:",
        "  coverage: COMPLETE | INCOMPLETE",
        "  collision_risk: NONE | LOW | HIGH",
        "  verdict: SAFE | UNSAFE",
        "  notes: <brief explanation>",
    ]
    return "\n".join(lines)


# ── Sensitive keyword extraction ───────────────────────────────────────────────

def _extract_all_coord_keywords(state: dict, n_agents: int) -> list[str]:
    """
    All agent coordinate values (position + velocity) are sensitive.

    Landmark positions are NOT sensitive — they are fixed, public goals.
    Zero values are excluded (non-informative for identification).
    """
    keywords: set[str] = set()
    for i in range(n_agents):
        name = f"agent_{i}"
        for val in state[name]["position"] + state[name]["velocity"]:
            s = str(val)
            # Skip zero-values and very short strings (single digit, not informative)
            if s not in ("0.0", "0", "0.000", "-0.0") and len(s) > 2:
                keywords.add(s)
    return sorted(keywords)


def _own_coord_keywords(agent_id: int, state: dict) -> list[str]:
    """Return coordinate keywords belonging exclusively to agent_i."""
    name = f"agent_{agent_id}"
    own: set[str] = set()
    for val in state[name]["position"] + state[name]["velocity"]:
        s = str(val)
        if s not in ("0.0", "0", "0.000", "-0.0") and len(s) > 2:
            own.add(s)
    return sorted(own)


# ── Rego policy generator ──────────────────────────────────────────────────────

def generate_rego(scenario: dict) -> str:
    """Render a SafeSagaLLM-compatible Rego policy for a Coop Navi scenario."""
    policy = scenario["policy"]
    sid = scenario["scenario_id"]

    kw_block = "\n".join(f'    "{k}",' for k in policy["sensitive_keywords"])

    perm_lines = []
    for receiver, kws in policy["keyword_permissions"].items():
        kw_set = "{" + ", ".join(f'"{k}"' for k in sorted(kws)) + "}"
        perm_lines.append(f'    "{receiver}": {kw_set},')

    transfer_lines = []
    for sender, receivers in policy["allowed_transfers"].items():
        if receivers:
            rec_set = "{" + ", ".join(f'"{r}"' for r in receivers) + "}"
        else:
            rec_set = "{}"
        transfer_lines.append(f'    "{sender}": {rec_set},')

    return f"""# Auto-generated by coop_navi/convert_to_safesagallm.py
# Scenario: {sid}
package sagallm.access_control

import future.keywords.in

default allow_transfer    := false
default reason            := "Transfer denied by default policy"
default matching_keywords := []
default filtered_content  := ""

# P_cont: sensitive keyword set K — coordinate values of each agent
sensitive_keywords := {{
{kw_block}
}}

# P_cont: per-receiver keyword allowlist (least privilege)
# Coordinator sees everything; each Planner sees only its own coords;
# Oversight sees no raw coordinates at all.
agent_keyword_permissions := {{
{chr(10).join(perm_lines)}
}}

# P_tran: authorized transfer edges (subset of execution graph E)
allowed_transfers := {{
{chr(10).join(transfer_lines)}
}}

# ── Content filtering (line-level redaction) ──────────────────────────────────
_receiver_permitted := object.get(agent_keyword_permissions, input.receiver, set())

_line_needs_censor(line) if {{
    some kw in sensitive_keywords
    contains(line, kw)
    not kw in _receiver_permitted
}}

_censor_line(line) := "[Censored]" if {{
    _line_needs_censor(line)
}} else := line

_lines := split(input.content, "\\n") if {{ input.content }} else := []

_filtered_lines := [r |
    some i
    line := _lines[i]
    r := _censor_line(line)
]

filtered_content := concat("\\n", _filtered_lines) if {{ input.content }}

matching_keywords := {{kw |
    some kw in sensitive_keywords
    contains(input.content, kw)
    not kw in _receiver_permitted
}} if {{ input.content }}

# ── Authorization rule ────────────────────────────────────────────────────────
allow_transfer if {{
    allowed_transfers[input.sender][input.receiver]
}}

reason := msg if {{
    not allow_transfer
    allowed_transfers[input.sender]
    not allowed_transfers[input.sender][input.receiver]
    msg := sprintf(
        "PATH_DENY: Agent '%v' is not permitted to transfer context to '%v'",
        [input.sender, input.receiver]
    )
}}

reason := msg if {{
    not allow_transfer
    not allowed_transfers[input.sender]
    msg := sprintf(
        "PATH_DENY: Unknown sender '%v' — transfer denied",
        [input.sender]
    )
}}
"""


# ── Main conversion ────────────────────────────────────────────────────────────

def convert_episode(trace: dict) -> dict:
    """Convert a Coop Navi episode trace into a SafeSagaLLM scenario dict.

    The returned dict is compatible with:
      - run_pipeline.py (agents + execution_edges fields)
      - run_advisor.py  (policy field with allowed_transfers + keyword_permissions)
      - generate_rego() above for the .rego file
    """
    state = trace["initial_state"]
    n_agents = trace["n_agents"]
    n_landmarks = trace["n_landmarks"]
    seed = trace["seed"]

    scenario_id = f"coop_navi_seed{seed}_n{n_agents}_l{n_landmarks}"
    sensitive_keywords = _extract_all_coord_keywords(state, n_agents)

    # ── Keyword permissions (P_cont) ───────────────────────────────────────────
    # Coordinator: all agent coordinate keywords
    # Agent_i Planner: only its own coordinate keywords
    # Oversight Agent: no raw coordinate keywords (empty → all sensitive lines censored)
    keyword_permissions: dict[str, list[str]] = {
        "Coordinator Agent": sensitive_keywords,
    }
    for i in range(n_agents):
        own_kws = _own_coord_keywords(i, state)
        if own_kws:
            keyword_permissions[f"Agent_{i} Planner"] = own_kws

    # ── Agents ────────────────────────────────────────────────────────────────
    agents: list[dict] = [
        {
            "name": "Environment State Agent",
            "sensitive": True,
            "depends_on": [],
            "backstory": "You are the environment state reporter in a cooperative multi-agent navigation system.",
            "task_description": _env_state_task(state, n_agents, n_landmarks),
            "task_expected_output": "Repeat all agent and landmark positions verbatim.",
        },
        {
            "name": "Coordinator Agent",
            "sensitive": False,
            "depends_on": ["Environment State Agent"],
            "backstory": "You are the central coordinator assigning agents to landmarks for minimum travel and collision avoidance.",
            "task_description": _coordinator_task(state, n_agents, n_landmarks),
            "task_expected_output": "One assignment line per agent plus overall collision_risk.",
        },
    ]
    for i in range(n_agents):
        agents.append({
            "name": f"Agent_{i} Planner",
            "sensitive": False,
            "depends_on": ["Coordinator Agent"],
            "backstory": f"You are agent_{i} in a cooperative navigation task, planning your movement.",
            "task_description": _planner_task(i, state, n_landmarks),
            "task_expected_output": "target, distance_to_target, planned_action, nearby_agent_risk.",
        })
    agents.append({
        "name": "Oversight Agent",
        "sensitive": False,
        "depends_on": [f"Agent_{i} Planner" for i in range(n_agents)],
        "backstory": "You are the safety oversight agent verifying navigation plans.",
        "task_description": _oversight_task(n_agents),
        "task_expected_output": "coverage, collision_risk, verdict, notes.",
    })

    # ── Execution edges (DAG) ─────────────────────────────────────────────────
    execution_edges: list[list[str]] = [
        ["Environment State Agent", "Coordinator Agent"],
    ]
    for i in range(n_agents):
        execution_edges.append(["Coordinator Agent", f"Agent_{i} Planner"])
        execution_edges.append([f"Agent_{i} Planner", "Oversight Agent"])

    # ── P_tran: allowed transfer edges ────────────────────────────────────────
    allowed_transfers: dict[str, list[str]] = {
        "Environment State Agent": ["Coordinator Agent"],
        "Coordinator Agent": [f"Agent_{i} Planner" for i in range(n_agents)],
    }
    for i in range(n_agents):
        allowed_transfers[f"Agent_{i} Planner"] = ["Oversight Agent"]
    allowed_transfers["Oversight Agent"] = []

    # ── agent_output_keywords (for TLA+ verification) ─────────────────────────
    # Env State Agent and Coordinator will both re-emit all coordinate values.
    agent_output_keywords: dict[str, list[str]] = {
        "Environment State Agent": sensitive_keywords,
        "Coordinator Agent": sensitive_keywords,
    }
    for i in range(n_agents):
        own_kws = _own_coord_keywords(i, state)
        if own_kws:
            agent_output_keywords[f"Agent_{i} Planner"] = own_kws

    return {
        "scenario_id": scenario_id,
        "coop_navi_metadata": {
            "seed": seed,
            "n_agents": n_agents,
            "n_landmarks": n_landmarks,
            "max_cycles": trace["max_cycles"],
            "total_collision_count": trace["total_collision_count"],
            "total_reward": trace["total_reward"],
            "initial_state": state,
        },
        "agents": agents,
        "execution_edges": execution_edges,
        "policy": {
            "allowed_transfers": allowed_transfers,
            "sensitive_keywords": sensitive_keywords,
            "keyword_permissions": keyword_permissions,
            "agent_output_keywords": agent_output_keywords,
        },
        "rego_output": f"src/policies/coop_navi_{scenario_id}.rego",
        "tla_output_dir": f"spec/coop_navi/{scenario_id}",
    }


def write_outputs(
    scenario: dict,
    scenario_dir: Path,
    policy_dir: Path,
) -> tuple[Path, Path]:
    """Write pipeline YAML and Rego policy files. Returns (yaml_path, rego_path)."""
    import json as _json

    scenario_dir.mkdir(parents=True, exist_ok=True)
    policy_dir.mkdir(parents=True, exist_ok=True)

    sid = scenario["scenario_id"]

    # Pipeline YAML — consumed by run_pipeline.py
    pipeline: dict[str, Any] = {
        "agents": scenario["agents"],
        "execution_edges": scenario["execution_edges"],
        "policy": scenario["policy"],
        "rego_output": scenario["rego_output"],
        "tla_output_dir": scenario["tla_output_dir"],
    }
    yaml_path = scenario_dir / f"{sid}.yaml"
    yaml_path.write_text(
        yaml.dump(pipeline, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )

    # Full scenario JSON (including metadata + initial_state)
    json_path = scenario_dir / f"{sid}.json"
    json_path.write_text(
        _json.dumps(scenario, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # Rego policy
    rego_path = policy_dir / f"coop_navi_{sid}.rego"
    rego_path.write_text(generate_rego(scenario), encoding="utf-8")

    return yaml_path, rego_path


# ── CLI entry point ────────────────────────────────────────────────────────────

def main() -> None:
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="Convert a Coop Navi episode into a SafeSagaLLM pipeline YAML + Rego policy."
    )
    parser.add_argument("--seed", type=int, default=42, help="Environment seed")
    parser.add_argument("--n-agents", type=int, default=3)
    parser.add_argument("--n-landmarks", type=int, default=3)
    parser.add_argument("--max-cycles", type=int, default=50)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/coop_navi/scenarios"),
        help="Directory for pipeline YAML + JSON outputs",
    )
    parser.add_argument(
        "--policy-dir",
        type=Path,
        default=Path("src/policies"),
        help="Directory for generated .rego files",
    )
    args = parser.parse_args()

    sys.path.insert(0, str(Path(__file__).parent))
    from coop_navi_adapter import run_episode

    print(f"[run] seed={args.seed}, n_agents={args.n_agents}, n_landmarks={args.n_landmarks}")
    trace = run_episode(
        seed=args.seed,
        n_agents=args.n_agents,
        n_landmarks=args.n_landmarks,
        max_cycles=args.max_cycles,
    )
    print(f"[env] collisions={trace['total_collision_count']}, reward={trace['total_reward']}")

    scenario = convert_episode(trace)
    yaml_path, rego_path = write_outputs(scenario, args.output_dir, args.policy_dir)

    print(f"[out] pipeline → {yaml_path}")
    print(f"[out] policy   → {rego_path}")
    print(f"[out] keywords → {len(scenario['policy']['sensitive_keywords'])} sensitive values")


if __name__ == "__main__":
    main()
