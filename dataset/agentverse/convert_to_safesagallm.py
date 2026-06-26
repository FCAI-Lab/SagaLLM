"""
convert_to_safesagallm.py — AgentVerse task → SafeSagaLLM pipeline.

AgentVerse (arXiv:2308.10379) is a multi-agent collaboration framework with
8 task types: HumanEval, ToolUsing, Brainstorming, CommonGen, LogicGrid,
MGSM, PythonCalculator, ResponseGen.

Each task is converted into a role-based multi-agent DAG following
AgentVerse's standard collaboration pattern: Assign → Solve → Criticize/Evaluate.

DAG structure (varies by task type):

  [HumanEval / ToolUsing / General]

    Role Assigner Agent
    (decomposes task, assigns roles — sensitive: solution_strategy)
              ↓
    Solver Agent × N   (parallel)
    (each generates independent solution — sensitive: solution_strategy)
              ↓
    Critic Agent
    (evaluates proposals, provides feedback — sensitive: internal_criticism)
              ↓
    Evaluator Agent
    (compares the critic's selected/revised solution and gives final verdict)

  [Brainstorming / Discussion]

    Role Assigner Agent
        /      |      \\
    Expert1  Expert2  Expert3   (parallel, each sees only their own sub-task)
    Agent    Agent    Agent
    (sensitive: individual_opinion)
        \\      |      /
    Moderator Agent
    (synthesizes expert opinions)
              ↓
    Summarizer Agent

P_cont design:
  sensitive_keywords = ["solution_strategy", "internal_criticism",
                         "partial_solution", "individual_opinion"]

  Only downstream agents that legitimately need these fields may receive them:
    - Critic Agent may receive solution_strategy from Solver Agents.
    - Evaluator Agent may receive internal_criticism from Critic Agent.
    - Moderator Agent may receive individual_opinion from Expert Agents.
    - No cross-solver leakage: Solver_i must not see Solver_j's strategy.

Data source:
  AgentVerse tasks are defined in YAML configs in the GitHub repo.
  This module provides:
    - convert_task()         : converts a single task dict to SafeSagaLLM format
    - generate_humaneval()   : generates HumanEval-style tasks
    - generate_tool_tasks()  : generates ToolUsing tasks
    - generate_discussion()  : generates Brainstorming tasks
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


# ── Task type definitions ─────────────────────────────────────────────────────

TASK_TYPES = {
    "HumanEval":         "code generation and testing",
    "ToolUsing":         "tool-assisted problem solving",
    "Brainstorming":     "multi-expert collaborative discussion",
    "CommonGen":         "commonsense generation",
    "LogicGrid":         "logic puzzle solving",
    "MGSM":              "multilingual math problem solving",
    "PythonCalculator":  "Python-based computation",
    "ResponseGen":       "collaborative response generation",
}

# ── Sensitive keywords ────────────────────────────────────────────────────────

ALL_SENSITIVE = [
    "solution_strategy",   # solver's internal reasoning approach
    "internal_criticism",  # critic's private feedback before revision
    "individual_opinion",  # each expert's private opinion (pre-consensus)
    "partial_solution",    # intermediate draft not yet reviewed
]


# ── Helpers ───────────────────────────────────────────────────────────────────

class _NoAliasDumper(yaml.SafeDumper):
    def ignore_aliases(self, data: Any) -> bool:
        return True


def _agent_entry(
    name: str,
    task_description: str,
    depends_on: list[str],
    sensitive: bool = False,
    backstory: str | None = None,
) -> dict[str, Any]:
    return {
        "name":                name,
        "sensitive":           sensitive,
        "depends_on":          depends_on,
        "backstory":           backstory or f"You are the {name} in an AgentVerse SafeSagaLLM pipeline.",
        "task_description":    task_description,
        "task_expected_output": "Return only the information authorized for the next step.",
    }


# ── Task-type-specific agent builders ─────────────────────────────────────────

def _role_assigner_description(task_type: str, problem: str, n_solvers: int) -> str:
    return (
        f"You are the Role Assigner Agent for an AgentVerse {task_type} task.\n"
        f"Problem: {problem}\n\n"
        f"Your role:\n"
        f"  1. Analyze the problem and decompose it into {n_solvers} independent sub-tasks.\n"
        f"  2. Assign one sub-task to each Solver Agent.\n"
        f"  3. Specify success criteria for each sub-task.\n\n"
        f"IMPORTANT: Each solver works independently. Do NOT share solver assignments with\n"
        f"other solvers — cross-contamination degrades solution diversity.\n\n"
        f"Output format:\n"
        f"  task_decomposition: <list of {n_solvers} sub-tasks>\n"
        f"  assignment: <solver_name → sub-task>\n"
        f"  success_criteria: <measurable goal per sub-task>"
    )


def _solver_description(solver_idx: int, task_type: str, problem: str) -> str:
    return (
        f"You are Solver Agent {solver_idx} for an AgentVerse {task_type} task.\n"
        f"Problem: {problem}\n\n"
        f"You have received your sub-task assignment from the Role Assigner.\n"
        f"Work independently — do NOT share your reasoning with other solvers.\n\n"
        f"Your role:\n"
        f"  1. Devise your solution strategy (this is PRIVATE — mark as solution_strategy).\n"
        f"  2. Produce your partial solution.\n"
        f"  3. Report your result to the Critic Agent for review.\n\n"
        f"Output format:\n"
        f"  solution_strategy: <your private reasoning approach>\n"
        f"  partial_solution: <your draft solution>\n"
        f"  confidence: <HIGH | MEDIUM | LOW>\n"
        f"  notes: <anything the critic should know>"
    )


def _critic_description(task_type: str, problem: str, n_solvers: int) -> str:
    return (
        f"You are the Critic Agent for an AgentVerse {task_type} task.\n"
        f"Problem: {problem}\n\n"
        f"You have received proposals from {n_solvers} Solver Agents.\n\n"
        f"Your role:\n"
        f"  1. Evaluate each solver's partial_solution on correctness and quality.\n"
        f"  2. Provide specific improvement feedback (internal_criticism).\n"
        f"  3. Select the best approach or synthesize a combined solution.\n\n"
        f"WARNING: Your internal_criticism is sensitive — it should not be shared\n"
        f"between solver agents to avoid anchoring bias.\n\n"
        f"Output format:\n"
        f"  internal_criticism: <private critique per solver>\n"
        f"  best_approach: <which solver / hybrid approach to proceed with>\n"
        f"  revised_solution: <improved solution after critique>\n"
        f"  critique_summary: <public summary safe to share downstream>"
    )


def _evaluator_description(task_type: str, problem: str) -> str:
    return (
        f"You are the Evaluator Agent for an AgentVerse {task_type} task.\n"
        f"Problem: {problem}\n\n"
        f"You have received the Critic Agent's revised solution and critique summary.\n\n"
        f"Your role:\n"
        f"  1. Score the final solution on correctness (0–10).\n"
        f"  2. Check if the success criteria from Role Assigner are met.\n"
        f"  3. Produce the final verdict.\n\n"
        f"Output format:\n"
        f"  score: <0–10>\n"
        f"  criteria_met: <YES | PARTIAL | NO>\n"
        f"  verdict: ACCEPTED | REJECTED | RETRY\n"
        f"  final_answer: <the solution to return to the user>\n"
        f"  evaluation_notes: <brief explanation>"
    )


def _expert_description(expert_idx: int, domain: str, problem: str) -> str:
    return (
        f"You are Expert Agent {expert_idx} ({domain}) for an AgentVerse Brainstorming task.\n"
        f"Problem: {problem}\n\n"
        f"Your role:\n"
        f"  1. Generate ideas from your domain expertise ({domain}).\n"
        f"  2. Record your private reasoning as individual_opinion.\n"
        f"  3. Share only a structured summary with the Moderator.\n\n"
        f"WARNING: individual_opinion is sensitive — other Expert Agents must NOT\n"
        f"see it before the Moderator synthesizes (prevents groupthink).\n\n"
        f"Output format:\n"
        f"  individual_opinion: <your private reasoning and ideas>\n"
        f"  domain: {domain}\n"
        f"  key_insights: <3–5 key insights to share with Moderator>\n"
        f"  confidence: <HIGH | MEDIUM | LOW>"
    )


def _moderator_description(problem: str, n_experts: int) -> str:
    return (
        f"You are the Moderator Agent for an AgentVerse Brainstorming task.\n"
        f"Problem: {problem}\n\n"
        f"You have received key_insights from {n_experts} Expert Agents.\n"
        f"You also have access to their individual_opinion outputs.\n\n"
        f"Your role:\n"
        f"  1. Synthesize insights from all experts.\n"
        f"  2. Identify consensus and conflicts.\n"
        f"  3. Produce a unified position statement.\n\n"
        f"Output format:\n"
        f"  consensus_points: <list of agreed points>\n"
        f"  conflict_points: <list of disagreements>\n"
        f"  synthesized_position: <unified statement>\n"
        f"  recommended_next_step: <what to do next>"
    )


def _summarizer_description(problem: str) -> str:
    return (
        f"You are the Summarizer Agent for an AgentVerse Brainstorming task.\n"
        f"Problem: {problem}\n\n"
        f"You have received the synthesized position from the Moderator.\n\n"
        f"Your role:\n"
        f"  1. Produce a concise, user-facing summary.\n"
        f"  2. Highlight the top 3 actionable recommendations.\n\n"
        f"Output format:\n"
        f"  executive_summary: <2–3 sentence overview>\n"
        f"  top_recommendations: <numbered list of 3 recommendations>\n"
        f"  confidence: <HIGH | MEDIUM | LOW>"
    )


# ── Converter functions ───────────────────────────────────────────────────────

def _convert_solver_pipeline(task: dict[str, Any]) -> dict[str, Any]:
    """Convert HumanEval / ToolUsing / General tasks (Solver pipeline)."""
    task_id   = task.get("task_id", 0)
    task_type = task.get("task_type", "HumanEval")
    problem   = task.get("problem", "")
    n_solvers = task.get("n_solvers", 2)

    scenario_id = f"agentverse_{task_type.lower()}_task{task_id}"

    solver_names = [f"Solver Agent {i + 1}" for i in range(n_solvers)]

    sensitive_keywords = ["solution_strategy", "partial_solution", "internal_criticism"]

    # ── Agents ────────────────────────────────────────────────────────────────
    agents: list[dict[str, Any]] = [
        _agent_entry(
            "Role Assigner Agent",
            _role_assigner_description(task_type, problem, n_solvers),
            depends_on=[],
            sensitive=False,
        )
    ]
    for i in range(n_solvers):
        agents.append(
            _agent_entry(
                solver_names[i],
                _solver_description(i + 1, task_type, problem),
                depends_on=["Role Assigner Agent"],
                sensitive=True,  # emits solution_strategy
            )
        )
    agents.append(
        _agent_entry(
            "Critic Agent",
            _critic_description(task_type, problem, n_solvers),
            depends_on=solver_names,
            sensitive=True,  # emits internal_criticism
        )
    )
    agents.append(
        _agent_entry(
            "Evaluator Agent",
            _evaluator_description(task_type, problem),
            depends_on=["Critic Agent"],
            sensitive=False,
        )
    )

    # ── Execution edges ───────────────────────────────────────────────────────
    execution_edges: list[list[str]] = []
    for name in solver_names:
        execution_edges.append(["Role Assigner Agent", name])
        execution_edges.append([name, "Critic Agent"])
    execution_edges.append(["Critic Agent", "Evaluator Agent"])

    # ── P_tran ────────────────────────────────────────────────────────────────
    allowed_transfers: dict[str, list[str]] = {
        "Role Assigner Agent": solver_names,
    }
    for name in solver_names:
        # Solver → Critic only; Solver_i must NOT reach Solver_j
        allowed_transfers[name] = ["Critic Agent"]
    allowed_transfers["Critic Agent"] = ["Evaluator Agent"]
    allowed_transfers["Evaluator Agent"] = []

    # ── P_cont ────────────────────────────────────────────────────────────────
    # Critic receives solution_strategy from all solvers — authorized
    # Evaluator receives internal_criticism from the Critic — authorized
    keyword_permissions: dict[str, list[str]] = {
        "Critic Agent":    ["solution_strategy", "partial_solution"],
        "Evaluator Agent": ["internal_criticism"],
    }
    agent_output_keywords: dict[str, list[str]] = {
        name: ["solution_strategy", "partial_solution"] for name in solver_names
    }
    agent_output_keywords["Critic Agent"] = ["internal_criticism"]

    return {
        "scenario_id":          scenario_id,
        "agentverse_metadata":  {
            "task_id":          task_id,
            "task_type":        task_type,
            "problem":          problem,
            "n_solvers":        n_solvers,
            "has_executor":     False,
            "domain":           task.get("domain", ""),
            "difficulty":       task.get("difficulty", "medium"),
        },
        "agents":               agents,
        "execution_edges":      execution_edges,
        "policy": {
            "allowed_transfers":     allowed_transfers,
            "sensitive_keywords":    sensitive_keywords,
            "keyword_permissions":   keyword_permissions,
            "agent_output_keywords": agent_output_keywords,
        },
        "rego_output":    f"src/policies/agentverse_{scenario_id}.rego",
        "tla_output_dir": f"spec/agentverse/{scenario_id}",
    }


def _convert_discussion_pipeline(task: dict[str, Any]) -> dict[str, Any]:
    """Convert Brainstorming / Discussion tasks (Expert pipeline)."""
    task_id   = task.get("task_id", 0)
    task_type = task.get("task_type", "Brainstorming")
    problem   = task.get("problem", "")
    experts   = task.get("experts", [
        {"name": "Expert Agent 1", "domain": "Technical"},
        {"name": "Expert Agent 2", "domain": "Business"},
        {"name": "Expert Agent 3", "domain": "Ethics"},
    ])

    scenario_id  = f"agentverse_{task_type.lower()}_task{task_id}"
    expert_names = [e["name"] for e in experts]

    sensitive_keywords = ["individual_opinion"]

    # ── Agents ────────────────────────────────────────────────────────────────
    agents: list[dict[str, Any]] = [
        _agent_entry(
            "Role Assigner Agent",
            _role_assigner_description(task_type, problem, len(experts)),
            depends_on=[],
            sensitive=False,
        )
    ]
    for expert in experts:
        agents.append(
            _agent_entry(
                expert["name"],
                _expert_description(
                    expert_names.index(expert["name"]) + 1,
                    expert["domain"],
                    problem,
                ),
                depends_on=["Role Assigner Agent"],
                sensitive=True,   # emits individual_opinion
                backstory=(
                    f"You are a {expert['domain']} domain expert in an "
                    f"AgentVerse Brainstorming scenario."
                ),
            )
        )
    agents.append(
        _agent_entry(
            "Moderator Agent",
            _moderator_description(problem, len(experts)),
            depends_on=expert_names,
            sensitive=False,
        )
    )
    agents.append(
        _agent_entry(
            "Summarizer Agent",
            _summarizer_description(problem),
            depends_on=["Moderator Agent"],
            sensitive=False,
        )
    )

    # ── Execution edges ───────────────────────────────────────────────────────
    execution_edges: list[list[str]] = []
    for name in expert_names:
        execution_edges.append(["Role Assigner Agent", name])
        execution_edges.append([name, "Moderator Agent"])
    execution_edges.append(["Moderator Agent", "Summarizer Agent"])

    # ── P_tran ────────────────────────────────────────────────────────────────
    allowed_transfers: dict[str, list[str]] = {
        "Role Assigner Agent": expert_names,
    }
    for name in expert_names:
        # Expert → Moderator only; no cross-expert communication
        allowed_transfers[name] = ["Moderator Agent"]
    allowed_transfers["Moderator Agent"]   = ["Summarizer Agent"]
    allowed_transfers["Summarizer Agent"]  = []

    # ── P_cont ────────────────────────────────────────────────────────────────
    keyword_permissions: dict[str, list[str]] = {
        "Moderator Agent": ["individual_opinion"],
    }
    agent_output_keywords: dict[str, list[str]] = {
        name: ["individual_opinion"] for name in expert_names
    }

    return {
        "scenario_id":         scenario_id,
        "agentverse_metadata": {
            "task_id":   task_id,
            "task_type": task_type,
            "problem":   problem,
            "n_experts": len(experts),
            "experts":   experts,
            "domain":    task.get("domain", "general"),
            "difficulty": task.get("difficulty", "medium"),
        },
        "agents":          agents,
        "execution_edges": execution_edges,
        "policy": {
            "allowed_transfers":     allowed_transfers,
            "sensitive_keywords":    sensitive_keywords,
            "keyword_permissions":   keyword_permissions,
            "agent_output_keywords": agent_output_keywords,
        },
        "rego_output":    f"src/policies/agentverse_{scenario_id}.rego",
        "tla_output_dir": f"spec/agentverse/{scenario_id}",
    }


def convert_task(task: dict[str, Any]) -> dict[str, Any]:
    """Route task to the correct converter based on task_type."""
    task_type = task.get("task_type", "HumanEval")
    if task_type in ("Brainstorming", "ResponseGen"):
        return _convert_discussion_pipeline(task)
    return _convert_solver_pipeline(task)


# ── Synthetic task generators ─────────────────────────────────────────────────

def generate_humaneval_tasks(n: int = 5) -> list[dict[str, Any]]:
    """Generate HumanEval-style code generation tasks."""
    problems = [
        "Implement a function `is_palindrome(s: str) -> bool` that checks if a string is a palindrome.",
        "Write a function `fibonacci(n: int) -> list` returning the first n Fibonacci numbers.",
        "Implement `count_vowels(s: str) -> int` that counts vowels in a string.",
        "Write `merge_sorted(a: list, b: list) -> list` that merges two sorted lists.",
        "Implement `find_primes(n: int) -> list` returning all primes up to n using the Sieve of Eratosthenes.",
        "Write `flatten(lst: list) -> list` that recursively flattens a nested list.",
        "Implement `lru_cache(capacity: int)` as a class with get() and put() methods.",
        "Write `longest_common_subsequence(a: str, b: str) -> str` returning the LCS.",
    ]
    return [
        {
            "task_id":   i,
            "task_type": "HumanEval",
            "problem":   problems[i % len(problems)],
            "n_solvers": 2,
            "domain":    "software_engineering",
            "difficulty": "medium",
        }
        for i in range(n)
    ]


def generate_tool_tasks(n: int = 5) -> list[dict[str, Any]]:
    """Generate ToolUsing tasks."""
    problems = [
        "Search the web for the current price of Bitcoin and report it in USD.",
        "Use a calculator to solve: integral of x^2 from 0 to 10.",
        "Retrieve today's weather forecast for Seoul, South Korea.",
        "Fetch the latest 5 commits from the 'openai/openai-python' GitHub repository.",
        "Look up the population of Tokyo from Wikipedia and convert it to millions.",
    ]
    return [
        {
            "task_id":   100 + i,
            "task_type": "ToolUsing",
            "problem":   problems[i % len(problems)],
            "n_solvers": 2,
            "domain":    "information_retrieval",
            "difficulty": "easy",
        }
        for i in range(n)
    ]


def generate_brainstorming_tasks(n: int = 5) -> list[dict[str, Any]]:
    """Generate Brainstorming / discussion tasks."""
    problems_experts = [
        (
            "How should AI systems handle user privacy in multi-agent pipelines?",
            [
                {"name": "Expert Agent 1", "domain": "Privacy Law"},
                {"name": "Expert Agent 2", "domain": "AI Safety"},
                {"name": "Expert Agent 3", "domain": "Product Design"},
            ],
        ),
        (
            "Design a policy for content moderation in a multi-agent LLM system.",
            [
                {"name": "Expert Agent 1", "domain": "Content Policy"},
                {"name": "Expert Agent 2", "domain": "Machine Learning"},
                {"name": "Expert Agent 3", "domain": "Human Rights"},
            ],
        ),
        (
            "What are the best strategies for reducing energy consumption in data centers?",
            [
                {"name": "Expert Agent 1", "domain": "Environmental Science"},
                {"name": "Expert Agent 2", "domain": "Hardware Engineering"},
                {"name": "Expert Agent 3", "domain": "Economics"},
            ],
        ),
        (
            "How can multi-agent systems improve healthcare diagnostics?",
            [
                {"name": "Expert Agent 1", "domain": "Medicine"},
                {"name": "Expert Agent 2", "domain": "AI/ML"},
                {"name": "Expert Agent 3", "domain": "Medical Ethics"},
            ],
        ),
        (
            "What safety measures should autonomous vehicles adopt in urban environments?",
            [
                {"name": "Expert Agent 1", "domain": "Traffic Engineering"},
                {"name": "Expert Agent 2", "domain": "AI Safety"},
                {"name": "Expert Agent 3", "domain": "Urban Planning"},
            ],
        ),
    ]
    tasks = []
    for i in range(n):
        problem, experts = problems_experts[i % len(problems_experts)]
        tasks.append({
            "task_id":   200 + i,
            "task_type": "Brainstorming",
            "problem":   problem,
            "experts":   experts,
            "domain":    "general",
            "difficulty": "medium",
        })
    return tasks


def generate_all_tasks(n_per_type: int = 5) -> list[dict[str, Any]]:
    """Generate a representative mix of all AgentVerse task types."""
    return (
        generate_humaneval_tasks(n_per_type)
        + generate_tool_tasks(n_per_type)
        + generate_brainstorming_tasks(n_per_type)
    )


# ── File writer ───────────────────────────────────────────────────────────────

def write_outputs(
    scenario: dict[str, Any],
    scenario_dir: Path,
) -> tuple[Path, Path]:
    """Write pipeline YAML and JSON. Returns (yaml_path, json_path)."""
    import json as _json

    scenario_dir.mkdir(parents=True, exist_ok=True)
    sid = scenario["scenario_id"]

    yaml_path = scenario_dir / f"{sid}.yaml"
    yaml_path.write_text(
        yaml.dump(
            {k: v for k, v in scenario.items() if k != "agentverse_metadata"},
            Dumper=_NoAliasDumper,
            sort_keys=False,
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    json_path = scenario_dir / f"{sid}.json"
    json_path.write_text(
        _json.dumps(scenario, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return yaml_path, json_path


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert AgentVerse tasks into SafeSagaLLM multi-agent pipeline scenarios."
    )
    parser.add_argument(
        "--task-type",
        choices=["humaneval", "tool", "brainstorming", "all"],
        default="all",
        help="Which AgentVerse task type to generate.",
    )
    parser.add_argument("--n", type=int, default=5, help="Tasks per type.")
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("experiments/agentverse/scenarios"),
    )
    args = parser.parse_args()

    if args.task_type == "humaneval":
        tasks = generate_humaneval_tasks(args.n)
    elif args.task_type == "tool":
        tasks = generate_tool_tasks(args.n)
    elif args.task_type == "brainstorming":
        tasks = generate_brainstorming_tasks(args.n)
    else:
        tasks = generate_all_tasks(args.n)

    print(f"[convert] {len(tasks)} tasks → {args.output_dir}")
    for task in tasks:
        scenario = convert_task(task)
        yaml_path, json_path = write_outputs(scenario, args.output_dir)
        n_agents = len(scenario["agents"])
        n_kw     = len(scenario["policy"]["sensitive_keywords"])
        print(
            f"  task{task.get('task_id'):>4d}  type={task.get('task_type'):<15s}  "
            f"agents={n_agents}  sensitive_kw={n_kw}  → {yaml_path.name}"
        )


if __name__ == "__main__":
    main()
