"""
huggingface.py — Package and upload SafeSagaLLM AgentVerse scenarios to HuggingFace.

Usage:
    python dataset/agentverse/huggingface.py --task-type all --n 10 --upload

Target repo: julee0323/agentverse
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_SCENARIO_DIR = Path("experiments/agentverse/scenarios")
DEFAULT_OUTPUT_DIR   = Path("experiments/agentverse/hf_dataset")
DEFAULT_REPO_ID      = "julee0323/agentverse"


def _json_field(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _scenario_row(data: dict[str, Any]) -> dict[str, Any]:
    meta     = data.get("agentverse_metadata", {})
    agents   = data.get("agents", [])
    edges    = data.get("execution_edges", [])
    policy   = data.get("policy", {})
    keywords = policy.get("sensitive_keywords", [])

    return {
        "scenario_id":             data.get("scenario_id"),
        "task_id":                 meta.get("task_id"),
        "task_type":               meta.get("task_type"),
        "problem":                 meta.get("problem", ""),
        "domain":                  meta.get("domain", ""),
        "difficulty":              meta.get("difficulty", "medium"),
        "n_solvers":               meta.get("n_solvers", meta.get("n_experts", 0)),
        "has_executor":            meta.get("has_executor", False),
        "agent_count":             len(agents),
        "edge_count":              len(edges),
        "sensitive_keyword_count": len(keywords),
        "sensitive_keywords":      json.dumps(keywords),
        "agents":                  _json_field(agents),
        "execution_edges":         _json_field(edges),
        "policy":                  _json_field(policy),
        "agentverse_metadata":     _json_field(meta),
        "rego_output":             data.get("rego_output"),
        "tla_output_dir":          data.get("tla_output_dir"),
    }


def load_scenarios(scenario_dir: Path) -> list[dict[str, Any]]:
    paths = sorted(scenario_dir.glob("*.json"))
    if not paths:
        raise FileNotFoundError(f"No JSON scenario files in {scenario_dir}")
    rows = []
    for path in paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        rows.append(_scenario_row(data))
    return rows


def write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_summary(rows: list[dict[str, Any]], path: Path) -> None:
    fieldnames = [
        "scenario_id", "task_id", "task_type", "domain",
        "difficulty", "n_solvers", "has_executor",
        "agent_count", "edge_count", "sensitive_keyword_count",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fieldnames})


def write_readme(rows: list[dict[str, Any]], path: Path, repo_id: str) -> None:
    task_types = Counter(row.get("task_type") for row in rows)
    domains    = Counter(row.get("domain") for row in rows)
    avg_agents = sum(row.get("agent_count", 0) for row in rows) / max(len(rows), 1)
    avg_kw     = sum(row.get("sensitive_keyword_count", 0) for row in rows) / max(len(rows), 1)

    type_lines   = "\n".join(f"- {k}: {v}" for k, v in sorted(task_types.items()))
    domain_lines = "\n".join(f"- {k}: {v}" for k, v in sorted(domains.items()))

    text = f"""---
license: mit
language:
- en
pretty_name: SafeSagaLLM AgentVerse Scenarios
tags:
- agentverse
- multi-agent-systems
- policy-verification
- safesagallm
- code-generation
- collaborative-ai
---

# SafeSagaLLM AgentVerse Scenarios

This dataset converts [AgentVerse](https://arxiv.org/abs/2308.10379) benchmark
tasks into SafeSagaLLM multi-agent pipeline scenarios.

Repository: `{repo_id}`

## Multi-Agent DAG Structures

### Solver Pipeline (HumanEval, ToolUsing, PythonCalculator)

```
Role Assigner Agent
         ↓
Solver Agent × N  ← sensitive: solution_strategy, partial_solution
         ↓
Critic Agent      ← sensitive: internal_criticism
         ↓
Executor Agent    ← sensitive: execution_output, tool_output  [if applicable]
         ↓
Evaluator Agent
```

### Discussion Pipeline (Brainstorming, ResponseGen)

```
Role Assigner Agent
    /      |      \\
Expert1  Expert2  Expert3  ← sensitive: individual_opinion
         ↓
Moderator Agent
         ↓
Summarizer Agent
```

## Dataset Statistics

Total scenarios: {len(rows)}
Average agents per scenario: {avg_agents:.1f}
Average sensitive keywords: {avg_kw:.1f}

Task types:
{type_lines}

Domains:
{domain_lines}

## Sensitive Information Policy

OPA `P_cont` prevents sensitive inter-agent information from leaking:

| Keyword | Emitter | Authorized Receivers |
|---|---|---|
| `solution_strategy` | Solver Agents | Critic Agent only |
| `partial_solution` | Solver Agents | Critic Agent only |
| `internal_criticism` | Critic Agent | Executor / Evaluator |
| `execution_output` | Executor Agent | Evaluator Agent only |
| `tool_output` | Executor Agent | Evaluator Agent only |
| `individual_opinion` | Expert Agents | Moderator Agent only |

Key security property: **Solver_i must NOT see Solver_j's solution_strategy**
(prevents anchoring bias and preserves solution diversity).

## Intended Use

Evaluate whether SafeSagaLLM's dual-layer OPA policy (P_tran + P_cont) prevents
internal agent reasoning from leaking between agents that should work independently.
"""
    path.write_text(text, encoding="utf-8")


def package_and_upload(
    scenario_dir: Path,
    output_dir: Path,
    repo_id: str,
    upload: bool,
) -> list[dict[str, Any]]:
    rows = load_scenarios(scenario_dir)
    data_dir = output_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    jsonl_path = data_dir / "scenarios.jsonl"
    write_jsonl(rows, jsonl_path)
    write_summary(rows, data_dir / "summary.csv")
    write_readme(rows, output_dir / "README.md", repo_id)

    print(f"[package] {len(rows)} scenarios → {output_dir}")
    print(f"[package] JSONL:   {jsonl_path}")
    print(f"[package] summary: {data_dir / 'summary.csv'}")

    if upload:
        from huggingface_hub import HfApi
        api = HfApi()
        api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)
        api.upload_folder(
            repo_id=repo_id,
            repo_type="dataset",
            folder_path=str(output_dir),
            commit_message=f"Add SafeSagaLLM AgentVerse scenarios ({len(rows)} tasks)",
        )
        print(f"[upload] pushed to https://huggingface.co/datasets/{repo_id}")

    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task-type",
        choices=["humaneval", "tool", "brainstorming", "all", "existing"],
        default="all",
    )
    parser.add_argument("--n",            type=int,  default=10)
    parser.add_argument("--scenario-dir", type=Path, default=DEFAULT_SCENARIO_DIR)
    parser.add_argument("--output-dir",   type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--repo-id",      default=DEFAULT_REPO_ID)
    parser.add_argument("--upload",       action="store_true")
    args = parser.parse_args()

    if args.task_type != "existing":
        import sys
        sys.path.insert(0, str(Path(__file__).parent))
        from convert_to_safesagallm import (
            convert_task, write_outputs,
            generate_humaneval_tasks, generate_tool_tasks,
            generate_brainstorming_tasks, generate_all_tasks,
        )

        if args.task_type == "humaneval":
            tasks = generate_humaneval_tasks(args.n)
        elif args.task_type == "tool":
            tasks = generate_tool_tasks(args.n)
        elif args.task_type == "brainstorming":
            tasks = generate_brainstorming_tasks(args.n)
        else:
            tasks = generate_all_tasks(args.n)

        args.scenario_dir.mkdir(parents=True, exist_ok=True)
        for task in tasks:
            scenario = convert_task(task)
            write_outputs(scenario, args.scenario_dir)
        print(f"[convert] {len(tasks)} tasks written to {args.scenario_dir}")

    package_and_upload(args.scenario_dir, args.output_dir, args.repo_id, args.upload)


if __name__ == "__main__":
    main()
