"""
huggingface.py — Package and upload SafeSagaLLM WebArena scenarios to HuggingFace.

Usage:
    # Synthetic tasks (offline, no WebArena install needed)
    python dataset/webarena/huggingface.py --source synthetic --n 50 --upload

    # From local WebArena test.json
    python dataset/webarena/huggingface.py --source json --input /path/to/test.json --upload

    # From HuggingFace WebArena dataset
    python dataset/webarena/huggingface.py --source hf --upload

Target repo: julee0323/webarena
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_SCENARIO_DIR = Path("experiments/webarena/scenarios")
DEFAULT_OUTPUT_DIR   = Path("experiments/webarena/hf_dataset")
DEFAULT_REPO_ID      = "julee0323/webarena"

JSON_ENCODED_FIELDS = [
    "agents",
    "execution_edges",
    "policy",
    "webarena_metadata",
]


def _json_field(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _scenario_row(data: dict[str, Any]) -> dict[str, Any]:
    meta     = data.get("webarena_metadata", {})
    agents   = data.get("agents", [])
    edges    = data.get("execution_edges", [])
    policy   = data.get("policy", {})
    keywords = policy.get("sensitive_keywords", [])

    return {
        "scenario_id":             data.get("scenario_id"),
        "task_id":                 meta.get("task_id"),
        "task_type":               meta.get("task_type"),
        "sites":                   json.dumps(meta.get("sites", [])),
        "require_login":           meta.get("require_login", False),
        "intent":                  meta.get("intent", ""),
        "agent_count":             len(agents),
        "edge_count":              len(edges),
        "sensitive_keyword_count": len(keywords),
        "sensitive_keywords":      json.dumps(keywords),
        "agents":                  _json_field(agents),
        "execution_edges":         _json_field(edges),
        "policy":                  _json_field(policy),
        "webarena_metadata":       _json_field(meta),
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
        "scenario_id", "task_id", "task_type", "sites",
        "require_login", "agent_count", "edge_count",
        "sensitive_keyword_count",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fieldnames})


def write_readme(rows: list[dict[str, Any]], path: Path, repo_id: str) -> None:
    task_types  = Counter(row.get("task_type") for row in rows)
    sites_all: Counter = Counter()
    for row in rows:
        for site in json.loads(row.get("sites", "[]")):
            sites_all[site] += 1
    login_count = sum(1 for row in rows if row.get("require_login"))
    avg_agents  = sum(row.get("agent_count", 0) for row in rows) / max(len(rows), 1)
    avg_kw      = sum(row.get("sensitive_keyword_count", 0) for row in rows) / max(len(rows), 1)

    type_lines  = "\n".join(f"- {k}: {v}" for k, v in sorted(task_types.items()))
    site_lines  = "\n".join(f"- {k}: {v}" for k, v in sorted(sites_all.items()))

    text = f"""---
license: mit
language:
- en
pretty_name: SafeSagaLLM WebArena Scenarios
tags:
- webarena
- multi-agent-systems
- policy-verification
- safesagallm
- web-navigation
---

# SafeSagaLLM WebArena Scenarios

This dataset converts [WebArena](https://arxiv.org/abs/2307.13854) web navigation
tasks into SafeSagaLLM multi-agent pipeline scenarios.

Repository: `{repo_id}`

## Multi-Agent DAG Structure

Each WebArena task is modeled as a 4–7 agent pipeline:

```
Task Interpreter Agent
         ↓
[Site Browser Agents]  (reddit / shopping / gitlab / wikipedia / map)
         ↓
Observation Aggregator Agent  ← sensitive: captures raw page content
         ↓
Action Decision Agent
         ↓
Answer Verifier Agent
```

## Dataset Statistics

Total scenarios: {len(rows)}
Tasks requiring login: {login_count}
Average agents per scenario: {avg_agents:.1f}
Average sensitive keywords per scenario: {avg_kw:.1f}

Task types:
{type_lines}

Site distribution:
{site_lines}

## Sensitive Information Policy

OPA `P_cont` enforces that sensitive web content (credentials, PII, financial data)
is filtered based on site type:

- **reddit**: private_message, inbox_content, user_session
- **shopping**: credit_card, purchase_history, shipping_address
- **shopping_admin**: admin_credential, user_database, transaction_log
- **gitlab**: access_token, api_key, repository_secret
- **map**: location_history, saved_places

## Field Descriptions

Complex columns (`agents`, `policy`, `execution_edges`, `webarena_metadata`)
are JSON-encoded strings for stable schema across tasks with different site sets.

## Intended Use

Evaluate whether SafeSagaLLM's dual-layer OPA policy (P_tran + P_cont) prevents
sensitive web data from leaking across unauthorized agents in a multi-step
web navigation pipeline.
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
            commit_message=f"Add SafeSagaLLM WebArena scenarios ({len(rows)} tasks)",
        )
        print(f"[upload] pushed to https://huggingface.co/datasets/{repo_id}")

    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", choices=["existing", "synthetic", "json", "hf"],
        default="existing",
        help=(
            "existing: package already-converted scenarios from --scenario-dir; "
            "synthetic/json/hf: generate + convert then package."
        ),
    )
    parser.add_argument("--input",        type=Path, help="Local test.json (--source json)")
    parser.add_argument("--n",            type=int,  default=None)
    parser.add_argument("--scenario-dir", type=Path, default=DEFAULT_SCENARIO_DIR)
    parser.add_argument("--output-dir",   type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--repo-id",      default=DEFAULT_REPO_ID)
    parser.add_argument("--upload",       action="store_true")
    args = parser.parse_args()

    # Auto-generate scenarios if requested
    if args.source != "existing":
        import sys
        sys.path.insert(0, str(Path(__file__).parent))
        from convert_to_safesagallm import (
            convert_task, generate_synthetic_tasks,
            load_from_json, load_from_huggingface, write_outputs,
        )

        if args.source == "synthetic":
            tasks = generate_synthetic_tasks(args.n or 50)
        elif args.source == "json":
            if not args.input:
                parser.error("--source json requires --input")
            tasks = load_from_json(args.input, args.n)
        else:
            tasks = load_from_huggingface(args.n)

        args.scenario_dir.mkdir(parents=True, exist_ok=True)
        for task in tasks:
            scenario = convert_task(task)
            write_outputs(scenario, args.scenario_dir)
        print(f"[convert] {len(tasks)} tasks written to {args.scenario_dir}")

    package_and_upload(args.scenario_dir, args.output_dir, args.repo_id, args.upload)


if __name__ == "__main__":
    main()
