#!/usr/bin/env python3
"""
Package and optionally upload SafeSagaLLM AgentDojo scenarios to Hugging Face.

Default input:
    experiments/agentdojo/safesagallm_scenarios/*.json

Default package:
    experiments/agentdojo/hf_dataset/
        README.md
        data/scenarios.jsonl
        data/summary.csv

Upload:
    python dataset/agentdojo/huggingface.py --upload --repo-id julee0323/agentdojo
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_INPUT_DIR = Path("experiments/agentdojo/safesagallm_scenarios")
DEFAULT_OUTPUT_DIR = Path("experiments/agentdojo/hf_dataset")

JSON_ENCODED_FIELDS = [
    "agents",
    "execution_edges",
    "dependency_model",
    "policy",
    "agentdojo_transfer_events",
    "agentdojo_inferred_policy_surfaces",
    "agentdojo_attack",
    "agentdojo_attack_labels",
    "agentdojo_attack_sensitive_keywords",
    "state_delta_labels",
]


def _json_field(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _scenario_row(data: dict[str, Any]) -> dict[str, Any]:
    metadata = data.get("agentdojo_metadata", {})
    agents = data.get("agents", [])
    edges = data.get("execution_edges", [])
    policy = data.get("policy", {})
    sensitive_keywords = policy.get("sensitive_keywords", [])

    return {
        "scenario_id": data.get("scenario_id"),
        "suite": metadata.get("suite_name"),
        "pipeline_name": metadata.get("pipeline_name"),
        "user_task_id": metadata.get("user_task_id"),
        "injection_task_id": metadata.get("injection_task_id"),
        "attack_type": metadata.get("attack_type"),
        "benchmark_version": metadata.get("benchmark_version"),
        "utility": metadata.get("utility"),
        "security": metadata.get("security"),
        "conversion": metadata.get("conversion"),
        "agent_count": len(agents),
        "edge_count": len(edges),
        "sensitive_keyword_count": len(sensitive_keywords),
        # Keep complex, dynamic-key structures JSON-encoded so Hugging Face
        # `datasets` can infer a stable schema across suites and attack types.
        "agents": _json_field(agents),
        "execution_edges": _json_field(edges),
        "dependency_model": _json_field(data.get("dependency_model", {})),
        "policy": _json_field(policy),
        "agentdojo_transfer_events": _json_field(data.get("agentdojo_transfer_events", [])),
        "agentdojo_inferred_policy_surfaces": _json_field(data.get("agentdojo_inferred_policy_surfaces", [])),
        "agentdojo_attack": _json_field(data.get("agentdojo_attack", {})),
        "agentdojo_attack_labels": _json_field(data.get("agentdojo_attack_labels", {})),
        "agentdojo_attack_sensitive_keywords": _json_field(data.get("agentdojo_attack_sensitive_keywords", [])),
        "state_delta_labels": _json_field(data.get("state_delta_labels", {})),
        "rego_output": data.get("rego_output"),
        "tla_output_dir": data.get("tla_output_dir"),
    }


def load_scenarios(input_dir: Path, attack_type: str | None = None) -> list[dict[str, Any]]:
    paths = sorted(input_dir.glob("*.json"))
    if not paths:
        raise FileNotFoundError(f"No scenario JSON files found in {input_dir}")
    rows = []
    for path in paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        row = _scenario_row(data)
        if attack_type and row.get("attack_type") != attack_type:
            continue
        rows.append(row)
    if not rows:
        label = f" with attack_type={attack_type}" if attack_type else ""
        raise FileNotFoundError(f"No scenario JSON files found in {input_dir}{label}")
    return rows


def load_packaged_rows(data_dir: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(data_dir.glob("*.jsonl")):
        with path.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
    return rows


def write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_summary(rows: list[dict[str, Any]], path: Path) -> None:
    fieldnames = [
        "scenario_id",
        "suite",
        "user_task_id",
        "injection_task_id",
        "attack_type",
        "agent_count",
        "edge_count",
        "sensitive_keyword_count",
        "utility",
        "security",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fieldnames})


def write_readme(rows: list[dict[str, Any]], path: Path, repo_id: str, data_files: list[str]) -> None:
    suites = Counter(row.get("suite") or "unknown" for row in rows)
    attacks = Counter(row.get("attack_type") or "unknown" for row in rows)
    agent_counts = Counter(row.get("agent_count") for row in rows)

    suite_lines = "\n".join(f"- {key}: {value}" for key, value in sorted(suites.items()))
    attack_lines = "\n".join(f"- {key}: {value}" for key, value in sorted(attacks.items()))
    agent_lines = "\n".join(f"- {key} agents: {value}" for key, value in sorted(agent_counts.items()))
    file_lines = "\n".join(f"- `data/{name}`" for name in sorted(data_files))

    text = f"""---
license: mit
language:
- en
pretty_name: SafeSagaLLM AgentDojo Scenarios
tags:
- agentdojo
- multi-agent-systems
- policy-verification
- safesagallm
---

# SafeSagaLLM AgentDojo Scenarios

This dataset contains AgentDojo-derived scenarios converted into SafeSagaLLM's
sender-receiver-content policy evaluation format.

Repository: `{repo_id}`

## Files

- Scenario JSONL files:
{file_lines}
- `data/summary.csv`: compact table for filtering and experiment planning.

## Scenario Format

Each scenario models AgentDojo tool use as a domain-level DAG:

```text
User Task Agent
-> LLM Planning Agent
-> Tool Data Agent
-> LLM Decision Agent
-> Tool Action Agent / Final Answer Agent
```

`Data Agent` nodes model read-side tool results entering LLM context. `Action
Agent` nodes model side-effecting tool calls such as sending email, deleting
files, booking travel, or transferring money.

Complex columns such as `agents`, `policy`, `execution_edges`,
`agentdojo_attack_labels`, and `state_delta_labels` are stored as JSON-encoded
strings. This keeps the Hugging Face `datasets` schema stable across suites,
because these fields contain dynamic agent names and tool-specific argument
keys.

## Current Package Summary

Total scenarios: {len(rows)}

Suites:
{suite_lines}

Attack types:
{attack_lines}

Agent count distribution:
{agent_lines}

## Intended Use

The scenarios are intended for evaluating whether SafeSagaLLM preserves data
isolation, content filtering, atomic termination, and compensation completeness
when AgentDojo prompt-injection scenarios are represented as multi-agent
workflows.
"""
    path.write_text(text, encoding="utf-8")


def package_dataset(
    input_dir: Path,
    output_dir: Path,
    repo_id: str,
    attack_type: str | None,
    data_file: str,
    replace: bool,
) -> tuple[Path, list[dict[str, Any]]]:
    rows = load_scenarios(input_dir, attack_type=attack_type)
    data_dir = output_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    if replace:
        for path in data_dir.glob("*.jsonl"):
            path.unlink()
    write_jsonl(rows, data_dir / data_file)

    all_rows = load_packaged_rows(data_dir)
    data_files = [path.name for path in data_dir.glob("*.jsonl")]
    write_summary(all_rows, data_dir / "summary.csv")
    write_readme(all_rows, output_dir / "README.md", repo_id, data_files)
    return output_dir, all_rows


def upload_dataset(output_dir: Path, repo_id: str) -> None:
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)
    api.upload_folder(
        repo_id=repo_id,
        repo_type="dataset",
        folder_path=str(output_dir),
        commit_message="Add SafeSagaLLM AgentDojo pilot scenarios",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--repo-id", default="julee0323/agentdojo")
    parser.add_argument("--attack-type", help="Only package scenarios with this attack type.")
    parser.add_argument("--data-file", default="scenarios.jsonl", help="JSONL file name under data/.")
    parser.add_argument("--replace", action="store_true", help="Delete existing data/*.jsonl before packaging.")
    parser.add_argument("--upload", action="store_true")
    args = parser.parse_args()

    output_dir, rows = package_dataset(
        args.input_dir,
        args.output_dir,
        args.repo_id,
        args.attack_type,
        args.data_file,
        args.replace,
    )
    print(f"[package] wrote package summary for {len(rows)} scenarios to {output_dir}")
    print(f"[package] JSONL: {output_dir / 'data' / args.data_file}")
    print(f"[package] CSV:   {output_dir / 'data' / 'summary.csv'}")

    if args.upload:
        upload_dataset(output_dir, args.repo_id)
        print(f"[upload] pushed to https://huggingface.co/datasets/{args.repo_id}")


if __name__ == "__main__":
    main()
