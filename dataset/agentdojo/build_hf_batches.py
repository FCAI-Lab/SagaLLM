#!/usr/bin/env python3
"""
Build SafeSagaLLM AgentDojo HF batches without writing per-scenario YAML/JSON.

This script converts AgentDojo registry scenarios directly into JSONL files:

    experiments/agentdojo/hf_dataset/data/{suite}_{attack}.jsonl

It is intended for the full automated dataset build after the converter logic
has been validated on small pilot batches.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from agentdojo.task_suite.load_suites import get_suite

from convert_to_safesagallm import (
    _add_attack_execution_surfaces,
    _attack_payload,
    _merge_attack_sensitive_policy,
    _message_text,
    _registry_tool_messages,
    _state_delta_oracle,
    convert_trace,
)
from huggingface import _scenario_row, load_packaged_rows, upload_dataset, write_readme, write_summary


DEFAULT_OUTPUT_DIR = Path("experiments/agentdojo/hf_dataset")
DEFAULT_BENCHMARK_VERSION = "v1.2.2"
DEFAULT_SUITES = ["workspace", "travel", "banking", "slack"]
DEFAULT_ATTACKS = [
    "direct",
    "ignore_previous",
    "important_instructions",
    "important_instructions_no_names",
    "important_instructions_no_user_name",
    "important_instructions_wrong_user_name",
    "injecagent",
    "system_message",
    "tool_knowledge",
]


def _sort_task_ids(ids: list[str]) -> list[str]:
    def key(value: str) -> tuple[str, int]:
        match = re.search(r"_(\d+)$", value)
        return (value[: match.start()] if match else value, int(match.group(1)) if match else -1)

    return sorted(ids, key=key)


def _registry_case_from_loaded_suite(
    suite: Any,
    suite_name: str,
    user_task_id: str,
    injection_task_id: str,
    attack_type: str,
    benchmark_version: str,
) -> dict[str, Any]:
    user_task = suite.get_user_task_by_id(user_task_id)
    injection_task = suite.get_injection_task_by_id(injection_task_id)

    env = suite.load_and_inject_default_environment({})
    try:
        task_env = user_task.init_environment(env.model_copy(deep=True))
        calls = user_task.ground_truth(task_env)
    except Exception:
        calls = []
    try:
        injection_calls = injection_task.ground_truth(env.model_copy(deep=True))
    except Exception:
        injection_calls = []

    injection_goal = getattr(injection_task, "GOAL", "")
    payload = _attack_payload(attack_type, injection_goal, injection_calls)

    trace = {
        "suite_name": suite_name,
        "pipeline_name": "agentdojo_registry",
        "user_task_id": user_task_id,
        "injection_task_id": injection_task_id,
        "attack_type": attack_type,
        "benchmark_version": benchmark_version,
        "utility": None,
        "security": None,
        "injections": {
            "registry_injection_goal": injection_goal,
            "registry_attack_payload": payload,
        },
        "messages": [
            {
                "role": "user",
                "content": _message_text(getattr(user_task, "PROMPT", "")),
            },
            *_registry_tool_messages(calls),
        ],
    }
    scenario = convert_trace(trace, source_path=None)
    scenario["agentdojo_metadata"]["source_file"] = "agentdojo_registry"
    scenario["agentdojo_metadata"]["registry_ground_truth_tools"] = [call.function for call in calls]
    scenario["agentdojo_attack"] = {
        "goal": injection_goal,
        "payload": payload,
        "payload_source": "agentdojo_attack_template",
    }
    scenario["state_delta_labels"] = _state_delta_oracle(calls, injection_calls)
    _add_attack_execution_surfaces(scenario, injection_calls)
    _merge_attack_sensitive_policy(scenario, injection_calls)
    return scenario


def _write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _batch_path(output_dir: Path, suite_name: str, attack_type: str) -> Path:
    legacy_workspace_tool = output_dir / "data" / "scenarios.jsonl"
    canonical = output_dir / "data" / f"{suite_name}_{attack_type}.jsonl"
    if suite_name == "workspace" and attack_type == "tool_knowledge" and legacy_workspace_tool.exists():
        return legacy_workspace_tool
    return canonical


def build_batch(
    suite_name: str,
    attack_type: str,
    output_dir: Path,
    benchmark_version: str,
) -> int:
    suite = get_suite(benchmark_version, suite_name)
    user_task_ids = _sort_task_ids(list(suite.user_tasks.keys()))
    injection_task_ids = _sort_task_ids(list(suite.injection_tasks.keys()))

    rows: list[dict[str, Any]] = []
    for user_task_id in user_task_ids:
        for injection_task_id in injection_task_ids:
            scenario = _registry_case_from_loaded_suite(
                suite=suite,
                suite_name=suite_name,
                user_task_id=user_task_id,
                injection_task_id=injection_task_id,
                attack_type=attack_type,
                benchmark_version=benchmark_version,
            )
            rows.append(_scenario_row(scenario))

    output_path = _batch_path(output_dir, suite_name, attack_type)
    _write_jsonl(rows, output_path)
    print(f"[batch] {suite_name}/{attack_type}: {len(rows)} -> {output_path}")
    return len(rows)


def refresh_metadata(output_dir: Path, repo_id: str) -> int:
    data_dir = output_dir / "data"
    rows = load_packaged_rows(data_dir)
    data_files = [path.name for path in sorted(data_dir.glob("*.jsonl"))]
    write_summary(rows, data_dir / "summary.csv")
    write_readme(rows, output_dir / "README.md", repo_id, data_files)
    print(f"[summary] {len(rows)} scenarios across {len(data_files)} jsonl files")
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suites", nargs="+", default=DEFAULT_SUITES)
    parser.add_argument("--attacks", nargs="+", default=DEFAULT_ATTACKS)
    parser.add_argument("--benchmark-version", default=DEFAULT_BENCHMARK_VERSION)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--repo-id", default="julee0323/agentdojo")
    parser.add_argument("--upload", action="store_true")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Do not rebuild suite/attack JSONL files that already exist.",
    )
    args = parser.parse_args()

    for suite_name in args.suites:
        for attack_type in args.attacks:
            path = args.output_dir / "data" / f"{suite_name}_{attack_type}.jsonl"
            path = _batch_path(args.output_dir, suite_name, attack_type)
            if args.skip_existing and path.exists():
                print(f"[skip] {suite_name}/{attack_type}: {path}")
                continue
            build_batch(
                suite_name=suite_name,
                attack_type=attack_type,
                output_dir=args.output_dir,
                benchmark_version=args.benchmark_version,
            )

    refresh_metadata(args.output_dir, args.repo_id)
    if args.upload:
        upload_dataset(args.output_dir, args.repo_id)
        print(f"[upload] pushed to https://huggingface.co/datasets/{args.repo_id}")


if __name__ == "__main__":
    main()
