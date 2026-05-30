"""
magpie_pipeline_yaml_generator.py — MAGPIE Scenario → SafeSagaLLM pipeline YAML
================================================================================
Builds the common SafeSagaLLM policy-advisor input format from MAGPIE metadata.

This is the reusable bridge for dataset-backed experiments:
    dataset row → adapter metadata → pipeline YAML → TLC/Advisor → Rego → OPA
"""

from pathlib import Path
from typing import Optional

import yaml

from magpie_policy_generator import _keyword_fragments


_DEFAULT_PIPELINE_DIR = Path(__file__).parent / "generated_pipelines"


def _scenario_stem(scenario_idx: int, scenario_id: str) -> str:
    safe_id = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in str(scenario_id))
    return f"magpie_s{scenario_idx}_{safe_id or 'unknown'}"


def _linear_allowed_transfers(agent_names: list[str]) -> dict[str, list[str]]:
    transfers = {}
    for i, name in enumerate(agent_names):
        transfers[name] = [agent_names[i + 1]] if i < len(agent_names) - 1 else []
    return transfers


def _private_keywords(value: str) -> list[str]:
    value = str(value).strip()
    return sorted(({value} | _keyword_fragments(value)) - {""})


def build_pipeline_yaml(metadata: dict, scenario_idx: int) -> dict:
    """
    Convert MAGPIE adapter metadata into the common SafeSagaLLM pipeline YAML dict.

    The YAML is intentionally conservative:
      - every MAGPIE agent with private preferences is marked sensitive
      - P_tran allows only the linear chain edges
      - P_cont grants no cross-agent private keyword permissions
      - agent_output_keywords records each agent's own private keywords/fragments
    """
    agent_names = metadata.get("agent_names", [])
    scenario_id = metadata.get("scenario_id", "unknown")
    stem = _scenario_stem(scenario_idx, scenario_id)

    per_agent_private = metadata.get("per_agent_private", {})
    all_keywords: set[str] = set()
    agent_output_keywords: dict[str, list[str]] = {}

    for agent_name in agent_names:
        owned_keywords: set[str] = set()
        for value in per_agent_private.get(agent_name, {}).values():
            owned_keywords.update(_private_keywords(value))
        if owned_keywords:
            agent_output_keywords[agent_name] = sorted(owned_keywords)
            all_keywords.update(owned_keywords)
        else:
            agent_output_keywords[agent_name] = ["normal_data"]

    agents = []
    for agent_name in agent_names:
        agents.append({
            "name": agent_name,
            "sensitive": bool(per_agent_private.get(agent_name)),
        })

    execution_edges = [
        [agent_names[i], agent_names[i + 1]]
        for i in range(max(0, len(agent_names) - 1))
    ]

    keyword_permissions = {agent_name: [] for agent_name in agent_names}

    return {
        "agents": agents,
        "execution_edges": execution_edges,
        "policy": {
            "allowed_transfers": _linear_allowed_transfers(agent_names),
            "sensitive_keywords": sorted(all_keywords),
            "keyword_permissions": keyword_permissions,
            "agent_output_keywords": agent_output_keywords,
        },
        "rego_output": "src/policies/magpie_generated.rego",
        "tla_output_dir": f"spec/magpie/{stem}",
    }


def save_pipeline_yaml(data: dict, scenario_idx: int, scenario_id: str, path: Optional[Path] = None) -> Path:
    out_path = Path(path) if path else _DEFAULT_PIPELINE_DIR / f"{_scenario_stem(scenario_idx, scenario_id)}.yaml"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")
    print(f"[magpie_pipeline_yaml_generator] Pipeline YAML written → {out_path}")
    return out_path

