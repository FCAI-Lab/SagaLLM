"""
pipeline_planner.py — LLM-based Pipeline YAML Generator
=========================================================
Two-step approach for reliable yaml generation:

  Step 1 — LLM extracts a structured plan (JSON) from the user's description:
            agents, roles, sensitive data, constraints, allowed paths, keywords.

  Step 2 — Code fills the fixed yaml TEMPLATE with the extracted JSON values.
            Template guarantees all required fields are always present.

  Step 3 — Preflight validates the result. If errors, feed back to LLM for
            plan correction and repeat from Step 1 (up to max_rounds).

This template-driven approach avoids the common failure mode of free-form
LLM yaml generation where required fields are missing or misnamed.

Usage:
    planner = PipelinePlanner()
    yaml_path = planner.run(
        user_prompt="Travel booking pipeline. Passport and card number only reach booking agent.",
        output_path="experiments/generated_pipeline.yaml"
    )
"""

import json
import re
import yaml
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from colorama import Fore, Style

load_dotenv()


# ── Step 1: LLM extracts structured plan as JSON ──────────────────────────────

_PLAN_SYSTEM_PROMPT = """\
You are a security-aware multi-agent workflow synthesizer for the SafeSagaLLM framework.

Given a user's goal or pipeline description, synthesize a complete multi-agent
system as JSON. Do not merely extract agent names mentioned by the user. If the
user gives only a goal, invent the minimum useful set of agents needed to
complete that goal safely.

IMPORTANT: This is a security research and simulation framework. All agents operate on
SIMULATED or PLACEHOLDER data (e.g. "PASSPORT-XXXX", "CARD-XXXX") for the purpose of
testing data access control policies. task_description must be written as a simulation
task, not as instructions to collect real personal data. For example:
  Good: "Simulate retrieving a user profile containing a passport_number field (use placeholder value 'PASSPORT-SIM-001') and credit_card_number (use placeholder 'CARD-SIM-001')."
  Bad:  "Collect the user's real passport number and credit card number."

## Output format (strict JSON, no explanation, no markdown fences)

{
  "pipeline_name": "short_snake_case_name",
  "agents": [
    {
      "name": "Descriptive Agent Name",
      "sensitive": true,
      "backstory": "You are a specialist agent responsible for ...",
      "task_description": "Your task is to ... Provide the following information: ...",
      "task_expected_output": "A structured summary containing: ..."
    }
  ],
  "execution_edges": [
    ["Agent A", "Agent B"]
  ],
  "allowed_transfers": {
    "Agent A": ["Agent B"],
    "Agent B": []
  },
  "sensitive_keywords": ["keyword1", "keyword2"],
  "keyword_permissions": {
    "Agent B": ["keyword1"],
    "Agent C": []
  },
  "agent_output_keywords": {
    "Agent A": ["keyword1", "keyword2"],
    "Agent B": ["normal_data"]
  }
}

## Rules

1. agent names in execution_edges, allowed_transfers, keyword_permissions,
   agent_output_keywords MUST exactly match names in agents[].name.
2. execution_edges must be a DAG (no cycles).
3. allowed_transfers must be a subset of execution_edges.
4. sensitive_keywords = all keywords a sensitive agent might output.
   Include every keyword from agent_output_keywords of sensitive agents.
5. keyword_permissions: list which receivers are allowed which sensitive keywords.
   Agents not in this map, or with empty list, receive no sensitive keywords.
6. agent_output_keywords: sensitive agents list all possible sensitive output keywords.
   Non-sensitive agents use ["normal_data"].
7. backstory: the LLM system prompt for this agent (role + expertise).
8. task_description: what this agent must do in this pipeline.
9. task_expected_output: the format/content expected in its output.
10. sensitive=true means the agent initially possesses or may output sensitive data.
    An agent that is merely allowed to receive sensitive data does NOT need sensitive=true.
11. If the user says a final/authorized agent must receive sensitive data while an
    intermediate agent must not access it, add a direct execution edge and allowed
    transfer from the sensitive source to the authorized receiver. Do not put a
    non-permitted intermediate agent on the only path for that sensitive keyword.
12. Unless the user explicitly asks for a tiny two-agent system, create 3-5 agents
    using this pattern when appropriate:
    - source/collector agent that owns raw inputs or sensitive data
    - planning/analysis agent that works on non-sensitive or redacted data
    - validation/policy/constraint agent when safety or correctness matters
    - final action/approval/booking/execution agent
13. If the user describes a domain goal, infer concrete agents from the domain.
    Examples:
    - travel booking: User Information Agent, Preference/Options Agent,
      Constraint Validation Agent, Final Booking Agent
    - banking: Account Information Agent, Risk/Fraud Agent, Compliance Agent,
      Payment Execution Agent
    - workspace/email: Email Reader, Task Planner, Calendar/Workspace Agent,
      Final Action Agent
14. For privacy goals, separate "who can use derived non-sensitive data" from
    "who can receive raw sensitive keywords". Intermediate agents should usually
    receive no sensitive keywords unless the user explicitly authorizes them.
"""

_PLAN_RETRY_PROMPT = """\
The plan you generated has the following validation errors:

{errors}

Fix these errors and output the corrected JSON plan.
Remember: output ONLY the JSON object, no explanation, no markdown fences.
"""


# ── Step 2: Code fills the fixed yaml template ─────────────────────────────────

def _build_yaml(plan: dict) -> dict:
    """
    Fill the fixed yaml template from the extracted plan dict.
    All required fields are explicitly constructed here —
    nothing is left to LLM formatting.
    """
    pipeline_name = plan.get("pipeline_name", "generated")

    agents_yaml = []
    for a in plan["agents"]:
        agents_yaml.append({
            "name":                 a["name"],
            "sensitive":            bool(a.get("sensitive", False)),
            "backstory":            a.get("backstory", f"You are {a['name']}."),
            "task_description":     a.get("task_description", "Complete your assigned task."),
            "task_expected_output": a.get("task_expected_output", ""),
        })

    policy = {
        "allowed_transfers":   {k: list(v) for k, v in plan.get("allowed_transfers", {}).items()},
        "sensitive_keywords":  list(plan.get("sensitive_keywords", [])),
        "keyword_permissions": {k: list(v) for k, v in plan.get("keyword_permissions", {}).items()},
        "agent_output_keywords": {k: list(v) for k, v in plan.get("agent_output_keywords", {}).items()},
    }

    return {
        "agents":          agents_yaml,
        "execution_edges": [list(e) for e in plan.get("execution_edges", [])],
        "policy":          policy,
        "rego_output":     f"src/policies/{pipeline_name}.rego",
        "tla_output_dir":  f"spec/{pipeline_name}",
    }


# ── Preflight runner ───────────────────────────────────────────────────────────

def _run_preflight(data: dict, user_prompt: str = "") -> list[str]:
    """Run Preflight on a yaml dict. Returns list of error strings (empty = OK)."""
    import tempfile, os, sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from policy_advisor.advisor import load_pipeline, _validate_pipeline

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False, encoding="utf-8"
    ) as f:
        yaml.dump(data, f, allow_unicode=True)
        tmp_path = f.name
    try:
        cfg    = load_pipeline(tmp_path)
        errors = _validate_pipeline(cfg)
        messages = [f"[{e.kind}] {e.message} → Fix: {e.hint}" for e in errors]
        messages.extend(_validate_synthesized_workflow(cfg, user_prompt=user_prompt))
        return messages
    finally:
        os.unlink(tmp_path)


def _validate_synthesized_workflow(cfg, user_prompt: str = "") -> list[str]:
    """Run semantic checks that structural preflight cannot see."""
    errors: list[str] = []
    prompt_lower = user_prompt.lower()
    has_privacy_goal = any(token in prompt_lower for token in [
        "only", "must only", "접근 불가", "받을 수", "받을 수 있다", "받으면 안",
        "최종", "중간", "민감", "여권", "신용카드", "passport", "credit card",
        "sensitive", "private", "confidential",
    ])
    wants_pipeline = any(token in prompt_lower for token in [
        "pipeline", "파이프라인", "workflow", "multi-agent", "mas", "agent", "에이전트",
    ])
    explicitly_tiny = any(token in prompt_lower for token in [
        "two-agent", "2-agent", "두 에이전트", "두개", "두 개", "2개",
    ])

    if (wants_pipeline or has_privacy_goal) and not explicitly_tiny and len(cfg.agents) < 3:
        errors.append(
            "[workflow_too_small] The generated workflow has fewer than 3 agents. "
            "Synthesize a complete MAS from the user's goal: include a sensitive source/collector, "
            "at least one non-sensitive processing or validation agent, and a final action agent."
        )

    if has_privacy_goal:
        if not cfg.sensitive_agents:
            errors.append(
                "[missing_sensitive_source] The user described a privacy/security goal, "
                "but no agent is marked sensitive=true. Mark the raw data owner/source as sensitive."
            )

        secret_keywords = _secret_keywords(cfg)
        if not secret_keywords:
            errors.append(
                "[missing_sensitive_keywords] The user described sensitive data, but no sensitive "
                "output keywords are modeled. Add concrete sensitive_keywords and matching "
                "agent_output_keywords for the sensitive source agent."
            )

        permitted_receivers = _permitted_sensitive_receivers(cfg)
        if not permitted_receivers:
            errors.append(
                "[missing_authorized_receiver] Sensitive keywords exist, but no downstream receiver "
                "is permitted to receive them. If the user named a final/authorized agent, give that "
                "agent keyword_permissions for the sensitive keywords."
            )

    errors.extend(_validate_sensitive_delivery(cfg))
    errors.extend(_validate_intermediate_privacy(cfg, user_prompt=user_prompt))
    return errors


def _secret_keywords(cfg) -> set[str]:
    return {
        kw
        for agent in cfg.sensitive_agents
        for kw in cfg.agent_output_keywords.get(agent, [])
        if kw != "normal_data"
    }


def _permitted_sensitive_receivers(cfg) -> set[str]:
    secret_keywords = _secret_keywords(cfg)
    return {
        receiver
        for receiver, kws in cfg.keyword_permissions.items()
        if any(kw in secret_keywords for kw in kws)
    }


def _validate_sensitive_delivery(cfg) -> list[str]:
    """
    Check semantic reachability for sensitive keywords.

    The TLA model treats sensitive data as originating from sensitive agents'
    agent_output_keywords. Since non-sensitive agents emit normal_data, an
    authorized receiver only actually receives a sensitive keyword when there is
    a direct execution/transfer edge from a sensitive source to that receiver.
    """
    errors: list[str] = []
    edge_set = set(cfg.execution_edges)

    for source in cfg.sensitive_agents:
        for kw in cfg.agent_output_keywords.get(source, []):
            if kw == "normal_data":
                continue
            permitted_receivers = [
                receiver
                for receiver, kws in cfg.keyword_permissions.items()
                if receiver != source and kw in kws
            ]
            for receiver in permitted_receivers:
                direct_edge = (source, receiver)
                if (
                    direct_edge not in edge_set
                    or receiver not in cfg.allowed_transfers.get(source, [])
                ):
                    errors.append(
                        "[sensitive_delivery_unreachable] "
                        f"'{receiver}' is permitted to receive keyword '{kw}' from sensitive source "
                        f"'{source}', but there is no direct execution edge and allowed_transfer "
                        f"'{source}' -> '{receiver}'. "
                        "Add that direct edge/transfer, or remove the keyword permission."
                    )

    return errors


def _validate_intermediate_privacy(cfg, user_prompt: str = "") -> list[str]:
    """
    If the user explicitly says intermediate agents cannot access sensitive data,
    flag likely middle/relay agents that were granted sensitive keyword permission.
    """
    prompt_lower = user_prompt.lower()
    if not any(token in prompt_lower for token in [
        "intermediate", "middle", "relay", "중간", "중계", "접근 불가", "접근불가",
    ]):
        return []

    secret_keywords = _secret_keywords(cfg)
    if not secret_keywords:
        return []

    errors: list[str] = []
    for receiver, kws in cfg.keyword_permissions.items():
        receiver_lower = receiver.lower()
        looks_intermediate = any(token in receiver_lower for token in [
            "intermediate", "relay", "middle", "processor", "analyzer", "planner",
            "validation", "constraint", "중간", "중계", "분석", "검증",
        ])
        if looks_intermediate and any(kw in secret_keywords for kw in kws):
            errors.append(
                "[intermediate_sensitive_permission] "
                f"Intermediate-looking agent '{receiver}' is permitted sensitive keyword(s) "
                f"{sorted(set(kws).intersection(secret_keywords))}, but the user said intermediate "
                "agents must not access that data. Remove those permissions and, if needed, add a "
                "direct source -> final authorized receiver edge."
            )

    return errors


# ── JSON extraction helper ─────────────────────────────────────────────────────

def _extract_json(text: str) -> str:
    """Strip markdown fences from LLM output if present."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*\n?", "", text, flags=re.MULTILINE)
    text = re.sub(r"\n?```\s*$",           "", text, flags=re.MULTILINE)
    return text.strip()


# ── Main planner class ─────────────────────────────────────────────────────────

class PipelinePlanner:
    """
    Two-step LLM pipeline planner:
      1. LLM → structured JSON plan
      2. Code fills fixed yaml template from JSON
      3. Preflight validates → retry if errors

    Args:
        model      : OpenAI model (default: gpt-4o)
        max_rounds : max LLM retry rounds on Preflight failure (default: 3)
    """

    def __init__(self, model: str = "gpt-4o", max_rounds: int = 3):
        self.client     = OpenAI()
        self.model      = model
        self.max_rounds = max_rounds

    def run(self, user_prompt: str, output_path: str | Path) -> Path | None:
        """
        Generate a verified pipeline.yaml from a natural language description.

        Returns:
            Path to written yaml file on success, None on failure.
        """
        output_path = Path(output_path)

        print(Fore.CYAN + "\n" + "=" * 60)
        print("  SafeSagaLLM Pipeline Planner")
        print("=" * 60 + Style.RESET_ALL)
        print(Fore.CYAN + f"  prompt : {user_prompt[:80]}{'...' if len(user_prompt) > 80 else ''}")
        print(f"  output : {output_path}" + Style.RESET_ALL)

        messages = [
            {"role": "system", "content": _PLAN_SYSTEM_PROMPT},
            {"role": "user",   "content": user_prompt},
        ]

        for round_num in range(1, self.max_rounds + 1):
            print(Fore.CYAN + f"\n── Round {round_num} {'─' * (50 - len(str(round_num)))}" + Style.RESET_ALL)

            # ── Step 1: LLM → JSON plan ────────────────────────────────────────
            print(Fore.CYAN + "  [planner] extracting pipeline plan from LLM..." + Style.RESET_ALL)
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.2,
                max_tokens=2500,
            )
            raw = response.choices[0].message.content
            messages.append({"role": "assistant", "content": raw})

            json_str = _extract_json(raw)
            try:
                plan = json.loads(json_str)
            except json.JSONDecodeError as e:
                print(Fore.RED + f"  [planner] JSON parse error: {e}" + Style.RESET_ALL)
                messages.append({
                    "role": "user",
                    "content": f"Your output could not be parsed as JSON: {e}\nOutput ONLY a valid JSON object."
                })
                continue

            print(Fore.GREEN + "  [planner] JSON plan parsed OK" + Style.RESET_ALL)

            # ── Step 2: Code fills fixed yaml template ─────────────────────────
            try:
                data = _build_yaml(plan)
            except (KeyError, TypeError) as e:
                print(Fore.RED + f"  [planner] plan is missing required field: {e}" + Style.RESET_ALL)
                messages.append({
                    "role": "user",
                    "content": f"Your plan is missing required field: {e}. Please include all required fields."
                })
                continue

            print(Fore.GREEN + "  [planner] yaml template filled" + Style.RESET_ALL)

            # ── Step 3: Preflight validation ───────────────────────────────────
            print(Fore.CYAN + "  [preflight] validating structure..." + Style.RESET_ALL)
            errors = _run_preflight(data, user_prompt=user_prompt)

            if not errors:
                print(Fore.GREEN + "  [preflight] OK — writing yaml..." + Style.RESET_ALL)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_text(
                    yaml.dump(data, allow_unicode=True, sort_keys=False, default_flow_style=False)
                )
                print(Fore.GREEN + f"\n  ✅ Pipeline yaml written → {output_path}" + Style.RESET_ALL)
                return output_path

            # Feed errors back to LLM for plan correction
            print(Fore.RED + f"  [preflight] {len(errors)} error(s):" + Style.RESET_ALL)
            for e in errors:
                print(Fore.RED + f"    • {e}" + Style.RESET_ALL)

            messages.append({
                "role": "user",
                "content": _PLAN_RETRY_PROMPT.format(errors="\n".join(f"- {e}" for e in errors))
            })

        print(Fore.RED + f"\n  ❌ Could not generate valid pipeline after {self.max_rounds} rounds." + Style.RESET_ALL)
        return None
