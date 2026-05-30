"""
magpie_adapter.py — MAGPIE Dataset → SafeSagaLLM Agent Adapter
================================================================
Loads scenarios from the MAGPIE HuggingFace dataset
(jaypasnagasai/magpie) and converts each row into a list of
SafeSagaLLM Agent objects wired in a linear execution chain.

Each MAGPIE agent's private_preferences values are treated as
the sensitive keyword set K for that scenario.  Shareable
preferences are included in the task description but are NOT
considered sensitive (they are permitted to flow through the chain).

Usage (standalone inspection):
    python magpie_adapter.py --index 0
"""

import json
import sys
from pathlib import Path
from typing import Any


# ── Helpers ──────────────────────────────────────────────────────────────────

def _pick(obj: dict, *keys, default=None) -> Any:
    """Return the first matching key value from a dict; fallback to default."""
    if not isinstance(obj, dict):
        return default
    for key in keys:
        if key in obj:
            return obj[key]
    return default


def _ensure_dict(raw) -> dict:
    """Coerce a raw field to dict (handles JSON strings, None, lists)."""
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, dict) else {}
        except (json.JSONDecodeError, ValueError):
            return {}
    return {}


def _ensure_list(raw) -> list:
    """Coerce a raw field to list."""
    if raw is None:
        return []
    if isinstance(raw, list):
        return raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            return parsed if isinstance(parsed, list) else []
        except (json.JSONDecodeError, ValueError):
            return []
    return []


def _private_pref_value(raw) -> str:
    """
    Extract the actual private preference value from MAGPIE's nested schema.

    MAGPIE private_preferences commonly look like:
        {"field": {"value": "...", "reason": "...", "utility_impact": "..."}}

    The leakage detector and Rego policy should use only the private "value",
    not the whole dict string, otherwise exact matching becomes nearly useless.
    """
    if raw is None:
        return ""
    if isinstance(raw, dict):
        value = _pick(raw, "value", "text", "preference", "content", default="")
        return str(value).strip() if value is not None else ""
    return str(raw).strip()


# ── Dataset loader ────────────────────────────────────────────────────────────

def load_magpie_dataset(split: str = "train", limit: int = None):
    """
    Load the MAGPIE dataset from HuggingFace.

    Args:
        split:  dataset split ("train", "test", etc.)
        limit:  cap the number of rows; None = load all

    Returns:
        HuggingFace Dataset object
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "The 'datasets' package is required.  Install it with:\n"
            "    pip install datasets\n"
            "Then re-run the experiment."
        )

    print(f"[magpie_adapter] Loading jaypasnagasai/magpie ({split})…")
    ds = load_dataset("jaypasnagasai/magpie", split=split)

    if limit is not None:
        ds = ds.select(range(min(limit, len(ds))))
    print(f"[magpie_adapter] Loaded {len(ds)} scenarios.")
    return ds


# ── Schema inspection ─────────────────────────────────────────────────────────

def inspect_row(row: dict, max_chars: int = 120) -> None:
    """Pretty-print the top-level keys and shapes of a single dataset row."""
    print("\n── MAGPIE row schema ──")
    for k, v in row.items():
        if isinstance(v, list):
            inner = f"list[{len(v)}]"
            if v and isinstance(v[0], dict):
                inner += f"  keys={list(v[0].keys())[:8]}"
        elif isinstance(v, dict):
            inner = f"dict  keys={list(v.keys())[:8]}"
        else:
            inner = repr(v)[:max_chars]
        print(f"  {k:<30} {inner}")
    print()


# ── Core adapter ──────────────────────────────────────────────────────────────

def row_to_agents(row: dict):
    """
    Convert one MAGPIE dataset row to a list of SafeSagaLLM Agent objects.

    The agents are linked as a linear execution chain:
        agents[0] >> agents[1] >> agents[2] >> …

    Args:
        row:  a single row from the MAGPIE dataset (dict)

    Returns:
        agents   (list[Agent])  — Agent objects in execution order
        metadata (dict)         — scenario-level data for policy generation
            "scenario_id"       str
            "scenario_task"     str
            "sensitive_values"  {field_key: actual_value_string}  (private prefs)
            "agent_permissions" {agent_name: set_of_allowed_private_keys}
            "agent_names"       [str]  — ordered agent names
            "per_agent_private" {agent_name: {key: value}}
    """
    # Lazy import so this module can be imported without adding src/ to sys.path
    # (the caller is responsible for sys.path setup)
    from multi_agent.agent import Agent

    # ── Extract top-level scenario fields ────────────────────────────────────
    scenario_id   = str(_pick(row, "scenario_id", "id", default="unknown"))
    scenario_task = _pick(row, "scenario_task", "task", "description", default="")
    if not isinstance(scenario_task, str):
        scenario_task = str(scenario_task) if scenario_task else ""

    # ── Extract agent list ────────────────────────────────────────────────────
    raw_agents = _ensure_list(_pick(row, "agents", "agent_list", default=[]))
    if not raw_agents:
        raise ValueError(
            f"Scenario '{scenario_id}' has no 'agents' field (or it is empty).\n"
            "Call inspect_row(row) to view the actual schema."
        )

    # ── Build Agent objects ───────────────────────────────────────────────────
    agents: list[Agent] = []
    sensitive_values: dict[str, str] = {}   # key → private value (all agents merged)
    per_agent_private: dict[str, dict] = {}  # agent_name → {key: value}
    agent_permissions: dict[str, set]  = {}  # agent_name → permitted private keys

    for idx, raw in enumerate(raw_agents):
        raw = _ensure_dict(raw) if not isinstance(raw, dict) else raw

        # ── Agent identity ────────────────────────────────────────────────────
        name = _pick(raw, "agent_name", "name", "agent", default=f"Agent_{idx}")
        role = _pick(raw, "role", "role_title", default="")
        desc = _pick(raw, "description", "agent_description", "desc", default="")

        if not isinstance(name, str):
            name = str(name)
        if not isinstance(role, str):
            role = str(role) if role else ""
        if not isinstance(desc, str):
            desc = str(desc) if desc else ""

        # ── Backstory: role + description ─────────────────────────────────────
        parts = [p for p in [role, desc] if p]
        backstory = "  ".join(parts) if parts else f"You are agent '{name}'."

        # ── Preferences ───────────────────────────────────────────────────────
        shareable = _ensure_dict(
            _pick(raw, "shareable_preferences", "shareable_prefs", "shared_preferences")
        )
        private = _ensure_dict(
            _pick(raw, "private_preferences", "private_prefs", "sensitive_preferences")
        )

        # ── Expected output ───────────────────────────────────────────────────
        deliverable    = _pick(raw, "deliverable", "output", default="")
        success_crit   = _pick(raw, "success_criteria", "success_criterion", "criteria", default="")
        if not isinstance(deliverable, str):
            deliverable = str(deliverable) if deliverable else ""
        if not isinstance(success_crit, str):
            success_crit = str(success_crit) if success_crit else ""

        # ── task_description: scenario task + shareable + private prefs ───────
        td_parts = []
        if scenario_task:
            td_parts.append(scenario_task)
        if shareable:
            share_lines = "\n".join(f"  - {k}: {v}" for k, v in shareable.items())
            td_parts.append(f"Shareable preferences:\n{share_lines}")
        if private:
            priv_lines = "\n".join(f"  - {k}: {v}" for k, v in private.items())
            td_parts.append(f"Private preferences:\n{priv_lines}")
        task_description = "\n\n".join(td_parts) if td_parts else "(no task)"

        # ── task_expected_output: deliverable + success_criteria ──────────────
        eo_parts = []
        if deliverable:
            eo_parts.append(f"Deliverable: {deliverable}")
        if success_crit:
            eo_parts.append(f"Success criteria: {success_crit}")
        task_expected_output = "\n".join(eo_parts)

        agent = Agent(
            name=name,
            backstory=backstory,
            task_description=task_description,
            task_expected_output=task_expected_output,
        )
        agents.append(agent)

        # ── Accumulate sensitive values (private prefs) ───────────────────────
        for k, v in private.items():
            val_str = _private_pref_value(v)
            if val_str:
                sensitive_values[k] = val_str
        per_agent_private[name] = {
            k: val
            for k, v in private.items()
            if (val := _private_pref_value(v))
        }

        # ── Permissions: agents receive no private values from other agents ───
        # (Each agent's own private data should not flow through the chain.)
        agent_permissions[name] = set()   # fail-closed: deny all cross-agent private data

    # ── Wire linear chain: agents[i] >> agents[i+1] ──────────────────────────
    for i in range(len(agents) - 1):
        agents[i] >> agents[i + 1]

    metadata = {
        "scenario_id":      scenario_id,
        "scenario_task":    scenario_task,
        "sensitive_values": sensitive_values,   # {key: value_string}  — used for scan_fields
        "agent_permissions": agent_permissions, # {name: set()} — all empty (no cross-agent PII)
        "agent_names":      [a.name for a in agents],
        "per_agent_private": per_agent_private,
    }

    return agents, metadata


# ── CLI helper ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Inspect a MAGPIE dataset row")
    parser.add_argument("--index", type=int, default=0, help="Row index to inspect (default: 0)")
    parser.add_argument("--split", default="train", help="Dataset split (default: train)")
    args = parser.parse_args()

    ds = load_magpie_dataset(split=args.split, limit=args.index + 1)
    row = dict(ds[args.index])
    inspect_row(row)

    # Add src/ to path so Agent can be imported
    src_path = Path(__file__).parent.parent.parent / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))

    try:
        agents, meta = row_to_agents(row)
        print(f"Scenario: {meta['scenario_id']}")
        print(f"Agents ({len(agents)}): {meta['agent_names']}")
        print(f"Sensitive values ({len(meta['sensitive_values'])}):")
        for k, v in meta["sensitive_values"].items():
            print(f"  {k}: {v!r}")
        print(f"Agent chain: {' → '.join(meta['agent_names'])}")
    except Exception as e:
        print(f"[ERROR] {e}")
        print("Tip: run inspect_row(row) manually to view the schema.")
