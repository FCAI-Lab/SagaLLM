"""
magpie_policy_generator.py — MAGPIE Scenario → Rego Policy Generator
======================================================================
Generates a SafeSagaLLM-compatible Rego policy (P_tran + P_cont) from
a MAGPIE scenario's metadata produced by magpie_adapter.row_to_agents().

Policy design decisions:
  P_tran:  Linear full-chain.  agents[i] → agents[i+1] for every consecutive
           pair in the scenario.  (Coordinator structures are not assumed
           because MAGPIE scenarios are sequential by construction.)
  P_cont:  sensitive_keywords = actual VALUES of all agents' private_preferences.
           agent_keyword_permissions = {} for every non-source agent.
           No agent is permitted to receive another agent's private data.

The generated policy is written to:
    SafeSagaLLM/src/policies/magpie_generated.rego

It uses package sagallm.access_control so the OPA client can call it at
    v1/data/sagallm/access_control
matching the transfer_path used in the field exposure experiment.

Usage:
    from magpie_adapter import row_to_agents
    from magpie_policy_generator import generate_policy, save_policy

    agents, metadata = row_to_agents(row)
    rego = generate_policy(metadata)
    save_policy(rego)
"""

import re
from pathlib import Path
from typing import Optional

# Default output path (relative to project root)
_DEFAULT_POLICY_PATH = (
    Path(__file__).parent.parent.parent / "src" / "policies" / "magpie_generated.rego"
)


# ── Rego escaping ─────────────────────────────────────────────────────────────

def _rego_str(value: str) -> str:
    """Escape a string for safe embedding inside a Rego string literal."""
    return value.replace("\\", "\\\\").replace('"', '\\"')


def _rego_set(values) -> str:
    """Format a Python set/list as a Rego set literal."""
    if not values:
        return "{}"
    items = ",\n    ".join(f'"{_rego_str(v)}"' for v in sorted(values))
    return "{\n    " + items + "\n}"


def _rego_allowed_transfers(agent_names: list[str]) -> str:
    """
    Build the P_tran allowed_transfers Rego map for a linear chain.

    agents[0] → agents[1] → … → agents[-1]

    Each agent maps to its immediate successor; the last agent maps to {}.
    """
    lines = ["allowed_transfers := {"]
    for i, name in enumerate(agent_names):
        if i < len(agent_names) - 1:
            successor = _rego_str(agent_names[i + 1])
            lines.append(f'    "{_rego_str(name)}": {{"{successor}"}},')
        else:
            lines.append(f'    "{_rego_str(name)}": {{}}')
    lines.append("}")
    return "\n".join(lines)


def _rego_keyword_permissions(agent_names: list[str]) -> str:
    """
    Build the P_cont agent_keyword_permissions Rego map.

    All agents have an empty allowlist: no agent is permitted to receive
    another agent's private values.  (Fail-closed privacy model.)
    """
    lines = ["agent_keyword_permissions := {"]
    for i, name in enumerate(agent_names):
        comma = "," if i < len(agent_names) - 1 else ""
        lines.append(f'    "{_rego_str(name)}": {{}}' + comma)
    lines.append("}")
    return "\n".join(lines)


_STOPWORDS = {
    "a", "about", "after", "again", "against", "also", "an", "and", "are",
    "as", "at", "because", "before", "being", "between", "but", "by", "can",
    "could", "currently", "does", "each", "for", "from", "have", "in", "into",
    "is", "it", "just", "make", "more", "must", "of", "on", "only", "or",
    "other", "over", "properly", "revealing", "share", "sharing", "should",
    "that", "the", "their", "them", "there", "these", "this", "through", "to",
    "under", "very", "was", "what", "when", "where", "which", "while", "will",
    "with", "would",
}


def _keyword_fragments(value: str, max_fragments: int = 80) -> set[str]:
    """
    Build shorter sensitive phrases from a long MAGPIE private value.

    Exact full-paragraph matching misses natural LLM paraphrases such as
    "critical memory leak".  These fragments keep the policy simple while
    making line-level redaction useful for MAGPIE's long private constraints.
    """
    text = " ".join(str(value).split())
    if not text:
        return set()

    fragments: set[str] = set()
    if len(text) <= 300:
        fragments.add(text)

    # Quoted terms often carry private entities, feature names, or allowed hints.
    for match in re.findall(r"'([^']{4,100})'|\"([^\"]{4,100})\"", text):
        quoted = (match[0] or match[1]).strip()
        if quoted:
            fragments.add(quoted)

    # Dates, money amounts, percentages, codes, and named entities are often
    # the highest-risk leakage units in MAGPIE scenarios.
    patterns = [
        r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\s+\d{1,2}(?:st|nd|rd|th)?(?:,\s+\d{4})?\b",
        r"\b\d{1,2}(?:st|nd|rd|th)?\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*(?:\s+\d{4})?\b",
        r"\$\d+(?:\.\d+)?\s*(?:k|m|million|billion)?\b",
        r"\b\d+(?:\.\d+)?%\b",
        r"\b[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){1,3}\b",
    ]
    for pattern in patterns:
        fragments.update(m.group(0).strip() for m in re.finditer(pattern, text))

    # Add compact 2-4 word n-grams around informative words.
    words = re.findall(r"[A-Za-z0-9][A-Za-z0-9'-]*", text)
    for n in (2, 3, 4):
        for i in range(max(0, len(words) - n + 1)):
            gram_words = words[i:i + n]
            lower_words = [w.lower().strip("'") for w in gram_words]
            if all(w in _STOPWORDS or len(w) <= 2 for w in lower_words):
                continue
            if lower_words[0] in _STOPWORDS or lower_words[-1] in _STOPWORDS:
                continue
            phrase = " ".join(gram_words).strip()
            if 10 <= len(phrase) <= 80:
                fragments.add(phrase)
            if len(fragments) >= max_fragments:
                return fragments

    return fragments


# ── Policy generator ──────────────────────────────────────────────────────────

def generate_policy(metadata: dict) -> str:
    """
    Generate a Rego policy string from MAGPIE scenario metadata.

    Args:
        metadata:  dict returned by magpie_adapter.row_to_agents()
                   Required keys: "agent_names", "sensitive_values"

    Returns:
        Complete Rego policy as a string.
    """
    agent_names: list[str]      = metadata.get("agent_names", [])
    sensitive_values: dict      = metadata.get("sensitive_values", {})
    scenario_id: str            = metadata.get("scenario_id", "unknown")

    if not agent_names:
        raise ValueError("metadata['agent_names'] is empty — cannot generate policy.")

    # sensitive_keywords: actual VALUE strings and shorter fragments from all
    # agents' private_preferences.
    kw_values: set[str] = set()
    for val in sensitive_values.values():
        v = str(val).strip()
        if v:
            kw_values.add(v)
            kw_values.update(_keyword_fragments(v))

    # Also include the KEY names so label-based lines are caught
    kw_keys: set[str] = set()
    for key in sensitive_values.keys():
        k = str(key).strip()
        if k:
            kw_keys.add(k)

    all_keywords = kw_values | kw_keys

    # Build Rego blocks
    sensitive_kw_block    = _rego_set(all_keywords)
    allowed_trans_block   = _rego_allowed_transfers(agent_names)
    kw_permissions_block  = _rego_keyword_permissions(agent_names)

    policy = f"""\
# magpie_generated.rego — Auto-generated by magpie_policy_generator.py
# Scenario: {_rego_str(scenario_id)}
#
# P_tran:  Linear chain  {' → '.join(agent_names)}
# P_cont:  sensitive_keywords = VALUES of all agents' private_preferences
#          All agents have empty allowlist (fail-closed: no cross-agent PII)
#
# Package path: sagallm.access_control
# OPA endpoint: v1/data/sagallm/access_control

package sagallm.access_control

import future.keywords.in

# ── Defaults (fail-closed) ────────────────────────────────────────────────────
default allow_transfer    := false
default reason            := "Transfer denied by default policy"
default matching_keywords := []
default filtered_content  := ""

# ── P_cont: sensitive keyword set K (private preference values + label keys) ──
sensitive_keywords := {sensitive_kw_block}

# ── P_cont: per-receiver keyword allowlist ────────────────────────────────────
# All agents have empty permissions: no agent receives another agent's private data.
{kw_permissions_block}

# ── P_tran: authorized transfer edges (linear chain) ─────────────────────────
{allowed_trans_block}

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

# ── Denial reasons ────────────────────────────────────────────────────────────
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

    return policy


# ── File writer ───────────────────────────────────────────────────────────────

def save_policy(rego: str, path: Optional[Path] = None) -> Path:
    """
    Write the generated Rego policy to disk.

    Args:
        rego:  policy string from generate_policy()
        path:  output path; defaults to SafeSagaLLM/src/policies/magpie_generated.rego

    Returns:
        Resolved Path of the written file.
    """
    out_path = Path(path) if path else _DEFAULT_POLICY_PATH
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(rego, encoding="utf-8")
    print(f"[magpie_policy_generator] Policy written → {out_path}")
    return out_path


# ── CLI helper ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Generate a Rego policy from a MAGPIE scenario")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--split", default="train")
    parser.add_argument("--output", default=None, help="Override output .rego path")
    args = parser.parse_args()

    src_path = Path(__file__).parent.parent.parent / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))

    from magpie_adapter import load_magpie_dataset, row_to_agents

    ds  = load_magpie_dataset(split=args.split, limit=args.index + 1)
    row = dict(ds[args.index])
    agents, meta = row_to_agents(row)

    rego = generate_policy(meta)
    out  = save_policy(rego, path=args.output)

    print("\n── Generated policy preview (first 40 lines) ──")
    for line in rego.splitlines()[:40]:
        print(" ", line)
    print(f"\nFull policy → {out}")
