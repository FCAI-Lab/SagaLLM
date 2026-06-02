---
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

Repository: `julee0323/agentdojo`

## Files

- Scenario JSONL files:
- `data/banking_direct.jsonl`
- `data/banking_ignore_previous.jsonl`
- `data/banking_important_instructions.jsonl`
- `data/banking_important_instructions_no_names.jsonl`
- `data/banking_important_instructions_no_user_name.jsonl`
- `data/banking_important_instructions_wrong_user_name.jsonl`
- `data/banking_injecagent.jsonl`
- `data/banking_system_message.jsonl`
- `data/banking_tool_knowledge.jsonl`
- `data/scenarios.jsonl`
- `data/slack_direct.jsonl`
- `data/slack_ignore_previous.jsonl`
- `data/slack_important_instructions.jsonl`
- `data/slack_important_instructions_no_names.jsonl`
- `data/slack_important_instructions_no_user_name.jsonl`
- `data/slack_important_instructions_wrong_user_name.jsonl`
- `data/slack_injecagent.jsonl`
- `data/slack_system_message.jsonl`
- `data/slack_tool_knowledge.jsonl`
- `data/travel_direct.jsonl`
- `data/travel_ignore_previous.jsonl`
- `data/travel_important_instructions.jsonl`
- `data/travel_important_instructions_no_names.jsonl`
- `data/travel_important_instructions_no_user_name.jsonl`
- `data/travel_important_instructions_wrong_user_name.jsonl`
- `data/travel_injecagent.jsonl`
- `data/travel_system_message.jsonl`
- `data/travel_tool_knowledge.jsonl`
- `data/workspace_direct.jsonl`
- `data/workspace_ignore_previous.jsonl`
- `data/workspace_important_instructions.jsonl`
- `data/workspace_important_instructions_no_names.jsonl`
- `data/workspace_important_instructions_no_user_name.jsonl`
- `data/workspace_important_instructions_wrong_user_name.jsonl`
- `data/workspace_injecagent.jsonl`
- `data/workspace_system_message.jsonl`
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

Total scenarios: 8541

Suites:
- banking: 1296
- slack: 945
- travel: 1260
- workspace: 5040

Attack types:
- direct: 949
- ignore_previous: 949
- important_instructions: 949
- important_instructions_no_names: 949
- important_instructions_no_user_name: 949
- important_instructions_wrong_user_name: 949
- injecagent: 949
- system_message: 949
- tool_knowledge: 949

Agent count distribution:
- 5 agents: 9
- 6 agents: 2610
- 7 agents: 1962
- 8 agents: 2511
- 9 agents: 711
- 10 agents: 558
- 11 agents: 108
- 12 agents: 72

## Intended Use

The scenarios are intended for evaluating whether SafeSagaLLM preserves data
isolation, content filtering, atomic termination, and compensation completeness
when AgentDojo prompt-injection scenarios are represented as multi-agent
workflows.
