"""
agent.py — Agent Node
=====================
Represents a single agent in the multi-agent Saga pipeline.

Each Agent wraps one ReactAgent (LLM call) and implements:
  - Agent registry metadata: model, capability, and clearance.
  - Dependency declaration via the >> operator (e.g., agent_a >> agent_b means
    b depends on a and must execute after a).
  - run(): invokes the LLM, then for every declared downstream dependent calls
    OPA to decide whether and how much context to forward (P_tran + P_cont).
  - Dual-store design: the raw LLM output is stored in Saga.context (SagaContext)
    for compensation; the filtered content goes into dependent.context
    (PromptContext) so the LLM only sees policy-sanitized data.

Class-level transfer_logs captures every OPA decision across all agents in a
run so that Saga.report_transfers() can print an audit trail afterward.
"""

from textwrap import dedent
from colorama import Fore, Style


class Agent:
    # Shared across all Agent instances — accumulates OPA decisions for reporting
    transfer_logs: list = []

    def __init__(
        self,
        name: str,
        backstory: str,
        task_description: str,
        task_expected_output: str = "",
        llm: str = "gpt-4o",
        mock_output: str | None = None,
        capability: str | list[str] | None = None,
        clearance: str = "internal",
    ):
        self.name = name
        self.model = llm
        self.capability = capability if capability is not None else "general"
        self.clearance = clearance
        self.backstory = backstory
        self.task_description = task_description
        self.task_expected_output = task_expected_output
        self.mock_output = mock_output
        self.context = ""          # PromptContext C_a: policy-filtered inputs from upstream agents
        self.dependencies: list[Agent] = []   # upstream agents this agent waits on
        self.dependents: list[Agent] = []     # downstream agents that wait on this agent

        from planning_agent.react_agent import ReactAgent
        self.react_agent = ReactAgent(
            model=llm, system_prompt=self.backstory, tools=[]
        )

    def __repr__(self):
        return self.name

    # ── Dependency declaration ─────────────────────────────────────────────────

    def __rshift__(self, other):
        """
        self >> other declares that 'other' depends on 'self'.

        Usage:
            agent_a >> agent_b            # b waits for a
            agent_a >> [agent_b, agent_c] # fan-out: b and c both wait for a
        """
        if isinstance(other, Agent):
            other.dependencies.append(self)
            self.dependents.append(other)
        elif isinstance(other, list):
            for o in other:
                o.dependencies.append(self)
                self.dependents.append(o)
        return other

    def __rrshift__(self, other):
        """[agent_a, agent_b] >> agent_c: c depends on both a and b (fan-in)."""
        if isinstance(other, list):
            for o in other:
                self.dependencies.append(o)
                o.dependents.append(self)
        return self

    # ── Context management ────────────────────────────────────────────────────

    def receive_context(self, data: str):
        """Append filtered upstream output to this agent's PromptContext."""
        self.context += data

    def create_prompt(self) -> str:
        """Build the full prompt injected into the LLM: task + expected output + context."""
        return dedent(f"""
            You are an AI agent working as part of a multi-agent team.

            <task_description>
            {self.task_description}
            </task_description>

            <task_expected_output>
            {self.task_expected_output}
            </task_expected_output>

            <context>
            {self.context if self.context else "(no context received)"}
            </context>

            Your response:
        """).strip()

    # ── Execution ─────────────────────────────────────────────────────────────

    def run(self, enforce_opa: bool = True) -> str:
        """
        Execute this agent:
          1. Call the LLM to get raw output (stored in SagaContext by the coordinator).
          2. For each downstream dependent, call OPA to evaluate P_tran + P_cont.
          3. Forward policy-filtered content to dependent.context (PromptContext).
          4. Log every transfer decision for audit reporting.

        With enforce_opa=False (baseline/SagaLLM mode), all content is forwarded
        unconditionally — no OPA call is made.
        """
        if self.mock_output is not None:
            output = self.mock_output
        else:
            output = self.react_agent.run(user_msg=self.create_prompt())

        from utils.opa_client import opa

        for dependent in self.dependents:
            if enforce_opa:
                # Single atomic OPA REST call evaluates both P_tran and P_cont
                allowed, reason, censored_keywords, filtered_content = opa.check_context_transfer(
                    self.name, dependent.name, content=output
                )
                # Classify the transfer outcome for the audit log
                if not allowed:
                    transfer_status = "DENIED"
                elif censored_keywords:
                    transfer_status = "CONTENT_FILTERED"
                else:
                    transfer_status = "ALLOWED"
                log = {
                    "sender":   self.name,
                    "receiver": dependent.name,
                    "status":   transfer_status,
                    "reason":   reason,
                    "keywords": list(censored_keywords),
                }
                Agent.transfer_logs.append(log)

                if not allowed:
                    # P_tran denied: contribute nothing to the dependent's PromptContext
                    print(Fore.RED +
                          f"\n🚫 OPA PATH DENIED: {self.name} → {dependent.name}"
                          f"\n   reason: {reason}" + Style.RESET_ALL)
                    continue

                if censored_keywords:
                    print(Fore.YELLOW +
                          f"\n🔶 OPA CONTENT FILTERED: {self.name} → {dependent.name}"
                          f"\n   censored keywords: {censored_keywords}" + Style.RESET_ALL)
                else:
                    print(Fore.GREEN +
                          f"\n✅ OPA ALLOWED TRANSFER: {self.name} → {dependent.name}" + Style.RESET_ALL)
            else:
                # Baseline mode: bypass OPA, forward raw output unchanged
                print(Fore.YELLOW +
                      f"\n⚠️  OPA BYPASSED: {self.name} → {dependent.name}" + Style.RESET_ALL)
                filtered_content = output

            # Inject into dependent's PromptContext with a sender header for traceability
            dependent.receive_context(f"[From {self.name}]:\n{filtered_content}\n\n")

        return output

    def rollback(self):
        """Compensating transaction — override in subclasses for domain-specific undo logic."""
        print(f"🔄 Rolling back {self.name}...")
