"""
saga.py — Saga Transaction Coordinator
=======================================
Implements the Saga pattern for multi-agent LLM pipelines.

The Saga class acts as the central orchestrator (Guard Coordinator in the paper):
  1. transaction_manager(): registers agents and resets shared transfer logs.
  2. saga_coordinator():    executes agents in topological dependency order.
                            On any failure, reverses completed agents in LIFO order
                            (compensating transactions).
  3. topological_sort():    Kahn's algorithm — derives execution order from the
                            agent dependency graph declared via the >> operator.
  4. report_*():            Diagnostic helpers that print OPA transfer decisions,
                            D_actual = E ∩ P comparisons, and sensitivity metrics.

State machine per agent (matches §3.2 of the paper):
    waiting → exec → done          (happy path)
    exec    → failed → compensated (rollback path)
"""

from collections import deque
from colorama import Fore, Style


class Saga:
    def __init__(self):
        self.agents = []
        self.context: dict = {}   # raw agent outputs keyed by agent name (SagaContext)
        self.enforce_opa: bool = True

    # ── Registration ──────────────────────────────────────────────────────────

    def transaction_manager(self, agents):
        """Register the agent list for this transaction and clear prior logs."""
        self.agents = agents
        from multi_agent.agent import Agent
        Agent.transfer_logs.clear()
        print(Fore.CYAN + f"{'='*55}")
        print(f"🛠  Transaction Manager: {len(agents)} agent(s) registered")
        print(f"{'='*55}" + Style.RESET_ALL)

    # ── Core execution loop ────────────────────────────────────────────────────

    def saga_coordinator(self, with_rollback=True):
        """
        Pure Saga transactional state machine:
          waiting → exec → done                (success path)
          exec → failed → compensated          (rollback path)

        Agents execute in topological order so dependencies always finish first.
        Each agent's raw output is stored in self.context for potential rollback.
        OPA enforcement is delegated to Agent.run(enforce_opa=...).
        """
        sorted_agents = self.topological_sort()
        agent_state: dict = {a.name: "waiting" for a in sorted_agents}
        executed = []

        def _set_state(agent, state: str):
            agent_state[agent.name] = state
            state_colors = {
                "exec":        Fore.CYAN,
                "done":        Fore.GREEN,
                "failed":      Fore.RED,
                "compensated": Fore.YELLOW,
            }
            color = state_colors.get(state, "")
            print(color + f"  [state] {agent.name}: {state}" + Style.RESET_ALL)

        try:
            for agent in sorted_agents:
                print(Fore.CYAN + f"\n{'='*55}")
                print(f"🚀 Running: {agent.name}")
                print(f"{'='*55}" + Style.RESET_ALL)

                _set_state(agent, "exec")
                result = agent.run(enforce_opa=self.enforce_opa)
                self.context[agent.name] = result   # persist raw output for compensation
                executed.append(agent)
                _set_state(agent, "done")
                print(Fore.GREEN + f"✅ {agent.name} completed." + Style.RESET_ALL)

        except Exception as e:
            print(Fore.RED + f"❌ ERROR: {e}" + Style.RESET_ALL)
            failed_agent = executed[-1] if executed else None
            if failed_agent:
                _set_state(failed_agent, "failed")

            if with_rollback:
                # Compensate in reverse order (LIFO), skipping the failed agent itself
                print(Fore.YELLOW + f"[rollback] compensating..." + Style.RESET_ALL)
                for done in reversed(executed[:-1] if failed_agent else executed):
                    _set_state(done, "comp")
                    try:
                        done.rollback()
                        _set_state(done, "compensated")
                    except Exception as re:
                        print(Fore.YELLOW + f"⚠️  Rollback failed for {done.name}: {re}" + Style.RESET_ALL)
                print(Fore.YELLOW + f"[rollback] compensated" + Style.RESET_ALL)

    # ── Diagnostic / reporting helpers ────────────────────────────────────────

    def intra_agent(self):
        """Print each agent's expected output vs actual LLM output (quality check)."""
        print(f"\n{'='*65}")
        print("📌 Intra-Agent: expected output vs actual output")
        print(f"{'='*65}")
        for agent in self.agents:
            actual = self.context.get(agent.name, "(not executed)").strip()
            print(f"\n🔹 {agent.name}")
            print(f"   [task_expected_output]")
            for line in agent.task_expected_output.strip().splitlines():
                print(f"     {line}")
            print(f"   [actual LLM output]")
            preview = actual
            for line in preview.splitlines():
                print(f"     {line}")

    def inter_agent(self):
        """Print the declared dependency graph (edges declared with >>)."""
        print(f"\n{'='*55}")
        print("🔗 Inter-Agent: structural dependencies (>>)")
        print(f"{'='*55}")
        for agent in self.agents:
            deps = [d.name for d in agent.dependencies]
            print(f"🔸 {agent.name}  ←depends on→  {', '.join(deps) if deps else 'None'}")

    def report_transfers(self):
        """
        Print the OPA transfer log for every (sender, receiver) pair.

        Status values:
          ALLOWED          — path and content both passed
          CONTENT_FILTERED — path allowed but sensitive keywords were redacted
          DENIED           — path blocked by P_tran
        """
        from multi_agent.agent import Agent

        print(f"\n{Style.BRIGHT}{Fore.MAGENTA}{'='*65}")
        print(f"OPA Transfer Control Report")
        print(f"{'='*65}{Style.RESET_ALL}")
        print(f"  {'Sender':<30} {'Receiver':<30} Result")
        print(f"  {'-'*63}")

        for log in Agent.transfer_logs:
            if log["status"] == "ALLOWED":
                status = Fore.GREEN + "✅ ALLOWED" + Style.RESET_ALL
            elif log["status"] == "CONTENT_FILTERED":
                kws = ", ".join(log["keywords"]) if log["keywords"] else ""
                status = Fore.YELLOW + f"🔶 CONTENT_FILTERED  [{kws}]" + Style.RESET_ALL
            else:
                status = Fore.RED + f"🚫 DENIED  ({log['reason']})" + Style.RESET_ALL
            print(f"  {log['sender']:<30} {log['receiver']:<30} {status}")

        print(f"{Fore.MAGENTA}{'='*65}{Style.RESET_ALL}\n")

    def report_dependency_vs_transfer(self):
        """
        Visualize D_actual = E ∩ P for all declared dependency edges.

        E        = declared dependency (>> operator)
        P        = OPA-allowed path (P_tran decision)
        D_actual = data actually forwarded (intersection of E and P)
        """
        from multi_agent.agent import Agent

        allowed_set = {
            (log["sender"], log["receiver"])
            for log in Agent.transfer_logs
            if log["status"] in ("ALLOWED", "CONTENT_FILTERED")
        }

        declared_pairs = []
        for agent in self.agents:
            for dep in agent.dependencies:
                declared_pairs.append((dep.name, agent.name))

        sep = "=" * 75
        print(f"\n{Style.BRIGHT}{Fore.CYAN}{sep}")
        print(f"  D_actual = E ∩ P")
        print(f"{sep}{Style.RESET_ALL}")
        print(f"  {'Sender':<30} {'Receiver':<28} {'E':^6} {'P':^6} {'D_actual':^10}")
        print(f"  {'-'*73}")

        if not declared_pairs:
            print(f"  {Fore.YELLOW}No declared dependency edges were found.{Style.RESET_ALL}")

        for sender, receiver in declared_pairs:
            in_e = True   # edge is declared, so always True
            in_p = (sender, receiver) in allowed_set
            d_actual = in_e and in_p

            e_mark  = Fore.GREEN + "  ✅" + Style.RESET_ALL
            p_mark  = Fore.GREEN + "  ✅" + Style.RESET_ALL if in_p  else Fore.RED + "  ❌" + Style.RESET_ALL
            d_mark  = Fore.GREEN + "  ✅" + Style.RESET_ALL if d_actual else Fore.RED + "  ❌" + Style.RESET_ALL

            print(f"  {sender:<30} {receiver:<28}{e_mark}{p_mark}{d_mark}")

        print(f"\n  Declared dependency edges: {len(declared_pairs)}")
        print(f"{Fore.CYAN}{sep}{Style.RESET_ALL}\n")

    def reset(self):
        """Clear all agent contexts and transfer logs between runs."""
        from multi_agent.agent import Agent
        for agent in self.agents:
            agent.context = ""
        self.context = {}
        Agent.transfer_logs.clear()

    def compare_agent_output(self, agent_name: str, unblocked_senders: list[str]):
        """Re-run a single agent with additional (normally blocked) sender context to
        demonstrate the impact of OPA enforcement (counterfactual comparison)."""
        agent = next((a for a in self.agents if a.name == agent_name), None)
        if not agent:
            print(Fore.RED + f"Agent '{agent_name}' not found." + Style.RESET_ALL)
            return

        with_opa_output = self.context.get(agent_name, "(not executed)").strip()

        original_context = agent.context
        extra = ""
        for sender_name in unblocked_senders:
            sender_output = self.context.get(sender_name, "")
            if sender_output:
                extra += f"[From {sender_name}]:\n{sender_output}\n\n"

        agent.context = original_context + extra
        print(Fore.YELLOW + f"\n🔄 Re-running {agent_name} without OPA restriction..." + Style.RESET_ALL)
        without_opa_output = agent.react_agent.run(user_msg=agent.create_prompt())
        agent.context = original_context   # restore

        sep = "=" * 70
        print(f"\n{Style.BRIGHT}{Fore.CYAN}{sep}")
        print(f"  {agent_name} ouput")
        print(f"  additionally received context: {', '.join(unblocked_senders)}")
        print(f"{sep}{Style.RESET_ALL}")

        print(f"\n{Fore.RED}[ WITHOUT OPA ] Task Output{Style.RESET_ALL}")
        for line in without_opa_output.strip().splitlines():
            print(f"    {line}")

        print(f"\n{Fore.GREEN}[ WITH OPA ] Task Output{Style.RESET_ALL}")
        for line in with_opa_output.strip().splitlines():
            print(f"    {line}")

        print(f"{Fore.CYAN}{sep}{Style.RESET_ALL}\n")

    def report_output_comparison(self, without_opa: dict, with_opa: dict):
        """Side-by-side output diff: two pre-collected result dicts (without vs with OPA)."""
        sep = "=" * 70
        print(f"\n{Style.BRIGHT}{Fore.CYAN}{sep}")
        print(f"{sep}{Style.RESET_ALL}")

        for agent in self.agents:
            name = agent.name
            out_before = without_opa.get(name, "(not executed)").strip()
            out_after  = with_opa.get(name, "(not executed)").strip()

            print(f"\n{Style.BRIGHT}{'─'*70}")
            print(f"  {name}")
            print(f"{'─'*70}{Style.RESET_ALL}")

            changed = out_before != out_after

            print(f"\n  {Fore.RED}[ WITHOUT OPA ]{Style.RESET_ALL}")
            preview = out_before.replace('\n', '\n    ')
            print(f"    {preview}")

            print(f"\n  {Fore.GREEN}[ WITH OPA ]{Style.RESET_ALL}")
            preview2 = out_after.replace('\n', '\n    ')
            print(f"    {preview2}")

            if changed:
                print(f"\n  {Fore.YELLOW}  ⚠️ Content changed after filtering {Style.RESET_ALL}")
            else:
                print(f"\n  {Fore.GREEN}  ✅ Content not changed after filtering {Style.RESET_ALL}")

        print(f"\n{Fore.CYAN}{sep}{Style.RESET_ALL}\n")

    def report_all_agent_comparison(self):
        """For every agent, compare what it would have received (no OPA) vs what it
        actually received (with OPA) by inspecting the injected promptContext."""
        sep = "=" * 70
        print(f"\n{Style.BRIGHT}{Fore.CYAN}{sep}")
        print(f"  Compare context ")
        print(f"{sep}{Style.RESET_ALL}")

        for agent in self.agents:
            print(f"\n{Style.BRIGHT}{'─'*70}")
            print(f"  Agent: {agent.name}")
            print(f"{'─'*70}{Style.RESET_ALL}")

            without_opa_parts = []
            for dep in agent.dependencies:
                dep_output = self.context.get(dep.name, "")
                if dep_output:
                    without_opa_parts.append((dep.name, dep_output))

            with_opa = agent.context.strip()

            # Determine which senders actually contributed to the prompt context
            actually_received_names = set()
            for part_name, part_content in without_opa_parts:
                marker = f"[From {part_name}]"
                if marker in with_opa:
                    actually_received_names.add(part_name)

            blocked_names = {name for name, _ in without_opa_parts} - actually_received_names

            if without_opa_parts:
                print(f"\n  {Fore.RED}[ WITHOUT OPA ]  context that would have been received (based on >> declarations){Style.RESET_ALL}")
                for dep_name, dep_output in without_opa_parts:
                    tag = f"{Fore.RED}  🚫 BLOCKED{Style.RESET_ALL}" if dep_name in blocked_names else f"{Fore.GREEN}  ✅ RECEIVED{Style.RESET_ALL}"
                    preview = dep_output.strip().replace('\n', '\n      ')
                    print(f"    from {dep_name}: {tag}")
                    print(f"      {preview}")
            else:
                print(f"\n  {Fore.RED}[ WITHOUT OPA ]  No dependency(Initial Agent) {Style.RESET_ALL}")

            print(f"\n  {Fore.GREEN}[ WITH OPA ] {Style.RESET_ALL}")
            if with_opa:
                preview = with_opa.replace('\n', '\n      ')
                print(f"      {preview}")
            else:
                print(f"     ")

            if blocked_names:
                print(f"\n  {Fore.YELLOW}  → Blocked Transfer: {', '.join(blocked_names)}{Style.RESET_ALL}")
            else:
                print(f"\n  {Fore.GREEN}  → Allowed Transfer {Style.RESET_ALL}")

        print(f"\n{Fore.CYAN}{sep}{Style.RESET_ALL}\n")

    def report_opa_comparison(self, blocked_sender: str, blocked_receiver: str):
        """Show the raw sender output next to the actual filtered prompt context
        received by blocked_receiver — illustrates P_tran denial in action."""
        from colorama import Fore, Style

        receiver = next((a for a in self.agents if a.name == blocked_receiver), None)
        sender_output = self.context.get(blocked_sender, "(not executed)")

        sep = "=" * 65
        print(f"\n{Style.BRIGHT}{Fore.CYAN}{sep}")
        print(f"  OPA Evaluation Path : {blocked_sender} → {blocked_receiver}")
        print(f"{sep}{Style.RESET_ALL}")

        print(f"\n{Fore.RED}[ WITHOUT OPA ]  {blocked_receiver} Prompt Context{Style.RESET_ALL}")
        print(f"{Fore.RED}  {blocked_sender} Push result:{Style.RESET_ALL}")
        preview = sender_output.strip()
        for line in preview.splitlines():
            print(f"    {line}")

        print(f"\n{Fore.GREEN}[ WITH OPA ]  {blocked_receiver} Actual Prompt Context{Style.RESET_ALL}")
        if receiver:
            actual = receiver.context.strip()
            if actual:
                preview2 = actual
                for line in preview2.splitlines():
                    print(f"    {line}")
            else:
                print(f"    No pused data")

        print(f"\n{Fore.YELLOW}  → {blocked_sender} data is not pushed to {blocked_receiver} {Style.RESET_ALL}")
        print(f"  → {blocked_receiver} receives data from only allowed agents")
        print(f"{Fore.CYAN}{sep}{Style.RESET_ALL}\n")

    def report_sensitivity_metrics(
        self,
        sensitive_fields: list[str],
        target_agent: str,
        blocked_sender: str,
    ):
        """
        Compute precision/recall style sensitivity metrics for a single
        (blocked_sender → target_agent) pair.  Used in RQ3 experiments.

        Fields present in sender_output but absent in target's promptContext
        count as successfully blocked (true negatives for the attacker).
        """
        sep = "=" * 65
        print(f"\n{Style.BRIGHT}{Fore.CYAN}{sep}")
        print(f"  [Experiment B] Sensitivity Metrics")
        print(f"  target: {target_agent}  |  blocked sender: {blocked_sender}")
        print(f"{sep}{Style.RESET_ALL}")

        sender_output    = self.context.get(blocked_sender, "")
        target           = next((a for a in self.agents if a.name == target_agent), None)
        context_with_opa = target.context if target else ""

        found_without = [f for f in sensitive_fields if f in sender_output]
        found_with    = [f for f in sensitive_fields if f in context_with_opa]

        total = len(sensitive_fields)
        rate_without = len(found_without) / total * 100 if total else 0
        rate_with    = len(found_with)    / total * 100 if total else 0

        print(f"\n  {'Field':<30} {'Without OPA':^15} {'With OPA':^15}")
        print(f"  {'-'*60}")
        for f in sensitive_fields:
            wo = Fore.RED   + "  ✅ DETECTED" + Style.RESET_ALL if f in found_without else Fore.GREEN + "  ❌ NOT FOUND" + Style.RESET_ALL
            wi = Fore.RED   + "  ✅ DETECTED" + Style.RESET_ALL if f in found_with    else Fore.GREEN + "  ❌ NOT FOUND" + Style.RESET_ALL
            print(f"  {f:<30}{wo:^15}{wi:^15}")

        print(f"\n  {'-'*60}")
        print(f"  {'Detection Rate':<30} {rate_without:>6.1f}%         {rate_with:>6.1f}%")

        print(f"\n  Report:")
        if rate_with == 0:
            print(Fore.GREEN + f"  ✅ Sensitive Data is Blocked with OPA" + Style.RESET_ALL)
        else:
            print(Fore.RED + f"  ⚠️ {rate_with:.1f}% of Sensitive Data is leaked with OPA" + Style.RESET_ALL)

        print(f"{Fore.CYAN}{sep}{Style.RESET_ALL}\n")

    # ── Graph utility ─────────────────────────────────────────────────────────

    def topological_sort(self):
        """
        Kahn's BFS topological sort over the agent dependency graph.
        Raises ValueError if a cycle is detected (acyclicity is a precondition
        for the Saga pattern — also verified by the TLA+ ExecutionOrderPreserved invariant).
        """
        in_degree = {a: len(a.dependencies) for a in self.agents}
        queue = deque([a for a in self.agents if in_degree[a] == 0])
        result = []
        while queue:
            cur = queue.popleft()
            result.append(cur)
            for dep in cur.dependents:
                in_degree[dep] -= 1
                if in_degree[dep] == 0:
                    queue.append(dep)
        if len(result) != len(self.agents):
            raise ValueError("Circular dependency detected.")
        return result
