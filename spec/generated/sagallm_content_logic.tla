----------------------- MODULE sagallm_content_logic -----------------------
\* SagaLLM + OPA Content-Based Filtering Logic  (Pure Saga)
\*
\* agentOutput[a]  = raw output keyword set (may contain sensitive data; serves as saga.context for rollback)
\* promptContext[a] = filtered input keyword set passed to the LLM
\*
\* Transactional State Machine (Pure Saga):
\*   waiting -> running -> done                   (normal path, local commit)
\*   running -> failed -> comp -> compensated     (failure path, compensating transaction)
\*
\* Properties:
\*   DataIsolation        : secret keywords are absent unless explicitly permitted
\*   ContentDataIsolation : policy K filters unpermitted secret keywords

EXTENDS Naturals, Sequences, FiniteSets, TLC

CONSTANTS Agents, SensitiveAgents, E, P,
          SensitiveKeywords, KeywordPermissions, AgentOutputKeywords

VARIABLES
    agentStatus,
    agentOutput,
    promptContext,
    executionOrder

vars == <<agentStatus, agentOutput, promptContext, executionOrder>>

\* ─── Helper Operators ───
Upstream(agent) == {src \in Agents : <<src, agent>> \in E}
OPAAllows(src, dst) == <<src, dst>> \in P

\* Secret keywords are the true sensitive tokens the model should protect.
\* SensitiveKeywords is the Rego policy's known keyword set K. If a secret
\* keyword is absent from K, the policy cannot redact it and TLC can expose
\* that as a ContentDataIsolation counterexample.
SecretKeywords == UNION {AgentOutputKeywords[a] : a \in SensitiveAgents}

\* Rego P_cont whitelist semantics:
\*   - if kw is not in SensitiveKeywords, the policy does not recognize it and it passes
\*   - if kw is in SensitiveKeywords, it passes only when receiver is permitted
ContentFiltered(src, dst) ==
    {kw \in agentOutput[src] :
        \/ kw \notin SensitiveKeywords
        \/ kw \in KeywordPermissions[dst]}

\* ─── 1. Init ───
Init ==
    /\ agentStatus   = [a \in Agents |-> "waiting"]
    /\ agentOutput   = [a \in Agents |-> {}]
    /\ promptContext = [a \in Agents |-> {}]
    /\ executionOrder = <<>>

\* ─── 2. Actions ───

\* waiting -> running: start when all upstream dependencies are done
StartAgent(agent) ==
    /\ agentStatus[agent] = "waiting"
    /\ \A dep \in Upstream(agent) : agentStatus[dep] = "done"
    /\ \A a \in Agents : agentStatus[a] /= "failed"
    /\ agentStatus' = [agentStatus EXCEPT ![agent] = "running"]
    /\ promptContext' = [promptContext EXCEPT
        ![agent] = UNION {
            ContentFiltered(src, agent)
            : src \in {s \in Upstream(agent) : OPAAllows(s, agent)}
        }]
    /\ UNCHANGED <<agentOutput, executionOrder>>

\* running -> done: normal completion (local commit)
SucceedAgent(agent) ==
    /\ agentStatus[agent] = "running"
    /\ agentStatus' = [agentStatus EXCEPT ![agent] = "done"]
    /\ agentOutput' = [agentOutput EXCEPT ![agent] = AgentOutputKeywords[agent]]
    /\ executionOrder' = Append(executionOrder, agent)
    /\ UNCHANGED <<promptContext>>

\* running -> failed: execution failure
FailAgent(agent) ==
    /\ agentStatus[agent] = "running"
    /\ agentStatus' = [agentStatus EXCEPT ![agent] = "failed"]
    /\ UNCHANGED <<agentOutput, promptContext, executionOrder>>

\* done -> comp: prepare compensation when a failure has occurred (Saga compensation)
CompStageAgent(agent) ==
    /\ agentStatus[agent] = "done"
    /\ \E f \in Agents : agentStatus[f] = "failed"
    /\ agentStatus' = [agentStatus EXCEPT ![agent] = "comp"]
    /\ UNCHANGED <<agentOutput, promptContext, executionOrder>>

\* comp -> compensated: compensating transaction complete
CompensateAgent(agent) ==
    /\ agentStatus[agent] = "comp"
    /\ agentStatus' = [agentStatus EXCEPT ![agent] = "compensated"]
    /\ UNCHANGED <<agentOutput, promptContext, executionOrder>>

\* ─── 3. Next & Spec ───
Next == \E a \in Agents :
    \/ StartAgent(a)
    \/ SucceedAgent(a)
    \/ FailAgent(a)
    \/ CompStageAgent(a)
    \/ CompensateAgent(a)

Spec == Init /\ [][Next]_vars /\ WF_vars(Next)

\* ══════════════════════════════════════════════════════════════
\* Properties
\* ══════════════════════════════════════════════════════════════

\* [P1] DataIsolation
\*      Secret keywords may enter promptContext only through OPA-authorized paths.
DataIsolation ==
    \A src \in SensitiveAgents :
        \A dst \in Agents :
            \A kw \in AgentOutputKeywords[src] :
                kw \in promptContext[dst] => <<src, dst>> \in P

\* [P1b] ContentDataIsolation
\*       Any unpermitted secret keyword reaching a receiver means P_cont did not
\*       redact it. This usually indicates a missing SensitiveKeywords entry.
ContentDataIsolation ==
    \A a \in Agents :
        \A kw \in SecretKeywords :
            kw \in promptContext[a] => kw \in KeywordPermissions[a]

\* [P2] AtomicTermination
\*      Normal: all agents done / Failure: all agents in terminal state
AtomicTermination ==
    <>(\/ \A a \in Agents : agentStatus[a] = "done"
       \/ \A b \in Agents : agentStatus[b] \in {"waiting", "failed", "compensated"})

\* [P3] CompensationCompleteness
CompensationCompleteness ==
    []( (\E f \in Agents : agentStatus[f] = "failed")
        => <>(\A a \in Agents : agentStatus[a] \in {"waiting", "failed", "compensated"}) )

\* [P4] ExecutionOrderPreserved
ExecutionOrderPreserved ==
    \A src \in Agents : \A dst \in Agents :
        ( <<src, dst>> \in E
          /\ agentStatus[src] = "done"
          /\ agentStatus[dst] = "done" )
        => \E i \in 1..Len(executionOrder) : \E j \in 1..Len(executionOrder) :
            /\ executionOrder[i] = src
            /\ executionOrder[j] = dst
            /\ i < j
=============================================================================
