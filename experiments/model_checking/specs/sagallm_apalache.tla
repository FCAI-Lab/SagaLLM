----------------------- MODULE sagallm_apalache -----------------------
\* Apalache-compatible version of sagallm_content_logic
\* Type annotations added for Snowcat type checker.
EXTENDS Naturals, Sequences, FiniteSets

CONSTANTS
    \* @type: Set(Str);
    Agents,
    \* @type: Set(Str);
    SensitiveAgents,
    \* @type: Set(<<Str, Str>>);
    E,
    \* @type: Set(<<Str, Str>>);
    P,
    \* @type: Set(Str);
    SensitiveKeywords,
    \* @type: Str -> Set(Str);
    KeywordPermissions,
    \* @type: Str -> Set(Str);
    AgentOutputKeywords

VARIABLES
    \* @type: Str -> Str;
    agentStatus,
    \* @type: Str -> Set(Str);
    agentOutput,
    \* @type: Str -> Set(Str);
    promptContext,
    \* @type: Seq(Str);
    executionOrder

vars == <<agentStatus, agentOutput, promptContext, executionOrder>>

Upstream(agent) == {src \in Agents : <<src, agent>> \in E}
OPAAllows(src, dst) == <<src, dst>> \in P

SecretKeywords == UNION {AgentOutputKeywords[a] : a \in SensitiveAgents}

ContentFiltered(src, dst) ==
    {kw \in agentOutput[src] :
        \/ kw \notin SensitiveKeywords
        \/ kw \in KeywordPermissions[dst]}

Init ==
    /\ agentStatus   = [a \in Agents |-> "waiting"]
    /\ agentOutput   = [a \in Agents |-> {}]
    /\ promptContext = [a \in Agents |-> {}]
    /\ executionOrder = <<>>

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

SucceedAgent(agent) ==
    /\ agentStatus[agent] = "running"
    /\ agentStatus' = [agentStatus EXCEPT ![agent] = "done"]
    /\ agentOutput' = [agentOutput EXCEPT ![agent] = AgentOutputKeywords[agent]]
    /\ executionOrder' = Append(executionOrder, agent)
    /\ UNCHANGED <<promptContext>>

FailAgent(agent) ==
    /\ agentStatus[agent] = "running"
    /\ agentStatus' = [agentStatus EXCEPT ![agent] = "failed"]
    /\ UNCHANGED <<agentOutput, promptContext, executionOrder>>

CompStageAgent(agent) ==
    /\ agentStatus[agent] = "done"
    /\ \E f \in Agents : agentStatus[f] = "failed"
    /\ agentStatus' = [agentStatus EXCEPT ![agent] = "comp"]
    /\ UNCHANGED <<agentOutput, promptContext, executionOrder>>

CompensateAgent(agent) ==
    /\ agentStatus[agent] = "comp"
    /\ agentStatus' = [agentStatus EXCEPT ![agent] = "compensated"]
    /\ UNCHANGED <<agentOutput, promptContext, executionOrder>>

IsTerminal ==
    \/ \A a \in Agents : agentStatus[a] = "done"
    \/ \A a \in Agents : agentStatus[a] \in {"waiting", "failed", "compensated"}

Terminating ==
    /\ IsTerminal
    /\ UNCHANGED vars

Next ==
    \/ \E a \in Agents :
        \/ StartAgent(a)
        \/ SucceedAgent(a)
        \/ FailAgent(a)
        \/ CompStageAgent(a)
        \/ CompensateAgent(a)
    \/ Terminating

Spec == Init /\ [][Next]_vars /\ WF_vars(Next)

\* Safety invariants
DataIsolation ==
    \A src \in SensitiveAgents :
        \A dst \in Agents :
            \A kw \in AgentOutputKeywords[src] :
                kw \in promptContext[dst] => <<src, dst>> \in P

ContentDataIsolation ==
    \A a \in Agents :
        \A kw \in SecretKeywords :
            kw \in promptContext[a] => kw \in KeywordPermissions[a]

=============================================================================
