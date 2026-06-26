------------------------- MODULE bench_fanout_12 -------------------------
EXTENDS sagallm_content_logic

\* Fan-out topology, 13 agents total: a0 -> {a1..a12}
\* Sensitive: {a0}
\*
\* Policy (partial — realistic):
\*   P_tran: a0 authorized to send to {a1..a6} only
\*           {a7..a12} are in E but NOT in P  (blocked)
\*   P_cont: keyword "secret"; only authorized receivers have permission

const_Agents    == {"a0","a1","a2","a3","a4","a5","a6",
                    "a7","a8","a9","a10","a11","a12"}
const_Sensitive == {"a0"}

\* E: structural fan-out edges (a0 -> all 12 receivers)
const_E == {
    <<"a0","a1">>,  <<"a0","a2">>,  <<"a0","a3">>,
    <<"a0","a4">>,  <<"a0","a5">>,  <<"a0","a6">>,
    <<"a0","a7">>,  <<"a0","a8">>,  <<"a0","a9">>,
    <<"a0","a10">>, <<"a0","a11">>, <<"a0","a12">>
}

\* P_tran: only first 6 receivers authorized
const_P == {
    <<"a0","a1">>, <<"a0","a2">>, <<"a0","a3">>,
    <<"a0","a4">>, <<"a0","a5">>, <<"a0","a6">>
}

const_K == {"secret"}

\* P_cont: authorized receivers may see "secret"; unauthorized get empty set
const_Perms == [a \in const_Agents |->
    CASE a \in {"a1","a2","a3","a4","a5","a6"} -> {"secret"}
    [] OTHER                                   -> {}]

const_Outputs == [a \in const_Agents |->
    CASE a = "a0" -> {"secret"}
    [] OTHER      -> {"normal"}]
=============================================================================
