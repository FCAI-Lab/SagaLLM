------------------------- MODULE bench_seq_12 -------------------------
EXTENDS sagallm_content_logic

\* Sequential topology, 12 agents: a1 -> a2 -> ... -> a12
\* Sensitive: {a1}, P = E (full policy)

const_Agents    == {"a1","a2","a3","a4","a5","a6",
                    "a7","a8","a9","a10","a11","a12"}
const_Sensitive == {"a1"}
const_E == {
    <<"a1","a2">>,   <<"a2","a3">>,   <<"a3","a4">>,
    <<"a4","a5">>,   <<"a5","a6">>,   <<"a6","a7">>,
    <<"a7","a8">>,   <<"a8","a9">>,   <<"a9","a10">>,
    <<"a10","a11">>, <<"a11","a12">>
}
const_P       == const_E
const_K       == {"secret"}
const_Perms   == [a \in const_Agents |-> {"secret"}]
const_Outputs == [a \in const_Agents |->
                  CASE a = "a1" -> {"secret"}
                  [] OTHER      -> {"normal"}]
=============================================================================
