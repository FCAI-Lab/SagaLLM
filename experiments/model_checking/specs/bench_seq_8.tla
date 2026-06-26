------------------------- MODULE bench_seq_8 -------------------------
EXTENDS sagallm_content_logic

\* Sequential topology, 8 agents: a1 -> a2 -> ... -> a8
\* Sensitive: {a1}, P = E (full policy), K = {"secret"}, full keyword perms

const_Agents    == {"a1","a2","a3","a4","a5","a6","a7","a8"}
const_Sensitive == {"a1"}
const_E == {
    <<"a1","a2">>, <<"a2","a3">>, <<"a3","a4">>,
    <<"a4","a5">>, <<"a5","a6">>, <<"a6","a7">>, <<"a7","a8">>
}
const_P         == const_E
const_K         == {"secret"}
const_Perms     == [a \in const_Agents |->
                    CASE a = "a1" -> {"secret"}
                    [] a = "a2" -> {"secret"}
                    [] a = "a3" -> {"secret"}
                    [] a = "a4" -> {"secret"}
                    [] a = "a5" -> {"secret"}
                    [] a = "a6" -> {"secret"}
                    [] a = "a7" -> {"secret"}
                    [] OTHER    -> {"secret"}]
const_Outputs   == [a \in const_Agents |->
                    CASE a = "a1" -> {"secret"}
                    [] OTHER      -> {"normal"}]
=============================================================================
