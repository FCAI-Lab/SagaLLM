------------------------- MODULE bench_apa_fanout_12 -------------------------
EXTENDS sagallm_apalache

\* @type: Set(Str);
const_Agents == {"a0","a1","a2","a3","a4","a5","a6",
                 "a7","a8","a9","a10","a11","a12"}

\* @type: Set(Str);
const_Sensitive == {"a0"}

\* @type: Set(<<Str, Str>>);
const_E == {
    <<"a0","a1">>,  <<"a0","a2">>,  <<"a0","a3">>,
    <<"a0","a4">>,  <<"a0","a5">>,  <<"a0","a6">>,
    <<"a0","a7">>,  <<"a0","a8">>,  <<"a0","a9">>,
    <<"a0","a10">>, <<"a0","a11">>, <<"a0","a12">>
}

\* @type: Set(<<Str, Str>>);
const_P == {
    <<"a0","a1">>, <<"a0","a2">>, <<"a0","a3">>,
    <<"a0","a4">>, <<"a0","a5">>, <<"a0","a6">>
}

\* @type: Set(Str);
const_K == {"secret"}

\* @type: Str -> Set(Str);
const_Perms == [a \in const_Agents |->
    CASE a \in {"a1","a2","a3","a4","a5","a6"} -> {"secret"}
    [] OTHER                                   -> {}]

\* @type: Str -> Set(Str);
const_Outputs == [a \in const_Agents |->
    CASE a = "a0" -> {"secret"}
    [] OTHER      -> {"normal"}]
=============================================================================
