------------------------- MODULE bench_apa_12 -------------------------
EXTENDS sagallm_apalache

\* @type: Set(Str);
const_Agents == {"a1","a2","a3","a4","a5","a6",
                 "a7","a8","a9","a10","a11","a12"}

\* @type: Set(Str);
const_Sensitive == {"a1"}

\* @type: Set(<<Str, Str>>);
const_E == {
    <<"a1","a2">>,   <<"a2","a3">>,   <<"a3","a4">>,
    <<"a4","a5">>,   <<"a5","a6">>,   <<"a6","a7">>,
    <<"a7","a8">>,   <<"a8","a9">>,   <<"a9","a10">>,
    <<"a10","a11">>, <<"a11","a12">>
}

\* @type: Set(<<Str, Str>>);
const_P == const_E

\* @type: Set(Str);
const_K == {"secret"}

\* @type: Str -> Set(Str);
const_Perms == [a \in const_Agents |-> {"secret"}]

\* @type: Str -> Set(Str);
const_Outputs == [a \in const_Agents |->
                  CASE a = "a1" -> {"secret"}
                  [] OTHER      -> {"normal"}]
=============================================================================
