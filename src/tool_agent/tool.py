"""
tool.py — Tool Wrapper for ReactAgent
======================================
Provides the @tool decorator and Tool class used by ReactAgent to expose
Python functions as callable actions within the ReAct loop.

Usage:
    @tool
    def search_web(query: str) -> str:
        "Search the web and return results."
        ...

    agent = ReactAgent(tools=[search_web], ...)

The decorator extracts the function signature via get_fn_signature(), which
produces a JSON schema consumed by the LLM as the tool description.
validate_arguments() coerces string-typed LLM outputs to the correct Python types.
"""

import json
from typing import Callable


def get_fn_signature(fn: Callable) -> dict:
    """
    Build a JSON-serialisable tool schema from a function's annotations.

    The resulting dict matches the structure expected by the ReAct system prompt
    and is also used by validate_arguments() for type coercion.
    """
    fn_signature: dict = {
        "name": fn.__name__,
        "description": fn.__doc__,
        "parameters": {"properties": {}},
    }
    schema = {
        k: {"type": v.__name__} for k, v in fn.__annotations__.items() if k != "return"
    }
    fn_signature["parameters"]["properties"] = schema
    return fn_signature


def validate_arguments(tool_call: dict, tool_signature: dict) -> dict:
    """
    Coerce LLM-supplied argument values to the types declared in the signature.

    LLMs frequently return numbers as strings; this ensures the underlying
    Python function receives correctly typed arguments.
    """
    properties = tool_signature["parameters"]["properties"]
    type_mapping = {"int": int, "str": str, "bool": bool, "float": float}

    for arg_name, arg_value in tool_call["arguments"].items():
        expected_type = properties[arg_name].get("type")
        if not isinstance(arg_value, type_mapping[expected_type]):
            tool_call["arguments"][arg_name] = type_mapping[expected_type](arg_value)

    return tool_call


class Tool:
    """Wrapper that pairs a callable with its JSON schema string."""

    def __init__(self, name: str, fn: Callable, fn_signature: str):
        self.name = name
        self.fn = fn
        self.fn_signature = fn_signature   # JSON string included in the system prompt

    def __str__(self):
        return self.fn_signature

    def run(self, **kwargs):
        return self.fn(**kwargs)


def tool(fn: Callable):
    """Decorator that converts a plain function into a Tool instance."""
    def wrapper():
        fn_signature = get_fn_signature(fn)
        return Tool(
            name=fn_signature.get("name"), fn=fn, fn_signature=json.dumps(fn_signature)
        )
    return wrapper()
