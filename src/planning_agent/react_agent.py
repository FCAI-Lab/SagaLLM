"""
react_agent.py — LLM Interface (ReAct Loop)
============================================
Wraps the OpenAI API with a lightweight ReAct (Reasoning + Acting) loop.

When tools are provided, the agent iterates up to max_rounds times:
  Thought → tool_call → observation → … → <response>

When no tools are supplied (the common case in SafeSagaLLM agents), a single
chat completion is issued and the response text is returned directly.

The system prompt is set per-agent (backstory) and merged with the ReAct
instruction block that describes available tools in XML format.
"""

import json
import os

from colorama import Fore
from dotenv import load_dotenv
from openai import OpenAI

from tool_agent.tool import Tool, validate_arguments
from utils.completions import build_prompt_structure, ChatHistory, completions_create, update_chat_history
from utils.extraction import extract_tag_content

load_dotenv()


def _make_client(model: str) -> tuple:
    """
    Create the appropriate API client based on model name prefix.

    Supported prefixes:
      - gpt-*         : OpenAI (default)
      - claude-*      : Anthropic  (requires ANTHROPIC_API_KEY)
      - gemini-*      : Google Gemini via OpenAI-compatible endpoint  (requires GEMINI_API_KEY)
      - deepseek-*    : DeepSeek via OpenAI-compatible endpoint  (requires DEEPSEEK_API_KEY)

    Returns (client, provider_str).
    """
    if model.startswith("claude-"):
        from anthropic import Anthropic
        return Anthropic(), "anthropic"

    if model.startswith("gemini-"):
        return (
            OpenAI(
                api_key=os.environ.get("GEMINI_API_KEY"),
                base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            ),
            "openai",
        )

    if model.startswith("deepseek-"):
        return (
            OpenAI(
                api_key=os.environ.get("DEEPSEEK_API_KEY"),
                base_url="https://api.deepseek.com",
            ),
            "openai",
        )

    # Default: OpenAI
    return OpenAI(), "openai"

BASE_SYSTEM_PROMPT = ""

# ReAct instruction appended when tools are available.
# The LLM is expected to emit <tool_call> blocks which are parsed and executed.
REACT_SYSTEM_PROMPT = """
You operate by running a loop with the following steps: Thought, Action, Observation.
You are provided with function signatures within <tools></tools> XML tags.
You may call one or more functions to assist with the user query. Don't make assumptions about what values to plug
into functions. Pay special attention to the properties 'types'. You should use those types as in a Python dict.

For each function call return a json object with function name and arguments within <tool_call></tool_call> XML tags as follows:

<tool_call>
{"name": <function-name>,"arguments": <args-dict>, "id": <monotonically-increasing-id>}
</tool_call>

Here are the available tools / actions:

<tools>
%s
</tools>

Example session:

<question>What's the current temperature in Madrid?</question>
<thought>I need to get the current weather in Madrid</thought>
<tool_call>{"name": "get_current_weather","arguments": {"location": "Madrid", "unit": "celsius"}, "id": 0}</tool_call>

You will be called again with this:

<observation>{0: {"temperature": 25, "unit": "celsius"}}</observation>

You then output:

<response>The current temperature in Madrid is 25 degrees Celsius</response>

Additional constraints:

- If the user asks you something unrelated to any of the tools above, answer freely enclosing your answer with <response></response> tags.
"""


class ReactAgent:
    """
    Thin LLM wrapper used by every Agent in the pipeline.

    - No tools: single chat completion, returns string response.
    - With tools: up to max_rounds of Thought/Action/Observation iterations
      until a <response> tag is emitted.
    """

    def __init__(
        self,
        tools: Tool | list[Tool],
        model: str = "gpt-4o",
        system_prompt: str = BASE_SYSTEM_PROMPT,
    ) -> None:
        self.client, self.provider = _make_client(model)
        self.model = model
        self.system_prompt = system_prompt
        self.tools = tools if isinstance(tools, list) else [tools]
        self.tools_dict = {tool.name: tool for tool in self.tools}

    def add_tool_signatures(self) -> str:
        """Concatenate all tool JSON schemas for injection into the system prompt."""
        return "".join([tool.fn_signature for tool in self.tools])

    def process_tool_calls(self, tool_calls_content: list) -> dict:
        """Parse, validate, and execute all tool calls from one LLM round."""
        observations = {}
        for tool_call_str in tool_calls_content:
            tool_call = json.loads(tool_call_str)
            tool_name = tool_call["name"]
            tool = self.tools_dict[tool_name]

            print(Fore.GREEN + f"\nUsing Tool: {tool_name}")
            validated_tool_call = validate_arguments(tool_call, json.loads(tool.fn_signature))
            print(Fore.GREEN + f"\nTool call dict: \n{validated_tool_call}")

            result = tool.run(**validated_tool_call["arguments"])
            print(Fore.GREEN + f"\nTool result: \n{result}")

            observations[validated_tool_call["id"]] = result
        return observations

    def run(self, user_msg: str, max_rounds: int = 5) -> str:
        """
        Run the agent for a single user message.

        Without tools: one LLM call, returns the completion string.
        With tools: iterate Thought/Action/Observation until <response> found
        or max_rounds is exhausted.
        """
        user_prompt = build_prompt_structure(prompt=user_msg, role="user", tag="question")

        system_prompt = self.system_prompt
        if self.tools:
            system_prompt += "\n" + REACT_SYSTEM_PROMPT % self.add_tool_signatures()

        chat_history = ChatHistory(
            [
                build_prompt_structure(prompt=system_prompt, role="system"),
                user_prompt,
            ]
        )

        if self.tools:
            for _ in range(max_rounds):
                completion = completions_create(self.client, chat_history, self.model, self.provider)

                # If the model emits a final <response>, we're done
                response = extract_tag_content(str(completion), "response")
                if response.found:
                    return response.content[0]

                thought = extract_tag_content(str(completion), "thought")
                tool_calls = extract_tag_content(str(completion), "tool_call")

                update_chat_history(chat_history, completion, "assistant")
                print(Fore.MAGENTA + f"\nThought: {thought.content[0]}")

                if tool_calls.found:
                    observations = self.process_tool_calls(tool_calls.content)
                    print(Fore.BLUE + f"\nObservations: {observations}")
                    update_chat_history(chat_history, f"{observations}", "user")

        # No tools (or no <response> found): return raw completion
        return completions_create(self.client, chat_history, self.model, self.provider)
