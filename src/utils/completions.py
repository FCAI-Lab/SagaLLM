"""
completions.py — OpenAI Chat Completion Helpers
================================================
Thin wrappers around the OpenAI chat completions API used by ReactAgent.

ChatHistory extends list with an optional bounded-window (total_length):
  - When the window is full, the oldest message is evicted on each append.
  - FixedFirstChatHistory preserves the system prompt at index 0 and
    evicts the second-oldest message instead.
"""


def completions_create(client, messages: list, model: str, provider: str = "openai") -> str:
    """
    Call the appropriate completions endpoint and return the response text.

    provider:
      "anthropic" — uses Anthropic Messages API (system prompt extracted separately)
      "openai"    — uses OpenAI-compatible chat completions API (OpenAI, Gemini, DeepSeek)
    """
    if provider == "anthropic":
        # Anthropic requires system prompt to be passed as a separate parameter,
        # not inside the messages list.
        system_parts = [m["content"] for m in messages if m["role"] == "system"]
        chat_msgs    = [m for m in messages if m["role"] != "system"]
        response = client.messages.create(
            model=model,
            system=system_parts[0] if system_parts else "",
            messages=chat_msgs,
            temperature=0.3,
            max_tokens=3000,
        )
        return response.content[0].text

    # OpenAI / OpenAI-compatible (Gemini, DeepSeek)
    response = client.chat.completions.create(
        messages=messages, model=model, temperature=0.3, max_tokens=3000
    )
    return str(response.choices[0].message.content)


def build_prompt_structure(prompt: str, role: str, tag: str = "") -> dict:
    """Wrap prompt text in an optional XML tag and return an OpenAI message dict."""
    if tag:
        prompt = f"<{tag}>{prompt}</{tag}>"
    return {"role": role, "content": prompt}


def update_chat_history(history: list, msg: str, role: str):
    """Append a new message to the chat history."""
    history.append(build_prompt_structure(prompt=msg, role=role))


class ChatHistory(list):
    """Bounded chat history list. Evicts oldest message when capacity is reached."""

    def __init__(self, messages: list | None = None, total_length: int = -1):
        if messages is None:
            messages = []
        super().__init__(messages)
        self.total_length = total_length

    def append(self, msg: str):
        if len(self) == self.total_length:
            self.pop(0)   # evict oldest
        super().append(msg)


class FixedFirstChatHistory(ChatHistory):
    """Chat history that always keeps the first message (system prompt) intact."""

    def append(self, msg: str):
        if len(self) == self.total_length:
            self.pop(1)   # evict second-oldest, preserving index 0
        super().append(msg)
