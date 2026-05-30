"""
opa_client.py — OPA REST API Client
=====================================
HTTP client for the Open Policy Agent (OPA) server that enforces
the two-layer access control policy P = (P_tran, P_cont).

A single POST to the transfer_control endpoint atomically evaluates:
  - P_tran: whether the (sender, receiver) path is authorized.
  - P_cont: which lines of the content contain restricted keywords for
            this receiver, and returns a redacted version.

Fail-closed design: any network error (connection refused, timeout) causes
check_context_transfer to return (allowed=False, ...) so the transfer is
denied rather than bypassing policy on transient failures.

A module-level singleton `opa` is created at import time for convenience.
"""

import requests
from colorama import Fore

OPA_BASE_URL    = "http://localhost:8181"
TRANSFER_PATH   = "v1/data/sagallm/access_control"


class OPAClient:
    def __init__(self, base_url: str = OPA_BASE_URL, transfer_path: str = TRANSFER_PATH):
        self.base_url      = base_url
        self.transfer_path = transfer_path
        self._session      = requests.Session()   # reuse TCP connection across calls

    def _query(self, policy_path: str, input_data: dict) -> tuple[bool, str, set, str]:
        """
        POST input_data to OPA and unpack the result fields.

        Returns:
            allowed          (bool)  — P_tran decision
            reason           (str)   — human-readable denial reason (empty if allowed)
            keywords         (set)   — sensitive keywords found and redacted
            filtered_content (str)   — content with restricted lines replaced by [Censored]
        """
        url = f"{self.base_url}/{policy_path}"
        try:
            response = self._session.post(url, json={"input": input_data}, timeout=2.0)
            response.raise_for_status()
            result           = response.json().get("result", {})
            allowed          = result.get("allow_transfer", False)
            reason           = result.get("reason", "Denied by policy") if not allowed else ""
            keywords         = set(result.get("matching_keywords", []))
            filtered_content = result.get("filtered_content", input_data.get("content", ""))
            return bool(allowed), reason, keywords, filtered_content
        except requests.exceptions.ConnectionError:
            msg = "OPA server unreachable — failing closed"
            print(Fore.YELLOW + f"⚠️  {msg}")
            return False, msg, set(), input_data.get("content", "")
        except requests.exceptions.Timeout:
            msg = "OPA request timed out — failing closed"
            print(Fore.YELLOW + f"⚠️  {msg}")
            return False, msg, set(), input_data.get("content", "")

    def check_context_transfer(
        self, sender: str, receiver: str, content: str = ""
    ) -> tuple[bool, str, set, str]:
        """
        Evaluate P_tran + P_cont for a proposed context transfer.

        Args:
            sender:   name of the sending agent
            receiver: name of the receiving agent
            content:  raw LLM output to be (possibly) forwarded

        Returns:
            (allowed, reason, censored_keywords, filtered_content)
        """
        payload: dict = {"sender": sender, "receiver": receiver}
        if content:
            payload["content"] = content
        return self._query(self.transfer_path, payload)


# Module-level singleton — imported directly by Agent.run()
opa = OPAClient()
