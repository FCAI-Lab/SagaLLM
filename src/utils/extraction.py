"""
extraction.py — XML Tag Content Extractor
==========================================
Extracts text between XML-style tags from LLM outputs.

Used by ReactAgent to parse <response>, <thought>, and <tool_call> blocks
from raw LLM completion strings.
"""

import re
from dataclasses import dataclass


@dataclass
class TagContentResult:
    content: list[str]   # all matched strings, stripped of leading/trailing whitespace
    found: bool          # True if at least one match was found


def extract_tag_content(text: str, tag: str) -> TagContentResult:
    """
    Find all occurrences of <tag>...</tag> in text (including multi-line content).

    Returns a TagContentResult with every captured group, in order.
    """
    tag_pattern = rf"<{tag}>(.*?)</{tag}>"
    matched_contents = re.findall(tag_pattern, text, re.DOTALL)
    return TagContentResult(
        content=[content.strip() for content in matched_contents],
        found=bool(matched_contents),
    )
