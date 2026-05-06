"""LLM client wrapper using LangChain. Supports any OpenAI-compatible API."""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Iterator

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser

from .config import LLMConfig

logger = logging.getLogger(__name__)


class LLMClient:
    """LangChain-based wrapper for OpenAI-compatible chat completion API."""

    def __init__(self, config: LLMConfig):
        self.config = config
        self.llm = ChatOpenAI(
            model=config.model,
            api_key=config.api_key,
            base_url=config.base_url,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            timeout=config.timeout,
            max_retries=3,
        )
        self.parser = StrOutputParser()

    def generate(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Generate text from messages. Returns content string."""
        lc_messages = _to_langchain_messages(messages)

        llm = self.llm
        if temperature is not None or max_tokens is not None:
            overrides = {}
            if temperature is not None:
                overrides["temperature"] = temperature
            if max_tokens is not None:
                overrides["max_tokens"] = max_tokens
            llm = llm.bind(**overrides) if not overrides else self.llm.model_copy(
                update=overrides
            )

        chain = llm | self.parser
        return chain.invoke(lc_messages)

    def generate_stream(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> Iterator[str]:
        """Stream text generation, yielding content chunks."""
        lc_messages = _to_langchain_messages(messages)

        llm = self.llm
        if temperature is not None:
            llm = llm.model_copy(update={"temperature": temperature})
        if max_tokens is not None:
            llm = llm.model_copy(update={"max_tokens": max_tokens})

        chain = llm | self.parser
        for chunk in chain.stream(lc_messages):
            if chunk:
                yield chunk


def _to_langchain_messages(messages: list[dict[str, str]]) -> list:
    """Convert dict-style messages to LangChain message objects."""
    lc_messages = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role == "system":
            lc_messages.append(SystemMessage(content=content))
        else:
            lc_messages.append(HumanMessage(content=content))
    return lc_messages


def extract_json_from_response(text: str) -> dict | None:
    """Best-effort JSON extraction from LLM response."""
    # Try direct parse
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        pass

    # Try extracting from ```json ... ``` code blocks
    match = re.search(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            pass

    # Try repairing common issues (trailing commas)
    cleaned = re.sub(r",\s*([}\]])", r"\1", text)
    try:
        return json.loads(cleaned)
    except (json.JSONDecodeError, TypeError):
        pass

    # Try finding the first { ... } block
    brace_match = re.search(r"\{.*\}", text, re.DOTALL)
    if brace_match:
        try:
            return json.loads(brace_match.group(0))
        except json.JSONDecodeError:
            cleaned = re.sub(r",\s*([}\]])", r"\1", brace_match.group(0))
            try:
                return json.loads(cleaned)
            except json.JSONDecodeError:
                pass

    return None
