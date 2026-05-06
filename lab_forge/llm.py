"""
Legacy OpenAI-compatible LLM client wrapper.

New code should use lab_forge.models.create_chat_model() which returns a LangChain
ChatOpenAI instance with built-in retry logic. This module is kept for compatibility
with older tests.
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    InternalServerError,
    OpenAI,
    RateLimitError,
)

from .config import LLMConfig

logger = logging.getLogger(__name__)

DEFAULT_MAX_RETRIES = 3
RETRYABLE_STATUS_CODES = {408, 409, 429, 500, 502, 503, 504, 529}


class LLMResponseError(RuntimeError):
    """Raised when an OpenAI-compatible backend returns an unusable response."""


class LLMClient:
    """Thin wrapper around OpenAI-compatible chat completion API."""

    def __init__(self, config: LLMConfig):
        self.config = config
        self.client = OpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
            timeout=config.timeout,
        )

    def chat(
        self,
        messages: list[dict[str, str]],
        tools: list[dict] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        """
        Send a chat completion request.

        Returns:
            dict with keys:
                - "content": str | None (text response)
                - "tool_calls": list[dict] | None (tool call requests)
                - "usage": dict (token usage)
        """
        kwargs: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "temperature": temperature if temperature is not None else self.config.temperature,
            "max_tokens": max_tokens or self.config.max_tokens,
            "top_p": self.config.top_p,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        logger.debug("LLM request: model=%s, messages=%d", self.config.model, len(messages))

        last_error: Exception | None = None
        for attempt in range(1, DEFAULT_MAX_RETRIES + 1):
            try:
                response = self.client.chat.completions.create(**kwargs)
                result = self._parse_response(response)
                logger.debug(
                    "LLM response: content_len=%d, tool_calls=%s",
                    len(result["content"] or ""),
                    len(result["tool_calls"]) if result["tool_calls"] else 0,
                )
                return result
            except (
                APIConnectionError,
                APITimeoutError,
                InternalServerError,
                RateLimitError,
                LLMResponseError,
            ) as exc:
                last_error = exc
                if attempt >= DEFAULT_MAX_RETRIES:
                    break

                sleep_s = min(2 ** (attempt - 1), 8)
                logger.warning(
                    "Retrying LLM request after attempt %d/%d failed: %s",
                    attempt,
                    DEFAULT_MAX_RETRIES,
                    self._describe_error(exc),
                )
                time.sleep(sleep_s)
            except APIStatusError as exc:
                last_error = exc
                if attempt >= DEFAULT_MAX_RETRIES or exc.status_code not in RETRYABLE_STATUS_CODES:
                    break

                sleep_s = min(2 ** (attempt - 1), 8)
                logger.warning(
                    "Retrying LLM request after status %s on attempt %d/%d: %s",
                    exc.status_code,
                    attempt,
                    DEFAULT_MAX_RETRIES,
                    self._describe_error(exc),
                )
                time.sleep(sleep_s)

        assert last_error is not None
        raise last_error

    def _parse_response(self, response: Any) -> dict[str, Any]:
        """Normalize OpenAI-compatible responses into the agent's expected shape."""
        choices = getattr(response, "choices", None)
        if not choices:
            raise LLMResponseError(
                "LLM returned no choices. "
                f"Payload preview: {self._response_preview(response)}"
            )

        choice = choices[0]
        message = getattr(choice, "message", None)
        if message is None:
            raise LLMResponseError(
                "LLM returned a choice without a message. "
                f"Payload preview: {self._response_preview(response)}"
            )

        result: dict[str, Any] = {
            "content": self._coerce_content(getattr(message, "content", None)),
            "tool_calls": None,
            "usage": {
                "prompt_tokens": (
                    getattr(getattr(response, "usage", None), "prompt_tokens", 0) or 0
                ),
                "completion_tokens": (
                    getattr(getattr(response, "usage", None), "completion_tokens", 0) or 0
                ),
            },
        }

        # Parse tool calls if present
        if getattr(message, "tool_calls", None):
            result["tool_calls"] = []
            for tc in message.tool_calls:
                raw_arguments = getattr(tc.function, "arguments", None)
                arguments = self._parse_tool_arguments(raw_arguments, tc.function.name)
                result["tool_calls"].append({
                    "id": tc.id,
                    "name": tc.function.name,
                    "arguments": arguments,
                })

        return result

    def generate_text(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
    ) -> str:
        """Simple text generation without tool calling."""
        result = self.chat(messages, temperature=temperature)
        return result["content"] or ""

    def _parse_tool_arguments(
        self, raw_arguments: Any, tool_name: str
    ) -> dict[str, Any]:
        """Best-effort parsing for providers that sometimes return invalid JSON."""
        if raw_arguments in (None, ""):
            return {}
        if isinstance(raw_arguments, dict):
            return raw_arguments

        try:
            return json.loads(raw_arguments)
        except (TypeError, json.JSONDecodeError):
            pass

        # Try to repair common JSON issues from LLMs:
        # 1. Trailing commas  2. Unescaped newlines in strings  3. Single quotes
        if isinstance(raw_arguments, str):
            repaired = raw_arguments
            # Remove trailing commas before } or ]
            repaired = re.sub(r",\s*([}\]])", r"\1", repaired)
            try:
                return json.loads(repaired)
            except (TypeError, json.JSONDecodeError):
                pass

        logger.warning(
            "Failed to decode tool arguments for %s; preserving raw arguments.",
            tool_name,
        )
        return {"__raw_arguments": raw_arguments}

    def _coerce_content(self, content: Any) -> str | None:
        """Flatten provider-specific content formats to a plain string."""
        if content is None or isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict):
                    text = item.get("text")
                    if isinstance(text, str):
                        parts.append(text)
            return "\n".join(part for part in parts if part) or None
        return str(content)

    def _response_preview(self, response: Any, max_len: int = 500) -> str:
        """Create a short response preview for logs and exceptions."""
        try:
            if hasattr(response, "model_dump"):
                preview = json.dumps(response.model_dump(), ensure_ascii=False)
            else:
                preview = repr(response)
        except Exception:
            preview = repr(response)
        if len(preview) > max_len:
            return preview[:max_len] + "...(truncated)"
        return preview

    def _describe_error(self, exc: Exception) -> str:
        """Create compact, readable retry logs across OpenAI-compatible backends."""
        status_code = getattr(exc, "status_code", None)
        if status_code is None:
            status_code = getattr(getattr(exc, "response", None), "status_code", None)
        if status_code is not None:
            return f"status={status_code}, error={exc}"
        return str(exc)
