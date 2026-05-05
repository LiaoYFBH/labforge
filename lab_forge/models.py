"""
Multi-model factory using LangChain's ChatOpenAI.

Replaces the hand-rolled LLMClient with LangChain-native model instances.
Supports any OpenAI-compatible API (AI Studio, MiniMax, DeepSeek, OpenAI, etc.).

Mod 3: ``create_chat_model`` returns ``ResilientChatOpenAI`` instead of the
raw ``ChatOpenAI``. The subclass defensively coerces ``tool_calls[i].args``
to a dict whenever the upstream provider emits a malformed or non-string
``function.arguments`` field. Several AI Studio models (notably
deepseek-v3 / ERNIE 4.5+ on long prompts) intermittently return arguments
that LangChain's strict parser leaves as a non-dict, which then triggers
``ValidationError: tool_calls.0.args Input should be a valid dictionary``
and poisons the LangGraph conversation thread. We've already added a
poison-history reset for this case (see agent.py), but the cleaner fix is
to never let the bad shape leave the LLM exit in the first place.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

from langchain_openai import ChatOpenAI

from .config import ModelConfig, resolve_api_key_for_endpoint

logger = logging.getLogger(__name__)






def _tolerant_json_loads(text: str) -> dict | None:
    """Best-effort JSON object parse for malformed LLM outputs.

    Returns the parsed dict on success, or ``None`` if no fix recovers a
    parseable object. Tries (in order):
      1. Plain ``json.loads``.
      2. Strip trailing commas before ``}`` / ``]`` (common LLM mistake).
      3. Find the first balanced ``{...}`` substring and parse that
         (LLMs sometimes wrap the JSON in stray prefixes/suffixes).

    Single-quote-to-double-quote and unquoted-key fixes are intentionally
    not attempted: they introduce more false positives than they cure when
    the original JSON happens to contain escaped quotes inside string values.
    """
    if not isinstance(text, str) or not text.strip():
        return None

    try:
        result = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        result = None
    else:
        return result if isinstance(result, dict) else None

    cleaned = re.sub(r",\s*([}\]])", r"\1", text)
    if cleaned != text:
        try:
            result = json.loads(cleaned)
        except (json.JSONDecodeError, TypeError):
            result = None
        else:
            return result if isinstance(result, dict) else None


    start = text.find("{")
    if start != -1:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start:i + 1]
                    candidate = re.sub(r",\s*([}\]])", r"\1", candidate)
                    try:
                        result = json.loads(candidate)
                    except (json.JSONDecodeError, TypeError):
                        return None
                    return result if isinstance(result, dict) else None
                    break
    return None


def _coerce_function_arguments(raw: Any) -> str:
    """Normalize a tool_call's ``function.arguments`` to a JSON string.

    Returns a string suitable for ``json.loads`` — that's the contract
    LangChain's ``parse_tool_call`` expects. We accept three input shapes:
      * Already a valid JSON string  → return as-is.
      * A dict / list                → JSON-encode.
      * A malformed JSON string      → repair with ``_tolerant_json_loads``,
        re-encode if recoverable; if not, return ``"{}"`` so downstream
        parsing produces an empty-args tool call rather than poisoning
        the conversation thread.
    """
    if raw is None:
        return "{}"
    if isinstance(raw, dict):
        try:
            return json.dumps(raw, ensure_ascii=False)
        except (TypeError, ValueError):
            return "{}"
    if isinstance(raw, list):
        try:
            return json.dumps(raw, ensure_ascii=False)
        except (TypeError, ValueError):
            return "{}"
    if isinstance(raw, str):

        if not raw.strip():
            return "{}"
        try:
            json.loads(raw)
            return raw
        except (json.JSONDecodeError, TypeError):
            repaired = _tolerant_json_loads(raw)
            if repaired is not None:
                logger.warning(
                    "Repaired malformed tool_call arguments via tolerant JSON "
                    "parser (length=%d).",
                    len(raw),
                )
                try:
                    return json.dumps(repaired, ensure_ascii=False)
                except (TypeError, ValueError):
                    return "{}"
            logger.warning(
                "Could not repair tool_call arguments; replacing with empty "
                "dict to keep the conversation thread valid. Sample: %r",
                raw[:200],
            )
            return "{}"

    return "{}"


def _normalize_response_dict(response_dict: dict) -> None:
    """In-place repair of ``response.choices[*].message.tool_calls[*].function.arguments``.

    Mutates the dict so ``parse_tool_call`` (called inside
    ``_convert_dict_to_message`` via parent ``_create_chat_result``) always
    sees a parseable JSON string.
    """
    if not isinstance(response_dict, dict):
        return
    choices = response_dict.get("choices")
    if not isinstance(choices, list):
        return
    for choice in choices:
        if not isinstance(choice, dict):
            continue
        message = choice.get("message")
        if not isinstance(message, dict):
            continue
        tool_calls = message.get("tool_calls")
        if not isinstance(tool_calls, list):
            continue
        for tc in tool_calls:
            if not isinstance(tc, dict):
                continue
            function = tc.get("function")
            if not isinstance(function, dict):
                continue
            function["arguments"] = _coerce_function_arguments(function.get("arguments"))


def _normalize_aimessage_tool_calls(message: Any) -> None:
    """Belt-and-suspenders: walk an ``AIMessage`` and coerce stray string
    ``args`` fields into dicts. Runs after parent ``_create_chat_result``.

    Even with the response-dict pre-pass above, some langchain versions
    or future provider quirks can still produce a tool_call where
    ``args`` is a string — the pre-pass only normalizes
    ``function.arguments``, not the post-conversion ``tool_calls[i].args``
    on the AIMessage itself. This second pass catches anything that slipped.
    """
    tool_calls = getattr(message, "tool_calls", None)
    if not tool_calls:
        return
    for tc in tool_calls:


        if not isinstance(tc, dict):
            continue
        args = tc.get("args")
        if isinstance(args, str):
            repaired = _tolerant_json_loads(args)
            if repaired is None:
                logger.warning(
                    "Stray string args in AIMessage.tool_calls[*]; replacing "
                    "with empty dict. Sample: %r",
                    args[:200],
                )
                tc["args"] = {}
            else:
                tc["args"] = repaired
































def _coerce_tool_args_to_dict(args: Any) -> dict:
    """Return a dict suitable for ``ToolCall.args``.

    - dict           -> returned as-is
    - JSON-string of a dict (single- or double-encoded) -> parsed dict
    - anything else  -> ``{}`` (so the call survives validation; the
      tool will then complain about missing fields, which the agent can
      recover from in a normal round, instead of poisoning the thread)
    """
    if isinstance(args, dict):
        return args
    if isinstance(args, str):
        try:
            parsed = json.loads(args)
        except (json.JSONDecodeError, TypeError):
            parsed = _tolerant_json_loads(args)


        if isinstance(parsed, str):
            try:
                inner = json.loads(parsed)
            except (json.JSONDecodeError, TypeError):
                inner = _tolerant_json_loads(parsed)
            if isinstance(inner, dict):
                return inner
        if isinstance(parsed, dict):
            return parsed
    return {}


def _install_tool_call_patch() -> None:
    """Monkey-patch ``tool_call`` in every module that holds a reference,
    forcing ``args`` to be a dict before the ToolCall pydantic model sees it.

    Idempotent: calling twice is a no-op (we tag the patched function).
    """
    try:
        from langchain_core.messages import tool as _lc_tool_module
        from langchain_core.messages import ai as _lc_ai_module
        from langchain_core.output_parsers import openai_tools as _lc_openai_tools_module
    except ImportError:
        logger.warning(
            "Could not import langchain_core tool_call modules; "
            "tool_call args coercion patch is inactive."
        )
        return

    original = _lc_tool_module.tool_call
    if getattr(original, "__lab_forge_patched__", False):
        return

    def safe_tool_call(*, name, args, id, **kwargs):
        coerced = _coerce_tool_args_to_dict(args)
        if not isinstance(args, dict):
            logger.warning(
                "Coerced non-dict tool_call args to dict (name=%r). Sample: %r",
                name, repr(args)[:200],
            )





        if isinstance(name, str):
            stripped = name.strip()
            if stripped != name:
                logger.warning(
                    "Stripped whitespace from tool_call name: %r -> %r",
                    name, stripped,
                )
                name = stripped
        return original(name=name, args=coerced, id=id, **kwargs)

    safe_tool_call.__lab_forge_patched__ = True

    _lc_tool_module.tool_call = safe_tool_call
    _lc_ai_module.create_tool_call = safe_tool_call
    _lc_openai_tools_module.create_tool_call = safe_tool_call

    logger.info(
        "Installed tool_call args coercion patch in 3 langchain modules "
        "(tool, ai, openai_tools)."
    )


_install_tool_call_patch()


class ResilientChatOpenAI(ChatOpenAI):
    """ChatOpenAI variant that won't poison the conversation when the
    upstream provider emits malformed tool_call arguments.

    See module docstring for context. The subclass overrides
    ``_create_chat_result`` (used by both ``_generate`` and the streaming
    paths via ``generate_from_stream``) so the fix applies regardless of
    streaming mode.
    """

    def _create_chat_result(self, response, generation_info=None):



        if hasattr(response, "model_dump") and not isinstance(response, dict):
            response_dict = response.model_dump()
        else:
            response_dict = response

        if isinstance(response_dict, dict):
            try:
                _normalize_response_dict(response_dict)
            except Exception:
                logger.exception("Pre-pass tool_call arg normalization failed")
            result = super()._create_chat_result(response_dict, generation_info)
        else:
            result = super()._create_chat_result(response, generation_info)



        try:
            for gen in getattr(result, "generations", []) or []:
                msg = getattr(gen, "message", None)
                if msg is not None:
                    _normalize_aimessage_tool_calls(msg)
        except Exception:
            logger.exception("Post-pass tool_call arg normalization failed")

        return result


def _get_first_env(*keys: str) -> str:
    """Return the first non-empty environment variable among ``keys``."""
    for key in keys:
        value = os.environ.get(key, "").strip()
        if value:
            return value
    return ""





MODEL_PRESETS: dict[str, dict[str, str]] = {
    "MiniMax-M2.7": {
        "model": "MiniMax-M2.7",
        "base_url": "https://api.minimaxi.com/v1",
    },
    "星河社区 · ERNIE 4.5 Turbo 128K (推荐)": {
        "model": "ernie-4.5-turbo-128k-preview",
        "base_url": "https://aistudio.baidu.com/llm/lmapi/v3",
    },
    "星河社区 · ERNIE 5.0 Thinking": {
        "model": "ernie-5.0-thinking-preview",
        "base_url": "https://aistudio.baidu.com/llm/lmapi/v3",
    },
    "星河社区 · ERNIE X1.1": {
        "model": "ernie-x1.1-preview",
        "base_url": "https://aistudio.baidu.com/llm/lmapi/v3",
    },
    "星河社区 · ERNIE X1 Turbo 32K": {
        "model": "ernie-x1-turbo-32k",
        "base_url": "https://aistudio.baidu.com/llm/lmapi/v3",
    },
    "星河社区 · DeepSeek-V3": {
        "model": "deepseek-v3",
        "base_url": "https://aistudio.baidu.com/llm/lmapi/v3",
    },
    "星河社区 · DeepSeek-R1": {
        "model": "deepseek-r1",
        "base_url": "https://aistudio.baidu.com/llm/lmapi/v3",
    },
    "星河社区 · Kimi K2 Instruct": {
        "model": "kimi-k2-instruct",
        "base_url": "https://aistudio.baidu.com/llm/lmapi/v3",
    },
    "星河社区 · Qwen3 Coder 30B": {
        "model": "qwen3-coder-30b-a3b-instruct",
        "base_url": "https://aistudio.baidu.com/llm/lmapi/v3",
    },
}









REVIEWER_PRESETS: dict[str, dict[str, str]] = {
    "星河社区 · DeepSeek-V3 (推荐)": {
        "model": "deepseek-v3",
        "base_url": "https://aistudio.baidu.com/llm/lmapi/v3",
    },
    "星河社区 · ERNIE 4.5 Turbo 128K": {
        "model": "ernie-4.5-turbo-128k-preview",
        "base_url": "https://aistudio.baidu.com/llm/lmapi/v3",
    },
    "星河社区 · ERNIE 5.0 Thinking": {
        "model": "ernie-5.0-thinking-preview",
        "base_url": "https://aistudio.baidu.com/llm/lmapi/v3",
    },
    "星河社区 · ERNIE X1 Turbo 32K": {
        "model": "ernie-x1-turbo-32k",
        "base_url": "https://aistudio.baidu.com/llm/lmapi/v3",
    },
    "星河社区 · DeepSeek-R1": {
        "model": "deepseek-r1",
        "base_url": "https://aistudio.baidu.com/llm/lmapi/v3",
    },
    "星河社区 · Qwen3 Coder 30B": {
        "model": "qwen3-coder-30b-a3b-instruct",
        "base_url": "https://aistudio.baidu.com/llm/lmapi/v3",
    },
    "星河社区 · Kimi K2 Instruct": {
        "model": "kimi-k2-instruct",
        "base_url": "https://aistudio.baidu.com/llm/lmapi/v3",
    },
    "(legacy) ERNIE 3.5 8K": {
        "model": "ernie-3.5-8k",
        "base_url": "https://aistudio.baidu.com/llm/lmapi/v3",
    },
    "(legacy) ERNIE Speed 128K": {
        "model": "ernie-speed-128k",
        "base_url": "https://aistudio.baidu.com/llm/lmapi/v3",
    },
}


def create_chat_model(config: ModelConfig, **overrides: Any) -> ChatOpenAI:
    """
    Create a ResilientChatOpenAI instance from a ModelConfig.

    Returns the ``ResilientChatOpenAI`` subclass (not the raw ``ChatOpenAI``)
    so malformed ``tool_calls.args`` from chatty providers (deepseek-v3,
    ERNIE 4.5+ on long prompts) get coerced rather than poisoning the
    conversation thread.

    Supports any OpenAI-compatible API (AI Studio, MiniMax, DeepSeek, etc.).
    API key is resolved from config or environment variables.
    """
    api_key = config.api_key or resolve_api_key_for_endpoint(
        base_url=config.base_url,
        model=config.model,
    )

    kwargs: dict[str, Any] = {
        "model": config.model,
        "api_key": api_key,
        "base_url": config.base_url,
        "temperature": config.temperature,
        "max_tokens": config.max_tokens,
        "timeout": config.timeout,
        "max_retries": 3,
    }
    kwargs.update(overrides)

    return ResilientChatOpenAI(**kwargs)
