"""
Base class and registry for agent tools.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class ToolResult:
    """Result returned by a tool execution."""

    output: str
    success: bool = True
    metadata: dict[str, Any] | None = None


class Tool(ABC):
    """Abstract base class for agent tools."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Tool name used in function calling."""
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        """Tool description for the LLM."""
        ...

    @property
    @abstractmethod
    def parameters(self) -> dict:
        """JSON Schema for the tool parameters."""
        ...

    @abstractmethod
    def execute(self, **kwargs) -> ToolResult:
        """Execute the tool with given arguments."""
        ...

    def to_openai_tool(self) -> dict:
        """Convert to OpenAI function calling format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class ToolRegistry:
    """Registry that holds all available tools."""

    def __init__(self):
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool):
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def execute(self, name: str, arguments: dict) -> ToolResult:
        tool = self._tools.get(name)
        if tool is None:
            return ToolResult(output=f"Error: Unknown tool '{name}'", success=False)

        # If the LLM returned unparseable JSON, __raw_arguments is set.
        # Try to recover by extracting known parameter names from the raw string.
        if "__raw_arguments" in arguments:
            raw = arguments["__raw_arguments"]
            recovered = self._recover_arguments(tool, raw)
            if recovered is not None:
                arguments = recovered
            else:
                return ToolResult(
                    output=(
                        f"Error: LLM returned malformed arguments for '{name}'. "
                        f"Please retry with valid JSON arguments. Raw: {str(raw)[:500]}"
                    ),
                    success=False,
                )

        try:
            return tool.execute(**arguments)
        except Exception as e:
            return ToolResult(output=f"Error executing {name}: {e}", success=False)

    @staticmethod
    def _recover_arguments(tool: "Tool", raw: str) -> dict | None:
        """Best-effort recovery of tool arguments from a raw string."""
        import json, re  # noqa: E401
        if not isinstance(raw, str):
            return None
        # Try to find a JSON object anywhere in the string
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            candidate = match.group(0)
            # Remove trailing commas
            candidate = re.sub(r",\s*([}\]])", r"\1", candidate)
            try:
                parsed = json.loads(candidate)
                if isinstance(parsed, dict):
                    return parsed
            except (json.JSONDecodeError, TypeError):
                pass
        # For single-parameter tools like execute_code(code=...), treat the
        # whole raw string as the parameter value.
        schema = tool.parameters
        required = schema.get("required", [])
        if len(required) == 1:
            return {required[0]: raw}
        return None

    def to_openai_tools(self) -> list[dict]:
        """Get all tools in OpenAI function calling format."""
        return [t.to_openai_tool() for t in self._tools.values()]

    @property
    def tool_names(self) -> list[str]:
        return list(self._tools.keys())
