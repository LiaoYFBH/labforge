"""
Submit result tool — agent calls this when it believes the task is complete.
"""

from __future__ import annotations

from .base import Tool, ToolResult


class SubmitResultTool(Tool):
    """Submit the final result for a research task."""

    def __init__(self):
        self._submitted = False
        self._result: str = ""

    @property
    def name(self) -> str:
        return "submit_result"

    @property
    def description(self) -> str:
        return (
            "Submit the final result when you have completed the research task. "
            "Provide the path to the output file (e.g., a generated plot, "
            "a trained model, or a result CSV) along with a brief summary. "
            "Call this ONLY when you are confident the task is done correctly."
        )

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "result_path": {
                    "type": "string",
                    "description": "Path to the output file in the workspace.",
                },
                "summary": {
                    "type": "string",
                    "description": "Brief summary of what was accomplished and key findings.",
                },
            },
            "required": ["result_path", "summary"],
        }

    def execute(self, result_path: str, summary: str) -> ToolResult:
        self._submitted = True
        self._result = summary
        return ToolResult(
            output=f"Result submitted.\nPath: {result_path}\nSummary: {summary}",
            success=True,
            metadata={"result_path": result_path, "summary": summary},
        )

    @property
    def submitted(self) -> bool:
        return self._submitted

    def reset(self):
        self._submitted = False
        self._result = ""
