"""
Code execution and file operation tools.
"""

from __future__ import annotations

from ..result_guardrails import (
    blocking_findings,
    format_findings,
    validate_workspace_results,
)
from ..sandbox import Sandbox
from .base import Tool, ToolResult


def _append_result_guardrail(output: str, sandbox: Sandbox) -> str:
    """Surface invalid numeric artifacts immediately after code execution."""

    findings = blocking_findings(validate_workspace_results(sandbox.working_dir))
    if not findings:
        return output
    return (
        f"{output}\n\n[Result guardrail]\n"
        f"{format_findings(findings)}\n"
        "Treat these as failed or unresolved experiment results. Fix the "
        "experiment, or explicitly disclose the invalid values before writing "
        "the report."
    )


class ExecuteCodeTool(Tool):
    """Execute Python code in a sandboxed environment."""

    def __init__(self, sandbox: Sandbox):
        self.sandbox = sandbox

    @property
    def name(self) -> str:
        return "execute_code"

    @property
    def description(self) -> str:
        return (
            "Execute Python code in a sandboxed environment. "
            "Use this to run experiments, process data, train models, "
            "generate plots, or perform any computation. "
            "The code runs in an isolated workspace with access to "
            "common scientific Python packages (numpy, pandas, scipy, "
            "sklearn, matplotlib, torch, etc.)."
        )

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Python code to execute.",
                },
            },
            "required": ["code"],
        }

    def execute(self, code: str) -> ToolResult:
        result = self.sandbox.execute_code(code, language="python")
        output = _append_result_guardrail(result.output, self.sandbox)
        return ToolResult(
            output=output,
            success=result.success,
            metadata={
                "exit_code": result.exit_code,
                "timed_out": result.timed_out,
                "script_path": result.script_path,
                "log_path": result.log_path,
            },
        )


class ExecuteBashTool(Tool):
    """Execute a bash command."""

    def __init__(self, sandbox: Sandbox):
        self.sandbox = sandbox

    @property
    def name(self) -> str:
        return "execute_bash"

    @property
    def description(self) -> str:
        return (
            "Execute a bash command in the sandbox. "
            "Use this for installing packages (pip install), "
            "downloading data (wget/curl), or file system operations."
        )

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "Bash command to execute.",
                },
            },
            "required": ["command"],
        }

    def execute(self, command: str) -> ToolResult:
        result = self.sandbox.execute_bash(command)
        output = _append_result_guardrail(result.output, self.sandbox)
        return ToolResult(
            output=output,
            success=result.success,
            metadata={
                "exit_code": result.exit_code,
                "script_path": result.script_path,
                "log_path": result.log_path,
            },
        )


class FileReadTool(Tool):
    """Read a file from the workspace."""

    def __init__(self, sandbox: Sandbox):
        self.sandbox = sandbox

    @property
    def name(self) -> str:
        return "file_read"

    @property
    def description(self) -> str:
        return "Read the contents of a file in the workspace."

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Relative path to the file in the workspace.",
                },
            },
            "required": ["path"],
        }

    def execute(self, path: str) -> ToolResult:
        content = self.sandbox.read_file(path)
        is_error = content.startswith("Error:")
        return ToolResult(output=content, success=not is_error)


class FileWriteTool(Tool):
    """Write a file to the workspace."""

    # Text-style artefacts the agent is allowed to author directly. Data files
    # (CSV, NPY, PNG, etc.) must come out of `execute_code` so the workspace
    # never contains AI-fabricated "results".
    ALLOWED_EXTENSIONS = {
        ".py", ".sh", ".md", ".txt",
        ".yaml", ".yml", ".toml", ".ini", ".cfg",
        ".tex", ".bib", ".rst",
        ".ipynb",
    }
    FORBIDDEN_EXTENSIONS = {
        ".csv", ".tsv", ".parquet", ".npy", ".npz", ".pkl", ".pickle",
        ".h5", ".hdf5", ".xlsx", ".xls", ".json", ".jsonl",
        ".feather", ".arrow",
        ".png", ".jpg", ".jpeg", ".svg", ".gif", ".bmp", ".tiff", ".pdf",
        ".wav", ".mp3", ".mp4", ".webp",
    }

    def __init__(self, sandbox: Sandbox):
        self.sandbox = sandbox

    @property
    def name(self) -> str:
        return "file_write"

    @property
    def description(self) -> str:
        return (
            "Write content to a file in the workspace. "
            "Use this ONLY for source code, scripts, markdown, and plain-text "
            "configuration files (e.g. .py, .sh, .md, .txt, .yaml). "
            "Data artefacts (CSV, NPY, PNG, JSON, parquet, images, etc.) MUST "
            "be produced by running real code via execute_code — never "
            "hand-written here — otherwise they will be rejected."
        )

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": (
                        "Relative path for the file in the workspace. "
                        "Only source/text files are allowed; data files must "
                        "be generated by execute_code."
                    ),
                },
                "content": {
                    "type": "string",
                    "description": "Content to write to the file.",
                },
            },
            "required": ["path", "content"],
        }

    def _validate_path(self, path: str) -> str | None:
        """Return an error message if ``path`` targets a forbidden file, else None."""
        from pathlib import PurePosixPath

        suffix = PurePosixPath(path).suffix.lower()
        if not suffix:
            # No extension — treat as plain text, allow.
            return None
        if suffix in self.FORBIDDEN_EXTENSIONS:
            return (
                f"file_write rejected: '{path}' looks like a data/binary artefact "
                f"(extension `{suffix}`). Generate it by running real code via "
                "execute_code so the workspace only contains machine-produced "
                "outputs. file_write is reserved for source/text files such as "
                ".py / .sh / .md / .txt / .yaml."
            )
        if suffix not in self.ALLOWED_EXTENSIONS:
            return (
                f"file_write rejected: extension `{suffix}` is not on the allow "
                "list. Use execute_code to generate data artefacts, or rename "
                "to one of: " + ", ".join(sorted(self.ALLOWED_EXTENSIONS))
            )
        return None

    def execute(self, path: str, content: str) -> ToolResult:
        error = self._validate_path(path)
        if error:
            return ToolResult(output=error, success=False)
        msg = self.sandbox.write_file(path, content)
        return ToolResult(output=msg, success=True)


class ListFilesTool(Tool):
    """List files in the workspace."""

    def __init__(self, sandbox: Sandbox):
        self.sandbox = sandbox

    @property
    def name(self) -> str:
        return "list_files"

    @property
    def description(self) -> str:
        return "List files and directories in the workspace."

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Relative path to list. Defaults to workspace root.",
                    "default": ".",
                },
            },
        }

    def execute(self, path: str = ".") -> ToolResult:
        content = self.sandbox.list_files(path)
        is_error = content.startswith("Error:")
        return ToolResult(output=content, success=not is_error)
