"""
LangChain tool wrappers for all LabForge tools.

Each wrapper delegates to the existing tool implementation (in tools/),
preserving all internal logic while providing LangChain-compatible interfaces.
"""

from __future__ import annotations

from typing import Optional

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from .sandbox import Sandbox
from .result_guardrails import (
    blocking_findings,
    format_findings,
    report_text_discloses_findings,
    validate_workspace_results,
)
from .tools.code_tools import (
    ExecuteBashTool,
    ExecuteCodeTool,
    FileReadTool,
    FileWriteTool,
    ListFilesTool,
)
from .tools.fulltext_tool import DEFAULT_CHUNK_SIZE, ReadPaperFullTextTool
from .tools.hf_papers_tool import LookupPaperCodeTool
from .tools.report_tool import GenerateReportTool
from .tools.search_tool import SearchLiteratureTool
from .tools.submit_tool import SubmitResultTool


def _truncate(text: str, max_len: int = 4000) -> str:
    if len(text) <= max_len:
        return text
    return text[:max_len] + f"\n... (truncated, {len(text)} chars total)"






class ExecuteCodeInput(BaseModel):
    code: str = Field(description="Python code to execute.")


class ExecuteBashInput(BaseModel):
    command: str = Field(description="Bash command to execute.")


class FileReadInput(BaseModel):
    path: str = Field(description="Relative path to the file in the workspace.")


class FileWriteInput(BaseModel):
    path: str = Field(description="Relative path for the file in the workspace.")
    content: str = Field(description="Content to write to the file.")


class ListFilesInput(BaseModel):
    path: str = Field(default=".", description="Relative path to list. Defaults to workspace root.")


class SearchLiteratureInput(BaseModel):
    query: str = Field(description="Search query for finding academic papers.")
    max_results: int = Field(default=10, description="Maximum number of results to return.")


class LookupPaperCodeInput(BaseModel):
    arxiv_id: str = Field(
        description=(
            "arXiv id of the paper, e.g. '2506.09781' or '2506.09781v2'. "
            "Full arxiv URLs are also accepted; the tool extracts the id."
        ),
    )


class ReadPaperFullTextInput(BaseModel):
    url: str = Field(default="", description="Paper landing page, arXiv URL, or direct PDF/image URL.")
    local_path: str = Field(default="", description="Local PDF/image path (relative to workspace or absolute).")
    title: str = Field(default="", description="Optional title hint used for naming saved files.")
    force_ocr: bool = Field(default=False, description="Force PaddleOCR-VL even when direct PDF text extraction succeeds.")
    chunk_size_chars: int = Field(default=DEFAULT_CHUNK_SIZE, description="Approximate size of each saved chunk file.")
    use_doc_orientation_classify: Optional[bool] = Field(default=None, description="PaddleOCR-VL: classify document orientation.")
    use_doc_unwarping: Optional[bool] = Field(default=None, description="PaddleOCR-VL: apply document unwarping.")
    use_chart_recognition: Optional[bool] = Field(default=None, description="PaddleOCR-VL: run chart recognition.")


class GenerateReportInput(BaseModel):
    """
    Generate the final report. 
    IMPORTANT: You MUST use STRICT LaTeX-safe Markdown!
    - Headers: `## 1. Introduction` (No HTML <h2>).
    - Bold/Italic: `**bold**` (No \\textbf{}).
    - Math: `$x$` or `$$x$$` ONLY.
    - Citations: `[1]` ONLY (No \\cite{} or Author-Year).
    - Tables: No markdown tables, use the `tables` parameter with CSV paths instead.
    """
    title: str = Field(description="Title of the research report.")
    results: str = Field(description="Experimental results with numbers and tables.")
    abstract: str = Field(default="", description="Short abstract covering the task, methods, and findings.")
    keywords: list[str] = Field(default_factory=list, description="Optional paper keywords.")
    introduction: str = Field(default="", description="Background and motivation for this research.")
    related_work: str = Field(default="", description="Summary of relevant prior work and methods found in literature.")
    methodology: str = Field(default="", description="Description of the methods/algorithms compared.")
    setup: str = Field(default="", description="Experimental setup: dataset, hyperparameters, evaluation metrics.")
    analysis: str = Field(default="", description="Analysis and discussion of the results.")
    conclusion: str = Field(default="", description="Key findings and conclusions.")
    references: list[str] = Field(
        default_factory=list,
        description="Verified references used in the report. Never include invented citations.",
    )
    figures: list[dict] = Field(
        default_factory=list,
        description="Saved chart/image files to include in the final paper. Each item should include path, caption, and optional section.",
    )
    tables: list[dict] = Field(
        default_factory=list,
        description="Saved CSV/TSV tables to include in the final paper. Each item should include path, caption, and optional section.",
    )


class SubmitResultInput(BaseModel):
    result_path: str = Field(description="Path to the output file in the workspace.")
    summary: str = Field(description="Brief summary of what was accomplished and key findings.")






def make_execute_code_tool(sandbox: Sandbox) -> StructuredTool:
    inner = ExecuteCodeTool(sandbox)

    def _run(code: str) -> str:
        result = inner.execute(code=code)
        output = _truncate(result.output)
        return output if result.success else f"ERROR: {output}"

    return StructuredTool(
        name="execute_code",
        description=inner.description,
        func=_run,
        args_schema=ExecuteCodeInput,
    )


def make_execute_bash_tool(sandbox: Sandbox) -> StructuredTool:
    inner = ExecuteBashTool(sandbox)

    def _run(command: str) -> str:
        result = inner.execute(command=command)
        output = _truncate(result.output)
        return output if result.success else f"ERROR: {output}"

    return StructuredTool(
        name="execute_bash",
        description=inner.description,
        func=_run,
        args_schema=ExecuteBashInput,
    )


def make_file_read_tool(sandbox: Sandbox) -> StructuredTool:
    inner = FileReadTool(sandbox)

    def _run(path: str) -> str:
        result = inner.execute(path=path)
        return _truncate(result.output) if result.success else f"ERROR: {result.output}"

    return StructuredTool(
        name="file_read",
        description=inner.description,
        func=_run,
        args_schema=FileReadInput,
    )


def make_file_write_tool(sandbox: Sandbox) -> StructuredTool:
    inner = FileWriteTool(sandbox)

    def _run(path: str, content: str) -> str:
        result = inner.execute(path=path, content=content)
        output = _truncate(result.output)




        return output if result.success else f"ERROR: {output}"

    return StructuredTool(
        name="file_write",
        description=inner.description,
        func=_run,
        args_schema=FileWriteInput,
    )


def make_list_files_tool(sandbox: Sandbox) -> StructuredTool:
    inner = ListFilesTool(sandbox)

    def _run(path: str = ".") -> str:
        result = inner.execute(path=path)
        return result.output if result.success else f"ERROR: {result.output}"

    return StructuredTool(
        name="list_files",
        description=inner.description,
        func=_run,
        args_schema=ListFilesInput,
    )


def make_search_literature_tool(
    working_dir: str | None = None,
    search_quota: int = 0,
) -> StructuredTool:
    inner = SearchLiteratureTool(working_dir=working_dir)
    if search_quota:
        inner.set_quota(search_quota)

    def _run(query: str, max_results: int = 10) -> str:
        result = inner.execute(query=query, max_results=max_results)
        output = _truncate(result.output)
        return output if result.success else f"ERROR: {output}"

    return StructuredTool(
        name="search_literature",
        description=inner.description,
        func=_run,
        args_schema=SearchLiteratureInput,
    )


def make_lookup_paper_code_tool() -> StructuredTool:
    inner = LookupPaperCodeTool()

    def _run(arxiv_id: str) -> str:
        result = inner.execute(arxiv_id=arxiv_id)
        output = _truncate(result.output)
        return output if result.success else f"ERROR: {output}"

    return StructuredTool(
        name="lookup_paper_code",
        description=inner.description,
        func=_run,
        args_schema=LookupPaperCodeInput,
    )


def make_read_paper_fulltext_tool(sandbox: Sandbox, ocr_enabled: bool = False) -> StructuredTool:
    inner = ReadPaperFullTextTool(sandbox, ocr_enabled=ocr_enabled)

    def _run(
        url: str = "",
        local_path: str = "",
        title: str = "",
        force_ocr: bool = False,
        chunk_size_chars: int = DEFAULT_CHUNK_SIZE,
        use_doc_orientation_classify: bool | None = None,
        use_doc_unwarping: bool | None = None,
        use_chart_recognition: bool | None = None,
    ) -> str:
        result = inner.execute(
            url=url,
            local_path=local_path,
            title=title,
            force_ocr=force_ocr,
            chunk_size_chars=chunk_size_chars,
            use_doc_orientation_classify=use_doc_orientation_classify,
            use_doc_unwarping=use_doc_unwarping,
            use_chart_recognition=use_chart_recognition,
        )
        output = _truncate(result.output)
        return output if result.success else f"ERROR: {output}"

    return StructuredTool(
        name="read_paper_fulltext",
        description=inner.description,
        func=_run,
        args_schema=ReadPaperFullTextInput,
    )


def make_generate_report_tool(
    working_dir: str,
    writer_llm=None,
    expand_sections: bool = True,
    task_description: str = "",
) -> StructuredTool:
    """Build the generate_report tool, optionally wired with a writer LLM.

    When ``writer_llm`` is provided (any LangChain-compatible chat model with
    ``.invoke(prompt) -> response``), the tool runs an outline → per-section
    expansion pass so each section reaches top-conference depth before the
    bundle is written. Without it, the legacy single-pass behaviour is kept.

    ``task_description`` is forwarded so the tool can detect SURVEY ONLY
    runs and skip the experiment-evidence gate (P5 fix).
    """
    writer_callable = None
    if writer_llm is not None:
        def _writer(prompt: str) -> str:
            response = writer_llm.invoke(prompt)
            content = getattr(response, "content", response)
            if isinstance(content, list):


                parts = []
                for item in content:
                    if isinstance(item, str):
                        parts.append(item)
                    elif isinstance(item, dict):
                        text = item.get("text")
                        if isinstance(text, str):
                            parts.append(text)
                content = "\n".join(parts)
            return str(content or "")
        writer_callable = _writer

    inner = GenerateReportTool(
        working_dir=working_dir,
        writer_llm=writer_callable,
        expand_sections=expand_sections,
        task_description=task_description,
    )

    def _run(
        title: str,
        results: str,
        abstract: str = "",
        keywords: list[str] | None = None,
        introduction: str = "",
        related_work: str = "",
        methodology: str = "",
        setup: str = "",
        analysis: str = "",
        conclusion: str = "",
        references: list[str] | None = None,
        figures: list[dict] | None = None,
        tables: list[dict] | None = None,
    ) -> str:
        result = inner.execute(
            title=title,
            results=results,
            abstract=abstract,
            keywords=keywords or [],
            introduction=introduction,
            related_work=related_work,
            methodology=methodology,
            setup=setup,
            analysis=analysis,
            conclusion=conclusion,
            references=references or [],
            figures=figures or [],
            tables=tables or [],
        )
        output = _truncate(result.output)
        return output if result.success else f"ERROR: {output}"

    tool = StructuredTool(
        name="generate_report",
        description=inner.description,
        func=_run,
        args_schema=GenerateReportInput,
    )






    tool._lab_forge_inner = inner
    return tool


def make_submit_result_tool(
    submit_tool: SubmitResultTool,
    working_dir: str | None = None,
) -> StructuredTool:
    """Wrap SubmitResultTool. Caller keeps a reference to submit_tool to check .submitted.

    When ``working_dir`` is provided, the tool blocks ``submit_result`` calls
    that come in before ``generate_report`` has produced
    ``research_report.md`` in the workspace. Without this guard the agent
    can declare "task complete" with only a textual summary, leaving the
    user with no markdown / bundle / PDF to download — but the chat panel
    happily congratulating the run as successful. The guard returns a
    tool-failure observation that explicitly tells the agent to call
    ``generate_report`` first; the run continues, the agent retries, and
    only the *real* success path ends with a downloadable report.
    """
    from pathlib import Path as _P

    def _run(result_path: str, summary: str) -> str:
        if working_dir:
            report_path = _P(working_dir) / "research_report.md"
            if not report_path.exists():
                return (
                    "ERROR: submit_result was REJECTED: there is no "
                    f"`research_report.md` in the workspace ({working_dir}). "
                    "Before calling submit_result you MUST call "
                    "`generate_report` with the verified findings, references, "
                    "figures and tables so a markdown report (and its "
                    "matching paperforge_bundle.json) is written to disk. "
                    "Call generate_report now, then call submit_result with "
                    "result_path=\"research_report.md\"."
                )
            findings = blocking_findings(validate_workspace_results(working_dir))
            if findings:
                try:
                    report_text = report_path.read_text(encoding="utf-8")
                except OSError:
                    report_text = ""
                if not report_text_discloses_findings(report_text + "\n" + summary, findings):
                    return (
                        "ERROR: submit_result was REJECTED by the result "
                        "guardrail.\n"
                        f"{format_findings(findings)}\n\n"
                        "The final report/summary does not clearly disclose "
                        "these invalid experiment results as failed or "
                        "unresolved. Fix the experiment, or regenerate the "
                        "report with an explicit caveat before submitting."
                    )
        result = submit_tool.execute(result_path=result_path, summary=summary)

        return f"TASK_COMPLETE: {result.output}"

    return StructuredTool(
        name="submit_result",
        description=submit_tool.description,
        func=_run,
        args_schema=SubmitResultInput,
    )






CODE_TOOL_NAMES = {
    "execute_code",
    "execute_bash",
    "file_read",
    "file_write",
    "list_files",
    "submit_result",
}

RESEARCH_TOOL_NAMES = CODE_TOOL_NAMES | {
    "search_literature",
    "read_paper_fulltext",
    "lookup_paper_code",
    "generate_report",
}


def create_all_tools(
    sandbox: Sandbox,
    working_dir: str,
    ocr_enabled: bool = False,
    submit_tool: SubmitResultTool | None = None,
    tool_names: set[str] | None = None,
    writer_llm=None,
    search_quota: int = 0,
    task_description: str = "",
) -> tuple[list[StructuredTool], SubmitResultTool]:
    """
    Create LangChain tools for the research agent.

    Args:
        sandbox: Code execution sandbox.
        working_dir: Workspace directory for file and report tools.
        ocr_enabled: Enable OCR-backed full-text reader.
        submit_tool: Optional pre-built submit tool; created if omitted.
        tool_names: Optional subset of tool names to include. Defaults to the
            full research tool set. Use ``CODE_TOOL_NAMES`` for benchmark runs.
        search_quota: Per-run cap on ``search_literature`` calls. ``0`` means
            no cap. The cap forces the agent to plan its searches up front
            rather than reactively re-searching.

    Returns:
        (tools_list, submit_tool_instance)
    """
    if submit_tool is None:
        submit_tool = SubmitResultTool()

    if tool_names is None:
        tool_names = RESEARCH_TOOL_NAMES

    factories = {
        "execute_code": lambda: make_execute_code_tool(sandbox),
        "execute_bash": lambda: make_execute_bash_tool(sandbox),
        "file_read": lambda: make_file_read_tool(sandbox),
        "file_write": lambda: make_file_write_tool(sandbox),
        "list_files": lambda: make_list_files_tool(sandbox),
        "search_literature": lambda: make_search_literature_tool(
            working_dir,
            search_quota=search_quota,
        ),
        "read_paper_fulltext": lambda: make_read_paper_fulltext_tool(sandbox, ocr_enabled=ocr_enabled),
        "lookup_paper_code": lambda: make_lookup_paper_code_tool(),
        "generate_report": lambda: make_generate_report_tool(
            working_dir, writer_llm=writer_llm, task_description=task_description,
        ),
        "submit_result": lambda: make_submit_result_tool(submit_tool, working_dir),
    }

    ordered = [
        "execute_code",
        "execute_bash",
        "file_read",
        "file_write",
        "list_files",
        "search_literature",
        "read_paper_fulltext",
        "lookup_paper_code",
        "generate_report",
        "submit_result",
    ]
    tools = [factories[name]() for name in ordered if name in tool_names]
    return tools, submit_tool
