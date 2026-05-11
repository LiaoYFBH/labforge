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
from .tools.scope_lock_tool import ScopeLockTool
from .tools.search_tool import SearchLiteratureTool
from .tools.submit_tool import SubmitResultTool


def _truncate(text: str, max_len: int = 4000) -> str:
    if len(text) <= max_len:
        return text
    return text[:max_len] + f"\n... (truncated, {len(text)} chars total)"


# ---------------------------------------------------------------------------
# Scope-lock hard-enforcement gate
# ---------------------------------------------------------------------------
#
# Every tool except ``submit_research_plan`` itself is wrapped so the agent
# sees a hard error until the plan is committed. Returning a string starting
# with "ERROR:" matches the existing convention all other gates use, so the
# trajectory step is recorded as a failure and the agent is steered back to
# the missing call without us having to touch the LangGraph wiring.

SCOPE_GATE_ERROR = (
    "ERROR: tool {tool_name!r} is BLOCKED — the research scope has not been "
    "locked yet.\n\n"
    "Before any other tool unlocks, you MUST call ``submit_research_plan`` "
    "ONCE with:\n"
    "  • natural_scope.methods (≥3 if the topic asks for comparison),\n"
    "  • natural_scope.axes_of_comparison,\n"
    "  • feasibility_split.will_execute / feasibility_split.literature_only,\n"
    "  • title_hypothesis (mirrors the topic, not the smallest dataset),\n"
    "  • rationale (2-4 sentences).\n\n"
    "This is a one-shot gate: one valid submit_research_plan call unblocks "
    "every other tool for the rest of the run."
)


def _scope_gated(tool_name: str, scope_lock: "ScopeLockTool | None"):
    """Return a ``(should_block, error_message)`` predicate closure.

    The closure returns ``(True, msg)`` when the tool must short-circuit and
    ``(False, "")`` when execution can proceed. ``scope_lock=None`` disables
    gating entirely, which is the path used by benchmark / unit-test runs
    that don't wire a ScopeLockTool.
    """
    if scope_lock is None:
        def _noop() -> tuple[bool, str]:
            return False, ""
        return _noop

    def _check() -> tuple[bool, str]:
        if scope_lock.is_locked():
            return False, ""
        return True, SCOPE_GATE_ERROR.format(tool_name=tool_name)

    return _check


# ---------------------------------------------------------------------------
# Pydantic input schemas
# ---------------------------------------------------------------------------

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


class _NaturalScopeInput(BaseModel):
    methods: list[str] = Field(
        default_factory=list,
        description=(
            "Distinct methods/algorithms/approaches the topic naturally "
            "calls for (≥3 when the topic asks for comparison)."
        ),
    )
    axes_of_comparison: list[str] = Field(
        default_factory=list,
        description=(
            "Dimensions along which methods will be contrasted — accuracy, "
            "sample efficiency, runtime, etc."
        ),
    )


class _FeasibilitySplitInput(BaseModel):
    will_execute: list[str] = Field(
        default_factory=list,
        description="Methods that will be run end-to-end with code.",
    )
    literature_only: list[str] = Field(
        default_factory=list,
        description=(
            "Methods covered via literature synthesis only (CPU/GPU/time/"
            "infra constraints). Do NOT leave empty just to shrink the topic."
        ),
    )


class SubmitResearchPlanInput(BaseModel):
    natural_scope: _NaturalScopeInput
    feasibility_split: _FeasibilitySplitInput
    title_hypothesis: str = Field(
        description=(
            "Paper title — mirrors the topic-level question, not the "
            "smallest dataset/env you'll run on."
        ),
    )
    rationale: str = Field(
        description=(
            "2-4 sentences justifying the methods, the execute/literature "
            "split, and the title."
        ),
    )


# ---------------------------------------------------------------------------
# Tool factory functions
# ---------------------------------------------------------------------------

def make_execute_code_tool(
    sandbox: Sandbox,
    scope_lock: "ScopeLockTool | None" = None,
) -> StructuredTool:
    inner = ExecuteCodeTool(sandbox)
    gate = _scope_gated("execute_code", scope_lock)

    def _run(code: str) -> str:
        blocked, msg = gate()
        if blocked:
            return msg
        result = inner.execute(code=code)
        output = _truncate(result.output)
        return output if result.success else f"ERROR: {output}"

    return StructuredTool(
        name="execute_code",
        description=inner.description,
        func=_run,
        args_schema=ExecuteCodeInput,
    )


def make_execute_bash_tool(
    sandbox: Sandbox,
    scope_lock: "ScopeLockTool | None" = None,
) -> StructuredTool:
    inner = ExecuteBashTool(sandbox)
    gate = _scope_gated("execute_bash", scope_lock)

    def _run(command: str) -> str:
        blocked, msg = gate()
        if blocked:
            return msg
        result = inner.execute(command=command)
        output = _truncate(result.output)
        return output if result.success else f"ERROR: {output}"

    return StructuredTool(
        name="execute_bash",
        description=inner.description,
        func=_run,
        args_schema=ExecuteBashInput,
    )


def make_file_read_tool(
    sandbox: Sandbox,
    scope_lock: "ScopeLockTool | None" = None,
) -> StructuredTool:
    inner = FileReadTool(sandbox)
    gate = _scope_gated("file_read", scope_lock)

    def _run(path: str) -> str:
        blocked, msg = gate()
        if blocked:
            return msg
        result = inner.execute(path=path)
        return _truncate(result.output) if result.success else f"ERROR: {result.output}"

    return StructuredTool(
        name="file_read",
        description=inner.description,
        func=_run,
        args_schema=FileReadInput,
    )


def make_file_write_tool(
    sandbox: Sandbox,
    scope_lock: "ScopeLockTool | None" = None,
) -> StructuredTool:
    inner = FileWriteTool(sandbox)
    gate = _scope_gated("file_write", scope_lock)

    def _run(path: str, content: str) -> str:
        blocked, msg = gate()
        if blocked:
            return msg
        result = inner.execute(path=path, content=content)
        output = _truncate(result.output)
        # Surface rejections (e.g. blocked data-file writes) with the same
        # "ERROR:" prefix other tools use, so the reviewer and phase tracker
        # register it as a failure and the agent knows to fall back to
        # execute_code.
        return output if result.success else f"ERROR: {output}"

    return StructuredTool(
        name="file_write",
        description=inner.description,
        func=_run,
        args_schema=FileWriteInput,
    )


def make_list_files_tool(
    sandbox: Sandbox,
    scope_lock: "ScopeLockTool | None" = None,
) -> StructuredTool:
    inner = ListFilesTool(sandbox)
    gate = _scope_gated("list_files", scope_lock)

    def _run(path: str = ".") -> str:
        blocked, msg = gate()
        if blocked:
            return msg
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
    scope_lock: "ScopeLockTool | None" = None,
) -> StructuredTool:
    inner = SearchLiteratureTool(working_dir=working_dir)
    if search_quota:
        inner.set_quota(search_quota)
    gate = _scope_gated("search_literature", scope_lock)

    def _run(query: str, max_results: int = 10) -> str:
        blocked, msg = gate()
        if blocked:
            return msg
        result = inner.execute(query=query, max_results=max_results)
        output = _truncate(result.output)
        return output if result.success else f"ERROR: {output}"

    return StructuredTool(
        name="search_literature",
        description=inner.description,
        func=_run,
        args_schema=SearchLiteratureInput,
    )


def make_lookup_paper_code_tool(
    scope_lock: "ScopeLockTool | None" = None,
) -> StructuredTool:
    inner = LookupPaperCodeTool()
    gate = _scope_gated("lookup_paper_code", scope_lock)

    def _run(arxiv_id: str) -> str:
        blocked, msg = gate()
        if blocked:
            return msg
        result = inner.execute(arxiv_id=arxiv_id)
        output = _truncate(result.output)
        return output if result.success else f"ERROR: {output}"

    return StructuredTool(
        name="lookup_paper_code",
        description=inner.description,
        func=_run,
        args_schema=LookupPaperCodeInput,
    )


def make_read_paper_fulltext_tool(
    sandbox: Sandbox,
    ocr_enabled: bool = False,
    scope_lock: "ScopeLockTool | None" = None,
) -> StructuredTool:
    inner = ReadPaperFullTextTool(sandbox, ocr_enabled=ocr_enabled)
    gate = _scope_gated("read_paper_fulltext", scope_lock)

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
        blocked, msg = gate()
        if blocked:
            return msg
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
    scope_lock: "ScopeLockTool | None" = None,
    fidelity_critic=None,
    outcome_judge_llm=None,
    vision_llm=None,
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
                # Some providers return content as a list of {"text": "..."}
                # blocks — flatten to a single string.
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
        fidelity_critic=fidelity_critic,
        outcome_judge_llm=outcome_judge_llm,
        vision_llm=vision_llm,
    )

    gate = _scope_gated("generate_report", scope_lock)

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
        blocked, msg = gate()
        if blocked:
            return msg
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
    # P5 fix: attach the inner GenerateReportTool to the StructuredTool so
    # the runtime caller (agent.run, etc.) can mutate ``task_description``
    # at task start. Without this hook the tool is built once at agent
    # __init__ and stays unaware of which task it is currently serving.
    # ``_lab_forge_inner`` is a private attribute name to avoid colliding
    # with future LangChain fields.
    tool._lab_forge_inner = inner  # type: ignore[attr-defined]
    return tool


def _read_reviewer_state(working_dir: str | None) -> dict | None:
    """Read the latest reviewer verdict mirrored by ``ReviewerCallback``.

    Returns ``None`` when no state file exists yet (typical for the first
    submit attempt when the report-review hasn't been triggered) or when
    the file is unreadable. Errors degrade silently to "no veto" so a
    broken state file can't deadlock the submit path.
    """
    if not working_dir:
        return None
    from pathlib import Path as _P

    from .callbacks import REVIEWER_STATE_FILENAME

    state_path = _P(working_dir) / REVIEWER_STATE_FILENAME
    if not state_path.exists():
        return None
    try:
        import json as _json
        return _json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def make_submit_result_tool(
    submit_tool: SubmitResultTool,
    working_dir: str | None = None,
    scope_lock: "ScopeLockTool | None" = None,
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

    from .callbacks import (
        REVIEWER_BLOCK_SCORE_THRESHOLD,
        REVIEWER_MAX_REWRITE_ATTEMPTS,
    )

    gate = _scope_gated("submit_result", scope_lock)
    # Counter for reviewer-driven rewrites. Lives on the SubmitResultTool
    # instance so it persists across retries within one run but resets
    # between runs (the agent reinstantiates the tool). Bounded by
    # REVIEWER_MAX_REWRITE_ATTEMPTS so a stuck reviewer can't deadlock
    # the run — same policy as the topic-fidelity critic.
    if not hasattr(submit_tool, "_reviewer_block_attempts"):
        submit_tool._reviewer_block_attempts = 0  # type: ignore[attr-defined]

    def _run(result_path: str, summary: str) -> str:
        blocked, msg = gate()
        if blocked:
            return msg
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

            # Reviewer-blocking gate. The reviewer callback persists its
            # latest report-checkpoint verdict to ``reviewer_state.json``;
            # if that verdict has score < threshold OR explicit issues, we
            # bounce the submit back to the agent with the must-fix list
            # — UP TO ``REVIEWER_MAX_REWRITE_ATTEMPTS`` times. After the
            # budget is exhausted we accept the submit but the report
            # tool's downstream consumer (Web UI / PDF export) sees
            # ``reviewer_unresolved_issues`` in the metadata and can
            # warn the user. This prevents the trajectory-194edbd4
            # failure mode where the agent ignored a Score=0.50 review
            # and submitted anyway, while NOT recreating the
            # death-loop the previous "veto" mechanism caused.
            reviewer_state = _read_reviewer_state(working_dir)
            if reviewer_state and not reviewer_state.get("is_infrastructure_error", False):
                score = float(reviewer_state.get("score", 1.0))
                issues = reviewer_state.get("issues") or []
                needs_fix = (
                    score < REVIEWER_BLOCK_SCORE_THRESHOLD
                    or (not reviewer_state.get("passed", True) and bool(issues))
                )
                if needs_fix:
                    attempts = getattr(submit_tool, "_reviewer_block_attempts", 0)
                    if attempts < REVIEWER_MAX_REWRITE_ATTEMPTS:
                        submit_tool._reviewer_block_attempts = attempts + 1  # type: ignore[attr-defined]
                        issues_text = "\n".join(f"  - {i}" for i in issues[:10])
                        return (
                            "ERROR: submit_result was REJECTED by the "
                            f"reviewer model (score {score:.2f} < "
                            f"{REVIEWER_BLOCK_SCORE_THRESHOLD}).\n\n"
                            f"Reviewer's must-fix list "
                            f"(rewrite attempt {attempts + 1}/"
                            f"{REVIEWER_MAX_REWRITE_ATTEMPTS}):\n"
                            f"{issues_text}\n\n"
                            f"Suggestion: {reviewer_state.get('suggestion', '')}\n\n"
                            "Re-call generate_report with these issues "
                            "addressed in the relevant sections, then call "
                            "submit_result again. After "
                            f"{REVIEWER_MAX_REWRITE_ATTEMPTS} bounced "
                            "attempts the submit will go through with the "
                            "reviewer's unresolved-issues list attached as "
                            "metadata so the user is warned in the UI."
                        )
                    # Budget exhausted — record for the caller's metadata
                    # so the UI / PDF export can show the warning.
                    submit_tool._reviewer_unresolved = {  # type: ignore[attr-defined]
                        "score": score,
                        "issues": issues,
                        "suggestion": reviewer_state.get("suggestion", ""),
                        "attempts_used": attempts + 1,
                    }
        result = submit_tool.execute(result_path=result_path, summary=summary)
        # Return a special marker that the custom executor can detect
        return f"TASK_COMPLETE: {result.output}"

    return StructuredTool(
        name="submit_result",
        description=submit_tool.description,
        func=_run,
        args_schema=SubmitResultInput,
    )


def _to_plain_dict(value) -> dict:
    """Coerce an LLM-supplied nested arg to a plain dict.

    LangChain instantiates nested Pydantic models from the args_schema
    (so ``natural_scope`` arrives as a ``_NaturalScopeInput`` instance,
    not a dict). ``ScopeLockTool.execute`` and ``ResearchPlan.from_dict``
    are dict-shaped on purpose — they're the same code paths used by
    ``load_plan_from_workspace`` reading JSON. We bridge the two here.

    Tolerant of: Pydantic v2 (``model_dump``), Pydantic v1 (``dict``),
    plain dicts, and ``None`` / arbitrary garbage (returns ``{}``).
    """
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        try:
            return dump()
        except Exception:
            pass
    legacy_dict = getattr(value, "dict", None)
    if callable(legacy_dict):
        try:
            return legacy_dict()
        except Exception:
            pass
    return {}


def make_scope_lock_tool(scope_lock: ScopeLockTool) -> StructuredTool:
    """LangChain wrapper for the mandatory ``submit_research_plan`` tool.

    Note: this tool intentionally does NOT pass through the scope-gate —
    it IS the gate. Calling it is what unlocks every other tool.
    """

    def _run(
        natural_scope=None,
        feasibility_split=None,
        title_hypothesis: str = "",
        rationale: str = "",
    ) -> str:
        result = scope_lock.execute(
            natural_scope=_to_plain_dict(natural_scope),
            feasibility_split=_to_plain_dict(feasibility_split),
            title_hypothesis=title_hypothesis or "",
            rationale=rationale or "",
        )
        output = _truncate(result.output)
        return output if result.success else f"ERROR: {output}"

    return StructuredTool(
        name="submit_research_plan",
        description=scope_lock.description,
        func=_run,
        args_schema=SubmitResearchPlanInput,
    )


# ---------------------------------------------------------------------------
# Convenience: create all tools at once
# ---------------------------------------------------------------------------

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
    "submit_research_plan",
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
    enforce_scope_lock: bool = True,
    fidelity_critic=None,
    outcome_judge_llm=None,
    vision_llm=None,
) -> tuple[list[StructuredTool], SubmitResultTool, "ScopeLockTool | None"]:
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
        enforce_scope_lock: When True (default), exposes ``submit_research_plan``
            and gates every other tool behind it. Benchmark / test callers
            that don't include "submit_research_plan" in ``tool_names`` get
            no gating regardless of this flag.
        fidelity_critic: Optional callable(topic, plan, draft) -> verdict
            wired into ``generate_report``. See ``topic_fidelity.py``.

    Returns:
        (tools_list, submit_tool_instance, scope_lock_tool_or_None)
    """
    if submit_tool is None:
        submit_tool = SubmitResultTool()

    if tool_names is None:
        tool_names = RESEARCH_TOOL_NAMES

    # Only instantiate ScopeLockTool when scope-lock is BOTH enabled and
    # listed in tool_names. Benchmark callers passing ``CODE_TOOL_NAMES``
    # (no submit_research_plan) get no gating — the closures see
    # scope_lock=None and short-circuit to no-op gates.
    scope_lock: ScopeLockTool | None = None
    if enforce_scope_lock and "submit_research_plan" in tool_names:
        scope_lock = ScopeLockTool(working_dir)

    factories = {
        "execute_code": lambda: make_execute_code_tool(sandbox, scope_lock=scope_lock),
        "execute_bash": lambda: make_execute_bash_tool(sandbox, scope_lock=scope_lock),
        "file_read": lambda: make_file_read_tool(sandbox, scope_lock=scope_lock),
        "file_write": lambda: make_file_write_tool(sandbox, scope_lock=scope_lock),
        "list_files": lambda: make_list_files_tool(sandbox, scope_lock=scope_lock),
        "search_literature": lambda: make_search_literature_tool(
            working_dir,
            search_quota=search_quota,
            scope_lock=scope_lock,
        ),
        "read_paper_fulltext": lambda: make_read_paper_fulltext_tool(
            sandbox, ocr_enabled=ocr_enabled, scope_lock=scope_lock,
        ),
        "lookup_paper_code": lambda: make_lookup_paper_code_tool(scope_lock=scope_lock),
        "generate_report": lambda: make_generate_report_tool(
            working_dir,
            writer_llm=writer_llm,
            task_description=task_description,
            scope_lock=scope_lock,
            fidelity_critic=fidelity_critic,
            outcome_judge_llm=outcome_judge_llm,
            vision_llm=vision_llm,
        ),
        "submit_result": lambda: make_submit_result_tool(
            submit_tool, working_dir, scope_lock=scope_lock,
        ),
        "submit_research_plan": (
            (lambda: make_scope_lock_tool(scope_lock)) if scope_lock is not None else None
        ),
    }

    ordered = [
        # submit_research_plan is listed FIRST so the agent sees it at the
        # top of the tool list — the order the LLM scans matters when it
        # picks an opening move.
        "submit_research_plan",
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
    tools = [
        factories[name]()
        for name in ordered
        if name in tool_names and factories.get(name) is not None
    ]
    return tools, submit_tool, scope_lock
