"""
Report generation tool — produces a structured Markdown research report.

Inspired by AI-Scientist (LaTeX paper generation) and AutoResearchClaw
(conference-format output), adapted for lightweight Markdown output.

This tool now uses a two-phase generation pipeline:
  1. Build the bundle from the agent-supplied section drafts (the
     "outline" pass — short bullet-y notes are fine here).
  2. If a writer-LLM is wired in via the factory, expand each section to
     top-conference depth one-by-one, with explicit word targets, must-
     cover checklists, and an anti-fabrication clamp so we never invent
     numbers or citations the agent didn't actually verify.

The per-section LLM call is intentionally separate from the agent's main
LLM so the agent can stay cheap on tool decisions while the writer model
produces long-form prose. When no writer LLM is provided we fall back to
the original behaviour (single-pass bundle build) — useful in tests and
benchmarks where length isn't the bottleneck.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Callable

from lab_forge.paper_bundle import (
    build_paper_bundle,
    paper_bundle_to_markdown,
    save_paper_bundle,
)
from lab_forge.result_guardrails import (
    blocking_findings,
    format_findings,
    report_text_discloses_findings,
    validate_workspace_results,
)

# Reuse paper_forge's LLM-output cleaner so the section-expansion path here
# benefits from R1+R2+R3 (\{}command -> \command, "Here is the polished..."
# prefix removal, \textbf->markdown, display-math isolation, etc.). The
# previous version just took the LLM string raw, which let \{} pollution
# and meta-narration prefixes ride straight into the bundle and the
# downstream PDF (trajectory 203c6c5c paper.tex line 69 had ~50 \{}
# residues from this exact gap).
try:
    from paper_forge.paper_writer import _clean_llm_section_output as _pf_clean_section
except ImportError:  # paper_forge unavailable in some test environments
    _pf_clean_section = None

from .base import Tool, ToolResult

logger = logging.getLogger(__name__)

FIGURE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".svg"}
TABLE_EXTENSIONS = {".csv", ".tsv"}


# Section name -> (target_words, minimum_words, role description, must-cover bullets).
# These mirror paper_forge.style_guide.SECTION_BLUEPRINTS but live here so the
# expansion still works when paper_forge isn't on the path (e.g. in tests).
SECTION_TARGETS: dict[str, dict[str, Any]] = {
    "introduction": {
        "target_words": 800,
        "minimum_words": 550,
        "role": "Introduction",
        "must_cover": (
            "领域宏观背景与意义",
            "明确的研究 gap / 挑战",
            "现有方法的至少 2 类不足",
            "本文方法的核心 insight",
            "至少 3 条 contribution（用粗体动词开头）",
            "论文组织结构一句话",
        ),
    },
    "related_work": {
        "target_words": 700,
        "minimum_words": 450,
        "role": "Related Work",
        "must_cover": (
            "按 2-3 个主题分组讨论",
            "每个主题给出代表性方法 + 关键差异",
            "结尾点出本文与已有工作的本质区别",
        ),
    },
    "methodology": {
        "target_words": 1100,
        "minimum_words": 750,
        "role": "Methodology",
        "must_cover": (
            "Problem formulation（输入/输出/约束）",
            "核心 insight 一句话",
            "正式的算法描述（编号步骤或伪代码）",
            "至少 1 个关键公式",
            "对方法可行性 / 复杂度的简要论证",
        ),
    },
    "setup": {
        "target_words": 600,
        "minimum_words": 400,
        "role": "Experimental Setup",
        "must_cover": (
            "数据集来源与划分",
            "评估指标的明确定义",
            "训练 / 推理超参数（≥ 3 个具体数值，全部来自实际执行）",
            "对比方法 / baseline 列表",
            "实现与硬件细节",
        ),
    },
    "results": {
        "target_words": 1100,
        "minimum_words": 700,
        "role": "Results",
        "must_cover": (
            "主结果表的解读（不能只列数字）",
            "至少 1 组消融或敏感性分析",
            "把每个数字与方法/实验设计联系起来",
            "对意外或失败结果的诚实讨论",
        ),
    },
    "analysis": {
        "target_words": 700,
        "minimum_words": 450,
        "role": "Analysis & Discussion",
        "must_cover": (
            "对结果做机制层面的解释",
            "至少 2 条本方法的局限",
            "至少 2 条具体可执行的未来方向",
        ),
    },
    "conclusion": {
        "target_words": 350,
        "minimum_words": 220,
        "role": "Conclusion",
        "must_cover": (
            "一句话点题：本文做了什么",
            "1-2 句话总结代表性量化结论",
            "对领域的影响一句话",
        ),
    },
}


# Type alias: a callable that takes a single string prompt and returns the
# model's reply. Lets us inject a langchain ChatOpenAI / Anthropic / mock
# without coupling the tool to a specific provider.
WriterLLM = Callable[[str], str]


def _word_count(text: str) -> int:
    """Cheap word/CJK char count — good enough for length checks."""
    if not text:
        return 0
    cjk = sum(1 for ch in text if "一" <= ch <= "鿿")
    ascii_words = len([w for w in text.replace("\n", " ").split() if w.strip()])
    return cjk + ascii_words


def _clean_writer_llm_output(raw: Any) -> str:
    """Run the LLM-output cleaner on a writer_llm response.

    Falls back to a plain ``str(raw).strip()`` when paper_forge is not
    importable (e.g. the test environment). When paper_forge IS available
    (the normal case in production), the call routes through R1+R2+R3:

      * ``\\{}command`` -> ``\\command`` (undo LLM's self-escape)
      * "Here is the polished … section:" prefix removed
      * ``\\textbf{x}`` -> ``**x**`` so markdown path renders correctly
      * display-math blocks isolated with surrounding blank lines
      * ``\\title{...}`` and other preamble commands stripped from prose

    Without this, the trajectory 203c6c5c case shipped a paper with ~50
    literal ``\\{}`` residues in the rendered PDF.
    """
    if raw is None:
        return ""
    text = raw if isinstance(raw, str) else str(raw)
    if _pf_clean_section is None:
        return text.strip()
    try:
        return _pf_clean_section(text)
    except Exception:
        logger.exception("paper_forge cleaner threw; falling back to raw strip")
        return text.strip()


def _format_bullets(items: tuple[str, ...]) -> str:
    return "\n".join(f"- {item}" for item in items)


EXPANSION_PROMPT = """\
你正在协助撰写一篇顶会风格的科研论文。请把下面的 **{role}** 草稿扩写成符合
NeurIPS / ICML / ACL 等顶会风格的章节正文。

# 严格规则（违反任何一条都视为整段无效）
1. 仅基于"运行事实清单"和"原始草稿"中已经出现的事实写作。
2. 严禁编造数字、百分比、运行时间、引用、作者、数据集名、模型名、硬件型号。
3. 训练超参（max_iter / kernel / hidden_layer_sizes / batch_size / epochs / 学习率
   等）、数据集划分大小、随机种子、baseline 名单、硬件、训练耗时——只能取自
   "运行事实清单"。清单未提及的，明确写"未在本次运行中记录"，绝不补默认值。
4. 引用沿用底稿里的占位符（例如 [1]、[ref-N]），不要新造引用编号。
5. 公式用 LaTeX 写真实数学表达式（``$...$`` 或 ``$$...$$``），不要用纯文字。
6. 第一人称用 "we / 我们"。每段以一句主题句开头，再展开。

# 运行事实清单（机器抽取，唯一可信的硬件/超参/数据/告警来源）
---
{run_facts}
---

# 写作目标
- 章节角色：{role}
- 目标字数：约 {target_words} 字（最少 {minimum_words} 字，越接近顶会
  正文密度越好）
- 必须涵盖的子要素：
{must_cover_bullets}

# 论文上下文（仅供你建立全局观，不要直接复用其原文）
- 论文标题：{title}
- 摘要预览：{abstract_preview}
- 已有引用占位符：{references_preview}

# 章节草稿（事实底稿）
---
{content}
---

# 输出
- 直接输出 Markdown 段落正文，不要重复章节标题。
- 不要使用 ``# / ##`` 一级 / 二级标题；如果需要小节请使用 ``### 子节标题``。
- 不要在开头写"Here is the expanded ... version:"、"以下是改写后的..."等元话术。
- 不要在末尾加 "Word count: N"、"This revision expands..."、
  "数学补充（根据需要嵌入）："等任何元说明。
"""

OUTLINE_NUDGE = """\
检测到 **{role}** 草稿过短（{actual_words} 字）。请按以下要点先在内部列一个详细
大纲（不必输出大纲），再输出展开后的正文，确保覆盖所有 must-cover 项：

{must_cover_bullets}

运行事实清单（数字/超参/告警唯一可信源，不在清单里的内容请写"未在本次运行中记录"）：
---
{run_facts}
---

事实底稿：
---
{content}
---

输出要求：纯正文 Markdown，目标 {target_words} 字。
不要写元话术（"Here is the expanded..."、"Word count: ..."、"This revision expands..."等）。"""


class GenerateReportTool(Tool):
    """Generate a structured Markdown research report from the agent's findings."""

    def __init__(
        self,
        working_dir: str,
        writer_llm: WriterLLM | None = None,
        expand_sections: bool = True,
        task_description: str = "",
    ):
        self.working_dir = Path(working_dir)
        # Optional writer-LLM used for the per-section expansion pass. When
        # None we keep the legacy single-pass behaviour so unit tests don't
        # need to wire up a model.
        self.writer_llm = writer_llm
        self.expand_sections = expand_sections
        # P5 fix: store the task_description so we can detect SURVEY ONLY
        # mode in ``_validate_experiment_evidence`` and skip the figure /
        # table requirement. Without this, a survey-only run is forced to
        # call execute_code purely to fabricate placeholder artifacts to
        # pass the gate (trajectory 477a3605 generated a ``placeholder.png``
        # literally containing "NO EXPERIMENTAL RESULTS (SURVEY ONLY TASK)"
        # just to clear the validator).
        self.task_description = task_description or ""

    def _is_survey_mode(self) -> bool:
        """Whether this run is a literature-survey-only task.

        Detection is text-based: ``workflow.topic_to_task`` writes a
        ``[MODE: SURVEY ONLY ...]`` marker into ``task_description``
        whenever the topic classifier picks the survey path. We treat
        that marker as authoritative — the agent is explicitly told not
        to run experiments in this mode, so a missing-figure / missing-
        table block here would be punishing the agent for following its
        instructions.
        """
        return "[MODE: SURVEY ONLY" in (self.task_description or "")

    @property
    def name(self) -> str:
        return "generate_report"

    @property
    def description(self) -> str:
        return (
            "Generate a structured Markdown research report. "
            "Provide the report title, content for each section (an outline / "
            "draft is fine — the tool will expand each section to top-conference "
            "depth automatically), verified references, and the real saved "
            "figure/table files. This tool refuses to run until experiments "
            "have produced successful execution logs, at least one CSV/TSV "
            "result table, and at least one PNG/JPG/SVG figure. The tool saves "
            "both a Markdown report and a PaperForge bundle in the workspace. "
            "Call this only after experiments and analysis are complete."
        )

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "Title of the research report."},
                "abstract": {
                    "type": "string",
                    "description": "Short abstract summarizing the task, methods, and findings.",
                },
                "keywords": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional paper keywords.",
                },
                "introduction": {
                    "type": "string",
                    "description": (
                        "Outline / draft for the Introduction. Bullet points or a "
                        "short paragraph are fine — the tool will expand it."
                    ),
                },
                "related_work": {
                    "type": "string",
                    "description": "Outline / draft for the Related Work section.",
                },
                "methodology": {
                    "type": "string",
                    "description": "Outline / draft for the Methodology section.",
                },
                "setup": {
                    "type": "string",
                    "description": "Outline / draft for the Experimental Setup.",
                },
                "results": {
                    "type": "string",
                    "description": (
                        "Experimental results with the actual numbers and tables. "
                        "Numbers MUST come from real execution outputs."
                    ),
                },
                "analysis": {
                    "type": "string",
                    "description": "Outline / draft for Analysis and Discussion.",
                },
                "conclusion": {
                    "type": "string",
                    "description": "Outline / draft for the Conclusion.",
                },
                "references": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Verified references used in the report. Only include papers "
                        "that were actually found via literature search or full-text reading."
                    ),
                },
                "figures": {
                    "type": "array",
                    "description": "Saved chart/image files that should be passed to PaperForge.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string"},
                            "caption": {"type": "string"},
                            "section": {"type": "string"},
                        },
                        "required": ["path"],
                    },
                },
                "tables": {
                    "type": "array",
                    "description": "Saved CSV/TSV result tables that should be passed to PaperForge.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string"},
                            "caption": {"type": "string"},
                            "section": {"type": "string"},
                        },
                        "required": ["path"],
                    },
                },
            },
            "required": ["title", "results"],
        }

    def execute(
        self,
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
    ) -> ToolResult:
        # Deterministic correctness gate. If the workspace contains machine-
        # detectable invalid results (inf / NaN / negative convergence rate)
        # the paper may proceed only when the agent explicitly frames them as
        # invalid or unresolved. This prevents the "broken experiment -> glossy
        # paper" failure mode even when the main LLM misses the issue.
        result_findings = blocking_findings(
            validate_workspace_results(self.working_dir)
        )
        report_fact_text = self._combined_report_text(
            title=title,
            abstract=abstract,
            keywords=keywords or [],
            introduction=introduction,
            related_work=related_work,
            methodology=methodology,
            setup=setup,
            results=results,
            analysis=analysis,
            conclusion=conclusion,
            references=references or [],
            figures=figures or [],
            tables=tables or [],
        )
        guardrail_summary = ""
        if result_findings:
            guardrail_summary = format_findings(result_findings)
            if not report_text_discloses_findings(report_fact_text, result_findings):
                return ToolResult(
                    output=(
                        f"{guardrail_summary}\n\n"
                        "generate_report was rejected because the report draft "
                        "does not explicitly disclose these invalid experiment "
                        "results as failures or unresolved issues. Rerun/fix the "
                        "experiment, or call generate_report again with results, "
                        "analysis, and conclusion fields that clearly state the "
                        "non-finite/invalid values and avoid positive claims."
                    ),
                    success=False,
                    metadata={"result_guardrail_findings": [f.__dict__ for f in result_findings]},
                )

        # Citation guardrail (Mod 1d): every reference the agent passes must
        # trace back to a literature_cache.jsonl hit (search_literature) or a
        # papers/**/manifest.json record (read_paper_fulltext). This prevents
        # the failure mode we saw repeatedly in 2026-04 trajectories where
        # the agent invented industry-report citations like "Gartner 2023" /
        # "MarketReport 2024" that no actual search produced.
        explicit_refs_input = list(references or [])
        unverified = self._find_unverified_references(explicit_refs_input)
        if unverified:
            return ToolResult(
                output=(
                    "generate_report rejected: "
                    f"{len(unverified)}/{len(explicit_refs_input)} reference(s) "
                    "did not match anything in this run's literature cache. "
                    "Every citation MUST trace back to a `search_literature` "
                    "result or a `read_paper_fulltext` success in THIS run.\n\n"
                    "Unverified references:\n"
                    + "\n".join(f"  - {ref}" for ref in unverified)
                    + "\n\nFix: either (a) drop these citations and rewrite "
                    "the affected paragraphs without them, or (b) call "
                    "search_literature with the paper's title/keywords and "
                    "verify it appears in the results before regenerating "
                    "the report. Do NOT cite from background knowledge."
                ),
                success=False,
                metadata={"unverified_references": unverified},
            )

        # Agents tend to forget to forward ``references=`` / ``figures=`` /
        # ``tables=`` even when their run produced all three. Rather than
        # let the bundle ship empty (and the rendered PDF therefore have
        # zero refs and zero figures, which is exactly what the user
        # complained about), we autodiscover from the workspace whenever
        # the agent left a slot blank. The agent's explicit values still
        # win — autodiscovery only fills the gaps.
        autodiscovery_log: list[str] = []
        references, ref_log = self._merge_references_with_cache(explicit_refs_input)
        if ref_log:
            autodiscovery_log.append(ref_log)
        if not figures:
            figures, fig_log = self._discover_figures_from_workspace()
            if fig_log:
                autodiscovery_log.append(fig_log)
        if not tables:
            tables, tbl_log = self._discover_tables_from_workspace()
            if tbl_log:
                autodiscovery_log.append(tbl_log)

        # P0a path-existence gate (REFACTOR_PLAN.md follow-up).
        #
        # Agents under a tight step budget sometimes pass figure/table paths
        # that don't actually exist on disk — e.g. trajectory c032bbf1
        # passed ``figures=[{'path': 'training_loss.png'}]`` and
        # ``tables=[{'path': 'optimizer_comparison.csv'}]`` despite never
        # calling plt.savefig / to_csv. The bundle then ships with broken
        # references; paper_writer LLM happily expands prose around the
        # nonexistent figures, the PDF compiler embeds nothing, and the
        # final paper is fabricated. Cheaper to catch it here.
        path_gate = self._validate_artifact_paths(figures or [], tables or [])
        if path_gate:
            return ToolResult(
                output=path_gate,
                success=False,
                metadata={"missing_artifact_paths": True},
            )

        experiment_gate = self._validate_experiment_evidence(figures or [], tables or [])
        if experiment_gate:
            return ToolResult(
                output=experiment_gate,
                success=False,
                metadata={"experiment_evidence_missing": True},
            )

        section_drafts: dict[str, str] = {
            "introduction": introduction,
            "related_work": related_work,
            "methodology": methodology,
            "setup": setup,
            "results": results,
            "analysis": analysis,
            "conclusion": conclusion,
        }

        expansion_log: list[str] = []
        if self.writer_llm is not None and self.expand_sections:
            section_drafts = self._expand_all_sections(
                title=title,
                abstract=abstract,
                references=references or [],
                drafts=section_drafts,
                log=expansion_log,
            )

        bundle = build_paper_bundle(
            working_dir=self.working_dir,
            title=title,
            abstract=abstract,
            keywords=keywords or [],
            introduction=section_drafts["introduction"],
            related_work=section_drafts["related_work"],
            methodology=section_drafts["methodology"],
            setup=section_drafts["setup"],
            results=section_drafts["results"],
            analysis=section_drafts["analysis"],
            conclusion=section_drafts["conclusion"],
            references=references or [],
            figures=figures or [],
            tables=tables or [],
        )
        report_content = paper_bundle_to_markdown(bundle)

        generated_files = self._list_artifacts()
        if generated_files:
            report_content += "\n## Appendix: Generated Files\n\n"
            report_content += "\n".join(f"- `{name}`" for name in generated_files)
            report_content += "\n"

        report_content += "\n---\n*Generated by LabForge*\n"

        report_name = "research_report.md"
        report_path = self.working_dir / report_name
        report_path.write_text(report_content, encoding="utf-8")
        bundle_name = "paperforge_bundle.json"
        bundle_path = save_paper_bundle(bundle, self.working_dir / bundle_name)

        expansion_summary = ""
        if expansion_log:
            expansion_summary = "\n\n[Expansion log]\n" + "\n".join(
                f"- {line}" for line in expansion_log
            )

        autodiscovery_summary = ""
        if autodiscovery_log:
            autodiscovery_summary = "\n\n[Auto-discovery]\n" + "\n".join(
                f"- {line}" for line in autodiscovery_log
            )

        guardrail_status = ""
        if guardrail_summary:
            guardrail_status = "\n\n[Result guardrail]\n" + guardrail_summary

        return ToolResult(
            output=(
                f"Report saved to {report_name}\n"
                f"PaperForge bundle saved to {bundle_path.name}"
                f"{guardrail_status}"
                f"{autodiscovery_summary}"
                f"{expansion_summary}\n\n"
                f"{report_content[:2000]}"
            ),
            success=True,
            metadata={
                "report_path": report_name,
                "bundle_path": bundle_name,
                "references_count": len(bundle.get("references", [])),
                "expansion_log": expansion_log,
                "autodiscovery_log": autodiscovery_log,
                "result_guardrail_findings": [f.__dict__ for f in result_findings],
            },
        )

    @staticmethod
    def _combined_report_text(**values: Any) -> str:
        parts: list[str] = []

        def _add(value: Any) -> None:
            if value is None:
                return
            if isinstance(value, str):
                parts.append(value)
            elif isinstance(value, dict):
                for item in value.values():
                    _add(item)
            elif isinstance(value, (list, tuple, set)):
                for item in value:
                    _add(item)
            else:
                parts.append(str(value))

        for value in values.values():
            _add(value)
        return "\n".join(parts)

    # ------------------------------------------------------------------
    # Per-section expansion pipeline
    # ------------------------------------------------------------------

    def _expand_all_sections(
        self,
        *,
        title: str,
        abstract: str,
        references: list[str],
        drafts: dict[str, str],
        log: list[str],
    ) -> dict[str, str]:
        """Expand every non-empty section in turn using the writer LLM.

        On any per-section failure we keep the original draft so the report
        is still produced; the failure is recorded in ``log`` and surfaced
        in the tool result so the agent / UI knows to inspect it.
        """
        abstract_preview = (abstract or "")[:280] or "(empty)"
        references_preview = self._format_references_preview(references)
        # Compute run_facts ONCE per generate_report call — scanning logs/ and
        # CSVs twice for every section is wasted IO.
        run_facts = self._collect_run_facts()

        expanded: dict[str, str] = {}
        for key, draft in drafts.items():
            text = (draft or "").strip()
            if not text:
                expanded[key] = ""
                continue
            target = SECTION_TARGETS.get(key)
            if target is None:
                expanded[key] = text
                continue

            try:
                rewritten = self._expand_one(
                    role=target["role"],
                    target_words=target["target_words"],
                    minimum_words=target["minimum_words"],
                    must_cover=target["must_cover"],
                    title=title,
                    abstract_preview=abstract_preview,
                    references_preview=references_preview,
                    run_facts=run_facts,
                    content=text,
                )
                final = rewritten.strip() or text
                actual = _word_count(final)
                log.append(
                    f"{key}: {_word_count(text)} → {actual} words "
                    f"(target {target['target_words']}, min {target['minimum_words']})"
                )
                expanded[key] = final
            except Exception as exc:
                # Don't lose the agent's draft if the writer LLM fails — fall
                # back to the original outline so the bundle is still valid.
                logger.warning("Section expansion failed for '%s': %s", key, exc)
                log.append(f"{key}: expansion FAILED ({exc}); kept original draft")
                expanded[key] = text

        return expanded

    def _expand_one(
        self,
        *,
        role: str,
        target_words: int,
        minimum_words: int,
        must_cover: tuple[str, ...],
        title: str,
        abstract_preview: str,
        references_preview: str,
        run_facts: str,
        content: str,
    ) -> str:
        """Call the writer LLM once, retry once if the result is too short."""
        prompt = EXPANSION_PROMPT.format(
            role=role,
            target_words=target_words,
            minimum_words=minimum_words,
            must_cover_bullets=_format_bullets(must_cover),
            title=title,
            abstract_preview=abstract_preview,
            references_preview=references_preview,
            run_facts=run_facts,
            content=content[:8000],
        )
        rewritten = _clean_writer_llm_output(self.writer_llm(prompt))

        if _word_count(rewritten) < minimum_words:
            logger.info(
                "Section '%s' came back short (%d < %d) — retrying with explicit nudge",
                role, _word_count(rewritten), minimum_words,
            )
            retry_prompt = OUTLINE_NUDGE.format(
                role=role,
                actual_words=_word_count(rewritten),
                must_cover_bullets=_format_bullets(must_cover),
                run_facts=run_facts,
                content=rewritten or content[:8000],
                target_words=target_words,
            )
            retry = _clean_writer_llm_output(self.writer_llm(retry_prompt))
            if _word_count(retry) > _word_count(rewritten):
                rewritten = retry

        return rewritten

    @staticmethod
    def _format_references_preview(references: list[str]) -> str:
        if not references:
            return "(no references provided — do not invent any)"
        preview = "; ".join(ref[:80] for ref in references[:8])
        if len(references) > 8:
            preview += f" … (+{len(references) - 8} more)"
        return preview

    # Cap run_facts to keep the writer LLM's prompt under ~6 KB. Anything past
    # this is almost certainly redundant log output, not extra hyperparameter
    # ground truth.
    _RUN_FACTS_BUDGET_CHARS = 6000
    _RUN_FACTS_PER_SCRIPT_BUDGET = 1500
    _RUN_FACTS_PER_CSV_ROW_LIMIT = 8

    def _collect_run_facts(self) -> str:
        """Snapshot of "what the run actually did" — the only ground truth
        the writer LLM is allowed to draw hyperparameters / hardware /
        baselines / convergence status from.

        Without this, ``_expand_one`` only gives the LLM the agent's outline
        and a short reference preview. The LLM then invents plausible-but-fake
        numbers (trajectory 5bfb6c8d's "Xeon E5-2678 v3 / 23.5 GB",
        "5 repetitions with seeds 2023–2027", "k-NN / Random Forest / LeNet-5
        baselines") because it has no other anchor. Surfacing the actual
        scripts + CSV headers + stderr warnings closes that gap.
        """
        parts: list[str] = []

        logs_dir = self.working_dir / "logs"
        if logs_dir.exists():
            for log_path in sorted(logs_dir.glob("*.log")):
                try:
                    text = log_path.read_text(encoding="utf-8", errors="replace")
                except OSError:
                    continue
                if not re.search(r"^tool\s*:\s*execute_code\s*$", text, re.MULTILINE):
                    continue
                exit_match = re.search(r"^exit_code\s*:\s*(\S+)\s*$", text, re.MULTILINE)
                exit_code = exit_match.group(1) if exit_match else "?"
                script_match = re.search(
                    r"---- SCRIPT ----\n(.*?)\n---- STDOUT ----", text, re.DOTALL
                )
                stderr_match = re.search(
                    r"---- STDERR ----\n(.*?)\n---- END ----", text, re.DOTALL
                )
                if not script_match:
                    continue
                script_body = script_match.group(1).strip()
                preview = script_body[: self._RUN_FACTS_PER_SCRIPT_BUDGET]
                if len(script_body) > len(preview):
                    preview += "\n# ... (truncated)"
                parts.append(f"### {log_path.name}  (exit_code={exit_code})")
                parts.append("```python")
                parts.append(preview)
                parts.append("```")
                stderr_body = stderr_match.group(1).strip() if stderr_match else ""
                warnings = []
                seen: set[str] = set()
                for line in stderr_body.splitlines():
                    lower = line.lower()
                    if (
                        "convergencewarning" in lower
                        or "did not converge" in lower
                        or "not converged" in lower
                        or "solver terminated early" in lower
                        or "max_iter" in lower and ("reach" in lower or "limit" in lower)
                    ):
                        key = line.strip()[:120]
                        if key in seen:
                            continue
                        seen.add(key)
                        warnings.append(line.strip()[:200])
                    if len(warnings) >= 6:
                        break
                if warnings:
                    parts.append(
                        "Stderr warnings (training did NOT fully converge — "
                        "treat these metrics as a smoke test, not a final result):"
                    )
                    for w in warnings:
                        parts.append(f"- {w}")

        for csv_path in sorted(self.working_dir.rglob("*.csv")):
            try:
                rel = csv_path.relative_to(self.working_dir)
            except ValueError:
                continue
            if self._is_skipped_path(rel):
                continue
            try:
                with csv_path.open("r", encoding="utf-8-sig") as f:
                    rows: list[str] = []
                    for i, line in enumerate(f):
                        if i >= self._RUN_FACTS_PER_CSV_ROW_LIMIT:
                            break
                        rows.append(line.rstrip("\r\n"))
            except OSError:
                continue
            if not rows:
                continue
            parts.append(
                f"### {rel.as_posix()}  (CSV first {len(rows)} row(s) — "
                "use these numbers verbatim; do not round or invent additional rows)"
            )
            parts.append("```")
            parts.extend(rows)
            parts.append("```")

        if not parts:
            return (
                "(no execute_code scripts under logs/ and no *.csv result "
                "files were found — do NOT invent hyperparameters, hardware, "
                "baselines, or numeric results. Write 'no experiment was run "
                "in this trajectory' wherever Setup / Results / Analysis "
                "would otherwise quote numbers.)"
            )

        facts_text = "\n".join(parts)
        if len(facts_text) > self._RUN_FACTS_BUDGET_CHARS:
            facts_text = (
                facts_text[: self._RUN_FACTS_BUDGET_CHARS]
                + "\n... (truncated; see workspace files for the rest)"
            )
        return facts_text

    def _list_artifacts(self) -> list[str]:
        """List relevant output files in workspace."""
        if not self.working_dir.exists():
            return []
        relevant_exts = {".png", ".jpg", ".csv", ".json", ".txt", ".pdf", ".svg"}
        files = []
        for f in sorted(self.working_dir.rglob("*")):
            if f.is_file() and f.suffix.lower() in relevant_exts:
                try:
                    files.append(str(f.relative_to(self.working_dir)))
                except ValueError:
                    files.append(f.name)
        return files

    # ------------------------------------------------------------------
    # Auto-discovery — used when the agent forgot to forward references /
    # figures / tables. We never invent content, only surface what's
    # actually on disk in the run's workspace.
    # ------------------------------------------------------------------

    # Subdirectories where auto-discovery should NOT look. Files here
    # belong to the user (uploads), the OCR cache (papers), or the
    # framework's own bookkeeping (logs, dictionary caches).
    _AUTODISCOVERY_SKIP_DIRS = (
        "uploads", "papers", "logs", ".remote_images",
        "literature_cache", "data", "node_modules",
    )

    def _is_skipped_path(self, rel_path) -> bool:
        parts = rel_path.parts if hasattr(rel_path, "parts") else ()
        return any(part in self._AUTODISCOVERY_SKIP_DIRS for part in parts)

    def _discover_references_from_cache(self) -> tuple[list[str], str]:
        """Read ``literature_cache.jsonl`` and turn it into a refs list.

        Returns ``(refs, log_line)`` where ``log_line`` is a one-line
        summary suitable for the tool's autodiscovery log, or empty
        when nothing was found.
        """
        cache_path = self.working_dir / "literature_cache.jsonl"
        if not cache_path.exists():
            return [], ""
        try:
            import json as _json
            seen: set[str] = set()
            refs: list[str] = []
            with cache_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = _json.loads(line)
                    except _json.JSONDecodeError:
                        continue
                    title = (rec.get("title") or "").strip()
                    if not title:
                        continue
                    authors = (rec.get("authors") or "").strip()
                    year = rec.get("year") or "n.d."
                    url = rec.get("url") or ""
                    key = title.lower()
                    if key in seen:
                        continue
                    seen.add(key)
                    cite = (
                        f"{authors} ({year}). {title}."
                        if authors
                        else f"{title} ({year})."
                    )
                    if url:
                        cite += f" {url}"
                    refs.append(cite)
            if not refs:
                return [], ""
            return (
                refs,
                f"references: agent left empty, auto-attached {len(refs)} entries from literature_cache.jsonl",
            )
        except Exception as exc:
            logger.warning("Could not read literature cache: %s", exc)
            return [], ""

    def _discover_references_from_fulltext_manifests(self) -> tuple[list[str], str]:
        """Read ``papers/**/manifest.json`` files created by read_paper_fulltext."""
        papers_root = self.working_dir / "papers"
        if not papers_root.exists():
            return [], ""
        try:
            import json as _json
            seen: set[str] = set()
            refs: list[str] = []
            for manifest_path in sorted(papers_root.glob("**/manifest.json")):
                try:
                    rec = _json.loads(manifest_path.read_text(encoding="utf-8"))
                except Exception:
                    continue
                title = (rec.get("title") or "").strip()
                if not title:
                    continue
                key = title.lower()
                if key in seen:
                    continue
                seen.add(key)
                url = (
                    (rec.get("resolved_url") or "").strip()
                    or (rec.get("pdf_url") or "").strip()
                    or (rec.get("original_input") or "").strip()
                )
                cite = title
                if url:
                    cite += f". {url}"
                refs.append(cite)
            if not refs:
                return [], ""
            return (
                refs,
                f"references: auto-attached {len(refs)} full-text paper(s) from papers/**/manifest.json",
            )
        except Exception as exc:
            logger.warning("Could not read full-text paper manifests: %s", exc)
            return [], ""

    @staticmethod
    def _normalize_for_match(text: str) -> str:
        """Lowercase + alphanumeric-only + collapse whitespace, for fuzzy match."""
        if not text:
            return ""
        norm = re.sub(r"[^a-z0-9 ]", " ", text.lower())
        return re.sub(r"\s+", " ", norm).strip()

    def _build_verified_reference_index(self) -> tuple[set[str], set[str]]:
        """Collect verified (title, url) sources from the workspace.

        Returns ``(normalized_titles, urls_lower)`` covering both
        ``literature_cache.jsonl`` (every search_literature hit) and
        ``papers/**/manifest.json`` (every successful read_paper_fulltext).
        Titles are normalized via ``_normalize_for_match`` so callers can use
        substring containment without worrying about punctuation differences.
        """
        titles: set[str] = set()
        urls: set[str] = set()

        cache_path = self.working_dir / "literature_cache.jsonl"
        if cache_path.exists():
            try:
                import json as _json
                with cache_path.open("r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            rec = _json.loads(line)
                        except _json.JSONDecodeError:
                            continue
                        title_norm = self._normalize_for_match(rec.get("title") or "")
                        if title_norm:
                            titles.add(title_norm)
                        for key in ("url", "pdf_url"):
                            u = (rec.get(key) or "").strip().lower()
                            if u:
                                urls.add(u)
            except Exception as exc:  # pragma: no cover - best-effort read
                logger.debug("Reading literature_cache.jsonl for ref-index failed: %s", exc)

        papers_root = self.working_dir / "papers"
        if papers_root.exists():
            try:
                import json as _json
                for manifest_path in papers_root.glob("**/manifest.json"):
                    try:
                        rec = _json.loads(manifest_path.read_text(encoding="utf-8"))
                    except Exception:
                        continue
                    title_norm = self._normalize_for_match(rec.get("title") or "")
                    if title_norm:
                        titles.add(title_norm)
                    for key in ("resolved_url", "pdf_url", "original_input"):
                        u = (rec.get(key) or "").strip().lower()
                        if u:
                            urls.add(u)
            except Exception as exc:  # pragma: no cover - best-effort read
                logger.debug("Reading papers/**/manifest.json for ref-index failed: %s", exc)

        return titles, urls

    @classmethod
    def _ref_matches_known(
        cls,
        ref: str,
        known_titles: set[str],
        known_urls: set[str],
    ) -> bool:
        """Return True if ``ref`` traces back to any known title or URL.

        Match strategy:
          * URL match — if any verified URL appears as a substring of the
            reference text, accept (strongest signal).
          * Title match — if any verified title (3+ words after normalization)
            appears as a substring of the normalized reference text, accept.
            The 3-word threshold suppresses false positives from very short
            generic titles like "BERT" that would otherwise match unrelated
            references containing the same word.
        """
        text_lower = ref.lower().strip()
        if not text_lower:
            return False
        for url in known_urls:
            if url and len(url) > 12 and url in text_lower:
                return True
        norm_ref = cls._normalize_for_match(ref)
        if not norm_ref:
            return False
        for title in known_titles:
            if not title:
                continue
            if len(title.split()) < 3:
                # Short titles get URL-only verification to keep false-positive
                # rate down. The agent should pass URLs anyway per system prompt.
                continue
            if title in norm_ref:
                return True
        return False

    def _find_unverified_references(self, explicit_refs: list[str]) -> list[str]:
        """Return the subset of ``explicit_refs`` that don't match any cached
        search/read source.

        Empty / whitespace-only entries are silently dropped (treated as a
        no-op rather than fabrication). Anything else must trace back to a
        cache hit or it is reported as unverified so ``generate_report`` can
        reject the call and force the agent to either remove the citation or
        run search_literature for the missing paper first.
        """
        cleaned = [r.strip() for r in explicit_refs if r and r.strip()]
        if not cleaned:
            return []
        titles, urls = self._build_verified_reference_index()
        # If the workspace has no cache at all, the agent claimed citations
        # without ever search_literature/read_paper_fulltext'ing — every
        # explicit ref is unverified and we surface them all.
        if not titles and not urls:
            return cleaned
        return [
            ref for ref in cleaned
            if not self._ref_matches_known(ref, titles, urls)
        ]

    def _merge_references_with_cache(self, explicit_refs: list[str]) -> tuple[list[str], str]:
        """Merge explicit refs with full-text paper manifests and search cache hits."""
        manifest_refs, _ = self._discover_references_from_fulltext_manifests()
        cache_refs, _ = self._discover_references_from_cache()
        merged: list[str] = []
        seen: set[str] = set()
        added_fulltext = 0
        added_cache = 0

        def add_refs(refs: list[str], source: str) -> None:
            nonlocal added_fulltext, added_cache
            for ref in refs:
                text = str(ref).strip()
                if not text:
                    continue
                key = re.sub(r"\s+", " ", text).casefold()
                if key in seen:
                    continue
                seen.add(key)
                merged.append(text)
                if source == "fulltext":
                    added_fulltext += 1
                elif source == "cache":
                    added_cache += 1

        add_refs(explicit_refs or [], "explicit")
        add_refs(manifest_refs, "fulltext")
        add_refs(cache_refs, "cache")

        logs: list[str] = []
        if added_fulltext:
            logs.append(
                f"references: merged {added_fulltext} full-text/read paper(s) from papers/**/manifest.json"
            )
        if added_cache:
            logs.append(
                f"references: merged {added_cache} additional abstract-level search result(s) from literature_cache.jsonl"
            )
        return merged, "; ".join(logs)

    # Iteration suffixes the agent appends when re-running an experiment with
    # tweaked hyperparameters. ``accuracy_comparison.png`` and
    # ``accuracy_comparison_refined.png`` are the same figure family — only
    # the latest belongs in the paper. We strip the suffix when computing
    # the family key, then keep the most-recently-modified path.
    _ITERATION_SUFFIX_RE = re.compile(
        r"_(?:refined|final|new|fixed|updated|improved|rev\d+|v\d+)$",
        re.IGNORECASE,
    )

    def _dedup_by_stem_family(
        self, entries: list[tuple["Path", dict]]
    ) -> tuple[list[dict], int]:
        """Keep only the freshest artifact per stem family.

        Trajectory 5bfb6c8d shipped both ``accuracy_comparison.png`` (from a
        max_iter=3 smoke test) and ``accuracy_comparison_refined.png`` (the
        max_iter=10 retry) into the same paper, plus their two contradicting
        CSVs. The reader sees two tables with different LR accuracy (0.906 vs
        0.898) for the same experiment.
        """
        family_best: dict[str, tuple["Path", dict]] = {}
        for path, entry in entries:
            stem = path.stem
            family_key = self._ITERATION_SUFFIX_RE.sub("", stem).lower()
            existing = family_best.get(family_key)
            try:
                mtime = path.stat().st_mtime
            except OSError:
                mtime = 0
            if existing is None:
                family_best[family_key] = (path, entry)
                continue
            try:
                existing_mtime = existing[0].stat().st_mtime
            except OSError:
                existing_mtime = 0
            if mtime > existing_mtime:
                family_best[family_key] = (path, entry)
            elif mtime == existing_mtime:
                # Test/CI envs can produce identical mtimes when files are
                # written within the same FS tick. Tie-break by stem length:
                # the longer stem is the suffixed variant (``foo_refined``
                # over ``foo``), which is the intended winner.
                if len(path.stem) > len(existing[0].stem):
                    family_best[family_key] = (path, entry)

        kept = [entry for _, entry in family_best.values()]
        dropped = len(entries) - len(kept)
        return kept, dropped

    def _discover_figures_from_workspace(self) -> tuple[list[dict], str]:
        """Find image artefacts produced during the run.

        Image extensions: PNG / JPG / SVG / PDF (only when next to a
        ``.tex`` it's NOT — but we don't bother distinguishing here;
        the LaTeX renderer skips anything it can't include).
        """
        if not self.working_dir.exists():
            return [], ""
        image_exts = {".png", ".jpg", ".jpeg", ".svg"}
        candidates: list[tuple["Path", dict]] = []
        for f in sorted(self.working_dir.rglob("*")):
            if not f.is_file():
                continue
            try:
                rel = f.relative_to(self.working_dir)
            except ValueError:
                continue
            if self._is_skipped_path(rel):
                continue
            if f.suffix.lower() not in image_exts:
                continue
            stem = f.stem.replace("_", " ").replace("-", " ").strip()
            caption = stem.capitalize() if stem else f.name
            candidates.append((f, {
                "path": str(rel),
                "caption": caption,
                "section": "results",
            }))
        if not candidates:
            return [], ""
        figures, dropped = self._dedup_by_stem_family(candidates)
        msg = f"figures: agent left empty, auto-attached {len(figures)} image(s) from workspace"
        if dropped:
            msg += (
                f" (deduped {dropped} older iteration(s) — only the freshest "
                "of each ``stem_refined.png`` family is kept)"
            )
        return figures, msg

    def _discover_tables_from_workspace(self) -> tuple[list[dict], str]:
        """Find tabular artefacts (CSV / TSV) produced during the run."""
        if not self.working_dir.exists():
            return [], ""
        table_exts = {".csv", ".tsv"}
        candidates: list[tuple["Path", dict]] = []
        for f in sorted(self.working_dir.rglob("*")):
            if not f.is_file():
                continue
            try:
                rel = f.relative_to(self.working_dir)
            except ValueError:
                continue
            if self._is_skipped_path(rel):
                continue
            if f.suffix.lower() not in table_exts:
                continue
            stem = f.stem.replace("_", " ").replace("-", " ").strip()
            caption = stem.capitalize() if stem else f.name
            candidates.append((f, {
                "path": str(rel),
                "caption": caption,
                "section": "results",
            }))
        if not candidates:
            return [], ""
        tables, dropped = self._dedup_by_stem_family(candidates)
        msg = f"tables: agent left empty, auto-attached {len(tables)} CSV/TSV from workspace"
        if dropped:
            msg += (
                f" (deduped {dropped} older iteration(s) — only the freshest "
                "of each ``stem_refined.csv`` family is kept)"
            )
        return tables, msg

    def _validate_artifact_paths(
        self,
        figures: list[dict],
        tables: list[dict],
    ) -> str:
        """Reject generate_report when any cited figure / table path does not exist.

        Catches the "agent hallucinated a file path" failure mode (trajectory
        c032bbf1: figures=[{'path': 'training_loss.png'}] but no plt.savefig
        ever ran). Returns ``""`` when every path is real, or a structured
        rejection message when one or more are missing — same shape as
        ``_validate_experiment_evidence`` so the caller treats them
        uniformly.

        Each path is resolved relative to ``self.working_dir`` (the run
        sandbox), then ``Path.exists()`` is called. We also flag empty /
        whitespace-only path strings, since those are equally useless to
        downstream paper rendering.
        """
        invalid: list[str] = []

        def _check(kind: str, entry: object) -> None:
            if not isinstance(entry, dict):
                invalid.append(f"{kind}: <non-dict entry, type={type(entry).__name__}>")
                return
            raw = (entry.get("path") or "").strip()
            if not raw:
                invalid.append(f"{kind}: <empty path>")
                return
            full = (self.working_dir / raw).resolve()
            if not full.exists():
                invalid.append(f"{kind}: {raw}")

        for fig in figures or []:
            _check("figures", fig)
        for tbl in tables or []:
            _check("tables", tbl)

        if not invalid:
            return ""

        return (
            "generate_report was rejected: you referenced files that do not "
            "exist in the run workspace.\n\n"
            "Missing artifacts:\n"
            + "\n".join(f"  - {p}" for p in invalid)
            + "\n\nDo NOT call generate_report with placeholder file names. "
            "Fix one of two ways:\n"
            "  (a) Run execute_code FIRST to actually save these files, "
            "for example:\n"
            "      import matplotlib.pyplot as plt\n"
            "      plt.plot(history); plt.savefig('training_loss.png')\n"
            "      pd.DataFrame(metrics).to_csv('optimizer_comparison.csv', index=False)\n"
            "      Then call generate_report again with the real paths.\n"
            "  (b) Remove these entries from figures= / tables= and rewrite "
            "the prose in results=/analysis= so it does not claim a chart "
            "or table that does not exist. An honest report with fewer "
            "figures is better than a fabricated one.\n\n"
            "Either path is acceptable — pick the one that fits how much "
            "step budget you have left."
        )

    def _validate_experiment_evidence(
        self,
        figures: list[dict],
        tables: list[dict],
    ) -> str:
        """Reject paper writing before real experiments and visual artifacts exist.

        P5 fix: in SURVEY ONLY mode this gate is skipped — the survey
        prompt explicitly instructs the agent NOT to run experiments,
        so demanding a result table / figure here would force the agent
        to fabricate one (trajectory 477a3605 created a placeholder.png
        with literal text "NO EXPERIMENTAL RESULTS (SURVEY ONLY TASK)"
        purely to satisfy this validator). Survey runs only need the
        reference-list gate that fires earlier in ``execute()``.
        """
        if self._is_survey_mode():
            return ""

        executed = self._has_successful_experiment_execution()
        figure_count = len(figures or [])
        table_count = len(tables or [])

        missing: list[str] = []
        if not executed:
            missing.append("no successful execute_code/execute_bash experiment log under logs/")
        if table_count == 0:
            missing.append("no CSV/TSV result table")
        if figure_count == 0:
            missing.append("no PNG/JPG/SVG result figure")

        if not missing:
            return ""

        return (
            "generate_report was rejected: the research run has not produced "
            "enough experiment evidence for a paper.\n\n"
            "Missing evidence:\n"
            + "\n".join(f"- {item}" for item in missing)
            + "\n\nBefore writing the paper, continue with execute_code to run "
            "the actual experiment, save at least one result table as CSV/TSV, "
            "and save at least one figure as PNG/JPG/SVG. Then call "
            "generate_report again with those artifacts."
        )

    def _has_successful_experiment_execution(self) -> bool:
        logs_dir = self.working_dir / "logs"
        if not logs_dir.exists():
            return False
        for log_path in sorted(logs_dir.glob("*.log")):
            try:
                text = log_path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            tool_match = re.search(r"^tool\s*:\s*(execute_code|execute_bash)\s*$", text, re.MULTILINE)
            exit_match = re.search(r"^exit_code\s*:\s*0\s*$", text, re.MULTILINE)
            if tool_match and exit_match:
                return True
        return False
