"""
LabForge interactive web UI.

This Gradio app provides:
- a cleaner research workspace
- persistent model / API / OCR settings
- direct PaperForge export using structured references, figures, and tables
"""

from __future__ import annotations

import argparse
import html as html_lib
import json
import logging
import os
import shlex
import sys
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import gradio as gr
from openai import OpenAI

from lab_forge.config import (
    AgentConfig,
    ModelConfig,
    ReviewerConfig,
    SandboxConfig,
    resolve_api_key_for_endpoint,
)
from lab_forge.env_utils import load_project_env
from lab_forge.models import MODEL_PRESETS, REVIEWER_PRESETS
from lab_forge.paper_bundle import load_bundle_images, load_paper_bundle
from lab_forge.trajectory import Trajectory
from lab_forge.ui_settings import (
    DEFAULT_SETTINGS_PATH,
    UISettings,
    load_ui_settings,
    mask_secret,
    save_ui_settings,
)
from lab_forge.workflow import (
    DEFAULT_SANDBOX_ROOT,
    prepare_run_sandbox,
    summarise_workspace,
    topic_to_task,
)

logger = logging.getLogger(__name__)

# ``web.py`` lives at ``<repo_root>/lab_forge/web.py``. The project's
# trajectories/ folder + paper_forge/ sibling and friends live one level
# UP, at ``<repo_root>/``. (Pre-refactor this module was at the repo root
# as ``ui.py`` so a single ``.parent`` resolved correctly; that's no longer
# true and using ``.parent`` here would silently route artefacts into the
# package directory.)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PAPER_FORGE_ROOT = PROJECT_ROOT / "paper_forge"
DEFAULT_TRAJ_DIR = PROJECT_ROOT / "trajectories"
SANDBOX_ROOT = DEFAULT_SANDBOX_ROOT
CUSTOM_PRESET_LABEL = "自定义"


def _env_bool(name: str, default: bool) -> bool:
    """Parse ``"true" / "1" / "yes"`` etc. from an env var."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on", "y")


# Feature flag for shared deployments (e.g., 星河社区 / AI Studio): when this is
# false, the "运行记录" history tab is hidden so other users can't browse the
# trajectories produced by previous sessions on the same instance. Toggle via
# ``SCIPRM_SHOW_HISTORY=false`` in .env (or env var) before launching the UI.
SHOW_HISTORY_TAB = _env_bool("SCIPRM_SHOW_HISTORY", True)

PHASE_LABELS = {
    "planning": "思考与规划",
    "literature_review": "文献检索",
    "source_reading": "资料读取",
    "implementation": "代码执行",
    "analysis": "结果分析",
    "submission": "结果提交",
    "report": "论文整理",
    "reviewer_review": "评审模型介入",
    "human": "用户介入",
    "unknown": "思考中",
}

PHASE_ICONS = {
    "planning": "Plan",
    "literature_review": "Search",
    "source_reading": "Read",
    "implementation": "Run",
    "analysis": "Analyze",
    "submission": "Submit",
    "report": "Report",
    "reviewer_review": "Review",
    "human": "Human",
    "unknown": "Step",
}



_CODE_FENCE_LANG = {
    "execute_code": "python",
    "execute_bash": "bash",
    "file_write": "",
    "file_read": "",
}

ACTION_PHASE_MAP = {
    "plan": "planning",
    "reviewer_review": "reviewer_review",
    "search_literature": "literature_review",
    "read_paper_fulltext": "source_reading",
    "execute_code": "implementation",
    "execute_bash": "implementation",
    "file_write": "implementation",
    "file_read": "analysis",
    "list_files": "analysis",
    "generate_report": "report",
    "submit_result": "submission",
    "finish": "submission",
    "human_feedback": "human",
    "human_abort": "human",
    "human_review": "human",
}

NON_TOOL_ACTIONS = {"plan", "human_feedback", "human_abort", "human_review"}


def _escape_html(value: Any) -> str:
    return html_lib.escape(str(value or ""), quote=True)


def _compact_text(value: Any, limit: int = 110) -> str:
    text = " ".join(str(value or "").strip().split())
    if len(text) > limit:
        return text[: limit - 1] + "..."
    return text


def _phase_for_action(action_name: str) -> str:
    return ACTION_PHASE_MAP.get(action_name, "unknown")


def _step_status_key(step_info: dict[str, Any]) -> str:
    action_name = str(step_info.get("action_name", "unknown"))
    success = bool(step_info.get("success", True))
    metadata = step_info.get("metadata") or {}
    if action_name == "reviewer_review" and metadata.get("phase") == "thinking":
        return "running"
    if action_name in {"human_review"}:
        return "review"
    if action_name == "reviewer_review" and not success:
        return "review"
    if not success:
        return "failed"
    return "completed"


def _status_badge_html(status_key: str) -> str:
    label_map = {
        "pending": "Pending",
        "running": "Running",
        "completed": "Completed",
        "failed": "Failed",
        "review": "Need Review",
    }
    css_map = {
        "pending": "status-pending",
        "running": "status-running",
        "completed": "status-completed",
        "failed": "status-failed",
        "review": "status-review",
    }
    key = status_key if status_key in label_map else "pending"
    return f'<span class="status-badge {css_map[key]}">{label_map[key]}</span>'


def _runtime_status_key(status: str) -> str:
    return {
        "idle": "pending",
        "running": "running",
        "done": "completed",
        "error": "failed",
    }.get(status, "pending")


def format_step_budget_label(max_steps: int | str | None) -> str:
    try:
        value = int(max_steps or 0)
    except (TypeError, ValueError):
        value = 0
    return "Adaptive" if value <= 0 else str(value)


def format_step_progress(step_count: int, max_steps: int | str | None) -> str:
    try:
        value = int(max_steps or 0)
    except (TypeError, ValueError):
        value = 0
    if value <= 0:
        return f"{step_count} · Adaptive"
    return f"{step_count}/{value}"


def _build_research_pipeline_html(
    step_infos: list[dict[str, Any]] | None = None,
    status: str = "idle",
    max_steps: int = 30,
    sandbox_name: str = "",
) -> str:
    steps = list(step_infos or [])
    max_steps_int = int(max_steps or 0)
    pct = min(100, int(len(steps) / max(max_steps_int, 1) * 100)) if max_steps_int > 0 else 0
    badge = _status_badge_html(_runtime_status_key(status))
    sandbox = _escape_html(sandbox_name or "-")
    progress_label = _escape_html(format_step_progress(len(steps), max_steps_int))

    if not steps:
        body = (
            '<div class="empty-state">'
            "等待研究任务启动。启动后这里会按 agent 已有步骤顺序显示 step name、status 和摘要。"
            "</div>"
        )
    else:
        items: list[str] = []
        for idx, step_info in enumerate(steps, start=1):
            action_name = str(step_info.get("action_name", "unknown"))
            phase = _phase_for_action(action_name)
            label = PHASE_LABELS.get(phase, phase)
            status_key = _step_status_key(step_info)
            summary = _compact_text(
                step_info.get("thought") or step_info.get("observation") or action_name,
                limit=92,
            )
            items.append(
                f'<div class="pipeline-item {status_key}">'
                '<span class="pipeline-dot"></span>'
                '<div>'
                '<div class="pipeline-step-head">'
                f'<span class="pipeline-step-name">{idx}. {_escape_html(label)}</span>'
                f'{_status_badge_html(status_key)}'
                '</div>'
                f'<div class="pipeline-step-meta">{_escape_html(action_name)} · {_escape_html(summary)}</div>'
                '</div>'
                '</div>'
            )
        body = '<div class="pipeline-list">' + "".join(items) + "</div>"

    return (
        '<section class="pipeline-shell">'
        '<div class="panel-title-row">'
        '<h3>Research Pipeline</h3>'
        f'{badge}'
        '</div>'
        f'<p class="panel-subtitle">Steps {progress_label} · Sandbox {sandbox}</p>'
        '<div class="pipeline-progress" aria-hidden="true">'
        f'<span style="width: {pct}%"></span>'
        '</div>'
        f'{body}'
        '</section>'
    )


def _build_evidence_trace_html(
    step_infos: list[dict[str, Any]] | None = None,
    status: str = "idle",
) -> str:
    steps = list(step_infos or [])
    evidence_items: list[str] = []
    tool_items: list[str] = []
    review_items: list[str] = []

    for idx, step_info in enumerate(steps, start=1):
        action_name = str(step_info.get("action_name", "unknown"))
        args = step_info.get("action_args") or {}
        metadata = step_info.get("metadata") or {}
        status_key = _step_status_key(step_info)
        success_class = "failed" if status_key == "failed" else "success"
        duration = metadata.get("exec_time_s")
        duration_label = f" · {duration}s" if duration else ""

        if action_name in {"search_literature", "read_paper_fulltext"}:
            source = (
                args.get("query")
                or args.get("q")
                or args.get("local_path")
                or args.get("url")
                or args.get("pdf_url")
                or args.get("title")
                or _compact_text(step_info.get("observation"), 80)
            )
            title = (
                "Literature search"
                if action_name == "search_literature"
                else "Uploaded/local paper read"
            )
            evidence_items.append(
                '<div class="evidence-item">'
                f'<div class="evidence-title">[{idx}] {_escape_html(title)}</div>'
                f'<div class="evidence-meta">{_escape_html(source)}{_escape_html(duration_label)}</div>'
                '</div>'
            )

        if action_name not in NON_TOOL_ACTIONS:
            tool_items.append(
                f'<div class="tool-row {success_class}">'
                '<span class="tool-dot"></span>'
                '<div>'
                f'<div class="tool-name">{_escape_html(action_name)}</div>'
                f'<div class="tool-meta">Step {idx} · {_escape_html(status_key)}{_escape_html(duration_label)}</div>'
                '</div>'
                '</div>'
            )

        if action_name == "reviewer_review":
            review_text = _compact_text(step_info.get("observation") or step_info.get("thought"), 160)
            review_status = "Need Review" if status_key == "review" else "Completed"
            review_items.append(
                '<div class="uncertainty-box">'
                f'<strong>{_escape_html(review_status)}</strong><br>'
                f'{_escape_html(review_text or "评审模型已完成该检查点复核。")}'
                '</div>'
            )

    if status == "running" and not tool_items:
        tool_items.append(
            '<div class="empty-state">Agent 正在准备第一步工具调用。</div>'
        )

    evidence_html = (
        "".join(evidence_items)
        if evidence_items
        else '<div class="empty-state">尚无文献检索或全文读取证据。</div>'
    )
    tool_html = (
        "".join(tool_items)
        if tool_items
        else '<div class="empty-state">尚无工具调用记录。</div>'
    )
    review_html = (
        "".join(review_items)
        if review_items
        else '<div class="uncertainty-box">尚无评审模型风险提示；运行到检查点后会在此展示不确定性。</div>'
    )

    return (
        '<section class="evidence-shell">'
        '<div class="panel-title-row"><h3>Evidence Trace</h3>'
        f'{_status_badge_html(_runtime_status_key(status))}</div>'
        '<p class="panel-subtitle">引用、工具调用和评审不确定性只从现有执行轨迹派生。</p>'
        '<div class="evidence-section">'
        '<h4>Evidence Items</h4>'
        f'<div class="evidence-list">{evidence_html}</div>'
        '</div>'
        '<div class="evidence-section">'
        '<h4>Tool Call Log</h4>'
        f'<div class="evidence-list">{tool_html}</div>'
        '</div>'
        '<div class="evidence-section">'
        '<h4>Uncertainty</h4>'
        f'{review_html}'
        '</div>'
        '</section>'
    )


def _runner_ui_snapshots() -> tuple[str, str]:
    with _runner._lock:
        step_infos = [dict(item) for item in getattr(_runner, "step_infos", [])]
        status = _runner.status
        max_steps = _runner.max_steps
        sandbox_name = _runner.sandbox_dir.name if _runner.sandbox_dir else ""
    return (
        _build_research_pipeline_html(step_infos, status, max_steps, sandbox_name),
        _build_evidence_trace_html(step_infos, status),
    )


def _safe_code_fence(body: str, lang: str = "") -> str:
    """Wrap ``body`` in a markdown code fence that cannot be broken by the
    body's own backticks. Picks a run of backticks one longer than the
    longest run found inside ``body`` (min 3, max 10).
    """
    if body is None:
        body = ""
    longest = 0
    run = 0
    for ch in body:
        if ch == "`":
            run += 1
            if run > longest:
                longest = run
        else:
            run = 0
    fence_len = max(3, min(longest + 1, 10))
    fence = "`" * fence_len
    return f"{fence}{lang}\n{body}\n{fence}"


def _format_action_input(action_name: str, args: dict[str, Any]) -> str:
    """Render tool arguments as an "**Input**" code block for the UI.

    We only surface arguments for tools where the input is meaningful to show
    (the code the agent wrote, the bash command it ran, the file it wrote).
    For literature/file-browsing tools we show the query / path inline.
    """
    if not args:
        return ""

    code = args.get("code")
    if action_name == "execute_code" and code:
        snippet = str(code)
        if len(snippet) > 3000:
            snippet = snippet[:3000] + "\n# … (truncated)"
        return "\n**Input**\n" + _safe_code_fence(snippet, "python") + "\n"

    command = args.get("command")
    if action_name == "execute_bash" and command:
        snippet = str(command)
        if len(snippet) > 2000:
            snippet = snippet[:2000] + "\n# … (truncated)"
        return "\n**Input**\n" + _safe_code_fence(snippet, "bash") + "\n"

    if action_name == "file_write":
        path = args.get("path") or args.get("filename") or ""
        body = args.get("content") or args.get("text") or ""
        snippet = str(body)
        if len(snippet) > 2500:
            snippet = snippet[:2500] + "\n… (truncated)"
        header = f"`{path}`\n" if path else ""
        return "\n**Input**\n" + header + _safe_code_fence(snippet) + "\n"

    if action_name == "file_read":
        path = args.get("path") or args.get("filename") or ""
        return f"\n**Input** `{path}`\n" if path else ""

    if action_name == "search_literature":
        query = args.get("query") or args.get("q") or ""
        return f"\n**Query** `{query}`\n" if query else ""

    if action_name == "read_paper_fulltext":
        source = args.get("local_path") or args.get("url") or args.get("pdf_url") or ""
        title = args.get("title") or ""
        details = []
        if source:
            details.append(f"**Source** `{source}`")
        if title:
            details.append(f"**Title** `{title}`")
        return "\n" + "\n".join(details) + "\n" if details else ""

    # Generic fallback: short repr of the args so the user still sees
    # *something* rather than a blank card.
    try:
        rendered = json.dumps(args, ensure_ascii=False, indent=2, default=str)
    except Exception:
        rendered = str(args)
    if len(rendered) > 1500:
        rendered = rendered[:1500] + "\n… (truncated)"
    return "\n**Input**\n" + _safe_code_fence(rendered, "json") + "\n"


def _format_step_card_from_dict(step_info: dict[str, Any], idx: int) -> str:
    action_name = step_info.get("action_name", "unknown")
    phase = _phase_for_action(action_name)

    # Render human-in-the-loop cards distinctly so users can see exactly when
    # their interventions reached the agent.
    if action_name in ("human_feedback", "human_abort", "human_review"):
        label = PHASE_LABELS.get("human", "用户介入")
        body = (step_info.get("observation") or step_info.get("thought") or "").strip()
        badge_map = {
            "human_feedback": "用户实时反馈",
            "human_abort": "用户中止",
            "human_review": "等待用户确认",
        }
        badge = badge_map.get(action_name, "用户介入")
        return (
            f"### Step {idx}: {label} · {badge}\n\n"
            f"{body}\n\n---\n"
        )
    label = PHASE_LABELS.get(phase, phase)
    status = "成功" if step_info.get("success", True) else "失败"

    thought = (step_info.get("thought") or "").strip()
    if len(thought) > 1200:
        thought = thought[:1200] + "…"

    observation = (step_info.get("observation") or "").strip()
    if len(observation) > 1600:
        observation = observation[:1600] + "\n… (truncated)"

    # "plan" steps have no tool output — render a compact thinking card.
    if action_name == "plan":
        reasoning_body = thought or "（模型本轮未输出显式推理）"
        return (
            f"### Step {idx}: {label}\n\n"
            f"{reasoning_body}\n\n---\n"
        )

    # Reviewer review cards — highlight pass/fail and show the review body
    # directly instead of wrapping it in a tool-call frame. The callback also
    # emits a lightweight "thinking" placeholder (phase=thinking in metadata)
    # right before calling the reviewer LLM so the UI doesn't go silent
    # during the review — render that with a distinct badge.
    if action_name == "reviewer_review":
        metadata = step_info.get("metadata") or {}
        if metadata.get("phase") == "thinking":
            return (
                f"### Step {idx}: {label} · 评审模型思考中\n\n"
                f"{thought or '评审模型正在检查本阶段产出...'}\n\n---\n"
            )
        badge = "通过" if step_info.get("success", True) else "需修正"
        intro = thought or "评审模型在当前检查点执行了一次复核。"
        return (
            f"### Step {idx}: {label} · {badge}\n\n"
            f"{intro}\n\n"
            f"{_safe_code_fence(observation)}\n\n---\n"
        )

    # For code / bash / file_write steps, show the actual input the agent
    # sent to the tool so the user can see the exact experiment being run
    # — not just the output it produced. Otherwise it's impossible to tell
    # a real execution apart from fabricated data.
    input_block = _format_action_input(action_name, step_info.get("action_args") or {})

    return (
        f"### Step {idx}: {label} [{status}]\n\n"
        f"**Reasoning**\n> {thought or '本步骤未输出显式推理，见工具输出。'}\n\n"
        f"**Tool**\n`{action_name}`\n"
        f"{input_block}"
        f"**Output**\n{_safe_code_fence(observation)}\n\n---\n"
    )


def _format_step_card(step, idx: int) -> str:
    if step.action_name == "plan":
        phase = "planning"
    elif step.action_name == "reviewer_review":
        phase = "reviewer_review"
    elif step.action_name == "generate_report":
        phase = "report"
    elif step.action_name in ("human_feedback", "human_abort", "human_review"):
        phase = "human"
    else:
        phase = step.phase
    label = PHASE_LABELS.get(phase, phase)
    status = "成功" if step.success else "失败"
    time_s = step.metadata.get("exec_time_s", "")
    time_str = f" · {time_s}s" if time_s else ""

    thought = step.thought.strip()
    if len(thought) > 1200:
        thought = thought[:1200] + "…"

    observation = step.observation.strip()
    if len(observation) > 1200:
        observation = observation[:1200] + "\n… (truncated)"

    if step.action_name == "plan":
        reasoning_body = thought or "（模型本轮未输出显式推理）"
        return (
            f"### Step {idx + 1}: {label}\n\n"
            f"{reasoning_body}\n\n---\n"
        )

    if step.action_name == "reviewer_review":
        badge = "通过" if step.success else "需修正"
        return (
            f"### Step {idx + 1}: {label} · {badge}\n\n"
            f"{thought or '评审模型在当前检查点完成了一次复核。'}\n\n"
            f"{_safe_code_fence(observation)}\n\n---\n"
        )

    input_block = _format_action_input(step.action_name, step.action_args or {})
    return (
        f"### Step {idx + 1}: {label} [{status}{time_str}]\n\n"
        f"**Reasoning**\n> {thought or 'No reasoning recorded.'}\n\n"
        f"**Tool**\n`{step.action_name}`\n"
        f"{input_block}"
        f"**Output**\n{_safe_code_fence(observation)}\n\n---\n"
    )


def _build_progress_header(steps_count: int, max_steps: int, status: str) -> str:
    if status == "running":
        if max_steps:
            pct = int(steps_count / max_steps * 100)
            bar = "█" * (pct // 5) + "░" * (20 - pct // 5)
            return f"**当前进度** `[{bar}]` {steps_count}/{max_steps}\n\n"
        return f"**当前进度** `{steps_count} 步 · Adaptive budget`\n\n"
    if status == "done":
        return "**状态** 已完成\n\n"
    if status == "error":
        return "**状态** 运行出错\n\n"
    return ""


class AgentRunner:
    """Manage the background agent thread and stream-friendly UI state."""

    def __init__(self):
        self.trajectory: Trajectory | None = None
        self.steps_md: list[str] = []
        self.step_infos: list[dict[str, Any]] = []
        self.status: str = "idle"
        self.error_msg: str = ""
        self.max_steps: int = 30
        self.sandbox_dir: Path | None = None
        self._lock = threading.Lock()

    def reset_for_run(self, max_steps: int, sandbox_dir: Path) -> None:
        with self._lock:
            self.trajectory = None
            self.steps_md = []
            self.step_infos = []
            self.status = "running"
            self.error_msg = ""
            self.max_steps = max_steps
            self.sandbox_dir = sandbox_dir

    def _step_callback(self, step_info: dict[str, Any]) -> None:
        with self._lock:
            idx = step_info.get("step_index", len(self.steps_md) + 1)
            stored_info = dict(step_info)
            self.step_infos.append(stored_info)
            self.steps_md.append(_format_step_card_from_dict(stored_info, idx))

    def get_display(self, tick: int = 0) -> str:
        with self._lock:
            header = _build_progress_header(len(self.steps_md), self.max_steps, self.status)
            if not self.steps_md:
                if self.status == "running":
                    dots = "." * ((tick % 3) + 1)
                    return header + f"*Agent 正在拆解任务并规划执行路径{dots}*"
                return ""
            body = "\n".join(self.steps_md)
            if self.status == "running":
                dots = "." * ((tick % 3) + 1)
                body += f"\n\n*继续执行下一步{dots}*"
            return header + body

    def run(
        self,
        config: AgentConfig,
        task_id: str,
        task_desc: str,
        expected_output: str,
        data_desc: str,
    ) -> None:
        try:
            from lab_forge.agent import ResearchAgent

            agent = ResearchAgent(config)
            trajectory = agent.run(
                task_id=task_id,
                task_description=task_desc,
                expected_output=expected_output,
                data_description=data_desc,
                step_callback=self._step_callback,
            )
            with self._lock:
                self.trajectory = trajectory
                self.status = "done"
        except Exception as exc:
            logger.exception("Agent run failed")
            with self._lock:
                self.status = "error"
                self.error_msg = str(exc)

    def get_output_files(self) -> list[str]:
        with self._lock:
            sandbox_dir = self.sandbox_dir
        if sandbox_dir is None or not sandbox_dir.exists():
            return []

        files: list[str] = []
        allowed_exts = {
            ".png",
            ".jpg",
            ".jpeg",
            ".svg",
            ".csv",
            ".tsv",
            ".json",
            ".txt",
            ".pdf",
            ".md",
            # Exec audit artefacts — the actual script the agent ran and the
            # matching stdout/stderr log. Surfacing them in the UI is the
            # whole point of the audit trail: the CSV is only trustworthy if
            # you can match it to a real script + log.
            ".log",
            ".py",
            ".sh",
        }
        for candidate in sandbox_dir.rglob("*"):
            if not candidate.is_file():
                continue
            if candidate.suffix.lower() not in allowed_exts:
                continue
            files.append(str(candidate))
        return sorted(files)


_runner = AgentRunner()


def _prepare_run_sandbox(task_id: str) -> Path:
    return prepare_run_sandbox(task_id, SANDBOX_ROOT)


def _build_workspace_summary_md(run_dir: Path) -> str:
    """Markdown summary of a run's workspace, broken down by file role.

    The UI used to only show a "download files" list, which didn't make clear
    *where* the code that produced those outputs lives. This panel surfaces the
    sandbox path, the persisted execute_code scripts, and the matching logs so
    the user can open the actual source of every artefact.
    """
    buckets = summarise_workspace(run_dir)
    sections: list[str] = [f"**工作区路径** `{run_dir}`"]
    labels = [
        ("code", "代码 / 脚本 (execute_code / file_write 真实源文件)"),
        ("log", "执行日志 (script_path + stdout/stderr)"),
        ("data", "数据 / 结果文件 (由实验程序生成)"),
        ("image", "图表"),
        ("report", "报告 / 论文"),
        ("other", "其他"),
    ]
    for key, label in labels:
        files = buckets.get(key, [])
        if not files:
            continue
        sections.append(f"\n**{label}**")
        for path in files[:8]:
            sections.append(f"- `{path}`")
        remaining = len(files) - 8
        if remaining > 0:
            sections.append(f"- ...还有 {remaining} 个文件")
    if len(sections) == 1:
        return ""
    return "\n".join(sections)


def _resolve_model_preset(preset_name: str) -> dict[str, str]:
    return MODEL_PRESETS.get(
        preset_name,
        {"model": "MiniMax-M2.7", "base_url": "https://api.minimaxi.com/v1"},
    )


def _resolve_reviewer_preset(preset_name: str) -> dict[str, str]:
    return REVIEWER_PRESETS.get(
        preset_name,
        {"model": "ernie-4.5-turbo-128k-preview", "base_url": "https://aistudio.baidu.com/llm/lmapi/v3"},
    )


def _match_preset(model: str, base_url: str, presets: dict[str, dict[str, str]]) -> str:
    for name, config in presets.items():
        if config["model"] == model and config["base_url"] == base_url:
            return name
    return CUSTOM_PRESET_LABEL


def _get_env_api_key() -> str:
    return resolve_api_key_for_endpoint()


def _resolve_api_key(api_key: str, base_url: str, model: str) -> str:
    return api_key.strip() or resolve_api_key_for_endpoint(
        base_url=base_url,
        model=model,
    )


def _get_env_ocr_token() -> str:
    for key in ("PADDLEOCR_TOKEN", "OCR_TOKEN", "PADDLEOCR_VL_TOKEN"):
        value = os.environ.get(key, "").strip()
        if value:
            return value
    return ""


def _get_env_ocr_url() -> str:
    for key in ("PADDLEOCR_API_URL", "OCR_API_URL", "PADDLEOCR_VL_API_URL"):
        value = os.environ.get(key, "").strip()
        if value:
            return value
    return ""


def _build_ui_settings(
    agent_model: str,
    agent_base_url: str,
    agent_api_key: str,
    reviewer_enabled: bool,
    reviewer_model: str,
    reviewer_base_url: str,
    reviewer_api_key: str,
    max_steps: int,
    temperature: float,
    ocr_enabled: bool,
    ocr_api_url: str,
    ocr_token: str,
    search_quota: int = 8,
    run_mode: str = "survey",
) -> UISettings:
    normalized_mode = run_mode if run_mode in ("survey", "experiment") else "survey"
    return UISettings(
        agent_model=agent_model.strip() or UISettings().agent_model,
        agent_base_url=agent_base_url.strip() or UISettings().agent_base_url,
        agent_api_key=agent_api_key.strip(),
        reviewer_enabled=bool(reviewer_enabled),
        reviewer_share_credentials=False,
        reviewer_model=reviewer_model.strip() or UISettings().reviewer_model,
        reviewer_base_url=reviewer_base_url.strip() or UISettings().reviewer_base_url,
        reviewer_api_key=reviewer_api_key.strip(),
        max_steps=int(max_steps),
        temperature=float(temperature),
        ocr_enabled=bool(ocr_enabled),
        ocr_api_url=ocr_api_url.strip() or UISettings().ocr_api_url,
        ocr_token=ocr_token.strip(),
        search_quota=int(search_quota or 0),
        run_mode=normalized_mode,
    )


def _settings_source_label(source: str) -> str:
    return {
        "file": "本地保存",
        "env": "环境变量",
        "default": "内置默认",
        "runtime": "当前表单",
    }.get(source, source)


def _copy_button_html(value: Any, label: str = "复制") -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    payload = _escape_html(json.dumps(text, ensure_ascii=False))
    safe_label = _escape_html(label)
    return (
        f'<button type="button" class="copy-btn" data-copy="{payload}" aria-label="{safe_label}" '
        'onclick="if(navigator.clipboard){navigator.clipboard.writeText(JSON.parse(this.dataset.copy));}'
        "this.classList.add('copied');setTimeout(()=>this.classList.remove('copied'),900)\">复制</button>"
    )


def _summary_status_badge(enabled: bool) -> str:
    label = "已启用" if enabled else "未启用"
    css_class = "enabled" if enabled else "disabled"
    return f'<span class="soft-badge {css_class}">{label}</span>'


def _summary_field_html(
    label: str,
    value: Any,
    *,
    copyable: bool = False,
    code: bool = False,
    fallback: str = "未配置",
) -> str:
    raw = str(value or "").strip()
    display = raw or fallback
    value_class = "truncate-value"
    if code:
        value_class += " code-value"
    copy_html = _copy_button_html(raw, f"复制{label}") if copyable and raw else ""
    return (
        '<div class="summary-field">'
        f"<dt>{_escape_html(label)}</dt>"
        "<dd>"
        f'<span class="{value_class}" title="{_escape_html(display)}">{_escape_html(display)}</span>'
        f"{copy_html}"
        "</dd>"
        "</div>"
    )


def _summary_secret_html(label: str, secret: str) -> str:
    display = f"已配置 {mask_secret(secret)}" if secret else "未配置"
    return (
        '<div class="summary-field">'
        f"<dt>{_escape_html(label)}</dt>"
        "<dd>"
        f'<span class="truncate-value secret-value" title="{_escape_html(display)}">{_escape_html(display)}</span>'
        "</dd>"
        "</div>"
    )


def _build_settings_summary_html(settings: UISettings, source: str) -> str:
    agent_key = _resolve_api_key(
        settings.agent_api_key,
        settings.agent_base_url,
        settings.agent_model,
    )
    reviewer_base = settings.reviewer_base_url or UISettings().reviewer_base_url
    reviewer_key = _resolve_api_key(
        settings.reviewer_api_key or "",
        reviewer_base,
        settings.reviewer_model,
    )
    ocr_url = settings.ocr_api_url or _get_env_ocr_url()
    ocr_token = settings.ocr_token or _get_env_ocr_token()
    reviewer_base_display = reviewer_base or settings.agent_base_url

    return (
        '<div class="settings-summary-card">'
        '<div class="summary-card-header">'
        "<div>"
        '<div class="summary-eyebrow">当前运行配置</div>'
        "<h3>模型 / OCR 摘要</h3>"
        "</div>"
        f'<span class="source-badge">{_escape_html(_settings_source_label(source))}</span>'
        "</div>"
        '<section class="summary-section">'
        '<div class="summary-section-head">'
        '<span class="summary-section-title">主模型</span>'
        '<span class="soft-badge enabled">已配置</span>'
        "</div>"
        '<dl class="summary-dl">'
        + _summary_field_html("模型", settings.agent_model)
        + _summary_field_html("Base URL", settings.agent_base_url, copyable=True, code=True)
        + _summary_secret_html("API Key", agent_key)
        + "</dl>"
        "</section>"
        '<section class="summary-section">'
        '<div class="summary-section-head">'
        '<span class="summary-section-title">独立评审</span>'
        + _summary_status_badge(settings.reviewer_enabled)
        + "</div>"
        '<dl class="summary-dl">'
        + _summary_field_html("模型", settings.reviewer_model)
        + _summary_field_html("Base URL", reviewer_base_display, copyable=True, code=True)
        + _summary_secret_html("API Key", reviewer_key)
        + "</dl>"
        "</section>"
        '<section class="summary-section">'
        '<div class="summary-section-head">'
        '<span class="summary-section-title">OCR 与默认参数</span>'
        + _summary_status_badge(settings.ocr_enabled)
        + "</div>"
        '<dl class="summary-dl">'
        + _summary_field_html("OCR URL", ocr_url, copyable=True, code=True, fallback="未配置 OCR 地址")
        + _summary_secret_html("OCR Token", ocr_token)
        + "</dl>"
        '<div class="summary-param-row">'
        f'<span class="mini-param"><b>Step Budget</b>{format_step_budget_label(settings.max_steps)}</span>'
        f'<span class="mini-param"><b>Temperature</b>{settings.temperature:.1f}</span>'
        "</div>"
        "</section>"
        "</div>"
    )


def _settings_to_ui_values(
    settings: UISettings,
    source: str,
    status_message: str,
) -> tuple[Any, ...]:
    agent_preset = _match_preset(settings.agent_model, settings.agent_base_url, MODEL_PRESETS)
    reviewer_base_for_match = settings.reviewer_base_url or UISettings().reviewer_base_url
    reviewer_preset = _match_preset(settings.reviewer_model, reviewer_base_for_match, REVIEWER_PRESETS)
    summary_html = _build_settings_summary_html(settings, source)
    return (
        agent_preset,
        settings.agent_model,
        settings.agent_base_url,
        settings.agent_api_key,
        settings.reviewer_enabled,
        reviewer_preset,
        settings.reviewer_model,
        settings.reviewer_base_url,
        settings.reviewer_api_key,
        settings.max_steps,
        settings.temperature,
        settings.ocr_enabled,
        settings.ocr_api_url,
        settings.ocr_token,
        settings.search_quota,
        settings.run_mode,
        status_message,
        summary_html,
        "",
    )


def load_saved_settings_handler() -> tuple[Any, ...]:
    settings, source = load_ui_settings(DEFAULT_SETTINGS_PATH)
    message = f"已载入{_settings_source_label(source)}设置。"
    return _settings_to_ui_values(settings, source, message)


def load_env_settings_handler() -> tuple[Any, ...]:
    settings = UISettings.from_env_defaults()
    message = "已从环境变量重新填充设置，尚未写入本地文件。"
    return _settings_to_ui_values(settings, "env", message)


def reset_default_settings_handler() -> tuple[Any, ...]:
    settings = UISettings()
    message = "已恢复到内置默认值，尚未写入本地文件。"
    return _settings_to_ui_values(settings, "default", message)


def save_settings_handler(
    agent_model: str,
    agent_base_url: str,
    agent_api_key: str,
    reviewer_enabled: bool,
    reviewer_model: str,
    reviewer_base_url: str,
    reviewer_api_key: str,
    max_steps: int,
    temperature: float,
    ocr_enabled: bool,
    ocr_api_url: str,
    ocr_token: str,
    search_quota: int = 8,
    run_mode: str = "survey",
) -> tuple[str, str]:
    settings = _build_ui_settings(
        agent_model,
        agent_base_url,
        agent_api_key,
        reviewer_enabled,
        reviewer_model,
        reviewer_base_url,
        reviewer_api_key,
        max_steps,
        temperature,
        ocr_enabled,
        ocr_api_url,
        ocr_token,
        search_quota=search_quota,
        run_mode=run_mode,
    )
    save_path = save_ui_settings(settings, DEFAULT_SETTINGS_PATH)
    message = f"设置已保存到 `{save_path}`。仅保存在本机，不会自动提交到仓库。"
    return message, _build_settings_summary_html(settings, "file")


def apply_agent_preset(
    preset_name: str,
    current_model: str,
    current_base_url: str,
) -> tuple[str, str]:
    if preset_name == CUSTOM_PRESET_LABEL:
        return current_model, current_base_url
    preset = _resolve_model_preset(preset_name)
    return preset["model"], preset["base_url"]


def apply_reviewer_preset(
    preset_name: str,
    current_model: str,
    current_base_url: str,
) -> tuple[str, str]:
    if preset_name == CUSTOM_PRESET_LABEL:
        return current_model, current_base_url
    preset = _resolve_reviewer_preset(preset_name)
    return preset["model"], preset["base_url"]


def test_connection_handler(api_key: str, base_url: str, model: str) -> str:
    resolved_api_key = _resolve_api_key(api_key, base_url, model)
    resolved_base_url = base_url.strip() or UISettings().agent_base_url
    resolved_model = model.strip() or UISettings().agent_model
    if not resolved_api_key:
        return "请先填写 API Key，或者确保环境变量里已经配置。"

    try:
        client = OpenAI(
            api_key=resolved_api_key,
            base_url=resolved_base_url,
            timeout=20,
        )
        client.chat.completions.create(
            model=resolved_model,
            messages=[{"role": "user", "content": "Reply with OK."}],
            max_tokens=1,
        )
        return f"连接测试成功：`{resolved_model}` 可用。"
    except Exception as exc:
        message = str(exc).strip()
        if len(message) > 260:
            message = message[:260] + "…"
        return f"连接测试失败：{message}"


def _apply_ocr_runtime_settings(enabled: bool, token: str, api_url: str) -> None:
    if not enabled:
        return

    resolved_token = token.strip() or _get_env_ocr_token()
    resolved_url = api_url.strip() or _get_env_ocr_url()

    if resolved_token:
        os.environ["PADDLEOCR_TOKEN"] = resolved_token
        os.environ["OCR_TOKEN"] = resolved_token
        os.environ["PADDLEOCR_VL_TOKEN"] = resolved_token
    if resolved_url:
        os.environ["PADDLEOCR_API_URL"] = resolved_url
        os.environ["OCR_API_URL"] = resolved_url
        os.environ["PADDLEOCR_VL_API_URL"] = resolved_url


def _build_runtime_config(
    agent_model: str,
    agent_base_url: str,
    agent_api_key: str,
    reviewer_enabled: bool,
    reviewer_model: str,
    reviewer_base_url: str,
    reviewer_api_key: str,
    max_steps: int,
    temperature: float,
    sandbox_dir: Path,
    ocr_enabled: bool,
    search_quota: int = 8,
) -> AgentConfig:
    resolved_agent_api_key = _resolve_api_key(agent_api_key, agent_base_url, agent_model)
    resolved_agent_base = agent_base_url.strip() or UISettings().agent_base_url
    resolved_reviewer_base = reviewer_base_url.strip() or UISettings().reviewer_base_url
    resolved_reviewer_key = _resolve_api_key(
        reviewer_api_key.strip() or "",
        resolved_reviewer_base,
        reviewer_model,
    )

    return AgentConfig(
        agent_model=ModelConfig(
            model=agent_model.strip() or UISettings().agent_model,
            api_key=resolved_agent_api_key,
            base_url=resolved_agent_base,
            temperature=temperature,
            max_tokens=8192,
        ),
        reviewer=ReviewerConfig(
            enabled=reviewer_enabled,
            model=ModelConfig(
                model=reviewer_model.strip() or UISettings().reviewer_model,
                api_key=resolved_reviewer_key,
                base_url=resolved_reviewer_base,
                temperature=0.0,
                max_tokens=2048,
            ),
            checkpoints=[
                "after_literature",
                "after_experiments",
                "before_report",
                "before_submit",
            ],
        ),
        sandbox=SandboxConfig(
            backend="subprocess",
            timeout=300,
            working_dir=str(sandbox_dir),
            # Empty → auto-detect (uses the interpreter that runs the UI).
            # Override via the SCIPRM_PYTHON_EXECUTABLE env var if you want
            # code to run under a different env, e.g. one with heavy domain
            # packages (rdkit, deepchem, etc.) pre-installed.
            python_executable=os.environ.get("SCIPRM_PYTHON_EXECUTABLE", ""),
        ),
        max_steps=max_steps,
        record_trajectory=True,
        trajectory_dir=str(DEFAULT_TRAJ_DIR),
        ocr_enabled=bool(ocr_enabled),
        verbose=True,
        search_quota=int(search_quota or 0),
    )


def _build_chat_status(status: str, model: str, step_count: int, max_steps: int) -> str:
    """Compose a compact research-workspace status bar."""
    state_html = _status_badge_html(_runtime_status_key(status))
    return (
        '<div class="chat-status-bar">'
        f'{state_html}'
        f'<span>Model · <code>{model or "—"}</code></span>'
        f'<span>Steps · <code>{format_step_progress(step_count, max_steps)}</code></span>'
        f'<span>Sandbox · <code>{_runner.sandbox_dir.name if _runner.sandbox_dir else "—"}</code></span>'
        '</div>'
    )


def chat_quick_inject(message: str):
    """Synchronous, ``queue=False``. Clear the input box and stash the typed
    text for the streaming generator to pick up.

    Returns: (cleared_input, stashed_message)
    Where ``stashed_message`` is the original text only when the agent was
    idle — that's the signal to ``chat_main_stream`` to start a new run.
    If the agent is currently running, ``stashed_message`` is empty so the
    queued main_stream call becomes a no-op.
    """
    text = (message or "").strip()
    if _runner.status == "running":
        return "", ""
    return "", text


def _stage_attachments_into_sandbox(
    sandbox_dir: Path,
    attachments,
) -> list[dict[str, str]]:
    """Copy user-attached files into ``<sandbox>/uploads/`` for the agent.

    The ``+ Attach`` button is a ``gr.UploadButton`` whose value is either a
    list of objects with a ``.name`` attribute (Gradio FileData) or a list
    of plain string paths. We treat each entry as a real local file and
    copy it under the run's sandbox so the agent can address it via a
    stable relative path. Returns a list of ``{"path", "name", "kind"}``
    dicts that the prompt-builder can use to brief the agent.
    """
    if not attachments:
        return []
    import shutil
    uploads_dir = sandbox_dir / "uploads"
    uploads_dir.mkdir(parents=True, exist_ok=True)
    staged: list[dict[str, str]] = []
    for entry in attachments:
        src_path = getattr(entry, "name", None) or str(entry)
        if not src_path:
            continue
        src = Path(src_path)
        if not src.exists() or not src.is_file():
            continue
        # Preserve the user-visible filename. If two uploads share a name,
        # prefix with a counter so neither is silently dropped.
        target = uploads_dir / src.name
        if target.exists():
            stem, suffix = target.stem, target.suffix
            counter = 1
            while True:
                candidate = uploads_dir / f"{stem}_{counter}{suffix}"
                if not candidate.exists():
                    target = candidate
                    break
                counter += 1
        try:
            shutil.copy2(src, target)
        except Exception as exc:
            logger.warning("Failed to stage attachment %s: %s", src, exc)
            continue
        suffix = target.suffix.lower()
        if suffix == ".pdf":
            kind = "PDF"
        elif suffix in {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".gif"}:
            kind = "image"
        elif suffix in {".md", ".markdown", ".txt", ".text"}:
            kind = "text"
        elif suffix in {".csv", ".tsv"}:
            kind = "table"
        elif suffix == ".json":
            kind = "json"
        elif suffix == ".zip":
            kind = "archive"
        else:
            kind = "file"
        staged.append({
            "path": str(target),
            "name": target.name,
            "kind": kind,
        })
    return staged


def _attachments_brief(staged: list[dict[str, str]], ocr_enabled: bool) -> str:
    """Render a short Chinese brief listing every staged attachment.

    The returned string is appended to the task's ``data_desc`` so the
    agent's system prompt explicitly tells it: "the user uploaded these
    files; use read_paper_fulltext / file_read to consume them".
    """
    if not staged:
        return ""
    has_pdf = any(item["kind"] == "PDF" for item in staged)
    lines = ["用户已上传以下本地资料（已放入沙箱 uploads/ 目录）："]
    for item in staged:
        lines.append(f"- `{item['path']}` （{item['kind']}）")
    lines.append("")
    needs_ocr = any(item["kind"] in ("PDF", "image") for item in staged)
    if needs_ocr and not ocr_enabled:
        lines.append(
            "注意：上传中包含 PDF / 图片资料，但当前 OCR 未启用 —— "
            "请提示用户在「模型与集成设置」中开启 PaddleOCR 后再次发送，"
            "或对于纯文本 PDF 用 read_paper_fulltext 直接抽取文本。"
        )
    else:
        lines.append(
            "请优先用 `read_paper_fulltext(local_path=...)` 读取 PDF / 图片"
            "（自动走 PaddleOCR-VL 抽取正文与公式），用 `file_read(path=...)` "
            "读取 .md / .txt / .json / .csv 等文本资料。读取后再回答用户的问题。"
        )

    # When the user uploaded papers, those uploads are almost always the
    # primary research artifact (e.g. "improve the algorithm in this paper").
    # Without this rule the agent treats arXiv search as the main entry
    # point and burns through its quota on tangential queries — exactly
    # what we saw on trajectory 32afa51f (19 search_literature calls, 11
    # quota-denied, while the actual uploaded paper was already on disk).
    if has_pdf:
        lines.append("")
        lines.append(
            "**研究对象优先级（hard rule）**：用户上传的论文 / PDF 是本次研究"
            "的主要参考文献和研究对象。请按以下顺序工作：\n"
            "  1. 先 `read_paper_fulltext(local_path=...)` 读完每一份上传"
            "的 PDF，提取核心方法、实验设置、基线对比与限制讨论。\n"
            "  2. 仅当上传论文中提到的具体方法 / 数据集 / 评估指标需要补充"
            "上下文时，才调 `search_literature` —— **每次外部搜索前先问自己：'"
            "这个问题真的没法靠上传的 PDF 回答吗？'**\n"
            "  3. `search_literature` 配额非常有限（默认每 run 8 次）。一旦"
            "看到 'quota exhausted' 错误，立刻停止该工具的调用，把剩余预算"
            "留给 `execute_code` 与报告生成。"
        )
    return "\n".join(lines)


def chat_main_stream(
    stashed_text: str,
    history: list[dict[str, str]] | None,
    agent_model: str,
    agent_base_url: str,
    agent_api_key: str,
    reviewer_enabled: bool,
    reviewer_model: str,
    reviewer_base_url: str,
    reviewer_api_key: str,
    max_steps: int,
    temperature: float,
    ocr_enabled: bool,
    ocr_api_url: str,
    ocr_token: str,
    search_quota: int = 8,
    attachments=None,
    run_mode: str = "survey",
):
    """Queued streaming generator for the central run, pipeline, and evidence panels.

    Always runs in auto mode end-to-end. If ``stashed_text`` is empty, this
    is a no-op (e.g. the user typed during a running job). Otherwise the
    text is treated as a fresh research topic.
    """
    text = (stashed_text or "").strip()
    if not text:
        yield gr.update(), gr.update(), gr.update(), gr.update()
        return

    history = list(history or [])

    resolved_api_key = _resolve_api_key(agent_api_key, agent_base_url, agent_model)
    if not resolved_api_key:
        history.append({"role": "user", "content": text})
        history.append({
            "role": "assistant",
            "content": (
                "还没有可用的 API Key。请到「模型与集成设置」页填写并保存，"
                "或通过 `.env` 设置 `AI_STUDIO_API_KEY` / `minimax_API_KEY` / `API_KEY` 等。"
            ),
        })
        pipeline, evidence = _runner_ui_snapshots()
        yield (
            history,
            _build_chat_status("error", agent_model, 0, max_steps),
            pipeline,
            evidence,
        )
        return

    # Append the user's bubble + a bootstrap assistant card.
    history.append({"role": "user", "content": text})
    history.append({
        "role": "assistant",
        "content": (
            f"正在拆解任务并准备沙箱...\n\n"
            f"- Model: `{agent_model}`\n"
            f"- Step budget: `{format_step_budget_label(max_steps)}`\n"
            f"- 自动模式：Agent 将一气呵成跑完整个研究流程并产出报告。"
        ),
    })

    task_desc, expected_output, data_desc = topic_to_task(text, "", mode=run_mode)
    task_id = str(uuid.uuid4())[:8]
    run_sandbox_dir = _prepare_run_sandbox(task_id)
    _apply_ocr_runtime_settings(bool(ocr_enabled), ocr_token, ocr_api_url)

    staged_attachments = _stage_attachments_into_sandbox(run_sandbox_dir, attachments)
    if staged_attachments:
        attachments_brief = _attachments_brief(staged_attachments, bool(ocr_enabled))
        if attachments_brief:
            data_desc = (data_desc + "\n\n" + attachments_brief).strip() if data_desc else attachments_brief
        history[-1]["content"] = (
            f"已附加 {len(staged_attachments)} 份资料到沙箱 `uploads/`，"
            f"Agent 将通过 read_paper_fulltext / file_read 读取后再作答。\n\n"
            + history[-1]["content"]
        )

    config = _build_runtime_config(
        agent_model=agent_model,
        agent_base_url=agent_base_url,
        agent_api_key=resolved_api_key,
        reviewer_enabled=reviewer_enabled,
        reviewer_model=reviewer_model,
        reviewer_base_url=reviewer_base_url,
        reviewer_api_key=reviewer_api_key,
        max_steps=max_steps,
        temperature=temperature,
        sandbox_dir=run_sandbox_dir,
        ocr_enabled=ocr_enabled,
        search_quota=search_quota,
    )

    _runner.reset_for_run(max_steps=max_steps, sandbox_dir=run_sandbox_dir)

    auto_instruction = (
        "UI RUN MODE: AUTO.\n"
        "Do not ask the user to confirm your plan, proposed direction, "
        "experiment design, or next phase. The user has already delegated "
        "the run to you. If you would normally write 'please confirm' or "
        "'是否同意', instead make the safest reasonable choice and call "
        "the next appropriate tool."
    )
    task_desc = f"{task_desc}\n\n{auto_instruction}"

    thread = threading.Thread(
        target=_runner.run,
        args=(config, task_id, task_desc, expected_output, data_desc),
        daemon=True,
    )
    thread.start()

    pipeline, evidence = _runner_ui_snapshots()
    yield (
        history,
        _build_chat_status("running", agent_model, 0, max_steps),
        pipeline,
        evidence,
    )

    last_seen = 0
    while _runner.status == "running":
        with _runner._lock:
            new_steps = _runner.steps_md[last_seen:]
            last_seen = len(_runner.steps_md)
        for step_md in new_steps:
            history.append({"role": "assistant", "content": step_md})
        pipeline, evidence = _runner_ui_snapshots()
        yield (
            history,
            _build_chat_status("running", agent_model, last_seen, max_steps),
            pipeline,
            evidence,
        )
        time.sleep(0.4)

    # Drain anything that arrived after the loop's last poll.
    with _runner._lock:
        new_steps = _runner.steps_md[last_seen:]
        last_seen = len(_runner.steps_md)
    for step_md in new_steps:
        history.append({"role": "assistant", "content": step_md})

    # Final summary turn. Whether to invite the user to "download the
    # report" depends on whether the agent actually wrote one — calling
    # submit_result with a textual summary alone does NOT produce
    # research_report.md / paperforge_bundle.json. Lying about it ("go
    # download the PDF") wastes the user's time. Check the sandbox.
    if _runner.status == "error":
        history.append({"role": "assistant", "content": f"运行出错：\n```\n{_runner.error_msg}\n```"})
    else:
        traj = _runner.trajectory
        if traj and traj.outcome == "success":
            summary = ""
            for step in reversed(traj.steps):
                if step.action_name == "submit_result":
                    summary = step.observation
                    break
            if summary:
                history.append({"role": "assistant", "content": f"## 研究结论\n\n{summary}"})

            report_path = run_sandbox_dir / "research_report.md"
            bundle_path = run_sandbox_dir / "paperforge_bundle.json"
            artefact_bits: list[str] = []
            if report_path.exists():
                artefact_bits.append("research_report.md")
            if bundle_path.exists():
                artefact_bits.append("paperforge_bundle.json")

            generate_report_called = any(
                step.action_name == "generate_report" for step in traj.steps
            )

            if artefact_bits:
                history.append({
                    "role": "assistant",
                    "content": (
                        f"任务完成 · 共 {len(traj.steps)} 步。\n\n"
                        f"工作区：`{run_sandbox_dir}`\n\n"
                        f"已生成：{', '.join('`' + n + '`' for n in artefact_bits)}。"
                        " 去「工作区与论文」面板下载 Markdown / PaperForge 包 / 编译 PDF。"
                    ),
                })
            else:
                # Agent submitted a text result but never wrote a real
                # report. Don't promise downloads that don't exist.
                missing_note = (
                    "`research_report.md` 与 `paperforge_bundle.json` 都未生成 —— "
                    "Agent 提交了文本总结但没调 `generate_report` 工具。"
                    if not generate_report_called
                    else "`generate_report` 调用未产出 markdown 报告，请检查工具日志。"
                )
                history.append({
                    "role": "assistant",
                    "content": (
                        f"任务完成 · 共 {len(traj.steps)} 步。\n\n"
                        f"工作区：`{run_sandbox_dir}`\n\n"
                        f"⚠ 没有可导出的论文产物：{missing_note}\n\n"
                        "下次跑研究任务时可以在指令里明确要求 \"完成后调 generate_report 工具"
                        "把研究过程整理为 research_report.md\"，或直接基于上面工作区里"
                        "的 csv / 图片 等中间产物自行整理。"
                    ),
                })
        elif traj:
            history.append({
                "role": "assistant",
                "content": f"Agent 最终状态：`{traj.outcome}` · {len(traj.steps)} 步",
            })

    pipeline, evidence = _runner_ui_snapshots()
    yield (
        history,
        _build_chat_status(_runner.status, agent_model, last_seen, max_steps),
        pipeline,
        evidence,
    )



def chat_clear_history() -> tuple[list, str, str, str, str]:
    """Reset the chat panel without touching settings.

    Note: a running agent thread cannot be interrupted from here — wait for
    it to finish before calling this. The button just clears UI state.
    """
    with _runner._lock:
        if _runner.status != "running":
            _runner.steps_md = []
            _runner.step_infos = []
            _runner.status = "idle"
    return (
        [],
        "",
        _build_chat_status("idle", "", 0, _runner.max_steps),
        _build_research_pipeline_html([], "idle", _runner.max_steps, ""),
        _build_evidence_trace_html([], "idle"),
    )


def _iter_report_workspaces() -> list[Path]:
    candidates: list[Path] = []
    if _runner.sandbox_dir and _runner.sandbox_dir.exists():
        candidates.append(_runner.sandbox_dir)

    if SANDBOX_ROOT.exists():
        extra_dirs = sorted(
            [path for path in SANDBOX_ROOT.iterdir() if path.is_dir()],
            key=lambda item: item.stat().st_mtime,
            reverse=True,
        )
        for path in extra_dirs:
            if path not in candidates:
                candidates.append(path)
    return candidates


def _find_paper_artifacts(
    preferred_dir: Path | None = None,
) -> tuple[Path | None, Path | None, Path | None, Path | None]:
    candidate_dirs: list[Path] = []
    if preferred_dir is not None and preferred_dir.exists():
        candidate_dirs.append(preferred_dir)
    for path in _iter_report_workspaces():
        if path not in candidate_dirs:
            candidate_dirs.append(path)

    for candidate_dir in candidate_dirs:
        report_path = candidate_dir / "research_report.md"
        bundle_path = candidate_dir / "paperforge_bundle.json"
        pdf_path = candidate_dir / "research_paper.pdf"
        if report_path.exists() or bundle_path.exists() or pdf_path.exists():
            return (
                candidate_dir,
                report_path if report_path.exists() else None,
                bundle_path if bundle_path.exists() else None,
                pdf_path if pdf_path.exists() else None,
            )
    return None, None, None, None


def _file_download_update(path: Path | None):
    if path is None or not path.exists():
        return gr.update(value=None, visible=False)
    return gr.update(value=str(path), visible=True)


# Subdirectories of a sandbox we deliberately exclude from the
# "Full bundle" zip — they're either user input (uploads), throwaway OCR
# downloads (papers — the cached PDF chunks read by read_paper_fulltext)
# or the framework's signed-URL temporary cache.
_FULL_BUNDLE_SKIP_DIRS = {"uploads", ".remote_images"}


def build_full_bundle_zip(workspace_dir: Path | None = None) -> Path | None:
    """Package every artefact of the latest run as a single ``.zip``.

    Includes (when present):
    * ``research_report.md`` — the markdown report
    * ``paperforge_bundle.json`` — the structured paper bundle
    * ``research_paper.pdf`` — the compiled PDF
    * ``.research_paper_latex/paper.tex`` and the ``figures/`` dir — the
      LaTeX source paper_forge produced (kept thanks to ``keep_tex=True``)
    * Every ``.py`` / ``.csv`` / ``.json`` / ``.png`` / ``.jpg`` / ``.svg``
      / ``.npy`` / ``.txt`` produced during the run — i.e. all experiment
      code, charts, intermediate results
    * ``logs/`` — the per-step bash + python logs the sandbox persisted
    * ``literature_cache.jsonl`` — the cached search hits

    Excludes ``uploads/`` (already on the user's disk) and the OCR /
    remote-image throwaway caches.

    Returns the zip path, or ``None`` when no run produced anything.
    """
    import zipfile

    if workspace_dir is None:
        workspace_dir, *_ = _find_paper_artifacts()
    if workspace_dir is None or not workspace_dir.exists():
        return None

    # Use a stable name keyed off the run id (the sandbox dir name) so
    # repeat downloads overwrite the previous bundle instead of piling
    # up an unbounded number of zips on disk.
    bundle_name = f"lab_forge_run_{workspace_dir.name}.zip"
    bundle_path = workspace_dir / bundle_name

    # Always rebuild — the agent may have appended new artefacts since
    # the last download, so a stale zip would mislead the user.
    if bundle_path.exists():
        try:
            bundle_path.unlink()
        except OSError:
            pass

    with zipfile.ZipFile(bundle_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in sorted(workspace_dir.rglob("*")):
            if not f.is_file():
                continue
            try:
                rel = f.relative_to(workspace_dir)
            except ValueError:
                continue
            # Don't recurse into excluded dirs.
            if any(part in _FULL_BUNDLE_SKIP_DIRS for part in rel.parts):
                continue
            # Don't include the bundle in itself.
            if f.name == bundle_name:
                continue
            try:
                zf.write(f, arcname=str(rel))
            except OSError as exc:
                logger.warning("Skipped %s in bundle zip: %s", rel, exc)

    if bundle_path.stat().st_size == 0:
        try:
            bundle_path.unlink()
        except OSError:
            pass
        return None
    return bundle_path


def download_full_bundle_zip():
    """Gradio handler: build (or refresh) the full-run zip and return its path.

    Designed for ``gr.DownloadButton.click(...)``. Returns a
    ``gr.update(...)`` so the button surface stays accurate even when no
    artefacts exist yet.
    """
    bundle_path = build_full_bundle_zip()
    return _file_download_update(bundle_path)


def _format_file_size(path: Path | None) -> str:
    if path is None or not path.exists():
        return "—"
    size = path.stat().st_size
    units = ["B", "KB", "MB", "GB"]
    value = float(size)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            if unit == "B":
                return f"{int(value)} {unit}"
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{size} B"


def _artifact_meta_html(path: Path | None, file_type: str, icon_label: str) -> str:
    if path is None or not path.exists():
        return ""
    full_path = str(path)
    return (
        '<div class="artifact-meta-card">'
        f'<span class="artifact-icon">{_escape_html(icon_label)}</span>'
        '<div class="artifact-name-wrap">'
        f'<div class="artifact-file-name" title="{_escape_html(full_path)}">{_escape_html(path.name)}</div>'
        '<div class="artifact-subrow">'
        f'<span class="type-badge">{_escape_html(file_type)}</span>'
        '</div>'
        '</div>'
        f'<span class="artifact-size">{_escape_html(_format_file_size(path))}</span>'
        '</div>'
    )


def _paper_artifacts_status_html(
    workspace_dir: Path | None,
    report_path: Path | None,
    bundle_path: Path | None,
    pdf_path: Path | None,
    status_message: str = "",
) -> str:
    if workspace_dir is None:
        message = status_message or "完成一次研究任务后，Markdown、PaperForge 包和 PDF 会在这里出现。"
        return (
            '<div class="artifact-empty">'
            '<div><strong>暂无导出产物</strong>'
            f'<div>{_escape_html(message)}</div></div>'
            '</div>'
        )

    available = []
    if report_path is not None:
        available.append("Markdown")
    if bundle_path is not None:
        available.append("PaperForge 包")
    if pdf_path is not None:
        available.append("PDF")

    directory_field = (
        '<span class="copy-code-field">'
        f'<code title="{_escape_html(str(workspace_dir))}">{_escape_html(workspace_dir.name)}</code>'
        f'{_copy_button_html(str(workspace_dir), "复制论文目录")}'
        '</span>'
    )
    status_lines = [
        f"<strong>当前论文目录</strong> {directory_field}",
        f"已检测到 {len(available)} 个产物：{_escape_html('、'.join(available) or '暂无')}。",
    ]
    if pdf_path is None and (report_path is not None or bundle_path is not None):
        status_lines.append("如需 PDF，请点击“导出 PDF”。")
    if status_message:
        status_lines.insert(0, _escape_html(status_message).replace("\n", "<br>"))
    return '<div class="artifact-status-card">' + "<br>".join(status_lines) + "</div>"


def load_latest_paper_artifacts(preferred_dir: Path | None = None, status_message: str = ""):
    workspace_dir, report_path, bundle_path, pdf_path = _find_paper_artifacts(preferred_dir)
    return (
        gr.update(visible=report_path is not None),
        gr.update(value=_artifact_meta_html(report_path, "Markdown", "MD"), visible=report_path is not None),
        _file_download_update(report_path),
        gr.update(visible=bundle_path is not None),
        gr.update(value=_artifact_meta_html(bundle_path, "PaperForge bundle", "JSON"), visible=bundle_path is not None),
        _file_download_update(bundle_path),
        gr.update(visible=pdf_path is not None),
        gr.update(value=_artifact_meta_html(pdf_path, "PDF", "PDF"), visible=pdf_path is not None),
        _file_download_update(pdf_path),
        gr.update(value=_paper_artifacts_status_html(workspace_dir, report_path, bundle_path, pdf_path, status_message)),
    )


def load_latest_paper_downloads(preferred_dir: Path | None = None):
    workspace_dir, report_path, bundle_path, pdf_path = _find_paper_artifacts(preferred_dir)
    if workspace_dir is None:
        return (
            _file_download_update(None),
            _file_download_update(None),
            _file_download_update(None),
            "*还没有检测到可下载的论文文件。先运行研究任务，生成 `research_report.md` 后这里会自动出现下载入口。*",
        )

    available = []
    if report_path is not None:
        available.append("Markdown 论文")
    if bundle_path is not None:
        available.append("PaperForge 导出包")
    if pdf_path is not None:
        available.append("论文 PDF")

    status = (
        f"当前论文目录：`{workspace_dir.name}`。"
        f" 可下载：{'、'.join(available)}。"
    )
    if pdf_path is None and (report_path is not None or bundle_path is not None):
        status += " 如需 PDF，请点击“导出 PDF”。"

    return (
        _file_download_update(report_path),
        _file_download_update(bundle_path),
        _file_download_update(pdf_path),
        status,
    )


def _paper_forge_import_error_message(exc: ImportError) -> str:
    missing_name = (getattr(exc, "name", "") or "").split(".")[0]
    package_map = {
        "fpdf": "fpdf2",
        "PIL": "Pillow",
    }
    suggested_packages = []
    if missing_name in package_map:
        suggested_packages.append(package_map[missing_name])
    else:
        suggested_packages.extend(["fpdf2", "Pillow"])
    install_cmd = f"{shlex.quote(sys.executable)} -m pip install {' '.join(suggested_packages)}"
    return (
        f"无法导入 paper_forge 依赖：{exc}。\n\n"
        f"当前 UI 使用的 Python：`{sys.executable}`\n"
        f"请在同一个环境执行：`{install_cmd}`\n"
        "如果你是新建环境启动这个项目，也可以重新执行：`pip install -r lab_forge/requirements.txt`"
    )


def convert_report_to_pdf(
    paper_language: str = "auto",
    template_key: str = "general_article",
    agent_model: str = "",
    agent_base_url: str = "",
    agent_api_key: str = "",
):
    """Export the latest research report to PDF through PaperForge.

    Always regenerates each section through the top-conference style guide
    in :mod:`paper_forge.style_guide` — strict anti-fabrication rules, per-
    section word targets, top-conf rhetorical structure. The LLM credentials
    reuse the agent's settings page (so users don't need to configure twice).
    """
    workspace_dir, report_path, bundle_path, _ = _find_paper_artifacts()

    if workspace_dir is None:
        return (
            _file_download_update(None),
            _file_download_update(None),
            _file_download_update(None),
            "没有找到可导出的报告。请先运行一次研究任务。",
        )

    try:
        if str(PAPER_FORGE_ROOT) not in sys.path:
            sys.path.insert(0, str(PAPER_FORGE_ROOT))
        from paper_forge import render_paper_pdf
        from paper_forge.pdf_quality_agent import format_quality_summary, inspect_paper_artifacts
        from paper_forge.paper_writer import _expand_sections, parse_markdown_paper
        from paper_forge.config import LLMConfig as PFLLMConfig
    except ImportError as exc:
        report_update, bundle_update, pdf_update, _ = load_latest_paper_downloads(workspace_dir)
        return (
            report_update,
            bundle_update,
            pdf_update,
            _paper_forge_import_error_message(exc),
        )

    try:
        if bundle_path is not None:
            paper = load_paper_bundle(bundle_path)
            images = load_bundle_images(paper, workspace_dir)
        elif report_path is not None:
            markdown_text = report_path.read_text(encoding="utf-8")
            try:
                from paper_forge.markdown_normalizer import normalize_markdown
                from langchain_openai import ChatOpenAI
                resolved_key = _resolve_api_key(api_key, base_url, effective_model)
                if resolved_key:
                    norm_llm = ChatOpenAI(api_key=resolved_key, base_url=effective_base_url, model=effective_model)
                    markdown_text = normalize_markdown(markdown_text, llm=norm_llm)
                else:
                    markdown_text = normalize_markdown(markdown_text)
            except Exception:
                pass
            paper = parse_markdown_paper(markdown_text, image_names=[])
            images = {}
        else:
            raise FileNotFoundError("Neither paperforge_bundle.json nor research_report.md exists.")

        forced_language = paper_language if paper_language in ("zh", "en") else None

        # Top-conference expansion always runs. Reuses the agent's LLM
        # credentials so the user doesn't configure a second model.
        expansion_status = ""
        saved_settings, _ = load_ui_settings()
        effective_model = (agent_model or "").strip() or saved_settings.agent_model
        effective_base_url = (agent_base_url or "").strip() or saved_settings.agent_base_url
        resolved_key = _resolve_api_key(
            agent_api_key, effective_base_url, effective_model
        )
        if not resolved_key:
            expansion_status = (
                "\n\n未检测到大模型 API Key，跳过顶会风格扩展，"
                "仅按 bundle 直接渲染。请在「模型与集成设置」配置 Key 后重试。"
            )
        else:
            logger.info(
                "Running PaperForge top-conf expansion (model=%s, forced_language=%s)",
                effective_model,
                forced_language or "auto",
            )
            pf_llm_config = PFLLMConfig(
                model=effective_model,
                api_key=resolved_key,
                base_url=effective_base_url,
                temperature=0.3,
                max_tokens=8192,
            )
            try:
                paper = _expand_sections(
                    paper,
                    pf_llm_config,
                    temperature=0.3,
                    target_total_words=None,
                    forced_language=forced_language,
                )
                expansion_status = (
                    "\n\n已按顶会论文写作范式扩展全部章节（受 style_guide "
                    "约束、禁止编造数字与引用）。"
                )
                if forced_language:
                    expansion_status += (
                        f"\n\n已强制将论文正文改写为 "
                        f"{'英文' if forced_language == 'en' else '中文'}。"
                    )
            except Exception as exc:
                logger.exception("Top-conf expansion failed")
                expansion_status = (
                    f"\n\nPaperForge 改写/扩展失败（已回退到原文渲染）：{exc}"
                )

        output_path = workspace_dir / "research_paper.pdf"
        render_paper_pdf(
            paper=paper,
            images=images,
            output_path=output_path,
            template_key=template_key if template_key else None,
            keep_tex=True,
            forced_language=forced_language,
        )
        qa_text = ""
        tex_path = workspace_dir / ".research_paper_latex" / "paper.tex"
        log_path = workspace_dir / ".research_paper_latex" / "paper.log"
        try:
            report = inspect_paper_artifacts(
                tex_path=tex_path,
                log_path=log_path,
                target_language=forced_language,
            )
            qa_text = "\n\n" + format_quality_summary(report)
        except Exception as exc:
            logger.warning("PDF QA summary failed: %s", exc)
        refs = len(paper.get("references", []))
        figs = sum(len(section.get("figures", [])) for section in paper.get("sections", []))
        tables = sum(len(section.get("tables", [])) for section in paper.get("sections", []))
        status = (
            f"PDF 已生成：`{output_path.name}`。"
            f" 复用 {refs} 条参考文献、{figs} 张图、{tables} 个表。"
            + expansion_status
            + qa_text
        )
        report_update, bundle_update, pdf_update, base_status = load_latest_paper_downloads(workspace_dir)
        return report_update, bundle_update, pdf_update, f"{status}\n\n{base_status}"
    except Exception as exc:
        logger.exception("PDF generation failed")
        report_update, bundle_update, pdf_update, base_status = load_latest_paper_downloads(workspace_dir)
        error_message = f"PDF 生成失败：{exc}"
        log_tail = getattr(exc, "log_tail", "")
        if log_tail:
            error_message += f"\n\nLaTeX 日志摘录：\n```text\n{log_tail.strip()}\n```"
        return report_update, bundle_update, pdf_update, f"{error_message}\n\n{base_status}"


def convert_report_to_pdf_artifacts(
    paper_language: str = "auto",
    template_key: str = "general_article",
    agent_model: str = "",
    agent_base_url: str = "",
    agent_api_key: str = "",
):
    _, _, _, status = convert_report_to_pdf(
        paper_language,
        template_key,
        agent_model,
        agent_base_url,
        agent_api_key,
    )
    return load_latest_paper_artifacts(status_message=status)


_OUTCOME_BADGES = {
    "success": "成功",
    "failure": "失败",
    "max_steps_exceeded": "超步数",
    "adaptive_step_budget_exhausted": "预算耗尽",
    "stalled_at_step_budget": "停滞",
    "stalled_without_submission": "未提交停滞",
    "aborted": "已中止",
    "incomplete": "… 未完成",
}


def _trajectory_label(path: Path) -> str:
    """Build a human-readable label for a trajectory file.

    Reads the JSON once to extract the research topic + outcome + step count
    + run time, so the dropdown shows what the task was about instead of an
    opaque 8-char hash.
    """
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return path.stem

    desc = (data.get("task_description") or "").strip()
    if not desc:
        preview = "(无主题描述)"
    else:
        # Use the first non-empty line as the headline; strip section markers
        # like "## Research Task" if the prompt template prefixed them.
        first = ""
        for line in desc.splitlines():
            line = line.strip().lstrip("#").strip()
            if line and not line.lower().startswith(("research task", "task:")):
                first = line
                break
        preview = first or desc.splitlines()[0].strip()
        if len(preview) > 50:
            preview = preview[:50] + "…"

    outcome = data.get("outcome", "incomplete")
    badge = _OUTCOME_BADGES.get(outcome, outcome)
    n_steps = len(data.get("steps", []))
    try:
        mtime = datetime.fromtimestamp(path.stat().st_mtime).strftime("%m-%d %H:%M")
    except Exception:
        mtime = ""
    pieces = [preview, f"{n_steps}步", badge]
    if mtime:
        pieces.append(mtime)
    return " · ".join(pieces)


def list_trajectory_choices() -> list[tuple[str, str]]:
    """Return ``[(label, task_id), ...]`` pairs sorted by recency.

    The label summarises the research topic so the user can pick a run
    by what it was about rather than by a random hex string.
    """
    if not DEFAULT_TRAJ_DIR.exists():
        return []
    files = sorted(
        DEFAULT_TRAJ_DIR.glob("*.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return [(_trajectory_label(path), path.stem) for path in files]


def load_trajectory_list():
    return gr.update(choices=list_trajectory_choices(), value=None)


def view_trajectory(task_id: str) -> str:
    if not task_id:
        return _history_empty_state_html()
    path = DEFAULT_TRAJ_DIR / f"{task_id}.json"
    if not path.exists():
        return f"未找到轨迹文件：`{path}`"
    trajectory = Trajectory.load(path)

    lines = [
        f"## Task `{trajectory.task_id}`",
        f"**状态** {trajectory.outcome} · **步数** {len(trajectory.steps)}",
        "",
        f"**任务描述** {trajectory.task_description[:300]}{'…' if len(trajectory.task_description) > 300 else ''}",
        "",
        "---",
        "",
    ]
    for idx, step in enumerate(trajectory.steps):
        lines.append(_format_step_card(step, idx))
    return "\n".join(lines)


def export_traj_json(task_id: str) -> str:
    if not task_id:
        return "{}"
    path = DEFAULT_TRAJ_DIR / f"{task_id}.json"
    if not path.exists():
        return "{}"
    return json.dumps(json.loads(path.read_text(encoding="utf-8")), indent=2, ensure_ascii=False)


def _storage_info_html() -> str:
    settings_path = str(DEFAULT_SETTINGS_PATH)
    return (
        '<div class="info-card storage-info">'
        "<ul>"
        "<li>设置仅保存在本机："
        '<span class="copy-code-field">'
        f'<code title="{_escape_html(settings_path)}">{_escape_html(settings_path)}</code>'
        f'{_copy_button_html(settings_path, "复制设置路径")}'
        "</span>"
        "</li>"
        "<li>该文件已加入 <code>lab_forge/.gitignore</code>，避免误提交。</li>"
        "<li>API Key 留空时，运行时会回退到环境变量。</li>"
        "</ul>"
        "</div>"
    )


def _history_empty_state_html() -> str:
    return (
        '<div class="history-empty">'
        '<div><strong>请选择一条运行记录</strong>'
        '<div>选择历史轨迹后，可查看结构化步骤或原始 JSON。</div></div>'
        '</div>'
    )
