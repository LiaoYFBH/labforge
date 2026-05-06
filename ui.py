"""
LabForge · Research Studio (redesigned UI).

A researcher-first, monochrome workbench rebuilt on top of the existing agent
backend. Every control here delegates to the canonical handlers exposed by
``lab_forge.ui`` — this module owns layout, typography and styling only.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import gradio as gr

# ---------------------------------------------------------------------------
# Backend reuse — every handler below comes from ``lab_forge.web`` (the chat
# runtime, settings handlers, paper-export handlers, trajectory readers, and
# all HTML rendering helpers). This file owns layout / wiring only.
# ---------------------------------------------------------------------------
_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR
if str(_REPO_ROOT) not in sys.path:
    # When ui-new.py is launched directly (``python ui-new.py``) the ``lab_forge``
    # package needs to be importable from the repo root.
    sys.path.insert(0, str(_REPO_ROOT))

from lab_forge import web as backend  # noqa: E402
legacy_ui = backend  # backward-compat alias used by older helpers below
from lab_forge.web import (  # noqa: E402
    CUSTOM_PRESET_LABEL,
    DEFAULT_SETTINGS_PATH,
    PAPER_FORGE_ROOT,
    SHOW_HISTORY_TAB,
    _artifact_meta_html,
    _build_chat_status,
    _build_evidence_trace_html,
    _build_research_pipeline_html,
    _build_settings_summary_html,
    _find_paper_artifacts,
    _match_preset,
    _settings_source_label,
    apply_agent_preset,
    apply_reviewer_preset,
    chat_clear_history,
    chat_main_stream,
    chat_quick_inject,
    convert_report_to_pdf_artifacts,
    download_full_bundle_zip,
    export_traj_json,
    format_step_budget_label,
    format_step_progress,
    list_trajectory_choices,
    load_env_settings_handler,
    load_latest_paper_artifacts,
    load_saved_settings_handler,
    load_trajectory_list,
    load_ui_settings,
    reset_default_settings_handler,
    save_settings_handler,
    test_connection_handler,
    view_trajectory,
)
from lab_forge.env_utils import load_project_env
from lab_forge.models import MODEL_PRESETS, REVIEWER_PRESETS

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# Static markup helpers (sidebar / header). All purely presentational.
# ---------------------------------------------------------------------------
def _brand_block_html() -> str:
    return (
        '<div class="brand-block">'
        '<div class="brand-mark">LF</div>'
        '<div class="brand-text">'
        '<div class="brand-title">LabForge</div>'
        '<div class="brand-tagline">Research Studio</div>'
        '</div>'
        '</div>'
    )


def _sidebar_section_label(eyebrow: str, title: str, count: int | None = None) -> str:
    badge = f'<span class="rail-count">{count}</span>' if count is not None else ""
    return (
        '<div class="rail-section-head">'
        f'<span class="rail-eyebrow">{eyebrow}</span>'
        f'<div class="rail-section-title">{title}{badge}</div>'
        '</div>'
    )


RAIL_PROJECT_SLOTS = 8


def _projects_header_html(item_count: int) -> str:
    return (
        '<section class="rail-section">'
        + _sidebar_section_label("01", "Research Projects", item_count)
        + "</section>"
    )


def _projects_empty_html() -> str:
    return (
        '<div class="rail-empty">还没有研究项目。<br/>'
        '<span class="muted">在右侧输入研究主题开始第一次任务。</span></div>'
    )


def _projects_more_html(extra: int) -> str:
    if extra <= 0:
        return ""
    return f'<div class="rail-more">+{extra} more — open Archive tab</div>'


def _projects_rail_state():
    """Snapshot the trajectory list and project it into rail-slot updates.

    Returns a tuple of:
      header_html, empty_html, [N button updates], [N task_ids], more_html
    """
    items = list_trajectory_choices() if SHOW_HISTORY_TAB else []
    n = len(items)
    btn_updates = []
    state_values = []
    for i in range(RAIL_PROJECT_SLOTS):
        if i < n:
            label, task_id = items[i]
            display = label if len(label) <= 56 else label[:54] + "…"
            btn_updates.append(gr.update(value=display, visible=True))
            state_values.append(task_id)
        else:
            btn_updates.append(gr.update(value="", visible=False))
            state_values.append("")
    header_html = _projects_header_html(n)
    empty_html = _projects_empty_html() if n == 0 else ""
    more_html = _projects_more_html(max(0, n - RAIL_PROJECT_SLOTS))
    return header_html, empty_html, btn_updates, state_values, more_html


def _papers_list_html() -> str:
    """Render the Saved Papers list using whatever artefacts the runner has."""
    workspace_dir, report_path, bundle_path, pdf_path = _find_paper_artifacts()
    artefacts = [
        (report_path, "Report", "MD"),
        (bundle_path, "Bundle", "JSON"),
        (pdf_path, "Paper", "PDF"),
    ]
    available = [(p, t, k) for p, t, k in artefacts if p is not None]

    head = _sidebar_section_label("02", "Saved Papers", len(available))
    if not available:
        body = (
            '<div class="rail-empty">尚未生成可下载的论文产物。<br/>'
            '<span class="muted">研究完成后这里会列出 report / bundle / PDF。</span></div>'
        )
    else:
        rows = []
        sandbox_name = workspace_dir.name if workspace_dir else "—"
        for path, kind, ext in available:
            size = _format_size(path)
            rows.append(
                '<div class="rail-paper">'
                f'<span class="rail-ext">{ext}</span>'
                '<div class="rail-paper-body">'
                f'<div class="rail-paper-name">{path.name}</div>'
                f'<div class="rail-paper-meta">{kind} · {size}</div>'
                '</div>'
                '</div>'
            )
        rows.append(
            f'<div class="rail-paper-source">workspace · <code>{sandbox_name}</code></div>'
        )
        body = '<div class="rail-list">' + "".join(rows) + "</div>"
    return f'<section class="rail-section">{head}{body}</section>'


def _task_queue_html(status: str = "idle", steps: int = 0, max_steps: int = 30, sandbox: str = "") -> str:
    """Tiny live indicator for the long-running agent task."""
    label_map = {
        "idle": ("Idle", "queue-idle", "等待新的研究指令"),
        "running": ("Running", "queue-running", f"步骤 {format_step_progress(steps, max_steps)}"),
        "done": ("Completed", "queue-done", "上一次任务已完成"),
        "error": ("Error", "queue-error", "上一次任务异常退出"),
    }
    label, css, hint = label_map.get(status, label_map["idle"])
    sandbox_html = (
        f'<div class="queue-sandbox">sandbox · <code>{sandbox}</code></div>'
        if sandbox
        else ""
    )
    head = _sidebar_section_label("03", "Task Queue")
    body = (
        f'<div class="queue-card {css}">'
        '<div class="queue-row">'
        f'<span class="queue-led"></span>'
        f'<span class="queue-label">{label}</span>'
        '</div>'
        f'<div class="queue-hint">{hint}</div>'
        f'{sandbox_html}'
        '</div>'
    )
    return f'<section class="rail-section">{head}{body}</section>'


def _pdf_preview_iframe(pdf_path: Path) -> str:
    """Embed an on-disk PDF inside the Manuscript tab as a real preview.

    We base64-encode the bytes and feed them to a ``<iframe>`` via a
    ``data:`` URI so the browser's built-in PDF viewer renders the
    actual paper — pagination, fonts, figures, equations and all. This
    avoids the previous "fake preview" (a few sentences of markdown
    typeset to look like a paper) which the user complained mis-sold a
    not-yet-finished export. Falls back to a small notice if reading
    the PDF fails.
    """
    try:
        import base64
        b64 = base64.b64encode(pdf_path.read_bytes()).decode("ascii")
    except Exception as exc:
        return (
            '<section class="doc-shell">'
            '<div class="doc-frame doc-frame-empty">'
            f'<div class="doc-empty-note">无法加载 PDF：{exc}</div>'
            '</div></section>'
        )
    size_kb = pdf_path.stat().st_size / 1024
    size_label = (
        f"{size_kb / 1024:.1f} MB" if size_kb >= 1024 else f"{size_kb:.0f} KB"
    )
    return (
        '<section class="doc-shell doc-shell-pdf">'
        '<div class="doc-meta">'
        '<span class="doc-meta-label">Live PDF</span>'
        f'<span class="doc-meta-name">{pdf_path.name}</span>'
        f'<span class="doc-meta-size">{size_label}</span>'
        '</div>'
        f'<iframe class="doc-pdf-frame" '
        f'src="data:application/pdf;base64,{b64}#toolbar=1&navpanes=0&view=FitH" '
        f'title="{pdf_path.name}"></iframe>'
        '</section>'
    )


def _document_preview_html(report_path: Path | None = None) -> str:
    """Render the Manuscript tab.

    Resolution order:
    1. ``research_paper.pdf`` next to the markdown report → embed as a real,
       scrollable PDF iframe.
    2. ``research_report.md`` only → show an explicit "not exported yet"
       placeholder. We intentionally do not render Markdown as a fake paper
       preview, because that made the Export button look meaningless.
    3. Nothing on disk → empty-state placeholder.
    """
    if report_path is None or not report_path.exists():
        return (
            '<section class="doc-shell">'
            '<div class="doc-frame doc-frame-empty">'
            '<div class="doc-watermark">EXPORT</div>'
            '<div class="doc-title">Manuscript Preview</div>'
            '<p class="doc-line w90"></p>'
            '<p class="doc-line w70"></p>'
            '<p class="doc-line w95"></p>'
            '<p class="doc-line w60"></p>'
            '<p class="doc-line w80"></p>'
            '<p class="doc-line w85"></p>'
            '<p class="doc-line w55"></p>'
            '<div class="doc-empty-note">点击 Export 面板中的 <code>Export PDF</code> 后，'
            '这里才会显示编译好的 <code>research_paper.pdf</code>。</div>'
            '</div>'
            '</section>'
        )

    pdf_candidate = report_path.parent / "research_paper.pdf"
    if pdf_candidate.exists() and pdf_candidate.stat().st_size > 0:
        return _pdf_preview_iframe(pdf_candidate)

    return (
        '<section class="doc-shell">'
        '<div class="doc-frame doc-frame-empty">'
        '<div class="doc-watermark">EXPORT</div>'
        '<div class="doc-title">PDF 尚未导出</div>'
        '<div class="doc-empty-note">已检测到 <code>research_report.md</code>。'
        '请到 Export 面板点击 <code>Export PDF</code>，完成 LaTeX 编译后这里会显示真正的 PDF 预览。</div>'
        '</div>'
        '</section>'
    )


def _format_size(path: Path | None) -> str:
    if path is None or not path.exists():
        return "—"
    size = path.stat().st_size
    if size < 1024:
        return f"{size} B"
    if size < 1024 * 1024:
        return f"{size / 1024:.1f} KB"
    return f"{size / (1024 * 1024):.1f} MB"


# ---------------------------------------------------------------------------
# Custom CSS — the heart of the redesign. Strict monochrome, 4/8pt grid,
# Serif content + Sans UI typography. No gradients, no glassmorphism.
# ---------------------------------------------------------------------------
CUSTOM_CSS = """
:root {
  --sp-bg: #ffffff;
  --sp-bg-soft: #fafafa;
  --sp-bg-rail: #f8f8f7;
  --sp-bg-tint: #f4f4f5;
  --sp-border: #ececec;
  --sp-border-strong: #d4d4d8;
  --sp-text-1: #0a0a0a;
  --sp-text-2: #525252;
  --sp-text-3: #a1a1aa;
  --sp-accent: #1d4ed8;
  --sp-accent-soft: #eef2ff;
  --sp-success: #166534;
  --sp-warning: #92400e;
  --sp-danger:  #991b1b;
  --sp-radius: 6px;
  --sp-radius-lg: 8px;
  --sp-shadow: 0 1px 0 0 rgba(15, 23, 42, 0.04);
  --sp-font-sans: "Inter", "PingFang SC", "Microsoft YaHei", system-ui, -apple-system, sans-serif;
  --sp-font-serif: "Source Serif 4", "Iowan Old Style", "Charter", Georgia, "Songti SC", serif;
  --sp-font-mono: "JetBrains Mono", "SF Mono", Menlo, Consolas, monospace;
}

/* Base reset of Gradio chrome --------------------------------------------- */
.gradio-container {
  background: var(--sp-bg) !important;
  color: var(--sp-text-1) !important;
  font-family: var(--sp-font-sans) !important;
  font-size: 13px !important;
  letter-spacing: 0;
  max-width: 100% !important;
  padding: 0 !important;
}
.gradio-container * { box-sizing: border-box; }
footer { display: none !important; }

/* Top bar ----------------------------------------------------------------- */
.sp-topbar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  height: 52px;
  padding: 0 24px;
  border-bottom: 1px solid var(--sp-border);
  background: var(--sp-bg);
}
.sp-topbar .brand-block { display: flex; align-items: center; gap: 10px; }
.sp-topbar .brand-mark {
  width: 28px; height: 28px;
  border-radius: var(--sp-radius);
  background: var(--sp-text-1);
  color: #fff;
  font-weight: 600;
  font-size: 12px;
  letter-spacing: 0;
  display: inline-flex;
  align-items: center;
  justify-content: center;
}
.sp-topbar .brand-title {
  font-weight: 600;
  font-size: 14px;
  color: var(--sp-text-1);
  line-height: 1;
}
.sp-topbar .brand-tagline {
  font-size: 11px;
  color: var(--sp-text-3);
  letter-spacing: 0;
  margin-top: 3px;
}
.sp-topbar-right { display: flex; align-items: center; gap: 16px; }
.sp-topbar-meta {
  font-size: 12px;
  color: var(--sp-text-3);
}
.sp-topbar-meta code {
  font-family: var(--sp-font-mono);
  background: var(--sp-bg-tint);
  border: 1px solid var(--sp-border);
  border-radius: 4px;
  padding: 1px 6px;
  color: var(--sp-text-2);
}

/* Tabs -------------------------------------------------------------------- */
.tab-nav, [role="tablist"] {
  background: transparent !important;
  border-bottom: 1px solid var(--sp-border) !important;
  padding: 0 24px !important;
}
.tab-nav button, [role="tab"] {
  background: transparent !important;
  color: var(--sp-text-2) !important;
  border: none !important;
  border-bottom: 2px solid transparent !important;
  border-radius: 0 !important;
  padding: 10px 14px !important;
  font-size: 13px !important;
  font-weight: 500 !important;
}
.tab-nav button.selected, [role="tab"][aria-selected="true"] {
  color: var(--sp-text-1) !important;
  border-bottom-color: var(--sp-text-1) !important;
}

/* Workspace grid ---------------------------------------------------------- */
.sp-workspace { padding: 0 !important; gap: 0 !important; }
.sp-workspace > .form,
.sp-workspace > div { gap: 0 !important; }
.sp-rail-left {
  background: var(--sp-bg-rail) !important;
  border-right: 1px solid var(--sp-border) !important;
  padding: 20px 16px !important;
  min-height: calc(100vh - 100px);
}
.sp-rail-right {
  border-left: 1px solid var(--sp-border) !important;
  padding: 20px 16px !important;
  background: var(--sp-bg) !important;
  min-height: calc(100vh - 100px);
}
.sp-main {
  padding: 16px 24px 0 24px !important;
  background: var(--sp-bg) !important;
  min-height: calc(100vh - 100px);
}

/* Sidebar sections -------------------------------------------------------- */
.rail-section { margin-bottom: 24px; }
.rail-section-head {
  display: flex; align-items: baseline; justify-content: space-between;
  margin-bottom: 8px;
  padding-bottom: 6px;
  border-bottom: 1px dashed var(--sp-border);
}
.rail-eyebrow {
  font-family: var(--sp-font-mono);
  font-size: 10px;
  letter-spacing: 0;
  color: var(--sp-text-3);
}
.rail-section-title {
  font-size: 13px;
  font-weight: 600;
  color: var(--sp-text-1);
  display: flex; align-items: center; gap: 6px;
}
.rail-count {
  font-family: var(--sp-font-mono);
  font-size: 11px;
  color: var(--sp-text-3);
  background: var(--sp-bg-tint);
  border-radius: 3px;
  padding: 1px 6px;
}
.rail-list { display: flex; flex-direction: column; gap: 2px; }

/* Project buttons in the rail — they look like list rows but are real
 * gr.Buttons that load the trajectory in the Archive tab. */
.rail-project-btn button,
button.rail-project-btn {
  text-align: left !important;
  width: 100% !important;
  justify-content: flex-start !important;
  padding: 6px 10px !important;
  background: transparent !important;
  border: 1px solid transparent !important;
  border-radius: 4px !important;
  font-size: 12px !important;
  color: var(--sp-text-2) !important;
  font-weight: 400 !important;
  white-space: nowrap !important;
  overflow: hidden !important;
  text-overflow: ellipsis !important;
  display: block !important;
  margin: 0 !important;
}
.rail-project-btn button:hover,
button.rail-project-btn:hover {
  background: var(--sp-bg-tint) !important;
  color: var(--sp-text-1) !important;
  border-color: var(--sp-border) !important;
}
.rail-empty {
  font-size: 12px;
  color: var(--sp-text-2);
  padding: 8px;
  background: var(--sp-bg);
  border: 1px dashed var(--sp-border);
  border-radius: var(--sp-radius);
  line-height: 1.55;
}
.rail-empty .muted { color: var(--sp-text-3); display: inline-block; margin-top: 2px; }
.rail-more {
  font-size: 11px;
  color: var(--sp-text-3);
  padding: 4px 8px;
  font-style: italic;
}
.rail-paper {
  display: flex; align-items: center; gap: 10px;
  padding: 8px;
  border: 1px solid var(--sp-border);
  border-radius: var(--sp-radius);
  background: var(--sp-bg);
}
.rail-paper + .rail-paper { margin-top: 6px; }
.rail-ext {
  font-family: var(--sp-font-mono);
  font-size: 10px;
  letter-spacing: 0;
  color: var(--sp-text-2);
  background: var(--sp-bg-tint);
  padding: 3px 6px;
  border-radius: 3px;
  border: 1px solid var(--sp-border);
}
.rail-paper-name {
  font-size: 12px; color: var(--sp-text-1);
  white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
  max-width: 150px;
}
.rail-paper-meta { font-size: 11px; color: var(--sp-text-3); margin-top: 2px; }
.rail-paper-source {
  font-size: 11px; color: var(--sp-text-3);
  margin-top: 8px; padding-top: 6px;
  border-top: 1px dashed var(--sp-border);
}
.rail-paper-source code {
  font-family: var(--sp-font-mono);
  color: var(--sp-text-2);
}

/* Task Queue card --------------------------------------------------------- */
.queue-card {
  border: 1px solid var(--sp-border);
  border-radius: var(--sp-radius);
  padding: 10px 12px;
  background: var(--sp-bg);
}
.queue-row { display: flex; align-items: center; gap: 8px; }
.queue-led {
  width: 8px; height: 8px;
  border-radius: 50%;
  background: var(--sp-text-3);
}
.queue-running .queue-led { background: var(--sp-accent); }
.queue-done .queue-led { background: var(--sp-success); }
.queue-error .queue-led { background: var(--sp-danger); }
.queue-label {
  font-size: 12px; font-weight: 600; color: var(--sp-text-1);
  letter-spacing: 0;
}
.queue-hint {
  font-size: 11px; color: var(--sp-text-3); margin-top: 4px;
}
.queue-sandbox {
  font-size: 11px; color: var(--sp-text-3);
  margin-top: 6px;
  padding-top: 6px;
  border-top: 1px dashed var(--sp-border);
}
.queue-sandbox code { font-family: var(--sp-font-mono); color: var(--sp-text-2); }

/* Main workspace heading & status bar ------------------------------------- */
.workspace-heading {
  display: flex; align-items: baseline; gap: 12px;
  padding: 4px 0 12px;
  border-bottom: 1px solid var(--sp-border);
  margin-bottom: 12px;
}
.workspace-heading h2 {
  font-family: var(--sp-font-serif);
  font-size: 22px;
  font-weight: 600;
  color: var(--sp-text-1);
  margin: 0;
  letter-spacing: 0;
}
.workspace-heading p {
  font-size: 12px;
  color: var(--sp-text-3);
  margin: 0;
}

/* Override the existing chat-status-bar to match the new monochrome look */
.chat-status-bar {
  display: flex;
  align-items: center;
  gap: 14px;
  padding: 8px 12px;
  border: 1px solid var(--sp-border) !important;
  background: var(--sp-bg-soft) !important;
  border-radius: var(--sp-radius);
  font-size: 12px;
  color: var(--sp-text-2);
  margin-bottom: 12px;
}
.chat-status-bar code {
  font-family: var(--sp-font-mono);
  font-size: 11px;
  color: var(--sp-text-1);
  background: transparent;
  padding: 0;
}

/* Status badge override (subdued, no saturated colour) */
.status-badge, .soft-badge {
  display: inline-flex;
  align-items: center;
  gap: 5px;
  font-size: 11px !important;
  font-weight: 500 !important;
  padding: 2px 8px !important;
  border-radius: 999px !important;
  border: 1px solid var(--sp-border) !important;
  background: var(--sp-bg) !important;
  color: var(--sp-text-2) !important;
  letter-spacing: 0;
}
.status-badge.running, .soft-badge.enabled {
  color: var(--sp-accent) !important;
  border-color: rgba(29, 78, 216, 0.18) !important;
  background: rgba(29, 78, 216, 0.05) !important;
}
.status-badge.error, .status-badge.failed {
  color: var(--sp-danger) !important;
  border-color: rgba(153, 27, 27, 0.18) !important;
  background: rgba(153, 27, 27, 0.04) !important;
}
.status-badge.done, .status-badge.success { color: var(--sp-success) !important; }

/* Chatbot placeholder (the "等待研究任务启动…" hint that Gradio renders
 * when the bot has no messages yet). Default Gradio styles it with strong
 * contrast and centers it, which reads as a giant ugly box. We fade it
 * down to a subtle hint and trim its size. ---------------------------- */
.chat-pane .placeholder,
.chat-pane [class*="placeholder"],
.chat-pane .empty,
.chat-pane .bubble-wrap > .placeholder {
  color: var(--sp-text-3) !important;
  font-size: 12.5px !important;
  font-weight: 400 !important;
  font-style: normal !important;
  letter-spacing: 0 !important;
  background: transparent !important;
  border: none !important;
  box-shadow: none !important;
  padding: 12px 16px !important;
  margin: 8px !important;
  text-align: left !important;
  max-width: 520px !important;
  line-height: 1.55 !important;
}

/* Chat panel — researcher chain-of-thought ------------------------------- */
/* Keep styling minimal so we don't fight Gradio's chatbot internals. Only  */
/* touch the outer wrapper + a few content-typography rules.                 */
.chat-pane {
  border: 1px solid var(--sp-border) !important;
  border-radius: var(--sp-radius-lg) !important;
  background: var(--sp-bg) !important;
  box-shadow: var(--sp-shadow);
  overflow: hidden;
}
.chat-pane h1, .chat-pane h2, .chat-pane h3, .chat-pane h4 {
  font-family: var(--sp-font-serif) !important;
  color: var(--sp-text-1) !important;
  letter-spacing: 0;
}
.chat-pane h2 { font-size: 16px !important; }
.chat-pane h3 { font-size: 14px !important; }
.chat-pane code, .chat-pane pre {
  font-family: var(--sp-font-mono) !important;
  font-size: 12px !important;
}
.chat-pane pre {
  background: var(--sp-bg-tint) !important;
  border: 1px solid var(--sp-border) !important;
  border-radius: 4px !important;
  padding: 10px 12px !important;
  white-space: pre-wrap !important;
}
.chat-pane blockquote {
  border-left: 2px solid var(--sp-border-strong) !important;
  padding-left: 10px !important;
  color: var(--sp-text-2) !important;
}

/* Citation-style cards inside the chat (the existing step renderer uses
 * .step-card / .citation-card classes — make them feel like citations). */
.citation-card, .step-card {
  border: 1px solid var(--sp-border) !important;
  background: var(--sp-bg) !important;
  border-radius: var(--sp-radius) !important;
  padding: 12px 14px !important;
  box-shadow: var(--sp-shadow);
}
.citation-card .title, .step-card .title {
  font-family: var(--sp-font-serif) !important;
  font-size: 14px !important;
  color: var(--sp-text-1) !important;
}
.citation-card .meta, .step-card .meta {
  font-size: 11px !important;
  color: var(--sp-text-3) !important;
  letter-spacing: 0;
}

/* Command input ----------------------------------------------------------- */
.sp-command-shell {
  position: sticky;
  bottom: 0;
  background: var(--sp-bg);
  padding-top: 12px;
  margin-top: 12px;
  border-top: 1px solid var(--sp-border);
}
.sp-command-row {
  display: flex; align-items: stretch; gap: 8px;
  border: 1px solid var(--sp-border-strong);
  border-radius: var(--sp-radius);
  background: var(--sp-bg);
  padding: 6px 6px 6px 10px;
  box-shadow: var(--sp-shadow);
}
.sp-command-row:focus-within {
  border-color: var(--sp-text-1);
}
.sp-command-row textarea {
  border: none !important;
  background: transparent !important;
  resize: none !important;
  font-size: 13px !important;
  color: var(--sp-text-1) !important;
  font-family: var(--sp-font-sans) !important;
  padding: 8px 4px !important;
  box-shadow: none !important;
}
.sp-command-row textarea::placeholder { color: var(--sp-text-3); }
.sp-command-row .input-icons {
  display: flex; align-items: center; gap: 4px;
}
.sp-command-actions { display: flex; align-items: center; gap: 6px; }

button.gr-button, .gr-button, button {
  border-radius: var(--sp-radius) !important;
  font-family: var(--sp-font-sans) !important;
  font-size: 12px !important;
  font-weight: 500 !important;
  letter-spacing: 0 !important;
  border: 1px solid var(--sp-border-strong) !important;
  background: var(--sp-bg) !important;
  color: var(--sp-text-1) !important;
  padding: 6px 12px !important;
  box-shadow: none !important;
  transition: background 120ms linear, border-color 120ms linear;
}
button:hover, .gr-button:hover { background: var(--sp-bg-tint) !important; }
button.primary, .gr-button-primary, button[variant="primary"] {
  background: var(--sp-text-1) !important;
  color: #fff !important;
  border-color: var(--sp-text-1) !important;
}
button.primary:hover, .gr-button-primary:hover { background: #1f1f1f !important; }
button.stop, .gr-button-stop, button[variant="stop"] {
  background: var(--sp-bg) !important;
  color: var(--sp-danger) !important;
  border-color: rgba(153, 27, 27, 0.22) !important;
}
button.icon-only {
  padding: 4px 8px !important;
  min-width: 32px !important;
}

/* Drag-drop attach zone (gr.Files component). It sits above the textarea
 * and acts as both a click-to-open file picker AND a real HTML5 drop
 * target. We render uploaded files as small chips inline, so the user
 * doesn't have to expand a separate "Attached files" accordion to see
 * what they uploaded. ------------------------------------------------ */
.sp-attach-zone,
.sp-attach-zone > div,
.sp-attach-zone > .form,
.sp-attach-zone .gr-block {
  border: 1px dashed var(--sp-border-strong) !important;
  border-radius: var(--sp-radius-lg) !important;
  background: var(--sp-bg-soft) !important;
  box-shadow: none !important;
  margin-bottom: 8px !important;
  transition: border-color 120ms linear, background 120ms linear;
}
.sp-attach-zone:hover,
.sp-attach-zone:focus-within {
  border-color: var(--sp-text-2) !important;
  background: var(--sp-bg-tint) !important;
}
/* Hide the giant default Gradio file-row separator + label decoration so
 * the dropzone reads as a single calm strip. */
.sp-attach-zone label > span:first-child,
.sp-attach-zone .label-wrap {
  font-size: 11.5px !important;
  font-weight: 500 !important;
  color: var(--sp-text-3) !important;
  letter-spacing: 0 !important;
}
.sp-attach-zone .upload-container,
.sp-attach-zone [data-testid="file-upload"] {
  background: transparent !important;
  border: none !important;
  padding: 6px 10px !important;
}
/* File chips: each uploaded file gets a small inline pill */
.sp-attach-zone .file-preview,
.sp-attach-zone [data-testid="file"] {
  display: inline-flex !important;
  align-items: center !important;
  gap: 6px !important;
  padding: 4px 10px !important;
  margin: 2px 4px 2px 0 !important;
  background: var(--sp-bg) !important;
  border: 1px solid var(--sp-border) !important;
  border-radius: 999px !important;
  font-size: 11.5px !important;
  color: var(--sp-text-1) !important;
  font-family: var(--sp-font-mono) !important;
}
.sp-attach-zone .file-preview button,
.sp-attach-zone [data-testid="file"] button {
  font-size: 10px !important;
  padding: 0 4px !important;
  background: transparent !important;
  border: none !important;
  color: var(--sp-text-3) !important;
}

/* Mode toggle (radio) ----------------------------------------------------- */
.run-mode-row .wrap { display: flex !important; gap: 4px !important; }
.run-mode-row label {
  border: 1px solid var(--sp-border) !important;
  border-radius: var(--sp-radius) !important;
  padding: 4px 10px !important;
  font-size: 12px !important;
  color: var(--sp-text-2) !important;
  background: var(--sp-bg) !important;
  cursor: pointer;
}
.run-mode-row input:checked + span,
.run-mode-row label:has(input:checked) {
  border-color: var(--sp-text-1) !important;
  color: var(--sp-text-1) !important;
  background: var(--sp-bg-tint) !important;
}

/* Pipeline / evidence override ------------------------------------------- */
.pipeline-shell, .evidence-shell {
  border: 1px solid var(--sp-border) !important;
  background: var(--sp-bg) !important;
  padding: 14px 16px !important;
  border-radius: var(--sp-radius-lg) !important;
  box-shadow: var(--sp-shadow);
}
.pipeline-shell + .pipeline-shell,
.pipeline-shell + .evidence-shell,
.evidence-shell + .evidence-shell { margin-top: 16px; }
.panel-title-row {
  display: flex; align-items: center; justify-content: space-between;
  margin-bottom: 6px;
}
.panel-title-row h3 {
  font-family: var(--sp-font-serif);
  font-size: 14px !important;
  font-weight: 600 !important;
  color: var(--sp-text-1) !important;
  margin: 0 !important;
  letter-spacing: 0;
}
.panel-subtitle {
  font-size: 11px !important;
  color: var(--sp-text-3) !important;
  margin: 0 0 10px !important;
}
.pipeline-progress {
  height: 2px;
  background: var(--sp-bg-tint);
  border-radius: 2px;
  overflow: hidden;
  margin-bottom: 12px;
}
.pipeline-progress span {
  display: block; height: 100%;
  background: var(--sp-text-1);
  transition: width 240ms ease;
}
.pipeline-list, .evidence-list { display: flex; flex-direction: column; gap: 6px; }
.pipeline-item {
  display: flex; gap: 8px;
  padding: 8px 10px;
  border: 1px solid var(--sp-border);
  border-radius: var(--sp-radius);
  background: var(--sp-bg);
}
.pipeline-dot {
  width: 6px; height: 6px;
  margin-top: 6px;
  border-radius: 50%;
  background: var(--sp-text-3);
  flex-shrink: 0;
}
.pipeline-item.running .pipeline-dot { background: var(--sp-accent); }
.pipeline-item.done .pipeline-dot,
.pipeline-item.success .pipeline-dot { background: var(--sp-success); }
.pipeline-item.failed .pipeline-dot { background: var(--sp-danger); }
.pipeline-step-head {
  display: flex; align-items: center; justify-content: space-between; gap: 8px;
}
.pipeline-step-name {
  font-size: 12px; font-weight: 600; color: var(--sp-text-1);
}
.pipeline-step-meta {
  font-size: 11px; color: var(--sp-text-3); margin-top: 2px; line-height: 1.45;
}

.evidence-section { margin-top: 12px; }
.evidence-section h4 {
  font-family: var(--sp-font-mono);
  font-size: 10px !important;
  letter-spacing: 0;
  color: var(--sp-text-3) !important;
  text-transform: uppercase;
  margin: 0 0 6px !important;
}
.evidence-item {
  border: 1px solid var(--sp-border);
  border-radius: var(--sp-radius);
  padding: 8px 10px;
  background: var(--sp-bg);
}
.evidence-title {
  font-family: var(--sp-font-serif);
  font-size: 12px;
  color: var(--sp-text-1);
  font-weight: 600;
}
.evidence-meta {
  font-size: 11px;
  color: var(--sp-text-3);
  margin-top: 2px;
}
.tool-row {
  display: flex; gap: 8px; align-items: flex-start;
  padding: 6px 10px;
  border: 1px solid var(--sp-border);
  border-radius: var(--sp-radius);
  background: var(--sp-bg);
}
.tool-row.failed { border-color: rgba(153, 27, 27, 0.18); }
.tool-dot {
  width: 6px; height: 6px;
  margin-top: 6px;
  border-radius: 50%;
  background: var(--sp-success);
  flex-shrink: 0;
}
.tool-row.failed .tool-dot { background: var(--sp-danger); }
.tool-name {
  font-family: var(--sp-font-mono);
  font-size: 12px; color: var(--sp-text-1);
}
.tool-meta {
  font-size: 11px; color: var(--sp-text-3); margin-top: 2px;
}
.uncertainty-box {
  border: 1px dashed var(--sp-border-strong);
  border-radius: var(--sp-radius);
  padding: 10px 12px;
  background: var(--sp-bg-soft);
  font-size: 12px;
  color: var(--sp-text-2);
  line-height: 1.55;
}
.empty-state {
  font-size: 12px;
  color: var(--sp-text-3);
  padding: 8px;
  background: var(--sp-bg);
  border: 1px dashed var(--sp-border);
  border-radius: var(--sp-radius);
  text-align: center;
}

/* Document preview ------------------------------------------------------- */
.doc-shell {
  border: 1px solid var(--sp-border);
  border-radius: var(--sp-radius-lg);
  background: var(--sp-bg);
  padding: 14px 16px;
  box-shadow: var(--sp-shadow);
}
.doc-shell-pdf {
  /* When the live PDF is mounted we want the iframe to fill, so the
   * outer shell drops vertical padding and lets the frame breathe. */
  padding: 10px 10px 4px 10px;
}
.doc-meta {
  display: flex; align-items: baseline; gap: 12px;
  font-size: 11px; color: var(--sp-text-3);
  padding: 0 4px 8px;
  border-bottom: 1px dashed var(--sp-border);
  margin-bottom: 10px;
}
.doc-meta-label {
  font-family: var(--sp-font-mono);
  text-transform: uppercase; letter-spacing: 0;
}
.doc-meta-name {
  font-family: var(--sp-font-mono);
  color: var(--sp-text-2);
  flex: 1 1 auto;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.doc-meta-size {
  font-family: var(--sp-font-mono);
  color: var(--sp-text-3);
}
.doc-pdf-frame {
  width: 100%;
  /* Sized to most reasonable monitors; on really tall screens the
   * browser viewer will still show pagination. */
  height: 78vh;
  min-height: 520px;
  border: 1px solid var(--sp-border);
  border-radius: var(--sp-radius);
  background: var(--sp-bg-tint);
}
.doc-frame {
  background: var(--sp-bg);
  border: 1px solid var(--sp-border);
  border-radius: var(--sp-radius);
  padding: 24px 28px;
  font-family: var(--sp-font-serif);
  color: var(--sp-text-1);
  line-height: 1.7;
  max-height: 480px;
  overflow-y: auto;
  position: relative;
}
.doc-frame.doc-frame-empty {
  text-align: center;
  padding: 36px 24px;
}
.doc-watermark {
  font-family: var(--sp-font-mono);
  font-size: 10px;
  letter-spacing: 0;
  color: var(--sp-text-3);
  margin-bottom: 12px;
}
.doc-title {
  font-family: var(--sp-font-serif);
  font-size: 16px;
  color: var(--sp-text-1);
  margin-bottom: 18px;
}
.doc-line {
  height: 6px;
  background: var(--sp-bg-tint);
  border-radius: 2px;
  margin: 6px auto;
}
.doc-line.w90 { width: 90%; } .doc-line.w70 { width: 70%; }
.doc-line.w95 { width: 95%; } .doc-line.w60 { width: 60%; }
.doc-line.w80 { width: 80%; } .doc-line.w85 { width: 85%; }
.doc-line.w55 { width: 55%; }
.doc-empty-note {
  margin-top: 16px;
  font-family: var(--sp-font-sans);
  font-size: 11px;
  color: var(--sp-text-3);
}
.doc-empty-note code {
  font-family: var(--sp-font-mono);
  background: var(--sp-bg-tint);
  padding: 1px 4px;
  border-radius: 3px;
}
.doc-h {
  font-family: var(--sp-font-serif);
  margin: 14px 0 6px !important;
}
.doc-h:first-child { margin-top: 0 !important; }
.doc-p {
  margin: 0 0 8px !important;
  font-family: var(--sp-font-serif) !important;
  font-size: 13px;
  color: var(--sp-text-1);
}
.doc-p.muted { color: var(--sp-text-3); }

/* Settings tab ----------------------------------------------------------- */
/* Scope the card chrome to our own ``surface-card`` class only. The
 * ``.gr-block`` / ``.gr-form`` / ``.form`` selectors used to live here, but
 * they were too greedy — they nested a border + background around every
 * internal Gradio block (including the chatbot's message viewport), which
 * collapsed its scrollable area and made the chat appear empty. */
.surface-card {
  border-radius: var(--sp-radius-lg) !important;
  border: 1px solid var(--sp-border) !important;
  background: var(--sp-bg) !important;
  box-shadow: var(--sp-shadow) !important;
  padding: 16px 18px !important;
  margin-bottom: 12px;
}
.card-heading h3 {
  font-family: var(--sp-font-serif);
  font-size: 14px !important;
  font-weight: 600 !important;
  color: var(--sp-text-1) !important;
  margin: 0 0 4px !important;
}
.card-heading p {
  font-size: 12px !important;
  color: var(--sp-text-3) !important;
  margin: 0 0 12px !important;
}
input[type="text"], input[type="password"], input[type="number"], textarea, select {
  border: 1px solid var(--sp-border) !important;
  border-radius: var(--sp-radius) !important;
  background: var(--sp-bg) !important;
  color: var(--sp-text-1) !important;
  font-size: 13px !important;
  font-family: var(--sp-font-sans) !important;
}
input:focus, textarea:focus, select:focus {
  border-color: var(--sp-text-1) !important;
  box-shadow: none !important;
  outline: none !important;
}
label, .gr-form > label, .label {
  font-size: 12px !important;
  font-weight: 500 !important;
  color: var(--sp-text-2) !important;
}

/* Settings summary card override (used on the workspace rail) */
.settings-summary-card {
  border: 1px solid var(--sp-border);
  border-radius: var(--sp-radius-lg);
  padding: 14px;
  background: var(--sp-bg);
}
.summary-card-header {
  display: flex; align-items: flex-start; justify-content: space-between;
  border-bottom: 1px dashed var(--sp-border);
  padding-bottom: 8px; margin-bottom: 10px;
}
.summary-eyebrow {
  font-family: var(--sp-font-mono);
  font-size: 10px; letter-spacing: 0;
  color: var(--sp-text-3);
}
.summary-card-header h3 {
  font-family: var(--sp-font-serif);
  font-size: 13px !important;
  margin: 2px 0 0 !important;
  color: var(--sp-text-1) !important;
}
.source-badge {
  font-family: var(--sp-font-mono);
  font-size: 10px;
  border: 1px solid var(--sp-border);
  border-radius: 999px;
  padding: 2px 8px;
  color: var(--sp-text-2);
  background: var(--sp-bg-soft);
}
.summary-section { margin-top: 10px; }
.summary-section-head {
  display: flex; align-items: center; justify-content: space-between;
  margin-bottom: 4px;
}
.summary-section-title {
  font-size: 11px; font-weight: 600;
  color: var(--sp-text-2);
  text-transform: uppercase; letter-spacing: 0;
}
.summary-dl { display: flex; flex-direction: column; gap: 4px; margin: 0; }
.summary-field {
  display: flex; gap: 6px; align-items: baseline; font-size: 11px;
}
.summary-field dt {
  width: 64px;
  flex-shrink: 0;
  color: var(--sp-text-3);
  font-weight: 500;
}
.summary-field dd { margin: 0; flex: 1; min-width: 0; }
.truncate-value {
  display: inline-block; max-width: 100%;
  white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
  color: var(--sp-text-1);
}
.code-value { font-family: var(--sp-font-mono); }
.copy-btn {
  font-family: var(--sp-font-mono);
  font-size: 10px !important;
  border: 1px solid var(--sp-border) !important;
  background: var(--sp-bg-soft) !important;
  color: var(--sp-text-2) !important;
  padding: 1px 6px !important;
  margin-left: 6px;
  border-radius: 3px !important;
  cursor: pointer;
}
.copy-btn.copied { color: var(--sp-success) !important; border-color: var(--sp-success) !important; }
.summary-param-row {
  display: flex; gap: 16px; margin-top: 8px;
  padding-top: 8px;
  border-top: 1px dashed var(--sp-border);
}
.mini-param {
  display: inline-flex; flex-direction: column; gap: 2px;
  font-size: 11px; color: var(--sp-text-1);
}
.mini-param b {
  font-size: 10px; color: var(--sp-text-3); font-weight: 500;
  letter-spacing: 0; text-transform: uppercase;
}

/* Artifact rows --------------------------------------------------------- */
.artifact-meta-card {
  display: flex; align-items: center; gap: 10px;
  border: 1px solid var(--sp-border);
  border-radius: var(--sp-radius);
  background: var(--sp-bg);
  padding: 8px 10px;
}
.artifact-icon {
  font-family: var(--sp-font-mono);
  font-size: 10px;
  letter-spacing: 0;
  color: var(--sp-text-2);
  background: var(--sp-bg-tint);
  padding: 3px 6px;
  border-radius: 3px;
  border: 1px solid var(--sp-border);
}
.artifact-name-wrap { flex: 1; min-width: 0; }
.artifact-file-name {
  font-size: 12px; color: var(--sp-text-1);
  font-family: var(--sp-font-mono);
  overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
}
.artifact-subrow .type-badge {
  font-size: 10px;
  color: var(--sp-text-3);
  text-transform: uppercase;
  letter-spacing: 0;
}
.artifact-size {
  font-family: var(--sp-font-mono);
  font-size: 11px;
  color: var(--sp-text-3);
}

/* History / archive table ----------------------------------------------- */
.history-shell { padding: 16px 18px; }
.history-header {
  display: flex; align-items: flex-start; justify-content: space-between;
  border-bottom: 1px solid var(--sp-border);
  padding-bottom: 10px; margin-bottom: 14px;
}
.history-header h3 {
  font-family: var(--sp-font-serif);
  font-size: 16px !important;
  margin: 0 !important;
}
.history-header p {
  font-size: 12px !important;
  color: var(--sp-text-3) !important;
  margin: 4px 0 0 !important;
}
.history-toolbar { gap: 8px; align-items: end; }
.history-view {
  font-family: var(--sp-font-serif);
  font-size: 13px;
  line-height: 1.7;
  color: var(--sp-text-1);
  padding: 0 4px;
}

/* Footer note ------------------------------------------------------------ */
.footer-note {
  text-align: center;
  font-size: 11px !important;
  color: var(--sp-text-3) !important;
  padding: 16px 24px !important;
  border-top: 1px solid var(--sp-border);
  margin-top: 16px;
}

/* Hide the default Gradio container labels in the rail */
.sp-rail-left .label-wrap, .sp-rail-right .label-wrap { display: none !important; }
"""


# ---------------------------------------------------------------------------
# Wrappers — extend a few backend handlers so the new sidebar widgets stay
# in sync with run state. They DO NOT touch backend behaviour, only fan out
# extra UI-only outputs that derive from already-published runner state.
# ---------------------------------------------------------------------------
def _live_sidebar_outputs():
    """Snapshot the runner and rebuild sidebar widgets that depend on it.

    Returns a flat tuple matching the order:
      queue_html, papers_html,
      projects_header_html, projects_empty_html,
      *[N button updates], *[N task_id state values],
      projects_more_html
    """
    runner = legacy_ui._runner
    with runner._lock:
        status = runner.status
        steps = len(runner.steps_md)
        max_steps = runner.max_steps
        sandbox = runner.sandbox_dir.name if runner.sandbox_dir else ""

    header_html, empty_html, btn_updates, state_values, more_html = _projects_rail_state()
    return (
        _task_queue_html(status, steps, max_steps, sandbox),
        _papers_list_html(),
        header_html,
        empty_html,
        *btn_updates,
        *state_values,
        more_html,
    )


def chat_main_stream_with_rails(*args, **kwargs):
    """Wrap the legacy generator so each yield refreshes the sidebar widgets."""
    for tup in chat_main_stream(*args, **kwargs):
        yield (*tup, *_live_sidebar_outputs())


def chat_clear_history_with_rails():
    return (*chat_clear_history(), *_live_sidebar_outputs())


def refresh_all_rails():
    """Manual refresh handler for the sidebar 'refresh' button."""
    return _live_sidebar_outputs()


def open_project_in_archive(task_id: str):
    """Click handler for a rail project button.

    Loads the chosen trajectory in the Archive tab via the existing legacy
    handlers and switches the tab to ``archive``. ``task_id`` is sourced
    from the per-button hidden state populated by ``_projects_rail_state``.
    """
    if not task_id:
        return (
            gr.update(),                          # tabs (no-op)
            gr.update(),                          # traj_select
            "*该项目还没有可读取的内容。*",        # traj_view
            "{}",                                 # traj_json_view
        )
    return (
        gr.update(selected="archive"),
        gr.update(value=task_id),
        view_trajectory(task_id),
        export_traj_json(task_id),
    )


def refresh_paper_with_doc():
    """Refresh paper artefacts AND the right-rail document preview."""
    artefact_tuple = load_latest_paper_artifacts()
    _, report_path, _, _ = _find_paper_artifacts()
    return (*artefact_tuple, _document_preview_html(report_path))


def convert_pdf_with_doc(*args, **kwargs):
    artefact_tuple = convert_report_to_pdf_artifacts(*args, **kwargs)
    _, report_path, _, _ = _find_paper_artifacts()
    return (*artefact_tuple, _document_preview_html(report_path))


# ---------------------------------------------------------------------------
# Build the Gradio app.
# ---------------------------------------------------------------------------
def create_ui() -> gr.Blocks:
    theme = gr.themes.Base(
        primary_hue="slate",
        secondary_hue="slate",
        neutral_hue="zinc",
        font=[
            gr.themes.GoogleFont("Inter"),
            gr.themes.GoogleFont("Source Serif 4"),
            "JetBrains Mono",
            "PingFang SC",
            "Microsoft YaHei",
            "system-ui",
            "sans-serif",
        ],
    ).set(
        body_background_fill="#ffffff",
        body_background_fill_dark="#ffffff",
        body_text_color="#0a0a0a",
        body_text_color_dark="#0a0a0a",
        background_fill_primary="#ffffff",
        background_fill_secondary="#fafafa",
        block_background_fill="#ffffff",
        block_label_background_fill="#ffffff",
        block_label_text_color="#525252",
        block_title_text_color="#0a0a0a",
        block_border_color="#ececec",
        input_background_fill="#ffffff",
        input_border_color="#d4d4d8",
        button_primary_background_fill="#0a0a0a",
        button_primary_background_fill_hover="#1f1f1f",
        button_primary_text_color="#ffffff",
        button_secondary_background_fill="#ffffff",
        button_secondary_background_fill_hover="#f4f4f5",
        button_secondary_text_color="#0a0a0a",
    )

    initial_settings, initial_source = load_ui_settings(DEFAULT_SETTINGS_PATH)
    initial_status_md = f"已载入{_settings_source_label(initial_source)}设置。"
    initial_summary_html = _build_settings_summary_html(initial_settings, initial_source)

    agent_preset_choices = list(MODEL_PRESETS.keys()) + [CUSTOM_PRESET_LABEL]
    reviewer_preset_choices = list(REVIEWER_PRESETS.keys()) + [CUSTOM_PRESET_LABEL]

    initial_traj_choices = list_trajectory_choices() if SHOW_HISTORY_TAB else []

    initial_paper_dir, initial_report_path, initial_bundle_path, _initial_pdf_path = _find_paper_artifacts()
    # Do not preload an existing PDF into the Export/Manuscript panels. A
    # previous run's PDF made the current run look exported before the user
    # pressed "Export PDF". Existing papers are still visible in Saved Papers;
    # the live Manuscript preview is populated by Export/Refresh actions.
    initial_pdf_path = None

    initial_pipeline_html = _build_research_pipeline_html(
        [], "idle", initial_settings.max_steps, ""
    )
    initial_evidence_html = _build_evidence_trace_html([], "idle")

    # Try to load PaperForge templates without requiring the dependency.
    try:
        if str(PAPER_FORGE_ROOT) not in sys.path:
            sys.path.insert(0, str(PAPER_FORGE_ROOT))
        from paper_forge import list_templates as _list_templates
        _template_choices = [(t.label, t.key) for t in _list_templates()]
        _default_template_key = _template_choices[0][1]
    except Exception:
        _template_choices = [("通用 article", "general_article")]
        _default_template_key = "general_article"

    # Gradio's Textbox `submit` event only fires on Ctrl+Enter for multi-line
    # boxes, so we wire the textarea's keydown ourselves: plain Enter clicks
    # the Send button, Shift+Enter falls through to the textarea's native
    # newline. We wrap in a MutationObserver because Gradio mounts components
    # asynchronously and the textarea may not exist when this script first runs.
    enter_send_js = """
() => {
  const wire = () => {
    const wrap = document.getElementById("chat-input-box");
    if (!wrap) return false;
    const ta = wrap.querySelector("textarea");
    if (!ta) return false;
    if (ta.dataset.enterSendWired === "1") return true;
    ta.dataset.enterSendWired = "1";
    ta.addEventListener("keydown", (e) => {
      if (e.key === "Enter" && !e.shiftKey && !e.isComposing) {
        e.preventDefault();
        const wrapBtn = document.getElementById("chat-send-btn");
        const btn = (wrapBtn && wrapBtn.tagName === "BUTTON")
                    ? wrapBtn
                    : (wrapBtn && wrapBtn.querySelector("button"));
        if (btn) btn.click();
      }
    });
    return true;
  };
  if (!wire()) {
    const obs = new MutationObserver(() => { if (wire()) obs.disconnect(); });
    obs.observe(document.body, { childList: true, subtree: true });
  }
}
"""

    with gr.Blocks(
        title="LabForge · Research Studio",
        theme=theme,
        css=CUSTOM_CSS,
        analytics_enabled=False,
        js=enter_send_js,
    ) as app:

        # ------------------------------------------------------------------
        # TOP BAR
        # ------------------------------------------------------------------
        gr.HTML(
            '<div class="sp-topbar">'
            f'{_brand_block_html()}'
            '<div class="sp-topbar-right">'
            f'<span class="sp-topbar-meta">model · <code>{initial_settings.agent_model}</code></span>'
            f'<span class="sp-topbar-meta">step budget · <code>{format_step_budget_label(initial_settings.max_steps)}</code></span>'
            '</div>'
            '</div>'
        )

        with gr.Tabs(elem_classes=["sp-tabs"]) as tabs_root:

            # ==================================================================
            # TAB 1 — RESEARCH WORKSPACE
            # ==================================================================
            with gr.Tab("Workspace", id="workspace"):
                with gr.Row(elem_classes=["sp-workspace"], equal_height=False):
                    # ------ LEFT SIDEBAR -------------------------------------
                    with gr.Column(scale=2, min_width=240, elem_classes=["sp-rail-left"]):
                        # Projects section: header + empty-state + N buttons + more-footer.
                        # Each button is a real gr.Button so a click can route to the
                        # Archive tab via existing ``view_trajectory`` / ``export_traj_json``.
                        _hdr0, _empty0, _btn_updates0, _states0, _more0 = _projects_rail_state()
                        projects_header_html = gr.HTML(_hdr0)
                        projects_empty_html = gr.HTML(_empty0)
                        project_btns: list[gr.Button] = []
                        project_states: list[gr.State] = []
                        for i in range(RAIL_PROJECT_SLOTS):
                            init_value = _btn_updates0[i].get("value", "") if isinstance(_btn_updates0[i], dict) else ""
                            init_visible = _btn_updates0[i].get("visible", False) if isinstance(_btn_updates0[i], dict) else False
                            btn = gr.Button(
                                value=init_value or " ",
                                visible=init_visible,
                                size="sm",
                                elem_classes=["rail-project-btn"],
                            )
                            project_btns.append(btn)
                            project_states.append(gr.State(_states0[i]))
                        projects_more_html = gr.HTML(_more0)

                        papers_html = gr.HTML(_papers_list_html())
                        queue_html = gr.HTML(_task_queue_html("idle", 0, initial_settings.max_steps, ""))
                        refresh_rail_btn = gr.Button("Refresh Sidebar", size="sm")

                    # ------ MAIN WORKSPACE -----------------------------------
                    with gr.Column(scale=8, min_width=560, elem_classes=["sp-main"]):
                        gr.HTML(
                            '<div class="workspace-heading">'
                            '<h2>Current Research Run</h2>'
                            '<p>Chain-of-thought, citations and final manuscript appear below.</p>'
                            '</div>'
                        )

                        chat_status_html = gr.HTML(
                            _build_chat_status("idle", initial_settings.agent_model, 0, initial_settings.max_steps)
                        )

                        chat_panel = gr.Chatbot(
                            label=None,
                            show_label=False,
                            elem_classes=["chat-pane"],
                            height=520,
                            placeholder=(
                                "在下方输入研究主题或上传论文后回车，"
                                "Agent 的思考、引用与结论会按时间顺序出现在这里。"
                            ),
                            avatar_images=(None, None),
                            render_markdown=True,
                            type="messages",
                            group_consecutive_messages=False,
                        )

                        # Command input — drag-drop attach zone above, then
                        # textarea + actions row. ``gr.Files`` doubles as a
                        # native drop target AND shows uploaded filenames as
                        # chips inline — no hidden accordion to click into.
                        with gr.Group(elem_classes=["sp-command-shell"]):
                            attach_btn = gr.Files(
                                label="拖拽或点击此处上传 PDF / 图片 / Markdown / 文本（PDF·图片自动走 PaddleOCR-VL）",
                                file_count="multiple",
                                file_types=[
                                    ".pdf",
                                    ".png", ".jpg", ".jpeg", ".bmp",
                                    ".tiff", ".tif", ".gif",
                                    ".csv", ".tsv",
                                    ".json", ".txt", ".md", ".zip",
                                ],
                                interactive=True,
                                height=120,
                                elem_classes=["sp-attach-zone"],
                            )

                            with gr.Row(elem_classes=["run-mode-row"]):
                                run_mode = gr.Radio(
                                    choices=[
                                        ("文献综述", "survey"),
                                        ("实验研究", "experiment"),
                                    ],
                                    value=initial_settings.run_mode,
                                    show_label=False,
                                    container=False,
                                    elem_id="run-mode-radio",
                                )

                            with gr.Row(elem_classes=["sp-command-row"]):
                                chat_input = gr.Textbox(
                                    placeholder=(
                                        "输入研究主题…  Enter 发送 · Shift+Enter 换行"
                                    ),
                                    lines=2,
                                    max_lines=8,
                                    show_label=False,
                                    container=False,
                                    scale=8,
                                    autofocus=True,
                                    elem_id="chat-input-box",
                                )
                                with gr.Column(scale=0, min_width=140, elem_classes=["sp-command-actions"]):
                                    with gr.Row():
                                        chat_clear_btn = gr.Button("Clear", size="sm")
                                        chat_send_btn = gr.Button(
                                            "Send →",
                                            size="sm",
                                            variant="primary",
                                            elem_id="chat-send-btn",
                                        )

                        # Hidden bridge between quick-inject and main_stream.
                        chat_stash = gr.State("")

                        # ``gr.Files`` already shows file chips inline, so the
                        # old "Attached files" accordion + status message are
                        # redundant — we skip them.
                        attach_status = gr.Markdown(visible=False)

                    # ------ RIGHT CONTEXT PANEL ------------------------------
                    with gr.Column(scale=4, min_width=320, elem_classes=["sp-rail-right"]):
                        with gr.Tabs():
                            with gr.Tab("Pipeline"):
                                pipeline_html = gr.HTML(initial_pipeline_html)
                                evidence_trace_html = gr.HTML(initial_evidence_html)

                            with gr.Tab("Manuscript"):
                                doc_preview_html = gr.HTML(_document_preview_html(None))

                            with gr.Tab("Export"):
                                gr.HTML(
                                    '<div class="card-heading">'
                                    '<h3>Paper Export</h3>'
                                    '<p>选择论文语言与 LaTeX 模板，导出 PaperForge 论文产物。</p>'
                                    '</div>'
                                )
                                paper_language = gr.Radio(
                                    choices=[
                                        ("Auto", "auto"),
                                        ("中文", "zh"),
                                        ("English", "en"),
                                    ],
                                    value="auto",
                                    label="Language",
                                    interactive=True,
                                )
                                paper_template = gr.Dropdown(
                                    choices=_template_choices,
                                    value=_default_template_key,
                                    label="Template",
                                    interactive=True,
                                    allow_custom_value=False,
                                )
                                with gr.Row():
                                    refresh_paper_btn = gr.Button("Refresh", size="sm")
                                    pdf_btn = gr.Button("Export PDF", size="sm", variant="primary")

                                with gr.Group(elem_classes=["surface-card"]):
                                    gr.HTML(
                                        '<div class="card-heading">'
                                        '<h3>Artifacts</h3>'
                                        '<p>Markdown report, PaperForge bundle, rendered PDF.</p>'
                                        '</div>'
                                    )
                                    with gr.Row(visible=initial_report_path is not None) as report_artifact_row:
                                        report_artifact = gr.HTML(
                                            _artifact_meta_html(initial_report_path, "Markdown", "MD"),
                                            visible=initial_report_path is not None,
                                        )
                                        report_download = gr.DownloadButton(
                                            label="Download",
                                            size="sm",
                                            value=str(initial_report_path) if initial_report_path else None,
                                            visible=initial_report_path is not None,
                                        )
                                    with gr.Row(visible=initial_bundle_path is not None) as bundle_artifact_row:
                                        bundle_artifact = gr.HTML(
                                            _artifact_meta_html(initial_bundle_path, "PaperForge bundle", "JSON"),
                                            visible=initial_bundle_path is not None,
                                        )
                                        bundle_download = gr.DownloadButton(
                                            label="Download",
                                            size="sm",
                                            value=str(initial_bundle_path) if initial_bundle_path else None,
                                            visible=initial_bundle_path is not None,
                                        )
                                    with gr.Row(visible=initial_pdf_path is not None) as pdf_artifact_row:
                                        pdf_artifact = gr.HTML(
                                            _artifact_meta_html(initial_pdf_path, "PDF", "PDF"),
                                            visible=initial_pdf_path is not None,
                                        )
                                        pdf_download = gr.DownloadButton(
                                            label="Download",
                                            size="sm",
                                            value=str(initial_pdf_path) if initial_pdf_path else None,
                                            visible=initial_pdf_path is not None,
                                        )

                                    # Full bundle (zip): LaTeX source +
                                    # experiment code + charts + logs +
                                    # literature cache + everything the
                                    # agent produced. Always-visible
                                    # button — handler builds the zip on
                                    # demand and returns the path; if
                                    # nothing exists yet it stays
                                    # disabled / hidden.
                                    gr.HTML(
                                        '<div class="card-heading bundle-heading">'
                                        '<h3>Full Run Bundle</h3>'
                                        '<p>一键打包 LaTeX 源、实验代码、图表、日志、文献缓存，方便存档或二次编译。</p>'
                                        '</div>'
                                    )
                                    full_bundle_download = gr.DownloadButton(
                                        label="Build & download full bundle (.zip)",
                                        size="sm",
                                        variant="primary",
                                    )

                                    paper_status = gr.HTML(
                                        '<div class="empty-state">研究完成后再点击 Export PDF 即可。</div>'
                                    )

            # ==================================================================
            # TAB 2 — CONFIGURATION
            # ==================================================================
            with gr.Tab("Configuration", id="configuration"):
                with gr.Row():
                    with gr.Column(scale=6, min_width=520):
                        with gr.Group(elem_classes=["surface-card"]):
                            gr.HTML(
                                '<div class="card-heading">'
                                '<h3>Primary Model</h3>'
                                '<p>负责执行科研任务的 OpenAI-compatible 模型。</p>'
                                '</div>'
                            )
                            agent_preset = gr.Dropdown(
                                label="预设",
                                choices=agent_preset_choices,
                                value=_match_preset(initial_settings.agent_model, initial_settings.agent_base_url, MODEL_PRESETS),
                            )
                            agent_model = gr.Textbox(
                                label="模型名称",
                                value=initial_settings.agent_model,
                                placeholder="例如：MiniMax-M2.7 / deepseek-v3",
                            )
                            agent_base_url = gr.Textbox(
                                label="Base URL",
                                value=initial_settings.agent_base_url,
                                placeholder="OpenAI-compatible API endpoint",
                            )
                            agent_api_key = gr.Textbox(
                                label="API Key",
                                type="password",
                                value=initial_settings.agent_api_key,
                                placeholder="留空时回退到环境变量",
                            )

                        with gr.Group(elem_classes=["surface-card"]):
                            gr.HTML(
                                '<div class="card-heading">'
                                '<h3>Reviewer Model</h3>'
                                '<p>用于关键节点核查引用、论证和潜在幻觉。</p>'
                                '</div>'
                            )
                            reviewer_enabled = gr.Checkbox(
                                label="启用独立评审",
                                value=initial_settings.reviewer_enabled,
                            )
                            reviewer_preset = gr.Dropdown(
                                label="预设",
                                choices=reviewer_preset_choices,
                                value=_match_preset(initial_settings.reviewer_model, initial_settings.reviewer_base_url, REVIEWER_PRESETS),
                            )
                            reviewer_model = gr.Textbox(
                                label="评审模型名称",
                                value=initial_settings.reviewer_model,
                            )
                            reviewer_base_url = gr.Textbox(
                                label="Base URL",
                                value=initial_settings.reviewer_base_url,
                            )
                            reviewer_api_key = gr.Textbox(
                                label="API Key",
                                type="password",
                                value=initial_settings.reviewer_api_key,
                            )

                        with gr.Group(elem_classes=["surface-card"]):
                            gr.HTML(
                                '<div class="card-heading">'
                                '<h3>OCR</h3>'
                                '<p>读取扫描 PDF 或图片文档时启用外部 OCR 服务。</p>'
                                '</div>'
                            )
                            ocr_enabled = gr.Checkbox(
                                label="运行任务时启用 OCR",
                                value=initial_settings.ocr_enabled,
                            )
                            ocr_api_url = gr.Textbox(
                                label="OCR API URL",
                                value=initial_settings.ocr_api_url,
                            )
                            ocr_token = gr.Textbox(
                                label="OCR Token",
                                type="password",
                                value=initial_settings.ocr_token,
                            )

                        with gr.Group(elem_classes=["surface-card"]):
                            gr.HTML(
                                '<div class="card-heading">'
                                '<h3>Literature Search</h3>'
                                '<p>使用 arXiv 免费搜索；配额限定每次研究的搜索次数，'
                                '避免 Agent 反复触发搜索调用。</p>'
                                '</div>'
                            )
                            search_quota = gr.Slider(
                                label="单次研究搜索配额",
                                info=(
                                    "限制每次研究运行内 search_literature 的最大调用次数。"
                                    "8 适合 4-6 个计划查询 + 少量临时补搜；0 表示不限。"
                                ),
                                minimum=0,
                                maximum=20,
                                value=initial_settings.search_quota,
                                step=1,
                            )

                    with gr.Column(scale=4, min_width=380):
                        with gr.Group(elem_classes=["surface-card"]):
                            gr.HTML(
                                '<div class="card-heading">'
                                '<h3>Run Parameters</h3>'
                                '<p>影响 Agent 单次执行的硬约束。</p>'
                                '</div>'
                            )
                            max_steps = gr.Slider(
                                label="Step Budget（0 = Adaptive）",
                                info=(
                                    "0 会让 LabForge 根据任务复杂度自适应步数，并保留有限硬上限；"
                                    "正数是初始软预算，任务仍在推进时会有限延长。"
                                ),
                                minimum=0,
                                maximum=240,
                                value=initial_settings.max_steps,
                                step=5,
                            )
                            temperature = gr.Slider(
                                label="Temperature",
                                minimum=0.0,
                                maximum=1.2,
                                value=initial_settings.temperature,
                                step=0.1,
                            )

                        with gr.Group(elem_classes=["surface-card"]):
                            gr.HTML(
                                '<div class="card-heading">'
                                '<h3>Actions</h3>'
                                '<p>保存、读取或验证当前模型与集成配置。</p>'
                                '</div>'
                            )
                            with gr.Row():
                                save_btn = gr.Button("Save", variant="primary", size="sm")
                                load_btn = gr.Button("Load saved", size="sm")
                                load_env_btn = gr.Button("From environment", size="sm")
                            with gr.Row():
                                test_btn = gr.Button("Test connection", size="sm")
                                reset_btn = gr.Button("Reset defaults", size="sm")
                            settings_status = gr.Markdown(initial_status_md)
                            connection_status = gr.Markdown("")

                        with gr.Group(elem_classes=["surface-card"]):
                            gr.HTML(
                                '<div class="card-heading">'
                                '<h3>Live Summary</h3>'
                                '<p>当前生效的模型 / 评审 / OCR 配置摘要。</p>'
                                '</div>'
                            )
                            settings_summary = gr.HTML(initial_summary_html)

            # ==================================================================
            # TAB 3 — ARCHIVE (only when SHOW_HISTORY_TAB)
            # ==================================================================
            traj_select = traj_view = traj_json_view = refresh_btn = None
            if SHOW_HISTORY_TAB:
                with gr.Tab("Archive", id="archive"):
                    with gr.Group(elem_classes=["surface-card", "history-shell"]):
                        gr.HTML(
                            '<div class="history-header">'
                            '<div><h3>Run Archive</h3>'
                            '<p>按主题选择历史轨迹，查看格式化摘要或原始 JSON。</p></div>'
                            '<span class="soft-badge">local trajectories</span>'
                            '</div>'
                        )
                        with gr.Row(elem_classes=["history-toolbar"]):
                            traj_select = gr.Dropdown(
                                label="Select run",
                                choices=initial_traj_choices,
                                value=None,
                                interactive=True,
                            )
                            refresh_btn = gr.Button("Refresh", size="sm")
                        with gr.Tabs():
                            with gr.Tab("Reading view"):
                                traj_view = gr.Markdown(
                                    "*选择一条记录查看内容。*",
                                    elem_classes=["history-view"],
                                )
                            with gr.Tab("Raw JSON"):
                                traj_json_view = gr.Textbox(
                                    label="Trajectory JSON",
                                    value="{}",
                                    lines=22,
                                    interactive=False,
                                )

        gr.HTML(
            '<div class="footer-note">'
            'LabForge · Research Studio  ·  monochrome workbench '
            'over the existing agent backend (research_report.md / paperforge_bundle.json / research_paper.pdf).'
            '</div>'
        )

        # ------------------------------------------------------------------
        # WIRING — every interactive element in the UI must call into the
        # canonical backend handler. The wrappers above only fan out
        # additional UI-only outputs that derive from already-public state.
        # ------------------------------------------------------------------

        # --- Configuration: presets, save/load/reset/test ----------------
        agent_preset.change(
            fn=apply_agent_preset,
            inputs=[agent_preset, agent_model, agent_base_url],
            outputs=[agent_model, agent_base_url],
        )
        reviewer_preset.change(
            fn=apply_reviewer_preset,
            inputs=[reviewer_preset, reviewer_model, reviewer_base_url],
            outputs=[reviewer_model, reviewer_base_url],
        )

        save_inputs = [
            agent_model, agent_base_url, agent_api_key,
            reviewer_enabled, reviewer_model, reviewer_base_url, reviewer_api_key,
            max_steps, temperature,
            ocr_enabled, ocr_api_url, ocr_token,
            search_quota,
            run_mode,
        ]
        save_btn.click(
            fn=save_settings_handler,
            inputs=save_inputs,
            outputs=[settings_status, settings_summary],
        )

        load_outputs = [
            agent_preset, agent_model, agent_base_url, agent_api_key,
            reviewer_enabled, reviewer_preset, reviewer_model, reviewer_base_url, reviewer_api_key,
            max_steps, temperature,
            ocr_enabled, ocr_api_url, ocr_token,
            search_quota,
            run_mode,
            settings_status, settings_summary, connection_status,
        ]
        load_btn.click(fn=load_saved_settings_handler, outputs=load_outputs)
        load_env_btn.click(fn=load_env_settings_handler, outputs=load_outputs)
        reset_btn.click(fn=reset_default_settings_handler, outputs=load_outputs)
        test_btn.click(
            fn=test_connection_handler,
            inputs=[agent_api_key, agent_base_url, agent_model],
            outputs=[connection_status],
        )

        # Auto-save on change so users don't have to hit save manually.
        autosave_components = [
            agent_model, agent_base_url, agent_api_key,
            reviewer_enabled, reviewer_model, reviewer_base_url, reviewer_api_key,
            ocr_enabled, ocr_api_url, ocr_token,
            search_quota,
            max_steps, temperature,
            run_mode,
        ]
        for component in autosave_components:
            component.change(
                fn=save_settings_handler,
                inputs=save_inputs,
                outputs=[settings_status, settings_summary],
                queue=False,
            )

        # --- Workspace: chat run -----------------------------------------
        stream_inputs = [
            chat_stash,
            chat_panel,
            agent_model, agent_base_url, agent_api_key,
            reviewer_enabled, reviewer_model, reviewer_base_url, reviewer_api_key,
            max_steps, temperature,
            ocr_enabled, ocr_api_url, ocr_token,
            search_quota,
            attach_btn,
            run_mode,
        ]
        # Sidebar fan-out: queue, papers, projects header/empty/N buttons/N states/more.
        sidebar_outputs = [
            queue_html, papers_html,
            projects_header_html, projects_empty_html,
            *project_btns,
            *project_states,
            projects_more_html,
        ]
        stream_outputs = [
            chat_panel,
            chat_status_html,
            pipeline_html,
            evidence_trace_html,
            *sidebar_outputs,
        ]

        chat_send_btn.click(
            fn=chat_quick_inject,
            inputs=[chat_input],
            outputs=[chat_input, chat_stash],
            queue=False,
        ).then(
            fn=chat_main_stream_with_rails,
            inputs=stream_inputs,
            outputs=stream_outputs,
        )

        chat_input.submit(
            fn=chat_quick_inject,
            inputs=[chat_input],
            outputs=[chat_input, chat_stash],
            queue=False,
        ).then(
            fn=chat_main_stream_with_rails,
            inputs=stream_inputs,
            outputs=stream_outputs,
        )

        chat_clear_btn.click(
            fn=chat_clear_history_with_rails,
            outputs=[
                chat_panel, chat_input,
                chat_status_html, pipeline_html, evidence_trace_html,
                *sidebar_outputs,
            ],
            queue=False,
        )

        refresh_rail_btn.click(
            fn=refresh_all_rails,
            outputs=sidebar_outputs,
            queue=False,
        )

        # ``gr.Files`` is its own drop target and renders chips for every
        # uploaded file inline — no extra status surface needed. The file
        # list is read straight off ``attach_btn`` when the user hits Send
        # (it's part of ``stream_inputs`` and gets staged into the sandbox
        # by ``_stage_attachments_into_sandbox`` in ``lab_forge.web``).

        # --- Paper export -------------------------------------------------
        paper_artifact_outputs = [
            report_artifact_row, report_artifact, report_download,
            bundle_artifact_row, bundle_artifact, bundle_download,
            pdf_artifact_row, pdf_artifact, pdf_download,
            paper_status,
        ]
        refresh_paper_btn.click(
            fn=refresh_paper_with_doc,
            outputs=[*paper_artifact_outputs, doc_preview_html],
        )
        pdf_btn.click(
            fn=convert_pdf_with_doc,
            inputs=[
                paper_language, paper_template,
                agent_model, agent_base_url, agent_api_key,
            ],
            outputs=[*paper_artifact_outputs, doc_preview_html],
        )
        full_bundle_download.click(
            fn=download_full_bundle_zip,
            outputs=[full_bundle_download],
        )

        # --- Archive tab --------------------------------------------------
        if SHOW_HISTORY_TAB and refresh_btn is not None:
            refresh_btn.click(fn=load_trajectory_list, outputs=[traj_select])
            traj_select.change(fn=view_trajectory, inputs=[traj_select], outputs=[traj_view])
            traj_select.change(fn=export_traj_json, inputs=[traj_select], outputs=[traj_json_view])

            # Rail project buttons jump to the Archive tab and surface the
            # picked trajectory using the same backend handlers as the dropdown.
            for btn, state in zip(project_btns, project_states):
                btn.click(
                    fn=open_project_in_archive,
                    inputs=[state],
                    outputs=[tabs_root, traj_select, traj_view, traj_json_view],
                    queue=False,
                )

    return app


def main():
    parser = argparse.ArgumentParser(description="LabForge Research Studio (redesigned UI)")
    parser.add_argument("--port", type=int, default=7861)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    load_project_env(project_root=PROJECT_ROOT, extra_search_dirs=[Path.cwd()])

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    app = create_ui()
    app.queue()
    app.launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
