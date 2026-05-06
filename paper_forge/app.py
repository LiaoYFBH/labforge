"""
PaperForge — 文档转学术论文 PDF 工作台。

特性：
- Monochrome Research Studio UI（与 lab-forge ui-new.py 同款主题）
- 支持 PaddleOCR 解析 PDF / 图片为 Markdown
- 支持星河社区（AI Studio）等任意 OpenAI 兼容大模型，UI 内可配置 / 切换
- 多种顶会 LaTeX 模板（IEEE / NeurIPS / ICML / ACL / ACM SIGCONF / 通用 article）
  可选，编译时自动从网络拉取对应会议样式文件
- LLM-write 与纯组装两种生成模式，可由调用方注入大模型
- 既能独立运行（单仓库开源），也能作为 lab-forge 等上游 agent 的论文生成模块
"""

from __future__ import annotations

import argparse
import logging
import os
import threading
import traceback
from pathlib import Path

import gradio as gr

from paper_forge.config import LLMConfig, OCRConfig, PDFStyleConfig
from paper_forge.env_utils import load_project_env
from paper_forge.llm_client import extract_json_from_response
from paper_forge.ocr_client import parse_multiple_documents
from paper_forge.paper_writer import (
    should_use_native_markdown_parser,
    structure_paper,
    structure_paper_stream,
)
from paper_forge import (
    PAPER_TEMPLATES,
    list_templates,
    load_paper_bundle,
    paper_bundle_to_markdown,
    render_paper_pdf,
)
from paper_forge.pdf_quality_agent import format_quality_summary, inspect_paper_artifacts
from paper_forge.utils import (
    ensure_output_dir,
    generate_output_filename,
    get_paper_labels,
)

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent

# ────────────────────────────────────────────────────────────────
# 模型预设：星河社区 (AI Studio) + MiniMax + 其它兼容接口
# ────────────────────────────────────────────────────────────────

MODEL_PRESETS: dict[str, dict[str, str]] = {
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
    "MiniMax-M2.7": {
        "model": "MiniMax-M2.7",
        "base_url": "https://api.minimaxi.com/v1",
    },
    "自定义": {
        "model": "",
        "base_url": "",
    },
}


# ────────────────────────────────────────────────────────────────
# Session 状态
# ────────────────────────────────────────────────────────────────

class SessionState:
    """Per-session state for tracking OCR results and images."""

    def __init__(self):
        self.ocr_markdown: str = ""
        self.ocr_images: dict[str, bytes] = {}
        self.extra_images: dict[str, bytes] = {}
        self.paper_json: dict | None = None
        self.status: str = "idle"
        self.lock = threading.Lock()

    @property
    def all_images(self) -> dict[str, bytes]:
        merged: dict[str, bytes] = {}
        merged.update(self.ocr_images)
        merged.update(self.extra_images)
        return merged


session = SessionState()


# ────────────────────────────────────────────────────────────────
# Modern tech CSS
# ────────────────────────────────────────────────────────────────

CUSTOM_CSS = """
/* PaperForge — monochrome Research Studio palette.
 *
 * The visual language matches the lab-forge ui-new.py file
 * so the two apps feel like siblings: white canvas, hairline borders,
 * serif headings, mono captions, single accent (Klein-blue) reserved for
 * primary actions and status badges. No glow, no neon, no gradient body.
 */
:root {
  --pf-bg: #ffffff;
  --pf-bg-soft: #fafafa;
  --pf-bg-rail: #f8f8f7;
  --pf-bg-tint: #f4f4f5;
  --pf-border: #ececec;
  --pf-border-strong: #d4d4d8;
  --pf-text-1: #0a0a0a;
  --pf-text-2: #525252;
  --pf-text-3: #a1a1aa;
  --pf-accent: #1d4ed8;
  --pf-accent-soft: #eef2ff;
  --pf-success: #166534;
  --pf-warning: #92400e;
  --pf-danger:  #991b1b;
  --pf-radius: 6px;
  --pf-radius-lg: 8px;
  --pf-shadow: 0 1px 0 0 rgba(15, 23, 42, 0.04);
  --pf-font-sans: "Inter", "PingFang SC", "Microsoft YaHei", system-ui, -apple-system, sans-serif;
  --pf-font-serif: "Source Serif 4", "Iowan Old Style", "Charter", Georgia, "Songti SC", serif;
  --pf-font-mono: "JetBrains Mono", "SF Mono", Menlo, Consolas, monospace;
}

body, .gradio-container {
  background: var(--pf-bg) !important;
  color: var(--pf-text-1) !important;
  font-family: var(--pf-font-sans) !important;
  font-size: 13px !important;
  letter-spacing: 0;
}

.gradio-container {
  max-width: 1280px !important;
  margin: 0 auto;
  padding: 0 24px 36px !important;
}

.gradio-container * { box-sizing: border-box; }
footer { display: none !important; }

/* Top bar — mirrors ui-new.py .sp-topbar ----------------------------------- */
.page-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 18px 0 14px;
  margin: 0 0 16px;
  border-bottom: 1px solid var(--pf-border);
  background: var(--pf-bg);
}
.page-header .brand-block { display: flex; align-items: center; gap: 12px; }
.page-header .brand-mark {
  width: 32px; height: 32px;
  border-radius: var(--pf-radius);
  background: var(--pf-text-1);
  color: #fff;
  font-weight: 600;
  font-size: 12px;
  letter-spacing: 0;
  display: inline-flex;
  align-items: center;
  justify-content: center;
}
.page-header h1 {
  font-family: var(--pf-font-serif) !important;
  font-size: 22px !important;
  font-weight: 600 !important;
  color: var(--pf-text-1) !important;
  margin: 0 !important;
  letter-spacing: 0;
  line-height: 1.1;
}
.page-header .pill {
  display: inline-block;
  font-family: var(--pf-font-mono);
  font-size: 10px;
  letter-spacing: 0;
  text-transform: uppercase;
  color: var(--pf-text-3);
  margin-bottom: 4px;
}
.page-header p {
  margin: 4px 0 0;
  color: var(--pf-text-3);
  font-size: 12px;
  line-height: 1.5;
  max-width: 640px;
}

/* Tabs — flat underline, no chrome ----------------------------------------- */
.tab-nav, [role="tablist"] {
  background: transparent !important;
  border-bottom: 1px solid var(--pf-border) !important;
  padding: 0 !important;
  margin: 0 0 16px !important;
}
.tab-nav button, [role="tab"] {
  background: transparent !important;
  color: var(--pf-text-2) !important;
  border: none !important;
  border-bottom: 2px solid transparent !important;
  border-radius: 0 !important;
  padding: 10px 14px !important;
  font-size: 13px !important;
  font-weight: 500 !important;
}
.tab-nav button.selected, [role="tab"][aria-selected="true"] {
  color: var(--pf-text-1) !important;
  border-bottom-color: var(--pf-text-1) !important;
  background: transparent !important;
}

/* Surface cards — single hairline border, no shadow drama ------------------ */
.surface-card,
.gradio-container .gr-form,
.gradio-container .gr-block.gr-group {
  border: 1px solid var(--pf-border) !important;
  border-radius: var(--pf-radius-lg) !important;
  background: var(--pf-bg) !important;
  box-shadow: var(--pf-shadow) !important;
}
.surface-card {
  padding: 14px 16px;
}
.surface-card h3, .gr-markdown h3 {
  margin: 2px 0 10px;
  padding-bottom: 8px;
  font-family: var(--pf-font-serif) !important;
  font-size: 15px !important;
  font-weight: 600 !important;
  color: var(--pf-text-1) !important;
  letter-spacing: 0;
  border-bottom: 1px dashed var(--pf-border);
}

.helper-note {
  color: var(--pf-text-3);
  font-size: 12px;
  line-height: 1.6;
}
.helper-note code, .gr-markdown code, code {
  font-family: var(--pf-font-mono) !important;
  background: var(--pf-bg-tint) !important;
  border: 1px solid var(--pf-border) !important;
  border-radius: 4px;
  color: var(--pf-text-2) !important;
  padding: 1px 6px;
  font-size: 11.5px;
}

/* Inputs ------------------------------------------------------------------- */
input, textarea, select,
.gradio-container input[type="text"],
.gradio-container input[type="number"],
.gradio-container textarea,
.gradio-container .gr-input,
.gradio-container .gr-textbox {
  background: var(--pf-bg) !important;
  color: var(--pf-text-1) !important;
  border: 1px solid var(--pf-border-strong) !important;
  border-radius: var(--pf-radius) !important;
  font-family: var(--pf-font-sans) !important;
  font-size: 13px !important;
}
input:focus, textarea:focus, select:focus {
  border-color: var(--pf-text-1) !important;
  box-shadow: none !important;
  outline: none !important;
}
label, .gr-input-label, .gr-checkbox label, .gr-radio label {
  color: var(--pf-text-2) !important;
  font-size: 12px !important;
  font-weight: 500 !important;
}

/* Buttons ------------------------------------------------------------------ */
button.gr-button, .gr-button, button {
  border-radius: var(--pf-radius) !important;
  font-family: var(--pf-font-sans) !important;
  font-size: 12px !important;
  font-weight: 500 !important;
  letter-spacing: 0 !important;
  border: 1px solid var(--pf-border-strong) !important;
  background: var(--pf-bg) !important;
  color: var(--pf-text-1) !important;
  padding: 6px 12px !important;
  box-shadow: none !important;
  transition: background 120ms linear, border-color 120ms linear;
}
button:hover, .gr-button:hover { background: var(--pf-bg-tint) !important; }
button.primary, .gr-button-primary, button.lg.primary, button[variant="primary"] {
  background: var(--pf-text-1) !important;
  color: #ffffff !important;
  border-color: var(--pf-text-1) !important;
  font-weight: 600 !important;
}
button.primary:hover, .gr-button-primary:hover {
  background: #1f1f1f !important;
  filter: none !important;
  transform: none !important;
  box-shadow: none !important;
}

/* Markdown body — keep the long-form output readable ----------------------- */
.gr-markdown { color: var(--pf-text-1) !important; }
.gr-markdown p, .gr-markdown li { color: var(--pf-text-2) !important; line-height: 1.65; }
.gr-markdown h1, .gr-markdown h2, .gr-markdown h3, .gr-markdown h4 {
  color: var(--pf-text-1) !important;
  font-family: var(--pf-font-serif) !important;
  letter-spacing: 0;
}
.gr-markdown blockquote {
  border-left: 2px solid var(--pf-border-strong) !important;
  padding-left: 10px !important;
  color: var(--pf-text-2) !important;
}

/* Template info (kept) — now monochrome dashed callout --------------------- */
.template-info {
  border: 1px dashed var(--pf-border-strong);
  border-radius: var(--pf-radius);
  padding: 10px 14px;
  background: var(--pf-bg-soft);
  color: var(--pf-text-2);
  font-size: 12px;
  line-height: 1.6;
}
.template-info code {
  background: var(--pf-bg-tint);
  border: 1px solid var(--pf-border);
}

/* Status pill colors used in helper banners ------------------------------- */
.status-badge {
  display: inline-flex;
  align-items: center;
  gap: 5px;
  font-size: 11px;
  font-weight: 500;
  padding: 2px 8px;
  border-radius: 999px;
  border: 1px solid var(--pf-border);
  background: var(--pf-bg);
  color: var(--pf-text-2);
}
.status-badge.ok {
  color: var(--pf-success);
  border-color: rgba(22, 101, 52, 0.18);
  background: rgba(22, 101, 52, 0.04);
}
.status-badge.warn {
  color: var(--pf-warning);
  border-color: rgba(146, 64, 14, 0.18);
  background: rgba(146, 64, 14, 0.04);
}
.status-badge.err {
  color: var(--pf-danger);
  border-color: rgba(153, 27, 27, 0.18);
  background: rgba(153, 27, 27, 0.04);
}

/* Radio / Checkbox — make the selected state unmistakable (filled box + ✓) */
.gradio-container input[type="radio"],
.gradio-container input[type="checkbox"] {
  -webkit-appearance: none !important;
  appearance: none !important;
  width: 16px !important;
  height: 16px !important;
  min-width: 16px !important;
  border: 1px solid var(--pf-border-strong) !important;
  background: var(--pf-bg) !important;
  cursor: pointer !important;
  position: relative;
  vertical-align: middle;
  margin: 0 6px 0 0 !important;
  padding: 0 !important;
  box-shadow: none !important;
  flex-shrink: 0;
}
.gradio-container input[type="radio"] { border-radius: 50% !important; }
.gradio-container input[type="checkbox"] { border-radius: 3px !important; }
.gradio-container input[type="radio"]:hover,
.gradio-container input[type="checkbox"]:hover {
  border-color: var(--pf-text-1) !important;
}
.gradio-container input[type="radio"]:checked,
.gradio-container input[type="checkbox"]:checked {
  background: var(--pf-text-1) !important;
  border-color: var(--pf-text-1) !important;
}
.gradio-container input[type="radio"]:checked::after,
.gradio-container input[type="checkbox"]:checked::after {
  content: "\2713";
  color: #ffffff;
  font-size: 12px;
  font-weight: 700;
  line-height: 1;
  position: absolute;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
}
.gradio-container .gr-radio label:has(input[type="radio"]:checked),
.gradio-container .gr-checkbox label:has(input[type="checkbox"]:checked),
.gradio-container label:has(> input[type="radio"]:checked),
.gradio-container label:has(> input[type="checkbox"]:checked) {
  color: var(--pf-text-1) !important;
  font-weight: 600 !important;
}
"""


# ────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────

TEXT_EXTENSIONS = {".md", ".txt", ".markdown", ".text"}
JSON_BUNDLE_EXTENSIONS = {".json"}
OCR_FILE_TYPES = [".pdf", ".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".gif"]
DOCUMENT_FILE_TYPES = [*OCR_FILE_TYPES, *sorted(TEXT_EXTENSIONS), *sorted(JSON_BUNDLE_EXTENSIONS)]


def _get_first_env(*keys: str) -> str:
    for key in keys:
        value = os.environ.get(key, "").strip()
        if value:
            return value
    return ""


def _default_llm_api_key() -> str:
    return _get_first_env("AI_STUDIO_API_KEY", "API_KEY", "OPENAI_API_KEY")


def _default_llm_base_url() -> str:
    return _get_first_env("API_BASE_URL", "OPENAI_BASE_URL") or "https://aistudio.baidu.com/llm/lmapi/v3"


def _default_llm_model() -> str:
    return _get_first_env("MODEL_NAME", "LLM_MODEL") or "ernie-4.5-turbo-128k-preview"


def _default_ocr_url() -> str:
    return _get_first_env("PADDLEOCR_API_URL", "OCR_API_URL", "PADDLEOCR_VL_API_URL") or ""


def _default_ocr_token() -> str:
    return _get_first_env("PADDLEOCR_TOKEN", "OCR_TOKEN", "PADDLEOCR_VL_TOKEN", "OCR_API_TOKEN")


def apply_model_preset(preset_label: str, current_model: str, current_base_url: str) -> tuple[str, str]:
    preset = MODEL_PRESETS.get(preset_label)
    if preset is None or preset_label == "自定义":
        return current_model, current_base_url
    return preset["model"], preset["base_url"]


def handle_test_llm(api_key: str, base_url: str, model: str) -> str:
    """Send a tiny ping to the LLM to verify configuration."""
    if not api_key.strip():
        return "请先填入 API Key（也可在 .env 里设置 AI_STUDIO_API_KEY / API_KEY）。"
    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key.strip(), base_url=base_url.strip(), timeout=20)
        client.chat.completions.create(
            model=model.strip(),
            messages=[{"role": "user", "content": "Reply with OK."}],
            max_tokens=2,
        )
        return f"✅ 连接成功：`{model}` 可用。"
    except Exception as exc:
        msg = str(exc)
        if len(msg) > 240:
            msg = msg[:240] + "…"
        return f"❌ 连接失败：{msg}"


def template_info_md(template_key: str) -> str:
    template = PAPER_TEMPLATES.get(template_key)
    if template is None:
        return ""
    extras = ""
    if template.download_urls:
        files = "、".join(name for name, _ in template.download_urls)
        extras = (
            f"\n**首次编译会自动从网络拉取的样式文件**：`{files}`"
            "\n下载失败时会回退到 TexLive 自带版本（若有）。"
        )
    return (
        f"**当前模板** {template.label}  \n"
        f"{template.description}  \n"
        f"模板文件：`{template.template_filename}`{extras}"
    )


# ────────────────────────────────────────────────────────────────
# OCR handler
# ────────────────────────────────────────────────────────────────

def handle_parse_documents(
    files,
    extra_image_files,
    ocr_url: str,
    ocr_token: str,
    use_orientation: bool,
    use_unwarping: bool,
    use_chart: bool,
):
    """Parse uploaded documents via PaddleOCR, or read text/markdown directly."""
    if not files:
        yield "请先上传至少一份文档（PDF / 图片 / Markdown / TXT）。", "", None
        return

    with session.lock:
        session.status = "parsing"
        session.ocr_images = {}
        session.extra_images = {}

    file_paths: list[Path] = []
    for f in files:
        if isinstance(f, str):
            file_paths.append(Path(f))
        elif hasattr(f, "name"):
            file_paths.append(Path(f.name))

    json_files = [fp for fp in file_paths if fp.suffix.lower() in JSON_BUNDLE_EXTENSIONS]
    text_files = [fp for fp in file_paths if fp.suffix.lower() in TEXT_EXTENSIONS]
    ocr_files = [
        fp for fp in file_paths
        if fp.suffix.lower() not in TEXT_EXTENSIONS | JSON_BUNDLE_EXTENSIONS
    ]

    yield f"开始处理 {len(file_paths)} 份文件…", "", None

    try:
        all_markdown_parts: list[str] = []

        for fp in text_files:
            try:
                content = fp.read_text(encoding="utf-8")
                all_markdown_parts.append(f"<!-- File: {fp.name} -->\n{content}")
            except Exception as e:
                all_markdown_parts.append(f"<!-- File: {fp.name} - 读取失败: {e} -->")

        for fp in json_files:
            try:
                paper = load_paper_bundle(fp)
                all_markdown_parts.append(
                    f"<!-- PaperForge bundle: {fp.name} -->\n"
                    f"{paper_bundle_to_markdown(paper)}"
                )
                with session.lock:
                    session.paper_json = paper
            except Exception as e:
                all_markdown_parts.append(f"<!-- Bundle: {fp.name} - 读取失败: {e} -->")

        if ocr_files:
            if not ocr_url or not ocr_token:
                yield (
                    f"已读取 {len(text_files)} 份文本文件；但还有 {len(ocr_files)} 份需要 OCR — "
                    "请先在「⚙️ 模型与 OCR 配置」标签里填好 PaddleOCR API URL 与 Token。",
                    "\n\n---\n\n".join(all_markdown_parts),
                    None,
                )
                return

            ocr_config = OCRConfig(
                api_url=ocr_url,
                token=ocr_token,
                use_doc_orientation_classify=use_orientation,
                use_doc_unwarping=use_unwarping,
                use_chart_recognition=use_chart,
            )

            result = parse_multiple_documents([str(fp) for fp in ocr_files], ocr_config)

            if result.markdown_text:
                all_markdown_parts.append(result.markdown_text)
            with session.lock:
                session.ocr_images = result.images

        if extra_image_files:
            for img_file in extra_image_files:
                img_path = img_file.name if hasattr(img_file, "name") else str(img_file)
                img_name = Path(img_path).name
                img_data = Path(img_path).read_bytes()
                session.extra_images[img_name] = img_data

        combined_markdown = "\n\n---\n\n".join(all_markdown_parts)

        with session.lock:
            session.ocr_markdown = combined_markdown

        img_count = len(session.all_images)
        bits = []
        if json_files:
            bits.append(f"{len(json_files)} 个 PaperForge JSON 包")
        if text_files:
            bits.append(f"{len(text_files)} 份文本文件")
        if ocr_files:
            bits.append(f"{len(ocr_files)} 份 OCR 文档")
        if img_count:
            bits.append(f"{img_count} 张图片")
        status = "解析完成 ✓ " + "、".join(bits)

        gallery_items = _build_gallery(session.all_images)

        with session.lock:
            session.status = "idle"

        yield status, combined_markdown, gallery_items

    except Exception as e:
        logger.error("Parsing failed: %s", traceback.format_exc())
        with session.lock:
            session.status = "error"
        yield f"解析失败：{e}", "", None


def _build_gallery(images: dict[str, bytes]):
    if not images:
        return None
    import tempfile
    items: list[tuple[str, str]] = []
    for name, data in images.items():
        suffix = Path(name).suffix or ".png"
        try:
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                tmp.write(data)
                items.append((tmp.name, name))
        except Exception:
            pass
    return items if items else None


# ────────────────────────────────────────────────────────────────
# Paper generation
# ────────────────────────────────────────────────────────────────

def handle_generate_paper(
    markdown_text: str,
    api_key: str,
    base_url: str,
    model: str,
    temperature: float,
    rewrite_mode: str,
    template_key: str,
    font_size: int,
    line_spacing: float,
    page_size: str,
    include_page_numbers: bool,
    target_total_words: int = 0,
    paper_language: str = "auto",
):
    """Generate academic paper PDF using the chosen LaTeX template.

    ``target_total_words`` only applies in ``Expand to Top-Conf`` mode.
    A value of ``0`` (default) keeps the canonical ~5100-word top-conf
    target. A positive value rescales every body section's word target
    proportionally so the rewritten paper hits the requested length.
    """
    if not markdown_text or not markdown_text.strip():
        yield "请先在「📄 上传与解析」里解析文档，或直接粘贴 Markdown 内容。", "", None
        return

    # rewrite_mode is one of:
    #   "Reformat Only"        → no LLM rewrite, just structure & format
    #   "Rewrite & Enhance"    → polish each section (preserves length)
    #   "Expand to Top-Conf"   → top-conference style expansion driven by
    #                            paper_forge.style_guide (multiplies length)
    expand_mode = rewrite_mode == "Expand to Top-Conf"
    user_target_words = int(target_total_words) if target_total_words and int(target_total_words) > 0 else None
    forced_language = paper_language if paper_language in ("zh", "en") else None
    language_rewrite = forced_language is not None
    rewrite = rewrite_mode in ("Rewrite & Enhance", "Expand to Top-Conf") or language_rewrite
    image_names = list(session.all_images.keys())
    use_native_markdown = should_use_native_markdown_parser(markdown_text)
    needs_llm = rewrite or not use_native_markdown

    if needs_llm and not api_key:
        yield "请在「⚙️ 模型与 OCR 配置」中填写大模型 API Key。", "", None
        return

    # User Step 7/9: Normalize the markdown strict formatting before doing anything else
    yield "Step 0/2: 正在规范化 Markdown 语法子集...", "", None
    try:
        if api_key:
            norm_llm = ChatOpenAI(api_key=api_key, base_url=base_url, model=model or "ernie-4.5-turbo-128k-preview")
            markdown_text = normalize_markdown(markdown_text, llm=norm_llm)
        else:
            markdown_text = normalize_markdown(markdown_text)
    except Exception as e:
        yield f"规范化异常: {e}", "", None

    if template_key not in PAPER_TEMPLATES:
        template_key = "general_article"

    llm_config = LLMConfig(
        model=model or "ernie-4.5-turbo-128k-preview",
        api_key=api_key,
        base_url=base_url or "https://aistudio.baidu.com/llm/lmapi/v3",
        temperature=temperature,
        max_tokens=8192,
    )

    pdf_style = PDFStyleConfig(
        page_size=page_size,
        font_size_body=font_size,
        line_spacing=line_spacing,
        include_page_numbers=include_page_numbers,
    )

    with session.lock:
        session.status = "generating"

    rewrite_mode_arg = "expand" if expand_mode else "polish"
    if expand_mode:
        rewrite_phase_label = "调用大模型按顶会风格扩展各章节…"
    elif language_rewrite and rewrite_mode == "Reformat Only":
        rewrite_phase_label = "调用大模型统一论文语言…"
    else:
        rewrite_phase_label = "调用大模型润色各章节…"

    try:
        if use_native_markdown:
            yield (
                "Step 1/2: 解析 Markdown 结构…"
                + (f" 同时{rewrite_phase_label}" if rewrite else ""),
                "", None,
            )
            paper = structure_paper(
                markdown_text, image_names, llm_config,
                rewrite=rewrite, temperature=temperature,
                rewrite_mode=rewrite_mode_arg,
                target_total_words=user_target_words if expand_mode else None,
                forced_language=forced_language,
            )
        else:
            yield "Step 1/2: 调用大模型重组论文结构…", "", None
            full_response: list[str] = []
            for chunk in structure_paper_stream(markdown_text, image_names, llm_config, temperature):
                full_response.append(chunk)
                if len(full_response) % 20 == 0:
                    preview = "".join(full_response)[-500:]
                    yield (
                        f"Step 1/2: 大模型生成中…（{len(''.join(full_response))} 字符）\n\n```\n…{preview}\n```",
                        "",
                        None,
                    )
            response_text = "".join(full_response)
            paper = extract_json_from_response(response_text)
            if paper is None:
                yield "Step 1/2: JSON 解析失败，重试中…", "", None
                paper = structure_paper(
                    markdown_text, image_names, llm_config,
                    rewrite=False, temperature=temperature,
                    forced_language=forced_language,
                )
            if rewrite and paper:
                yield f"Step 1/2: {rewrite_phase_label}", "", None
                if expand_mode:
                    from paper_forge.paper_writer import _expand_sections
                    paper = _expand_sections(
                        paper, llm_config, temperature,
                        target_total_words=user_target_words,
                        forced_language=forced_language,
                    )
                else:
                    from paper_forge.paper_writer import _rewrite_sections
                    paper = _rewrite_sections(paper, llm_config, temperature, forced_language)

        if paper is None:
            yield "❌ 论文结构化失败：大模型未返回有效 JSON。", "", None
            return

        with session.lock:
            session.paper_json = paper

        yield f"Step 2/2: 使用模板 `{template_key}` 渲染并编译 PDF…", "", None

        output_dir = ensure_output_dir()
        title = paper.get("title", "paper")
        filename = generate_output_filename(title)
        output_path = output_dir / filename

        render_paper_pdf(
            paper=paper,
            images=session.all_images,
            output_path=output_path,
            template_key=template_key,
            line_spacing=pdf_style.line_spacing,
            margin_mm=pdf_style.margin_mm,
            keep_tex=True,
            forced_language=forced_language,
        )
        qa_report = inspect_paper_artifacts(
            tex_path=output_path.parent / f".{output_path.stem}_latex" / "paper.tex",
            log_path=output_path.parent / f".{output_path.stem}_latex" / "paper.log",
            target_language=forced_language,
        )

        with session.lock:
            session.status = "done"

        labels = get_paper_labels(paper)
        preview = _build_preview(paper)
        figs = sum(len(s.get("figures", [])) for s in paper.get("sections", []))
        tables = sum(len(s.get("tables", [])) for s in paper.get("sections", []))

        yield (
            f"✅ 生成完成。\n\n"
            f"- 模板：`{template_key}`\n"
            f"- 标题：{paper.get('title', 'N/A')}\n"
            f"- 章节：{len(paper.get('sections', []))}\n"
            f"- {labels['figure']}：{figs}\n"
            f"- {labels['table']}：{tables}\n"
            f"- {labels['references']}：{len(paper.get('references', []))}\n\n"
            f"{format_quality_summary(qa_report)}",
            preview,
            str(output_path),
        )

    except Exception as e:
        logger.error("Paper generation failed: %s", traceback.format_exc())
        with session.lock:
            session.status = "error"
        log_tail = getattr(e, "log_tail", "")
        msg = f"❌ 生成失败：{e}"
        if log_tail:
            msg += f"\n\nLaTeX 日志摘录：\n```text\n{log_tail.strip()}\n```"
        yield msg, "", None


def _build_preview(paper: dict) -> str:
    """Build a markdown preview of the structured paper."""
    lines: list[str] = []
    labels = get_paper_labels(paper)
    title = paper.get("title", "Untitled")
    lines.append(f"# {title}\n")

    authors = paper.get("authors", [])
    if authors:
        lines.append(f"*{', '.join(authors)}*\n")

    abstract = paper.get("abstract", "")
    if abstract:
        lines.append(f"**{labels['abstract']}：** {abstract}\n")

    keywords = paper.get("keywords", [])
    if keywords:
        lines.append(f"**{labels['keywords']}：** {', '.join(keywords)}\n")

    lines.append("---\n")

    for section in paper.get("sections", []):
        heading = section.get("heading", "")
        level = section.get("level", 1)
        prefix = "#" * (level + 1)
        lines.append(f"{prefix} {heading}\n")
        for para in section.get("paragraphs", []):
            lines.append(f"{para}\n")
        for fig in section.get("figures", []):
            lines.append(f"*[{labels['figure']}: {fig.get('caption', '')}]*\n")
        for tbl in section.get("tables", []):
            cap = tbl.get("caption", "")
            headers = tbl.get("headers", [])
            rows = tbl.get("rows", [])
            if headers:
                if cap:
                    lines.append(f"\n**{cap}**\n")
                lines.append("| " + " | ".join(str(h) for h in headers) + " |")
                lines.append("| " + " | ".join("---" for _ in headers) + " |")
                for row in rows[:10]:
                    lines.append("| " + " | ".join(str(c) for c in row) + " |")
                lines.append("")

    refs = paper.get("references", [])
    if refs:
        lines.append(f"\n---\n## {labels['references']}\n")
        for ref in refs:
            lines.append(f"- {ref}")
    return "\n".join(lines)


# ────────────────────────────────────────────────────────────────
# UI
# ────────────────────────────────────────────────────────────────

def _hero_html() -> str:
    return """
<section class="page-header">
  <div class="brand-block">
    <div class="brand-mark">PF</div>
    <div>
      <span class="pill">PaperForge · LaTeX · OCR · LLM</span>
      <h1>PaperForge · 论文锻造工坊</h1>
      <p>上传 PDF / 图片 / Markdown 资料 → PaddleOCR 识别 → OpenAI 兼容大模型重组并扩展为顶会论文 → 选择 IEEE / NeurIPS / ICML / ACL / ACM 等模板 → 编译出可投稿的论文 PDF。可独立运行，也可作为 lab-forge 的论文生成模块。</p>
    </div>
  </div>
</section>
"""


def build_ui() -> gr.Blocks:
    template_list = list_templates()
    default_template = template_list[0].key if template_list else "general_article"
    template_choices_list = [(tpl.label, tpl.key) for tpl in template_list]

    # Monochrome Research Studio palette — matches lab-forge's ui-new.py.
    # Primary hue is intentionally slate (not blue/cyan) so the only
    # saturated accent in the page is the primary action button.
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
    )

    with gr.Blocks(title="PaperForge · 论文锻造工坊", theme=theme, css=CUSTOM_CSS) as demo:
        gr.HTML(_hero_html())

        with gr.Tabs(selected="upload"):
            # ── Tab 1: 模型 & OCR 配置 ──
            with gr.Tab("Model & OCR", id="settings"):
                with gr.Row():
                    with gr.Column(scale=1):
                        with gr.Group(elem_classes=["surface-card"]):
                            gr.Markdown("### 大模型配置（OpenAI 兼容 / 星河社区）")
                            preset_label = gr.Dropdown(
                                label="模型预设",
                                choices=list(MODEL_PRESETS.keys()),
                                value="星河社区 · ERNIE 4.5 Turbo 128K (推荐)",
                            )
                            api_key = gr.Textbox(
                                label="API Key (Access Token)",
                                placeholder="星河社区 access token / OpenAI 兼容 API Key",
                                type="password",
                                value=_default_llm_api_key(),
                            )
                            base_url = gr.Textbox(
                                label="Base URL",
                                value=_default_llm_base_url(),
                            )
                            model = gr.Textbox(
                                label="模型名称",
                                value=_default_llm_model(),
                            )
                            with gr.Row():
                                test_btn = gr.Button("测试连接", size="sm")
                            llm_status = gr.Markdown("")

                    with gr.Column(scale=1):
                        with gr.Group(elem_classes=["surface-card"]):
                            gr.Markdown("### PaddleOCR 配置")
                            ocr_url = gr.Textbox(
                                label="API URL",
                                placeholder="https://xxx.aistudio-app.com/layout-parsing",
                                value=_default_ocr_url(),
                            )
                            ocr_token = gr.Textbox(
                                label="Token",
                                placeholder="星河社区 PaddleOCR-VL 访问令牌",
                                type="password",
                                value=_default_ocr_token(),
                            )
                            gr.Markdown(
                                "PaddleOCR 用于识别上传的 PDF / 图片中的文字、表格与图像，是把扫描资料"
                                "转成可结构化 Markdown 的关键。",
                                elem_classes=["helper-note"],
                            )

                with gr.Accordion("PDF 排版选项", open=False):
                    with gr.Row():
                        page_size = gr.Radio(
                            label="页面尺寸",
                            choices=["A4", "Letter"],
                            value="A4",
                        )
                        font_size = gr.Slider(
                            label="正文字号",
                            minimum=9, maximum=14, step=1, value=11,
                        )
                        line_spacing = gr.Slider(
                            label="行距",
                            minimum=1.0, maximum=2.0, step=0.1, value=1.4,
                        )
                        include_page_numbers = gr.Checkbox(
                            label="显示页码",
                            value=True,
                        )

            # ── Tab 2: 上传与解析 ──
            with gr.Tab("Upload & Parse", id="upload"):
                with gr.Row():
                    with gr.Column(scale=2):
                        with gr.Group(elem_classes=["surface-card"]):
                            gr.Markdown("### 上传资料 (PDF / 图片 / Markdown / TXT)")
                            doc_files = gr.File(
                                label="文档 (将通过 PaddleOCR 解析或直接读取)",
                                file_count="multiple",
                                file_types=DOCUMENT_FILE_TYPES,
                                type="filepath",
                            )
                            extra_images = gr.File(
                                label="额外图片 (会随论文一起插入)",
                                file_count="multiple",
                                file_types=[".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tiff"],
                                type="filepath",
                            )

                            with gr.Accordion("OCR 高级选项", open=False):
                                use_orientation = gr.Checkbox(label="文档方向校正", value=False)
                                use_unwarping = gr.Checkbox(label="文档去畸变", value=False)
                                use_chart = gr.Checkbox(label="图表识别", value=False)

                            parse_btn = gr.Button("🔍 解析文档", variant="primary", size="lg")

                    with gr.Column(scale=3):
                        with gr.Group(elem_classes=["surface-card"]):
                            gr.Markdown("### 解析结果")
                            parse_status = gr.Markdown("等待解析…")
                            ocr_result = gr.Textbox(
                                label="OCR 结果 (Markdown，可编辑)",
                                lines=20,
                                max_lines=40,
                                interactive=True,
                            )
                            ocr_gallery = gr.Gallery(
                                label="提取出的图片",
                                columns=3,
                                height="auto",
                            )

            # ── Tab 3: 选择模板并生成论文 ──
            with gr.Tab("Template & Generate", id="generate"):
                with gr.Row():
                    with gr.Column(scale=3):
                        with gr.Group(elem_classes=["surface-card"]):
                            gr.Markdown("### 编辑论文内容（Markdown）")
                            edit_markdown = gr.Textbox(
                                label="论文内容",
                                lines=22,
                                max_lines=50,
                                interactive=True,
                                placeholder="在这里粘贴或编辑 Markdown 内容…\n\n或先到「📄 上传与解析」里抽取文档内容。",
                            )

                    with gr.Column(scale=2):
                        with gr.Group(elem_classes=["surface-card"]):
                            gr.Markdown("### 顶会 LaTeX 模板")
                            template_dropdown = gr.Dropdown(
                                label="选择论文模板",
                                choices=template_choices_list,
                                value=default_template,
                            )
                            template_info = gr.Markdown(
                                template_info_md(default_template),
                                elem_classes=["template-info"],
                            )
                            gr.Markdown(
                                "首次选定带有 `.sty` 依赖的模板时，编译流程会自动从网络拉取该会议的"
                                "样式文件并缓存到 `templates/_assets/`。下载失败时回退到 TexLive 自带版本。",
                                elem_classes=["helper-note"],
                            )

                        with gr.Group(elem_classes=["surface-card"]):
                            gr.Markdown("### 生成参数")
                            paper_language = gr.Radio(
                                label="论文语言",
                                choices=[
                                    ("Auto", "auto"),
                                    ("中文", "zh"),
                                    ("English", "en"),
                                ],
                                value="auto",
                                info="选择中文/英文会强制改写正文语言；Auto 按输入内容自动判断。",
                            )
                            rewrite_mode = gr.Radio(
                                label="生成模式",
                                choices=[
                                    "Reformat Only",
                                    "Rewrite & Enhance",
                                    "Expand to Top-Conf",
                                ],
                                value="Expand to Top-Conf",
                                info=(
                                    "Reformat Only · 仅按结构整理；"
                                    "Rewrite & Enhance · 润色当前内容、字数大致不变；"
                                    "Expand to Top-Conf · 按顶会论文写作范式逐章节"
                                    "扩展到目标字数，受 paper_forge.style_guide 约束、"
                                    "禁止编造数字与引用（最慢，但是最完整）。"
                                ),
                            )
                            temperature = gr.Slider(
                                label="Temperature",
                                minimum=0.0, maximum=1.0, step=0.05, value=0.3,
                            )
                            target_total_words = gr.Number(
                                label="自定义正文总字数（仅 Expand 模式生效，0 = 顶会默认 ≈ 5100 词）",
                                value=0,
                                precision=0,
                                minimum=0,
                                maximum=20000,
                                step=500,
                                info="留 0 走顶会默认；填正数（如 8000）按比例放大每节字数，仅在 Expand to Top-Conf 时使用。",
                            )
                            generate_btn = gr.Button("生成论文 PDF", variant="primary", size="lg")

                gen_status = gr.Markdown("准备生成。")
                gr.Markdown("---")

                with gr.Row():
                    with gr.Column(scale=3):
                        with gr.Group(elem_classes=["surface-card"]):
                            gr.Markdown("### 论文预览")
                            paper_preview = gr.Markdown("生成后会在这里显示论文预览。")
                    with gr.Column(scale=1):
                        with gr.Group(elem_classes=["surface-card"]):
                            gr.Markdown("### 下载")
                            pdf_download = gr.File(label="论文 PDF", interactive=False)

        # ── Wiring ──

        preset_label.change(
            fn=apply_model_preset,
            inputs=[preset_label, model, base_url],
            outputs=[model, base_url],
        )

        test_btn.click(
            fn=handle_test_llm,
            inputs=[api_key, base_url, model],
            outputs=[llm_status],
        )

        template_dropdown.change(
            fn=template_info_md,
            inputs=[template_dropdown],
            outputs=[template_info],
        )

        parse_btn.click(
            fn=handle_parse_documents,
            inputs=[
                doc_files, extra_images,
                ocr_url, ocr_token,
                use_orientation, use_unwarping, use_chart,
            ],
            outputs=[parse_status, ocr_result, ocr_gallery],
        )

        ocr_result.change(
            fn=lambda x: x,
            inputs=[ocr_result],
            outputs=[edit_markdown],
        )

        generate_btn.click(
            fn=handle_generate_paper,
            inputs=[
                edit_markdown,
                api_key, base_url, model, temperature,
                rewrite_mode,
                template_dropdown,
                font_size, line_spacing, page_size, include_page_numbers,
                target_total_words,
                paper_language,
            ],
            outputs=[gen_status, paper_preview, pdf_download],
        )

    return demo


# ────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="PaperForge — 论文锻造工坊")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    load_project_env(project_root=PROJECT_ROOT)

    demo = build_ui()
    demo.queue()
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
    )


if __name__ == "__main__":
    main()
