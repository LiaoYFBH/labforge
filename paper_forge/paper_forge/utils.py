"""Utility functions for PaperForge."""

from __future__ import annotations

import re
import shutil
import tempfile
import uuid
from pathlib import Path

OUTPUT_DIR = Path(__file__).parent.parent / "output"
CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
LATIN_RE = re.compile(r"[A-Za-z]")


def create_workspace() -> Path:
    """Create a temporary workspace directory for a session."""
    workspace = Path(tempfile.mkdtemp(prefix="paperforge_"))
    return workspace


def cleanup_workspace(workspace: Path):
    """Remove a temporary workspace."""
    if workspace.exists() and "paperforge_" in workspace.name:
        shutil.rmtree(workspace, ignore_errors=True)


def ensure_output_dir() -> Path:
    """Ensure the output directory exists and return its path."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR


def generate_output_filename(title: str = "paper") -> str:
    """Generate a unique output filename."""
    safe_title = "".join(c for c in title[:30] if c.isalnum() or c in " _-").strip()
    safe_title = safe_title.replace(" ", "_") or "paper"
    short_id = uuid.uuid4().hex[:6]
    return f"{safe_title}_{short_id}.pdf"


def detect_paper_language(paper: dict | str) -> str:
    """Detect whether the paper content is primarily Chinese or English."""
    if isinstance(paper, dict):
        parts: list[str] = []
        parts.append(str(paper.get("title", "")))
        parts.append(str(paper.get("abstract", "")))
        parts.extend(str(author) for author in paper.get("authors", []))
        parts.extend(str(keyword) for keyword in paper.get("keywords", []))
        parts.extend(str(ref) for ref in paper.get("references", []))
        for section in paper.get("sections", []):
            parts.append(str(section.get("heading", "")))
            parts.extend(str(para) for para in section.get("paragraphs", []))
            for fig in section.get("figures", []):
                parts.append(str(fig.get("caption", "")))
            for tbl in section.get("tables", []):
                parts.append(str(tbl.get("caption", "")))
                parts.extend(str(header) for header in tbl.get("headers", []))
                for row in tbl.get("rows", []):
                    parts.extend(str(cell) for cell in row)
        text = "\n".join(part for part in parts if part)
    else:
        text = str(paper)

    cjk_count = len(CJK_RE.findall(text))
    latin_count = len(LATIN_RE.findall(text))
    return "zh" if cjk_count >= max(8, latin_count) else "en"


def get_paper_labels(paper: dict | str) -> dict[str, str]:
    """Return localized labels for the paper based on its primary language."""
    language = detect_paper_language(paper)
    return get_labels_for_language(language)


def get_labels_for_language(language: str) -> dict[str, str]:
    """Return localized labels for an explicit language code."""
    if language == "zh":
        return {
            "abstract": "摘要",
            "keywords": "关键词",
            "references": "参考文献",
            "figure": "图",
            "table": "表",
            "image_not_available": "图片文件 '{filename}' 不可用",
            "image_render_failed": "图片渲染失败",
        }

    return {
        "abstract": "Abstract",
        "keywords": "Keywords",
        "references": "References",
        "figure": "Figure",
        "table": "Table",
        "image_not_available": "image '{filename}' not available",
        "image_render_failed": "image rendering failed",
    }


def language_name(code: str) -> str:
    """Map internal language code to a human-readable language name."""
    return {
        "zh": "Chinese",
        "en": "English",
    }.get(code, "the original language")
