"""PaperForge bundle IO.

This keeps the PaperForge package independent from LabForge while accepting
the same plain-JSON paper schema LabForge writes as ``paperforge_bundle.json``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def normalize_paper_bundle(data: dict[str, Any]) -> dict[str, Any]:
    """Return a defensive copy of a PaperForge-compatible paper dict."""

    paper = {
        "title": str(data.get("title") or "Untitled Paper").strip() or "Untitled Paper",
        "authors": _text_list(data.get("authors")),
        "abstract": str(data.get("abstract") or "").strip(),
        "keywords": _text_list(data.get("keywords")),
        "sections": [],
        "references": _text_list(data.get("references")),
    }
    for section in data.get("sections") or []:
        if not isinstance(section, dict):
            continue
        paragraphs = section.get("paragraphs")
        if isinstance(paragraphs, str):
            paragraphs = [paragraphs]
        elif not isinstance(paragraphs, list):
            paragraphs = []
        normalized = {
            "heading": str(section.get("heading") or "Section").strip() or "Section",
            "level": int(section.get("level") or 1),
            "paragraphs": [str(p).strip() for p in paragraphs if str(p).strip()],
            "figures": [f for f in section.get("figures", []) if isinstance(f, dict)],
            "tables": [t for t in section.get("tables", []) if isinstance(t, dict)],
        }
        if normalized["paragraphs"] or normalized["figures"] or normalized["tables"]:
            paper["sections"].append(normalized)
    return paper


def load_paper_bundle(path: str | Path) -> dict[str, Any]:
    """Load a LabForge/PaperForge JSON bundle from disk."""

    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("PaperForge bundle must be a JSON object.")
    return normalize_paper_bundle(data)


def paper_bundle_to_markdown(paper: dict[str, Any]) -> str:
    """Render a bundle to Markdown so the standalone UI can edit it."""

    normalized = normalize_paper_bundle(paper)
    lines: list[str] = [f"# {normalized['title']}", ""]
    if normalized["authors"]:
        lines.extend([", ".join(normalized["authors"]), ""])
    if normalized["abstract"]:
        lines.extend(["## Abstract", "", normalized["abstract"], ""])
    if normalized["keywords"]:
        lines.extend([f"**Keywords:** {', '.join(normalized['keywords'])}", ""])

    for section in normalized["sections"]:
        lines.extend([f"## {section['heading']}", ""])
        for paragraph in section.get("paragraphs", []):
            lines.extend([paragraph, ""])
        for fig in section.get("figures", []):
            filename = str(fig.get("filename") or fig.get("path") or "").strip()
            if not filename:
                continue
            caption = str(fig.get("caption") or Path(filename).stem).strip()
            lines.extend([f"![{caption}]({Path(filename).name})", "", f"*{caption}*", ""])
        for table in section.get("tables", []):
            caption = str(table.get("caption") or "").strip()
            if caption:
                lines.extend([f"**{caption}**", ""])
            lines.extend(_markdown_table(table))
            lines.append("")

    if normalized["references"]:
        lines.extend(["## References", ""])
        lines.extend(f"- {ref}" for ref in normalized["references"])
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def _text_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        values = value.splitlines()
    elif isinstance(value, list):
        values = value
    else:
        values = [value]
    return [str(v).strip() for v in values if str(v).strip()]


def _markdown_table(table: dict[str, Any]) -> list[str]:
    headers = [str(cell) for cell in table.get("headers", [])]
    rows = [[str(cell) for cell in row] for row in table.get("rows", [])]
    if not headers and rows:
        headers = [f"Col {idx + 1}" for idx in range(len(rows[0]))]
    if not headers:
        return []
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        padded = row + [""] * max(0, len(headers) - len(row))
        lines.append("| " + " | ".join(padded[: len(headers)]) + " |")
    return lines
