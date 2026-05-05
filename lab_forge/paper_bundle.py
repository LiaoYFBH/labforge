"""Helpers for building and loading structured paper bundles for PaperForge."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable

SECTION_ORDER: list[tuple[str, str]] = [
    ("introduction", "1. Introduction"),
    ("related_work", "2. Related Work"),
    ("methodology", "3. Methodology"),
    ("setup", "4. Experimental Setup"),
    ("results", "5. Results"),
    ("analysis", "6. Analysis & Discussion"),
    ("conclusion", "7. Conclusion"),
]

SECTION_ALIASES = {
    "intro": "introduction",
    "introduction": "introduction",
    "background": "related_work",
    "related_work": "related_work",
    "relatedwork": "related_work",
    "literature": "related_work",
    "method": "methodology",
    "methods": "methodology",
    "methodology": "methodology",
    "setup": "setup",
    "experimental_setup": "setup",
    "experiment": "setup",
    "results": "results",
    "result": "results",
    "analysis": "analysis",
    "discussion": "analysis",
    "conclusion": "conclusion",
    "conclusions": "conclusion",
}

FIGURE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".svg", ".gif", ".bmp"}
TABLE_EXTENSIONS = {".csv", ".tsv"}
ARTIFACT_DIR_HINTS = (
    "",
    "output",
    "outputs",
    "artifacts",
    "figures",
    "plots",
    "results",
)
MARKDOWN_TABLE_PREVIEW_ROWS = 10
TABLE_PREVIEW_MAX_ROWS = 10
TABLE_PREVIEW_MAX_COLUMNS = 7
TABLE_TEXT_MAX_CHARS = 18


def build_paper_bundle(
    *,
    working_dir: str | Path,
    title: str,
    abstract: str = "",
    keywords: Iterable[str] | str | None = None,
    introduction: str = "",
    related_work: str = "",
    methodology: str = "",
    setup: str = "",
    results: str = "",
    analysis: str = "",
    conclusion: str = "",
    references: Iterable[str] | str | None = None,
    figures: Iterable[dict[str, Any]] | None = None,
    tables: Iterable[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Create a PaperForge-compatible structured paper bundle."""
    root = Path(working_dir).resolve()
    section_text = {
        "introduction": introduction,
        "related_work": related_work,
        "methodology": methodology,
        "setup": setup,
        "results": results,
        "analysis": analysis,
        "conclusion": conclusion,
    }

    section_map = _build_sections(section_text)
    paper = {
        "title": (title or "").strip() or "Untitled Research Report",
        "authors": [],
        "abstract": (abstract or "").strip(),
        "keywords": _coerce_text_list(keywords),
        "sections": [],
        "references": _dedupe_preserve_order(_coerce_text_list(references)),
    }

    figure_specs = _merge_artifact_specs(
        _coerce_artifact_specs(figures),
        _discover_artifact_specs(root, FIGURE_EXTENSIONS),
    )
    for spec in figure_specs:
        path = _resolve_artifact_path(spec.get("path", ""), root, FIGURE_EXTENSIONS)
        if path is None:
            continue
        section_key = _normalize_section_key(spec.get("section"))
        target = _ensure_section(section_map, section_key)
        target["figures"].append(
            {
                "filename": path.name,
                "caption": _clean_text(spec.get("caption")) or _humanize_filename(path.name),
                "path": _bundle_path(path, root),
            }
        )

    table_specs = _merge_artifact_specs(
        _coerce_artifact_specs(tables),
        _discover_artifact_specs(root, TABLE_EXTENSIONS),
    )
    for spec in table_specs:
        path = _resolve_artifact_path(spec.get("path", ""), root, TABLE_EXTENSIONS)
        if path is None:
            continue
        parsed = _parse_table_file(path)
        if parsed is None:
            continue
        section_key = _normalize_section_key(spec.get("section"))
        target = _ensure_section(section_map, section_key)
        target["tables"].append(
            {
                "caption": _clean_text(spec.get("caption")) or _humanize_filename(path.stem),
                "headers": parsed["headers"],
                "rows": parsed["rows"],
                "source_rows": parsed.get("source_rows", len(parsed["rows"])),
                "source_columns": parsed.get("source_columns", len(parsed["headers"])),
                "preview_rows": parsed.get("preview_rows", len(parsed["rows"])),
                "preview_columns": parsed.get("preview_columns", len(parsed["headers"])),
                "filename": path.name,
                "path": _bundle_path(path, root),
            }
        )

    sections: list[dict[str, Any]] = []
    for key, _heading in SECTION_ORDER:
        section = section_map.get(key)
        if section and _section_has_content(section):
            sections.append(section)
    if not sections:
        sections.append(
            {
                "heading": "5. Results",
                "level": 1,
                "paragraphs": ["No results were recorded."],
                "figures": [],
                "tables": [],
            }
        )
    paper["sections"] = sections
    return paper


def paper_bundle_to_markdown(paper: dict[str, Any]) -> str:
    """Render a structured bundle back to Markdown for human review and fallback parsing."""
    lines: list[str] = [f"# {paper.get('title', 'Untitled Research Report')}", ""]

    abstract = _clean_text(paper.get("abstract"))
    if abstract:
        lines.extend(["## Abstract", "", abstract, ""])

    keywords = _coerce_text_list(paper.get("keywords"))
    if keywords:
        lines.extend([f"**Keywords:** {', '.join(keywords)}", ""])

    for section in paper.get("sections", []):
        if not _section_has_content(section):
            continue
        heading = _clean_text(section.get("heading")) or "Section"
        lines.extend([f"## {heading}", ""])

        for paragraph in section.get("paragraphs", []):
            paragraph_text = _clean_text(paragraph)
            if paragraph_text:
                lines.extend([paragraph_text, ""])

        for figure in section.get("figures", []):
            filename = _clean_text(figure.get("filename"))
            if not filename:
                continue
            caption = _clean_text(figure.get("caption")) or _humanize_filename(filename)
            lines.extend([f"![{caption}]({filename})", "", f"*{caption}*", ""])

        for table in section.get("tables", []):
            caption = _clean_text(table.get("caption"))
            if caption:
                lines.extend([f"**{caption}**", ""])
            lines.extend(_markdown_table_lines(table))
            lines.append("")
            filename = _clean_text(table.get("filename"))
            if filename and _table_is_preview_only(table):
                lines.extend(
                    [
                        f"*Table preview only. Full data is kept in `{filename}`.*",
                        "",
                    ]
                )

    references = _coerce_text_list(paper.get("references"))
    if references:
        lines.extend(["## References", ""])
        lines.extend(f"- {ref}" for ref in references)
        lines.append("")

    return "\n".join(lines).strip() + "\n"


def save_paper_bundle(bundle: dict[str, Any], path: str | Path) -> Path:
    """Persist a structured paper bundle as JSON."""
    output_path = Path(path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(bundle, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return output_path


def load_paper_bundle(path: str | Path) -> dict[str, Any]:
    """Load a structured paper bundle from JSON."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_bundle_images(bundle: dict[str, Any], working_dir: str | Path) -> dict[str, bytes]:
    """Load figure bytes for a bundle so PaperForge can render them into the PDF."""
    root = Path(working_dir).resolve()
    images: dict[str, bytes] = {}
    for section in bundle.get("sections", []):
        for figure in section.get("figures", []):
            candidate = _resolve_bundle_asset_path(figure, root, FIGURE_EXTENSIONS)
            if candidate is None:
                continue
            try:
                images[figure.get("filename", candidate.name)] = candidate.read_bytes()
            except OSError:
                continue
    return images


def _build_sections(section_text: dict[str, str]) -> dict[str, dict[str, Any]]:
    sections: dict[str, dict[str, Any]] = {}
    for key, heading in SECTION_ORDER:
        text = _clean_text(section_text.get(key))
        sections[key] = {
            "heading": heading,
            "level": 1,
            "paragraphs": [text] if text else [],
            "figures": [],
            "tables": [],
        }
    return sections


def _coerce_text_list(value: Iterable[str] | str | None) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        candidates = value.replace("\r", "\n").split("\n")
    else:
        candidates = list(value)
    cleaned: list[str] = []
    for item in candidates:
        text = _clean_text(item)
        if not text:
            continue
        if text.startswith("- "):
            text = text[2:].strip()
        cleaned.append(text)
    return cleaned


def _coerce_artifact_specs(value: Iterable[dict[str, Any]] | None) -> list[dict[str, Any]]:
    if not value:
        return []
    specs: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        path = _clean_text(item.get("path") or item.get("filename"))
        if not path:
            continue
        specs.append(
            {
                "path": path,
                "caption": _clean_text(item.get("caption")),
                "section": _clean_text(item.get("section")),
            }
        )
    return specs


def _merge_artifact_specs(
    explicit_specs: list[dict[str, Any]],
    discovered_specs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    seen: set[str] = set()
    for spec in explicit_specs + discovered_specs:
        key = (spec.get("path") or "").strip()
        if not key or key in seen:
            continue
        seen.add(key)
        merged.append(spec)
    return merged


def _discover_artifact_specs(root: Path, allowed_suffixes: set[str]) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    seen: set[Path] = set()
    for artifact_dir in _iter_artifact_directories(root):
        if not artifact_dir.exists():
            continue
        for candidate in artifact_dir.rglob("*"):
            if candidate in seen or not candidate.is_file():
                continue
            seen.add(candidate)
            if candidate.suffix.lower() not in allowed_suffixes:
                continue
            specs.append(
                {
                    "path": _bundle_path(candidate, root),
                    "caption": _humanize_filename(candidate.stem),
                    "section": "results",
                }
            )
    return specs


def _iter_artifact_directories(root: Path) -> list[Path]:
    candidates: list[Path] = []
    for hint in ARTIFACT_DIR_HINTS:
        path = root / hint if hint else root
        if path.exists():
            candidates.append(path)
    deduped: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        deduped.append(resolved)
    return deduped


def _resolve_artifact_path(
    raw_path: str,
    root: Path,
    allowed_suffixes: set[str],
) -> Path | None:
    cleaned = _clean_text(raw_path)
    if not cleaned:
        return None
    candidate = Path(cleaned)
    if not candidate.is_absolute():
        candidate = (root / candidate).resolve()
    if candidate.exists() and candidate.suffix.lower() in allowed_suffixes:
        return candidate
    return _find_artifact_by_name(Path(cleaned).name, root, allowed_suffixes)


def _find_artifact_by_name(name: str, root: Path, allowed_suffixes: set[str]) -> Path | None:
    for artifact_dir in _iter_artifact_directories(root):
        for candidate in artifact_dir.rglob(name):
            if candidate.is_file() and candidate.suffix.lower() in allowed_suffixes:
                return candidate.resolve()
    return None


def _resolve_bundle_asset_path(
    asset: dict[str, Any],
    root: Path,
    allowed_suffixes: set[str],
) -> Path | None:
    return _resolve_artifact_path(
        asset.get("path") or asset.get("filename") or "",
        root,
        allowed_suffixes,
    )


def _normalize_section_key(value: str | None) -> str:
    normalized = (value or "").strip().lower().replace(" ", "_")
    return SECTION_ALIASES.get(normalized, "results")


def _ensure_section(
    section_map: dict[str, dict[str, Any]],
    section_key: str,
) -> dict[str, Any]:
    if section_key not in section_map:
        heading_lookup = dict(SECTION_ORDER)
        section_map[section_key] = {
            "heading": heading_lookup.get(section_key, "5. Results"),
            "level": 1,
            "paragraphs": [],
            "figures": [],
            "tables": [],
        }
    return section_map[section_key]


def _section_has_content(section: dict[str, Any]) -> bool:
    return bool(
        any(_clean_text(paragraph) for paragraph in section.get("paragraphs", []))
        or section.get("figures")
        or section.get("tables")
    )


def _parse_table_file(path: Path) -> dict[str, list[list[str]] | list[str]] | None:
    delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle, delimiter=delimiter)
            rows = [row for row in reader if any(cell.strip() for cell in row)]
    except UnicodeDecodeError:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.reader(handle, delimiter=delimiter)
            rows = [row for row in reader if any(cell.strip() for cell in row)]
    except OSError:
        return None

    if not rows:
        return None
    headers = [str(cell).strip() for cell in rows[0]]
    body_rows = [[str(cell).strip() for cell in row] for row in rows[1:]]
    source_columns = max(len(headers), max((len(row) for row in body_rows), default=0))
    if source_columns == 0:
        source_columns = len(headers)

    if len(headers) < source_columns:
        headers.extend(f"Col {idx + 1}" for idx in range(len(headers), source_columns))

    if headers and not headers[0] and any(row and row[0].strip() for row in body_rows):
        headers[0] = "Model"

    preview_columns = min(source_columns, TABLE_PREVIEW_MAX_COLUMNS)
    preview_headers = [
        _format_table_cell(headers[idx], max_chars=TABLE_TEXT_MAX_CHARS, numeric=False)
        for idx in range(preview_columns)
    ]
    preview_rows: list[list[str]] = []
    for row in body_rows[:TABLE_PREVIEW_MAX_ROWS]:
        padded = row + [""] * max(0, source_columns - len(row))
        preview_rows.append(
            [
                _format_table_cell(
                    padded[idx],
                    max_chars=TABLE_TEXT_MAX_CHARS,
                )
                for idx in range(preview_columns)
            ]
        )

    return {
        "headers": preview_headers,
        "rows": preview_rows,
        "source_rows": len(body_rows),
        "source_columns": source_columns,
        "preview_rows": len(preview_rows),
        "preview_columns": len(preview_headers),
    }


def _markdown_table_lines(table: dict[str, Any]) -> list[str]:
    headers = [str(cell) for cell in table.get("headers", [])]
    rows = [[str(cell) for cell in row] for row in table.get("rows", [])]
    if not headers and not rows:
        return []

    if not headers and rows:
        headers = [f"Col {idx + 1}" for idx in range(len(rows[0]))]

    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows[:MARKDOWN_TABLE_PREVIEW_ROWS]:
        padded = row + [""] * max(0, len(headers) - len(row))
        lines.append("| " + " | ".join(padded[: len(headers)]) + " |")
    return lines


def _table_is_preview_only(table: dict[str, Any]) -> bool:
    source_rows = int(table.get("source_rows") or len(table.get("rows", [])))
    source_columns = int(table.get("source_columns") or len(table.get("headers", [])))
    preview_rows = len(table.get("rows", []))
    preview_columns = len(table.get("headers", []))
    return source_rows > preview_rows or source_columns > preview_columns


def _format_table_cell(value: Any, *, max_chars: int, numeric: bool = True) -> str:
    text = _clean_text(value)
    if not text:
        return ""

    if numeric:
        compact_number = _compact_number(text)
        if compact_number is not None:
            return compact_number

    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1] + "…"


def _compact_number(text: str) -> str | None:
    normalized = text.replace(",", "").strip()
    if not normalized:
        return None
    try:
        value = float(normalized)
    except ValueError:
        return None

    if value.is_integer():
        return str(int(value))

    abs_value = abs(value)
    if abs_value >= 1000 or (0 < abs_value < 0.001):
        return f"{value:.2e}"
    if abs_value >= 1:
        return f"{value:.4f}".rstrip("0").rstrip(".")
    return f"{value:.4g}"


def _bundle_path(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root).as_posix()
    except ValueError:
        return str(path.resolve())


def _humanize_filename(name: str) -> str:
    cleaned = name.replace("_", " ").replace("-", " ").strip()
    return cleaned.title() if cleaned else name


def _clean_text(value: Any) -> str:
    return str(value or "").strip()


def _dedupe_preserve_order(items: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result
