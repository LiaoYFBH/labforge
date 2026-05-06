"""
Academic paper PDF renderer using fpdf2.

Generates well-formatted PDF papers from structured paper data (JSON).
Pure Python, no LaTeX required.
"""

from __future__ import annotations

import io
import logging
import re
import tempfile
from pathlib import Path
from typing import Any

from fpdf import FPDF
from fpdf.enums import XPos, YPos

from .config import PDFStyleConfig
from .paper_writer import _parse_markdown_table
from .utils import get_paper_labels

logger = logging.getLogger(__name__)
MAX_RENDERED_TABLES_PER_SECTION = 4

# Common font search paths
FONT_SEARCH_PATHS = [
    # Linux system fonts
    Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
    Path("/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf"),
    Path("/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf"),
    # WenQuanYi CJK
    Path("/usr/share/fonts/truetype/wqy/wqy-microhei.ttc"),
    Path("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"),
    # Noto CJK
    Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
    Path("/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc"),
    # Bundled fonts
    Path(__file__).parent.parent / "fonts",
]


def _find_font(name_hints: list[str]) -> Path | None:
    """Find a font file matching any of the name hints."""
    for path in FONT_SEARCH_PATHS:
        if path.is_dir():
            for f in path.iterdir():
                fname = f.name.lower()
                for hint in name_hints:
                    if hint.lower() in fname:
                        return f
        elif path.is_file():
            fname = path.name.lower()
            for hint in name_hints:
                if hint.lower() in fname:
                    return path
    return None


class AcademicPaperPDF(FPDF):
    """FPDF subclass with academic paper header/footer."""

    def __init__(self, style: PDFStyleConfig, paper_title: str = ""):
        super().__init__()
        self.style = style
        self.paper_title = paper_title
        self._fonts_loaded = False
        self._has_cjk = False
        self._body_font = "helvetica"
        self._bold_font = "helvetica"

    def setup_fonts(self):
        """Load Unicode fonts if available."""
        if self._fonts_loaded:
            return

        # Try to find a CJK-capable font
        cjk_font = _find_font(["wqy", "notosans-cjk", "notosanscjk", "notoserif"])
        if cjk_font:
            try:
                self.add_font("cjk", "", str(cjk_font), uni=True)
                self.add_font("cjk", "B", str(cjk_font), uni=True)
                self._body_font = "cjk"
                self._bold_font = "cjk"
                self._has_cjk = True
                logger.info("Loaded CJK font: %s", cjk_font.name)
            except Exception as e:
                logger.warning("Failed to load CJK font %s: %s", cjk_font, e)

        # Try DejaVu for better Unicode coverage (if no CJK found)
        if not self._has_cjk:
            dejavu = _find_font(["dejavusans", "dejavu"])
            dejavu_bold = _find_font(["dejavusans-bold", "dejavu"])
            if dejavu:
                try:
                    self.add_font("dejavu", "", str(dejavu), uni=True)
                    bold_path = str(dejavu_bold) if dejavu_bold else str(dejavu)
                    self.add_font("dejavu", "B", bold_path, uni=True)
                    self._body_font = "dejavu"
                    self._bold_font = "dejavu"
                    logger.info("Loaded DejaVu font")
                except Exception as e:
                    logger.warning("Failed to load DejaVu: %s", e)

        self._fonts_loaded = True

    def header(self):
        if self.page_no() > 1 and self.paper_title:
            self.set_font(self._body_font, "", 8)
            self.set_text_color(128, 128, 128)
            self.cell(0, 8, self.paper_title[:80], align="C",
                      new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            self.line(
                self.l_margin, self.get_y(),
                self.w - self.r_margin, self.get_y()
            )
            self.ln(3)
            self.set_text_color(0, 0, 0)

    def footer(self):
        if self.style.include_page_numbers:
            self.set_y(-15)
            self.set_font(self._body_font, "", 8)
            self.set_text_color(128, 128, 128)
            self.cell(0, 10, str(self.page_no()), align="C")
            self.set_text_color(0, 0, 0)


def render_paper(
    paper: dict[str, Any],
    images: dict[str, bytes],
    style: PDFStyleConfig,
    output_path: str | Path | None = None,
) -> Path:
    """
    Render a structured paper dict to PDF.

    Args:
        paper: Structured paper data from paper_writer.
        images: Dict of image_name -> image_bytes.
        style: PDF styling configuration.
        output_path: Where to save the PDF. If None, uses a temp file.

    Returns:
        Path to the generated PDF file.
    """
    if output_path is None:
        output_path = Path(tempfile.mktemp(suffix=".pdf"))
    else:
        output_path = Path(output_path)

    title = paper.get("title", "Untitled Paper")
    labels = get_paper_labels(paper)
    pdf = AcademicPaperPDF(style, paper_title=title)
    pdf.setup_fonts()

    margin = style.margin_mm
    pdf.set_margins(margin, margin, margin)
    pdf.set_auto_page_break(auto=True, margin=margin)
    pdf.add_page()

    body_font = pdf._body_font
    line_h = style.font_size_body * 0.5 * style.line_spacing

    # ── Title ──
    pdf.set_font(body_font, "B", style.font_size_title)
    pdf.multi_cell(0, style.font_size_title * 0.6, title, align="C",
                   new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(4)

    # ── Authors ──
    authors = paper.get("authors", [])
    if authors:
        pdf.set_font(body_font, "", style.font_size_body + 1)
        authors_text = ", ".join(authors) if isinstance(authors, list) else str(authors)
        pdf.multi_cell(0, line_h, authors_text, align="C",
                       new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.ln(4)

    # ── Abstract ──
    abstract = paper.get("abstract", "")
    if abstract:
        pdf.ln(2)
        pdf.set_font(body_font, "B", style.font_size_body)
        pdf.cell(0, line_h, labels["abstract"], new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.ln(1)

        # Indent abstract
        old_l = pdf.l_margin
        old_r = pdf.r_margin
        pdf.set_left_margin(old_l + 10)
        pdf.set_right_margin(old_r + 10)
        pdf.set_x(old_l + 10)
        pdf.set_font(body_font, "", style.font_size_body - 1)
        pdf.multi_cell(0, line_h * 0.9, abstract, align="J",
                       new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_left_margin(old_l)
        pdf.set_right_margin(old_r)
        pdf.ln(2)

    # ── Keywords ──
    keywords = paper.get("keywords", [])
    if keywords:
        pdf.set_font(body_font, "B", style.font_size_body - 1)
        kw_text = f"{labels['keywords']}: "
        pdf.set_x(margin + 10)
        pdf.cell(pdf.get_string_width(kw_text), line_h, kw_text)
        pdf.set_font(body_font, "", style.font_size_body - 1)
        pdf.multi_cell(0, line_h * 0.9, ", ".join(keywords),
                       new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.ln(4)

    # ── Divider ──
    _draw_divider(pdf, margin)
    pdf.ln(4)

    # ── Sections ──
    figure_counter = 0
    table_counter = 0

    for section in paper.get("sections", []):
        heading = section.get("heading", "")
        level = section.get("level", 1)
        paragraphs = section.get("paragraphs", [])
        figures = section.get("figures", [])
        tables = section.get("tables", [])

        # Section heading
        if heading:
            if level == 1:
                pdf.ln(4)
                pdf.set_font(body_font, "B", style.font_size_section)
                pdf.multi_cell(0, style.font_size_section * 0.55, heading,
                               new_x=XPos.LMARGIN, new_y=YPos.NEXT)
                pdf.ln(2)
            else:
                pdf.ln(2)
                pdf.set_font(body_font, "B", style.font_size_subsection)
                pdf.multi_cell(0, style.font_size_subsection * 0.55, heading,
                               new_x=XPos.LMARGIN, new_y=YPos.NEXT)
                pdf.ln(1)

        # Paragraphs
        for para in paragraphs:
            if not para or not para.strip():
                continue
            table_counter = _render_markdownish_paragraph(
                pdf=pdf,
                paragraph=para,
                table_counter=table_counter,
                style=style,
                body_font=body_font,
                line_h=line_h,
                margin=margin,
                labels=labels,
            )

        # Figures
        for fig in figures:
            figure_counter += 1
            filename = fig.get("filename", "")
            caption = fig.get("caption", "")

            img_data = images.get(filename)
            if img_data:
                _render_figure(pdf, img_data, caption, figure_counter, style, margin, labels)
            else:
                # Image not available, render placeholder
                pdf.ln(2)
                pdf.set_font(body_font, "", style.font_size_body - 1)
                placeholder = (
                    f"[{labels['figure']} {figure_counter}"
                    f"{': ' + caption if caption else ''} - "
                    f"{labels['image_not_available'].format(filename=filename)}]"
                )
                pdf.multi_cell(0, line_h, placeholder, align="C",
                               new_x=XPos.LMARGIN, new_y=YPos.NEXT)
                pdf.ln(2)

        # Tables
        for tbl in tables[:MAX_RENDERED_TABLES_PER_SECTION]:
            table_counter += 1
            _render_table(pdf, tbl, table_counter, style, body_font, line_h, margin, labels)
        if len(tables) > MAX_RENDERED_TABLES_PER_SECTION:
            omitted = len(tables) - MAX_RENDERED_TABLES_PER_SECTION
            pdf.set_font(body_font, "", max(style.font_size_body - 2, 8))
            pdf.set_text_color(110, 110, 110)
            pdf.multi_cell(
                0,
                line_h * 0.9,
                f"Detailed result tables omitted from the PDF preview ({omitted} more). "
                "Use the downloadable CSV files for the full results.",
                new_x=XPos.LMARGIN,
                new_y=YPos.NEXT,
            )
            pdf.set_text_color(0, 0, 0)
            pdf.ln(2)

    # ── References ──
    references = paper.get("references", [])
    if references:
        pdf.ln(4)
        _draw_divider(pdf, margin)
        pdf.ln(4)
        pdf.set_font(body_font, "B", style.font_size_section)
        pdf.cell(0, style.font_size_section * 0.55, labels["references"],
                 new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.ln(3)

        pdf.set_font(body_font, "", style.font_size_body - 2)
        ref_line_h = (style.font_size_body - 2) * 0.5 * style.line_spacing
        for ref in references:
            ref_text = ref.strip() if isinstance(ref, str) else str(ref)
            if ref_text:
                pdf.set_x(margin + 5)
                pdf.multi_cell(
                    pdf.w - 2 * margin - 5, ref_line_h,
                    ref_text, align="L",
                    new_x=XPos.LMARGIN, new_y=YPos.NEXT,
                )
                pdf.ln(0.5)

    # Save
    pdf.output(str(output_path))
    logger.info("PDF saved to %s", output_path)
    return output_path


def _draw_divider(pdf: FPDF, margin: float):
    """Draw a horizontal divider line."""
    y = pdf.get_y()
    pdf.set_draw_color(180, 180, 180)
    pdf.line(margin, y, pdf.w - margin, y)
    pdf.set_draw_color(0, 0, 0)


def _render_figure(
    pdf: AcademicPaperPDF,
    img_data: bytes,
    caption: str,
    fig_num: int,
    style: PDFStyleConfig,
    margin: float,
    labels: dict[str, str],
):
    """Render a figure with caption into the PDF."""
    pdf.ln(3)

    # Write image to temp file for fpdf2
    try:
        suffix = _guess_image_suffix(img_data)
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(img_data)
            tmp_path = tmp.name

        # Calculate available width and max height
        avail_w = pdf.w - 2 * margin
        max_img_w = avail_w * 0.8  # 80% of text width
        max_img_h = 120  # mm

        # Check if we need a new page
        if pdf.get_y() + max_img_h + 20 > pdf.h - margin:
            pdf.add_page()

        # Center the image
        pdf.image(
            tmp_path,
            x=(pdf.w - max_img_w) / 2,
            w=max_img_w,
            h=0,  # Auto height
        )

        # Clean up temp file
        Path(tmp_path).unlink(missing_ok=True)

    except Exception as e:
        logger.warning("Failed to render figure %d: %s", fig_num, e)
        pdf.set_font(pdf._body_font, "", style.font_size_body - 1)
        pdf.cell(0, 5, f"[{labels['figure']} {fig_num}: {labels['image_render_failed']}]", align="C",
                 new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    # Caption
    pdf.ln(2)
    pdf.set_font(pdf._body_font, "", style.font_size_body - 1)
    caption = caption.strip()
    caption_text = (
        f"{labels['figure']} {fig_num}. {caption}"
        if caption else f"{labels['figure']} {fig_num}"
    )
    pdf.multi_cell(0, style.font_size_body * 0.45, caption_text, align="C",
                   new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(3)


def _render_table(
    pdf: AcademicPaperPDF,
    tbl: dict,
    tbl_num: int,
    style: PDFStyleConfig,
    body_font: str,
    line_h: float,
    margin: float,
    labels: dict[str, str],
):
    """Render a table with caption."""
    caption = str(tbl.get("caption", "")).strip()
    headers = tbl.get("headers", [])
    rows = tbl.get("rows", [])

    if not headers and not rows:
        return

    pdf.ln(3)


def _render_markdownish_paragraph(
    *,
    pdf: AcademicPaperPDF,
    paragraph: str,
    table_counter: int,
    style: PDFStyleConfig,
    body_font: str,
    line_h: float,
    margin: float,
    labels: dict[str, str],
) -> int:
    lines = paragraph.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    paragraph_buffer: list[str] = []
    table_lines: list[str] = []
    pending_table_caption = ""

    def flush_text() -> None:
        nonlocal paragraph_buffer
        cleaned_lines = [_strip_inline_markdown(line) for line in paragraph_buffer]
        cleaned_lines = [line for line in cleaned_lines if line]
        if not cleaned_lines:
            paragraph_buffer = []
            return
        pdf.set_font(body_font, "", style.font_size_body)
        pdf.set_x(margin + 5)
        pdf.multi_cell(
            pdf.w - 2 * margin - 5,
            line_h,
            " ".join(cleaned_lines),
            align="J",
            new_x=XPos.LMARGIN,
            new_y=YPos.NEXT,
        )
        pdf.ln(1)
        paragraph_buffer = []

    def flush_table() -> None:
        nonlocal table_lines, table_counter, pending_table_caption
        if not table_lines:
            return
        parsed = _parse_markdown_table(table_lines)
        table_lines = []
        if not parsed:
            return
        if pending_table_caption:
            parsed["caption"] = pending_table_caption
        table_counter += 1
        _render_table(pdf, parsed, table_counter, style, body_font, line_h, margin, labels)
        pending_table_caption = ""

    for raw_line in lines:
        stripped = raw_line.strip()
        if not stripped:
            flush_text()
            flush_table()
            continue

        if stripped.startswith("|") and stripped.endswith("|"):
            flush_text()
            table_lines.append(stripped)
            continue

        flush_table()

        heading_match = re.match(r"^(#{2,6})\s+(.*)$", stripped)
        if heading_match:
            flush_text()
            marks, title = heading_match.groups()
            clean_title = _strip_inline_markdown(title)
            pending_table_caption = clean_title
            _render_inline_heading(pdf, clean_title, len(marks), style, body_font, margin)
            continue

        bullet_match = re.match(r"^[-*]\s+(.*)$", stripped)
        if bullet_match:
            flush_text()
            _render_list_item(
                pdf,
                f"• {_strip_inline_markdown(bullet_match.group(1))}",
                style,
                body_font,
                line_h,
                margin,
            )
            continue

        ordered_match = re.match(r"^(\d+)\.\s+(.*)$", stripped)
        if ordered_match:
            flush_text()
            number, content = ordered_match.groups()
            _render_list_item(
                pdf,
                f"{number}. {_strip_inline_markdown(content)}",
                style,
                body_font,
                line_h,
                margin,
            )
            continue

        paragraph_buffer.append(stripped)

    flush_text()
    flush_table()
    return table_counter


def _render_inline_heading(
    pdf: AcademicPaperPDF,
    text: str,
    level: int,
    style: PDFStyleConfig,
    body_font: str,
    margin: float,
) -> None:
    clean_text = text.strip()
    if not clean_text:
        return
    size = style.font_size_subsection if level >= 3 else style.font_size_section
    pdf.ln(2)
    pdf.set_font(body_font, "B", size)
    pdf.set_x(margin)
    pdf.multi_cell(0, size * 0.55, clean_text, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(1)


def _render_list_item(
    pdf: AcademicPaperPDF,
    text: str,
    style: PDFStyleConfig,
    body_font: str,
    line_h: float,
    margin: float,
) -> None:
    clean_text = text.strip()
    if not clean_text:
        return
    pdf.set_font(body_font, "", style.font_size_body)
    pdf.set_x(margin + 6)
    pdf.multi_cell(
        pdf.w - 2 * margin - 6,
        line_h,
        clean_text,
        align="L",
        new_x=XPos.LMARGIN,
        new_y=YPos.NEXT,
    )
    pdf.ln(0.5)


def _strip_inline_markdown(text: str) -> str:
    value = str(text or "").strip()
    if not value:
        return ""
    value = re.sub(r"!\[([^\]]*)\]\([^)]+\)", r"\1", value)
    value = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", value)
    value = value.replace("**", "").replace("__", "")
    value = value.replace("`", "")
    value = re.sub(r"^\s*#+\s*", "", value)
    value = re.sub(r"\s+", " ", value)
    return value.strip()

    # Table caption (above table)
    pdf.set_font(body_font, "B", style.font_size_body - 1)
    caption_text = (
        f"{labels['table']} {tbl_num}. {caption}"
        if caption else f"{labels['table']} {tbl_num}"
    )
    pdf.multi_cell(0, line_h * 0.9, caption_text, align="C",
                   new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(2)

    all_cols = max(len(headers), max((len(r) for r in rows), default=0)) if (headers or rows) else 1
    avail_w = pdf.w - 2 * margin
    table_font_size = max(style.font_size_body - (2 if all_cols >= 6 else 1), 8)
    pdf.set_font(body_font, "", table_font_size)
    col_widths = _calculate_table_widths(pdf, headers, rows, avail_w)

    cell_h = max(line_h * 0.8, table_font_size * 0.55 + 2.8)
    start_x = pdf.l_margin
    body_alignments = [
        _table_cell_alignment(
            next(
                (row[idx] for row in rows if idx < len(row) and str(row[idx]).strip()),
                "",
            )
        )
        for idx in range(all_cols)
    ]

    def draw_header() -> None:
        if not headers:
            return
        pdf.set_font(body_font, "B", table_font_size)
        pdf.set_fill_color(236, 240, 245)
        current_y = pdf.get_y()
        x = start_x
        for j, header in enumerate(headers):
            width = col_widths[j]
            pdf.set_xy(x, current_y)
            pdf.cell(
                width,
                cell_h,
                _clip_table_text(str(header), 22),
                border=1,
                fill=True,
                align="C",
            )
            x += width
        pdf.set_y(current_y + cell_h)

    draw_header()

    pdf.set_font(body_font, "", table_font_size)
    for row_idx, row in enumerate(rows):
        if pdf.get_y() + cell_h > pdf.h - margin:
            pdf.add_page()
            draw_header()
            pdf.set_font(body_font, "", table_font_size)

        fill = row_idx % 2 == 1
        pdf.set_fill_color(250, 250, 252)
        current_y = pdf.get_y()
        x = start_x
        for j in range(all_cols):
            cell_val = str(row[j]) if j < len(row) else ""
            width = col_widths[j]
            pdf.set_xy(x, current_y)
            pdf.cell(
                width,
                cell_h,
                _clip_table_text(cell_val, 20 if all_cols <= 5 else 16),
                border=1,
                align=body_alignments[j],
                fill=fill,
            )
            x += width
        pdf.set_y(current_y + cell_h)

    preview_note = _table_preview_note(tbl)
    if preview_note:
        pdf.ln(1)
        pdf.set_font(body_font, "", max(table_font_size - 1, 7))
        pdf.set_text_color(110, 110, 110)
        pdf.multi_cell(
            0,
            line_h * 0.8,
            preview_note,
            new_x=XPos.LMARGIN,
            new_y=YPos.NEXT,
        )
        pdf.set_text_color(0, 0, 0)

    pdf.ln(3)


def _calculate_table_widths(
    pdf: AcademicPaperPDF,
    headers: list[Any],
    rows: list[list[Any]],
    avail_w: float,
) -> list[float]:
    all_cols = max(len(headers), max((len(r) for r in rows), default=0)) if (headers or rows) else 1
    sample_rows = rows[: min(len(rows), 8)]
    widths: list[float] = []
    for idx in range(all_cols):
        samples = [str(headers[idx]) if idx < len(headers) else ""]
        for row in sample_rows:
            samples.append(str(row[idx]) if idx < len(row) else "")
        measured = max(pdf.get_string_width(_clip_table_text(text, 24)) for text in samples) + 6
        max_width = 42 if idx > 1 else 52
        widths.append(min(max(measured, 16), max_width))

    total = sum(widths)
    if total <= 0:
        return [avail_w / all_cols for _ in range(all_cols)]
    if total > avail_w:
        scale = avail_w / total
        widths = [max(12.0, width * scale) for width in widths]
        overflow = sum(widths) - avail_w
        if overflow > 0:
            shrink_room = sum(max(0.0, width - 12.0) for width in widths)
            if shrink_room > 0:
                widths = [
                    width - overflow * (max(0.0, width - 12.0) / shrink_room)
                    for width in widths
                ]
    else:
        widths[0] += avail_w - total
    return widths


def _clip_table_text(text: str, max_chars: int) -> str:
    value = str(text or "").strip()
    if len(value) <= max_chars:
        return value
    return value[: max_chars - 1] + "…"


def _table_cell_alignment(value: str) -> str:
    normalized = str(value or "").strip().replace(",", "")
    if not normalized:
        return "C"
    try:
        float(normalized)
    except ValueError:
        return "L"
    return "R"


def _table_preview_note(tbl: dict[str, Any]) -> str:
    source_rows = int(tbl.get("source_rows") or len(tbl.get("rows", [])))
    source_columns = int(tbl.get("source_columns") or len(tbl.get("headers", [])))
    preview_rows = len(tbl.get("rows", []))
    preview_columns = len(tbl.get("headers", []))
    filename = str(tbl.get("filename", "")).strip()

    notes: list[str] = []
    if source_rows > preview_rows:
        notes.append(f"{preview_rows} of {source_rows} rows shown")
    if source_columns > preview_columns:
        notes.append(f"{preview_columns} of {source_columns} columns shown")

    if not notes:
        return ""

    note = "Table preview: " + ", ".join(notes) + "."
    if filename:
        note += f" Full data remains in {filename}."
    return note


def _guess_image_suffix(data: bytes) -> str:
    """Guess image format from magic bytes."""
    if data[:8] == b'\x89PNG\r\n\x1a\n':
        return ".png"
    if data[:2] == b'\xff\xd8':
        return ".jpg"
    if data[:4] == b'GIF8':
        return ".gif"
    if data[:4] == b'BM':
        return ".bmp"
    return ".png"  # default
