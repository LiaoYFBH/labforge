"""PaperForge - Document to Academic Paper PDF Converter."""

from __future__ import annotations

__version__ = "0.3.0"

from .latex_renderer import render_paper_to_tex
from .bundle_io import load_paper_bundle, normalize_paper_bundle, paper_bundle_to_markdown
from .pdf_compiler import LatexCompileError, compile_tex_to_pdf
from .pdf_quality_agent import (
    PdfQualityIssue,
    PdfQualityReport,
    format_quality_summary,
    inspect_paper_artifacts,
    write_quality_report,
)
from .templates_catalog import (
    PaperTemplate,
    PAPER_TEMPLATES,
    ensure_assets,
    get_template,
    list_templates,
    template_choices,
)


def render_paper_pdf(
    paper,
    images=None,
    output_path=None,
    template_dir=None,
    template_name="article.tex.j2",
    template_key: str | None = None,
    line_spacing: float = 1.25,
    margin_mm: int = 25,
    keep_tex: bool = True,
    engine: str | None = None,
    forced_language: str | None = None,
):
    """
    High-level helper: render a paper dict to PDF via LaTeX.

    ``output_path`` is the desired final PDF path. The .tex source and
    figure files are written next to it (so relative paths inside the
    .tex resolve). If ``keep_tex`` is False, the auxiliary files are
    cleaned up afterwards.

    If ``template_key`` is provided, it overrides ``template_name`` and
    additionally pulls down any conference-specific .sty/.cls assets next
    to the rendered .tex so xelatex can find them at compile time.

    Returns the final PDF path.
    """
    from pathlib import Path
    import shutil

    if output_path is None:
        raise ValueError("output_path is required")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    work_dir = output_path.parent / f".{output_path.stem}_latex"
    work_dir.mkdir(parents=True, exist_ok=True)

    template: PaperTemplate | None = None
    if template_key is not None:
        template = get_template(template_key)
        template_name = template.template_filename
        # Pull in conference-specific style files (best-effort).
        ensure_assets(template, work_dir)

    tex_path = render_paper_to_tex(
        paper=paper,
        images=images or {},
        output_dir=work_dir,
        template_dir=template_dir,
        template_name=template_name,
        line_spacing=line_spacing,
        margin_mm=margin_mm,
        forced_language=forced_language,
    )

    produced_pdf = compile_tex_to_pdf(tex_path, engine=engine)
    shutil.move(str(produced_pdf), str(output_path))
    log_path = tex_path.with_suffix(".log")
    quality_report = inspect_paper_artifacts(
        tex_path=tex_path,
        log_path=log_path,
        target_language=forced_language,
    )
    write_quality_report(quality_report, work_dir / "pdf_quality_report.json")

    # Discard the downloaded remote-image cache: the bytes are now baked
    # into the .tex's figures/ directory (or the PDF), so there's no reason
    # to keep the signed-URL temporaries. With keep_tex=True the user
    # still gets paper.tex + figures/, just not the raw download dump.
    remote_cache = work_dir / ".remote_images"
    if remote_cache.exists():
        shutil.rmtree(remote_cache, ignore_errors=True)

    # Surface the .tex source + figures/ next to the PDF so users can
    # tweak rendering issues by hand without spelunking into the hidden
    # ``.{stem}_latex/`` work dir. Filename mirrors the PDF stem so it's
    # obvious which .tex produced which PDF.
    if keep_tex:
        try:
            visible_tex = output_path.with_suffix(".tex")
            shutil.copyfile(str(tex_path), str(visible_tex))
            # Also copy figures/ so the visible .tex compiles standalone
            # if the user takes it out and runs xelatex by hand.
            src_figs = work_dir / "figures"
            if src_figs.is_dir():
                dst_figs = output_path.parent / "figures"
                if dst_figs.exists():
                    # Refresh: replace stale copy
                    shutil.rmtree(dst_figs, ignore_errors=True)
                shutil.copytree(str(src_figs), str(dst_figs))
            # Also drop the .log next to it so users can read xelatex
            # warnings without finding the hidden dir.
            if log_path.exists():
                shutil.copyfile(str(log_path), str(output_path.with_suffix(".log")))
        except OSError:
            # Filesystem hiccup is non-fatal: the PDF + hidden work_dir
            # still exist, the visible copy is just a convenience.
            pass

    if not keep_tex:
        shutil.rmtree(work_dir, ignore_errors=True)

    return output_path


__all__ = [
    "__version__",
    "render_paper_to_tex",
    "load_paper_bundle",
    "normalize_paper_bundle",
    "paper_bundle_to_markdown",
    "compile_tex_to_pdf",
    "render_paper_pdf",
    "LatexCompileError",
    "PdfQualityIssue",
    "PdfQualityReport",
    "format_quality_summary",
    "inspect_paper_artifacts",
    "write_quality_report",
    "PaperTemplate",
    "PAPER_TEMPLATES",
    "ensure_assets",
    "get_template",
    "list_templates",
    "template_choices",
]
