"""
Compile a rendered .tex file to PDF via xelatex/latexmk.

The compiler runs the chosen engine in the directory that holds the
.tex file (so relative ``\\includegraphics`` paths resolve), captures
logs for error reporting, and returns the path of the produced PDF.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)


class LatexCompileError(RuntimeError):
    """Raised when the LaTeX engine exits non-zero."""

    def __init__(self, message: str, log_tail: str = ""):
        super().__init__(message)
        self.log_tail = log_tail


def _find_engine(preferred: str | None = None) -> str:
    candidates: list[str] = []
    if preferred:
        candidates.append(preferred)
    candidates.extend(["latexmk", "xelatex", "pdflatex"])
    for name in candidates:
        if shutil.which(name):
            return name
    raise LatexCompileError(
        "No LaTeX engine found. Install TeX Live (xelatex / latexmk / pdflatex)."
    )


def _tail(text: str, max_lines: int = 60) -> str:
    lines = text.splitlines()
    if len(lines) <= max_lines:
        return text
    return "\n".join(lines[-max_lines:])


def compile_tex_to_pdf(
    tex_path: str | Path,
    engine: str | None = None,
    runs: int = 2,
    timeout: int = 180,
) -> Path:
    """
    Compile ``tex_path`` to PDF.

    Args:
        tex_path: Path to a .tex file.
        engine: Preferred engine (latexmk/xelatex/pdflatex). Auto-detected otherwise.
        runs: Number of engine invocations for non-latexmk engines (for refs).
        timeout: Per-invocation timeout in seconds.

    Returns:
        Path to the produced PDF.
    """
    tex_path = Path(tex_path).resolve()
    if not tex_path.exists():
        raise FileNotFoundError(tex_path)

    work_dir = tex_path.parent
    stem = tex_path.stem
    engine_name = _find_engine(engine)
    logger.info("Compiling %s with %s", tex_path.name, engine_name)

    if engine_name == "latexmk":
        cmds = [[
            "latexmk",
            "-xelatex",
            "-interaction=nonstopmode",
            "-halt-on-error",
            "-file-line-error",
            f"-output-directory={work_dir}",
            tex_path.name,
        ]]
    else:
        single = [
            engine_name,
            "-interaction=nonstopmode",
            "-halt-on-error",
            "-file-line-error",
            tex_path.name,
        ]
        cmds = [single for _ in range(max(1, runs))]

    combined_log = ""
    for cmd in cmds:
        try:
            proc = subprocess.run(
                cmd,
                cwd=str(work_dir),
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as exc:
            raise LatexCompileError(
                f"LaTeX compilation timed out after {timeout}s",
                log_tail=_tail(exc.stdout or "") + "\n" + _tail(exc.stderr or ""),
            ) from exc

        combined_log += proc.stdout + proc.stderr
        if proc.returncode != 0:
            log_file = work_dir / f"{stem}.log"
            log_tail = combined_log
            if log_file.exists():
                log_tail = _tail(log_file.read_text(encoding="utf-8", errors="replace"))
            raise LatexCompileError(
                f"{engine_name} failed with exit code {proc.returncode}",
                log_tail=log_tail,
            )

    pdf_path = work_dir / f"{stem}.pdf"
    if not pdf_path.exists():
        raise LatexCompileError(
            f"{engine_name} finished but {pdf_path.name} was not produced",
            log_tail=_tail(combined_log),
        )

    logger.info("PDF produced at %s", pdf_path)
    return pdf_path
