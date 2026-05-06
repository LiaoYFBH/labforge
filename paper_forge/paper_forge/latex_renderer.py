"""
LaTeX renderer for paper dicts.

Converts a structured paper (the same schema produced by paper_writer)
into a .tex file using a Jinja2 template. The rendered .tex is written
to an output directory alongside any referenced image files so that
xelatex can compile it in place.
"""

from __future__ import annotations

import logging
import re
import shutil
from pathlib import Path
from typing import Any, Iterable

from jinja2 import Environment, FileSystemLoader, StrictUndefined

logger = logging.getLogger(__name__)

DEFAULT_TEMPLATE_DIR = Path(__file__).resolve().parent.parent / "templates"
DEFAULT_TEMPLATE = "article.tex.j2"

_CJK_PATTERN = re.compile(r"[\u3000-\u9fff\uff00-\uffef]")
_SUPPORTED_FIGURE_SUFFIXES = {".pdf", ".png", ".jpg", ".jpeg", ".bmp"}
_FIGURE_SUFFIX_PREFERENCE = {
    ".pdf": 0,
    ".png": 1,
    ".jpg": 2,
    ".jpeg": 2,
    ".bmp": 3,
}

# LaTeX special characters that need escaping in text content.
#
# IMPORTANT: this map is consumed by a *single-pass* regex substitution in
# ``escape_latex``. The previous implementation looped ``str.replace`` over
# the table sequentially; that was wrong because the replacement for ``\\``
# is ``\textbackslash{}`` — which contains ``{`` and ``}`` that the later
# rules then re-escaped. ``\theta`` would walk through:
#   ``\theta`` → ``\textbackslash{}theta`` → ``\textbackslash\{}theta``
#   → ``\textbackslash\{\}theta`` and render in the PDF as ``\{}theta``.
# The single-pass regex below visits each source char exactly once, so the
# replacement strings are never themselves re-scanned.
_LATEX_CHAR_MAP = {
    "\\": r"\textbackslash{}",
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
    "<": r"\textless{}",
    ">": r"\textgreater{}",
}
_LATEX_ESCAPE_RE = re.compile(r"[\\&%$#_{}~^<>]")


def escape_latex(text: Any) -> str:
    """Escape a string so it can be safely embedded in a LaTeX document.

    This is a pure character-level escape: every ``\\``, ``{``, ``}`` etc.
    becomes its LaTeX-safe equivalent, regardless of whether the source
    looked like a LaTeX command. Callers that want to preserve specific
    commands (math blocks, ``\\cite``, ``\\ref``, etc.) MUST pull those
    aside via ``_protect_math`` before calling this.
    """
    if text is None:
        return ""
    return _LATEX_ESCAPE_RE.sub(
        lambda m: _LATEX_CHAR_MAP[m.group(0)], str(text)
    )


def _contains_cjk(*values: Any) -> bool:
    for v in values:
        if v and _CJK_PATTERN.search(str(v)):
            return True
    return False


# Markdown inline patterns. Evaluated left-to-right; the first matching
# pattern on a line wins for that span. Ordering matters: `**bold**` must be
# checked before `*italic*` so `**` is not misread as "open italic then close".
_INLINE_SENTINEL = "\0INNER\0"
_MD_INLINE_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    # Fenced inline code: `x` -> \texttt{x}
    (re.compile(r"`([^`\n]+?)`"), "\\texttt{" + _INLINE_SENTINEL + "}"),
    # **bold**
    (re.compile(r"\*\*([^*\n]+?)\*\*"), "\\textbf{" + _INLINE_SENTINEL + "}"),
    (re.compile(r"__([^_\n]+?)__"), "\\textbf{" + _INLINE_SENTINEL + "}"),
    # *italic* / _italic_  (single, non-greedy)
    (re.compile(r"(?<!\*)\*([^*\n]+?)\*(?!\*)"), "\\textit{" + _INLINE_SENTINEL + "}"),
    (re.compile(r"(?<!_)_([^_\n]+?)_(?!_)"), "\\textit{" + _INLINE_SENTINEL + "}"),
    # [text](url) -> \href{url}{text}
    (re.compile(r"\[([^\]\n]+?)\]\(([^)\n]+?)\)"), "LINK"),
]

_MD_HEADING_PREFIX_RE = re.compile(r"^\s{0,3}(#{1,6})\s+")
_MD_NUMERIC_PREFIX_RE = re.compile(r"^\s*\d+(?:\.\d+)*[.)]?\s+")
_MD_UNORDERED_BULLET_RE = re.compile(r"^\s*[-*+]\s+")
_MD_ORDERED_BULLET_RE = re.compile(r"^\s*\d+[.)]\s+")
_MD_TABLE_ROW_RE = re.compile(r"^\s*\|(.+)\|\s*$")
# Separator row: |---|:---:|---:| — dashes with optional colons, pipes around.
_MD_TABLE_SEP_RE = re.compile(r"^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)+\|?\s*$")

# Math placeholders. We pull math expressions out before running
# ``escape_latex`` so the dollar signs and backslashes inside them aren't
# mangled (escape would turn ``$x$`` into ``\$x\$`` and the formula would
# render as literal ``$x$`` instead of typeset math). The placeholder uses
# a NUL-byte sentinel that can't appear in real input.
_MATH_PLACEHOLDER_PREFIX = "\0MATH\0"
_MATH_ENV_RE = re.compile(
    r"\\begin\{(equation\*?|align\*?|gather\*?|multline\*?)\}[\s\S]+?\\end\{\1\}"
)
_BRACKET_DISPLAY_MATH_RE = re.compile(r"\\\[([\s\S]+?)\\\]")
_PAREN_INLINE_MATH_RE = re.compile(r"\\\(([\s\S]+?)\\\)")
_DISPLAY_MATH_RE = re.compile(r"\$\$([\s\S]+?)\$\$")
# Inline math: a non-empty payload between single ``$`` that isn't itself
# a display ``$$``. We require at least one non-whitespace char inside so
# stray dollars in prose (e.g. "$10") don't accidentally swallow text.
_INLINE_MATH_RE = re.compile(r"(?<!\$)\$([^\n$]+?)\$(?!\$)")
_LONG_MATH_MIN_CHARS = 72

# Last-line-of-defense for the LLM "self-escaped backslash" pattern
# ``\{}command`` (see paper_writer._undo_llm_self_escape for the full
# rationale). We replicate the same regex here, in the renderer's own
# entry point, so that ANY paragraph reaching the tex source — regardless
# of whether it came from the agent's generate_report path, paper_forge's
# LLM rewrite/expand path, or a user-edited bundle — has the pattern
# stripped before escape_latex would mangle it into the literal
# ``\textbackslash\{\}command`` string the user keeps seeing in the PDF.
# Idempotent: running it on already-clean text changes nothing.
_RENDERER_SELF_ESCAPED_BACKSLASH_RE = re.compile(
    r"\\\{\}(?=[A-Za-z\[\]\(\)\{\}\|,;:!\\])"
)


def _undo_llm_self_escape_at_render(text: str) -> str:
    """Renderer-side R1 guard. Mirrors paper_writer._undo_llm_self_escape."""
    if not text or "\\{}" not in text:
        return text
    return _RENDERER_SELF_ESCAPED_BACKSLASH_RE.sub(r"\\", text)


# R1.5: After self-escape removal, the LLM may have left literal LaTeX
# list environments (``\begin{itemize} \item ... \end{itemize}``) inline
# in prose. ``_protect_math`` doesn't recognize these as math, so without
# intervention they fall through to ``escape_latex`` and re-emerge as
# ``\textbackslash\{\}begin\{itemize\}`` in the final tex (the b7a6bd6f
# itemize-block bug). We rewrite them into Markdown list items so the
# renderer's regular list pipeline turns them back into proper LaTeX.
_LATEX_ITEMIZE_RE = re.compile(
    r"\\begin\{itemize\}([\s\S]+?)\\end\{itemize\}"
)
_LATEX_ENUMERATE_RE = re.compile(
    r"\\begin\{enumerate\}([\s\S]+?)\\end\{enumerate\}"
)
_LATEX_ITEM_SPLIT_RE = re.compile(r"\\item\s+")


def _latex_lists_to_markdown(text: str) -> str:
    """Convert LLM-emitted ``\\begin{itemize/enumerate}...\\end`` blocks to
    Markdown list lines so the markdown→LaTeX pipeline handles them safely.
    """
    if not text or "\\begin{" not in text:
        return text

    def _itemize_to_md(match: re.Match) -> str:
        body = match.group(1)
        parts = [p.strip() for p in _LATEX_ITEM_SPLIT_RE.split(body) if p.strip()]
        if not parts:
            return ""
        return "\n\n" + "\n".join("- " + p for p in parts) + "\n\n"

    def _enumerate_to_md(match: re.Match) -> str:
        body = match.group(1)
        parts = [p.strip() for p in _LATEX_ITEM_SPLIT_RE.split(body) if p.strip()]
        if not parts:
            return ""
        return "\n\n" + "\n".join(f"{i+1}. {p}" for i, p in enumerate(parts)) + "\n\n"

    text = _LATEX_ITEMIZE_RE.sub(_itemize_to_md, text)
    text = _LATEX_ENUMERATE_RE.sub(_enumerate_to_md, text)
    return text


# R2 (latex_renderer half): allowlist of LaTeX commands that may appear
# inline in prose paragraphs and whose argument we want to pass through
# verbatim instead of escaping. Text-formatting commands (textbf / textit
# / texttt / emph) are deliberately NOT here — paper_writer rewrites
# those into Markdown upstream, where the normal markdown→LaTeX pipeline
# handles them. The commands listed here have no Markdown equivalent and
# must reach the tex source untouched.
#
# Each entry can take one or two ``{...}`` groups with non-nested bodies.
# Rare nested cases (``\href{url}{text \textbf{x}}``) will fall through
# to escape; that's preferable to greedy matching that could swallow the
# wrong text.
_INLINE_LATEX_ALLOWLIST_NAMES: tuple[str, ...] = (
    "href", "url",                                      # links
    "cite", "citep", "citet", "citealp", "citeauthor",  # citations
    "ref", "eqref", "pageref", "label",                 # cross-refs
    "footnote", "footnotemark", "footnotetext",         # footnotes
)
_INLINE_LATEX_ALLOWLIST_RE = re.compile(
    r"\\(?:" + "|".join(_INLINE_LATEX_ALLOWLIST_NAMES) + r")"
    r"(?:\{[^{}\n]*\}){0,2}"
)

_CODE_FENCE_LINE_RE = re.compile(r"^\s*[`'\"]{0,2}```\s*([A-Za-z0-9_-]+)?\s*$")
# Strict Python-code-line detector. The previous version matched any line
# containing ` class`, ` from`, ` def`, ` for`, ` if`, etc. as a substring,
# which deleted ordinary academic prose (trajectory 203c6c5c lost the
# entire Conclusion section because "classification" contained " class"
# and "calls from [21]" contained " from"). Now each Python keyword is
# anchored at the beginning of the line AND followed by the syntax token
# the language requires, so "We classify…" / "calls from [21]" no longer
# match. The library-method markers (np., pd., plt.) are kept; they're
# unambiguous code signals and almost never appear in prose.
_CODE_LIKE_LINE_RE = re.compile(
    r"^\s*(?:"
    r"import\s+\w"
    r"|from\s+[\w.]+\s+import\s"
    r"|def\s+\w+\s*\("
    r"|class\s+\w+\s*[\(:]"
    r"|return\s"
    r"|for\s+\w+\s+in\s.+:\s*$"
    r"|while\s.+:\s*$"
    r"|if\s.+:\s*$"
    r"|assert\s+\w"
    r")"
    r"|np\.|pd\.|plt\.|torch\.|sklearn\.|\.append\(|==\s|:=\s|print\("
)


def _protect_math(text: str) -> tuple[str, list[str]]:
    """Pull ``$...$`` and ``$$...$$`` out of ``text`` so escape can't break them.

    Also pulls out the small set of inline LaTeX commands that paper_writer
    is allowed to emit straight into prose (links, citations, references,
    footnotes — see ``_INLINE_LATEX_ALLOWLIST_NAMES``). Without that, an
    LLM-emitted ``\\cite{kingma2014}`` in an Introduction paragraph gets
    escape-mangled into ``\\textbackslash\\{\\}cite\\{kingma2014\\}`` and
    the citation never resolves.

    Display math goes first (so the inline matcher doesn't bite into it).
    The allowlist commands are protected last, after math, so a
    ``\\cite{...}`` that happens to live inside a math block stays
    inside the math fragment rather than being double-protected.

    Returns the placeholder-bearing text and a list of raw LaTeX
    fragments. Restore via :func:`_restore_math`.
    """
    has_math = any(marker in text for marker in ("$", "\\(", "\\[", "\\begin{"))
    has_allowlist = bool(text) and any(
        f"\\{name}" in text for name in _INLINE_LATEX_ALLOWLIST_NAMES
    )
    if not text or (not has_math and not has_allowlist):
        return text, []

    raw_fragments: list[str] = []

    def _swap(match: re.Match[str], wrapper: tuple[str, str]) -> str:
        body = match.group(1).strip()
        if not body:
            return match.group(0)
        idx = len(raw_fragments)
        raw_fragments.append(wrapper[0] + body + wrapper[1])
        return f"{_MATH_PLACEHOLDER_PREFIX}{idx}\0"

    def _swap_raw(match: re.Match[str]) -> str:
        body = match.group(0).strip()
        if not body:
            return match.group(0)
        idx = len(raw_fragments)
        raw_fragments.append(body)
        return f"{_MATH_PLACEHOLDER_PREFIX}{idx}\0"

    if has_math:
        text = _MATH_ENV_RE.sub(_swap_raw, text)
        text = _BRACKET_DISPLAY_MATH_RE.sub(lambda m: _swap(m, ("\\[", "\\]")), text)
        text = _PAREN_INLINE_MATH_RE.sub(lambda m: _swap(m, ("\\(", "\\)")), text)
        text = _DISPLAY_MATH_RE.sub(lambda m: _swap(m, ("\\[", "\\]")), text)
        text = _INLINE_MATH_RE.sub(lambda m: _swap(m, ("\\(", "\\)")), text)
    if has_allowlist:
        text = _INLINE_LATEX_ALLOWLIST_RE.sub(_swap_raw, text)
    return text, raw_fragments


def _restore_math(text: str, raw_fragments: list[str]) -> str:
    """Replace math placeholders with their raw LaTeX bodies."""
    if not raw_fragments:
        return text
    pattern = re.compile(re.escape(_MATH_PLACEHOLDER_PREFIX) + r"(\d+)\0")

    def _restore(match: re.Match[str]) -> str:
        idx = int(match.group(1))
        if 0 <= idx < len(raw_fragments):
            return _fit_math_fragment(raw_fragments[idx])
        return match.group(0)

    return pattern.sub(_restore, text)


def _fit_math_fragment(fragment: str) -> str:
    """Constrain long math fragments to the current column width."""
    value = fragment.strip()
    body = ""
    if value.startswith("\\[") and value.endswith("\\]"):
        body = value[2:-2].strip()
    elif value.startswith("\\(") and value.endswith("\\)"):
        body = value[2:-2].strip()
    else:
        return value
    compact = re.sub(r"\s+", "", body)
    if len(compact) < _LONG_MATH_MIN_CHARS:
        return value
    if "\\begin{" in body or "\\end{" in body:
        return value
    return (
        "\\begin{center}\n"
        "\\resizebox{0.98\\linewidth}{!}{$\\displaystyle "
        + body
        + "$}\n"
        "\\end{center}"
    )


def _strip_markdown_code_artifacts(text: str) -> str:
    """Drop fenced code blocks and obvious code lines from prose paragraphs."""
    if not text:
        return ""
    lines = str(text).replace("\r\n", "\n").replace("\r", "\n").splitlines()
    cleaned: list[str] = []
    in_fence = False
    for raw_line in lines:
        stripped = raw_line.strip()
        if _CODE_FENCE_LINE_RE.match(stripped):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        if stripped.lower() in {"markdown", "python", "latex", "tex"}:
            continue
        if _CODE_LIKE_LINE_RE.search(stripped):
            continue
        cleaned.append(raw_line)
    return "\n".join(cleaned).strip()


def _split_md_table_row(line: str) -> list[str]:
    """Split ``| a | b | c |`` into ``['a', 'b', 'c']``."""
    inner = line.strip()
    if inner.startswith("|"):
        inner = inner[1:]
    if inner.endswith("|"):
        inner = inner[:-1]
    return [cell.strip() for cell in inner.split("|")]


def _markdown_inline_to_latex(text: str) -> str:
    """Convert a single line's markdown inline syntax into LaTeX commands.

    Plain-text runs between commands are escaped with ``escape_latex``; the
    emitted LaTeX commands themselves are left verbatim so they render as
    real bold / italic / typewriter / hyperlink output instead of literal
    ``**foo**`` / ``###`` / ``` `foo` ``` in the PDF.

    Math segments (``$x$`` / ``$$x$$``) are pulled aside before everything
    else so neither the markdown matcher nor ``escape_latex`` mangles them
    — they are restored verbatim as ``\\(x\\)`` / ``\\[x\\]`` at the end.
    """
    if not text:
        return ""

    text, math_fragments = _protect_math(text)

    tokens: list[tuple[str, str]] = []  # ("text", raw) | ("latex", rendered)

    def _append_text(chunk: str) -> None:
        if chunk:
            tokens.append(("text", chunk))

    cursor = 0
    while cursor < len(text):
        best: tuple[re.Match[str], str] | None = None
        for pattern, replacement in _MD_INLINE_PATTERNS:
            m = pattern.search(text, cursor)
            if m is None:
                continue
            if best is None or m.start() < best[0].start():
                best = (m, replacement)
        if best is None:
            _append_text(text[cursor:])
            break

        match, replacement = best
        if match.start() > cursor:
            _append_text(text[cursor:match.start()])

        if replacement == "LINK":
            label, url = match.group(1), match.group(2)
            rendered = (
                "\\href{" + escape_latex(url) + "}{" + escape_latex(label) + "}"
            )
        else:
            inner = escape_latex(match.group(1))
            rendered = replacement.replace(_INLINE_SENTINEL, inner)
        tokens.append(("latex", rendered))
        cursor = match.end()

    rendered_text = "".join(
        escape_latex(chunk) if kind == "text" else chunk
        for kind, chunk in tokens
    )
    return _restore_math(rendered_text, math_fragments)


def _flush_list(buffer: list[tuple[str, str]], out: list[str]) -> None:
    """Flush a buffered ordered/unordered list into LaTeX items."""
    if not buffer:
        return
    kind = buffer[0][0]
    env = "itemize" if kind == "ul" else "enumerate"
    out.append("\\begin{" + env + "}")
    for _, body in buffer:
        out.append("  \\item " + body)
    out.append("\\end{" + env + "}")
    buffer.clear()


def _paragraph_to_latex(text: str) -> str:
    """Convert a paragraph (possibly containing markdown) to LaTeX.

    Handles: inline bold/italic/code/links, heading markers that leaked into
    the paragraph body (``### Foo``), and bullet/ordered lists composed of
    consecutive ``- foo`` or ``1. foo`` lines.
    """
    if not text:
        return ""
    # Last-line-of-defense R1 guard. If the upstream cleaner is bypassed
    # for any reason — stale module cache, a third-party path that calls
    # ``_paragraph_to_latex`` directly without going through paper_writer,
    # a manually-edited bundle.json — strip the LLM self-escape pattern
    # here so the renderer never produces ``\textbackslash\{\}command``
    # in the final tex.
    text = _undo_llm_self_escape_at_render(text)
    # R1.5: rewrite literal \begin{itemize}/\begin{enumerate} blocks into
    # Markdown list lines so the markdown→LaTeX pipeline takes them.
    text = _latex_lists_to_markdown(text)
    stripped = _strip_markdown_code_artifacts(str(text)).strip()
    if not stripped:
        return ""

    lines = stripped.replace("\r\n", "\n").split("\n")
    out: list[str] = []
    list_buf: list[tuple[str, str]] = []
    plain_buf: list[str] = []

    def _flush_plain() -> None:
        if plain_buf:
            # Soft-wrapped source lines inside a single paragraph: join with
            # a space so LaTeX can re-flow the text. The earlier `\\`
            # behaviour turned every 80-column wrap into a hard line break,
            # which produced the "one word per line" ragged output the user
            # flagged. True paragraph breaks come from the *paragraph list*
            # the parser emits, not from newlines inside a paragraph.
            out.append(" ".join(plain_buf))
            plain_buf.clear()

    def _render_table(header_cells: list[str], body_rows: list[list[str]]) -> str:
        n = max(len(header_cells), *(len(r) for r in body_rows)) if body_rows else len(header_cells)
        if n == 0:
            return ""
        col_spec = "l" + "l" * (n - 1) if n > 1 else "l"

        def _pad(cells: list[str]) -> list[str]:
            padded = [_markdown_inline_to_latex(c) for c in cells]
            padded.extend([""] * (n - len(padded)))
            return padded[:n]

        lines_: list[str] = []
        lines_.append("\\begin{table}[H]")
        lines_.append("\\centering")
        lines_.append("\\small")
        lines_.append("\\begin{tabular}{" + col_spec + "}")
        lines_.append("\\toprule")
        if header_cells:
            lines_.append(" & ".join(_pad(header_cells)) + " \\\\")
            lines_.append("\\midrule")
        for row in body_rows:
            lines_.append(" & ".join(_pad(row)) + " \\\\")
        lines_.append("\\bottomrule")
        lines_.append("\\end{tabular}")
        lines_.append("\\end{table}")
        return "\n".join(lines_)

    i = 0
    while i < len(lines):
        raw_line = lines[i]
        line = raw_line.rstrip()
        stripped_line = line.strip()
        if not stripped_line:
            i += 1
            continue

        # Display math: ``$$ x $$`` on a single line.
        single_math = re.match(r"^\$\$\s*(.+?)\s*\$\$$", stripped_line)
        if single_math:
            _flush_list(list_buf, out)
            _flush_plain()
            out.append("\\[" + single_math.group(1).strip() + "\\]")
            i += 1
            continue

        # Horizontal rule mapping
        if stripped_line == "---":
            _flush_list(list_buf, out)
            _flush_plain()
            out.append(r"\vspace{1em} \hrule \vspace{1em}")
            i += 1
            continue

        # Display math: ``$$`` opener line, body lines, ``$$`` closer.
        if stripped_line == "$$":
            _flush_list(list_buf, out)
            _flush_plain()
            buf: list[str] = []
            j = i + 1
            closed = False
            while j < len(lines):
                inner = lines[j].rstrip().strip()
                if inner == "$$" or inner.endswith("$$"):
                    if inner != "$$":
                        buf.append(inner[:-2].rstrip())
                    closed = True
                    j += 1
                    break
                buf.append(lines[j])
                j += 1
            body = "\n".join(buf).strip()
            if closed and body:
                out.append("\\[" + body + "\\]")
                i = j
                continue
            # Fall through: dangling ``$$`` line, treat as plain text.

        # Markdown table detection: a pipe row followed by a separator row.
        if (
            _MD_TABLE_ROW_RE.match(line)
            and i + 1 < len(lines)
            and _MD_TABLE_SEP_RE.match(lines[i + 1].strip())
        ):
            _flush_list(list_buf, out)
            _flush_plain()
            header_cells = _split_md_table_row(line)
            j = i + 2
            body: list[list[str]] = []
            while j < len(lines) and _MD_TABLE_ROW_RE.match(lines[j].strip()):
                body.append(_split_md_table_row(lines[j]))
                j += 1
            out.append(_render_table(header_cells, body))
            i = j
            continue

        heading_match = _MD_HEADING_PREFIX_RE.match(line)
        if heading_match:
            _flush_list(list_buf, out)
            _flush_plain()
            level = len(heading_match.group(1))
            body = _markdown_inline_to_latex(line[heading_match.end():].strip())
            cmd = "\\subsubsection*" if level >= 3 else "\\subsubsection*"
            out.append(cmd + "{" + body + "}")
            i += 1
            continue

        ul_match = _MD_UNORDERED_BULLET_RE.match(line)
        if ul_match:
            _flush_plain()
            if list_buf and list_buf[0][0] != "ul":
                _flush_list(list_buf, out)
            body = _markdown_inline_to_latex(line[ul_match.end():].strip())
            list_buf.append(("ul", body))
            i += 1
            continue

        ol_match = _MD_ORDERED_BULLET_RE.match(line)
        if ol_match:
            _flush_plain()
            if list_buf and list_buf[0][0] != "ol":
                _flush_list(list_buf, out)
            body = _markdown_inline_to_latex(line[ol_match.end():].strip())
            list_buf.append(("ol", body))
            i += 1
            continue

        _flush_list(list_buf, out)
        plain_buf.append(_markdown_inline_to_latex(line.strip()))
        i += 1

    _flush_list(list_buf, out)
    _flush_plain()
    return "\n".join(out)


def _clean_heading(raw: str) -> str:
    """Remove leading ``#`` markers and numeric prefixes from a heading.

    Section numbers like ``1.`` or ``2.3`` are dropped because the LaTeX
    template emits numbering via ``\\section{}``; leaving them in produced
    visible double numbers such as ``1 1. Introduction``.
    """
    if not raw:
        return ""
    cleaned = raw.strip()
    m = _MD_HEADING_PREFIX_RE.match(cleaned)
    if m:
        cleaned = cleaned[m.end():].strip()
    m2 = _MD_NUMERIC_PREFIX_RE.match(cleaned)
    if m2:
        cleaned = cleaned[m2.end():].strip()
    return cleaned


def _prepare_table(tbl: dict) -> dict | None:
    headers = tbl.get("headers") or []
    rows = tbl.get("rows") or []
    if not headers and not rows:
        return None

    n_cols = max(len(headers), max((len(r) for r in rows), default=0))
    if n_cols == 0:
        return None

    col_spec = "l" + "r" * (n_cols - 1) if n_cols > 1 else "l"

    def _pad(row: Iterable[Any]) -> list[str]:
        values = [_markdown_inline_to_latex(str(v)) for v in row]
        values.extend([""] * (n_cols - len(values)))
        return values[:n_cols]

    header_row = " & ".join(_pad(headers)) if headers else " & ".join([""] * n_cols)
    body_rows = [" & ".join(_pad(r)) for r in rows]

    return {
        "caption": escape_latex(tbl.get("caption", "")),
        "headers": headers,
        "rows": rows,
        "col_spec": col_spec,
        "header_row": header_row,
        "body_rows": body_rows,
    }


def _copy_figure(
    filename: str,
    images: dict[str, bytes],
    output_dir: Path,
) -> str | None:
    """Write figure bytes into the output directory, return relative path."""
    if not filename:
        return None
    if not _is_supported_figure_file(filename):
        logger.warning(
            "Skipping figure %s because %s is not supported by the LaTeX PDF pipeline",
            filename,
            Path(filename).suffix or "<no extension>",
        )
        return None
    data = images.get(filename)
    if data is None:
        logger.warning("Figure %s not found in images dict", filename)
        return None
    target = output_dir / "figures" / Path(filename).name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)
    return f"figures/{target.name}"


def _is_supported_figure_file(filename: str) -> bool:
    return Path(filename).suffix.lower() in _SUPPORTED_FIGURE_SUFFIXES


def _figure_group_key(filename: str, index: int) -> str:
    stem = Path(filename).stem.strip().casefold()
    return stem or f"__figure_{index}"


def _figure_preference(filename: str) -> tuple[int, str]:
    path = Path(filename)
    suffix = path.suffix.lower()
    return (
        _FIGURE_SUFFIX_PREFERENCE.get(suffix, len(_FIGURE_SUFFIX_PREFERENCE)),
        path.name.casefold(),
    )


def _resolve_supported_figure_filename(
    filename: str,
    images: dict[str, bytes],
) -> str | None:
    if not filename:
        return None
    if filename in images and _is_supported_figure_file(filename):
        return filename

    stem = Path(filename).stem.casefold()
    candidates = [
        name for name in images
        if Path(name).stem.casefold() == stem and _is_supported_figure_file(name)
    ]
    if not candidates:
        return None
    return min(candidates, key=_figure_preference)


def _select_figure_variants(
    figures: list[dict],
    images: dict[str, bytes],
) -> list[tuple[dict, str | None]]:
    grouped: dict[str, list[tuple[int, dict]]] = {}
    for index, fig in enumerate(figures):
        key = _figure_group_key(fig.get("filename", ""), index)
        grouped.setdefault(key, []).append((index, fig))

    selected: list[tuple[dict, str | None]] = []
    for group_key, items in grouped.items():
        best: tuple[tuple[int, tuple[int, str], int], dict, str] | None = None
        for index, fig in items:
            original_filename = fig.get("filename", "")
            resolved_filename = _resolve_supported_figure_filename(
                original_filename,
                images,
            )
            if resolved_filename is None:
                continue
            rank = (
                0 if resolved_filename == original_filename else 1,
                _figure_preference(resolved_filename),
                index,
            )
            if best is None or rank < best[0]:
                best = (rank, fig, resolved_filename)

        if best is not None:
            _, chosen_fig, chosen_filename = best
            skipped = [
                item.get("filename", "")
                for _, item in items
                if item is not chosen_fig and item.get("filename", "")
            ]
            if skipped:
                logger.info(
                    "Using figure %s for group %s; skipped duplicate variants: %s",
                    chosen_filename,
                    group_key,
                    ", ".join(skipped),
                )
            selected.append((chosen_fig, chosen_filename))
            continue

        first_fig = items[0][1]
        original_filename = first_fig.get("filename", "")
        if original_filename:
            suffix = Path(original_filename).suffix.lower()
            if suffix:
                logger.warning(
                    "Skipping figure %s because no LaTeX-renderable variant was found",
                    original_filename,
                )
            else:
                logger.warning(
                    "Skipping figure without a usable filename in group %s",
                    group_key,
                )
        selected.append((first_fig, None))

    return selected


def _materialize_remote_figure(
    fig: dict,
    images: dict[str, bytes],
    cache_dir: Path,
    download_session=None,
) -> None:
    """Download a remote ``src_url`` figure into ``images`` if not already cached.

    LaTeX cannot read ``\\includegraphics`` from an http(s) URL, so any
    figure whose source is a remote URL must be downloaded to a local file
    before compile. The bytes are added to ``images`` keyed by the figure's
    stable filename (the ``_stable_url_filename`` hash assigned by the
    parser); subsequent compilation copies them out of ``images`` like any
    OCR-supplied figure.
    """
    src_url = (fig.get("src_url") or "").strip()
    if not src_url:
        return
    filename = fig.get("filename", "")
    if not filename or filename in images:
        return
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / Path(filename).name
    if cache_path.exists() and cache_path.stat().st_size > 0:
        try:
            images[filename] = cache_path.read_bytes()
            return
        except OSError:
            pass
    try:
        import requests  # local import keeps requests optional at module load
    except ImportError:
        logger.warning(
            "Cannot download remote figure %s — `requests` is not installed.",
            src_url,
        )
        return
    try:
        session = download_session or requests
        resp = session.get(src_url, timeout=60)
        resp.raise_for_status()
        cache_path.write_bytes(resp.content)
        images[filename] = resp.content
    except Exception as exc:
        logger.warning("Failed to download remote figure %s: %s", src_url, exc)


def _materialize_all_remote_figures(
    paper: dict,
    images: dict[str, bytes],
    cache_dir: Path,
) -> None:
    """Walk every figure in ``paper`` and ensure remote ones are local.

    Idempotent: figures already present in ``images`` are skipped, and
    duplicate URLs share a single cache file (the parser assigns the same
    hashed filename to identical URLs).
    """
    try:
        import requests
        session = requests.Session()
    except ImportError:
        session = None

    for section in paper.get("sections", []) or []:
        for fig in section.get("figures", []) or []:
            _materialize_remote_figure(fig, images, cache_dir, session)


def _prepare_figures(
    figures: list[dict],
    images: dict[str, bytes],
    output_dir: Path,
) -> list[dict]:
    prepared: list[dict] = []
    for fig, resolved_filename in _select_figure_variants(figures, images):
        filename = fig.get("filename", "")
        rendered_path = None
        if resolved_filename:
            rendered_path = _copy_figure(resolved_filename, images, output_dir)
        prepared.append({
            "filename": filename,
            "caption": escape_latex(fig.get("caption", "")),
            "rendered_path": rendered_path,
        })
    return prepared


def _prepare_section(
    section: dict,
    images: dict[str, bytes],
    output_dir: Path,
) -> dict:
    return {
        "heading": _markdown_inline_to_latex(
            _clean_heading(section.get("heading", ""))
        ),
        "level": int(section.get("level", 1) or 1),
        "paragraphs": [
            _paragraph_to_latex(p) for p in section.get("paragraphs", []) if p
        ],
        "figures": _prepare_figures(section.get("figures", []), images, output_dir),
        "tables": [
            t for t in (_prepare_table(tbl) for tbl in section.get("tables", []))
            if t is not None
        ],
    }


def _make_env(template_dir: Path) -> Environment:
    env = Environment(
        loader=FileSystemLoader(str(template_dir)),
        block_start_string="((*",
        block_end_string="*))",
        variable_start_string="(((",
        variable_end_string=")))",
        comment_start_string="((=",
        comment_end_string="=))",
        trim_blocks=True,
        lstrip_blocks=True,
        autoescape=False,
        undefined=StrictUndefined,
    )
    return env


def render_paper_to_tex(
    paper: dict[str, Any],
    images: dict[str, bytes] | None = None,
    output_dir: str | Path | None = None,
    template_dir: str | Path | None = None,
    template_name: str = DEFAULT_TEMPLATE,
    line_spacing: float = 1.25,
    margin_mm: int = 25,
    forced_language: str | None = None,
    download_remote_images: bool = True,
) -> Path:
    """
    Render ``paper`` to a LaTeX source file.

    Images referenced in ``paper['sections'][*]['figures']`` are copied
    alongside the .tex file in a ``figures/`` subdirectory so the file
    can be compiled in place. Figures whose source was a remote URL
    (``src_url`` set on the figure dict, e.g. PaddleOCR-VL signed URLs)
    are downloaded into ``output_dir/.remote_images/`` and added to
    ``images`` automatically. Set ``download_remote_images=False`` if the
    caller has already materialized them.

    Returns the path of the written .tex file.
    """
    images = images or {}
    template_dir = Path(template_dir) if template_dir else DEFAULT_TEMPLATE_DIR
    if output_dir is None:
        raise ValueError("output_dir is required")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if download_remote_images:
        remote_cache = output_dir / ".remote_images"
        _materialize_all_remote_figures(paper, images, remote_cache)

    env = _make_env(template_dir)
    template = env.get_template(template_name)

    title = paper.get("title") or "Untitled"
    authors = paper.get("authors") or []
    abstract = paper.get("abstract") or ""
    keywords = paper.get("keywords") or []
    references = paper.get("references") or []
    sections_raw = paper.get("sections") or []

    # has_cjk decides whether the ctex CJK font package is loaded. It MUST
    # track the real content — if the paper has any Chinese characters and
    # we skip ctex, xelatex silently drops every CJK codepoint and the
    # output turns into the shattered English-only fragment the user sees.
    # forced_language is therefore a LABEL override only; it never suppresses
    # CJK font loading.
    has_cjk = _contains_cjk(
        title,
        abstract,
        " ".join(str(a) for a in authors),
        " ".join(str(k) for k in keywords),
        " ".join(
            (s.get("heading", "") + " " + " ".join(s.get("paragraphs", [])))
            for s in sections_raw
        ),
    )

    if forced_language == "zh":
        label_lang = "zh"
    elif forced_language == "en":
        label_lang = "en"
    else:
        label_lang = "zh" if has_cjk else "en"

    labels = {
        "abstract": "摘要" if label_lang == "zh" else "Abstract",
        "keywords": "关键词" if label_lang == "zh" else "Keywords",
        "references": "参考文献" if label_lang == "zh" else "References",
    }

    sections = [_prepare_section(s, images, output_dir) for s in sections_raw]

    rendered = template.render(
        title=_markdown_inline_to_latex(_clean_heading(title)),
        authors=[_markdown_inline_to_latex(a) for a in authors],
        abstract=_paragraph_to_latex(abstract),
        keywords=[_markdown_inline_to_latex(k) for k in keywords],
        sections=sections,
        references=[_markdown_inline_to_latex(r) for r in references],
        has_cjk=has_cjk,
        labels=labels,
        line_spacing=f"{line_spacing:.2f}",
        margin_mm=int(margin_mm),
    )

    tex_path = output_dir / "paper.tex"
    tex_path.write_text(rendered, encoding="utf-8")
    logger.info("LaTeX source written to %s", tex_path)
    return tex_path


def list_available_templates(template_dir: str | Path | None = None) -> list[str]:
    template_dir = Path(template_dir) if template_dir else DEFAULT_TEMPLATE_DIR
    if not template_dir.exists():
        return []
    return sorted(p.name for p in template_dir.glob("*.tex.j2"))
