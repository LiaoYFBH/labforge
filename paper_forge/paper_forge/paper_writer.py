"""LLM-based paper structuring and rewriting using LangChain."""

from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path
from typing import Iterator

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from .config import LLMConfig
from .llm_client import extract_json_from_response
from .style_guide import (
    SECTION_BLUEPRINTS,
    SectionBlueprint,
    blueprint_for,
    blueprint_for_in,
    compose_style_guide,
    scaled_blueprints,
    section_brief,
)
from .utils import detect_paper_language, get_labels_for_language, get_paper_labels, language_name

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are an expert academic paper formatter. Your task is to take raw text \
(extracted from documents via OCR) and restructure it into a well-formatted \
academic paper.

Rules:
1. Preserve ALL factual content, data, and findings from the original.
2. Organize into standard academic sections: Title, Abstract, Introduction, \
Related Work / Background, Methodology, Experiments / Results, Discussion, \
Conclusion, References.
3. If the original text already has clear section structure, respect it but \
improve formatting, coherence, and academic tone.
4. For tables, output structured data as JSON arrays (headers + rows).
5. For figures, preserve the original image references using the exact \
filenames provided in the available images list.
6. Write in the SAME LANGUAGE as the original document.
7. Maintain formal academic tone throughout.
8. Do not output Markdown fences, programming code blocks, or implementation \
instructions. Describe methods as academic prose, equations, and algorithms.
9. Number all sections and subsections (e.g., "1. Introduction", "2.1 Dataset").
10. Do not collapse multiple sections into short summaries. Keep every major \
section from the source and retain the key details in each section.

# STRICT MARKDOWN FORMATTING RULES (MANDATORY LATEX-SAFE SUBSET)
To ensure seamless LaTeX rendering without breaking pipelines, YOU MUST OBEY THESE RULES:
1. Headers MUST follow exact spacing: `## 1. Introduction` or `### 1.2 Background`. No raw HTML (`<h2>`).
2. Do NOT use raw LaTeX macros in the text text (like `\\textbf{{}}`, `\\section{{}}`). Use normal markdown (`**bold**`).
3. Math MUST use `$inline$` or `$$block$$`. Do NOT use `\\[ \\]`, `\\( \\)`, or `\\begin{{equation}}`.
4. For citations, use STRICT bracketed indices: `[1]` or `[1, 2, 3]`. Never use `\\cite{{key}}` or `[Author, Year]`.
5. NEVER write markdown tables (like `| col | col |`). If tabular data is required, describe it in text or define a CSV ref.
6. NO HTML tags (`<br>`, `<sub>`, etc.). Use empty lines for paragraphs.
"""

STRUCTURING_TEMPLATE = """\
Given the following OCR-extracted text from one or more documents, restructure \
it into a well-formatted academic paper. Output ONLY valid JSON with this exact schema:

{{{{
  "title": "Paper title",
  "authors": ["Author Name"],
  "abstract": "Paper abstract text...",
  "keywords": ["keyword1", "keyword2"],
  "sections": [
    {{{{
      "heading": "1. Introduction",
      "level": 1,
      "paragraphs": ["First paragraph...", "Second paragraph..."],
      "figures": [
        {{{{"filename": "exact_image_filename.png", "caption": "Figure description"}}}}
      ],
      "tables": [
        {{{{"caption": "Table title", "headers": ["Col1", "Col2"], "rows": [["a", "b"]]}}}}
      ]
    }}}}
  ],
  "references": ["[1] Author et al. Title. Journal, Year."]
}}}}

Available images from OCR: {image_list}

Source text:
---
{ocr_text}
---
"""

REWRITE_TEMPLATE = """\
You are rewriting the following section of an academic paper to improve its \
clarity, coherence, and academic tone.

Rules:
1. Preserve the same factual content, data, and conclusions.
2. Do not drop any concrete points, numbered items, comparisons, or metrics.
3. If the input is a rough outline or bullet list, expand it into polished \
academic prose while keeping every original point.
4. Preserve references to figures, tables, and filenames when present.
5. The whole output MUST be written in {target_language_name}.
6. Do not mix languages in connective prose. If the target language is Chinese, \
write the prose fully in natural academic Chinese and keep only unavoidable \
proper nouns or model names in their original form.
7. Do not output Markdown fences (```markdown, ```python, etc.), raw code \
snippets, import/assert statements, or programming instructions. Convert \
implementation details into formal academic prose or mathematical notation.

Section heading: {heading}

Original content:
{content}

Rewrite this section with improved academic writing. Output only the rewritten text.
"""

LANGUAGE_FIX_TEMPLATE = """\
The following academic passage should be written in {target_language_name}.

Rules:
1. Convert the passage into fluent {target_language_name}.
2. Preserve all facts, numbers, terminology, figure/table references, and meaning.
3. Do not add or remove substantive content.
4. Keep only unavoidable proper nouns or model names in their original form.
5. Remove Markdown fences and raw code snippets. The result must read like a \
paper section, not a notebook or implementation guide.

Section heading: {heading}

Passage:
{content}

Output only the corrected passage in {target_language_name}.
"""

LATEX_SAFE_NORMALIZER_TEMPLATE = """\
You are a strict LaTeX-safe markdown normalizer. Take the academic prose
below and rewrite ONLY its formatting so it conforms exactly to the
markdown subset described under REQUIRED MARKDOWN SUBSET. Output the
normalized text with no explanations, no fences, no extra prose.

# REQUIRED MARKDOWN SUBSET

1. Math:
   - Inline math MUST be wrapped as `$...$`. Display math MUST be wrapped
     as `$$...$$` on its own line(s).
   - ANY LaTeX math command appearing OUTSIDE `$..$` or `$$..$$` MUST be
     wrapped. Examples of commands that must always live inside math
     delimiters: `\\frac`, `\\sum`, `\\prod`, `\\int`, `\\min`, `\\max`,
     `\\sqrt`, `\\hat`, `\\bar`, `\\tilde`, `\\nabla`, `\\partial`,
     `\\theta`, `\\lambda`, `\\sigma`, `\\alpha`, `\\beta`, `\\gamma`,
     `\\delta`, `\\epsilon`, `\\mu`, `\\pi`, `\\phi`, `\\psi`, `\\omega`,
     `\\infty`, `\\cdot`, `\\times`, `\\leq`, `\\geq`, `\\neq`,
     `\\approx`, `\\propto`, `\\in`, `\\notin`, `\\subset`,
     `\\rightarrow`, `\\leftarrow`, `\\forall`, `\\exists`, `\\mathbb`,
     `\\mathcal`, `\\mathbf`, `\\quad`, `\\qquad`, `\\,`, `\\;`, `\\!`.
   - `\\(...\\)` MUST be rewritten as `$...$`. `\\[...\\]` MUST be
     rewritten as `$$...$$` on its own lines.
   - `\\begin{{equation}}...\\end{{equation}}`, `\\begin{{align}}...`,
     `\\begin{{aligned}}...`, `\\begin{{gather}}...` MUST be rewritten
     as `$$...$$` (keep `\\begin{{aligned}}...\\end{{aligned}}` content
     INSIDE the `$$..$$` if it is genuinely an aligned block).
   - Subscripts (`x_i`, `\\theta_{{i,j}}`) and superscripts (`x^2`) only
     appear inside math delimiters — never bare.

2. Headings: `## 1. Foo` or `### 1.2 Foo`. No raw HTML.

3. Bold: `**bold**`. Italic: `*italic*`. Code: `` `code` ``. NEVER
   `\\textbf{{...}}`, `\\textit{{...}}`, `\\texttt{{...}}`, `\\emph{{...}}`.

4. Citations: bracketed indices `[1]`, `[1, 2, 3]`. NEVER `\\cite{{...}}`
   or `(Author, Year)` style.

5. Lists: lines starting with `- ` for unordered, `1. ` for ordered.
   NEVER `\\begin{{itemize}}` / `\\begin{{enumerate}}`.

6. NO HTML tags (`<br>`, `<sub>`, `<sup>`, `<div>`, `<p>`, ...). Use
   blank lines for paragraph breaks.

7. NO Markdown tables. NO fenced code blocks (no triple backticks).

# HARD RULES

- DO NOT add or remove any factual content, numbers, citations, or
  meaning. ONLY fix formatting/syntax.
- DO NOT change the section heading.
- DO NOT translate the text into a different language.
- If the input already conforms, return it verbatim.

# INPUT

{content}

# OUTPUT

Output ONLY the normalized markdown.
"""

structuring_prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", STRUCTURING_TEMPLATE),
])

rewrite_prompt = ChatPromptTemplate.from_messages([
    ("human", REWRITE_TEMPLATE),
])

latex_safe_normalize_prompt = ChatPromptTemplate.from_messages([
    ("human", LATEX_SAFE_NORMALIZER_TEMPLATE),
])

language_fix_prompt = ChatPromptTemplate.from_messages([
    ("human", LANGUAGE_FIX_TEMPLATE),
])

# ──────────────────────────────────────────────────────────────────────
# "Expand to top-conf style" rewrite — uses the style guide for context.
# ──────────────────────────────────────────────────────────────────────

EXPAND_SYSTEM_PROMPT = """\
You are a senior academic writer trained on top-tier ML/AI conference papers
(NeurIPS, ICML, CVPR, ACL). Your task is to take a section of a research paper
draft and rewrite it as a publication-quality {target_language_name} section,
following the structural and stylistic patterns of top conferences.

You must obey the style guide below. The style guide includes hard
anti-fabrication rules — violating them invalidates the whole output.

{style_guide}

# STRICT MARKDOWN FORMATTING RULES (MANDATORY LATEX-SAFE SUBSET)
To ensure seamless LaTeX rendering without breaking pipelines, YOU MUST OBEY THESE RULES:
1. Headers MUST follow exact spacing: `## 1. Introduction` or `### 1.2 Background`. No raw HTML (`<h2>`).
2. Do NOT use raw LaTeX macros in the text text (like `\\textbf{{}}`, `\\section{{}}`). Use normal markdown (`**bold**`).
3. Math MUST use `$inline$` or `$$block$$`. Do NOT use `\\[ \\]`, `\\( \\)`, or `\\begin{{equation}}`.
4. For citations, use STRICT bracketed indices: `[1]` or `[1, 2, 3]`. Never use `\\cite{{key}}` or `[Author, Year]`.
5. NEVER write markdown tables (like `| col | col |`). If tabular data is required, describe it in text or define a CSV ref.
6. NO HTML tags (`<br>`, `<sub>`, etc.). Use empty lines for paragraphs.

"""

EXPAND_TEMPLATE = """\
You are an academic-writing assistant. Your job is to take the upstream
research agent's structured input below and write **one section** of the
paper in clean academic prose. You are NOT a content generator — you are
a polisher. Every fact, number, citation, and claim in your output MUST
trace back to the input below; you have no other source of information.

{section_brief}

The paper title is: {paper_title}

# Allowed materials (use these and ONLY these)

## Abstract context (for tone/coherence reference)
{abstract_preview}

## Verified references collected this run
The agent searched the literature and verified each of these papers exists.
You may cite any subset of them by their existing wording. You may NOT
introduce any author / year / title combination not in this list.
---
{references_preview}
---

## Figures saved in this section
{figures_preview}

## Tables saved in this section
{tables_preview}

## Original section draft (the agent's factual ground truth — this is the
## sole source of numbers, dataset names, and method specifics)
---
{content}
---

# What you must output

Rewrite the section in {target_language_name} as polished academic prose.
Aim for approximately {target_words} words AS AN UPPER BOUND. **If the
materials above are not rich enough to fill {target_words} words, write a
shorter, honest section. Do NOT pad with invented content.** A 400-word
honest section is strictly better than a 1000-word section with even one
fabricated stat or citation.

# HARD anti-fabrication rules (output is invalid if violated)

You are forbidden to:
  ✗ Cite any author / year / paper not in the "Verified references" block.
    No "Smith et al., 2023", "Zhang et al., 2022", "Chen 2021" unless
    Smith / Zhang / Chen actually appears in that block above.
  ✗ Invent industry sources of any kind: "Industry Whitepaper",
    "Gartner", "MarketReport", "Annual Survey", "NLP Industry Report".
    Search returned academic papers only — these sources do NOT exist
    in your input.
  ✗ Use placeholder citations like "[ref-12]", "[REF-N]", "(Smith)".
  ✗ Invent numerical claims: "X% of users", "Y billion samples",
    "F1=0.85 on news data", "60% of vectors". Every number in your output
    must already appear verbatim in the original draft above.
  ✗ Invent results, metrics, or method behaviour beyond what the original
    draft reports. If the draft says K-means achieved silhouette 0.015,
    you cannot rephrase that as "K-means achieved respectable performance"
    — that's a quality judgment the draft did not make.

You are encouraged to:
  ✓ Improve transitions, motivation sentences, and rhetorical structure.
  ✓ Reorganize material from the draft for clarity.
  ✓ Translate informal phrasing into academic tone.
  ✓ When the draft lacks a fact, write "the experiment did not report
    this metric" / "未报告" instead of inventing one.
  ✓ When you want to discuss related work but no fitting paper is in the
    verified references, write "no closely related prior work was
    retrieved during this run's literature search" rather than naming
    a paper.

# Format requirements

1. Output the section body ONLY — do NOT print the section heading again.
2. Reuse the existing figure/table captions and filenames verbatim when
   referring to them.
3. Use real LaTeX math notation (\\(...\\) or \\[...\\]) for equations.
4. Do NOT output Markdown fences, fenced code blocks, raw Python /
   pseudocode syntax, import / assert statements, or "# ..." comments.
5. Cover every "must cover" point listed in the section brief.
6. Open with a topic sentence; close with a transition sentence.
7. For introductions: end with a contributions list of 3+ bullets, each
   starting with a bold action verb. Each bullet's claim must be backed
   by something in the original draft above.
"""

expand_prompt = ChatPromptTemplate.from_messages([
    ("system", EXPAND_SYSTEM_PROMPT),
    ("human", EXPAND_TEMPLATE),
])


def _word_count(text: str) -> int:
    """Approximate word count that works for both English and Chinese.

    For English we count alphabetic runs; for Chinese we additionally count
    each CJK character as one word. Mixed-language paragraphs add the two.
    """
    if not text:
        return 0
    en = len(re.findall(r"[A-Za-z]+", text))
    cjk = len(re.findall(r"[一-鿿]", text))
    return en + cjk


def _truncate_for_prompt(items: list[str] | None, max_chars: int = 600) -> str:
    if not items:
        return "(none)"
    rendered = "; ".join(str(x) for x in items)
    if len(rendered) > max_chars:
        rendered = rendered[:max_chars] + "…"
    return rendered


def _format_references_for_prompt(items: list[str] | None, max_chars: int = 8000) -> str:
    """Render the FULL references list for the expander prompt.

    The expander's anti-fabrication contract relies on the LLM seeing every
    paper the agent actually verified — if we truncate the list to 600 chars
    (as we did historically), the LLM sees 3-4 references and assumes the
    rest must be its own knowledge, which is exactly how phantom citations
    like "Zhang et al., 2022" get injected.

    Each reference goes on its own numbered line so the LLM can scan for
    "is X et al. <YEAR> in the list?" without ambiguity. The 8000-char
    soft cap is a safety net (~40 typical refs); when exceeded we keep
    the head and append a tail marker so the model knows truncation
    happened (so it can be conservative about citing "the rest").
    """
    if not items:
        return "(no verified references — do NOT cite any paper)"
    cleaned = [str(x).strip() for x in items if str(x).strip()]
    if not cleaned:
        return "(no verified references — do NOT cite any paper)"
    lines: list[str] = []
    total = 0
    for i, ref in enumerate(cleaned, start=1):
        line = f"  [{i}] {ref}"
        # +1 for the trailing newline we'll add at join time.
        if total + len(line) + 1 > max_chars and lines:
            lines.append(
                f"  ... ({len(cleaned) - i + 1} additional verified "
                "references omitted from this prompt for length; rely on "
                "the head of the list above)"
            )
            break
        lines.append(line)
        total += len(line) + 1
    return "\n".join(lines)


def _figures_preview(section: dict) -> str:
    figs = section.get("figures") or []
    if not figs:
        return "(none)"
    return "; ".join(
        f"{f.get('filename', '?')} → {f.get('caption', '')[:80]}"
        for f in figs
    )


def _tables_preview(section: dict) -> str:
    tbls = section.get("tables") or []
    if not tbls:
        return "(none)"
    return "; ".join(
        f"{t.get('caption', '?')[:60]} ({len(t.get('headers', []))} cols × "
        f"{len(t.get('rows', []))} rows)"
        for t in tbls
    )

HEADING_RE = re.compile(r"^\s{0,3}(#{1,6})\s+(.*?)\s*$")
HORIZONTAL_RULE_RE = re.compile(r"^\s{0,3}([-*_])(?:\s*\1){2,}\s*$")
IMAGE_RE = re.compile(r"!\[(.*?)\]\((.*?)\)")
# OCR backends like PaddleOCR-VL emit images as raw HTML ``<img src="...">``
# (often wrapped in a ``<div style="text-align:center">``) instead of the
# Markdown ``![](path)`` form. We detect both. ``src`` may be a remote URL
# (with signed query string) or a local filename, both are routed to the
# per-figure download/copy step downstream.
HTML_IMG_RE = re.compile(
    r"""<img\s+[^>]*?src\s*=\s*["']([^"']+)["'][^>]*?(?:/\s*>|>\s*</img\s*>|>)""",
    re.IGNORECASE,
)
HTML_IMG_ALT_RE = re.compile(
    r"""alt\s*=\s*["']([^"']*)["']""",
    re.IGNORECASE,
)
HTML_DIV_OPEN_RE = re.compile(r"^\s*<div\b[^>]*>\s*$", re.IGNORECASE)
HTML_DIV_CLOSE_RE = re.compile(r"^\s*</div\s*>\s*$", re.IGNORECASE)
# ``$$ ... $$`` display math, possibly on a single OCR line. The parser
# also recognises multi-line blocks below in the streaming loop.
DISPLAY_MATH_LINE_RE = re.compile(r"^\s*\$\$\s*(.+?)\s*\$\$\s*$")
DISPLAY_MATH_OPEN_RE = re.compile(r"^\s*\$\$\s*$")
KEYWORDS_RE = re.compile(
    r"^\s*(?:\*\*)?\s*(keywords?|keyword|关键词)\s*(?:\*\*)?\s*[:：]\s*(.+?)\s*$",
    re.IGNORECASE,
)
ABSTRACT_HEADINGS = {"abstract", "摘要"}
REFERENCE_HEADINGS = {"references", "reference", "参考文献", "参考资料"}
FIGURE_TARGET_KEYWORDS = (
    "design", "system", "architecture", "framework", "method", "workflow",
    "设计", "系统", "架构", "框架", "方法", "流程",
)
TRANSIENT_LLM_ERROR_PATTERNS = (
    "访问过于频繁",
    "请稍候再试",
    "rate limit",
    "too many requests",
    "try again later",
    "temporarily unavailable",
    "overloaded",
    "service unavailable",
)
REWRITE_REQUEST_INTERVAL_SECONDS = 1.0
LLM_RETRY_ATTEMPTS = 4
LLM_RETRY_BASE_DELAY_SECONDS = 2.0
LLM_RETRY_MAX_DELAY_SECONDS = 12.0


def _create_llm(config: LLMConfig, temperature: float | None = None) -> ChatOpenAI:
    """Create a LangChain ChatOpenAI instance from config.

    paper_forge's writer prompts don't bind tools (``ChatOpenAI |
    StrOutputParser``), so the lab-forge ``ResilientChatOpenAI`` subclass —
    which exists to coerce malformed ``tool_calls.args`` — adds no value
    here. The plain ``ChatOpenAI`` is correct for this surface.
    """
    return ChatOpenAI(
        model=config.model,
        api_key=config.api_key,
        base_url=config.base_url,
        temperature=temperature if temperature is not None else config.temperature,
        max_tokens=config.max_tokens,
        timeout=config.timeout,
        max_retries=3,
    )


def _is_transient_llm_error(exc: Exception) -> bool:
    """Best-effort detection of provider throttling / temporary failures."""
    message = str(exc).lower()
    return any(pattern.lower() in message for pattern in TRANSIENT_LLM_ERROR_PATTERNS)


def _retry_delay_seconds(attempt: int) -> float:
    """Exponential backoff delay for transient LLM errors."""
    return min(
        LLM_RETRY_BASE_DELAY_SECONDS * (2 ** (attempt - 1)),
        LLM_RETRY_MAX_DELAY_SECONDS,
    )


def _invoke_chain_with_retry(chain, payload: dict, label: str) -> str:
    """Invoke a LangChain chain with backoff for transient provider errors."""
    for attempt in range(1, LLM_RETRY_ATTEMPTS + 1):
        try:
            return chain.invoke(payload)
        except Exception as exc:
            if not _is_transient_llm_error(exc) or attempt >= LLM_RETRY_ATTEMPTS:
                raise

            delay = _retry_delay_seconds(attempt)
            logger.warning(
                "%s hit transient LLM throttling (attempt %d/%d), retrying in %.1fs: %s",
                label,
                attempt,
                LLM_RETRY_ATTEMPTS,
                delay,
                exc,
            )
            time.sleep(delay)

    raise RuntimeError(f"{label} failed after retries")


# Markdown→LaTeX normalizer. Detects raw LaTeX commands that escaped the
# expand prompt's "math MUST be inside $..$" rule and re-asks an LLM to
# wrap them. The detector is intentionally over-eager (false positives
# just cost one extra LLM call; false negatives leave broken math in the
# PDF), so we list every common math command rather than try to be clever.
_RAW_LATEX_MATH_COMMAND_NAMES = (
    "frac", "sum", "prod", "int", "min", "max", "sqrt",
    "hat", "bar", "tilde", "vec", "dot", "ddot",
    "nabla", "partial", "infty",
    "alpha", "beta", "gamma", "delta", "epsilon", "varepsilon",
    "zeta", "eta", "theta", "vartheta", "iota", "kappa",
    "lambda", "mu", "nu", "xi", "pi", "varpi", "rho", "varrho",
    "sigma", "varsigma", "tau", "upsilon", "phi", "varphi",
    "chi", "psi", "omega",
    "Gamma", "Delta", "Theta", "Lambda", "Xi", "Pi",
    "Sigma", "Upsilon", "Phi", "Psi", "Omega",
    "cdot", "times", "div", "pm", "mp", "ast", "star",
    "leq", "geq", "neq", "approx", "equiv", "propto", "sim", "simeq",
    "in", "notin", "subset", "supset", "subseteq", "supseteq",
    "rightarrow", "leftarrow", "Rightarrow", "Leftarrow",
    "mapsto", "to",
    "forall", "exists",
    "mathbb", "mathcal", "mathbf", "mathrm", "mathit", "mathsf",
    "operatorname", "underline", "overline", "underset", "overset",
    "begin", "end",
    "left", "right",
    "quad", "qquad",
)
_RAW_LATEX_MATH_COMMAND_RE = re.compile(
    r"\\(?:" + "|".join(re.escape(n) for n in _RAW_LATEX_MATH_COMMAND_NAMES) + r")\b"
)
_DISPLAY_MATH_BLOCK_RE = re.compile(r"\$\$[\s\S]+?\$\$")
_INLINE_MATH_SPAN_RE = re.compile(r"(?<!\$)\$[^\n$]+?\$(?!\$)")
_BRACKET_DISPLAY_MATH_OUTER_RE = re.compile(r"\\\[[\s\S]+?\\\]")
_PAREN_INLINE_MATH_OUTER_RE = re.compile(r"\\\([\s\S]+?\\\)")


def _has_raw_latex_outside_math(text: str) -> bool:
    """Return True when LaTeX math commands appear outside ``$..$`` / ``$$..$$``.

    We strip every wrapped math span first, then look for surviving
    ``\\frac``, ``\\theta``, ``\\begin{...}`` etc. — the patterns the
    rule-based renderer cannot escape losslessly. Also flags ``\\[..\\]``
    and ``\\(..\\)`` because the renderer's math protection occasionally
    misses them when they straddle paragraph boundaries.
    """
    if not text or "\\" not in text:
        return False
    stripped = _DISPLAY_MATH_BLOCK_RE.sub("", text)
    stripped = _INLINE_MATH_SPAN_RE.sub("", stripped)
    if _BRACKET_DISPLAY_MATH_OUTER_RE.search(stripped):
        return True
    if _PAREN_INLINE_MATH_OUTER_RE.search(stripped):
        return True
    if _RAW_LATEX_MATH_COMMAND_RE.search(stripped):
        return True
    return False


def _normalize_to_latex_safe_markdown(
    text: str, latex_safe_chain, *, label: str = "section",
) -> str:
    """Re-prompt the LLM to coerce ``text`` into our LaTeX-safe markdown subset.

    Only fires when the cheap heuristic ``_has_raw_latex_outside_math``
    detects something the rule-based renderer would mangle. When the input
    is already conformant we skip the call entirely so paper export cost
    only grows on the sections that actually need it.
    """
    if not text or not text.strip():
        return text
    if not _has_raw_latex_outside_math(text):
        return text
    try:
        normalized = _invoke_chain_with_retry(
            latex_safe_chain,
            {"content": text},
            f"latex-safe normalize '{label}'",
        ).strip()
    except Exception as exc:
        logger.warning(
            "LaTeX-safe normalizer failed for '%s' (keeping pre-normalized text): %s",
            label, exc,
        )
        return text
    if not normalized:
        return text
    return normalized


_RESIDUAL_CJK_THRESHOLD = 30  # P3: more than this much CJK in a section
                              # targeting English == force a normalize pass.
_RESIDUAL_LATIN_THRESHOLD = 200  # Same idea for a Chinese-target run, but
                                 # we accept much more Latin because tech
                                 # papers always cite English terms / formulas.


def _needs_language_normalization(text: str, target_language_code: str) -> bool:
    """Check whether rewritten text drifted away from the desired document language.

    P3 fix: ``detect_paper_language`` returns the *majority* language, so a
    section that is 95% English with one Chinese paragraph still reads as
    "en" and slips through the normalization pass. For English-target runs
    we additionally treat any section with ``cjk_count > 30`` as needing a
    rewrite (catches the c032bbf1 case where the rendered PDF carried 416
    stray CJK characters across the document).
    """
    if not text.strip():
        return False
    detected = detect_paper_language(text)
    if detected != target_language_code:
        return True

    # Same target language overall — but check for stray content from
    # the wrong language. Imported here to avoid a top-level circular
    # import; ``utils`` does not import this module so this is safe.
    from .utils import CJK_RE, LATIN_RE
    if target_language_code == "en":
        cjk_count = len(CJK_RE.findall(text))
        return cjk_count > _RESIDUAL_CJK_THRESHOLD
    if target_language_code == "zh":
        latin_count = len(LATIN_RE.findall(text))
        # Higher threshold: Chinese papers legitimately quote English
        # terms (e.g. "Transformer", method names), so we only flag
        # very obvious leftover English paragraphs.
        return latin_count > _RESIDUAL_LATIN_THRESHOLD
    return False


_FENCE_LINE_RE = re.compile(r"^\s*[`'\"]{0,2}```\s*([A-Za-z0-9_-]+)?\s*$")
_MARKDOWN_FENCE_LANGS = {"", "markdown", "md", "tex", "latex", "text", "txt"}
_CODE_FENCE_LANGS = {
    "python", "py", "bash", "sh", "shell", "javascript", "js", "typescript",
    "ts", "java", "cpp", "c", "r", "sql", "json", "yaml", "toml",
}
# Same hardening as latex_renderer._CODE_LIKE_LINE_RE — see the comment
# there. The old substring-anchored version killed academic prose lines
# whenever they happened to contain " class", " from", " for", " if",
# etc. as English words.
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

# P2 fix: display-math blocks (\[...\] / \begin{equation}...) coming out of
# the LLM expansion are routinely glued into the middle of a paragraph, e.g.
#     "...polished methodology section: \[f_\theta(x) = W_2\sigma(...)\] where ..."
# xelatex compiles the LaTeX correctly but the visual layout collapses
# because display math wants its own paragraph. We force a blank line on
# each side so the renderer treats it as a standalone displayed equation.
# Inline math \(...\) is intentionally NOT touched — it belongs in-paragraph.
_DISPLAY_MATH_BRACKET_RE = re.compile(r"\\\[.*?\\\]", re.DOTALL)
_DISPLAY_MATH_ENV_RE = re.compile(
    r"\\begin\{(equation|align|gather|displaymath|eqnarray|multline)\*?\}"
    r".*?"
    r"\\end\{\1\*?\}",
    re.DOTALL,
)

# P4 fix: LLMs sometimes emit document-level LaTeX preamble commands inline
# in section prose (we saw \title{...} on line 33 of trajectory c032bbf1's
# paper.tex). The renderer's own template owns these; section prose must
# not. We drop the entire offending line.
_LATEX_META_LINE_RE = re.compile(
    r"^\s*\\(?:"
    r"title|author|date|maketitle|documentclass|usepackage|"
    r"begin\s*\{\s*document\s*\}|end\s*\{\s*document\s*\}|"
    r"newcommand|renewcommand|providecommand|"
    r"input\s*\{|include\s*\{|"
    r"setlength|setcounter"
    r")\b",
)


def _isolate_display_math(text: str) -> str:
    """Wrap every display-math block in blank lines so xelatex centres it.

    Idempotent on already-isolated blocks (the trailing ``\\n{3,}`` collapse
    keeps it from accumulating blank lines on repeated passes).
    """
    if not text:
        return text

    def _wrap(match: re.Match) -> str:
        return f"\n\n{match.group(0).strip()}\n\n"

    text = _DISPLAY_MATH_BRACKET_RE.sub(_wrap, text)
    text = _DISPLAY_MATH_ENV_RE.sub(_wrap, text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


def _strip_latex_meta_lines(text: str) -> str:
    """Drop any line that starts with a document-level LaTeX command.

    These belong in the renderer's preamble template, never in section prose.
    Conservative: only matches at start-of-line so a stray ``\\title``
    embedded inside a sentence — unlikely but possible — is left alone for
    a human to notice.
    """
    if not text:
        return text
    kept = [line for line in text.splitlines() if not _LATEX_META_LINE_RE.match(line)]
    return "\n".join(kept)


def _strip_outer_markdown_fence(text: str) -> str:
    """Unwrap a whole-section ```markdown fence without dropping the section."""
    lines = text.strip().splitlines()
    if len(lines) < 2:
        return text.strip()
    first = lines[0].strip()
    last = lines[-1].strip()
    first_match = _FENCE_LINE_RE.match(first)
    if first_match and last.startswith("```"):
        lang = (first_match.group(1) or "").lower()
        if lang in _MARKDOWN_FENCE_LANGS:
            return "\n".join(lines[1:-1]).strip()
    return text.strip()


# R1: ``\{}xyz`` -> ``\xyz``. Match the literal three-char sequence
# ``\{}`` when it is immediately followed by anything that could be the
# start of a real LaTeX command:
#   - letters: ``\{}min``, ``\{}textbf``, ``\{}sigma`` (named commands)
#   - brackets: ``\{}[`` (open display math), ``\{}]`` (close),
#     ``\{}(``, ``\{}{``, ``\{}}``
#   - punctuation that LaTeX accepts as a single-char command:
#     ``\,`` ``\;`` ``\:`` ``\!`` ``\|`` (spacing / norm symbol),
#     ``\\`` (linebreak — unusual at line end but seen in tabular cells)
# Trailing ``\{}`` at the very end of a sentence (no follower) is left
# alone since it might be a legitimate decorative empty-group rather
# than a self-escape.
_LLM_SELF_ESCAPED_BACKSLASH_RE = re.compile(
    r"\\\{\}(?=[A-Za-z\[\]\(\)\{\}\|,;:!\\])"
)


# R2 (paper_writer half): convert LLM-emitted text-formatting LaTeX
# commands back into Markdown so the existing markdown→LaTeX pipeline in
# latex_renderer handles them through ``_MD_INLINE_PATTERNS``. This way
# the paragraph stays in the "prose" path (escape_latex applied to
# everything outside math), while the formatting still survives — instead
# of the LLM's literal ``\textbf{...}`` getting escape-mangled into
# ``\textbackslash\{\}textbf\{...\}`` (the c032bbf1 / 80ef0f0e bug).
#
# Limited to single-level argument: ``\textbf{nested \textit{...}}`` would
# require balanced-brace matching. We accept the rare miss; the latex
# renderer's R2 half (allowlist-protected commands) catches the more
# important inline-link / citation commands.
_LATEX_TO_MARKDOWN_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\\textbf\{([^{}\n]+?)\}"), r"**\1**"),
    (re.compile(r"\\textit\{([^{}\n]+?)\}"), r"*\1*"),
    (re.compile(r"\\emph\{([^{}\n]+?)\}"), r"*\1*"),
    (re.compile(r"\\texttt\{([^{}\n]+?)\}"), r"`\1`"),
)


def _convert_inline_latex_to_markdown(text: str) -> str:
    """Rewrite a few well-known text-formatting LaTeX commands back into
    Markdown so the renderer's markdown pipeline handles them safely.

    Only affects the listed commands (textbf/textit/emph/texttt). Other
    ``\\command`` calls are left alone and will be protected by the
    latex_renderer's allowlist (see R2 half there).
    """
    if not text or "\\" not in text:
        return text
    for pat, repl in _LATEX_TO_MARKDOWN_PATTERNS:
        text = pat.sub(repl, text)
    return text


def _undo_llm_self_escape(text: str) -> str:
    """Reverse the LLM's self-applied LaTeX-escape style ``\\{}command``.

    Some LLMs, when asked to emit LaTeX-bearing prose, defensively prefix
    every backslash with ``\\{}`` (a no-op pair: a literal backslash
    followed by an empty group) on the theory that this neutralizes any
    accidental command interpretation. In practice it converts every real
    LaTeX command (``\\frac``, ``\\textbf``, ``\\[`` …) into the literal
    string ``\\{}command`` — and the latex_renderer's ``_protect_math``
    regex no longer recognizes math blocks (``\\{}[ ... \\{}]`` does not
    match ``\\[ ... \\]``), so the whole paragraph falls through to
    ``escape_latex`` and the PDF shows literal junk like ``\\{}textbf``.

    Reverse by replacing ``\\{}`` with a single backslash whenever it
    immediately precedes an alpha character or ``[`` / ``(`` — those are
    the LaTeX commands and math-bracket openers an LLM would have
    self-escaped. Trailing ``\\{}`` (no follower) is left untouched so we
    don't rewrite legitimate empty-group decoration.
    """
    if not text or "\\{}" not in text:
        return text
    return _LLM_SELF_ESCAPED_BACKSLASH_RE.sub(r"\\", text)


# R3: an LLM asked to "rewrite this section" routinely opens with a
# meta-narration like ``Here is the polished methodology section ...:``
# before getting to the actual prose. Drop those leading phrases so they
# don't leak into the rendered PDF as the first sentence of the section.
_LLM_PREFIX_PATTERNS: tuple[re.Pattern[str], ...] = (
    # English: "Here is/are/'s the [...descriptors...] <section_name> ...:".
    # The descriptor block is *any* sequence of up to 4 short words
    # (≤ 18 chars each, no punctuation other than hyphens) sitting
    # between "the/an/a" and the section noun. This covers multi-word
    # descriptors the strict allowlist missed: "academic prose version",
    # "expanded 1100-word", "polished publication-ready", etc.
    # We anchor on the section noun + colon so this can't run away.
    re.compile(
        r"^\s*here(?:\s*['’`]\s*s|\s*['’`]\s*re|\s+(?:is|are|was|were))\s+"
        r"(?:the|an|a|my|your)?\s*"
        # Up to 6 short tokens (numbers, hyphenated descriptors, markdown
        # bold like ``**Methodology**``, "of", "the", etc.) sit between
        # the article and the section noun. Each token is ≤ 24 chars;
        # allowing ``*`` handles markdown wrappers from real traces
        # ("expanded **Analysis & Discussion** section").
        r"(?:[\w\-*&]{1,24}\s+){0,6}"
        # Section nouns + length / form descriptors. Adding
        # ``version|draft|revision|rewrite|expansion`` covers leaks like
        # "Here is the expanded 800-word version ...:" seen in trajectory
        # 5bfb6c8d, where the strict section-noun list missed the trailing
        # noun ``version`` and the whole meta-prose escaped into the PDF.
        r"(?:abstract|introduction|related\s+work|background|method(?:ology)?|"
        r"experimental\s+setup|setup|results?|experiments?|analysis|discussion|"
        r"conclusion|section|paragraph|passage|"
        r"version|draft|revision|rewrite|expansion|expanded\s+text)"
        r"[^:\n]{0,200}:\s*",
        re.IGNORECASE,
    ),
    # Chinese counterparts seen in real traces ("以下是改写后的方法部分：" etc.)
    re.compile(
        r"^\s*(?:以下是|这里是|下面是)[^：\n]{0,80}(?:版本|章节|部分|段落)?[：:]\s*",
    ),
    re.compile(
        r"^\s*(?:已为您|为您|本次|根据要求)[^：\n]{0,80}(?:章节|部分|段落)[：:]\s*",
    ),
)


def _strip_llm_meta_prefix(text: str) -> str:
    """Drop ``Here is the polished … section:`` LLM self-narration prefix.

    Only strips at the very start of the section content (after any leading
    whitespace) — middle-of-paragraph occurrences are left alone since
    those are likely substantive content (e.g. quoting feedback).
    """
    if not text:
        return text
    head = text.lstrip()
    leading_ws = text[: len(text) - len(head)]
    for pattern in _LLM_PREFIX_PATTERNS:
        match = pattern.match(head)
        if match:
            head = head[match.end():].lstrip()
            return leading_ws + head
    return text


# Trailing meta-prose the writer LLM tacks onto a section: word-count
# annotations, self-reflective change-logs, Chinese "math appendix" hints.
# These leaked into trajectory 5bfb6c8d's final PDF as ``*Word count: 1,128*``
# and a numbered "This revision expands the discussion by:" block under the
# Conclusion. None of them are actual content.
_LLM_SUFFIX_PATTERNS: tuple[re.Pattern[str], ...] = (
    # ``*Word count: 1,128*`` — italic-wrapped single-line counter at EOF.
    re.compile(
        r"(?:\n+|\A)\s*\*?\s*(?:Word\s+count|字数(?:统计)?)\s*[:：][^\n]*\*?\s*\Z",
        re.IGNORECASE,
    ),
    # English self-reflective tail: "This revision/version/expansion expands /
    # adds / introduces ..." possibly preceded by an HR. Eats to EOF so
    # numbered-list reflection blocks don't bleed into the PDF.
    re.compile(
        r"(?:\n+(?:---+\s*\n+)?)"
        r"(?:This\s+(?:revision|expansion|version|rewrite|update)\s+"
        r"(?:expands|adds|incorporates|enhances|introduces|improves|highlights))"
        r"[\s\S]+\Z",
        re.IGNORECASE,
    ),
    # Chinese trailing self-reflection ("本次扩写..." / "本版本添加...").
    re.compile(
        r"(?:\n+(?:---+\s*\n+)?)"
        r"(?:本次|本版本|这次|这版)(?:修订|改写|扩写|更新|改写后)"
        r"(?:在|添加|引入|增强|加入|强化)"
        r"[\s\S]+\Z",
    ),
    # Chinese meta hint that introduces a placeholder math block:
    # "数学补充（根据需要嵌入）：\n$$ ... $$"
    # Keep math elsewhere; just remove this orphaned appendix.
    re.compile(
        r"\n+数学补充\s*[（(][^）)]*[）)]\s*[:：]\s*\n+\$\$[\s\S]+?\$\$\s*\Z",
    ),
)


def _strip_llm_meta_suffix(text: str) -> str:
    """Remove trailing LLM meta-prose (word counts, change-logs, …)."""
    if not text:
        return text
    cleaned = text
    # Apply repeatedly so a section ending with both a word-count line AND
    # a change-log paragraph drops both, regardless of order.
    for _ in range(len(_LLM_SUFFIX_PATTERNS)):
        before = cleaned
        for pattern in _LLM_SUFFIX_PATTERNS:
            cleaned = pattern.sub("", cleaned)
        if cleaned == before:
            break
    return cleaned.rstrip() if cleaned.strip() else cleaned


def _clean_llm_section_output(text: str) -> str:
    """Remove Markdown/code artifacts that should never reach the PDF."""
    # R1: undo the LLM's "self-escape" pattern \\{}command -> \\command
    # before any other processing, so downstream regex passes (display-math
    # isolation, latex_renderer.protect_math) can recognize real LaTeX.
    text = _undo_llm_self_escape(text or "")
    # R3: strip the "Here is the polished ... section:" meta-narration so
    # it doesn't become the first sentence of the rendered paper.
    text = _strip_llm_meta_prefix(text)
    # R3b: strip trailing meta-prose (``*Word count: 1,128*`` / ``This
    # revision expands by ...`` / 数学补充：…). Trajectory 5bfb6c8d shipped
    # all three of these into the final PDF.
    text = _strip_llm_meta_suffix(text)
    text = _strip_outer_markdown_fence(text)
    if not text:
        return ""
    # R2 (paper_writer half): convert text-formatting LaTeX back to
    # Markdown so the renderer's markdown path handles it. Done here, after
    # the fence/prefix strip, so the conversion sees only real prose.
    text = _convert_inline_latex_to_markdown(text)

    cleaned_lines: list[str] = []
    in_fence = False
    keep_fenced_markdown = False
    for raw_line in text.splitlines():
        stripped = raw_line.strip()
        fence_match = _FENCE_LINE_RE.match(stripped)
        if fence_match:
            lang = (fence_match.group(1) or "").lower()
            if not in_fence:
                in_fence = True
                keep_fenced_markdown = lang in _MARKDOWN_FENCE_LANGS
            else:
                in_fence = False
                keep_fenced_markdown = False
            continue

        if in_fence:
            if keep_fenced_markdown:
                cleaned_lines.append(raw_line)
            continue

        if stripped.lower() in {"markdown", "python", "latex", "tex"}:
            continue
        if _CODE_LIKE_LINE_RE.search(stripped):
            continue
        cleaned_lines.append(raw_line)

    cleaned = "\n".join(cleaned_lines).strip()
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    # P4: drop document-level LaTeX preamble commands the LLM sometimes
    # emits inside section prose.
    cleaned = _strip_latex_meta_lines(cleaned)
    # P2: wrap display-math blocks in blank lines so xelatex renders them
    # as standalone centred equations rather than glueing them into the
    # middle of the surrounding paragraph.
    cleaned = _isolate_display_math(cleaned)
    return cleaned.strip()


def _rewrite_heading_label(paper: dict, section_heading: str) -> str:
    """Use localized headings for synthetic sections like abstract."""
    if section_heading == "Abstract":
        return get_paper_labels(paper)["abstract"]
    return section_heading


_HEADING_TRANSLATIONS = {
    "en": {
        "摘要": "Abstract",
        "引言": "Introduction",
        "导言": "Introduction",
        "简介": "Introduction",
        "相关工作": "Related Work",
        "背景": "Background",
        "研究现状": "Related Work",
        "方法": "Methodology",
        "方法与实现": "Methodology",
        "方法与实现的验证": "Methodology and Validation",
        "实验": "Experiments",
        "实验结果": "Experimental Results",
        "结果": "Results",
        "结果分析": "Analysis",
        "分析": "Analysis",
        "讨论": "Discussion",
        "局限": "Limitations",
        "局限性": "Limitations",
        "结论": "Conclusion",
        "总结": "Conclusion",
    },
    "zh": {
        "abstract": "摘要",
        "introduction": "引言",
        "related work": "相关工作",
        "background": "背景",
        "method": "方法",
        "methods": "方法",
        "methodology": "方法",
        "experiments": "实验",
        "experimental results": "实验结果",
        "results": "结果",
        "analysis": "结果分析",
        "discussion": "讨论",
        "limitations": "局限性",
        "conclusion": "结论",
    },
}


def _normalize_heading_language(heading: str, target_language_code: str) -> str:
    """Translate common academic headings when the UI forces a language."""
    if not heading:
        return heading
    prefix_match = re.match(r"^(\s*\d+(?:\.\d+)*[.)]?\s*)(.+?)\s*$", heading)
    prefix = prefix_match.group(1) if prefix_match else ""
    body = prefix_match.group(2) if prefix_match else heading
    body_clean = _strip_markdown_adornments(body).strip().strip(":：")
    key = re.sub(r"\s+", " ", body_clean).strip().lower()
    mapping = _HEADING_TRANSLATIONS.get(target_language_code, {})
    replacement = mapping.get(body_clean) or mapping.get(key)
    if not replacement:
        return heading
    return f"{prefix}{replacement}".strip()


def _apply_target_language_to_headings(paper: dict, target_language_code: str) -> None:
    """Best-effort heading localization without invoking the LLM."""
    for section in paper.get("sections", []):
        heading = section.get("heading", "")
        section["heading"] = _normalize_heading_language(heading, target_language_code)


def _strip_markdown_adornments(text: str) -> str:
    """Remove simple markdown adornments around a single line."""
    cleaned = text.strip()
    cleaned = re.sub(r"^\s{0,3}#{1,6}\s*", "", cleaned)
    cleaned = re.sub(r"^\*\*(.*?)\*\*$", r"\1", cleaned)
    cleaned = re.sub(r"^\*(.*?)\*$", r"\1", cleaned)
    return cleaned.strip()


def _normalize_leading_line(text: str) -> str:
    """Normalize a heading-like line for comparison."""
    cleaned = _strip_markdown_adornments(text)
    cleaned = cleaned.strip().rstrip(":：.。")
    return _normalize_heading_label(cleaned)


def _remove_duplicate_heading_prefix(text: str, heading: str) -> str:
    """Strip a repeated heading line if the rewrite echoed the section title."""
    text = _clean_llm_section_output(text)
    if not text.strip() or not heading.strip():
        return text.strip()

    lines = text.splitlines()
    first_nonempty_idx = next((i for i, line in enumerate(lines) if line.strip()), None)
    if first_nonempty_idx is None:
        return text.strip()

    first_line = lines[first_nonempty_idx]
    if _normalize_leading_line(first_line) != _normalize_heading_label(heading):
        return text.strip()

    remaining_lines = lines[first_nonempty_idx + 1 :]
    while remaining_lines and not remaining_lines[0].strip():
        remaining_lines.pop(0)
    return "\n".join(remaining_lines).strip()


def _remove_duplicate_section_label(text: str, label: str) -> str:
    """Strip a repeated abstract/keyword label if the rewrite echoed it."""
    text = _clean_llm_section_output(text)
    if not text.strip() or not label.strip():
        return text.strip()

    lines = text.splitlines()
    first_nonempty_idx = next((i for i, line in enumerate(lines) if line.strip()), None)
    if first_nonempty_idx is None:
        return text.strip()

    first_line = _strip_markdown_adornments(lines[first_nonempty_idx]).strip()
    label_norm = label.strip().lower()
    if first_line.lower().rstrip(":：.。") != label_norm:
        return text.strip()

    remaining_lines = lines[first_nonempty_idx + 1 :]
    while remaining_lines and not remaining_lines[0].strip():
        remaining_lines.pop(0)
    return "\n".join(remaining_lines).strip()


def should_use_native_markdown_parser(text: str) -> bool:
    """Detect structured Markdown that is better parsed locally than regenerated wholesale."""
    if not text or not text.strip():
        return False

    heading_matches = [
        match for line in text.splitlines()
        if (match := HEADING_RE.match(line))
    ]
    headings = [match.group(2).strip() for match in heading_matches]
    if len(headings) < 2:
        return False

    academic_hits = 0
    for heading in headings:
        normalized = _normalize_heading_label(heading)
        if normalized in ABSTRACT_HEADINGS or normalized in REFERENCE_HEADINGS:
            academic_hits += 1
        elif re.match(r"^\d+(?:\.\d+)*\s+\S+", heading):
            academic_hits += 1

    if academic_hits >= 1:
        return True

    has_h1 = any(match.group(1) == "#" for match in heading_matches)
    has_h2 = any(match.group(1) == "##" for match in heading_matches)
    return has_h1 and has_h2


def _stable_url_filename(url: str, suffix_hint: str = ".jpg") -> str:
    """Synthesize a stable, filesystem-safe filename for a remote image URL.

    Two URLs that point at the same resource (signed-URL with rotating
    `authorization=` query) hash to the same filename, so duplicate
    downloads are coalesced and the LaTeX template can reference a stable
    relative path.
    """
    import hashlib
    from urllib.parse import urlsplit

    parts = urlsplit(url)
    digest = hashlib.md5(url.split("?", 1)[0].encode("utf-8")).hexdigest()[:12]
    base_path = parts.path or ""
    suffix = ""
    if "." in base_path:
        candidate = base_path.rsplit(".", 1)[-1].lower()
        if 1 <= len(candidate) <= 5 and candidate.isalnum():
            suffix = "." + candidate
    if not suffix:
        suffix = suffix_hint or ".jpg"
    return f"remote_{digest}{suffix}"


def _extract_html_image(line: str) -> dict | None:
    """Pull the first ``<img>`` tag off a line (with optional ``<div>`` wrap).

    Returns a dict with keys ``src``, ``alt``, ``filename``, ``is_remote``,
    or None if the line is not a self-contained image element.
    """
    match = HTML_IMG_RE.search(line)
    if not match:
        return None
    src = match.group(1).strip()
    if not src:
        return None
    alt_match = HTML_IMG_ALT_RE.search(match.group(0))
    alt_text = alt_match.group(1).strip() if alt_match else ""
    is_remote = src.lower().startswith(("http://", "https://"))
    if is_remote:
        filename = _stable_url_filename(src)
    else:
        filename = Path(src).name
    return {
        "src": src,
        "alt": alt_text,
        "filename": filename,
        "is_remote": is_remote,
    }


def parse_markdown_paper(markdown_text: str, image_names: list[str]) -> dict:
    """Parse structured Markdown into the internal paper JSON schema."""
    text = _strip_outer_markdown_fence(
        markdown_text.replace("\r\n", "\n").replace("\r", "\n")
    )
    lines = text.split("\n")

    paper = {
        "title": "",
        "authors": [],
        "abstract": "",
        "keywords": [],
        "sections": [],
        "references": [],
    }
    preamble_blocks: list[str] = []
    referenced_images: set[str] = set()
    current_section: dict | None = None
    paragraph_lines: list[str] = []
    table_lines: list[str] = []
    in_code_block = False

    def ensure_section() -> dict:
        nonlocal current_section
        if current_section is None:
            current_section = {
                "heading": "1. Content",
                "level": 1,
                "paragraphs": [],
                "figures": [],
                "tables": [],
            }
            paper["sections"].append(current_section)
        return current_section

    def flush_paragraph() -> None:
        nonlocal paragraph_lines
        if not paragraph_lines:
            return
        block = "\n".join(line.rstrip() for line in paragraph_lines).strip()
        paragraph_lines = []
        if not block:
            return
        if current_section is None:
            preamble_blocks.append(block)
        else:
            current_section["paragraphs"].append(block)

    def flush_table() -> None:
        nonlocal table_lines
        if not table_lines:
            return
        table_block = [line.rstrip() for line in table_lines]
        table_lines = []
        parsed = _parse_markdown_table(table_block)
        if parsed is None:
            block = "\n".join(table_block).strip()
            if not block:
                return
            if current_section is None:
                preamble_blocks.append(block)
            else:
                current_section["paragraphs"].append(block)
            return
        ensure_section()["tables"].append(parsed)

    line_idx = 0
    while line_idx < len(lines):
        raw_line = lines[line_idx]
        line = raw_line.rstrip("\n")
        stripped = line.strip()

        if not in_code_block and stripped.startswith("<!--") and stripped.endswith("-->"):
            line_idx += 1
            continue

        if stripped.startswith("```"):
            flush_table()
            if not in_code_block:
                flush_paragraph()
            in_code_block = not in_code_block
            if not in_code_block:
                flush_paragraph()
            line_idx += 1
            continue

        if in_code_block:
            line_idx += 1
            continue

        # ── Display math: ``$$ ... $$`` on a single line ──────────────────
        single_math = DISPLAY_MATH_LINE_RE.match(stripped)
        if single_math:
            flush_paragraph()
            flush_table()
            ensure_section()["paragraphs"].append(
                "$$" + single_math.group(1).strip() + "$$"
            )
            line_idx += 1
            continue

        # ── Display math: ``$$`` then content lines then ``$$`` ────────────
        if DISPLAY_MATH_OPEN_RE.match(stripped):
            flush_paragraph()
            flush_table()
            buf: list[str] = []
            j = line_idx + 1
            closed = False
            while j < len(lines):
                inner = lines[j].rstrip("\n").strip()
                if inner == "$$" or inner.endswith("$$"):
                    if inner != "$$":
                        buf.append(inner[:-2].strip())
                    closed = True
                    j += 1
                    break
                buf.append(lines[j].rstrip("\n"))
                j += 1
            if closed and any(line.strip() for line in buf):
                ensure_section()["paragraphs"].append(
                    "$$\n" + "\n".join(buf).strip() + "\n$$"
                )
                line_idx = j
                continue
            # Unclosed ``$$`` — treat the opener as plain text and continue.

        # ── HTML ``<img>`` (PaddleOCR-VL emits these instead of ``![]()``) ──
        # The OCR markdown often wraps the image in a ``<div>``; absorb the
        # wrapper too so the URL doesn't leak into the prose.
        if HTML_DIV_OPEN_RE.match(line) and line_idx + 1 < len(lines):
            inner_line = lines[line_idx + 1].rstrip("\n")
            inner_image = _extract_html_image(inner_line)
            if inner_image is not None:
                # If the next-next line is a closing div, eat all three.
                consume_to = line_idx + 2
                if (
                    consume_to < len(lines)
                    and HTML_DIV_CLOSE_RE.match(lines[consume_to].rstrip("\n"))
                ):
                    consume_to += 1
                flush_paragraph()
                flush_table()
                target = ensure_section()
                referenced_images.add(inner_image["filename"])
                target["figures"].append({
                    "filename": inner_image["filename"],
                    "caption": inner_image["alt"] or _humanize_filename(inner_image["filename"]),
                    "src_url": inner_image["src"] if inner_image["is_remote"] else "",
                })
                line_idx = consume_to
                continue

        # Bare ``<img>`` line, possibly wrapped in a same-line ``<div>``.
        bare_image = _extract_html_image(line)
        if bare_image is not None and not stripped.startswith("<!--"):
            # Strip the matched ``<img>`` and any surrounding ``<div>`` /
            # ``</div>`` wrapper (which OCR backends like PaddleOCR-VL emit
            # on the same line). If what's left is just whitespace, treat
            # the whole line as a figure block.
            without_tag = HTML_IMG_RE.sub("", line, count=1)
            without_tag = re.sub(r"<div\b[^>]*>", "", without_tag, flags=re.IGNORECASE)
            without_tag = re.sub(r"</div\s*>", "", without_tag, flags=re.IGNORECASE)
            if not without_tag.strip():
                flush_paragraph()
                flush_table()
                target = ensure_section()
                referenced_images.add(bare_image["filename"])
                target["figures"].append({
                    "filename": bare_image["filename"],
                    "caption": bare_image["alt"] or _humanize_filename(bare_image["filename"]),
                    "src_url": bare_image["src"] if bare_image["is_remote"] else "",
                })
                line_idx += 1
                continue

        heading_match = HEADING_RE.match(line)
        if heading_match:
            flush_paragraph()
            flush_table()
            heading_marks, heading_text = heading_match.groups()
            if not paper["title"] and len(heading_marks) == 1 and not paper["sections"]:
                paper["title"] = heading_text.strip()
                current_section = None
                line_idx += 1
                continue

            current_section = {
                "heading": heading_text.strip(),
                "level": max(1, len(heading_marks) - (1 if paper["title"] else 0)),
                "paragraphs": [],
                "figures": [],
                "tables": [],
            }
            paper["sections"].append(current_section)
            line_idx += 1
            continue

        if HORIZONTAL_RULE_RE.match(line):
            flush_paragraph()
            flush_table()
            line_idx += 1
            continue

        if stripped.startswith("|") and stripped.endswith("|"):
            flush_paragraph()
            table_lines.append(line.rstrip())
            line_idx += 1
            continue

        if table_lines and stripped:
            flush_table()

        if not stripped:
            flush_paragraph()
            flush_table()
            line_idx += 1
            continue

        image_matches = list(IMAGE_RE.finditer(line))
        if image_matches:
            flush_paragraph()
            flush_table()
            target = ensure_section()
            remaining_text = line
            for match in image_matches:
                caption = match.group(1).strip()
                src = match.group(2).strip()
                if src.lower().startswith(("http://", "https://")):
                    filename = _stable_url_filename(src)
                    src_url = src
                else:
                    filename = Path(src).name
                    src_url = ""
                referenced_images.add(filename)
                target["figures"].append({
                    "filename": filename,
                    "caption": caption or _humanize_filename(filename),
                    "src_url": src_url,
                })
                remaining_text = remaining_text.replace(match.group(0), "").strip()
            if remaining_text:
                paragraph_lines.append(remaining_text)
            line_idx += 1
            continue

        paragraph_lines.append(line.rstrip())
        line_idx += 1

    flush_paragraph()
    flush_table()

    if not paper["title"]:
        paper["title"] = _extract_title_from_text(text)

    if preamble_blocks:
        maybe_authors = _maybe_extract_authors(preamble_blocks[0])
        if maybe_authors:
            paper["authors"] = maybe_authors
            preamble_blocks = preamble_blocks[1:]

    if preamble_blocks:
        if paper["sections"]:
            paper["sections"][0]["paragraphs"] = (
                preamble_blocks + paper["sections"][0]["paragraphs"]
            )
        else:
            paper["sections"].append({
                "heading": "1. Content",
                "level": 1,
                "paragraphs": preamble_blocks,
                "figures": [],
                "tables": [],
            })

    normalized_sections = []
    for section in paper["sections"]:
        paragraphs, keywords = _extract_keywords_from_paragraphs(section["paragraphs"])
        if keywords and not paper["keywords"]:
            paper["keywords"] = keywords

        section["paragraphs"] = paragraphs
        heading_kind = _classify_heading(section.get("heading", ""))

        if heading_kind == "abstract":
            abstract_text = "\n\n".join(paragraphs).strip()
            if abstract_text:
                paper["abstract"] = (
                    f"{paper['abstract']}\n\n{abstract_text}".strip()
                    if paper["abstract"]
                    else abstract_text
                )
            continue

        if heading_kind == "references":
            paper["references"].extend(_extract_references(paragraphs))
            continue

        normalized_sections.append(section)

    paper["sections"] = normalized_sections
    _attach_unreferenced_images(paper, image_names, referenced_images)
    return paper


def structure_paper(
    ocr_text: str,
    image_names: list[str],
    llm_config: LLMConfig,
    rewrite: bool = False,
    temperature: float | None = None,
    rewrite_mode: str = "polish",
    target_total_words: int | None = None,
    forced_language: str | None = None,
) -> dict | None:
    """
    Structure OCR/Markdown text into an academic paper.

    Args:
        rewrite: when True, run a follow-up rewrite pass over the structured paper.
        rewrite_mode:
            * ``"polish"``  — light rewrite, preserves length (legacy behaviour).
            * ``"expand"``  — top-conference style expansion. Each section is
              regenerated against the :mod:`style_guide` blueprints with strict
              anti-fabrication rules and per-section word targets.
        target_total_words: optional override for the total body word count
            in ``"expand"`` mode. ``None`` keeps the canonical top-conf
            defaults (~5100 body words, matching the NeurIPS reference).
            A positive value rescales every body section blueprint
            proportionally; abstract / conclusion stay at canonical size.
        forced_language: optional ``"zh"`` or ``"en"`` override from the UI.
            When provided, rewrite/expand prompts and common section headings
            are normalized to that language instead of auto-detecting from
            the source bundle.
    """
    if should_use_native_markdown_parser(ocr_text):
        logger.info("Structured Markdown detected; using native parser")
        paper = parse_markdown_paper(ocr_text, image_names)
        if rewrite and paper and "sections" in paper:
            if rewrite_mode == "expand":
                paper = _expand_sections(
                    paper, llm_config, temperature, target_total_words, forced_language
                )
            else:
                paper = _rewrite_sections(paper, llm_config, temperature, forced_language)
        elif forced_language in ("zh", "en"):
            _apply_target_language_to_headings(paper, forced_language)
        return paper

    llm = _create_llm(llm_config, temperature)
    parser = StrOutputParser()
    chain = structuring_prompt | llm | parser

    image_list = ", ".join(image_names) if image_names else "(no images)"

    logger.info("Calling LLM for paper structuring (model=%s)", llm_config.model)
    response_text = _invoke_chain_with_retry(
        chain,
        {
            "image_list": image_list,
            "ocr_text": ocr_text[:100000],
        },
        "paper structuring",
    )

    paper = extract_json_from_response(response_text)

    if paper is None:
        logger.warning("Failed to parse LLM output as JSON, using fallback")
        paper = _fallback_structure(ocr_text, image_names)

    if rewrite and paper and "sections" in paper:
        if rewrite_mode == "expand":
            paper = _expand_sections(
                paper, llm_config, temperature, target_total_words, forced_language
            )
        else:
            paper = _rewrite_sections(paper, llm_config, temperature, forced_language)
    elif forced_language in ("zh", "en"):
        _apply_target_language_to_headings(paper, forced_language)

    return paper


def structure_paper_stream(
    ocr_text: str,
    image_names: list[str],
    llm_config: LLMConfig,
    temperature: float | None = None,
) -> Iterator[str]:
    """Stream the structuring response for real-time UI updates."""
    if should_use_native_markdown_parser(ocr_text):
        yield json.dumps(
            parse_markdown_paper(ocr_text, image_names),
            ensure_ascii=False,
            indent=2,
        )
        return

    llm = _create_llm(llm_config, temperature)
    parser = StrOutputParser()
    chain = structuring_prompt | llm | parser

    image_list = ", ".join(image_names) if image_names else "(no images)"

    for chunk in chain.stream({
        "image_list": image_list,
        "ocr_text": ocr_text[:100000],
    }):
        if chunk:
            yield chunk


def _expand_sections(
    paper: dict,
    llm_config: LLMConfig,
    temperature: float | None,
    target_total_words: int | None = None,
    forced_language: str | None = None,
) -> dict:
    """Expand each section to top-conference quality using the style guide.

    Unlike :func:`_rewrite_sections` (which only polishes the existing prose),
    this pass:
      * picks a section blueprint (intro / method / experiments / …) for each
        section based on its heading,
      * sends the section content through the LLM with the blueprint's word
        target + must-cover checklist + reference paper rhetorical arc,
      * verifies the rewritten output is at least the blueprint's minimum
        word count; if not, retries once with an explicit "expand more" nudge,
      * preserves figures, tables, and reference handles verbatim — the
        prompt is locked against fabricating new numbers or citations.

    ``target_total_words`` overrides the canonical body-word target. When
    None, the default top-conf blueprints (~5100 body words) are used.
    """
    target_language_code = forced_language if forced_language in ("zh", "en") else detect_paper_language(paper)
    target_language_name = language_name(target_language_code)
    labels = get_labels_for_language(target_language_code)
    _apply_target_language_to_headings(paper, target_language_code)

    llm = _create_llm(llm_config, temperature)
    parser = StrOutputParser()
    expand_chain = expand_prompt | llm | parser
    language_fix_chain = language_fix_prompt | llm | parser
    # LaTeX-safe normalizer: enforces the markdown subset our renderer can
    # convert losslessly. Runs as a separate LLM pass per section so the
    # main expand prompt can stay focused on content quality while this
    # one obsesses over math wrapping / banned commands. See
    # ``_normalize_to_latex_safe_markdown`` for the gate logic.
    latex_safe_chain = latex_safe_normalize_prompt | llm | parser

    blueprints = scaled_blueprints(target_total_words)
    style_guide = compose_style_guide(target_language_name)
    title = paper.get("title", "Untitled Paper")
    abstract_preview = (paper.get("abstract") or "")[:280]
    references_preview = _format_references_for_prompt(paper.get("references") or [])

    if target_total_words:
        logger.info(
            "Starting style-guided expansion for '%s' (%d sections, %s, "
            "user-target ≈ %d total body words)",
            title,
            len(paper.get("sections", [])),
            target_language_name,
            target_total_words,
        )
    else:
        logger.info(
            "Starting style-guided expansion for paper '%s' with %d section(s) in %s",
            title,
            len(paper.get("sections", [])),
            target_language_name,
        )

    # 1. Expand the abstract first (its own blueprint).
    abstract_bp = blueprint_for_in("abstract", blueprints) or blueprints[0]
    abstract = (paper.get("abstract") or "").strip()
    if abstract:
        try:
            logger.info("Expanding abstract (target=%dw)", abstract_bp.target_words)
            new_abstract = _expand_one_section(
                expand_chain=expand_chain,
                language_fix_chain=language_fix_chain,
                latex_safe_chain=latex_safe_chain,
                paper_title=title,
                abstract_preview=abstract_preview,
                references_preview=references_preview,
                section={"figures": [], "tables": []},
                content=abstract,
                blueprint=abstract_bp,
                target_language_name=target_language_name,
                target_language_code=target_language_code,
                style_guide=style_guide,
                heading_label=labels["abstract"],
                duplicate_label=labels["abstract"],
            )
            if new_abstract:
                paper["abstract"] = _clean_llm_section_output(new_abstract)
        except Exception as exc:
            logger.warning("Failed to expand abstract: %s", exc)

    # 2. Expand each body section.
    has_sent_request = bool(abstract)
    fallback_kind_order = (
        "introduction", "related_work", "preliminaries",
        "method", "experiments", "discussion", "conclusion",
    )
    fallback_idx = 0

    for section in paper.get("sections", []):
        heading = section.get("heading", "")
        paragraphs = section.get("paragraphs", [])
        if not paragraphs:
            continue
        content = "\n\n".join(paragraphs)

        bp = blueprint_for_in(heading, blueprints)
        if bp is None:
            # No alias match — use the next blueprint in canonical order so
            # unlabeled sections still get a sensible expansion target.
            kind = fallback_kind_order[
                min(fallback_idx, len(fallback_kind_order) - 1)
            ]
            fallback_idx += 1
            bp = next((b for b in blueprints if b.kind == kind), None)
        if bp is None:
            continue

        try:
            if has_sent_request:
                time.sleep(REWRITE_REQUEST_INTERVAL_SECONDS)
            logger.info(
                "Expanding section '%s' as %s (target=%dw)",
                heading, bp.kind, bp.target_words,
            )
            new_body = _expand_one_section(
                expand_chain=expand_chain,
                language_fix_chain=language_fix_chain,
                latex_safe_chain=latex_safe_chain,
                paper_title=title,
                abstract_preview=abstract_preview,
                references_preview=references_preview,
                section=section,
                content=content,
                blueprint=bp,
                target_language_name=target_language_name,
                target_language_code=target_language_code,
                style_guide=style_guide,
                heading_label=heading,
                duplicate_label=heading,
            )
            if new_body:
                section["paragraphs"] = [_clean_llm_section_output(new_body)]
        except Exception as exc:
            logger.warning("Failed to expand section '%s': %s", heading, exc)
        finally:
            has_sent_request = True

    return paper


def _expand_one_section(
    *,
    expand_chain,
    language_fix_chain,
    latex_safe_chain,
    paper_title: str,
    abstract_preview: str,
    references_preview: str,
    section: dict,
    content: str,
    blueprint: SectionBlueprint,
    target_language_name: str,
    target_language_code: str,
    style_guide: str,
    heading_label: str,
    duplicate_label: str,
) -> str:
    """Run the expand prompt for a single section, with retry-if-too-short."""
    payload = {
        "target_language_name": target_language_name,
        "style_guide": style_guide,
        "section_brief": section_brief(blueprint, target_language_name),
        "paper_title": paper_title,
        "abstract_preview": abstract_preview or "(empty)",
        "references_preview": references_preview,
        "figures_preview": _figures_preview(section),
        "tables_preview": _tables_preview(section),
        "content": content,
        "target_words": blueprint.target_words,
        "minimum_words": blueprint.minimum_words,
    }

    rewritten = _invoke_chain_with_retry(
        expand_chain, payload, f"expand section '{blueprint.kind}'",
    ).strip()

    # Note: we deliberately do NOT retry-if-too-short here. The previous
    # version sent a "your output is too short, please expand" follow-up
    # whenever the LLM's response fell below ``blueprint.minimum_words``,
    # which directly forced the model to invent stats / citations to
    # reach the word target when the input fact-pack didn't have enough
    # raw material. ``target_words`` is now treated strictly as an upper
    # bound: a 400-word honest section is preferred over a 1000-word
    # padded one. If a section comes back unusably short, the right fix
    # is upstream (give the agent more time to gather facts), not here.
    if _word_count(rewritten) < blueprint.minimum_words:
        logger.info(
            "Section '%s' came back short (%d < %d) — accepting as-is "
            "to avoid pressuring the LLM into fabrication. Upstream agent "
            "should gather more facts if a longer section is needed.",
            blueprint.kind, _word_count(rewritten), blueprint.minimum_words,
        )

    if _needs_language_normalization(rewritten, target_language_code):
        logger.info("Normalizing '%s' back to %s", heading_label, target_language_name)
        rewritten = _invoke_chain_with_retry(
            language_fix_chain,
            {
                "target_language_name": target_language_name,
                "heading": heading_label,
                "content": rewritten,
            },
            f"normalize '{blueprint.kind}' language",
        ).strip()

    rewritten = _remove_duplicate_heading_prefix(rewritten, heading_label)
    rewritten = _remove_duplicate_section_label(rewritten, duplicate_label)
    rewritten = _normalize_to_latex_safe_markdown(
        rewritten, latex_safe_chain, label=blueprint.kind,
    )
    return rewritten


def _rewrite_sections(
    paper: dict,
    llm_config: LLMConfig,
    temperature: float | None,
    forced_language: str | None = None,
) -> dict:
    """Rewrite abstract and body sections for improved academic quality."""
    llm = _create_llm(llm_config, temperature)
    parser = StrOutputParser()
    chain = rewrite_prompt | llm | parser
    language_fix_chain = language_fix_prompt | llm | parser
    has_sent_request = False
    target_language_code = forced_language if forced_language in ("zh", "en") else detect_paper_language(paper)
    target_language_name = language_name(target_language_code)
    labels = get_labels_for_language(target_language_code)
    _apply_target_language_to_headings(paper, target_language_code)
    logger.info(
        "Starting LLM rewrite pass for paper '%s' with %d section(s) in %s",
        paper.get("title", "Untitled Paper"),
        len(paper.get("sections", [])),
        target_language_name,
    )

    abstract = paper.get("abstract", "").strip()
    if abstract:
        try:
            logger.info("Calling LLM to rewrite abstract")
            rewritten_abstract = _invoke_chain_with_retry(
                chain,
                {
                    "target_language_name": target_language_name,
                    "heading": labels["abstract"],
                    "content": abstract,
                },
                "rewrite abstract",
            ).strip()
            if _needs_language_normalization(rewritten_abstract, target_language_code):
                logger.info("Normalizing abstract back to %s", target_language_name)
                rewritten_abstract = _invoke_chain_with_retry(
                    language_fix_chain,
                    {
                        "target_language_name": target_language_name,
                        "heading": labels["abstract"],
                        "content": rewritten_abstract,
                    },
                    "normalize abstract language",
                ).strip()
            rewritten_abstract = _remove_duplicate_section_label(
                rewritten_abstract,
                labels["abstract"],
            )
            rewritten_abstract = _remove_duplicate_section_label(
                rewritten_abstract,
                "Abstract",
            )
            if rewritten_abstract:
                paper["abstract"] = _clean_llm_section_output(rewritten_abstract)
        except Exception as e:
            logger.warning("Failed to rewrite abstract: %s", e)
        finally:
            has_sent_request = True

    for section in paper.get("sections", []):
        heading = section.get("heading", "")
        paragraphs = section.get("paragraphs", [])
        if not paragraphs:
            continue

        content = "\n\n".join(paragraphs)

        try:
            if has_sent_request:
                time.sleep(REWRITE_REQUEST_INTERVAL_SECONDS)

            logger.info("Calling LLM to rewrite section '%s'", heading)
            rewritten = _invoke_chain_with_retry(
                chain,
                {
                    "target_language_name": target_language_name,
                    "heading": heading,
                    "content": content,
                },
                f"rewrite section '{heading}'",
            ).strip()
            if _needs_language_normalization(rewritten, target_language_code):
                logger.info(
                    "Normalizing section '%s' back to %s",
                    heading,
                    target_language_name,
                )
                rewritten = _invoke_chain_with_retry(
                    language_fix_chain,
                    {
                        "target_language_name": target_language_name,
                        "heading": heading,
                        "content": rewritten,
                    },
                    f"normalize section '{heading}' language",
                ).strip()
            rewritten = _remove_duplicate_heading_prefix(rewritten, heading)
            if rewritten:
                section["paragraphs"] = [_clean_llm_section_output(rewritten)]
        except Exception as e:
            logger.warning("Failed to rewrite section '%s': %s", heading, e)
        finally:
            has_sent_request = True

    return paper


def _fallback_structure(ocr_text: str, image_names: list[str]) -> dict:
    """Create a basic paper structure from raw text when LLM fails."""
    lines = ocr_text.strip().split("\n")

    title = "Untitled Paper"
    content_start = 0
    for i, line in enumerate(lines):
        stripped = line.strip().lstrip("#").strip()
        if stripped:
            title = stripped
            content_start = i + 1
            break

    paragraphs = []
    current = []
    for line in lines[content_start:]:
        if line.strip():
            current.append(line.strip())
        elif current:
            paragraphs.append(" ".join(current))
            current = []
    if current:
        paragraphs.append(" ".join(current))

    sections = []
    if paragraphs:
        sections.append({
            "heading": "1. Content",
            "level": 1,
            "paragraphs": paragraphs,
            "figures": [
                {"filename": name, "caption": f"Figure {i+1}"}
                for i, name in enumerate(image_names)
            ],
            "tables": [],
        })

    return {
        "title": title,
        "authors": [],
        "abstract": "",
        "keywords": [],
        "sections": sections,
        "references": [],
    }


def _parse_markdown_table(lines: list[str]) -> dict | None:
    """Parse a GitHub-style Markdown table block."""
    if len(lines) < 2:
        return None

    headers = _split_markdown_row(lines[0])
    separators = _split_markdown_row(lines[1])
    if not headers or len(headers) != len(separators):
        return None
    if not all(re.fullmatch(r":?-{3,}:?", cell) for cell in separators):
        return None

    rows = []
    for line in lines[2:]:
        cells = _split_markdown_row(line)
        if not cells:
            continue
        if len(cells) < len(headers):
            cells.extend([""] * (len(headers) - len(cells)))
        rows.append(cells[:len(headers)])

    return {
        "caption": "",
        "headers": headers,
        "rows": rows,
    }


def _split_markdown_row(line: str) -> list[str]:
    stripped = line.strip()
    if not stripped.startswith("|") or not stripped.endswith("|"):
        return []
    return [cell.strip() for cell in stripped.strip("|").split("|")]


def _normalize_heading_label(heading: str) -> str:
    text = heading.strip()
    text = re.sub(r"^\d+(?:\.\d+)*\s*", "", text)
    text = text.strip(" .:：-")
    return re.sub(r"\s+", " ", text).lower()


def _classify_heading(heading: str) -> str:
    normalized = _normalize_heading_label(heading)
    if normalized in ABSTRACT_HEADINGS:
        return "abstract"
    if normalized in REFERENCE_HEADINGS:
        return "references"
    return "section"


def _extract_keywords_from_paragraphs(paragraphs: list[str]) -> tuple[list[str], list[str]]:
    cleaned: list[str] = []
    keywords: list[str] = []

    for paragraph in paragraphs:
        match = KEYWORDS_RE.match(paragraph.strip())
        if match and not keywords:
            keywords = _split_keywords(match.group(2))
            continue
        cleaned.append(paragraph)

    return cleaned, keywords


def _split_keywords(text: str) -> list[str]:
    return [
        keyword.strip()
        for keyword in re.split(r"[;,，；、]", text)
        if keyword.strip()
    ]


def _extract_references(paragraphs: list[str]) -> list[str]:
    refs: list[str] = []
    for paragraph in paragraphs:
        for line in paragraph.splitlines():
            stripped = line.strip()
            stripped = re.sub(r"^[-*]\s+", "", stripped)
            stripped = re.sub(r"^\d+[.)]\s+", "", stripped)
            if stripped:
                refs.append(stripped)
    return refs


def _extract_title_from_text(text: str) -> str:
    for line in text.splitlines():
        stripped = line.strip().lstrip("#").strip()
        if stripped and not (stripped.startswith("<!--") and stripped.endswith("-->")):
            return stripped
    return "Untitled Paper"


def _maybe_extract_authors(block: str) -> list[str]:
    candidate = block.strip().strip("*").strip()
    if not candidate or "\n" in candidate or len(candidate) > 120:
        return []
    if any(mark in candidate for mark in (":", "：", "。", ";", "；", "|")):
        return []
    if not any(sep in candidate for sep in (",", "，", " and ", " & ")):
        return []

    authors = re.split(r",|，|\sand\s|\s&\s", candidate)
    return [author.strip() for author in authors if author.strip()]


def _attach_unreferenced_images(
    paper: dict,
    image_names: list[str],
    referenced_images: set[str],
) -> None:
    existing = {
        fig.get("filename", "")
        for section in paper.get("sections", [])
        for fig in section.get("figures", [])
    }
    remaining_images = [
        name for name in image_names
        if name not in referenced_images and name not in existing
    ]
    if not remaining_images:
        return

    target = _pick_figure_target_section(paper.get("sections", []))
    if target is None:
        target = {
            "heading": "Figures",
            "level": 1,
            "paragraphs": [],
            "figures": [],
            "tables": [],
        }
        paper.setdefault("sections", []).append(target)

    for name in remaining_images:
        target["figures"].append({
            "filename": name,
            "caption": _humanize_filename(name),
        })


def _pick_figure_target_section(sections: list[dict]) -> dict | None:
    for section in sections:
        heading = _normalize_heading_label(section.get("heading", ""))
        if any(keyword in heading for keyword in FIGURE_TARGET_KEYWORDS):
            return section
    return sections[-1] if sections else None


def _humanize_filename(filename: str) -> str:
    caption = Path(filename).stem.replace("_", " ").replace("-", " ").strip()
    return caption or filename
