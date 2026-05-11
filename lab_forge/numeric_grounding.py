"""
Numeric-claim grounding — catch fabricated numbers in report prose.

Why this exists
---------------
Trajectory 194edbd4 shipped a paper containing claims like
"reducing multi-agent path conflicts by 63%", "92% success rates",
"Amazon's warehouse robots exceeding 750,000 units in 2023" — none
of which appeared in any data file the run produced. Pure LLM
hallucination, undetectable by any of our existing gates because:
  • the citation guardrail only checks reference list entries (paper
    titles), not inline numeric claims;
  • the result guardrail only checks for inf / NaN in CSVs;
  • the topic-fidelity critic judges scope, not numeric truth.

Frontier solution: groundedness verification. Bing Chat / Perplexity
trace each cited claim to a source span; FActScore (Min et al. 2023)
extracts atomic facts and checks each. We do a deterministic Python
version: extract claim-shaped numbers from prose, then look for each
in the run's data files (CSV / JSON / log) and the literature cache.
Numbers we can't find → flag as potentially fabricated.

Design choices to minimise false positives
------------------------------------------
The prose of a real research paper is full of numbers that legitimately
do NOT appear in data files: years, citation indices, page numbers,
math constants, hyperparameter settings written in prose without being
logged. We aggressively whitelist these:

  • years 1990-2099                          → skip
  • citation forms ``[N]`` and ``(N)``       → skip
  • section numbers ``§3.1`` / ``1.2.3``     → skip
  • single digits ≤ 9                        → skip (too noisy)
  • numbers paired with math constants
    (``ε = 10^-3``, ``α = 0.1``)              → skip
  • numbers inside the agent's research_plan.json's rationale → skip
    (the agent's planning text isn't a numerical claim about the run)

The remaining "claim-shaped" numbers are:
  • percentages (``42%``, ``42.5%``)
  • large counts with separators (``750,000``, ``1.7 million``)
  • metrics with units (``42ms``, ``42x``, ``42 fps``)
  • "improved by N", "achieved N accuracy", "N reduction", etc.

For each, we check:
  1) numeric value appears (with tolerance) in any CSV / JSON / log
  2) numeric value appears in any literature_cache.jsonl abstract
  3) numeric value appears in any papers/**/manifest.json content

If none → flag as ungrounded. The downstream gate in report_tool turns
findings into a soft warning by default (logged + returned in metadata)
since the false-positive risk is real; a future tightening could make
it a hard reject for percentages-only or for numbers > 100.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class UngroundedClaim:
    """One numeric claim from prose that we couldn't find in evidence files."""

    value: float
    matched_text: str  # the regex-matched substring, e.g. "92%" or "750,000"
    context: str  # surrounding ~80 chars of prose for human review

    def render(self) -> str:
        return f"{self.matched_text!r} (value≈{self.value:g}) — context: …{self.context}…"


# ---------------------------------------------------------------------------
# Extraction patterns
# ---------------------------------------------------------------------------

# Numbers we extract as "claim-shaped". Each pattern captures a number
# substring + optionally a unit. We deliberately do NOT match every
# integer in the prose — that would produce too many false positives.
# The patterns target the specific shapes that LLMs tend to fabricate
# in scientific prose.
_PERCENT_RE = re.compile(r"(?<![A-Za-z])(\d{1,3}(?:\.\d+)?)\s*%(?!\w)")
_LARGE_COUNT_RE = re.compile(
    r"(?<![\d.])"
    r"(\d{1,3}(?:,\d{3})+(?:\.\d+)?)"  # 1,000 or 1,000,000.5
    r"(?![\d.])"
)
_MILLION_BILLION_RE = re.compile(
    r"(?<![A-Za-z\d])(\d+(?:\.\d+)?)\s*(million|billion|thousand|k|m|bn)\b",
    re.IGNORECASE,
)
_METRIC_WITH_UNIT_RE = re.compile(
    r"(?<![A-Za-z\d])"
    r"(\d{1,5}(?:\.\d+)?)\s*"
    r"(ms|fps|gb|mb|kb|gflops|tflops|x|×)"
    r"\b",
    re.IGNORECASE,
)

# Patterns we explicitly EXCLUDE. Each captures something that looks
# numeric but isn't a fabricable scientific claim.
_YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")
_CITATION_RE = re.compile(r"[\[\(]\s*\d{1,3}\s*[\]\)]")  # [3], (12)
_SECTION_NUM_RE = re.compile(
    # Require explicit "§" prefix or "Section/Chapter/Figure/Table" word
    # before the number — without this guard the previous regex matched
    # any "X.Y" in prose ("exceed 1.7 million" was wrongly classified
    # as a section reference and the claim slipped through).
    r"(?:§|\b[Ss]ection\s+|\b[Cc]hapter\s+|\b[Ff]igure\s+|\b[Tt]able\s+|\b[Ee]q(?:uation)?\.?\s+)\s*\d+(?:\.\d+){1,2}"
)
_MATH_CONSTANT_RE = re.compile(
    r"\b(?:epsilon|alpha|beta|gamma|delta|sigma|lambda|"
    r"e|pi|tau|phi|η|ε|α|β|γ|δ|σ|λ|τ|φ|μ|ρ|θ)\s*=\s*\d",
    re.IGNORECASE,
)
_HYPER_TINY_RE = re.compile(r"10\s*[\^*]\s*-?\d+")  # 10^-3, 10^3


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def find_ungrounded_numeric_claims(
    *,
    working_dir: str | Path,
    prose: str,
    tolerance: float = 0.01,
) -> list[UngroundedClaim]:
    """Return numeric claims from ``prose`` that we cannot trace to evidence.

    ``tolerance`` is the relative tolerance for matching (default 1%) —
    "92%" in prose matches "0.92" or "91.5%" or "92.4%" in evidence.
    Set lower for stricter matching.
    """
    if not prose or not prose.strip():
        return []
    workdir = Path(working_dir)
    if not workdir.exists():
        # No workspace = no evidence to ground against. Surface every
        # claim so the caller sees the (unusual) situation.
        # In production the report tool always passes a real workspace.
        return _extract_claims(prose)

    evidence_text = _gather_evidence_text(workdir)
    evidence_numbers = _extract_all_numbers(evidence_text)

    findings: list[UngroundedClaim] = []
    seen: set[str] = set()
    for claim in _extract_claims(prose):
        key = f"{claim.value:.4f}|{claim.matched_text.lower()}"
        if key in seen:
            continue
        seen.add(key)
        if not _is_grounded(claim.value, evidence_numbers, tolerance):
            findings.append(claim)
    return findings


# ---------------------------------------------------------------------------
# Claim extraction
# ---------------------------------------------------------------------------

def _extract_claims(prose: str) -> list[UngroundedClaim]:
    """Find every claim-shaped number in ``prose``."""
    text = prose
    out: list[UngroundedClaim] = []

    for match in _PERCENT_RE.finditer(text):
        if _in_excluded_span(match, text):
            continue
        try:
            value = float(match.group(1))
        except ValueError:
            continue
        # Percentages > 100 are usually citations or fictional ("90% of
        # 750000 = 675000"). Keep them — fabrication risk is real.
        if value < 0:
            continue
        out.append(UngroundedClaim(
            value=value,
            matched_text=match.group(0).strip(),
            context=_context_around(text, match),
        ))

    for match in _LARGE_COUNT_RE.finditer(text):
        if _in_excluded_span(match, text):
            continue
        try:
            value = float(match.group(1).replace(",", ""))
        except ValueError:
            continue
        if value < 100:  # below 100 a comma-formatted number is nonsense
            continue
        out.append(UngroundedClaim(
            value=value,
            matched_text=match.group(0).strip(),
            context=_context_around(text, match),
        ))

    for match in _MILLION_BILLION_RE.finditer(text):
        if _in_excluded_span(match, text):
            continue
        try:
            base = float(match.group(1))
        except ValueError:
            continue
        unit = match.group(2).lower()
        scale = {
            "thousand": 1_000.0, "k": 1_000.0,
            "million": 1_000_000.0, "m": 1_000_000.0,
            "billion": 1_000_000_000.0, "bn": 1_000_000_000.0,
        }.get(unit, 1.0)
        out.append(UngroundedClaim(
            value=base * scale,
            matched_text=match.group(0).strip(),
            context=_context_around(text, match),
        ))

    for match in _METRIC_WITH_UNIT_RE.finditer(text):
        if _in_excluded_span(match, text):
            continue
        try:
            value = float(match.group(1))
        except ValueError:
            continue
        out.append(UngroundedClaim(
            value=value,
            matched_text=match.group(0).strip(),
            context=_context_around(text, match),
        ))

    return out


def _in_excluded_span(match: re.Match, text: str) -> bool:
    """True when ``match`` overlaps a pattern we explicitly skip.

    Avoids flagging years (``2023``), citation refs (``[5]``), section
    numbers (``§3.1``), and math constants (``ε = 10^-3``) as claims.
    """
    span = match.span()
    for excl_re in (
        _YEAR_RE,
        _CITATION_RE,
        _SECTION_NUM_RE,
        _MATH_CONSTANT_RE,
        _HYPER_TINY_RE,
    ):
        for excl in excl_re.finditer(text):
            es = excl.span()
            if not (span[1] <= es[0] or es[1] <= span[0]):
                return True
    return False


def _context_around(text: str, match: re.Match, radius: int = 40) -> str:
    """Return ~80 chars of prose around the match for human review."""
    start = max(0, match.start() - radius)
    end = min(len(text), match.end() + radius)
    snippet = text[start:end].replace("\n", " ").strip()
    return snippet


# ---------------------------------------------------------------------------
# Evidence collection
# ---------------------------------------------------------------------------

# Files we treat as evidence. Anything we'd legitimately cite from in
# the paper has to pass through one of these (csv/json/log) or live in
# the literature cache (paper abstracts).
_EVIDENCE_GLOBS = (
    "*.csv", "*.tsv", "*.json", "*.jsonl",
    "logs/*.log",
    "papers/**/manifest.json",
)
# Files we EXCLUDE from evidence collection because they contain the
# agent's PROSE, not measured data. Counting prose as evidence would
# make every fabricated number self-grounding.
_EXCLUDED_EVIDENCE = (
    "research_report.md",
    "paperforge_bundle.json",
    "research_plan.json",  # plan rationale is prose
    "research_paper.tex",
)


def _gather_evidence_text(workdir: Path) -> str:
    """Concatenate the textual content of every evidence file."""
    parts: list[str] = []
    seen: set[Path] = set()
    for pattern in _EVIDENCE_GLOBS:
        for path in workdir.glob(pattern):
            if not path.is_file():
                continue
            if path.name in _EXCLUDED_EVIDENCE:
                continue
            if path in seen:
                continue
            seen.add(path)
            try:
                parts.append(path.read_text(encoding="utf-8", errors="replace"))
            except OSError:
                continue
    return "\n".join(parts)


# Match any number-like substring in the evidence: integers, decimals,
# scientific notation. Captures the raw token; downstream parses to
# float.
_EVIDENCE_NUM_RE = re.compile(
    r"(?<![A-Za-z\d_])"
    r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?"
    r"(?![A-Za-z\d_])"
)


def _extract_all_numbers(text: str) -> list[float]:
    """Pull every numeric token out of evidence text as a list of floats.

    We deliberately do NOT bucket — bucketing might miss a ``92.34`` that
    matches a prose claim of ``92%``. The grounding check uses tolerance
    matching downstream.
    """
    nums: list[float] = []
    for match in _EVIDENCE_NUM_RE.finditer(text):
        try:
            nums.append(float(match.group(0)))
        except ValueError:
            continue
    return nums


# ---------------------------------------------------------------------------
# Grounding check
# ---------------------------------------------------------------------------

def _is_grounded(claim: float, evidence: Iterable[float], tolerance: float) -> bool:
    """True iff some number in ``evidence`` is within tolerance of ``claim``.

    For percentages, the matching unit could be either "92" (when written
    as a percentage) or "0.92" (when written as a fraction). Try both —
    a small tolerance still keeps the check meaningful.
    """
    if claim == 0:
        # Zero appears everywhere in evidence as boilerplate ("step 0",
        # "iter 0", ""). Don't bother grounding zero claims.
        return True
    for ev in evidence:
        if _close(ev, claim, tolerance):
            return True
        # Percentage <-> fraction equivalence
        if _close(ev * 100.0, claim, tolerance):
            return True
        if _close(ev, claim / 100.0, tolerance):
            return True
    return False


def _close(a: float, b: float, tolerance: float) -> bool:
    """Relative-tolerance comparison, with absolute fallback for tiny vals."""
    if a == b:
        return True
    denom = max(abs(a), abs(b))
    if denom < 1.0:
        return abs(a - b) <= tolerance
    return abs(a - b) / denom <= tolerance


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def render_findings(findings: Iterable[UngroundedClaim], *, max_items: int = 8) -> str:
    """Format findings as a single-paragraph rejection message."""
    items = list(findings)
    if not items:
        return ""
    head = items[:max_items]
    rendered = "\n".join(f"  • {f.render()}" for f in head)
    extra = ""
    if len(items) > max_items:
        extra = f"\n  • ... and {len(items) - max_items} more ungrounded claim(s)."
    return rendered + extra
