"""
Vision-LLM figure quality check (opt-in).

What it catches
---------------
The blank-PNG guardrail in ``result_guardrails.validate_figure_content``
catches all-white / all-black images via std-of-pixels. But it cannot
catch figures that are technically populated yet semantically wrong:

  * a "learning curve" that's a flat line at zero
  * a "path planning" figure showing the agent stuck in one cell
  * a scatter plot whose axes don't reflect the experiment description
  * an ROC where TPR=FPR (random predictor disguised as a result)

These need a model that can SEE the image and judge whether it
plausibly corresponds to the topic / methodology described in the
report. Frontier coding agents (Claude Code's Computer Use, Cursor's
Composer with vision, GPT-4V tool wrappers) all use a vision-capable
LLM for this — there's no robust pure-Python alternative.

Default behaviour: SKIP
-----------------------
This module is opt-in. The user's deployment may not have a vision-
capable LLM configured (e.g. AI Studio's deepseek-v3 is text-only).
When ``vision_llm`` is ``None``:
  * ``assess_figure_semantics()`` returns ``[]`` immediately
  * The report tool does NOT block on missing vision capability

That keeps the existing pipeline functional for text-only setups while
exposing a clean integration point. To enable, the user wires a vision-
capable LangChain chat model into ``GenerateReportTool``.

When enabled
------------
For each figure, encode to base64, attach as an image_url message part
(LangChain v1+ multimodal format), and ask:
  "Given the topic ``<topic>`` and the report's claim that this figure
   shows ``<caption>``, does the image plausibly support that claim?
   Output JSON: {sensible: true|false, reason: '...'}"

A ``sensible: false`` verdict becomes a finding the report tool can
reject on. Same fail-open contract as the text critics — any LLM error
is logged and treated as "skip this figure", never as "reject the
paper".
"""

from __future__ import annotations

import base64
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Protocol

logger = logging.getLogger(__name__)


# Hard caps. Vision LLM calls are slow and expensive — limit how many
# figures we send and how big each payload is.
_MAX_FIGURES_PER_RUN = 8
_MAX_IMAGE_BYTES = 1_500_000  # 1.5 MB; bigger gets skipped with a warning


class _VisionLLMLike(Protocol):
    """LangChain-compatible chat model that accepts multimodal content."""

    def invoke(self, messages: list[Any], /, **kwargs: Any) -> Any: ...


@dataclass(frozen=True)
class FigureSemanticFinding:
    """One figure that the vision LLM flagged as not-sensible."""

    path: str
    reason: str

    def render(self) -> str:
        return f"{self.path} — {self.reason}"


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def assess_figure_semantics(
    *,
    working_dir: str | Path,
    figure_entries: Iterable[Any],
    topic: str,
    vision_llm: _VisionLLMLike | None,
) -> list[FigureSemanticFinding]:
    """Return findings for figures the vision LLM judges not-sensible.

    Returns ``[]`` immediately when:
      * ``vision_llm`` is None (no vision capability configured) — the
        most common case; deepseek-v3 / non-vision text models;
      * no figure entries supplied;
      * the workspace doesn't exist.

    The check is rate-limited to ``_MAX_FIGURES_PER_RUN`` figures to
    keep wall time and cost bounded; figures past the cap are skipped
    with a single info-level log line.
    """
    if vision_llm is None or not figure_entries:
        return []
    workdir = Path(working_dir)
    if not workdir.exists():
        return []

    paths = _extract_figure_paths(figure_entries)
    if not paths:
        return []
    if len(paths) > _MAX_FIGURES_PER_RUN:
        logger.info(
            "vision figure check capped at %d/%d figures (set _MAX_FIGURES_PER_RUN to widen)",
            _MAX_FIGURES_PER_RUN, len(paths),
        )
        paths = paths[:_MAX_FIGURES_PER_RUN]

    findings: list[FigureSemanticFinding] = []
    for entry, raw in paths:
        full = (workdir / raw).resolve()
        if not full.is_file():
            continue
        if full.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
            # SVG/PDF require a different attachment shape; skip cleanly.
            continue
        try:
            data = full.read_bytes()
        except OSError as exc:
            logger.debug("vision check skip (unreadable): %s — %s", raw, exc)
            continue
        if len(data) > _MAX_IMAGE_BYTES:
            logger.info("vision check skip (too large %d B): %s", len(data), raw)
            continue
        finding = _judge_one_figure(
            llm=vision_llm,
            topic=topic,
            caption=entry.get("caption", "") if isinstance(entry, dict) else "",
            image_bytes=data,
            mime=_mime_for_suffix(full.suffix),
            relative_path=raw,
        )
        if finding is not None:
            findings.append(finding)
    return findings


# ---------------------------------------------------------------------------
# Per-figure judge
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a vision reviewer for a research-agent system.

You judge ONE thing: does the attached image plausibly support the
claim the agent made about it (the figure caption + the research topic)?

You do NOT evaluate aesthetics, axis labels, or publication style. You
only check semantic sensibility:
  • Is the figure populated with data (not a flat line, not a single
    point oscillating, not all-zeros)?
  • Does the apparent shape match the kind of result described in the
    caption (e.g. a "learning curve" should trend; a "path on a grid"
    should traverse cells; a "comparison bar chart" should have visible
    bars)?
  • Are there signs the underlying experiment failed (a "training
    accuracy" plot stuck at chance, a "loss curve" that is flat, a
    path-planning trajectory frozen at one cell)?

Reply with ONE JSON object, no prose around it:
  {"sensible": true | false, "reason": "<short string. when false, name
   the specific visual cue. when true, the empty string is fine.>"}

Be strict but not pedantic. ``sensible: false`` is for figures that
clearly do not show what the caption claims; minor styling issues are
``sensible: true``. Fail-open on any genuine ambiguity.
"""


def _judge_one_figure(
    *,
    llm: _VisionLLMLike,
    topic: str,
    caption: str,
    image_bytes: bytes,
    mime: str,
    relative_path: str,
) -> FigureSemanticFinding | None:
    b64 = base64.b64encode(image_bytes).decode("ascii")
    text_part = (
        f"Research topic: {topic or '(unspecified)'}\n"
        f"Figure path: {relative_path}\n"
        f"Caption the agent attached: {caption or '(no caption)'}\n\n"
        "Apply the rubric and output the JSON object."
    )
    user_msg = {
        "role": "user",
        "content": [
            {"type": "text", "text": text_part},
            {
                "type": "image_url",
                "image_url": {"url": f"data:{mime};base64,{b64}"},
            },
        ],
    }
    try:
        response = llm.invoke([
            {"role": "system", "content": _SYSTEM_PROMPT},
            user_msg,
        ])
    except Exception as exc:  # noqa: BLE001
        logger.warning("vision figure check LLM call failed for %s: %s", relative_path, exc)
        return None

    content = _extract_text(response)
    parsed = _parse_verdict(content)
    if parsed is None:
        return None
    sensible, reason = parsed
    if sensible:
        return None
    return FigureSemanticFinding(path=relative_path, reason=reason or "vision judge flagged image as not-sensible")


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


def _parse_verdict(text: str) -> tuple[bool, str] | None:
    """Best-effort parse of the vision LLM's JSON reply.

    Returns ``None`` (treated as fail-open: don't reject the figure)
    on any parse failure.
    """
    text = (text or "").strip()
    if not text:
        return None
    candidate: str | None = None
    fence_match = _JSON_FENCE_RE.search(text)
    if fence_match:
        candidate = fence_match.group(1)
    else:
        first = text.find("{")
        last = text.rfind("}")
        if first != -1 and last > first:
            candidate = text[first : last + 1]
    if candidate is None:
        return None
    try:
        data = json.loads(candidate)
    except json.JSONDecodeError:
        return None
    sensible_raw = data.get("sensible")
    if isinstance(sensible_raw, bool):
        sensible = sensible_raw
    elif isinstance(sensible_raw, str):
        sensible = sensible_raw.strip().lower() not in {"false", "no", "0"}
    else:
        # Unknown verdict shape: fail open
        return None
    reason = str(data.get("reason") or "").strip()[:300]
    return sensible, reason


def _extract_text(response: Any) -> str:
    if isinstance(response, str):
        return response
    content = getattr(response, "content", response)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                txt = item.get("text")
                if isinstance(txt, str):
                    parts.append(txt)
        return "\n".join(parts)
    return str(content or "")


def _extract_figure_paths(entries: Iterable[Any]) -> list[tuple[Any, str]]:
    """Pull (entry, path) pairs out of figure entries."""
    out: list[tuple[Any, str]] = []
    for entry in entries:
        if isinstance(entry, dict):
            raw = (entry.get("path") or "").strip()
            if raw:
                out.append((entry, raw))
        elif isinstance(entry, str) and entry.strip():
            out.append(({}, entry.strip()))
    return out


def _mime_for_suffix(suffix: str) -> str:
    suffix = suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        return "image/jpeg"
    return "image/png"


def render_findings(findings: Iterable[FigureSemanticFinding], *, max_items: int = 6) -> str:
    items = list(findings)
    if not items:
        return ""
    head = items[:max_items]
    rendered = "\n".join(f"  • {f.render()}" for f in head)
    extra = ""
    if len(items) > max_items:
        extra = f"\n  • ... and {len(items) - max_items} more."
    return rendered + extra
