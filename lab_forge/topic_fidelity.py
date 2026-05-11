"""
Topic-fidelity critic — single-purpose LLM call invoked at generate_report.

What it does
------------
Given (a) the original research topic, (b) the scope plan the agent
committed to via ``submit_research_plan``, and (c) the report the agent
is about to ship, this module asks an LLM ONE focused question:

    "Does the report's title and scope honour the original topic and
    the committed scope plan?"

Nothing else. Not "is the code correct" (other gates handle that), not
"are the references real" (the citation guardrail handles that), not
"is the prose well-written" (the writer LLM handles that). The narrow
prompt is the entire point — Claude Code / Codex achieve scope adherence
by stacking single-purpose critic calls, not by asking one critic to
juggle every concern at once.

Why we don't ship a keyword/method dictionary
---------------------------------------------
The previous attempt (see ``lab-forge/scope_check.py`` in the WIP
sister directory) hard-coded a ``KNOWN_METHODS`` table — Q-learning,
DQN, SARSA, … — and matched against it with regex. That works for one
domain (RL) and breaks the moment the topic shifts to "compare different
optimizers" or "compare different image classifiers". This module
deliberately uses no keyword tables; the LLM infers method-equivalence
from the topic + plan + report at call time, which is domain-agnostic
by construction.

Public API
----------
``build_topic_fidelity_critic(llm)`` returns a callable
``critic(topic, plan, draft) -> FidelityVerdict`` ready to plug into
``GenerateReportTool``. The critic is tolerant of LLM hiccups: any
parsing error or model unavailability degrades to a "neutral" verdict
that does not reject the report (no false-positive rejections), and is
logged for diagnosis.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

from .tools.scope_lock_tool import ResearchPlan

logger = logging.getLogger(__name__)


# Soft caps on what we ship to the LLM. The verdict prompt is small by
# design — we want the critic to read the whole topic + plan + a
# representative slice of the report, not the entire bundle.
_REPORT_TEXT_BUDGET = 6000  # chars
_MIN_FIDELITY_PASS_SCORE = 7  # 0-10; below this we reject the first time

LimitationsCb = Callable[[str], None]


class _LLMLike(Protocol):
    def invoke(self, prompt: str | list[Any], /, **kwargs: Any) -> Any: ...


@dataclass
class FidelityVerdict:
    """Structured output of the critic."""

    fidelity_score: int  # 0-10
    title_matches_topic: bool
    coverage_gaps: list[str] = field(default_factory=list)
    must_fix: list[str] = field(default_factory=list)
    limitations_text: str = ""
    raw_response: str = ""
    parsing_error: str = ""

    @property
    def is_pass(self) -> bool:
        """True iff the report is in good enough shape to ship without rewrite.

        A parsing error fails OPEN (treated as a pass) — the critic is a
        quality enhancement, not a safety gate. We never reject a draft
        because the model wrapped its JSON differently than expected.
        """
        if self.parsing_error:
            return True
        return (
            self.fidelity_score >= _MIN_FIDELITY_PASS_SCORE
            and self.title_matches_topic
            and not self.must_fix
        )

    def render_rejection(self) -> str:
        """Human-readable rejection message returned to the agent."""
        lines = [
            "generate_report REJECTED by topic-fidelity critic.",
            "",
            f"Fidelity score: {self.fidelity_score}/10  "
            f"(needs ≥{_MIN_FIDELITY_PASS_SCORE} to pass)",
            f"Title matches topic: {self.title_matches_topic}",
        ]
        if self.coverage_gaps:
            lines.append("")
            lines.append("Coverage gaps (committed in your plan but missing in the report):")
            for gap in self.coverage_gaps:
                lines.append(f"  • {gap}")
        if self.must_fix:
            lines.append("")
            lines.append("Required fixes before resubmitting:")
            for fix in self.must_fix:
                lines.append(f"  • {fix}")
        lines.append("")
        lines.append(
            "Resubmit generate_report with these fixes applied. You have ONE "
            "more attempt before the report is accepted with an explicit "
            "Limitations section appended automatically."
        )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a topic-fidelity reviewer for a research-agent system.

You judge ONE thing: does the report's title and overall scope honour the
original research topic and the scope plan the agent committed to before
running experiments?

You do NOT evaluate:
  • whether the code is correct — another gate handles that
  • whether references are real — another gate handles that
  • whether prose flows well — the writer LLM handles that
  • whether numbers are accurate — a results guardrail handles that

Stay narrow. Reject only when the report's title or coverage is materially
narrower than the topic + plan promise. Do NOT reject for stylistic issues.

Reply with ONE JSON object, no prose around it:
{
  "fidelity_score": <integer 0-10>,
  "title_matches_topic": <true | false>,
  "coverage_gaps": [<short strings: items the plan committed but the report omits>],
  "must_fix": [<short strings: concrete edits the agent should apply on rewrite. Empty list when fidelity_score >= 7>],
  "limitations_text": "<1-2 sentence Limitations paragraph honestly stating what was NOT covered. Always populate; used as a fallback if rewrite still fails>"
}

Scoring rubric:
  10 — title and scope mirror the topic exactly; every plan item is reflected
   8 — minor narrowing acknowledged in abstract; reasonable
   7 — acceptable: some narrowing but title still answers the topic-level question
   5 — title is dataset-specific or env-specific when the topic was abstract
   3 — only one or two plan items covered; title silently shrunk
   0 — report addresses a different topic entirely
"""


_USER_TEMPLATE = """\
ORIGINAL RESEARCH TOPIC (verbatim from the user):
---
{topic}
---

SCOPE PLAN the agent committed to BEFORE running experiments:
---
{plan_json}
---

RUN ENVIRONMENT NOTES (deterministic facts about this run):
{env_notes}

REPORT the agent is now trying to submit:

Title:
{title}

Abstract:
{abstract}

Body excerpt (truncated to {budget} chars):
---
{body}
---

Apply the rubric. Output the JSON object."""


_LITERATURE_UNAVAILABLE_NOTE = (
    "  • Literature search was UNAVAILABLE for this run (empty "
    "literature_cache.jsonl, no successful read_paper_fulltext call). "
    "Treat literature_only methods in the plan as legitimately "
    "uncoverable — do NOT count them as coverage_gaps and do NOT add "
    "them to must_fix. The report tool has already auto-prepended a "
    "Limitations paragraph disclosing this; you should still verify "
    "the title/abstract honour the topic-level question."
)
_LITERATURE_AVAILABLE_NOTE = (
    "  • Literature search was available for this run; ordinary "
    "coverage rules apply (literature_only methods should appear in "
    "Related Work / Methodology)."
)


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------

def build_topic_fidelity_critic(
    llm: _LLMLike | None,
) -> Callable[..., FidelityVerdict] | None:
    """Return a critic callable, or ``None`` when no LLM is available.

    The returned callable signature::

        critic(
            *,
            topic: str,                  # original user topic
            plan: ResearchPlan | None,   # locked scope plan; may be None
            draft: dict[str, str],       # title, abstract, body
            literature_unavailable: bool = False,  # opt-in env note
        ) -> FidelityVerdict

    ``literature_unavailable`` softens the rubric so methods committed
    to ``feasibility_split.literature_only`` aren't flagged as
    coverage_gaps when this run literally couldn't reach a literature
    backend. The report tool detects this state from the workspace
    (see ``result_guardrails.is_literature_unavailable``) and threads
    it through here.
    """
    if llm is None:
        return None

    def _critic(
        topic: str,
        plan: "ResearchPlan | None",
        draft: dict[str, str],
        literature_unavailable: bool = False,
    ) -> FidelityVerdict:
        return _run_critic(
            llm, topic, plan, draft, literature_unavailable=literature_unavailable
        )

    return _critic


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

def _run_critic(
    llm: _LLMLike,
    topic: str,
    plan: "ResearchPlan | None",
    draft: dict[str, str],
    literature_unavailable: bool = False,
) -> FidelityVerdict:
    plan_json = json.dumps(
        plan.to_dict() if plan is not None else {"_note": "no plan was locked for this run"},
        ensure_ascii=False,
        indent=2,
    )
    body = (draft.get("body") or "")[:_REPORT_TEXT_BUDGET]
    env_notes = (
        _LITERATURE_UNAVAILABLE_NOTE if literature_unavailable
        else _LITERATURE_AVAILABLE_NOTE
    )
    user = _USER_TEMPLATE.format(
        topic=(topic or "").strip() or "(empty topic)",
        plan_json=plan_json,
        env_notes=env_notes,
        title=draft.get("title", "").strip() or "(empty title)",
        abstract=draft.get("abstract", "").strip() or "(empty abstract)",
        body=body or "(empty body)",
        budget=_REPORT_TEXT_BUDGET,
    )

    try:
        response = llm.invoke([
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ])
    except Exception as exc:  # noqa: BLE001 — model errors must not crash the run
        logger.warning("Topic-fidelity critic LLM call failed: %s", exc)
        return _neutral_verdict(parsing_error=f"llm_invoke_error: {exc}")

    content = _extract_text(response)
    return _parse_verdict(content)


def _extract_text(response: Any) -> str:
    """Pull a text payload out of a LangChain message-like response."""
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


_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


def _parse_verdict(text: str) -> FidelityVerdict:
    """Best-effort parse of the critic's JSON reply.

    LLMs sometimes wrap the JSON in ```json fences``` or precede it with
    a sentence of prose. We strip both.
    """
    text = (text or "").strip()
    if not text:
        return _neutral_verdict(parsing_error="empty_response")

    candidate: str | None = None
    fence_match = _JSON_FENCE_RE.search(text)
    if fence_match:
        candidate = fence_match.group(1)
    else:
        # Find the first balanced { ... } block.
        first = text.find("{")
        last = text.rfind("}")
        if first != -1 and last > first:
            candidate = text[first : last + 1]

    if candidate is None:
        return _neutral_verdict(
            raw=text,
            parsing_error="no_json_object_found",
        )

    try:
        data = json.loads(candidate)
    except json.JSONDecodeError as exc:
        return _neutral_verdict(
            raw=text,
            parsing_error=f"json_decode_error: {exc}",
        )

    score_raw = data.get("fidelity_score", 5)
    try:
        score = int(score_raw)
    except (TypeError, ValueError):
        score = 5
    score = max(0, min(10, score))

    return FidelityVerdict(
        fidelity_score=score,
        title_matches_topic=bool(data.get("title_matches_topic", True)),
        coverage_gaps=_coerce_str_list(data.get("coverage_gaps")),
        must_fix=_coerce_str_list(data.get("must_fix")),
        limitations_text=str(data.get("limitations_text") or "").strip(),
        raw_response=text,
    )


def _neutral_verdict(*, raw: str = "", parsing_error: str = "") -> FidelityVerdict:
    """A non-rejecting verdict used when we can't trust the critic's output.

    Failing open is the right call here: the critic is a quality
    enhancement, not a safety gate. Other guardrails (citations, results,
    artifact paths) still run unconditionally.
    """
    return FidelityVerdict(
        fidelity_score=_MIN_FIDELITY_PASS_SCORE,
        title_matches_topic=True,
        coverage_gaps=[],
        must_fix=[],
        limitations_text="",
        raw_response=raw,
        parsing_error=parsing_error,
    )


def _coerce_str_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        # Tolerate comma-separated single-string replies.
        return [s.strip() for s in value.split(",") if s.strip()]
    if isinstance(value, list):
        out: list[str] = []
        for item in value:
            if isinstance(item, str):
                txt = item.strip()
            elif isinstance(item, dict):
                txt = str(item.get("text") or item.get("name") or "").strip()
            else:
                txt = str(item).strip()
            if txt:
                out.append(txt)
        return out
    return []
