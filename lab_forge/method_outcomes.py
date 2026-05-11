"""
Outcome-aware writing — LLM-judged per-method outcomes from raw evidence.

Why this exists
---------------
Trajectory 194edbd4 shipped a paper that praised "Potential Fields"
("92% success rate", "63% conflict reduction") even though the run's
own ``robust_path_planning_comparison.csv`` showed PF failing both
scenarios with ``path_length=inf``. The agent's prose contradicted its
own data and no gate caught it: the result-guardrail saw the inf and
the disclosure-sniff was satisfied by stray "failures" / "crash" words
elsewhere in the prose.

The frontier-agent answer (Reflexion / Self-Refine / outcome-aware
writing — see Madaan et al. 2023, Shinn et al. 2023) is to make a
SEPARATE LLM call whose only job is to read the raw experimental
artifacts and label each method as success / partial / failure with
evidence pointers. That structured outcome table is then injected
into the report as ground truth — the writer can elaborate around it
but cannot contradict it without the contradiction being obvious.

This module is the LabForge implementation of that pattern. It is:
  * single-purpose (one LLM call, narrow prompt — does NOT judge the
    whole paper, just the per-method outcomes);
  * domain-agnostic (no method dictionaries, no topic templates —
    the LLM infers outcomes from whatever evidence the workspace has);
  * fail-safe (any LLM error returns an empty outcome map; the gate
    becomes a no-op rather than crashing the run);
  * cheap to skip (when no plan is locked or no evidence files exist
    we don't even invoke the LLM).
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Protocol

from .tools.scope_lock_tool import ResearchPlan

logger = logging.getLogger(__name__)


# Filename relative to workspace; persisted so the report tool can audit
# what it injected and downstream review can verify the agent's prose
# matches.
OUTCOMES_FILENAME = "method_outcomes.json"

# Hard caps on what we feed the LLM — single-purpose prompts should stay
# small. We sample evidence rather than dump everything to keep the call
# cheap and the verdict focused.
_MAX_LOG_FILES_PER_METHOD = 3
_MAX_LOG_CHARS = 800
_MAX_CSV_ROWS = 12


class _LLMLike(Protocol):
    def invoke(self, prompt: list[dict] | str, /, **kwargs: Any) -> Any: ...


@dataclass
class MethodOutcome:
    """Structured outcome for one method, derived from raw evidence."""

    method: str
    outcome: str  # "success" | "partial" | "implementation_failure" | "data_failure" | "not_attempted"
    evidence_pointer: str  # short human-readable evidence trail
    honest_summary: str  # 1-2 sentence neutral statement of what happened

    def to_dict(self) -> dict[str, str]:
        return {
            "method": self.method,
            "outcome": self.outcome,
            "evidence_pointer": self.evidence_pointer,
            "honest_summary": self.honest_summary,
        }


# Lowercased outcome labels we accept from the LLM. Other labels are
# remapped to "partial" since the LLM may emit synonyms.
_VALID_OUTCOMES = {
    "success",
    "partial",
    "implementation_failure",
    "data_failure",
    "not_attempted",
}


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


def assess_method_outcomes(
    *,
    working_dir: str | Path,
    plan: ResearchPlan | None,
    llm: _LLMLike | None,
) -> list[MethodOutcome]:
    """Return one MethodOutcome per ``plan.will_execute`` method.

    Returns ``[]`` when:
      * no plan was locked (nothing to assess against);
      * no LLM available (cannot judge);
      * an unrecoverable error occurred (logged, fail-open).

    Persists the outcome map to ``method_outcomes.json`` on success so
    the report tool can audit what was injected.
    """
    if plan is None or not plan.will_execute or llm is None:
        return []
    workdir = Path(working_dir)
    if not workdir.exists():
        return []

    evidence = _collect_evidence(workdir)
    try:
        outcomes = _run_outcome_llm(plan.will_execute, evidence, llm)
    except Exception as exc:  # noqa: BLE001 — never crash the report tool
        logger.warning("Method-outcome LLM call failed; skipping: %s", exc)
        return []

    if outcomes:
        try:
            (workdir / OUTCOMES_FILENAME).write_text(
                json.dumps([o.to_dict() for o in outcomes], ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except OSError as exc:
            logger.debug("method_outcomes.json write skipped: %s", exc)
    return outcomes


def render_outcomes_block(outcomes: Iterable[MethodOutcome]) -> str:
    """Render the outcomes as a markdown block to inject into the report.

    Format is a fenced fact block — the writer LLM expansion sees this
    as authoritative source content and won't paraphrase it away. The
    reader sees a clean "Method-level outcomes" subsection.
    """
    items = list(outcomes)
    if not items:
        return ""
    lines = [
        "**Method-level outcomes (auto-generated from this run's execution evidence — do NOT contradict in prose):**",
        "",
    ]
    for o in items:
        lines.append(f"- **{o.method}** — outcome: `{o.outcome}`. {o.honest_summary} _Evidence: {o.evidence_pointer}_")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Evidence collection (deterministic, no LLM)
# ---------------------------------------------------------------------------


def _collect_evidence(workdir: Path) -> dict[str, Any]:
    """Gather a compact evidence dossier from the workspace."""
    return {
        "csv_files": _summarise_csvs(workdir),
        "log_excerpts": _sample_logs(workdir),
        "figure_files": sorted(
            str(p.relative_to(workdir))
            for p in workdir.glob("*.png")
        )[:20],
    }


def _summarise_csvs(workdir: Path) -> list[dict[str, Any]]:
    """For each CSV, capture header + first ~12 rows as raw text.

    Keeps the evidence small (LLM call is cheap) while exposing the
    actual numbers — including ``inf`` / ``nan`` markers — so the
    judge can see the failure signals directly.
    """
    out: list[dict[str, Any]] = []
    for path in sorted(workdir.glob("*.csv")):
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        rows = text.splitlines()[: _MAX_CSV_ROWS + 1]
        out.append({
            "path": path.name,
            "preview": "\n".join(rows),
            "total_lines": text.count("\n") + 1,
        })
    return out


def _sample_logs(workdir: Path) -> list[dict[str, Any]]:
    """Pick the most-recent N execute_code logs and capture their tails.

    The TAIL is what matters — that's where the print() output lives.
    Errors / final values typically appear in the last few KB of stdout.
    """
    logs_dir = workdir / "logs"
    if not logs_dir.exists():
        return []
    log_paths = sorted(
        logs_dir.glob("step_*.log"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )[: _MAX_LOG_FILES_PER_METHOD * 4]  # over-collect, the LLM picks
    out: list[dict[str, Any]] = []
    for path in log_paths:
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        tail = text[-_MAX_LOG_CHARS:] if len(text) > _MAX_LOG_CHARS else text
        out.append({"path": str(path.relative_to(workdir)), "tail": tail})
    return out


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------


_SYSTEM_PROMPT = """\
You are an outcome judge for a research-agent system.

Your ONLY job: for each method the agent committed to executing, label
the outcome based STRICTLY on the provided evidence files. You are NOT
asked to judge whether the methods are intrinsically good, only whether
THIS RUN actually produced valid evidence for each one.

Outcome label MUST be one of:
  • success                 — the method ran end-to-end and produced
                              meaningful results (non-trivial path,
                              non-degenerate metric, etc.)
  • partial                 — ran but only some scenarios / settings
                              succeeded
  • implementation_failure  — the code ran without crashing but the
                              METHOD did not work (success_rate=0,
                              path_length=inf, NaN metrics, agent stuck
                              in same cell, loss not decreasing, etc.)
  • data_failure            — environment / dataset / API issue prevented
                              evaluation regardless of method
  • not_attempted           — no evidence the method was ever run

Reply with ONE JSON array, no prose around it. Each element is:
  {"method": "<name from input>",
   "outcome": "<one of the labels above>",
   "evidence_pointer": "<short pointer like 'foo.csv: success_rate=0.0' OR 'logs/step_009_code.log tail: NameError'>",
   "honest_summary": "<1-2 sentences. Be neutral, name the failure mode if any, do not soften it>"}

Critical rules:
  • If a CSV row shows path_length=inf, success=False, or success_rate=0.0
    for a method, the outcome IS implementation_failure or data_failure
    (NOT success / partial). Do not let absence of explicit error message
    override the numeric evidence.
  • If a log tail shows the agent's path coordinates oscillating in the
    same cell (e.g. all entries are [2,2]), outcome IS
    implementation_failure regardless of what the agent wrote in
    follow-up cells.
  • If the evidence does not mention the method by name AND no related
    file exists, outcome is not_attempted.
"""


_USER_TEMPLATE = """\
METHODS TO ASSESS (from research_plan.json::feasibility_split.will_execute):
{methods}

EVIDENCE FROM THE WORKSPACE:

CSV files (truncated):
{csv_block}

Recent execute_code log tails:
{log_block}

Figure files saved (paths only):
{figure_block}

Output the JSON array. One element per method above, in the same order."""


_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(\[.*?\])\s*```", re.DOTALL)


def _run_outcome_llm(
    methods: list[str],
    evidence: dict[str, Any],
    llm: _LLMLike,
) -> list[MethodOutcome]:
    csv_block = _format_csv_block(evidence["csv_files"])
    log_block = _format_log_block(evidence["log_excerpts"])
    figure_block = ", ".join(evidence["figure_files"]) or "(none)"
    methods_block = "\n".join(f"  • {m}" for m in methods)

    user = _USER_TEMPLATE.format(
        methods=methods_block,
        csv_block=csv_block or "(none)",
        log_block=log_block or "(none)",
        figure_block=figure_block,
    )
    response = llm.invoke([
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ])
    text = _extract_text(response)
    return _parse_outcomes(text, methods)


def _format_csv_block(csvs: list[dict[str, Any]]) -> str:
    if not csvs:
        return ""
    parts = []
    for entry in csvs:
        parts.append(
            f"--- {entry['path']} ({entry['total_lines']} lines total) ---\n"
            f"{entry['preview']}"
        )
    return "\n\n".join(parts)


def _format_log_block(logs: list[dict[str, Any]]) -> str:
    if not logs:
        return ""
    parts = []
    for entry in logs:
        parts.append(f"--- {entry['path']} (tail) ---\n{entry['tail']}")
    return "\n\n".join(parts)


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


def _parse_outcomes(text: str, expected_methods: list[str]) -> list[MethodOutcome]:
    """Parse the LLM's JSON array, tolerating ```json fences``` and prose.

    Methods not present in the LLM's output get a fallback "not_attempted"
    entry so the caller's downstream consumer always sees a complete map.
    """
    text = (text or "").strip()
    if not text:
        return _fallback_all(expected_methods)

    candidate: str | None = None
    fence_match = _JSON_FENCE_RE.search(text)
    if fence_match:
        candidate = fence_match.group(1)
    else:
        first = text.find("[")
        last = text.rfind("]")
        if first != -1 and last > first:
            candidate = text[first : last + 1]

    if candidate is None:
        return _fallback_all(expected_methods)

    try:
        items = json.loads(candidate)
    except json.JSONDecodeError:
        return _fallback_all(expected_methods)
    if not isinstance(items, list):
        return _fallback_all(expected_methods)

    by_method: dict[str, MethodOutcome] = {}
    for raw in items:
        if not isinstance(raw, dict):
            continue
        method = str(raw.get("method") or "").strip()
        if not method:
            continue
        outcome = str(raw.get("outcome") or "").strip().lower()
        if outcome not in _VALID_OUTCOMES:
            outcome = "partial"  # safest bucket for unrecognised labels
        by_method[method] = MethodOutcome(
            method=method,
            outcome=outcome,
            evidence_pointer=str(raw.get("evidence_pointer") or "").strip()[:300],
            honest_summary=str(raw.get("honest_summary") or "").strip()[:500],
        )

    out: list[MethodOutcome] = []
    for m in expected_methods:
        # Try exact then case-insensitive match — LLMs sometimes
        # title-case the input or strip parentheticals.
        if m in by_method:
            out.append(by_method[m])
            continue
        match = next(
            (v for k, v in by_method.items() if k.lower() == m.lower()),
            None,
        )
        out.append(match or _fallback(m))
    return out


def _fallback(method: str) -> MethodOutcome:
    return MethodOutcome(
        method=method,
        outcome="not_attempted",
        evidence_pointer="(no evidence found in workspace)",
        honest_summary=(
            "The outcome judge could not verify this method ran end-to-end. "
            "Treat coverage as not-attempted unless the report's prose "
            "demonstrates otherwise with cited evidence."
        ),
    )


def _fallback_all(expected_methods: list[str]) -> list[MethodOutcome]:
    return [_fallback(m) for m in expected_methods]


# ---------------------------------------------------------------------------
# Disk loader
# ---------------------------------------------------------------------------


def load_outcomes_from_workspace(working_dir: str | Path) -> list[MethodOutcome]:
    """Read a previously-persisted outcome list. Returns [] when missing."""
    path = Path(working_dir) / OUTCOMES_FILENAME
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    if not isinstance(data, list):
        return []
    out: list[MethodOutcome] = []
    for raw in data:
        if not isinstance(raw, dict):
            continue
        out.append(MethodOutcome(
            method=str(raw.get("method") or ""),
            outcome=str(raw.get("outcome") or ""),
            evidence_pointer=str(raw.get("evidence_pointer") or ""),
            honest_summary=str(raw.get("honest_summary") or ""),
        ))
    return out
