"""
Task-checklist: keep the agent anchored to the original task throughout a long run.

LLM agents drift. The original task description is injected once at run start
inside the system prompt, but after 30+ rounds of intermediate work, reviewer
feedback, and tool outputs, the original verbs ("propose a NEW algorithm",
"outperform baselines") get buried in token history. The model's attention
shifts to the most recent reviewer complaints (about NaN disclosure, citation
formatting, etc) and the high-level task fades from working memory.

Two layers — only the static one is wired up in agent.py now:

  1. ``parse_task_checklist`` — at run start, one LLM call turns the task
     description into a flat checklist of 3-5 concrete required artifacts.
     This is **kept**: the parsed list is rendered into a static reminder
     each round so the agent doesn't forget what it agreed to deliver.

  2. ``format_topic_reminder`` — every 5 rounds, render a deterministic
     anti-drift block that combines:
       (a) the original topic verbatim,
       (b) the parsed checklist items (no LLM re-evaluation),
       (c) a filesystem listing of artifacts already produced in the run
           workspace (CSVs, charts, reports — purely os.scandir-based).
     The agent sees what was asked + what already exists on disk + a clear
     "if you've covered the topic, call submit_result to end the run".

  3. ``evaluate_checklist_status`` / ``format_checklist_for_round`` /
     ``format_submit_block_message`` — **DEPRECATED but still defined**
     (UI / older tests still import them). Replaced because:
       - they re-evaluate completion via an LLM each refresh,
       - the LLM's verdict swings turn-to-turn (e.g. trajectory 72d6ad6d
         was blocked 12 times, each time citing a *different* "missing"
         item even though the artifacts existed on disk),
       - the gate that used these in agent.py to *block* submit_result
         created the multi-hour death loop SWE-agent / PaperQA2 / MLE-bench
         all explicitly avoid (see REFACTOR_PLAN.md).
     New code MUST NOT call ``evaluate_checklist_status``. The reminder
     mechanism is in ``format_topic_reminder``.

Why every 5 rounds (not every round): the reminder is plain-text, but
re-injecting it every round would just clutter the conversation. 5 rounds
keeps the topic fresh in the LLM's attention without spam.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


# Status taxonomy for each checklist item.
STATUS_DONE = "done"
STATUS_IN_PROGRESS = "in_progress"
STATUS_PENDING = "pending"
_VALID_STATUSES = {STATUS_DONE, STATUS_IN_PROGRESS, STATUS_PENDING}


@dataclass
class ChecklistItem:
    """One concrete required artifact from the task description.

    ``id`` is a small int (1..N) used purely as a stable handle in prompts
    and logs. ``title`` is the short-form requirement the agent must satisfy.
    ``verification_hint`` is a sentence telling the evaluator (and the
    agent) what kind of evidence in the trajectory would mark this item
    "done" — it makes evaluation deterministic across rounds.
    """

    id: int
    title: str
    verification_hint: str


@dataclass
class ChecklistStatus:
    """Per-item evaluation result. Refreshed every 5 rounds."""

    item_id: int
    status: str   # one of STATUS_*
    evidence: str = ""

    def is_done(self) -> bool:
        return self.status == STATUS_DONE


@dataclass
class TaskChecklist:
    """Full checklist + latest evaluation state for a run."""

    items: list[ChecklistItem] = field(default_factory=list)
    last_status: list[ChecklistStatus] = field(default_factory=list)
    last_evaluated_step: int = -1

    def is_complete(self) -> bool:
        """All items done. Returns False if checklist is empty (no items
        to verify means we haven't parsed yet — better to block than to
        let an unverified submit through)."""
        if not self.items or not self.last_status:
            return False
        status_by_id = {s.item_id: s for s in self.last_status}
        return all(
            (s := status_by_id.get(item.id)) is not None and s.is_done()
            for item in self.items
        )

    def pending_items(self) -> list[ChecklistItem]:
        """Items whose latest status is NOT done. Used in the submit gate."""
        if not self.items:
            return []
        if not self.last_status:
            return list(self.items)
        status_by_id = {s.item_id: s for s in self.last_status}
        out: list[ChecklistItem] = []
        for item in self.items:
            s = status_by_id.get(item.id)
            if s is None or not s.is_done():
                out.append(item)
        return out


# ---------------------------------------------------------------------------
# Parsing the task description into a checklist (one-shot at run start)
# ---------------------------------------------------------------------------

_PARSE_PROMPT = """\
You are a research project manager. Read the research task below and break it
into 3 to 5 concrete, verifiable artifact requirements that the executing
agent must produce. Each requirement must be specific enough that a reviewer
could check the agent's tool history and decide "done" / "not done".

Output STRICT JSON with this exact shape (no markdown, no commentary):

{{
  "items": [
    {{
      "title": "short one-line requirement (imperative)",
      "verification_hint": "what evidence in the agent's tool outputs would prove this is done"
    }}
  ]
}}

Examples of GOOD requirements:
  - title: "Implement a custom (not library-only) clustering algorithm"
    verification_hint: "execute_code defines a class or function implementing
                        the method; library calls like sklearn.cluster.KMeans
                        alone do NOT count"
  - title: "Run >=2 baselines from the literature for comparison"
    verification_hint: "execute_code runs at least 2 distinct named baseline
                        methods and saves their metrics to a CSV/log"
  - title: "The new method outperforms baselines on the majority of metrics"
    verification_hint: "the saved results table shows the new method
                        winning on >=50% of reported metrics"
  - title: "Validation done on a real (non-synthetic) text dataset"
    verification_hint: "execute_code loads a public dataset (e.g.
                        20newsgroups, MNIST, CIFAR-10 subset) — pure
                        synthetic / make_classification does NOT count"

Bad requirements (too vague):
  - "Do good research"
  - "Write a paper"

The TASK is:
---
{task_description}
---

Respond with JSON only.
"""


def _safe_json_parse(text: str) -> dict | None:
    """Parse the LLM JSON output; tolerate fence wrappers / trailing prose."""
    if not text:
        return None
    text = text.strip()
    # Strip ``` fences when present.
    fence_match = re.search(r"```(?:json)?\s*\n?(.*?)```", text, re.DOTALL)
    if fence_match:
        text = fence_match.group(1).strip()
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        pass
    # Fall back: extract the first balanced { ... } block.
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start:i + 1])
                except (json.JSONDecodeError, TypeError):
                    return None
    return None


def parse_task_checklist(
    task_description: str,
    llm: Any,
    max_items: int = 5,
) -> TaskChecklist:
    """One-shot LLM call: turn ``task_description`` into a verifiable checklist.

    Returns an empty checklist on parse failure — callers should treat
    "empty checklist" as "no goal-anchoring this run" rather than
    aborting; goal-anchoring is supplemental, not load-bearing.

    ``llm`` should be a LangChain ChatOpenAI-compatible runnable (the same
    instance the agent uses). The function calls ``llm.invoke([...])``.
    """
    if not task_description or not task_description.strip():
        return TaskChecklist()

    try:
        from langchain_core.messages import HumanMessage
    except ImportError:  # pragma: no cover - langchain always present in agent runs
        logger.warning("langchain_core unavailable; checklist disabled")
        return TaskChecklist()

    prompt = _PARSE_PROMPT.format(task_description=task_description.strip())
    try:
        result = llm.invoke([HumanMessage(content=prompt)])
        content = getattr(result, "content", "") or ""
    except Exception as exc:
        logger.warning("Task checklist parse LLM call failed: %s", exc)
        return TaskChecklist()

    parsed = _safe_json_parse(content)
    if not parsed or "items" not in parsed:
        logger.warning(
            "Task checklist parser returned invalid JSON; checklist disabled. "
            "First 200 chars of response: %r", content[:200],
        )
        return TaskChecklist()

    raw_items = parsed.get("items") or []
    if not isinstance(raw_items, list):
        return TaskChecklist()

    items: list[ChecklistItem] = []
    for idx, raw in enumerate(raw_items[:max_items], start=1):
        if not isinstance(raw, dict):
            continue
        title = (raw.get("title") or "").strip()
        if not title:
            continue
        hint = (raw.get("verification_hint") or "").strip()
        items.append(ChecklistItem(id=idx, title=title, verification_hint=hint))

    if not items:
        logger.warning("Task checklist parser returned no usable items")
    else:
        logger.info(
            "Parsed task checklist: %d items — %s",
            len(items),
            "; ".join(f"[{i.id}] {i.title}" for i in items),
        )
    return TaskChecklist(items=items)


# ---------------------------------------------------------------------------
# Evaluating checklist status against the trajectory (every 5 rounds)
# ---------------------------------------------------------------------------

_EVAL_PROMPT = """\
You are evaluating a research agent's progress against a task checklist.
For each checklist item, decide whether the agent's tool history shows
clear evidence the item is DONE, IN_PROGRESS (started but not yet
verifiable), or PENDING (no evidence of work yet).

Be STRICT — only mark "done" when the trajectory clearly contains the
required artifact. If the agent says it's done but the tool outputs don't
prove it (e.g. claims a custom method but only sklearn calls visible),
that's IN_PROGRESS, not DONE. Library-only experiments are NOT a custom
method implementation.

Output STRICT JSON with this exact shape (no markdown, no commentary):

{{
  "evaluations": [
    {{
      "item_id": <int>,
      "status": "done" | "in_progress" | "pending",
      "evidence": "<one short sentence quoting or summarizing the tool output that supports this status>"
    }}
  ]
}}

Checklist items:
{checklist_block}

Recent agent trajectory (most recent {trajectory_window} steps):
{trajectory_block}

Respond with JSON only.
"""


def _format_trajectory_for_eval(steps: list[Any], window: int = 30) -> str:
    """Render the last ``window`` steps as compact lines for the evaluator.

    Each line shows step index + action + truncated observation. We exclude
    failed runtime_error steps (noise) and shorten very long outputs so the
    LLM's input cost stays manageable.
    """
    if not steps:
        return "(no tool calls yet)"
    visible: list[Any] = []
    for s in steps:
        action_name = getattr(s, "action_name", "") or ""
        if action_name == "runtime_error":
            continue
        visible.append(s)
    sliced = visible[-window:]
    lines: list[str] = []
    for s in sliced:
        idx = getattr(s, "step_index", "?")
        action = getattr(s, "action_name", "?") or "?"
        success = getattr(s, "success", True)
        flag = "✓" if success else "✗"
        obs = (getattr(s, "observation", "") or "").strip()
        # Compact obs: keep first 240 chars of the first non-empty line
        compact = ""
        for line in obs.splitlines():
            line = line.strip()
            if line:
                compact = line[:240]
                break
        if not compact:
            compact = "[no output]"
        lines.append(f"step {idx} {flag} {action}: {compact}")
    return "\n".join(lines) if lines else "(no productive tool calls yet)"


def _format_checklist_block(items: list[ChecklistItem]) -> str:
    return "\n".join(
        f"  [{item.id}] {item.title}\n      hint: {item.verification_hint}"
        for item in items
    )


def evaluate_checklist_status(
    checklist: TaskChecklist,
    trajectory_steps: list[Any],
    llm: Any,
    *,
    trajectory_window: int = 30,
) -> list[ChecklistStatus]:
    """Run a single LLM call to refresh checklist statuses.

    The evaluation is best-effort: parse failures or LLM errors return the
    previous status (or all-pending if there was none yet). Callers should
    NOT block on this — the goal is to keep the agent anchored, and a stale
    checklist is better than a missing one.
    """
    if not checklist.items:
        return []

    try:
        from langchain_core.messages import HumanMessage
    except ImportError:
        return checklist.last_status or [
            ChecklistStatus(item_id=i.id, status=STATUS_PENDING) for i in checklist.items
        ]

    prompt = _EVAL_PROMPT.format(
        checklist_block=_format_checklist_block(checklist.items),
        trajectory_window=trajectory_window,
        trajectory_block=_format_trajectory_for_eval(trajectory_steps, trajectory_window),
    )

    try:
        result = llm.invoke([HumanMessage(content=prompt)])
        content = getattr(result, "content", "") or ""
    except Exception as exc:
        logger.warning("Checklist status eval LLM call failed: %s", exc)
        return checklist.last_status or [
            ChecklistStatus(item_id=i.id, status=STATUS_PENDING) for i in checklist.items
        ]

    parsed = _safe_json_parse(content)
    if not parsed or "evaluations" not in parsed:
        logger.warning(
            "Checklist status eval returned invalid JSON; keeping last status. "
            "First 200 chars: %r", content[:200],
        )
        return checklist.last_status or [
            ChecklistStatus(item_id=i.id, status=STATUS_PENDING) for i in checklist.items
        ]

    raw_evals = parsed.get("evaluations") or []
    valid_ids = {item.id for item in checklist.items}
    out: list[ChecklistStatus] = []
    seen_ids: set[int] = set()
    for raw in raw_evals:
        if not isinstance(raw, dict):
            continue
        try:
            item_id = int(raw.get("item_id"))
        except (TypeError, ValueError):
            continue
        if item_id not in valid_ids or item_id in seen_ids:
            continue
        status = (raw.get("status") or "").strip().lower()
        if status not in _VALID_STATUSES:
            status = STATUS_PENDING
        evidence = (raw.get("evidence") or "").strip()
        out.append(ChecklistStatus(item_id=item_id, status=status, evidence=evidence[:240]))
        seen_ids.add(item_id)

    # Make sure every checklist item has a status entry (default to pending).
    for item in checklist.items:
        if item.id not in seen_ids:
            out.append(ChecklistStatus(item_id=item.id, status=STATUS_PENDING))

    out.sort(key=lambda s: s.item_id)
    return out


# ---------------------------------------------------------------------------
# Rendering for round injection
# ---------------------------------------------------------------------------

def format_checklist_for_round(
    checklist: TaskChecklist,
    *,
    round_idx: int,
    max_rounds: int | None = None,
) -> str:
    """Compact human-readable block injected into every agent round.

    Items appear in checklist order (not status order) so the agent sees a
    stable layout across rounds. Pending items get a leading "☐" and the
    verification hint inlined; completed items get "✓" + evidence quote.
    """
    if not checklist.items:
        return ""

    status_by_id = {s.item_id: s for s in checklist.last_status}
    pending_count = sum(
        1 for item in checklist.items
        if (s := status_by_id.get(item.id)) is None or not s.is_done()
    )

    header = f"=== TASK CHECKLIST · round {round_idx}"
    if max_rounds is not None:
        header += f"/{max_rounds}"
    header += f" · {pending_count} of {len(checklist.items)} still open ==="
    lines = [header]

    for item in checklist.items:
        s = status_by_id.get(item.id)
        if s is None:
            lines.append(f"☐ [{item.id}] {item.title}")
            lines.append(f"     status: pending — no evidence checked yet")
            lines.append(f"     hint:   {item.verification_hint}")
        elif s.status == STATUS_DONE:
            lines.append(f"✓ [{item.id}] {item.title}")
            if s.evidence:
                lines.append(f"     evidence: {s.evidence}")
        elif s.status == STATUS_IN_PROGRESS:
            lines.append(f"◐ [{item.id}] {item.title}")
            lines.append(f"     status: in_progress — started but not yet verifiable")
            if s.evidence:
                lines.append(f"     evidence so far: {s.evidence}")
            lines.append(f"     hint to finish:  {item.verification_hint}")
        else:  # pending
            lines.append(f"☐ [{item.id}] {item.title}")
            lines.append(f"     status: pending — no evidence in tool history yet")
            lines.append(f"     hint:   {item.verification_hint}")

    if pending_count > 0:
        lines.append("")
        lines.append(
            "⚠️  Until every ☐ / ◐ item has clear tool-history evidence, "
            "do NOT call submit_result. Address the open items first by "
            "calling execute_code / search_literature / lookup_paper_code "
            "as needed."
        )
    else:
        lines.append("")
        lines.append("All checklist items satisfied. submit_result is now appropriate.")

    return "\n".join(lines)


def format_submit_block_message(
    pending: list[ChecklistItem],
    block_count: int,
) -> str:
    """DEPRECATED. Compose the HumanMessage injected when submit_result fires too early.

    Was used by ``agent.py``'s post-submit branch when the checklist still
    had pending items. Replaced by the static-reminder design — see
    ``format_topic_reminder`` and the rationale at the top of this file.
    Kept defined because UI code and tests still import it.
    """
    if not pending:
        return ""
    bullets = "\n".join(
        f"  ☐ [{item.id}] {item.title}\n      to satisfy: {item.verification_hint}"
        for item in pending
    )
    return (
        f"Your submit_result was BLOCKED (block {block_count}) because the "
        f"task checklist still has {len(pending)} unaddressed item(s). The "
        "checklist is the executive specification of THIS task — submitting "
        "before every item is verifiable in your tool history is shipping "
        "incomplete work.\n\n"
        f"Still missing:\n{bullets}\n\n"
        "Do NOT call submit_result again until each open item has direct "
        "evidence in your execute_code / search_literature / read_paper_fulltext "
        "outputs. For each open item, plan ONE concrete tool call that would "
        "produce that evidence, run it, then re-check progress. Only after "
        "every item is verifiably done should you regenerate the report and "
        "submit."
    )


# ---------------------------------------------------------------------------
# Static anti-drift reminder (replaces the LLM-eval-loop above)
# ---------------------------------------------------------------------------

# File categories rendered in the workspace listing — kept narrow on purpose
# so we don't spam every .pyc and .log into the agent's working memory.
_REMINDER_CATEGORIES = (
    ("data tables",  (".csv", ".tsv", ".jsonl", ".parquet")),
    ("figures",      (".png", ".jpg", ".jpeg", ".svg")),
    ("reports",      (".md", ".pdf")),
    ("scripts",      (".py", ".sh")),
)
_REMINDER_MAX_PER_CAT = 6
_REMINDER_SKIP_DIRS = {"__pycache__", "logs", ".cache"}


def _list_workspace_artifacts(workspace_dir: str | None) -> dict[str, list[str]]:
    """Return ``{category: [filename, ...]}`` for files visible in the run dir.

    Pure ``os.scandir`` walk — no LLM, no parsing, deterministic. Returns
    empty dict if the dir does not exist (early in a run, or when the
    caller did not set a sandbox dir).
    """
    if not workspace_dir:
        return {}
    from pathlib import Path
    root = Path(workspace_dir)
    if not root.exists() or not root.is_dir():
        return {}
    bucket: dict[str, list[str]] = {label: [] for label, _ in _REMINDER_CATEGORIES}
    try:
        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            if any(part in _REMINDER_SKIP_DIRS for part in path.parts):
                continue
            suffix = path.suffix.lower()
            for label, exts in _REMINDER_CATEGORIES:
                if suffix in exts:
                    rel = path.relative_to(root).as_posix()
                    bucket[label].append(rel)
                    break
    except OSError:
        # Filesystem hiccup is non-fatal for the reminder.
        pass
    return bucket


def _format_locked_plan_block(workspace_dir: str | None) -> str:
    """Render the scope plan persisted by submit_research_plan, if any.

    Imported lazily so tests that only exercise the checklist don't pay
    for the scope-lock import. Returns ``""`` when no plan exists or
    when the workspace dir is not provided — the reminder caller treats
    an empty return as "skip this section", so it stays out of the way
    on benchmark / unit-test runs that don't use the scope-lock tool.
    """
    if not workspace_dir:
        return ""
    try:
        from .tools.scope_lock_tool import load_plan_from_workspace
    except ImportError:
        return ""
    plan = load_plan_from_workspace(workspace_dir)
    if plan is None:
        return ""
    lines = [
        "Scope plan you committed to via submit_research_plan "
        "(re-injected so you don't drift):",
        f"  • Title hypothesis: {plan.title_hypothesis or '(none)'}",
    ]
    if plan.natural_scope_methods:
        lines.append(
            "  • Natural scope methods: " + ", ".join(plan.natural_scope_methods)
        )
    if plan.axes_of_comparison:
        lines.append(
            "  • Axes of comparison: " + ", ".join(plan.axes_of_comparison)
        )
    if plan.will_execute:
        lines.append("  • Will execute (must show up in results): "
                     + ", ".join(plan.will_execute))
    if plan.literature_only:
        lines.append("  • Literature-only (must show up in Related Work): "
                     + ", ".join(plan.literature_only))
    lines.append(
        "  At submit time the topic-fidelity critic will compare your "
        "report against THIS plan. If you're missing coverage, queue "
        "another tool call now."
    )
    return "\n".join(lines)


def format_topic_reminder(
    *,
    topic: str,
    checklist: TaskChecklist,
    workspace_dir: str | None,
    round_idx: int,
    max_rounds: int | None = None,
    mode: str = "survey",
) -> str:
    """Deterministic anti-drift block injected every N rounds (no LLM call).

    Replaces ``format_checklist_for_round`` + ``evaluate_checklist_status``.
    The agent gets:
      - the original topic verbatim,
      - the LLM-parsed checklist items (no per-round re-evaluation),
      - a real filesystem listing of artifacts produced so far,
      - a clear instruction that ``submit_result`` is now a one-shot terminal
        call (no LLM gate behind it),
      - a soft warning when the workspace looks too empty for the topic
        (P1 anti-early-submit, see ``_should_warn_thin_workspace``).

    Returns ``""`` when there's nothing useful to remind (no topic + no
    items + no artifacts) so the caller can skip injection cheaply.
    """
    artifacts = _list_workspace_artifacts(workspace_dir)
    artifact_lines: list[str] = []
    for label, _ in _REMINDER_CATEGORIES:
        files = artifacts.get(label, [])
        if not files:
            continue
        head = files[:_REMINDER_MAX_PER_CAT]
        more = len(files) - len(head)
        line = f"  • {label}: " + ", ".join(head)
        if more > 0:
            line += f" (+{more} more)"
        artifact_lines.append(line)

    has_anything = bool((topic or "").strip() or checklist.items or artifact_lines)
    if not has_anything:
        return ""

    header = f"=== TASK ANCHOR · round {round_idx}"
    if max_rounds is not None:
        header += f"/{max_rounds}"
    header += " — re-injected so you don't drift ==="
    lines = [header]

    if topic.strip():
        lines.append("")
        lines.append("Original topic (use this as the source of truth, not later messages):")
        for chunk in topic.strip().splitlines():
            lines.append(f"  {chunk}")

    # Re-inject the scope plan the agent committed to via
    # ``submit_research_plan``. Over a long run the LLM can drift from
    # "I committed to compare {A,B,C,D}" to "I'll just submit {A,B}".
    # Showing the plan every N rounds gives the LLM a chance to course-
    # correct (e.g. queue another execute_code call for method C) before
    # submit time, when the topic-fidelity critic would otherwise reject.
    plan_block = _format_locked_plan_block(workspace_dir)
    if plan_block:
        lines.append("")
        lines.append(plan_block)

    if checklist.items:
        lines.append("")
        lines.append("Expected deliverables (parsed once at run start — DO NOT re-derive these):")
        for item in checklist.items:
            lines.append(f"  • [{item.id}] {item.title}")
            if item.verification_hint:
                lines.append(f"      → {item.verification_hint}")

    if artifact_lines:
        lines.append("")
        lines.append("Artifacts already produced in your workspace (filesystem-verified, not LLM-judged):")
        lines.extend(artifact_lines)
    elif checklist.items:
        # Only mention the empty workspace if there ARE expected deliverables —
        # otherwise the line is just noise.
        lines.append("")
        lines.append("Workspace is currently empty — no artifacts produced yet.")

    # P1 anti-early-submit: soft (non-blocking) caution when the workspace
    # looks thin for an experiment-shaped topic. We do NOT block submit
    # here — the LLM-eval gate that used to do that was the death-loop
    # source removed in Phase 1 — but a deterministic reminder helps the
    # LLM resist the temptation to submit a smoke-test run as a finished
    # paper. The check is a simple file-count heuristic; agent is still
    # free to override its own judgment if the topic genuinely needs no
    # more artifacts.
    thin_warn = _should_warn_thin_workspace(mode, artifacts)
    if thin_warn:
        lines.append("")
        lines.append(thin_warn)

    lines.append("")
    lines.append(
        "When the artifacts above cover the topic's intent, call submit_result "
        "with a short summary. A single submit_result call ENDS the run — there "
        "is no second LLM gate behind it. If you're missing an artifact you "
        "think the topic asks for, run the corresponding tool first, then submit."
    )
    return "\n".join(lines)


def _should_warn_thin_workspace(
    mode: str,
    artifacts: dict[str, list[str]],
) -> str:
    """Return a soft caution line when an experiment-mode run has no
    figures AND no data tables yet — return ``""`` otherwise.

    The two file categories chosen here (``figures`` + ``data tables``)
    are the ones a reader expects in a comparative-experiment paper. If
    both are absent, the agent is almost certainly about to submit prose
    that claims results it never saved (the c032bbf1 failure mode).

    Survey-mode runs are skipped — a literature survey does not need
    experimental artifacts.
    """
    if mode != "experiment":
        return ""

    figure_count = len(artifacts.get("figures", []))
    table_count = len(artifacts.get("data tables", []))
    if figure_count == 0 and table_count == 0:
        return (
            "⚠ Soft check (non-blocking): your topic looks like a comparison "
            "experiment but the workspace currently has zero figures and zero "
            "result CSVs. Submitting now would mean writing a comparison paper "
            "with no real data — generate_report's path-existence gate will "
            "reject any figures=/tables= entries you make up. Save real charts "
            "via plt.savefig(...) and real metric tables via "
            "pd.DataFrame(...).to_csv(...) before calling generate_report."
        )
    if figure_count == 0:
        return (
            "⚠ Soft check (non-blocking): you have result tables but no figures. "
            "A comparison experiment is much clearer with at least one chart "
            "(e.g. accuracy / loss curve). Consider plt.savefig before submit."
        )
    if table_count == 0:
        return (
            "⚠ Soft check (non-blocking): you have figures but no metric CSV. "
            "A comparison study should also save the underlying numbers as a "
            "CSV (pd.DataFrame(...).to_csv(...)) so the generated table in "
            "the report is grounded in real numbers."
        )
    return ""
