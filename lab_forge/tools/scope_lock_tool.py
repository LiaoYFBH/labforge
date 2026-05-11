"""
Scope-lock tool — agent's mandatory FIRST tool call.

Why this exists
---------------
The previous experiment-mode prompt mixed two decisions in one LLM call:
  1) "what to research" (topic-level scope: which methods, axes of comparison)
  2) "what's feasible" (CPU-only, ≤300 s per execute_code, small datasets)

The feasibility signal is concrete and quantitative; the scope signal is
fuzzy. The cheaper signal won — runs converged on tiny demos (e.g. "Q-learning
vs DQN on CartPole") and the paper title silently shrank to match the demo
instead of the topic.

The fix is structural, not lexical. We force the agent to commit to a
scope BEFORE it sees execution constraints, by gating every other tool
behind one mandatory call to ``submit_research_plan``. The plan is a
pure-LLM artifact (no keyword tables, no domain dictionaries) — the LLM
infers the natural scope of methods/axes from the topic itself, then
splits them into "I will run these" vs "I will only cover these via
literature". The latter slot is the pressure-release valve that keeps the
title from shrinking when CPU constraints bite.

The plan is persisted to ``research_plan.json`` so:
  - the topic-anchor reminder can re-inject it every N rounds,
  - the topic-fidelity critic can compare report ↔ committed scope at
    submit time,
  - the run is auditable after the fact.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .base import Tool, ToolResult

# Filename relative to the workspace. Kept short and stable so other
# subsystems (reminder, critic) can hardcode a single import-free read.
PLAN_FILENAME = "research_plan.json"


@dataclass
class ResearchPlan:
    """Structured representation of the agent's locked-in scope.

    All list fields are normalised to ``list[str]`` even when the LLM
    submits dicts — keeping the downstream readers (reminder, critic)
    schema-stable.
    """

    natural_scope_methods: list[str]
    axes_of_comparison: list[str]
    will_execute: list[str]
    literature_only: list[str]
    title_hypothesis: str
    rationale: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "natural_scope": {
                "methods": list(self.natural_scope_methods),
                "axes_of_comparison": list(self.axes_of_comparison),
            },
            "feasibility_split": {
                "will_execute": list(self.will_execute),
                "literature_only": list(self.literature_only),
            },
            "title_hypothesis": self.title_hypothesis,
            "rationale": self.rationale,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ResearchPlan":
        scope = data.get("natural_scope") or {}
        feas = data.get("feasibility_split") or {}
        return cls(
            natural_scope_methods=_coerce_str_list(scope.get("methods")),
            axes_of_comparison=_coerce_str_list(scope.get("axes_of_comparison")),
            will_execute=_coerce_str_list(feas.get("will_execute")),
            literature_only=_coerce_str_list(feas.get("literature_only")),
            title_hypothesis=str(data.get("title_hypothesis") or "").strip(),
            rationale=str(data.get("rationale") or "").strip(),
        )


def _coerce_str_list(value: Any) -> list[str]:
    """Best-effort coerce to ``list[str]``.

    Tolerant of: list of strings, list of dicts (uses ``name`` / ``method``
    fields), comma-separated string, ``None``. Empty / unrecognised inputs
    yield ``[]``.
    """
    if value is None:
        return []
    if isinstance(value, str):
        # LLMs sometimes shove a comma-separated string into a list field.
        return [s.strip() for s in value.split(",") if s.strip()]
    if isinstance(value, list):
        out: list[str] = []
        for item in value:
            if isinstance(item, str):
                txt = item.strip()
            elif isinstance(item, dict):
                txt = str(
                    item.get("name")
                    or item.get("method")
                    or item.get("title")
                    or ""
                ).strip()
            else:
                txt = str(item).strip()
            if txt:
                out.append(txt)
        return out
    return []


class ScopeLockTool(Tool):
    """Mandatory first tool call. Locks the topic-level scope of the run.

    Hard-enforcement is implemented at the ``langchain_tools`` layer:
    every other tool's ``_run`` closure short-circuits with a clear
    error pointing the agent back here when ``is_locked()`` is False.
    """

    def __init__(self, working_dir: str | Path):
        self.working_dir = Path(working_dir)
        # In-memory mirror of the disk state. We re-check disk in
        # ``is_locked`` so a previous run's plan (left from a sandbox
        # reuse) is honoured without re-asking the agent.
        self._plan: ResearchPlan | None = None
        self._load_existing()

    @property
    def name(self) -> str:
        return "submit_research_plan"

    @property
    def description(self) -> str:
        return (
            "MANDATORY first tool call. Lock the topic-level scope of this "
            "run BEFORE any other tool is callable.\n\n"
            "Read the original research topic and infer — from the topic "
            "alone, NOT from execution constraints — the natural scope a "
            "credible paper on this topic should cover:\n"
            "  • natural_scope.methods: the distinct methods/algorithms/"
            "approaches a fair treatment of the topic should compare. If "
            "the topic uses plural framing ('compare different X', 'a "
            "survey of Y methods') aim for ≥3 entries.\n"
            "  • natural_scope.axes_of_comparison: the dimensions along "
            "which methods will be contrasted (e.g. 'sample efficiency', "
            "'final accuracy', 'wall-clock cost').\n\n"
            "Then split the methods into two buckets reflecting what the "
            "machine can actually run:\n"
            "  • feasibility_split.will_execute: methods you will run "
            "end-to-end with code.\n"
            "  • feasibility_split.literature_only: methods covered only "
            "via literature synthesis (because of CPU/GPU/time limits, or "
            "because they require infra you don't have). LIST THEM HERE — "
            "do not silently drop them, that's how titles shrink.\n\n"
            "Finally:\n"
            "  • title_hypothesis: the title the paper will carry. It "
            "MUST mirror the topic-level question, NOT the smallest "
            "dataset/env you happen to run on.\n"
            "  • rationale: 2-4 sentences explaining the split — why "
            "these methods, why this title, why these go to literature.\n\n"
            "Once locked, the plan is persisted to research_plan.json and "
            "every other tool unblocks. The plan is read by the topic-"
            "anchor reminder (every N rounds) and by the topic-fidelity "
            "critic at report submit time."
        )

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "natural_scope": {
                    "type": "object",
                    "description": (
                        "The scope a credible paper on this topic should "
                        "cover, derived from the topic alone."
                    ),
                    "properties": {
                        "methods": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": (
                                "Distinct methods / algorithms / approaches "
                                "to be compared. ≥3 when the topic asks for "
                                "comparison across plural items."
                            ),
                        },
                        "axes_of_comparison": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": (
                                "Dimensions along which methods will be "
                                "contrasted (sample efficiency, accuracy, "
                                "cost, etc.)."
                            ),
                        },
                    },
                    "required": ["methods", "axes_of_comparison"],
                },
                "feasibility_split": {
                    "type": "object",
                    "description": (
                        "Split natural_scope.methods into what will run "
                        "vs. what will only be reviewed in the paper."
                    ),
                    "properties": {
                        "will_execute": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": (
                                "Methods that will be run end-to-end with "
                                "code on this machine."
                            ),
                        },
                        "literature_only": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": (
                                "Methods covered via literature synthesis "
                                "instead of execution (CPU/GPU/time/infra "
                                "constraints). DO NOT leave empty just to "
                                "shrink the topic."
                            ),
                        },
                    },
                    "required": ["will_execute", "literature_only"],
                },
                "title_hypothesis": {
                    "type": "string",
                    "description": (
                        "The paper title. MUST mirror the topic's framing, "
                        "not the smallest dataset/env you'll run on."
                    ),
                },
                "rationale": {
                    "type": "string",
                    "description": (
                        "2-4 sentences justifying the method set, the "
                        "execute/literature split, and the title."
                    ),
                },
            },
            "required": [
                "natural_scope",
                "feasibility_split",
                "title_hypothesis",
                "rationale",
            ],
        }

    def execute(
        self,
        natural_scope: dict | None = None,
        feasibility_split: dict | None = None,
        title_hypothesis: str = "",
        rationale: str = "",
    ) -> ToolResult:
        plan = ResearchPlan.from_dict(
            {
                "natural_scope": natural_scope or {},
                "feasibility_split": feasibility_split or {},
                "title_hypothesis": title_hypothesis,
                "rationale": rationale,
            }
        )
        problems = self._validate(plan)
        if problems:
            return ToolResult(
                output=(
                    "submit_research_plan REJECTED — fix and resubmit:\n"
                    + "\n".join(f"  • {p}" for p in problems)
                    + "\n\nThe plan must commit to a scope BEFORE any other "
                    "tool is callable. Resubmit with the missing fields."
                ),
                success=False,
                metadata={"validation_errors": problems},
            )
        self._persist(plan)
        self._plan = plan
        return ToolResult(
            output=(
                "Scope locked. research_plan.json written to the workspace.\n\n"
                f"Methods (natural scope, n={len(plan.natural_scope_methods)}): "
                f"{', '.join(plan.natural_scope_methods)}\n"
                f"Axes of comparison: {', '.join(plan.axes_of_comparison)}\n"
                f"Will execute (n={len(plan.will_execute)}): "
                f"{', '.join(plan.will_execute) or '(none — pure literature run)'}\n"
                f"Literature-only (n={len(plan.literature_only)}): "
                f"{', '.join(plan.literature_only) or '(none)'}\n"
                f"Title hypothesis: {plan.title_hypothesis}\n\n"
                "All other tools are now unlocked. Proceed with the run.\n"
                "Reminder: at report submit time, a topic-fidelity critic "
                "will compare the report against THIS plan. Methods you "
                "list here as 'literature_only' must show up in the report's "
                "Related Work / Methodology coverage; methods you list as "
                "'will_execute' must show up in the experiment results."
            ),
            success=True,
            metadata={"plan": plan.to_dict()},
        )

    # ------------------------------------------------------------------
    # Public predicate used by the gating layer
    # ------------------------------------------------------------------
    def is_locked(self) -> bool:
        """True iff a valid plan has been submitted (this run or a previous one)."""
        if self._plan is not None:
            return True
        # Disk fallback: a sandbox that was reused across runs may carry an
        # earlier plan. Treat that as locked so we don't make the agent
        # re-derive it. The fidelity critic still uses whatever's on disk.
        self._load_existing()
        return self._plan is not None

    @property
    def plan(self) -> ResearchPlan | None:
        if self._plan is None:
            self._load_existing()
        return self._plan

    def reset(self) -> None:
        """Drop the cached plan AND its disk file. Used by tests."""
        self._plan = None
        try:
            (self.working_dir / PLAN_FILENAME).unlink()
        except FileNotFoundError:
            pass

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _validate(self, plan: ResearchPlan) -> list[str]:
        problems: list[str] = []
        if not plan.natural_scope_methods:
            problems.append(
                "natural_scope.methods is empty — list the methods/algorithms "
                "the topic naturally calls for."
            )
        if not plan.axes_of_comparison:
            problems.append(
                "natural_scope.axes_of_comparison is empty — name at least "
                "one dimension you'll compare on (e.g. accuracy, "
                "sample efficiency, runtime)."
            )
        if not plan.title_hypothesis:
            problems.append("title_hypothesis is empty.")
        if len(plan.rationale) < 30:
            problems.append(
                "rationale is too short (<30 chars) — explain why these "
                "methods, why this title, and why the execute/literature split."
            )
        # When natural_scope has multiple methods but BOTH buckets of the
        # split are empty, the plan is structurally inconsistent. We don't
        # police "you only put 1 method in will_execute" here — the critic
        # at submit time judges semantic fidelity. This rule only catches
        # the obviously-empty case where the agent forgot to fill the split.
        if (
            len(plan.natural_scope_methods) >= 1
            and not plan.will_execute
            and not plan.literature_only
        ):
            problems.append(
                "feasibility_split has no will_execute AND no literature_only "
                "methods — at least one bucket must be populated. Use "
                "literature_only when the machine cannot run a method."
            )
        return problems

    def _persist(self, plan: ResearchPlan) -> None:
        self.working_dir.mkdir(parents=True, exist_ok=True)
        (self.working_dir / PLAN_FILENAME).write_text(
            json.dumps(plan.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _load_existing(self) -> None:
        path = self.working_dir / PLAN_FILENAME
        if not path.exists():
            return
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return
        try:
            self._plan = ResearchPlan.from_dict(data)
        except Exception:
            # Defensive: a malformed plan on disk shouldn't deadlock a new run.
            self._plan = None


def load_plan_from_workspace(working_dir: str | Path) -> ResearchPlan | None:
    """Read a persisted plan from disk without instantiating the tool.

    Used by the topic-anchor reminder and the fidelity critic — both run
    in code paths that don't otherwise touch the tool registry.
    """
    path = Path(working_dir) / PLAN_FILENAME
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    try:
        return ResearchPlan.from_dict(data)
    except Exception:
        return None
