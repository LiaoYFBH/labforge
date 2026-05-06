"""Adaptive step-budget guardrails for LabForge runs.

The budget is intentionally a *soft* completion budget plus a finite hard cap.
That lets a complex research run keep going when it is still making progress,
while still giving runaway loops a clear ceiling.
"""

from __future__ import annotations

import math
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


IGNORED_BUDGET_ACTIONS = {
    # Reviewer / human / system steps are not "agent work" and should not
    # eat the adaptive budget.
    "reviewer_review",
    "human_feedback",
    "human_abort",
    "human_review",
    "runtime_error",
    # Quality-gate steps: the agent tried to submit / move on, but a gate
    # (checklist, narration spiral) intercepted it. Charging the agent for
    # being intercepted would make every gate hit shrink the runway it
    # has to actually fix the gap, which is the opposite of what we want.
    "checklist_block",
    "narration_spiral_abort",
}

PRODUCTIVE_ACTIONS = {
    "search_literature",
    "read_paper_fulltext",
    "execute_code",
    "execute_bash",
    "file_write",
    "file_read",
    "list_files",
    "generate_report",
    "submit_result",
}


def _safe_int_env(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, "").strip() or default)
    except ValueError:
        return default


def _clamp(value: int, low: int, high: int) -> int:
    return max(low, min(high, value))


def _contains_any(text: str, patterns: tuple[str, ...]) -> bool:
    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns)


def infer_initial_step_budget(
    task_description: str = "",
    expected_output: str = "",
    data_description: str = "",
) -> int:
    """Infer a sensible initial budget from task complexity."""

    text = "\n".join([task_description or "", expected_output or "", data_description or ""])
    lowered = text.lower()
    budget = 55

    if len(text) > 1200:
        budget += 10
    if len(text) > 2600:
        budget += 15

    if data_description and "no additional data provided" not in lowered:
        budget += 12
    if _contains_any(lowered, (r"\buploads?/", r"\.pdf\b", r"\.csv\b", r"\.tsv\b", r"\.json\b")):
        budget += 10
    if _contains_any(lowered, (r"paperforge", r"paper forge", r"论文", r"manuscript", r"\bpdf\b", r"latex")):
        budget += 12
    if _contains_any(
        lowered,
        (
            r"experiment",
            r"benchmark",
            r"ablation",
            r"train",
            r"evaluate",
            r"dataset",
            r"compare",
            r"实验",
            r"评测",
            r"训练",
            r"数据集",
            r"对比",
            r"消融",
        ),
    ):
        budget += 18
    if _contains_any(lowered, (r"literature", r"survey", r"related work", r"文献", r"综述", r"调研")):
        budget += 8
    if _contains_any(lowered, (r"scienceagentbench", r"\bsab\b", r"official eval", r"benchmark task")):
        budget += 15

    return _clamp(budget, 45, 125)


def budget_step_count(steps: list[Any]) -> int:
    """Count only agent work steps that should consume the adaptive budget."""

    count = 0
    for step in steps:
        action = getattr(step, "action_name", None)
        if action not in IGNORED_BUDGET_ACTIONS:
            count += 1
    return count


def recent_productive_progress(steps: list[Any], window: int = 8) -> bool:
    """Return true when recent steps include successful task progress."""

    for step in steps[-window:]:
        action = getattr(step, "action_name", "")
        success = bool(getattr(step, "success", False))
        if success and action in PRODUCTIVE_ACTIONS:
            return True
    return False


def workspace_completion_flags(workspace_dir: str | Path | None) -> dict[str, bool]:
    """Inspect durable artifacts without requiring the UI layer."""

    if workspace_dir is None:
        return {"report": False, "bundle": False, "table": False, "figure": False}
    root = Path(workspace_dir)
    if not root.exists():
        return {"report": False, "bundle": False, "table": False, "figure": False}

    table_exts = {".csv", ".tsv"}
    figure_exts = {".png", ".jpg", ".jpeg", ".svg"}
    flags = {
        "report": (root / "research_report.md").exists(),
        "bundle": (root / "paperforge_bundle.json").exists(),
        "table": False,
        "figure": False,
    }
    try:
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            suffix = path.suffix.lower()
            if suffix in table_exts:
                flags["table"] = True
            elif suffix in figure_exts:
                flags["figure"] = True
            if flags["table"] and flags["figure"]:
                break
    except OSError:
        pass
    return flags


@dataclass
class AdaptiveStepBudget:
    """A finite, extensible budget for one LabForge run."""

    requested_steps: int
    initial_budget: int
    current_budget: int
    hard_cap: int
    adaptive: bool
    max_rounds: int
    max_nudges: int
    max_stalled_rounds: int
    extension_count: int = 0
    extension_limit: int = 4
    no_progress_rounds: int = 0
    stop_reason: str = ""
    last_wrap_hint_at: int = -1

    @classmethod
    def for_task(
        cls,
        requested_steps: int | None,
        task_description: str = "",
        expected_output: str = "",
        data_description: str = "",
    ) -> "AdaptiveStepBudget":
        requested = max(0, int(requested_steps or 0))
        absolute_cap = max(20, _safe_int_env("LAB_FORGE_ABSOLUTE_STEP_CAP", 240))

        if requested <= 0:
            initial = infer_initial_step_budget(task_description, expected_output, data_description)
            hard_cap = min(absolute_cap, max(120, math.ceil(initial * 1.8)))
            adaptive = True
        elif requested <= 20:
            # Tiny values are usually tests or deliberate smoke runs. Respect
            # them strictly so short-run callers do not unexpectedly expand.
            initial = requested
            hard_cap = requested
            adaptive = False
        else:
            initial = requested
            hard_cap = min(absolute_cap, max(initial + 20, math.ceil(initial * 1.5)))
            adaptive = hard_cap > initial

        max_rounds = _clamp(math.ceil(hard_cap / 6) + 6, 8, 48)
        max_nudges = _clamp(math.ceil(initial / 12), 3, 10)
        max_stalled_rounds = 4 if adaptive else 3

        return cls(
            requested_steps=requested,
            initial_budget=initial,
            current_budget=initial,
            hard_cap=hard_cap,
            adaptive=adaptive,
            max_rounds=max_rounds,
            max_nudges=max_nudges,
            max_stalled_rounds=max_stalled_rounds,
        )

    @property
    def label(self) -> str:
        if self.requested_steps <= 0:
            return f"Adaptive {self.current_budget}/{self.hard_cap}"
        if self.current_budget == self.hard_cap:
            return str(self.current_budget)
        return f"{self.current_budget}/{self.hard_cap}"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def consumed(self, steps: list[Any]) -> int:
        return budget_step_count(steps)

    def reached_soft_limit(self, steps: list[Any]) -> bool:
        return self.consumed(steps) >= self.current_budget

    def reached_hard_cap(self, steps: list[Any]) -> bool:
        return self.consumed(steps) >= self.hard_cap

    def should_stop_stream_after_step(self, steps: list[Any], submitted: bool) -> bool:
        if submitted:
            return False
        return self.reached_soft_limit(steps)

    def record_round_progress(self, new_budget_steps: int) -> None:
        if new_budget_steps > 0:
            self.no_progress_rounds = 0
        else:
            self.no_progress_rounds += 1

    def maybe_extend(self, steps: list[Any]) -> str | None:
        """Extend the soft budget when the run is still productively moving."""

        consumed = self.consumed(steps)
        if consumed < self.current_budget:
            return None
        if consumed >= self.hard_cap:
            self.stop_reason = "adaptive_step_budget_exhausted"
            return None
        if not self.adaptive or self.extension_count >= self.extension_limit:
            self.stop_reason = "adaptive_step_budget_exhausted"
            return None
        if not recent_productive_progress(steps):
            self.stop_reason = "stalled_at_step_budget"
            return None

        previous = self.current_budget
        room = self.hard_cap - self.current_budget
        extension = min(room, max(12, math.ceil(self.current_budget * 0.25)))
        self.current_budget += extension
        self.extension_count += 1

        return (
            "Adaptive step budget extension granted.\n"
            f"- Previous soft budget: {previous} agent steps\n"
            f"- New soft budget: {self.current_budget} agent steps\n"
            f"- Absolute hard cap: {self.hard_cap} agent steps\n\n"
            "You are not done yet, but you are still making useful progress. "
            "Continue with the missing work. Do not stop until you have a real "
            "report from generate_report and then submit_result, unless the "
            "hard cap is reached."
        )

    def maybe_wrapup_hint(self, steps: list[Any], workspace_dir: str | Path | None) -> str | None:
        consumed = self.consumed(steps)
        threshold = max(1, math.floor(self.current_budget * 0.8))
        if consumed < threshold:
            return None
        if self.last_wrap_hint_at >= 0 and consumed - self.last_wrap_hint_at < 10:
            return None

        flags = workspace_completion_flags(workspace_dir)
        missing: list[str] = []
        if not flags["table"]:
            missing.append("at least one CSV/TSV result table")
        if not flags["figure"]:
            missing.append("at least one saved figure")
        if not flags["report"]:
            missing.append("research_report.md via generate_report")
        if not flags["bundle"]:
            missing.append("paperforge_bundle.json via generate_report")

        self.last_wrap_hint_at = consumed
        missing_text = ", ".join(missing) if missing else "only final submit_result"
        return (
            "You are approaching the current adaptive step budget. "
            "Do not terminate early; focus on completion.\n"
            f"Missing or unverified: {missing_text}.\n"
            "Finish the smallest valid path to completion: repair any blocking "
            "errors, generate the report, then call submit_result."
        )

    def should_stop_for_stall(self, steps: list[Any]) -> bool:
        if self.reached_hard_cap(steps):
            self.stop_reason = "adaptive_step_budget_exhausted"
            return True
        if self.no_progress_rounds >= self.max_stalled_rounds:
            self.stop_reason = "stalled_without_submission"
            return True
        return False
