"""
Tracking utilities for LabForge.

Provides:
- PhaseTracker: tracks tool usage counts for phase hint logic
- ReviewerCallback: triggers LLM reviews at key checkpoints
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Callable

from .reviewer import Reviewer, ReviewResult
from .trajectory import Trajectory

logger = logging.getLogger(__name__)

StepCallback = Callable[[dict[str, Any]], None]


# Reviewer's latest verdict at the report / submit checkpoint is mirrored
# to this file so other tools (notably ``submit_result``) can consult it
# without holding a reference to the live callback object. Keeping the
# state on disk also means the gate works correctly when the same sandbox
# is replayed across runs.
REVIEWER_STATE_FILENAME = "reviewer_state.json"
# Score below which we treat the verdict as "must fix before submit".
# Calibrated to the existing reviewer scale (0.0-1.0). 0.6 catches the
# trajectory-194edbd4 case (final review = 0.50) without being so strict
# that minor-issue reviews block the run.
REVIEWER_BLOCK_SCORE_THRESHOLD = 0.6
# Maximum number of consecutive submit_result calls we'll bounce back to
# the agent before accepting the draft with an auto-injected
# "reviewer-flagged-but-unresolved" notice. Picked at 2 to mirror the
# topic-fidelity critic policy ("reject once, accept on retry") so the
# total worst-case rewrite cost is bounded across all gates.
REVIEWER_MAX_REWRITE_ATTEMPTS = 2


class PhaseTracker:
    """
    Tracks tool usage to determine the current research phase.

    Used by the agent to decide when to inject phase hints.
    """

    def __init__(self, max_steps: int = 30):
        self.max_steps = max_steps
        self.tool_counts: dict[str, int] = {}
        self.step_count: int = 0
        self.consecutive_failures: int = 0
        # Literature tracking
        self.literature_attempts: int = 0
        self.literature_failures: int = 0
        self.literature_successes: int = 0

    def record_tool_call(self, tool_name: str, success: bool) -> None:
        """Record a tool execution."""
        self.tool_counts[tool_name] = self.tool_counts.get(tool_name, 0) + 1
        self.step_count += 1

        if success:
            self.consecutive_failures = 0
        else:
            self.consecutive_failures += 1

        if tool_name in ("search_literature", "read_paper_fulltext"):
            self.literature_attempts += 1
            if success:
                self.literature_successes += 1
            else:
                self.literature_failures += 1

    @property
    def implementation_count(self) -> int:
        return sum(
            self.tool_counts.get(t, 0)
            for t in ("execute_code", "execute_bash", "file_write")
        )

    def get_phase_hint(self) -> str | None:
        """
        Return a phase transition hint if conditions are met, or None.

        Logic mirrors the original agent._get_phase_hint().
        """
        from .prompts import PHASE_HINTS, ERROR_RECOVERY_PROMPT

        lit_success = self.literature_successes
        impl = self.implementation_count
        step = self.step_count

        # Error recovery after 3 consecutive failures
        if self.consecutive_failures >= 3:
            self.consecutive_failures = 0
            return ERROR_RECOVERY_PROMPT

        # After 2+ successful literature searches and no implementation → plan
        if lit_success >= 2 and impl == 0 and step >= 3:
            return PHASE_HINTS["literature_to_planning"]

        # Partial literature → plan
        if (
            lit_success >= 1
            and self.literature_attempts >= 2
            and self.literature_failures > 0
            and impl == 0
            and step >= 3
        ):
            return PHASE_HINTS["literature_partial_to_planning"]

        # All literature searches failed
        if (
            lit_success == 0
            and self.literature_attempts >= 2
            and impl == 0
            and step >= 3
        ):
            return PHASE_HINTS["literature_search_failed"]

        # After enough implementation → analysis/report
        if impl >= 5 and step >= int(self.max_steps * 0.5):
            return PHASE_HINTS["analysis_to_report"]

        # Near the end → wrap up
        if step == int(self.max_steps * 0.8):
            return (
                "You are running low on steps. Please wrap up your work:\n"
                "1. Save any remaining results\n"
                "2. Generate a report with generate_report\n"
                "3. Call submit_result to finish"
            )

        return None


class ReviewerCallback:
    """
    Trigger LLM reviews at key checkpoints and surface them live.

    Unlike the previous version which buffered review events until the agent
    loop drained them between rounds, this one emits each review card the
    moment the reviewer LLM returns, so the UI and the terminal see it in
    real time instead of sitting silent during the review call.

    Failed reviews write their feedback into ``pending_feedback`` so the
    agent loop can inject it as a HumanMessage on the next round.
    """

    def __init__(
        self,
        reviewer: Reviewer,
        task_description: str,
        trajectory: Trajectory,
        step_callback: StepCallback | None = None,
        checkpoints: list[str] | None = None,
        workspace_dir: str | Path | None = None,
    ):
        self.reviewer = reviewer
        self.task_description = task_description
        self.trajectory = trajectory
        self.step_callback = step_callback
        self.checkpoints = set(checkpoints or ["after_literature", "before_report"])
        # Workspace dir is optional so legacy callers / unit tests can
        # construct the callback without a sandbox. When supplied, the
        # reviewer's latest report-checkpoint verdict is mirrored to
        # ``<workspace>/reviewer_state.json`` so ``submit_result`` can
        # consult it without holding a callback reference (see the
        # ``REVIEWER_STATE_FILENAME`` constant above).
        self.workspace_dir = Path(workspace_dir) if workspace_dir else None

        # State tracking
        self._literature_evidence: list[str] = []
        # Backward-compatible alias for older tests/extensions that still
        # inspect the previous name directly.
        self._search_results = self._literature_evidence
        self._code_outputs: list[str] = []
        self._tool_history_lines: list[str] = []
        self._literature_count: int = 0
        self._code_count: int = 0
        self._literature_reviewed: bool = False
        self._experiment_reviewed: bool = False

        # Pending feedback to inject into the next round's HumanMessage.
        self.pending_feedback: str | None = None
        # True whenever the most recent review failed — the agent loop uses
        # this to refuse an early submit-break so the reviewer's correction
        # actually reaches the main model before the run terminates.
        self.last_review_failed: bool = False
        # Count *consecutive* infrastructure failures (e.g. 401 invalid_model,
        # network timeout). After ``MAX_INFRA_FAILURES`` we stop blocking the
        # agent so a permanently broken reviewer cannot freeze the run forever.
        self._consecutive_infra_failures: int = 0
        self._infra_disabled: bool = False
        self._MAX_INFRA_FAILURES = 3

    def on_tool_end(self, output: str, **kwargs: Any) -> None:
        """Track tool outputs and trigger reviews at checkpoints."""
        tool_name = kwargs.get("name", "")

        # Track history. Literature evidence can include title/URL/manifest
        # details that are needed by the reviewer, so preserve a little more.
        history_limit = (
            600 if tool_name in ("search_literature", "read_paper_fulltext") else 200
        )
        self._tool_history_lines.append(
            f"[{tool_name}] {'SUCCESS' if not output.startswith('ERROR:') else 'FAILED'}: "
            f"{output[:history_limit]}"
        )

        if tool_name in ("search_literature", "read_paper_fulltext"):
            self._literature_count += 1
            if not output.startswith("ERROR:"):
                evidence_kind = (
                    "arXiv search result"
                    if tool_name == "search_literature"
                    else "uploaded/local full-text document"
                )
                self._literature_evidence.append(f"{evidence_kind}:\n{output[:2500]}")
            if (
                "after_literature" in self.checkpoints
                and self._literature_count >= 2
                and not self._literature_reviewed
            ):
                self._literature_reviewed = True
                self._run_literature_review()

        elif tool_name in ("execute_code", "execute_bash"):
            self._code_count += 1
            if not output.startswith("ERROR:"):
                self._code_outputs.append(output[:2000])
            if (
                "after_experiments" in self.checkpoints
                and self._code_count >= 5
                and not self._experiment_reviewed
            ):
                self._experiment_reviewed = True
                self._run_experiment_review(output)

        elif tool_name == "generate_report":
            if "before_report" in self.checkpoints:
                self._run_report_review(output)

        elif tool_name == "submit_result":
            if "before_submit" in self.checkpoints:
                self._run_submission_review(output)

    # ------------------------------------------------------------------
    # Individual review runners
    # ------------------------------------------------------------------
    def _run_literature_review(self) -> None:
        agent_summary = "\n".join(self._tool_history_lines[-5:])
        self._emit_thinking_card("文献评审")
        result = self.reviewer.review_literature(self._literature_evidence, agent_summary)
        self._handle_review_result("文献评审", result)

    def _run_experiment_review(self, latest_output: str) -> None:
        self._emit_thinking_card("实验评审")
        result = self.reviewer.review_experiment(
            self.task_description, self._code_outputs, latest_output
        )
        self._handle_review_result("实验评审", result)

    def _run_report_review(self, report_content: str) -> None:
        tool_history = "\n".join(self._tool_history_lines[-20:])
        if self._literature_evidence:
            evidence_text = "\n\n---\n\n".join(self._literature_evidence[-8:])
            tool_history = (
                f"{tool_history}\n\n[actual_literature_evidence]\n"
                f"{evidence_text[:4000]}"
            )
        self._emit_thinking_card("报告评审")
        result = self.reviewer.review_report(
            self.task_description, tool_history, report_content
        )
        self._handle_review_result("报告评审", result)

    def _run_submission_review(self, submission_output: str) -> None:
        trajectory_summary = "\n".join(self._tool_history_lines[-10:])
        self._emit_thinking_card("最终评审")
        result = self.reviewer.review_submission(
            self.task_description, trajectory_summary
        )
        self._handle_review_result("最终评审", result)

    # ------------------------------------------------------------------
    # Emitters
    # ------------------------------------------------------------------
    def _emit_thinking_card(self, checkpoint: str) -> None:
        """Emit a lightweight 'reviewer is working' card before the LLM call.

        We only push it to the live step_callback (UI + terminal). It's NOT
        added to the trajectory because the follow-up result card would just
        duplicate it in the saved trajectory JSON.
        """
        if self.step_callback is None:
            return
        try:
            self.step_callback({
                "step_index": len(self.trajectory.steps) + 1,
                "thought": f"评审模型 ({checkpoint}) 正在检查本阶段产出...",
                "action_name": "reviewer_review",
                "action_args": {},
                "observation": "（评审进行中）",
                "success": True,
                "metadata": {"source": "reviewer", "phase": "thinking"},
            })
        except Exception:
            logger.exception("Reviewer thinking-card emit failed")

    def _persist_state(self, checkpoint_name: str, result: ReviewResult) -> None:
        """Mirror the latest review verdict to disk so ``submit_result``
        (which doesn't hold a callback reference) can gate on it.

        We only persist verdicts from the report / submit checkpoints —
        literature/experiment reviews are advisory mid-run and shouldn't
        block the final submit. Best-effort: an OS error here just
        means the submit gate degrades to "no reviewer veto", which is
        the pre-existing behaviour.
        """
        if self.workspace_dir is None:
            return
        if checkpoint_name not in ("报告评审", "最终评审"):
            return
        try:
            self.workspace_dir.mkdir(parents=True, exist_ok=True)
            (self.workspace_dir / REVIEWER_STATE_FILENAME).write_text(
                json.dumps(
                    {
                        "checkpoint": checkpoint_name,
                        "score": float(result.score),
                        "passed": bool(result.passed),
                        "issues": list(result.issues or []),
                        "suggestion": result.suggestion or "",
                        "is_infrastructure_error": bool(getattr(result, "error", False)),
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
        except OSError as exc:
            logger.debug("reviewer_state.json write skipped: %s", exc)

    def _handle_review_result(self, checkpoint_name: str, result: ReviewResult) -> None:
        logger.info(
            "Review [%s]: passed=%s, score=%.2f, issues=%d, error=%s",
            checkpoint_name, result.passed, result.score, len(result.issues),
            getattr(result, "error", False),
        )
        self._persist_state(checkpoint_name, result)

        # Track consecutive infrastructure failures so a permanently broken
        # reviewer (e.g. wrong model name → 401) doesn't lock the run.
        if getattr(result, "error", False):
            self._consecutive_infra_failures += 1
            if self._consecutive_infra_failures >= self._MAX_INFRA_FAILURES and not self._infra_disabled:
                self._infra_disabled = True
                logger.error(
                    "Reviewer infrastructure failed %d times in a row at "
                    "checkpoint %s — disabling reviewer for the rest of this run "
                    "so it does not block submission. Please verify the reviewer "
                    "model / API Key configuration.",
                    self._consecutive_infra_failures, checkpoint_name,
                )
        else:
            self._consecutive_infra_failures = 0

        feedback: str | None = None
        if not result.passed and (result.issues or result.suggestion):
            issues_text = "\n".join(f"- {issue}" for issue in result.issues) or "- （评审模型未列出具体条目）"
            suggestion = result.suggestion or "按上面的问题逐条修正，再重新生成报告并提交。"
            feedback = (
                f"REVIEWER FEEDBACK ({checkpoint_name}):\n"
                f"Score: {result.score:.1f}/1.0\n"
                f"Issues found:\n{issues_text}\n"
                f"Suggestion: {suggestion}\n"
                f"Please fix the above issues before proceeding."
            )
            # Only block the submit when this is a *content* rejection — or
            # while we still have retry budget for infrastructure errors. Once
            # infra failures exceed the threshold we surface the feedback but
            # stop blocking so the agent can still produce a final result.
            self.pending_feedback = feedback
            self.last_review_failed = not self._infra_disabled
        else:
            self.last_review_failed = False

        # Build the user-visible observation for the review card.
        status_emoji = "✅" if result.passed else "⚠️"
        body_lines = [
            f"{status_emoji} {checkpoint_name} · Score {result.score:.2f}/1.0 "
            f"· {'通过' if result.passed else '需修正'}",
        ]
        if result.issues:
            body_lines.append("发现的问题：")
            body_lines.extend(f"- {x}" for x in result.issues)
        if result.suggestion:
            body_lines.append(f"建议：{result.suggestion}")
        if feedback and not result.passed:
            body_lines.append("主模型将在下一轮收到以上反馈并据此修正。")
        observation = "\n".join(body_lines)

        review_step = self.trajectory.add_step(
            thought=f"[REVIEWER · {checkpoint_name}]",
            action_name="reviewer_review",
            action_args={},
            observation=observation[:4000],
            success=bool(result.passed),
            metadata={
                "source": "reviewer",
                "checkpoint": checkpoint_name,
                "score": result.score,
                "passed": bool(result.passed),
            },
        )
        if self.step_callback is not None:
            try:
                self.step_callback({
                    "step_index": review_step.step_index + 1,
                    "thought": (
                        f"评审模型在 **{checkpoint_name}** 节点介入。"
                        + (" 主模型将按建议纠正。" if not result.passed else "")
                    ),
                    "action_name": "reviewer_review",
                    "action_args": {},
                    "observation": observation[:2000],
                    "success": bool(result.passed),
                    "metadata": {
                        "source": "reviewer",
                        "checkpoint": checkpoint_name,
                    },
                })
            except Exception:
                logger.exception("Reviewer result-card emit failed")
