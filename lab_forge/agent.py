"""
Core agent using LangGraph's create_react_agent.

Architecture:
  - LangGraph create_react_agent (CompiledStateGraph) with MemorySaver
  - Multi-round execution: reviewer feedback & phase hints are injected as
    new HumanMessages between rounds so the agent actually sees them
  - Trajectory recording
  - Compatible with any OpenAI-compatible API (AI Studio, MiniMax, etc.)
"""

from __future__ import annotations

import logging
import re
import uuid
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

from .callbacks import PhaseTracker, ReviewerCallback
from .config import AgentConfig
from .langchain_tools import create_all_tools
from .models import create_chat_model
from .prompts import STEP_FORMAT_HINT, SYSTEM_PROMPT, TASK_PROMPT_TEMPLATE
from .reviewer import Reviewer
from .result_guardrails import (
    blocking_findings,
    format_findings,
    report_text_discloses_findings,
    validate_workspace_results,
)
from .sandbox import Sandbox
from .step_budget import AdaptiveStepBudget, budget_step_count, recent_productive_progress
from .task_checklist import (
    TaskChecklist,
    format_topic_reminder,
    parse_task_checklist,
)
from .tools.base import ToolResult
from .tools.submit_tool import SubmitResultTool
from .trajectory import Trajectory

logger = logging.getLogger(__name__)


# Reviewer feedback that mentions any of these signals is treated as an
# experiment-class issue: the agent should fix code/data, not regenerate the
# report. Patterns are case-insensitive substring matches against the raw
# feedback text. Bilingual on purpose because the AI Studio reviewer mixes
# Chinese and English in the same response.
_EXPERIMENTAL_FEEDBACK_SIGNALS = (
    "nan",
    "非有限",
    "negative loss",
    "负损失",
    "负值",
    "负数",
    "loss值为负",
    "损失为负",
    "为负数",
    "loss=-",
    "loss: -",
    "loss值为-",
    "损失值为-",
    "数值不稳定",
    "instabil",   # instability / unstable
    "unstable",
    "diverg",     # divergent / diverge
    "数值发散",
    "梯度爆炸",
    "gradient explod",
    "训练失败",
    "实验失败",
    "training fail",
    "experiment fail",
    "timeout",
    "timed out",
    "超时",
    "未收敛",
    "did not converge",
)

# Reviewer feedback classed as ``scope`` means "you haven't done what the task
# asked" — missing baselines, no novel algorithm, perfect-too-good metrics on
# toy data, etc. The remedy is more experimental work (or more literature
# review), NOT another generate_report iteration. We keep this list
# high-precision: each phrase clearly accuses a scope/task-fulfillment gap
# rather than a wording or numerical problem (those have their own classes).
_SCOPE_FEEDBACK_SIGNALS = (
    # "you didn't build / didn't compare what the task required"
    "未开发",
    "未实现",
    "未对比",
    "未做对比",
    "未做baseline",
    "未做基线",
    "缺少基线",
    "无基线",
    "no baseline",
    "no comparison",
    "lacks comparison",
    "did not develop",
    "did not implement",
    "did not propose",
    "did not present",
    "did not introduce",
    "did not compare",
    "did not test",
    "did not validate",
    # comparison-quality complaints (caught from real trajectories where
    # the agent compared two existing methods but didn't address novelty)
    "对比不充分",
    "对比不足",
    "缺乏对比",
    "对比实验不充分",
    "对比实验不足",
    "全面对比",          # 通常出现在「未能展示与基线方法的全面对比」上下文
    "完整的基线",
    "完整对比",
    "缺乏完整",
    # "task goal not (fully) reached" — common phrasing variants
    "未达到目标",
    "未达到预期",
    "任务目标未完全完成",
    "目标未完全完成",
    "未完全完成任务",
    "未完全达标",
    "未达标",
    "部分完成但未",
    "任务目标部分完成",
    # "didn't propose / didn't innovate" — direct novelty gap signals
    "未提出新方法",
    "未提出新算法",
    "未提出创新",
    "未提出.*创新",       # regex-flavored substring; ``in`` check still works
    "未设计新",
    # "did not prove the method's superiority"
    "未证明优越性",
    "未证明方法优越性",
    "未能证明",
    "did not prove",
    "did not demonstrate superiority",
    "did not outperform",
    "未超越基线",
    "未超过基线",
    # over-fit / toy-data signals (reviewer flagging "results too good to be true")
    "完美指标",
    "perfect metric",
    "perfect precision",
    "perfect recall",
    "perfect f1",
    "perfect score",
    "合成数据过于简单",
    "toy synthetic",
    "过于简单的合成",
    "overfit",
    "over-fitting",
    "over fit",
    "过拟合模型",
    # "no validation on real data"
    "未验证真实",
    "缺乏对真实数据",
    "缺乏对真实世界",
    "未在真实数据集",
    # "no innovation / not novel"
    "缺乏创新",
    "缺乏新颖",
    "no novelty",
    "lacks innovation",
    "lacks novelty",
    # explicit experimental-design complaint
    "实验设计缺陷",
    "experiment design flaw",
)


def _classify_reviewer_feedback(feedback: str) -> str:
    """Return one of ``"experimental"`` | ``"scope"`` | ``"writing"`` based on
    which class of issue the reviewer is flagging.

    The distinction drives the coaching: an *experimental* issue (NaN losses,
    divergence, timeouts) needs another ``execute_code`` to fix the code; a
    *scope* issue (missing baseline, no novel algorithm, perfect-too-good
    metrics on toy data) needs more experimental work or more literature
    review — definitely NOT another ``generate_report``; a *writing* issue
    can be addressed by regenerating the report with corrected content.

    Priority when multiple signals are present: experimental > scope > writing.
    Experimental wins because it's the most concrete failure mode (a NaN is
    a NaN) and tends to block any meaningful scope work downstream.
    """
    if not feedback:
        return "writing"
    text = feedback.lower()
    if any(sig in text for sig in _EXPERIMENTAL_FEEDBACK_SIGNALS):
        return "experimental"
    if any(sig in text for sig in _SCOPE_FEEDBACK_SIGNALS):
        return "scope"
    return "writing"


# Path B (honest negative-result framing) is gated so the agent can't escape
# debugging by declaring failure too early. It only becomes mentionable after
# enough blocks AND enough real debugging effort. These knobs are intentionally
# strict — the goal is for the agent to actually solve the problem; Path B is
# a last-resort safety valve, not a peer option.
_PATH_B_UNLOCK_BLOCK = 5
_PATH_B_MIN_EXEC_ATTEMPTS = 3


def _count_exec_attempts_after(steps, after_idx: int) -> int:
    """Count execute_code / execute_bash steps strictly after ``after_idx``.

    Failed attempts still count as "trying" — what we want to see is the agent
    actually engaging with the experiment, not whether each attempt succeeded.
    """
    return sum(
        1
        for i, s in enumerate(steps)
        if i > after_idx and s.action_name in {"execute_code", "execute_bash"}
    )


def _compose_submit_block_message(
    feedback: str,
    block_count: int,
    feedback_class: str,
    exec_attempts_since_first_block: int,
) -> str:
    """Build the ``HumanMessage`` body that's injected when reviewer rejects
    a ``submit_result``.

    Escalation policy (the goal is for the agent to actually fix the bug;
    Path B exists only so the agent doesn't have to lie when the bug genuinely
    can't be fixed in the remaining budget):

    * Block 1 — relay reviewer feedback verbatim with a standard fix prompt.
    * Block 2 (experimental) — redirect: next tool MUST be ``execute_code``,
      not ``generate_report``.
    * Block 2 (writing) — sharpen the writing-fix framing.
    * Blocks 3, 4 — Path A only, with progressively pointed diagnostics. No
      mention of Path B yet — the agent should be debugging, not looking for
      an exit.
    * Block 5+ — Path A is still the primary instruction. Path B (honest
      negative result) only unlocks if the agent has run ``execute_code`` at
      least :data:`_PATH_B_MIN_EXEC_ATTEMPTS` times since the first block —
      i.e. it has demonstrably tried to fix the bug. If it hasn't, Path B
      stays hidden and the agent is told it has not yet earned the off-ramp.
    """
    header = f"REVIEWER FEEDBACK (block {block_count}):\n{feedback}"

    if block_count <= 1:
        return (
            f"{header}\n\n"
            "Your previous submit was blocked because the reviewer rejected the "
            "work. Address every issue above, regenerate the report (call "
            "generate_report again with corrected content), then call "
            "submit_result."
        )

    if block_count == 2 and feedback_class == "experimental":
        return (
            f"{header}\n\n"
            "This is your SECOND block for what looks like an EXPERIMENTAL "
            "issue (NaN / negative loss / divergence / instability / timeout), "
            "not a writing issue. The reviewer is correct: re-wording the "
            "report will not fix broken experimental results.\n\n"
            "DO NOT call generate_report next. Your next tool call MUST be "
            "execute_code, with a concrete plan:\n"
            "  1. Re-read your most recent execute_code stdout/stderr — what "
            "     numbers actually appeared, what was unexpected?\n"
            "  2. Identify the ROOT CAUSE in the code (common contrastive-"
            "     learning bugs: missing log, wrong sign in InfoNCE, missing "
            "     L2-normalization of embeddings, learning rate too high, "
            "     temperature parameter wrong, in-batch negatives mis-aligned).\n"
            "  3. Apply a minimal code fix and run it.\n"
            "  4. Verify the fix produced sane numbers (loss in expected range, "
            "     no NaN/inf) BEFORE calling generate_report again."
        )

    if block_count == 2 and feedback_class == "scope":
        return (
            f"{header}\n\n"
            "This is your SECOND block, and the reviewer is telling you that "
            "what you submitted DOES NOT match what the task asked for "
            "(missing baseline, no novel algorithm, perfect-too-good metrics "
            "on toy data, no validation on real data, etc). This is NOT a "
            "writing problem — re-wording the report will not fix it.\n\n"
            "DO NOT call generate_report next. Your next tool call MUST be "
            "either:\n"
            "  - search_literature (within remaining quota) — find the "
            "    standard baseline / dataset / method the task implies, OR\n"
            "  - execute_code — actually build/run the missing artifact "
            "    (e.g. a real baseline comparison, a real-data evaluation, "
            "    a non-toy variant of the experiment), OR\n"
            "  - lookup_paper_code — find a reference implementation you can "
            "    clone and adapt (per the EXECUTION ENVIRONMENT block).\n\n"
            "Re-read the reviewer bullets above and identify the SPECIFIC "
            "missing piece. State it as one sentence in your reasoning, then "
            "execute the tool call that addresses it. Do not regenerate "
            "the report with the same evidence and a different framing."
        )

    if block_count == 2:
        return (
            f"{header}\n\n"
            "This is your SECOND block. The reviewer's complaints are about "
            "report content/disclosure. Read each bullet carefully, fix it "
            "concretely in the report (not by paraphrasing), and call "
            "submit_result. If the reviewer is asking for evidence you don't "
            "have, run execute_code first to produce it."
        )

    # Blocks 3+. Build the diagnosis once.
    if feedback_class == "experimental":
        diagnosis = (
            "The reviewer keeps flagging the SAME class of EXPERIMENTAL issue "
            "(NaN / negative loss / divergence / instability / timeout). "
            "Re-writing the report will not make this go away — the bug is in "
            "your code or experimental setup."
        )
    elif feedback_class == "scope":
        diagnosis = (
            "The reviewer keeps flagging the SAME SCOPE gap — what you've "
            "submitted does not match what the task asked for (missing "
            "baseline, no real novel algorithm, perfect-too-good metrics on "
            "toy data, no real-data validation, etc). Re-writing the report "
            "will NOT close this gap; you need to actually do the missing "
            "work."
        )
    else:
        diagnosis = (
            "The reviewer keeps flagging report-content issues that haven't "
            "been resolved. Either the fix wasn't actually applied, or the "
            "evidence the reviewer is asking for is missing from the workspace."
        )

    # Blocks 3, 4: Path A only. No Path B mention; the agent should be
    # debugging, not negotiating an exit. Add cumulatively sharper concrete
    # debug suggestions so each block contributes new information.
    if block_count < _PATH_B_UNLOCK_BLOCK:
        if feedback_class == "scope":
            debug_steps = (
                "  1. Read the reviewer bullets and write down the EXACT "
                "     missing artifact in one sentence (e.g. 'no comparison "
                "     to a real baseline', 'no evaluation on a real dataset', "
                "     'no novel mechanism beyond the baseline').\n"
                "  2. Build that artifact with execute_code: implement / "
                "     train / evaluate the missing baseline, swap the toy "
                "     synthetic data for a real (small) dataset, add the "
                "     novelty mechanism the task requested. ONE missing "
                "     piece per execute_code call so you can tell what "
                "     landed.\n"
                "  3. If the gap is 'no novel method' and you genuinely "
                "     don't have one, search_literature for related work "
                "     and use lookup_paper_code on a key paper to find a "
                "     reference implementation you can adapt as your "
                "     starting point.\n"
                "  4. Verify the new artifacts exist in the workspace "
                "     (CSV / PNG / log) BEFORE regenerating the report."
            )
        else:
            debug_steps = (
                "  1. file_read your latest execute_code log; copy the actual "
                "     numeric outputs into your reasoning.\n"
                "  2. Form a SPECIFIC hypothesis about what's wrong — name the "
                "     variable / line / formula you suspect. Generic 'numerical "
                "     instability' is not a hypothesis.\n"
                "  3. execute_code with a TARGETED fix (one change at a time so "
                "     you can tell what helped). For contrastive learning, common "
                "     culprits: missing log in InfoNCE, wrong sign, embeddings not "
                "     L2-normalized, temperature too small/large, learning rate, "
                "     mis-aligned positive/negative pairs.\n"
                "  4. Verify the new numbers are sane (loss ≥ 0, no NaN/inf, "
                "     decreases over epochs) BEFORE regenerating the report."
            )
        if block_count == 3:
            preface = (
                "You have been blocked 3 times. The first two rounds didn't "
                "fix the underlying issue."
            )
        else:  # block_count == 4
            preface = (
                "You have been blocked 4 times. So far you keep landing on "
                "the same failure mode. STOP and break the pattern: a "
                "different fix, not the same fix re-attempted."
            )
        return (
            f"{header}\n\n"
            f"{preface} {diagnosis}\n\n"
            "Your next move MUST be Path A — fix the underlying problem:\n"
            f"{debug_steps}\n\n"
            "Do NOT call generate_report or submit_result until execute_code "
            "shows the issue is resolved. There is no shortcut."
        )

    # Block 5+. Decide whether Path B is unlocked based on debugging effort.
    path_b_earned = exec_attempts_since_first_block >= _PATH_B_MIN_EXEC_ATTEMPTS

    if not path_b_earned:
        return (
            f"{header}\n\n"
            f"You have been blocked {block_count} times, but you have only "
            f"called execute_code {exec_attempts_since_first_block} time(s) "
            f"since the first block — the threshold for considering an honest "
            f"negative-result writeup is "
            f"{_PATH_B_MIN_EXEC_ATTEMPTS} debug attempts. {diagnosis}\n\n"
            "You have not earned an off-ramp yet. Your next tool call MUST be "
            "execute_code (or file_read on a prior log). Diagnose with a "
            "specific hypothesis, apply ONE targeted change, and verify the "
            "numbers. No more generate_report / submit_result until then."
        )

    # Path B unlocked: lead with Path A, present Path B only as last resort
    # with explicit qualifications about when it applies. The Path A and Path
    # B copy is class-specific — for "scope" issues, Path A means doing the
    # missing experimental work (not a debug fix), and Path B is a negative
    # result framing that quantifies WHY the task spec was unreachable rather
    # than a numerical-failure characterization.
    if feedback_class == "scope":
        path_a_block = (
            "PATH A — DO THE MISSING WORK (still the right answer if there's "
            "any concrete missing piece you can build):\n"
            "  - Identify ONE specific gap (missing baseline, missing real-"
            "    data evaluation, missing novelty mechanism) and build it "
            "    with execute_code or search_literature + lookup_paper_code.\n"
            "  - Verify the artifact exists in the workspace (CSV / PNG / "
            "    log). Only then regenerate the report and submit_result."
        )
        path_b_block = (
            "PATH B — HONEST 'TASK NOT ACHIEVABLE WITHIN BUDGET' REPORT "
            "(last resort; choose only if you've tried at least 2 distinct "
            "approaches and can specifically explain why the task as stated "
            "can't be completed in the remaining budget):\n"
            "  Write the report explicitly as a scoped-down attempt:\n"
            "    1. Title and abstract say 'we attempted <task X> on <real "
            "       constraints>; we were able to complete <reduced "
            "       scope Y>; the full task <Z> remains open'.\n"
            "    2. Quantify exactly what was achieved vs what was asked "
            "       (e.g. baseline trained but no novelty implemented; or "
            "       method works on synthetic data but not the real "
            "       dataset).\n"
            "    3. Describe the specific obstacle (compute / time / "
            "       missing reference implementation / unstable training).\n"
            "    4. 'Limitations' / 'Future work' names what's needed to "
            "       finish (more compute, more time, a specific dataset, etc).\n"
            "  Then submit_result. An honest scoped-down attempt is a "
            "  legitimate output; falsely claiming the full task on "
            "  toy/synthetic data is not."
        )
    else:
        path_a_block = (
            "PATH A — FIX THE UNDERLYING PROBLEM (still the right answer if "
            "you have any plausible angle left):\n"
            "  - Your next tool call should be execute_code with a fix you "
            "    have NOT tried before. Look at what the previous attempts "
            "    actually changed and pick a different lever (loss "
            "    formulation, normalization, learning rate, temperature, "
            "    batch construction).\n"
            "  - Verify with new numeric outputs. Only then regenerate the "
            "    report and submit_result."
        )
        path_b_block = (
            "PATH B — HONEST NEGATIVE-RESULT REPORT (last resort; choose "
            "only if you have tried at least 2 distinct fixes that did not "
            "work, AND you have a specific reason to believe further attempts "
            "within the remaining step budget won't help):\n"
            "  Write the report explicitly as a negative result. The "
            "  reviewer accepts this when:\n"
            "    1. Title and abstract say 'attempted X; observed Y; the "
            "       approach failed because Z'.\n"
            "    2. Numbers are quantified (which loss values, which epochs, "
            "       which configurations failed). Do not claim success "
            "       anywhere.\n"
            "    3. A root-cause hypothesis is stated based on what you tried.\n"
            "    4. 'Limitations' / 'Future work' section names what would be "
            "       needed to make it work.\n"
            "  Then submit_result. A well-characterized failure is a "
            "  legitimate research contribution; a falsely-positive report "
            "  is not."
        )

    return (
        f"{header}\n\n"
        f"You have been blocked {block_count} times and have run "
        f"execute_code {exec_attempts_since_first_block} time(s) since the "
        f"first block. {diagnosis}\n\n"
        f"{path_a_block}\n\n"
        f"{path_b_block}\n\n"
        "Default to Path A unless you can name two distinct attempts you've "
        "already made. Path B is not a shortcut — you must justify it."
    )


class ResearchAgent:
    """
    LangGraph-based scientific research agent.

    Uses create_react_agent with MemorySaver checkpointer so that
    reviewer feedback and phase hints can be injected between rounds
    and the agent sees the full conversation history.
    """

    def __init__(
        self,
        config: AgentConfig,
        system_prompt: str | None = None,
        enable_phase_hints: bool = True,
        tool_names: set[str] | None = None,
    ):
        self.config = config
        self.enable_phase_hints = enable_phase_hints
        self.sandbox = Sandbox(config.sandbox)

        # Backward-compatible guardrail state used by older unit tests and by
        # callers that still inspect the pre-LangGraph agent surface.
        self._phase_actions: dict[str, int] = {
            "literature": 0,
            "implementation": 0,
            "analysis": 0,
            "report": 0,
        }
        self._literature_attempts = 0
        self._literature_failures = 0
        self._literature_surnames: set[str] = set()

        # Create LLM instances
        self.agent_llm = create_chat_model(config.agent_model)

        # Create reviewer (if enabled). Pass the agent's main LLM as a
        # fallback: if the configured reviewer model is rejected by the
        # provider (e.g. AI Studio retired ERNIE 3.5/Speed/Lite/Tiny in
        # 2026 and returns invalid_model), the reviewer auto-migrates to
        # the agent LLM for the rest of the run instead of silently going
        # dark and letting hallucinated work ship.
        self.reviewer: Reviewer | None = None
        if config.reviewer.enabled:
            reviewer_llm = create_chat_model(config.reviewer.model)
            self.reviewer = Reviewer(reviewer_llm, fallback_llm=self.agent_llm)

        # Create tools (optionally restrict the set for benchmark runs).
        # We reuse the main agent model as the per-section "writer" LLM
        # inside generate_report so the tool can do outline → per-section
        # expansion automatically. The agent only sees a single tool call;
        # the multi-LLM-call expansion happens inside the tool.
        self.submit_tool = SubmitResultTool()
        # Build the topic-fidelity critic. It runs at generate_report submit
        # time to compare the report against the locked scope plan.
        # Reuses the reviewer LLM when available so we don't pay for an
        # extra model; falls back to the agent LLM otherwise. Returns None
        # if neither is available, in which case the critic becomes a no-op
        # — the run still proceeds, just without the extra fidelity gate.
        from .topic_fidelity import build_topic_fidelity_critic

        fidelity_critic = build_topic_fidelity_critic(
            llm=(
                self.reviewer.llm
                if (self.reviewer is not None and getattr(self.reviewer, "llm", None) is not None)
                else self.agent_llm
            ),
        )
        # Outcome judge — same fallback pattern as the fidelity critic.
        # Prefer the reviewer LLM (it's the "second opinion" model) so
        # the agent's main LLM doesn't grade its own work; fall back to
        # the agent LLM when no reviewer is configured.
        outcome_judge_llm = (
            self.reviewer.llm
            if (self.reviewer is not None and getattr(self.reviewer, "llm", None) is not None)
            else self.agent_llm
        )
        # Vision LLM is opt-in. The default deepseek-v3 deployment is
        # text-only, so we leave this None unless explicitly configured.
        # When None the vision figure-semantics gate skips silently and
        # the existing blank-figure pixel check remains the primary
        # figure guardrail.
        vision_llm = getattr(config, "vision_llm_model", None)
        if vision_llm is not None:
            try:
                vision_llm = create_chat_model(vision_llm)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to instantiate vision LLM; vision figure check disabled: %s", exc)
                vision_llm = None

        self.tools, self.submit_tool, self.scope_lock = create_all_tools(
            sandbox=self.sandbox,
            working_dir=config.sandbox.working_dir,
            ocr_enabled=config.ocr_enabled,
            submit_tool=self.submit_tool,
            tool_names=tool_names,
            writer_llm=self.agent_llm,
            search_quota=int(getattr(config, "search_quota", 0) or 0),
            fidelity_critic=fidelity_critic,
            outcome_judge_llm=outcome_judge_llm,
            vision_llm=vision_llm,
        )

        # Build the LangGraph react agent with MemorySaver for state persistence
        if system_prompt is None:
            system_prompt = SYSTEM_PROMPT + "\n\n" + STEP_FORMAT_HINT
        self.checkpointer = MemorySaver()
        self.graph = create_react_agent(
            self.agent_llm,
            self.tools,
            prompt=system_prompt,
            checkpointer=self.checkpointer,
        )

    def _heal_pending_tool_calls(self, thread_config: dict, error_note: str) -> list:
        """
        If the last AIMessage in the thread has pending tool_calls (no matching
        ToolMessage responses), synthesize ToolMessages with an error note so
        the provider stops rejecting the replayed conversation.

        Returns the list of synthetic messages (possibly empty).
        """
        try:
            state = self.graph.get_state(thread_config)
        except Exception as exc:
            logger.warning("Could not inspect graph state for healing: %s", exc)
            return []

        messages = (state.values or {}).get("messages", []) if state else []
        if not messages:
            return []

        responded_ids = {
            getattr(m, "tool_call_id", None)
            for m in messages
            if isinstance(m, ToolMessage)
        }

        synthetic: list = []
        for msg in messages:
            if not isinstance(msg, AIMessage):
                continue
            for tc in getattr(msg, "tool_calls", None) or []:
                tc_id = tc.get("id") if isinstance(tc, dict) else getattr(tc, "id", None)
                if not tc_id or tc_id in responded_ids:
                    continue
                tc_name = tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", "unknown")
                synthetic.append(
                    ToolMessage(
                        content=f"ERROR: {error_note}",
                        name=tc_name or "unknown",
                        tool_call_id=tc_id,
                    )
                )
                responded_ids.add(tc_id)
        if synthetic:
            logger.info(
                "Healing %d unmatched tool_call(s) with synthetic error response",
                len(synthetic),
            )
        else:
            logger.info(
                "No unmatched tool_calls found in %d-message thread state",
                len(messages),
            )
        return synthetic

    def _stream_round(
        self,
        input_messages: dict,
        thread_config: dict,
        trajectory: Trajectory,
        phase_tracker: PhaseTracker | None,
        reviewer_cb: ReviewerCallback | None,
        step_callback=None,
        should_stop_after_step=None,
    ) -> str | None:
        """
        Run one round of graph.stream() and record events.

        Because MemorySaver is used, the graph accumulates all messages
        across rounds under the same thread_id.
        """
        # In the ReAct loop, the agent node emits reasoning (AIMessage.content)
        # BEFORE the corresponding tool call runs. We carry that reasoning in
        # pending_thought and attach it to the next tool step — so the UI shows
        # "why I am about to run this tool," not a blank thought.
        pending_thought = ""
        # tool_call_id -> {"name": str, "args": dict} captured from the
        # agent node. Used to surface the actual code / bash / file_write
        # payload on the corresponding tool step in the UI.
        pending_tool_args: dict[str, dict] = {}

        def _step_limit_reached() -> bool:
            if should_stop_after_step is None:
                return False
            try:
                return bool(should_stop_after_step(trajectory))
            except Exception:
                logger.exception("Adaptive step-budget callback failed")
                return False

        for event in self.graph.stream(
            input_messages,
            config=thread_config,
            stream_mode="updates",
        ):
            for node_name, node_output in event.items():
                if node_name == "tools":
                    messages = node_output.get("messages", [])
                    for msg in messages:
                        tool_name = getattr(msg, "name", "unknown")
                        content = msg.content if hasattr(msg, "content") else str(msg)
                        is_error = str(content).startswith("ERROR:")

                        tool_call_id = getattr(msg, "tool_call_id", None)
                        tool_info = pending_tool_args.pop(tool_call_id, None) if tool_call_id else None
                        args_dict = (tool_info or {}).get("args") or {}

                        step_thought = pending_thought[:2000]
                        step = trajectory.add_step(
                            thought=step_thought,
                            action_name=tool_name,
                            action_args=args_dict,
                            observation=str(content)[:4000],
                            success=not is_error,
                            metadata={"exec_time_s": 0.0},
                        )
                        # Consume the thought so the next tool in the same
                        # round (rare but possible) doesn't reuse it.
                        pending_thought = ""

                        if phase_tracker is not None:
                            phase_tracker.record_tool_call(tool_name, not is_error)

                        if reviewer_cb:
                            reviewer_cb.on_tool_end(str(content), name=tool_name)

                        if step_callback:
                            step_callback({
                                "step_index": step.step_index + 1,
                                "thought": step.thought,
                                "action_name": tool_name,
                                "action_args": args_dict,
                                "observation": str(content)[:4000],
                                "success": not is_error,
                            })

                    if _step_limit_reached():
                        return "soft_step_budget_reached"

                elif node_name == "agent":
                    messages = node_output.get("messages", [])
                    for msg in messages:
                        content = msg.content if hasattr(msg, "content") else ""
                        tool_calls = getattr(msg, "tool_calls", None) or []

                        # Record the args of each outgoing tool call so the
                        # corresponding ToolMessage can surface them on the UI
                        # (e.g. the actual `code` that execute_code will run).
                        for tc in tool_calls:
                            if isinstance(tc, dict):
                                tc_id = tc.get("id")
                                tc_name = tc.get("name")
                                tc_args = tc.get("args") or {}
                            else:
                                tc_id = getattr(tc, "id", None)
                                tc_name = getattr(tc, "name", None)
                                tc_args = getattr(tc, "args", {}) or {}
                            if tc_id:
                                pending_tool_args[tc_id] = {
                                    "name": tc_name or "unknown",
                                    "args": tc_args,
                                }

                        text = str(content).strip() if content else ""
                        if not text:
                            continue

                        # A "substantive" reasoning block gets its own "plan"
                        # card so the user sees the research plan before the
                        # tool action — even when the agent produced the plan
                        # in the same message as the tool call. Heuristic:
                        # long text, multi-line text, or text that looks like
                        # structured planning (phase headers, numbered lists).
                        is_long = len(text) >= 200
                        is_multiline = text.count("\n") >= 2
                        looks_structured = any(
                            marker in text
                            for marker in (
                                "Phase ", "phase ", "阶段", "方案",
                                "## ", "### ", "1.", "1)", "- ",
                            )
                        )
                        no_tool_call = not tool_calls
                        emit_plan_card = (
                            no_tool_call
                            or is_long
                            or is_multiline
                            or looks_structured
                        )

                        if emit_plan_card:
                            plan_step = trajectory.add_step(
                                thought=text[:2000],
                                action_name="plan",
                                action_args={},
                                observation="",
                                success=True,
                                metadata={"exec_time_s": 0.0},
                            )
                            if step_callback:
                                step_callback({
                                    "step_index": plan_step.step_index + 1,
                                    "thought": plan_step.thought,
                                    "action_name": "plan",
                                    "observation": "",
                                    "success": True,
                                })
                            # If this AI message also contains tool calls,
                            # let the matching ToolMessages run before pausing;
                            # stopping here would leave pending tool calls in
                            # the LangGraph thread and poison the next round.
                            if not tool_calls and _step_limit_reached():
                                return "soft_step_budget_reached"
                            # The plan card already carries the reasoning, so
                            # don't duplicate it on the next tool card.
                            pending_thought = ""
                        else:
                            # Short, one-line reasoning — keep it inline on
                            # the next tool card rather than adding a card.
                            pending_thought = text

            if self.submit_tool.submitted:
                break

        return None

    def run(
        self,
        task_id: str,
        task_description: str,
        expected_output: str = "",
        data_description: str = "",
        step_callback=None,
    ) -> Trajectory:
        """
        Run the agent on a single research task.

        Uses a multi-round loop with MemorySaver: after each round,
        reviewer feedback and phase hints are injected as new
        HumanMessages into the same thread so the agent sees them
        alongside its full conversation history.
        """
        self.submit_tool.reset()

        # P5 fix: forward task_description into the per-task tool state so
        # ``GenerateReportTool._validate_experiment_evidence`` can detect
        # SURVEY ONLY mode and skip the figure / table requirement (which
        # would otherwise force the agent to fabricate a placeholder.png).
        for _tool in self.tools:
            inner = getattr(_tool, "_lab_forge_inner", None)
            if inner is not None and hasattr(inner, "task_description"):
                inner.task_description = task_description

        task_prompt = TASK_PROMPT_TEMPLATE.format(
            task_description=task_description,
            expected_output=expected_output or "See task description.",
            data_description=data_description or "No additional data provided.",
        )

        trajectory = Trajectory(task_id=task_id, task_description=task_description)
        step_budget = AdaptiveStepBudget.for_task(
            requested_steps=self.config.max_steps,
            task_description=task_description,
            expected_output=expected_output,
            data_description=data_description,
        )
        trajectory.metadata["step_budget"] = step_budget.to_dict()
        phase_tracker = (
            PhaseTracker(max_steps=step_budget.current_budget)
            if self.enable_phase_hints
            else None
        )

        # Goal-anchoring (Mod D): one-shot LLM call at run start parses the
        # task description into a verifiable checklist of 3-5 concrete required
        # artifacts. The checklist is then re-evaluated every 5 rounds and
        # injected as the first item in each round's HumanMessage so the agent
        # never loses sight of the original task spec — even after dozens of
        # rounds of intermediate work and reviewer feedback. Empty checklist
        # (parse failed / trivial task) silently disables the anchoring.
        task_checklist: TaskChecklist = parse_task_checklist(task_description, self.agent_llm)
        if task_checklist.items:
            trajectory.metadata["task_checklist"] = {
                "items": [
                    {"id": i.id, "title": i.title, "verification_hint": i.verification_hint}
                    for i in task_checklist.items
                ],
            }
        # Re-inject the static topic-anchor reminder every TOPIC_REMINDER_INTERVAL
        # rounds. The reminder is filesystem-verified and LLM-free (see
        # task_checklist.format_topic_reminder), so unlike the old
        # evaluate_checklist_status path it cannot flip turn-to-turn and
        # cannot deadlock submit_result.
        TOPIC_REMINDER_INTERVAL = 5

        # Pull the original topic out of the task_description so the reminder
        # block can show it verbatim. ``topic_to_task`` writes
        # ``"Research Topic: <topic>\n\n"`` as the first line; if the caller
        # bypassed it (e.g. raw --task in main.py), we fall back to the
        # first non-empty line as a best-effort topic.
        original_topic = ""
        for raw_line in (task_description or "").splitlines():
            line = raw_line.strip()
            if not line:
                continue
            if line.lower().startswith("research topic:"):
                original_topic = line.split(":", 1)[1].strip()
                break
            original_topic = line
            break

        # Recover the run mode from the [MODE: …] banner that ``topic_to_task``
        # injects, so the topic reminder can issue the experiment-only
        # thin-workspace warning. Raw --task runs that don't include the
        # banner default to "survey" (the safer branch — no false alarms).
        if "[MODE: EXPERIMENT" in (task_description or ""):
            original_mode = "experiment"
        else:
            original_mode = "survey"

        reviewer_cb: ReviewerCallback | None = None
        if self.reviewer:
            reviewer_cb = ReviewerCallback(
                reviewer=self.reviewer,
                task_description=task_description,
                trajectory=trajectory,
                step_callback=step_callback,
                checkpoints=self.config.reviewer.checkpoints,
                workspace_dir=self.config.sandbox.working_dir,
            )

        # Each run gets a unique thread so MemorySaver doesn't mix tasks.
        #
        # ``recursion_limit`` is the hard wall LangGraph imposes on the
        # number of (think → tool → observe) micro-steps it will execute.
        # A single visible "Step N" in the UI typically consumes 3-4
        # recursion units (one for the AI message, one per tool call, one
        # for the tool response). ``max_steps * 2`` was therefore cutting
        # the agent off mid-plan around step ~30 even when the user asked
        # for 50; bumping to ``× 6`` plus a 200-floor gives the run room
        # to reach actual completion before LangGraph kills the graph.
        def _new_thread_config() -> dict:
            tid = f"{task_id}_{uuid.uuid4().hex[:8]}"
            return {
                "configurable": {"thread_id": tid},
                "recursion_limit": max(step_budget.hard_cap * 6, 200),
            }

        thread_config = _new_thread_config()
        thread_id = thread_config["configurable"]["thread_id"]

        # First round: send the task prompt
        current_input = {"messages": [HumanMessage(content=task_prompt)]}

        logger.info("Starting task: %s (thread: %s)", task_id, thread_id)

        # Adaptive injection rounds prevent both premature fixed-round exits
        # and unbounded loops. The step budget may extend the soft budget when
        # recent work is productive, but the hard cap stays finite.
        max_rounds = step_budget.max_rounds
        nudged_rounds = 0
        max_nudges = step_budget.max_nudges
        # Narration-spiral guard (regression fix). When the LLM responds with
        # markdown code blocks in its reasoning text but never emits an actual
        # tool_call, the round produces zero non-plan trajectory steps. Pre-Mod-D
        # this was implicitly handled by the "if not injections: continuation
        # nudge" branch — but Mod D's checklist injection always populates
        # ``injections``, so the nudge stopped firing. We track narration-only
        # rounds explicitly here and prepend a strong format-violation message
        # ahead of every other injection. Counter resets when a real tool
        # call lands, so transient narration doesn't poison subsequent rounds.
        narration_only_rounds = 0
        # Quota self-stop counter. ``search_literature`` returns a clear
        # "quota exhausted" error when the budget is up, but the agent has
        # historically retried 10+ times anyway. We watch the trajectory for
        # consecutive quota-exhausted refusals and inject a hard-stop
        # instruction once we see two in a row, freeing the rest of the
        # step budget for productive work.
        quota_exhausted_consecutive = 0
        poison_resets = 0
        # Each AI Studio call has a ~5–10% chance of returning tool_calls.0.args
        # as a JSON string (instead of a dict), or producing other wire-format
        # errors that need a thread reset to clear. With the previous cap of 2
        # any 100-step run would die before finishing, even though every reset
        # preserves a progress summary and the agent recovers cleanly between
        # them. Bumped to 8 to absorb the realistic noise rate; the
        # consecutive-reset guard below catches the degenerate "reset without
        # making progress" case so we don't loop forever.
        max_poison_resets = 8
        # Trajectory length captured at the previous poison reset, so we can
        # detect "two resets in a row with no productive work between them" —
        # the only condition under which more resets cannot help.
        last_reset_step_count = -1
        consecutive_unproductive_resets = 0
        max_consecutive_unproductive_resets = 2
        # We don't cap submit blocks anymore — the previous "accept with caveats
        # at N=3" produced papers based on broken experiments. Instead we
        # escalate the coaching message so the agent learns to (a) treat
        # experiment-class feedback as a code-fix task rather than a writing
        # task, and (b) fall back to an honest negative-result framing when it
        # genuinely cannot fix the underlying issue. The hard step budget
        # remains the safety net for runs that can't converge.
        # NOTE (Phase-1 refactor): submit_blocks_used / checklist_blocks_used /
        # first_block_step_idx are no longer read by the submit path (the LLM
        # gates were removed — see the ``if self.submit_tool.submitted`` branch
        # below). Kept declared so any leftover diagnostic code can compile;
        # queued for deletion in Phase-2 alongside the helpers
        # ``_count_exec_attempts_after`` and ``_compose_submit_block_message``.
        submit_blocks_used = 0
        # Step index immediately after the first reviewer-block. Used as the
        # baseline for counting whether the agent has actually tried fixing
        # the bug (execute_code attempts) before the off-ramp to Path B
        # (honest negative-result framing) becomes mentionable.
        first_block_step_idx: int | None = None
        # Mod D: counter for checklist-gated blocks (separate from
        # reviewer-driven submit_blocks_used so we can distinguish
        # "executive function caught early submit" from "reviewer rejected
        # the work" in trajectory metadata + UI cards).
        checklist_blocks_used = 0

        def _is_poisoned_history_error(exc_obj) -> bool:
            text = str(exc_obj)
            # Provider-specific wire errors (the original cases).
            if "invalid function arguments" in text:
                return True
            if "invalid params" in text and "tool_call_id" in text:
                return True
            # LangChain/pydantic rejection when an OpenAI-compatible upstream
            # serializes ``tool_calls.*.args`` as a JSON string instead of an
            # object. The conversation is poisoned exactly the same way: the
            # AIMessage can't be replayed, so the next round will keep failing
            # until the thread is reset. Without this match we bleed steps on
            # runtime_error placeholders forever.
            if (
                "validation error" in text
                and "tool_calls" in text
                and ("valid dictionary" in text or "dict_type" in text)
            ):
                return True
            # Out-of-order tool/user messages produced by an earlier poisoned
            # turn ("Message format error, index[N] should be [tool] but is
            # [user]"). Same remedy: reset the thread.
            if "Message format error" in text and "should be [tool]" in text:
                return True
            return False

        def _summarize_progress() -> str:
            lines = []
            for st in trajectory.steps[-8:]:
                if st.action_name == "runtime_error":
                    continue
                obs = (st.observation or "").strip().replace("\n", " ")
                if len(obs) > 200:
                    obs = obs[:200] + "…"
                lines.append(f"- {st.action_name}: {obs or '[no output]'}")
            return "\n".join(lines) if lines else "(no successful tool calls yet)"

        for round_idx in range(max_rounds):
            try:
                budget_steps_before = budget_step_count(trajectory.steps)
                trajectory_len_before_round = len(trajectory.steps)
                self._stream_round(
                    input_messages=current_input,
                    thread_config=thread_config,
                    trajectory=trajectory,
                    phase_tracker=phase_tracker,
                    reviewer_cb=reviewer_cb,
                    step_callback=step_callback,
                    should_stop_after_step=lambda traj: step_budget.should_stop_stream_after_step(
                        traj.steps,
                        self.submit_tool.submitted,
                    ),
                )
                budget_steps_after = budget_step_count(trajectory.steps)
                step_budget.record_round_progress(budget_steps_after - budget_steps_before)

                # Narration / quota counters update (P0 + P1).
                # ``new_steps`` are the trajectory steps this round produced.
                # We classify them: a "real tool step" is anything that
                # actually invoked the sandbox / external API — plan and
                # runtime_error don't count. If zero real tool steps landed,
                # the LLM almost certainly emitted prose-only output (the
                # narration-spiral failure mode) and we need to override
                # whatever else we'd inject with a format-violation reminder.
                new_steps = trajectory.steps[trajectory_len_before_round:]
                _NON_TOOL_ACTIONS = {
                    "plan", "runtime_error", "human_feedback",
                    "human_abort", "human_review", "checklist_block",
                    "reviewer_block_cap_reached",
                }
                real_tool_steps = [
                    s for s in new_steps
                    if (s.action_name or "") not in _NON_TOOL_ACTIONS
                ]
                if real_tool_steps:
                    narration_only_rounds = 0
                else:
                    narration_only_rounds += 1
                    logger.warning(
                        "Narration-only round %d (no real tool calls; %d "
                        "consecutive). LLM likely emitted code in plan text "
                        "instead of calling execute_code.",
                        round_idx + 1, narration_only_rounds,
                    )

                # search_literature quota refusals — observation contains the
                # "quota exhausted" string when the tool denied the call.
                quota_refusals_this_round = sum(
                    1 for s in new_steps
                    if (s.action_name or "") == "search_literature"
                    and not s.success
                    and "quota exhausted" in (s.observation or "").lower()
                )
                if quota_refusals_this_round > 0:
                    quota_exhausted_consecutive += quota_refusals_this_round
                else:
                    # Reset only when the round contained a productive search
                    # call (success), not just any non-search round — we don't
                    # want every plan-only round to clear the counter.
                    if any(
                        (s.action_name or "") == "search_literature" and s.success
                        for s in new_steps
                    ):
                        quota_exhausted_consecutive = 0

                # Hard stop on persistent narration spirals. We've already
                # injected a format-violation nudge for ``narration_only_rounds
                # >= 1``; if four straight rounds still produce zero tool
                # calls, the LLM is genuinely stuck and burning the rest of
                # the step budget on useless prose generation. End the run
                # here so the user can debug rather than wait for max_rounds.
                if narration_only_rounds >= 4:
                    logger.error(
                        "Narration spiral persisted %d rounds (no tool calls); "
                        "aborting run. The LLM is emitting code in reasoning "
                        "text without ever wrapping it as a tool_call.",
                        narration_only_rounds,
                    )
                    trajectory.add_step(
                        thought="[SYSTEM]",
                        action_name="narration_spiral_abort",
                        action_args={"narration_only_rounds": narration_only_rounds},
                        observation=(
                            f"Run aborted: {narration_only_rounds} consecutive "
                            "rounds produced zero tool calls. The agent kept "
                            "writing markdown code blocks in reasoning text "
                            "instead of invoking execute_code. This is an LLM "
                            "format failure that the format-violation nudge "
                            "could not break. Recommended fix: switch to a "
                            "model with stronger function-calling reliability "
                            "(e.g. larger reasoning model), or shorten the "
                            "task so the prompt history stays smaller."
                        ),
                        success=False,
                        metadata={
                            "source": "narration_spiral_guard",
                            "narration_only_rounds": narration_only_rounds,
                        },
                    )
                    break
            except Exception as exc:
                # Don't abort the whole run on a single parse/tool error —
                # feed the error back and give the agent a chance to recover.
                logger.warning("Stream round %d failed: %s", round_idx, exc)
                trajectory.add_step(
                    thought="[SYSTEM]",
                    action_name="runtime_error",
                    action_args={},
                    observation=f"Previous turn failed: {exc}. Try again with a different approach.",
                    success=False,
                    metadata={"source": "exception"},
                )

                if _is_poisoned_history_error(exc):
                    # Decide whether to spend another reset. Two ceilings:
                    #   1. Hard cap: max_poison_resets (absorb realistic noise).
                    #   2. Consecutive unproductive resets: if the last reset
                    #      was followed by zero successful tool calls before
                    #      this poison error, the agent is genuinely stuck and
                    #      more resets won't help.
                    productive_since_last_reset = (
                        last_reset_step_count == -1
                        or any(
                            s.success
                            and s.action_name not in ("runtime_error", "plan", "human_feedback")
                            for s in trajectory.steps[last_reset_step_count:]
                        )
                    )
                    if productive_since_last_reset:
                        consecutive_unproductive_resets = 0
                    else:
                        consecutive_unproductive_resets += 1

                    if poison_resets >= max_poison_resets:
                        logger.warning(
                            "Poisoned history persists after %d reset(s) (cap reached); aborting task loop",
                            poison_resets,
                        )
                        break
                    if consecutive_unproductive_resets >= max_consecutive_unproductive_resets:
                        logger.warning(
                            "Poisoned history with %d consecutive unproductive resets; aborting task loop",
                            consecutive_unproductive_resets,
                        )
                        break

                    poison_resets += 1
                    last_reset_step_count = len(trajectory.steps)
                    progress = _summarize_progress()
                    thread_config = _new_thread_config()
                    thread_id = thread_config["configurable"]["thread_id"]
                    logger.info(
                        "Poisoned history detected; resetting thread to %s (reset %d/%d, "
                        "consecutive_unproductive=%d/%d)",
                        thread_id, poison_resets, max_poison_resets,
                        consecutive_unproductive_resets, max_consecutive_unproductive_resets,
                    )
                    reset_prompt = (
                        f"{task_prompt}\n\n"
                        "Previously you made progress but one of your tool calls had "
                        "malformed JSON arguments and was rejected by the API. "
                        "Continue the task from scratch, but here's what was already "
                        "verified via earlier successful tool calls:\n"
                        f"{progress}\n\n"
                        "Reuse the findings above when they help and make sure every "
                        "tool call has valid JSON arguments."
                    )
                    current_input = {"messages": [HumanMessage(content=reset_prompt)]}
                    continue

                healing_msgs = self._heal_pending_tool_calls(
                    thread_config,
                    error_note=(
                        "The previous tool call had malformed arguments and was rejected. "
                        "Reformulate the call with valid JSON arguments."
                    ),
                )
                nudge = HumanMessage(content=(
                    f"The previous turn raised an error: {exc}. "
                    "Ignore that attempt and continue the task using the remaining tools."
                ))
                current_input = {"messages": healing_msgs + [nudge]}
                continue

            # ReviewerCallback now emits its cards live (both "thinking" and
            # final verdict) as reviews happen, so there is no per-round drain
            # to run here. We still consult pending_feedback + last_review_failed
            # below to decide whether to force another round.

            if self.submit_tool.submitted:
                # Phase-1 refactor (REFACTOR_PLAN.md §3 Plan A): submit_result
                # is a single-shot terminal call. No LLM checklist re-eval,
                # no reviewer veto, no escalation message — the same design
                # as SWE-agent's `submit` and PaperQA2's `complete`. The LLM
                # eval gate that used to live here was the death-loop source
                # in trajectory 72d6ad6d (12 checklist_blocks, never escaped)
                # and c03d85d8 (6 checklist_blocks → narration spiral abort).
                #
                # Reviewer feedback is still useful, but only as a quality
                # signal recorded on the trajectory, not as a hard gate.
                # If the reviewer flagged issues at the report or submit
                # checkpoint, we tag the run as a degraded success so the
                # caller / UI can surface the warning to the user.
                if (
                    reviewer_cb
                    and reviewer_cb.last_review_failed
                    and reviewer_cb.pending_feedback
                ):
                    feedback_text = reviewer_cb.pending_feedback
                    feedback_class = _classify_reviewer_feedback(feedback_text)
                    trajectory.metadata["reviewer_warning_at_submit"] = {
                        "feedback": feedback_text[:4000],
                        "class": feedback_class,
                    }
                    logger.info(
                        "Submit accepted with reviewer warning (class=%s) — "
                        "recorded as metadata, no block.",
                        feedback_class,
                    )
                    # Drain the reviewer state so a stale warning doesn't
                    # leak into the next run if this agent instance is reused.
                    reviewer_cb.pending_feedback = None
                    reviewer_cb.last_review_failed = False

                logger.info("Task submitted at step %d", len(trajectory.steps))
                break

            # Collect feedback to inject for the next round
            injections: list[str] = []

            # -2. Format-violation guard (P0). When the LLM emitted prose code
            # blocks instead of calling tools, we MUST get this reminder in
            # front of the next prompt — it's load-bearing for breaking the
            # narration spiral. Goes ahead of the checklist (which would
            # otherwise be the first injection and bury the format problem).
            if narration_only_rounds >= 1:
                format_msg = (
                    "⚠️ FORMAT VIOLATION DETECTED — your previous response "
                    "contained Python code in REASONING TEXT (markdown code "
                    "blocks like ```python ... ```) instead of calling the "
                    "execute_code tool. **Code in reasoning text is NEVER "
                    "executed by the sandbox.** Only code passed as the "
                    "`code` argument to the execute_code tool actually runs.\n\n"
                    f"This is round {narration_only_rounds} in a row with no "
                    "tool call — your trajectory has zero new evidence to "
                    "show for the time. To fix:\n"
                    "  1. Take ONE concrete code snippet you've been writing "
                    "     in your plan text.\n"
                    "  2. Pass it AS THE `code` ARGUMENT to the execute_code "
                    "     tool. Not in reasoning text. Not in a markdown "
                    "     fence. As the tool argument.\n"
                    "  3. Verify it ran by reading the tool's stdout in the "
                    "     observation.\n\n"
                    "If you keep narrating without tool calls, the run will "
                    "exhaust its step budget without producing anything. "
                    "Make the next response a tool call."
                )
                injections.append(format_msg)
                logger.warning(
                    "Injecting format-violation nudge (narration_only_rounds=%d)",
                    narration_only_rounds,
                )

            # -1.5. Quota self-stop (P1). After two consecutive
            # quota-exhausted denials, tell the agent the search budget is
            # gone and it should stop trying. This frees the rest of the
            # step budget for productive work (execute_code / report).
            if quota_exhausted_consecutive >= 2:
                injections.append(
                    "⚠️ search_literature QUOTA EXHAUSTED for this run "
                    f"({quota_exhausted_consecutive} consecutive denials). "
                    "Do NOT call search_literature again — the tool will keep "
                    "refusing and you're burning rounds. Work with the papers "
                    "already retrieved (they're saved in literature_cache.jsonl "
                    "and any read_paper_fulltext outputs). If you need more "
                    "literature coverage, document this as a limitation in "
                    "the report's `related_work` section instead."
                )
                logger.warning(
                    "Injecting quota self-stop (consecutive=%d)",
                    quota_exhausted_consecutive,
                )

            # -1. Topic anchor (replaces the old LLM-eval checklist gate).
            # Re-inject a deterministic, filesystem-verified reminder of:
            #   - the original topic (so the LLM doesn't drift after long
            #     reviewer feedback chains),
            #   - the parsed deliverables list (one-shot LLM parse at run
            #     start; never re-evaluated per round),
            #   - actual artifacts present on disk in the run workspace.
            # We inject every TOPIC_REMINDER_INTERVAL rounds (and at round 1)
            # so the reminder stays fresh without spamming.
            if (
                round_idx == 0
                or (round_idx + 1) % TOPIC_REMINDER_INTERVAL == 0
            ):
                topic_block = format_topic_reminder(
                    topic=original_topic,
                    checklist=task_checklist,
                    workspace_dir=self.config.sandbox.working_dir,
                    round_idx=round_idx + 1,
                    max_rounds=max_rounds,
                    mode=original_mode,
                )
                if topic_block:
                    injections.append(topic_block)

            # 0. Adaptive budget guardrail. Hitting the soft budget is not an
            # automatic failure: if the agent is still producing useful work,
            # extend the soft budget and inject a clear completion-focused
            # instruction. Only the hard cap / repeated stalls end the run.
            budget_extension = step_budget.maybe_extend(trajectory.steps)
            if budget_extension:
                injections.append(budget_extension)
                trajectory.metadata["step_budget"] = step_budget.to_dict()
                if phase_tracker is not None:
                    phase_tracker.max_steps = step_budget.current_budget
                logger.info(
                    "Adaptive step budget extended to %d/%d",
                    step_budget.current_budget,
                    step_budget.hard_cap,
                )
            elif step_budget.stop_reason:
                logger.info("Adaptive budget stopping run: %s", step_budget.stop_reason)
                break

            wrapup_hint = step_budget.maybe_wrapup_hint(
                trajectory.steps,
                self.config.sandbox.working_dir,
            )
            if wrapup_hint:
                injections.append(wrapup_hint)
                logger.info("Injecting adaptive wrap-up hint")

            # Reviewer feedback
            if reviewer_cb and reviewer_cb.pending_feedback:
                feedback = reviewer_cb.pending_feedback
                reviewer_cb.pending_feedback = None
                injections.append(feedback)
                logger.info("Injecting reviewer feedback: %s", feedback[:200])

            # 3. Phase hints
            if phase_tracker is not None:
                phase_tracker.max_steps = step_budget.current_budget
            hint = phase_tracker.get_phase_hint() if phase_tracker is not None else None
            if hint:
                injections.append(hint)
                logger.info("Injecting phase hint: %s", hint[:200])

            # 4. Continuation nudge — the model sometimes stops mid-task and
            # returns narrative text instead of another tool call. Give it a
            # bounded number of prompts to keep going before giving up.
            if not injections and nudged_rounds < max_nudges:
                nudged_rounds += 1
                injections.append(
                    "You have not submitted a final result yet. "
                    "If your last message asked the user to confirm a plan, "
                    "treat it as approved because this run is in automatic "
                    "execution mode unless an explicit human-review message "
                    "was injected. "
                    "Continue the task by calling the appropriate tool. "
                    "If the solution is ready, call `submit_result`. "
                    "If you hit an error earlier, fix it and proceed. "
                    "Do not reply with prose — only tool calls.\n\n"
                    f"Adaptive budget status: {budget_step_count(trajectory.steps)}/"
                    f"{step_budget.current_budget} soft steps, hard cap "
                    f"{step_budget.hard_cap}."
                )
                logger.info("Injecting continuation nudge (%d/%d)", nudged_rounds, max_nudges)

            if not injections:
                if step_budget.should_stop_for_stall(trajectory.steps):
                    logger.info("Adaptive budget stopping run: %s", step_budget.stop_reason)
                    break
                if recent_productive_progress(trajectory.steps):
                    injections.append(
                        "You are still making progress but have not submitted "
                        "the task. Continue from the latest verified artifacts. "
                        "Take the smallest remaining valid step toward completion: "
                        "finish missing experiments or analysis, call generate_report, "
                        "then call submit_result. Do not ask for confirmation."
                    )
                    logger.info("Injecting adaptive progress continuation")
                else:
                    break

            combined = "\n\n".join(injections)
            current_input = {"messages": [HumanMessage(content=combined)]}
            logger.info("Round %d: injecting feedback into conversation", round_idx + 2)

        # Finalize
        if self.submit_tool.submitted:
            trajectory.finish("success")
        else:
            trajectory.finish(step_budget.stop_reason or "max_steps_exceeded")

        trajectory.metadata["step_budget"] = step_budget.to_dict()

        if self.config.record_trajectory:
            path = trajectory.save(self.config.trajectory_dir)
            logger.info("Trajectory saved to %s", path)

        return trajectory

    # ------------------------------------------------------------------
    # Compatibility guardrail helpers
    # ------------------------------------------------------------------
    def _get_phase_hint(self, step_idx: int | None = None) -> str | None:
        """Compatibility wrapper around the old phase-hint behavior."""

        from .prompts import PHASE_HINTS

        lit_success = self._phase_actions.get("literature", 0)
        impl = self._phase_actions.get("implementation", 0)
        step = step_idx if step_idx is not None else 0

        if lit_success >= 2 and impl == 0 and step >= 3:
            return PHASE_HINTS["literature_to_planning"]
        if (
            lit_success >= 1
            and self._literature_attempts >= 2
            and self._literature_failures > 0
            and impl == 0
            and step >= 3
        ):
            return PHASE_HINTS["literature_partial_to_planning"]
        if (
            lit_success == 0
            and self._literature_attempts >= 2
            and impl == 0
            and step >= 3
        ):
            return PHASE_HINTS["literature_search_failed"]
        return None

    def _validate_generate_report_request(self, arguments: dict[str, Any]) -> ToolResult | None:
        """Validate a would-be generate_report call before paper writing.

        The active LangGraph tool path enforces this inside GenerateReportTool.
        This method is kept for tests and external callers using the previous
        agent API.
        """

        combined = self._flatten_text(arguments)
        issues: list[str] = []

        unsupported = self._unsupported_citation_surnames(combined)
        if unsupported:
            issues.append(
                "Report cites unsupported literature not present in successful "
                f"search results: {', '.join(sorted(unsupported))}."
            )

        findings = blocking_findings(
            validate_workspace_results(self.config.sandbox.working_dir)
        )
        if findings and not report_text_discloses_findings(combined, findings):
            if any(finding.source.endswith("convergence_rates.csv") for finding in findings):
                issues.append(
                    "convergence_rates.csv contains invalid convergence values. "
                    "Do not claim normal convergence until the experiment is fixed "
                    "or the invalid values are explicitly disclosed."
                )
            else:
                issues.append(format_findings(findings))

        if not issues:
            return None
        return ToolResult(output="\n".join(issues), success=False)

    @staticmethod
    def _flatten_text(value: Any) -> str:
        parts: list[str] = []

        def _walk(item: Any) -> None:
            if item is None:
                return
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                for child in item.values():
                    _walk(child)
            elif isinstance(item, (list, tuple, set)):
                for child in item:
                    _walk(child)
            else:
                parts.append(str(item))

        _walk(value)
        return "\n".join(parts)

    def _unsupported_citation_surnames(self, text: str) -> set[str]:
        if not self._literature_surnames:
            return set()
        supported = {name.casefold() for name in self._literature_surnames}
        candidates: set[str] = set()
        for match in re.finditer(r"\b([A-Z][A-Za-z][A-Za-z'-]{1,30})\s+(?:et\s+al\.|and)\b", text):
            surname = match.group(1)
            if surname.casefold() not in supported:
                candidates.add(surname)
        return candidates
