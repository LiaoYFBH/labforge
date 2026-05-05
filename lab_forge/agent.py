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
    "instabil",
    "unstable",
    "diverg",
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







_SCOPE_FEEDBACK_SIGNALS = (

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


    "对比不充分",
    "对比不足",
    "缺乏对比",
    "对比实验不充分",
    "对比实验不足",
    "全面对比",
    "完整的基线",
    "完整对比",
    "缺乏完整",

    "未达到目标",
    "未达到预期",
    "任务目标未完全完成",
    "目标未完全完成",
    "未完全完成任务",
    "未完全达标",
    "未达标",
    "部分完成但未",
    "任务目标部分完成",

    "未提出新方法",
    "未提出新算法",
    "未提出创新",
    "未提出.*创新",
    "未设计新",

    "未证明优越性",
    "未证明方法优越性",
    "未能证明",
    "did not prove",
    "did not demonstrate superiority",
    "did not outperform",
    "未超越基线",
    "未超过基线",

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

    "未验证真实",
    "缺乏对真实数据",
    "缺乏对真实世界",
    "未在真实数据集",

    "缺乏创新",
    "缺乏新颖",
    "no novelty",
    "lacks innovation",
    "lacks novelty",

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
        else:
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



        self._phase_actions: dict[str, int] = {
            "literature": 0,
            "implementation": 0,
            "analysis": 0,
            "report": 0,
        }
        self._literature_attempts = 0
        self._literature_failures = 0
        self._literature_surnames: set[str] = set()


        self.agent_llm = create_chat_model(config.agent_model)







        self.reviewer: Reviewer | None = None
        if config.reviewer.enabled:
            reviewer_llm = create_chat_model(config.reviewer.model)
            self.reviewer = Reviewer(reviewer_llm, fallback_llm=self.agent_llm)






        self.submit_tool = SubmitResultTool()
        self.tools, self.submit_tool = create_all_tools(
            sandbox=self.sandbox,
            working_dir=config.sandbox.working_dir,
            ocr_enabled=config.ocr_enabled,
            submit_tool=self.submit_tool,
            tool_names=tool_names,
            writer_llm=self.agent_llm,
            search_quota=int(getattr(config, "search_quota", 0) or 0),
        )


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




        pending_thought = ""



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




                            if not tool_calls and _step_limit_reached():
                                return "soft_step_budget_reached"


                            pending_thought = ""
                        else:


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








        task_checklist: TaskChecklist = parse_task_checklist(task_description, self.agent_llm)
        if task_checklist.items:
            trajectory.metadata["task_checklist"] = {
                "items": [
                    {"id": i.id, "title": i.title, "verification_hint": i.verification_hint}
                    for i in task_checklist.items
                ],
            }





        TOPIC_REMINDER_INTERVAL = 5






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
            )











        def _new_thread_config() -> dict:
            tid = f"{task_id}_{uuid.uuid4().hex[:8]}"
            return {
                "configurable": {"thread_id": tid},
                "recursion_limit": max(step_budget.hard_cap * 6, 200),
            }

        thread_config = _new_thread_config()
        thread_id = thread_config["configurable"]["thread_id"]


        current_input = {"messages": [HumanMessage(content=task_prompt)]}

        logger.info("Starting task: %s (thread: %s)", task_id, thread_id)




        max_rounds = step_budget.max_rounds
        nudged_rounds = 0
        max_nudges = step_budget.max_nudges









        narration_only_rounds = 0






        quota_exhausted_consecutive = 0
        poison_resets = 0








        max_poison_resets = 8



        last_reset_step_count = -1
        consecutive_unproductive_resets = 0
        max_consecutive_unproductive_resets = 2













        submit_blocks_used = 0




        first_block_step_idx: int | None = None




        checklist_blocks_used = 0

        def _is_poisoned_history_error(exc_obj) -> bool:
            text = str(exc_obj)

            if "invalid function arguments" in text:
                return True
            if "invalid params" in text and "tool_call_id" in text:
                return True






            if (
                "validation error" in text
                and "tool_calls" in text
                and ("valid dictionary" in text or "dict_type" in text)
            ):
                return True



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



                quota_refusals_this_round = sum(
                    1 for s in new_steps
                    if (s.action_name or "") == "search_literature"
                    and not s.success
                    and "quota exhausted" in (s.observation or "").lower()
                )
                if quota_refusals_this_round > 0:
                    quota_exhausted_consecutive += quota_refusals_this_round
                else:



                    if any(
                        (s.action_name or "") == "search_literature" and s.success
                        for s in new_steps
                    ):
                        quota_exhausted_consecutive = 0







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






            if self.submit_tool.submitted:













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


                    reviewer_cb.pending_feedback = None
                    reviewer_cb.last_review_failed = False

                logger.info("Task submitted at step %d", len(trajectory.steps))
                break


            injections: list[str] = []






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


            if reviewer_cb and reviewer_cb.pending_feedback:
                feedback = reviewer_cb.pending_feedback
                reviewer_cb.pending_feedback = None
                injections.append(feedback)
                logger.info("Injecting reviewer feedback: %s", feedback[:200])


            if phase_tracker is not None:
                phase_tracker.max_steps = step_budget.current_budget
            hint = phase_tracker.get_phase_hint() if phase_tracker is not None else None
            if hint:
                injections.append(hint)
                logger.info("Injecting phase hint: %s", hint[:200])




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


        if self.submit_tool.submitted:
            trajectory.finish("success")
        else:
            trajectory.finish(step_budget.stop_reason or "max_steps_exceeded")

        trajectory.metadata["step_budget"] = step_budget.to_dict()

        if self.config.record_trajectory:
            path = trajectory.save(self.config.trajectory_dir)
            logger.info("Trajectory saved to %s", path)

        return trajectory




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
