"""
Main entry point and CLI for LabForge (auto mode only).

The CLI runs the agent end-to-end on a single task. The Gradio Studio
(``ui.py``) is the other supported entry point. Interactive terminal
mode and benchmark evaluation have been removed to keep the surface small.

Usage:
    # One-shot run
    python -m lab_forge.main run --config configs/default.yaml \
        --topic "compare RF/SVM/XGB on iris"
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import uuid
from pathlib import Path

from .agent import ResearchAgent
from .config import AgentConfig
from .workflow import (
    DEFAULT_SANDBOX_ROOT,
    format_workspace_summary,
    prepare_run_sandbox,
    topic_to_task,
)


_MODE_LABELS = {
    "experiment": "EXPERIMENT (轻量实验：搜文献 + 跑代码 + 出表 + 写报告)",
    "survey": "SURVEY (文献调研：只搜 + 读 + 写综述，不跑实验)",
}


def _format_mode_banner(mode: str) -> str:
    """Return a one-line banner showing which workflow the user picked."""
    return f"Mode        : {_MODE_LABELS.get(mode, mode)}"


def setup_logging(verbose: bool = True):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def _print_step(step_info: dict) -> None:
    """Print a single agent step to the terminal.

    Surfaces the persisted script / log paths for execute_code / execute_bash
    so the user can always open the exact source that produced the outputs.
    Reviewer cards render with a distinct badge so the reviewer's role is
    visible in the terminal just like it is in the UI.
    """
    idx = step_info.get("step_index", "?")
    tool = step_info.get("action_name", "")
    ok = "OK" if step_info.get("success") else "FAIL"
    thought = (step_info.get("thought") or "").strip()
    obs = (step_info.get("observation") or "").strip()
    action_args = step_info.get("action_args") or {}
    metadata = step_info.get("metadata") or {}

    if tool == "reviewer_review":
        if metadata.get("phase") == "thinking":
            header = f"--- Step {idx} [reviewer] THINKING ---"
        else:
            badge = "PASS" if step_info.get("success") else "FAIL"
            header = f"--- Step {idx} [reviewer] {badge} ---"
        print(f"\n{header}")
        if thought:
            print(f"  {thought[:400]}")
        if obs:
            for line in obs.splitlines()[:12]:
                print(f"  {line}")
        return

    if tool == "plan":
        print(f"\n--- Step {idx} [plan] ---")
        if thought:
            for line in thought.splitlines()[:12]:
                print(f"  {line}")
        return

    print(f"\n--- Step {idx} [{tool}] {ok} ---")
    if thought:
        print(f"  Thought: {thought[:200]}")

    if tool == "execute_code":
        code = str(action_args.get("code", "")).strip()
        if code:
            preview = code if len(code) <= 400 else code[:400] + "\n    ..."
            print("  Code:")
            for line in preview.splitlines():
                print(f"    {line}")
    elif tool == "execute_bash":
        command = str(action_args.get("command", "")).strip()
        if command:
            print(f"  Command: {command[:300]}")
    elif tool == "file_write":
        path = action_args.get("path") or action_args.get("filename") or ""
        if path:
            print(f"  Writing: {path}")

    if obs:
        preview = obs if len(obs) <= 600 else obs[:600] + "\n    ..."
        print("  Output:")
        for line in preview.splitlines():
            print(f"    {line}")


def _override_sandbox_dir(config: AgentConfig, run_dir: Path) -> None:
    """Force the agent to use the per-run sandbox instead of the config default."""
    config.sandbox.working_dir = str(run_dir)


def _print_run_summary(trajectory, run_dir: Path) -> None:
    """Print the final summary with outcome, report, and workspace files."""
    print("\n" + "=" * 70)
    print(f"Task outcome : {trajectory.outcome}")
    print(f"Total steps  : {len(trajectory.steps)}")

    if trajectory.outcome == "success":
        for step in reversed(trajectory.steps):
            if step.action_name == "submit_result":
                summary = (step.observation or "").strip()
                if summary:
                    print("\nFinal summary:")
                    for line in summary.splitlines()[:20]:
                        print(f"  {line}")
                break

    report_path = run_dir / "research_report.md"
    bundle_path = run_dir / "paperforge_bundle.json"
    if report_path.exists():
        print(f"\nReport markdown: {report_path}")
    if bundle_path.exists():
        print(f"PaperForge bundle: {bundle_path}")

    print("\n" + format_workspace_summary(run_dir))
    print("=" * 70)


def _build_task_from_args(args) -> tuple[str, str, str]:
    """Return (task_description, expected_output, data_description)."""
    if args.task_file:
        with open(args.task_file) as f:
            task_data = json.load(f)
        return (
            task_data.get("description", ""),
            task_data.get("expected_output", ""),
            task_data.get("data_description", ""),
        )

    if args.topic:
        return topic_to_task(args.topic, args.detail or "", mode=args.mode)

    return (
        args.task or "",
        args.expected_output or "",
        args.data_description or "",
    )


def cmd_run(args):
    """Run the agent on a research task in auto mode."""
    config = AgentConfig.from_yaml(args.config)
    setup_logging(config.verbose)

    task_id = args.task_id or str(uuid.uuid4())[:8]
    run_dir = prepare_run_sandbox(task_id, DEFAULT_SANDBOX_ROOT)
    _override_sandbox_dir(config, run_dir)

    task_desc, expected_output, data_desc = _build_task_from_args(args)
    if not task_desc:
        print("Error: task description is required (--topic, --task, or --task-file)")
        sys.exit(1)

    print("=" * 70)
    print(f"Task id     : {task_id}")
    print(f"Workspace   : {run_dir}")
    print(f"Config      : {args.config}")
    print(f"Agent model : {config.agent_model.model}")
    print(f"Reviewer    : "
          + (f"{config.reviewer.model.model} "
             f"(checkpoints={','.join(config.reviewer.checkpoints)})"
             if config.reviewer.enabled else "disabled"))
    if args.topic:
        print(_format_mode_banner(args.mode))
    print("=" * 70)

    agent = ResearchAgent(config)
    trajectory = agent.run(
        task_id=task_id,
        task_description=task_desc,
        expected_output=expected_output,
        data_description=data_desc,
        step_callback=_print_step,
    )

    _print_run_summary(trajectory, run_dir)


def main():
    parser = argparse.ArgumentParser(
        description="LabForge: LangChain-based Scientific Research Agent (auto mode)"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    run_parser = subparsers.add_parser("run", help="Run agent on a task (auto mode)")
    run_parser.add_argument("--config", required=True, help="Path to YAML config")
    run_parser.add_argument("--topic", type=str,
                            help="Research topic (auto-wrapped with the same 5-phase prompt the UI uses)")
    run_parser.add_argument("--detail", type=str, default="",
                            help="Optional additional requirements for the topic")
    run_parser.add_argument("--mode", choices=["survey", "experiment"], default="survey",
                            help="Pick workflow: 'survey' (literature only) or 'experiment' (run code). "
                                 "Defaults to 'survey' (the safer branch — never fabricates results).")
    run_parser.add_argument("--task", type=str,
                            help="Raw task description (bypasses topic wrapping)")
    run_parser.add_argument("--task-file", type=str, help="Path to task JSON file")
    run_parser.add_argument("--task-id", type=str, help="Task ID")
    run_parser.add_argument("--expected-output", type=str, help="Expected output description")
    run_parser.add_argument("--data-description", type=str, help="Available data description")

    args = parser.parse_args()

    if args.command == "run":
        cmd_run(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
