"""
Trajectory recording for agent runs.

Each trajectory records the full history of an agent run:
  - Task description
  - Each step: (thought, action, observation, metadata)
  - Final outcome (success/failure)
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Step:
    """A single step in the agent trajectory."""

    step_index: int
    thought: str  # agent's reasoning
    action_name: str  # tool name
    action_args: dict[str, Any]  # tool arguments
    observation: str  # tool output
    success: bool  # whether the tool call succeeded
    timestamp: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def phase(self) -> str:
        """Infer the research phase from the action (Level 0 step type)."""
        phase_map = {
            "plan": "planning",
            "reviewer_review": "reviewer_review",
            "search_literature": "literature_review",
            "read_paper_fulltext": "source_reading",
            "execute_code": "implementation",
            "execute_bash": "implementation",
            "file_write": "implementation",
            "file_read": "analysis",
            "list_files": "analysis",
            "generate_report": "report",
            "submit_result": "submission",
        }
        return phase_map.get(self.action_name, "unknown")


@dataclass
class Trajectory:
    """Full trajectory of an agent run on a single task."""

    task_id: str
    task_description: str
    steps: list[Step] = field(default_factory=list)
    outcome: str = "incomplete"  # "success", "failure", "incomplete"
    total_tokens: int = 0
    start_time: float = field(default_factory=time.time)
    end_time: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_step(
        self,
        thought: str,
        action_name: str,
        action_args: dict,
        observation: str,
        success: bool,
        metadata: dict | None = None,
    ) -> Step:
        step = Step(
            step_index=len(self.steps),
            thought=thought,
            action_name=action_name,
            action_args=action_args,
            observation=observation,
            success=success,
            metadata=metadata or {},
        )
        self.steps.append(step)
        return step

    def finish(self, outcome: str):
        self.outcome = outcome
        self.end_time = time.time()

    def save(self, directory: str | Path):
        """Save trajectory to a JSON file."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{self.task_id}.json"
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2, ensure_ascii=False)
        return path

    @classmethod
    def load(cls, path: str | Path) -> "Trajectory":
        """Load trajectory from a JSON file, ignoring unknown legacy fields."""
        with open(path, "r") as f:
            data = json.load(f)
        step_fields = {f.name for f in Step.__dataclass_fields__.values()}
        raw_steps = data.pop("steps", [])
        steps = [Step(**{k: v for k, v in s.items() if k in step_fields}) for s in raw_steps]
        traj_fields = {f.name for f in cls.__dataclass_fields__.values()}
        traj = cls(**{k: v for k, v in data.items() if k in traj_fields})
        traj.steps = steps
        return traj
