"""
Configuration for LabForge (LangChain edition).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .env_utils import load_project_env


def _get_first_env(*keys: str) -> str:
    """Return the first non-empty environment variable among ``keys``."""
    for key in keys:
        value = os.environ.get(key, "").strip()
        if value:
            return value
    return ""


def resolve_api_key_for_endpoint(base_url: str = "", model: str = "") -> str:
    """Pick the most likely API key for the current endpoint/model."""
    endpoint = (base_url or "").strip().lower()
    model_name = (model or "").strip().lower()



    if "aistudio.baidu.com" in endpoint:
        return _get_first_env("AI_STUDIO_API_KEY", "API_KEY")

    if any(token in endpoint for token in ("minimaxi.com", "minimax")) or "minimax" in model_name:
        return _get_first_env("minimax_API_KEY", "MINIMAX_API_KEY", "API_KEY")

    if "deepseek" in endpoint or "deepseek" in model_name:
        return _get_first_env("deepseek_API_KEY", "DEEPSEEK_API_KEY", "API_KEY")

    if model_name.startswith("ernie"):
        return _get_first_env("AI_STUDIO_API_KEY", "API_KEY")

    if "openai" in endpoint or model_name.startswith("gpt"):
        return _get_first_env("OPENAI_API_KEY", "API_KEY")

    return _get_first_env(
        "API_KEY",
        "OPENAI_API_KEY",
        "AI_STUDIO_API_KEY",
        "minimax_API_KEY",
        "MINIMAX_API_KEY",
        "deepseek_API_KEY",
        "DEEPSEEK_API_KEY",
    )






_FALLBACK_MODEL = "MiniMax-M2.7"
_FALLBACK_BASE_URL = "https://api.minimaxi.com/v1"


@dataclass
class ModelConfig:
    """Configuration for a single LLM endpoint (OpenAI-compatible).

    支持任意 OpenAI 兼容后端：MiniMax、星河社区 (AI Studio)、DeepSeek 等。

    解析优先级（``model`` / ``base_url``）：
      1. 显式传入的非空值（YAML、UI、Python 直接构造）
      2. 环境变量 ``API_BASE_URL`` / ``MODEL_NAME``（仅当 1 没提供时）
      3. ``_FALLBACK_*`` 兜底（仅当 1 + 2 都没提供）

    历史 bug：之前用「值等于硬编码默认就当作没设」做启发式判断，结果
    YAML 里把 base_url 显式写成 ``https://api.minimaxi.com/v1``（恰好等于
    默认值）就会被环境变量错误覆盖，导致请求带着 ``MiniMax-M2.7`` 模型名
    打到 AI Studio endpoint，回 ``暂不支持该模型``。
    现在用空串 sentinel：YAML 显式写值 → 非空 → 不会被 env 覆盖。
    """


    model: str = ""
    api_key: str = ""
    base_url: str = ""
    temperature: float = 0.0
    max_tokens: int = 8192
    top_p: float = 1.0
    timeout: int = 120

    def __post_init__(self):
        env_base = _get_first_env("API_BASE_URL", "OPENAI_BASE_URL")
        env_model = _get_first_env("MODEL_NAME", "LLM_MODEL")
        if not self.base_url:
            self.base_url = env_base or _FALLBACK_BASE_URL
        if not self.model:
            self.model = env_model or _FALLBACK_MODEL

        if not self.api_key:
            self.api_key = resolve_api_key_for_endpoint(
                base_url=self.base_url,
                model=self.model,
            )



LLMConfig = ModelConfig






@dataclass
class SandboxConfig:
    """Configuration for the code execution sandbox."""

    backend: str = "subprocess"
    timeout: int = 300
    max_output_length: int = 10000
    working_dir: str = "/tmp/lab_forge_workspace"
    python_executable: str = ""
    docker_image: str = "python:3.10-slim"






@dataclass
class ReviewerConfig:
    """Configuration for the LLM-based reviewer that checks for hallucinations."""

    enabled: bool = True
    model: ModelConfig = field(default_factory=lambda: ModelConfig(




        model="deepseek-v3",
        base_url="https://aistudio.baidu.com/llm/lmapi/v3",
        temperature=0.0,
        max_tokens=2048,
    ))

    checkpoints: list[str] = field(default_factory=lambda: [
        "after_literature",
        "after_experiments",
        "before_report",
        "before_submit",
    ])






@dataclass
class AgentConfig:
    """Top-level agent configuration."""

    agent_model: ModelConfig = field(default_factory=ModelConfig)


    llm: ModelConfig | None = None
    reviewer: ReviewerConfig = field(default_factory=ReviewerConfig)
    sandbox: SandboxConfig = field(default_factory=SandboxConfig)




    max_steps: int = 0
    record_trajectory: bool = True
    trajectory_dir: str = "./trajectories"
    ocr_enabled: bool = False
    verbose: bool = True




    search_quota: int = 0

    def __post_init__(self):
        if self.llm is not None:
            self.agent_model = self.llm

    @classmethod
    def from_yaml(cls, path: str | Path) -> "AgentConfig":
        """Load config from a YAML file. Supports both old and new formats."""
        config_path = Path(path).resolve()
        load_project_env(
            project_root=config_path.parent.parent,
            extra_search_dirs=[Path.cwd()],
        )

        with open(path, "r") as f:
            data = yaml.safe_load(f) or {}


        if "llm" in data and "agent_model" not in data:
            return cls._from_legacy_yaml(data)

        return cls._from_new_yaml(data)

    @classmethod
    def _from_new_yaml(cls, data: dict) -> "AgentConfig":
        """Parse the new YAML config format."""
        agent_model_data = data.get("agent_model", {})
        agent_model = ModelConfig(**agent_model_data)

        sandbox_data = data.get("sandbox", {})
        sandbox_config = SandboxConfig(**sandbox_data)

        reviewer_data = data.get("reviewer", {})
        reviewer_model_data = reviewer_data.pop("model", {})
        reviewer_model = ModelConfig(**reviewer_model_data) if reviewer_model_data else ModelConfig(
            model="deepseek-v3",
            base_url="https://aistudio.baidu.com/llm/lmapi/v3",
        )
        reviewer_config = ReviewerConfig(model=reviewer_model, **reviewer_data)

        top_keys = {"max_steps", "record_trajectory", "trajectory_dir", "verbose", "ocr_enabled"}
        top_data = {k: v for k, v in data.items() if k in top_keys}

        return cls(
            agent_model=agent_model,
            reviewer=reviewer_config,
            sandbox=sandbox_config,
            **top_data,
        )

    @classmethod
    def _from_legacy_yaml(cls, data: dict) -> "AgentConfig":
        """Parse the old YAML config format that used an 'llm' key."""
        llm_data = data.get("llm", {})
        agent_model = ModelConfig(**llm_data)

        sandbox_data = data.get("sandbox", {})
        sandbox_config = SandboxConfig(**sandbox_data)

        reviewer_model = ModelConfig(
            model="deepseek-v3",
            base_url="https://aistudio.baidu.com/llm/lmapi/v3",
        )
        reviewer_config = ReviewerConfig(enabled=False, model=reviewer_model)

        top_keys = {"max_steps", "record_trajectory", "trajectory_dir", "verbose", "ocr_enabled"}
        top_data = {k: v for k, v in data.items() if k in top_keys}

        return cls(
            agent_model=agent_model,
            reviewer=reviewer_config,
            sandbox=sandbox_config,
            **top_data,
        )
