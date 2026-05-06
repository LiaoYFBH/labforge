"""Persistence helpers for the LabForge web UI settings."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

from .config import resolve_api_key_for_endpoint

DEFAULT_AGENT_BASE_URL = "https://api.minimaxi.com/v1"
DEFAULT_AGENT_MODEL = "MiniMax-M2.7"
DEFAULT_REVIEWER_BASE_URL = "https://aistudio.baidu.com/llm/lmapi/v3"
# 2026 年 AI Studio 已把老的 ernie-3.5-8k / speed / lite / tiny 系列从
# 默认开放清单撤下，新账户走这些模型会拿到 invalid_model。改用 deepseek-v3
# 作为默认评审模型 —— 和 agent 主模型走同一个套餐，付费等级一致就不会
# 出现 "agent 能跑 reviewer 401" 的不对称失败。
DEFAULT_REVIEWER_MODEL = "deepseek-v3"
# Backward compat alias
DEFAULT_BASE_URL = DEFAULT_AGENT_BASE_URL
DEFAULT_OCR_URL = "https://j5j557k6rbo1c6f4.aistudio-app.com/layout-parsing"
DEFAULT_SETTINGS_PATH = Path(__file__).resolve().parent.parent / ".ui_settings.json"


def _get_first_env(*keys: str) -> str:
    for key in keys:
        value = os.environ.get(key, "").strip()
        if value:
            return value
    return ""


@dataclass
class UISettings:
    agent_model: str = DEFAULT_AGENT_MODEL
    agent_base_url: str = DEFAULT_AGENT_BASE_URL
    agent_api_key: str = ""

    reviewer_enabled: bool = True
    reviewer_share_credentials: bool = False
    reviewer_model: str = DEFAULT_REVIEWER_MODEL
    reviewer_base_url: str = DEFAULT_REVIEWER_BASE_URL
    reviewer_api_key: str = ""

    # ``0`` means no user-facing hard step cap. The previous 30-step default
    # was cutting agents off before experiments / PaperForge export finished.
    # Internally LangGraph still receives a large finite recursion limit to
    # prevent a genuinely infinite loop, but the UI no longer stops at 30.
    max_steps: int = 0
    temperature: float = 0.0

    ocr_enabled: bool = False
    ocr_api_url: str = DEFAULT_OCR_URL
    ocr_token: str = ""

    # Literature search uses arXiv. ``search_quota`` caps total
    # search_literature calls per run so reactive re-searching can't loop
    # forever — set to 0 to disable the cap. 8 covers a planned 4-6 query
    # batch with headroom.
    search_quota: int = 8

    # Workflow mode chosen by the user via the UI radio. Always one of
    # ``"survey"`` (literature-only) or ``"experiment"`` (runs code).
    # There is intentionally no "auto" — keyword-based auto-classification
    # used to mis-route prompts like "做实验分析参数的作用" to survey.
    run_mode: str = "survey"

    @classmethod
    def from_env_defaults(cls) -> "UISettings":
        env_agent_base = _get_first_env("API_BASE_URL", "OPENAI_BASE_URL") or DEFAULT_AGENT_BASE_URL
        env_model = _get_first_env("MODEL_NAME", "LLM_MODEL") or DEFAULT_AGENT_MODEL
        env_reviewer_model = _get_first_env("REVIEWER_MODEL") or DEFAULT_REVIEWER_MODEL
        env_reviewer_base = _get_first_env("REVIEWER_BASE_URL") or DEFAULT_REVIEWER_BASE_URL
        env_ocr_url = _get_first_env(
            "PADDLEOCR_API_URL",
            "OCR_API_URL",
            "PADDLEOCR_VL_API_URL",
        ) or DEFAULT_OCR_URL
        return cls(
            agent_model=env_model,
            agent_base_url=env_agent_base,
            agent_api_key=resolve_api_key_for_endpoint(
                base_url=env_agent_base,
                model=env_model,
            ),
            reviewer_model=env_reviewer_model,
            reviewer_base_url=env_reviewer_base,
            reviewer_api_key=resolve_api_key_for_endpoint(
                base_url=env_reviewer_base,
                model=env_reviewer_model,
            ),
            ocr_api_url=env_ocr_url,
            ocr_token=_get_first_env(
                "PADDLEOCR_TOKEN",
                "OCR_TOKEN",
                "PADDLEOCR_VL_TOKEN",
            ),
        )

    @classmethod
    def from_dict(cls, data: dict | None) -> "UISettings":
        base = cls.from_env_defaults()
        if not data:
            return base
        return cls(
            agent_model=str(data.get("agent_model", base.agent_model) or base.agent_model),
            agent_base_url=str(data.get("agent_base_url", base.agent_base_url) or base.agent_base_url),
            agent_api_key=str(data.get("agent_api_key", "")),
            reviewer_enabled=bool(data.get("reviewer_enabled", base.reviewer_enabled)),
            reviewer_share_credentials=bool(
                data.get("reviewer_share_credentials", base.reviewer_share_credentials)
            ),
            reviewer_model=str(
                data.get("reviewer_model", base.reviewer_model) or base.reviewer_model
            ),
            reviewer_base_url=str(
                data.get("reviewer_base_url", base.reviewer_base_url) or base.reviewer_base_url
            ),
            reviewer_api_key=str(data.get("reviewer_api_key", "")),
            max_steps=(
                0
                if int(data.get("max_steps", base.max_steps)) == 30
                else int(data.get("max_steps", base.max_steps))
            ),
            temperature=float(data.get("temperature", base.temperature)),
            ocr_enabled=bool(data.get("ocr_enabled", base.ocr_enabled)),
            ocr_api_url=str(data.get("ocr_api_url", base.ocr_api_url) or base.ocr_api_url),
            ocr_token=str(data.get("ocr_token", "")),
            search_quota=int(data.get("search_quota", base.search_quota) or 0),
            run_mode=(
                str(data.get("run_mode", base.run_mode))
                if str(data.get("run_mode", base.run_mode)) in ("survey", "experiment")
                else base.run_mode
            ),
        )

    def to_dict(self) -> dict:
        return asdict(self)


def load_ui_settings(path: str | Path = DEFAULT_SETTINGS_PATH) -> tuple[UISettings, str]:
    """Load saved UI settings and report whether they came from file, env, or defaults."""
    settings_path = Path(path)
    if settings_path.exists():
        try:
            payload = json.loads(settings_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            payload = {}
        return UISettings.from_dict(payload), "file"

    settings = UISettings.from_env_defaults()
    source = "env" if any(
        [
            settings.agent_api_key,
            settings.ocr_token,
            settings.agent_model != DEFAULT_AGENT_MODEL,
            settings.agent_base_url != DEFAULT_AGENT_BASE_URL,
        ]
    ) else "default"
    return settings, source


def save_ui_settings(
    settings: UISettings,
    path: str | Path = DEFAULT_SETTINGS_PATH,
) -> Path:
    """Persist UI settings locally as JSON."""
    settings_path = Path(path).resolve()
    settings_path.parent.mkdir(parents=True, exist_ok=True)
    settings_path.write_text(
        json.dumps(settings.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    try:
        os.chmod(settings_path, 0o600)
    except OSError:
        pass
    return settings_path


def mask_secret(secret: str) -> str:
    """Mask a credential for UI display."""
    value = (secret or "").strip()
    if not value:
        return ""
    if len(value) <= 8:
        return "********"
    return value[:4] + "********" + value[-4:]
