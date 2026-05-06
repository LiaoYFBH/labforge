"""Configuration dataclasses for PaperForge."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .env_utils import load_project_env


def _get_first_env(*keys: str) -> str:
    for key in keys:
        value = os.environ.get(key, "").strip()
        if value:
            return value
    return ""


@dataclass
class OCRConfig:
    api_url: str = ""
    token: str = ""
    use_doc_orientation_classify: bool = False
    use_doc_unwarping: bool = False
    use_chart_recognition: bool = False

    def __post_init__(self):
        if not self.api_url:
            self.api_url = _get_first_env(
                "PADDLEOCR_API_URL", "OCR_API_URL"
            )
        if not self.token:
            self.token = _get_first_env(
                "PADDLEOCR_TOKEN", "OCR_TOKEN"
            )


# Resolution order for LLMConfig fields:
#   1. Whatever the caller explicitly passed (string, including the default).
#   2. Environment variable (when nothing explicit was passed).
#   3. Hard-coded default below.
# We use ``None`` as the "unset" sentinel so an explicit ``model="deepseek-v3"``
# passed by the caller is NEVER overwritten by a stale ``MODEL_NAME`` env entry.
# (The previous "field == default-string" sentinel could not distinguish
# "caller passed the default explicitly" from "caller passed nothing", which is
# exactly how lab-forge's PaperForge integration was breaking: upstream passed
# ``deepseek-v3`` but ``MODEL_NAME=ernie-4.5-turbo-128k-preview`` in .env
# replaced it, causing AI Studio 40405 "暂不支持该模型".)
#
# 2026 年 AI Studio 把老的 ERNIE preview 系列从默认开放清单撤了下来，新付费
# 账户走旧模型会拿到 40405。default 改成 ``deepseek-v3`` —— 它是当下 AI Studio
# 现役免费/付费配额都开放的模型，独立 paper_forge 开箱即用，不再首发就 400。
_LLM_DEFAULT_MODEL = "deepseek-v3"
_LLM_DEFAULT_BASE_URL = "https://aistudio.baidu.com/llm/lmapi/v3"


@dataclass
class LLMConfig:
    model: str | None = None
    api_key: str = ""
    base_url: str | None = None
    temperature: float = 0.3
    max_tokens: int = 8192
    timeout: int = 120

    def __post_init__(self):
        if not self.api_key:
            self.api_key = _get_first_env(
                "AI_STUDIO_API_KEY", "API_KEY", "OPENAI_API_KEY"
            )
        if self.model is None:
            self.model = _get_first_env("MODEL_NAME", "LLM_MODEL") or _LLM_DEFAULT_MODEL
        if self.base_url is None:
            self.base_url = (
                _get_first_env("API_BASE_URL", "OPENAI_BASE_URL")
                or _LLM_DEFAULT_BASE_URL
            )


@dataclass
class PDFStyleConfig:
    page_size: str = "A4"
    font_size_body: int = 11
    font_size_title: int = 22
    font_size_section: int = 15
    font_size_subsection: int = 13
    margin_mm: int = 25
    two_column: bool = False
    line_spacing: float = 1.4
    include_page_numbers: bool = True


@dataclass
class AppConfig:
    ocr: OCRConfig = field(default_factory=OCRConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    pdf_style: PDFStyleConfig = field(default_factory=PDFStyleConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "AppConfig":
        config_path = Path(path).resolve()
        load_project_env(
            project_root=config_path.parent.parent,
            extra_search_dirs=[Path.cwd()],
        )

        with open(path, "r") as f:
            data = yaml.safe_load(f) or {}

        return cls(
            ocr=OCRConfig(**data.get("ocr", {})),
            llm=LLMConfig(**data.get("llm", {})),
            pdf_style=PDFStyleConfig(**data.get("pdf_style", {})),
        )
