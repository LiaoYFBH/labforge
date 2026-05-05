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
