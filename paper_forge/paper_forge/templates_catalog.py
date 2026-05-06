"""
顶会论文 LaTeX 模板目录。

每个 :class:`PaperTemplate` 对应一个 Jinja2 模板文件 (``*.tex.j2``)，并附带
- ``label`` / ``description``：UI 上展示用
- ``required_packages``：编译需要的常见 LaTeX 包
- ``download_urls``：(filename, url) 元组列表，用于在编译前从网络拉取该会议
  官方提供的 ``.sty`` / ``.bst`` / ``.cls`` 文件，确保文档真正使用对应会议的
  排版风格。下载是惰性的：第一次选定模板编译时才会触发，文件会缓存到
  ``templates/_assets`` 目录，后续直接复用。

模板写好之后，渲染时只是把结构化论文 dict 灌进 Jinja2 模板，完整的
LaTeX 源会写入工作目录与下载好的 .sty/.cls 一起被 ``xelatex`` 编译。
"""

from __future__ import annotations

import logging
import shutil
import urllib.request
import urllib.error
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

TEMPLATES_DIR = Path(__file__).resolve().parent.parent / "templates"
ASSETS_DIR = TEMPLATES_DIR / "_assets"


@dataclass
class PaperTemplate:
    """One LaTeX template option offered to the user."""

    key: str
    label: str
    description: str
    template_filename: str
    download_urls: list[tuple[str, str]] = field(default_factory=list)


# Conference template registry.
#
# The download_urls point at the official assets each conference distributes.
# At compile time we copy the assets next to the rendered .tex so the local
# TexLive can find them. If a download fails we still try to compile because
# many of these files (IEEEtran, acmart) are also bundled with TexLive.
# All templates are self-contained: they only rely on standard TexLive
# packages (article.cls, IEEEtran.cls, acmart.cls, multicol, geometry, …).
# No runtime downloads — that path was unreliable behind GFW / proxies and
# left users with broken PDFs when CTAN/conference mirrors timed out.
# ``download_urls`` remains in the dataclass for forward-compatibility (you
# can still drop a real conference .sty into ``templates/_assets/`` manually
# if you want pixel-perfect formatting), but the catalog ships empty lists.
PAPER_TEMPLATES: dict[str, PaperTemplate] = {
    "general_article": PaperTemplate(
        key="general_article",
        label="通用 article (默认 · 最稳)",
        description="标准 article 类单栏排版。任何 TexLive 都能直接编译。",
        template_filename="article.tex.j2",
    ),
    "ieee_conference": PaperTemplate(
        key="ieee_conference",
        label="IEEE 会议 (CVPR / ICCV / ICRA)",
        description=(
            "IEEEtran 双栏会议风格 — 适合 CVPR / ICCV / ICRA / INFOCOM 等。"
            "依赖 ``IEEEtran.cls``（TexLive 包 ``texlive-publishers`` 自带）。"
        ),
        template_filename="ieee_conference.tex.j2",
    ),
    "neurips_2024": PaperTemplate(
        key="neurips_2024",
        label="NeurIPS 风格 (单栏机器学习)",
        description="NeurIPS / ICLR 类单栏排版，Times 字体。无外部依赖，TexLive 即可编译。",
        template_filename="neurips_2024.tex.j2",
    ),
    "icml_2024": PaperTemplate(
        key="icml_2024",
        label="ICML 风格 (双栏机器学习)",
        description="ICML 类双栏排版，Times 字体。无外部依赖，TexLive 即可编译。",
        template_filename="icml_2024.tex.j2",
    ),
    "acl_2023": PaperTemplate(
        key="acl_2023",
        label="ACL / EMNLP 风格 (NLP 双栏)",
        description="ACL / EMNLP / NAACL 类双栏 NLP 论文排版。无外部依赖，TexLive 即可编译。",
        template_filename="acl_2023.tex.j2",
    ),
    "acm_sigconf": PaperTemplate(
        key="acm_sigconf",
        label="ACM SIGCONF (KDD / WWW / SIGIR)",
        description=(
            "ACM acmart sigconf 双栏风格 — 适合 KDD / WWW / SIGIR / CIKM 等。"
            "依赖 ``acmart.cls``（TexLive 包 ``texlive-publishers`` 自带）。"
        ),
        template_filename="acm_sigconf.tex.j2",
    ),
}


def list_templates() -> list[PaperTemplate]:
    return list(PAPER_TEMPLATES.values())


def get_template(key: str) -> PaperTemplate:
    if key not in PAPER_TEMPLATES:
        raise KeyError(f"Unknown paper template: {key}")
    return PAPER_TEMPLATES[key]


def template_choices() -> list[tuple[str, str]]:
    """Return a list of (label, key) tuples suitable for Gradio Dropdown."""
    return [(tpl.label, tpl.key) for tpl in PAPER_TEMPLATES.values()]


def _download_one(url: str, dest: Path, timeout: int = 30) -> bool:
    """Download a file. Returns True on success, False on failure (logs and moves on)."""
    if dest.exists() and dest.stat().st_size > 0:
        logger.debug("Reusing cached asset %s", dest)
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        logger.info("Downloading %s -> %s", url, dest.name)
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "PaperForge/0.2 (+https://github.com)"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = resp.read()
        if not data:
            logger.warning("Empty payload from %s", url)
            return False
        dest.write_bytes(data)
        return True
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as exc:
        logger.warning("Failed to download %s: %s", url, exc)
        return False


def ensure_assets(template: PaperTemplate, output_dir: Path) -> dict[str, bool]:
    """Make sure ``template``'s download_urls are resolved and copied next to the .tex.

    Returns a mapping of filename → True/False (downloaded successfully).
    Files that fail to download are silently skipped — TexLive often has
    them bundled, so the compile may still succeed.
    """
    if not template.download_urls:
        return {}

    output_dir.mkdir(parents=True, exist_ok=True)
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    results: dict[str, bool] = {}

    for filename, url in template.download_urls:
        cached = ASSETS_DIR / filename
        ok = _download_one(url, cached)
        results[filename] = ok
        if ok:
            try:
                shutil.copy2(cached, output_dir / filename)
            except OSError as exc:
                logger.warning("Failed to copy %s into %s: %s", filename, output_dir, exc)
                results[filename] = False
    return results
