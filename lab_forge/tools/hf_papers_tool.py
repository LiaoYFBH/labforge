"""
HuggingFace Papers code lookup tool.

Given an arXiv id (e.g. ``2506.09781`` or ``2506.09781v2``), this tool fetches
the matching HuggingFace Papers page and extracts the open-source assets that
the community has linked there:

  * GitHub repositories (paper authors' implementations, third-party reproductions)
  * HuggingFace model checkpoints
  * HuggingFace datasets
  * HuggingFace Spaces (live demos)

The tool is intentionally read-only and does NOT clone anything. The agent is
expected to call ``execute_bash("git clone ...")`` afterwards and adapt the
code to fit the run's EXECUTION ENVIRONMENT block (CPU vs GPU, dataset size,
epoch budget, ~2-hour total runtime cap). Letting the agent drive the clone
+ adaptation keeps its judgment in the loop instead of hiding behind a black
box.

Why HTML scraping rather than a JSON API: the public ``/api/papers/{id}``
endpoint returns paper metadata but does not include linked repos or HF
artefacts. The HTML page at ``/papers/{id}`` does, so we fetch and regex-extract.
"""

from __future__ import annotations

import logging
import re
from urllib.parse import urlparse

import requests
from requests.exceptions import (
    ChunkedEncodingError,
    ConnectionError as RequestsConnectionError,
    SSLError,
    Timeout,
)

from .base import Tool, ToolResult

logger = logging.getLogger(__name__)

HF_PAPERS_PAGE = "https://huggingface.co/papers/{arxiv_id}"
HF_PAPERS_API = "https://huggingface.co/api/papers/{arxiv_id}"

HTTP_HEADERS = {
    "User-Agent": "LabForge/0.1 (paper code lookup; contact: local-user)",
    "Accept": "text/html,application/json",
}

# arXiv id patterns we accept: 4-digit YYMM.NNNNN with optional vN suffix.
_ARXIV_ID_PATTERN = re.compile(r"^\d{4}\.\d{4,5}(?:v\d+)?$")
# Older-style ids like "cs.CV/0301001" or "math/0211159" — also acceptable.
# Subject suffix is 2 letters; we already lowercase the input above so the
# pattern matches against ``[a-z]{2}`` rather than ``[A-Z]{2}``.
_OLD_ARXIV_ID_PATTERN = re.compile(r"^[a-z\-]+(?:\.[a-z]{2})?/\d{7}(?:v\d+)?$")

# GitHub repo extraction: match owner/repo pairs but exclude `tree`, `blob`,
# and other non-repo paths so we end up with clone-friendly URLs.
_GITHUB_REPO_RE = re.compile(
    r"https?://github\.com/([A-Za-z0-9][A-Za-z0-9\-_.]*)/([A-Za-z0-9][A-Za-z0-9\-_.]*)(?=[\"'/?#\s]|$)"
)
_GITHUB_NON_REPO_OWNERS = {
    "topics", "search", "trending", "sponsors", "marketplace",
    "settings", "notifications", "explore",
}

# HF model / dataset / space references: parse anchor hrefs.
_HF_MODEL_RE = re.compile(r"https?://huggingface\.co/([A-Za-z0-9][A-Za-z0-9\-_.]+)/([A-Za-z0-9][A-Za-z0-9\-_.]+)(?=[\"'/?#\s]|$)")
# Reserved HF org-level paths that aren't model repos.
_HF_RESERVED_PATHS = {
    "papers", "datasets", "spaces", "blog", "docs", "models",
    "tasks", "pricing", "settings", "join", "login", "api",
    "huggingface", "transformers", "docs.huggingface.co",
}


def _normalize_arxiv_id(raw: str) -> str | None:
    """Canonicalize an arXiv id input. Accepts ``arxiv:`` prefix and full URLs."""
    text = (raw or "").strip()
    if not text:
        return None
    text = text.lower().removeprefix("arxiv:").strip()
    # Strip URL prefixes
    if "arxiv.org/abs/" in text:
        text = text.split("arxiv.org/abs/", 1)[1]
    if "arxiv.org/pdf/" in text:
        text = text.split("arxiv.org/pdf/", 1)[1].removesuffix(".pdf")
    if text.startswith("/"):
        text = text.lstrip("/")
    text = text.split("?", 1)[0].split("#", 1)[0].strip()
    if _ARXIV_ID_PATTERN.match(text) or _OLD_ARXIV_ID_PATTERN.match(text):
        return text
    return None


def _fetch_paper_metadata(arxiv_id: str) -> dict:
    """Fetch JSON metadata from /api/papers/{id}. Returns ``{}`` on failure."""
    try:
        resp = requests.get(
            HF_PAPERS_API.format(arxiv_id=arxiv_id),
            headers=HTTP_HEADERS,
            timeout=10,
        )
        if resp.status_code == 200:
            return resp.json() or {}
    except (
        SSLError,
        RequestsConnectionError,
        Timeout,
        ChunkedEncodingError,
        requests.exceptions.RequestException,
        ValueError,
    ) as exc:
        logger.debug("HF paper metadata fetch failed for %s: %s", arxiv_id, exc)
    return {}


def _fetch_paper_page_html(arxiv_id: str) -> tuple[str, int | None]:
    """Fetch the HTML page. Returns ``(html, status_code)`` or ``("", code)``."""
    try:
        resp = requests.get(
            HF_PAPERS_PAGE.format(arxiv_id=arxiv_id),
            headers=HTTP_HEADERS,
            timeout=15,
        )
        return resp.text or "", resp.status_code
    except (
        SSLError,
        RequestsConnectionError,
        Timeout,
        ChunkedEncodingError,
        requests.exceptions.RequestException,
    ) as exc:
        logger.debug("HF paper page fetch failed for %s: %s", arxiv_id, exc)
        return "", None


def _extract_github_repos(html: str) -> list[str]:
    """Extract distinct ``https://github.com/<owner>/<repo>`` URLs from the page."""
    seen: set[str] = set()
    repos: list[str] = []
    for match in _GITHUB_REPO_RE.finditer(html):
        owner, repo = match.group(1), match.group(2)
        if owner.lower() in _GITHUB_NON_REPO_OWNERS:
            continue
        # Strip trailing punctuation / .git suffix for canonical clone URL.
        repo = repo.removesuffix(".git")
        url = f"https://github.com/{owner}/{repo}"
        if url not in seen:
            seen.add(url)
            repos.append(url)
    return repos


def _extract_hf_artefacts(html: str) -> dict[str, list[str]]:
    """Extract HF models / datasets / spaces referenced on the page.

    HF page anchors look like ``href="/owner/repo"`` for models,
    ``href="/datasets/owner/repo"`` for datasets, ``href="/spaces/owner/repo"``
    for spaces. We parse anchor URLs from the page so we don't accidentally
    include CDN/asset links.
    """
    models: list[str] = []
    datasets: list[str] = []
    spaces: list[str] = []
    seen_models: set[str] = set()
    seen_datasets: set[str] = set()
    seen_spaces: set[str] = set()

    for href_match in re.finditer(r'href="(/[^"]+)"', html):
        path = href_match.group(1)
        # datasets and spaces paths are well-known prefixes
        if path.startswith("/datasets/"):
            tail = path[len("/datasets/"):].split("?", 1)[0].split("#", 1)[0]
            parts = tail.split("/")
            if len(parts) >= 2 and parts[0] and parts[1]:
                full = f"https://huggingface.co/datasets/{parts[0]}/{parts[1]}"
                if full not in seen_datasets:
                    seen_datasets.add(full)
                    datasets.append(full)
            continue
        if path.startswith("/spaces/"):
            tail = path[len("/spaces/"):].split("?", 1)[0].split("#", 1)[0]
            parts = tail.split("/")
            if len(parts) >= 2 and parts[0] and parts[1]:
                full = f"https://huggingface.co/spaces/{parts[0]}/{parts[1]}"
                if full not in seen_spaces:
                    seen_spaces.add(full)
                    spaces.append(full)
            continue
        # Models: /owner/repo (no leading namespace prefix). Strict filter to
        # avoid catching docs / API / non-repo paths.
        parts = path.lstrip("/").split("?", 1)[0].split("#", 1)[0].split("/")
        if len(parts) == 2 and parts[0] and parts[1] and parts[0] not in _HF_RESERVED_PATHS:
            full = f"https://huggingface.co/{parts[0]}/{parts[1]}"
            if full not in seen_models:
                seen_models.add(full)
                models.append(full)

    # Cap each list to 10 to keep the tool output manageable.
    return {
        "models": models[:10],
        "datasets": datasets[:10],
        "spaces": spaces[:10],
    }


def _format_lookup_output(
    arxiv_id: str,
    title: str,
    repos: list[str],
    hf_artefacts: dict[str, list[str]],
) -> str:
    lines = [
        f"HuggingFace Papers lookup for arXiv:{arxiv_id}",
    ]
    if title:
        lines.append(f"  Title: {title}")
    lines.append(f"  Page: https://huggingface.co/papers/{arxiv_id}")
    lines.append("")

    if repos:
        lines.append(f"GitHub repositories ({len(repos)}):")
        for url in repos[:10]:
            lines.append(f"  - {url}")
        if len(repos) > 10:
            lines.append(f"  (+ {len(repos) - 10} more, omitted for brevity)")
    else:
        lines.append("GitHub repositories: none linked.")

    if hf_artefacts.get("models"):
        lines.append("")
        lines.append(f"HuggingFace models ({len(hf_artefacts['models'])}):")
        for url in hf_artefacts["models"]:
            lines.append(f"  - {url}")
    if hf_artefacts.get("datasets"):
        lines.append("")
        lines.append(f"HuggingFace datasets ({len(hf_artefacts['datasets'])}):")
        for url in hf_artefacts["datasets"]:
            lines.append(f"  - {url}")
    if hf_artefacts.get("spaces"):
        lines.append("")
        lines.append(f"HuggingFace spaces / demos ({len(hf_artefacts['spaces'])}):")
        for url in hf_artefacts["spaces"]:
            lines.append(f"  - {url}")

    if not (repos or any(hf_artefacts.values())):
        lines.append("")
        lines.append(
            "(no open-source code or HF artefacts linked on this page — the "
            "authors may not have released code, or the community hasn't "
            "claimed it on HF yet)."
        )

    lines.append("")
    lines.append(
        "Next step: if a repository looks promising, clone it with "
        "execute_bash, then ADAPT it to fit the EXECUTION ENVIRONMENT block "
        "in your task description (CPU vs GPU, dataset size, epoch count, "
        "<= 2 hours total runtime budget). Document the actual data volume / "
        "config you ended up running in the report."
    )
    return "\n".join(lines)


class LookupPaperCodeTool(Tool):
    """Find open-source code linked to a paper via HuggingFace Papers."""

    @property
    def name(self) -> str:
        return "lookup_paper_code"

    @property
    def description(self) -> str:
        return (
            "Look up open-source code, model weights, datasets, and live demos "
            "associated with an arXiv paper via HuggingFace Papers. Returns the "
            "linked GitHub repositories plus any HF models / datasets / spaces "
            "the community has registered for the paper. Use this AFTER picking "
            "a key reference paper (search_literature + read_paper_fulltext) and "
            "BEFORE writing experimental code from scratch — if a working "
            "reference implementation exists, clone+adapt it to your machine "
            "(see EXECUTION ENVIRONMENT in the task description) instead of "
            "rebuilding from scratch. Input: the arXiv id (e.g. '2506.09781' or "
            "'2506.09781v2'). The tool itself does not clone — call execute_bash "
            "with 'git clone ...' on a returned URL when you decide to use it."
        )

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "arxiv_id": {
                    "type": "string",
                    "description": (
                        "arXiv identifier of the paper, e.g. '2506.09781', "
                        "'2506.09781v2', or 'cs.CV/0301001'. The full arxiv URL "
                        "is also accepted; the tool extracts the id."
                    ),
                },
            },
            "required": ["arxiv_id"],
        }

    def execute(self, arxiv_id: str) -> ToolResult:
        normalized = _normalize_arxiv_id(arxiv_id)
        if not normalized:
            return ToolResult(
                output=(
                    f"Invalid arXiv id: {arxiv_id!r}. Expected something like "
                    "'2506.09781', '2506.09781v2', or an arxiv URL."
                ),
                success=False,
            )

        meta = _fetch_paper_metadata(normalized)
        title = (meta.get("title") or "").strip()

        html, status = _fetch_paper_page_html(normalized)
        if not html:
            return ToolResult(
                output=(
                    f"Could not fetch HuggingFace Papers page for arXiv:"
                    f"{normalized} (status={status}). The paper may not be "
                    "indexed on HF Papers yet, or the page is temporarily "
                    "unreachable. Proceed by writing code from scratch, or "
                    "try again later."
                ),
                success=False,
                metadata={"arxiv_id": normalized, "status": status},
            )

        if status == 404:
            return ToolResult(
                output=(
                    f"arXiv:{normalized} is not indexed on HuggingFace Papers. "
                    "Either the community hasn't added it yet, or the id is "
                    "wrong. Write code from scratch (or search the paper "
                    "authors' GitHub manually via execute_bash + curl)."
                ),
                success=False,
                metadata={"arxiv_id": normalized, "status": 404},
            )

        repos = _extract_github_repos(html)
        hf_artefacts = _extract_hf_artefacts(html)

        return ToolResult(
            output=_format_lookup_output(normalized, title, repos, hf_artefacts),
            success=True,
            metadata={
                "arxiv_id": normalized,
                "github_repos_found": len(repos),
                "hf_models_found": len(hf_artefacts.get("models", [])),
                "hf_datasets_found": len(hf_artefacts.get("datasets", [])),
                "hf_spaces_found": len(hf_artefacts.get("spaces", [])),
            },
        )
