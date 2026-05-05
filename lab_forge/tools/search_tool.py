"""
Literature search tool. Uses the public arXiv API.

The tool persists every successful query to ``literature_cache.jsonl`` so
downstream guardrails (citation validation in ``generate_report``) can compare
against a single source-of-truth list of papers the agent actually saw this
run.
"""

from __future__ import annotations

import logging
import random
import re
import time
import xml.etree.ElementTree as ET

import requests
from requests import Session
from requests.exceptions import (
    ChunkedEncodingError,
    ConnectionError as RequestsConnectionError,
    ProxyError,
    SSLError,
    Timeout,
)

from .base import Tool, ToolResult

logger = logging.getLogger(__name__)

ARXIV_API = "https://export.arxiv.org/api/query"
HTTP_HEADERS = {
    "User-Agent": "LabForge/0.1 (literature search; contact: local-user)"
}





_RETRYABLE_EXCEPTIONS = (
    SSLError,
    RequestsConnectionError,
    Timeout,
    ChunkedEncodingError,
)
_RETRYABLE_STATUS = {429, 500, 502, 503, 504}

DEFAULT_MAX_ATTEMPTS = 4
INITIAL_BACKOFF_SECONDS = 1.5
MAX_BACKOFF_SECONDS = 15.0







RELEVANCE_THRESHOLD = 0.5
_TOKEN_RE = re.compile(r"[a-z0-9]{3,}")
_STOPWORDS = frozenset({
    "the", "and", "for", "with", "from", "that", "this", "into", "onto",
    "are", "was", "were", "but", "not", "any", "all", "via", "based",
    "approach", "method", "methods", "technique", "techniques", "study",
    "paper", "research", "comparison", "comparing", "evaluation", "review",
    "survey", "analysis", "model", "models", "task", "tasks",
})


def _query_tokens(query: str) -> set[str]:
    """Tokenize a query into stemmed content words for relevance scoring."""
    raw = _TOKEN_RE.findall(query.lower())
    tokens: set[str] = set()
    for tok in raw:
        if tok in _STOPWORDS:
            continue


        if len(tok) > 4 and tok.endswith("s"):
            tok = tok[:-1]
        tokens.add(tok)
    return tokens


def _relevance_score(q_tokens: set[str], paper: dict) -> float:
    """Fraction of distinct query tokens that appear in title+abstract."""
    if not q_tokens:
        return 1.0
    text = (
        f"{paper.get('title', '')} {paper.get('abstract', '')}"
    ).lower()
    hit = sum(1 for tok in q_tokens if tok in text)
    return hit / len(q_tokens)


def _filter_by_relevance(
    query: str, papers: list[dict]
) -> tuple[list[dict], list[tuple[str, float]]]:
    """Split ``papers`` into (kept, dropped_with_score) by relevance threshold."""
    q_tokens = _query_tokens(query)
    if not q_tokens:
        return papers, []
    kept: list[dict] = []
    dropped: list[tuple[str, float]] = []
    for paper in papers:
        score = _relevance_score(q_tokens, paper)
        if score >= RELEVANCE_THRESHOLD:
            kept.append(paper)
        else:
            dropped.append((paper.get("title", "") or "(untitled)", score))
    return kept, dropped


def _sleep_backoff(attempt: int, *, retry_after: float | None = None) -> float:
    """Exponential backoff with jitter; honours ``Retry-After`` when given."""
    if retry_after is not None and retry_after > 0:
        delay = min(retry_after, MAX_BACKOFF_SECONDS)
    else:
        delay = min(INITIAL_BACKOFF_SECONDS * (2 ** (attempt - 1)), MAX_BACKOFF_SECONDS)
        delay += random.uniform(0, 0.5)
    time.sleep(delay)
    return delay


def _parse_retry_after(resp: requests.Response) -> float | None:
    raw = resp.headers.get("Retry-After")
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def _do_http_get(url: str, *, timeout: int, params: dict | None) -> requests.Response:
    """Single GET attempt, with one fallback when env proxies misbehave."""
    try:
        return requests.get(url, params=params, timeout=timeout, headers=HTTP_HEADERS)
    except ProxyError as exc:
        logger.warning("Proxy failed for %s, retrying without env proxy: %s", url, exc)
        session = Session()
        session.trust_env = False
        return session.get(url, params=params, timeout=timeout, headers=HTTP_HEADERS)


def _http_get_with_retry(
    url: str,
    *,
    timeout: int = 30,
    params: dict | None = None,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    attempt_log: list[str] | None = None,
) -> requests.Response:
    """GET with retries on transient network errors and retryable HTTP status codes.

    Retries on SSL/connection/timeout errors and on 429/5xx responses, using
    exponential backoff with jitter. ``Retry-After`` is honoured when present.
    Raises the last exception (or ``HTTPError`` for retryable status codes
    that never recovered) once attempts are exhausted.
    """
    last_exc: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            resp = _do_http_get(url, timeout=timeout, params=params)
        except _RETRYABLE_EXCEPTIONS as exc:
            last_exc = exc
            note = f"attempt {attempt}/{max_attempts} {type(exc).__name__}: {exc}"
            logger.warning("arXiv transient error, %s", note)
            if attempt_log is not None:
                attempt_log.append(note)
            if attempt == max_attempts:
                raise
            _sleep_backoff(attempt)
            continue

        if resp.status_code in _RETRYABLE_STATUS:
            note = f"attempt {attempt}/{max_attempts} HTTP {resp.status_code}"
            logger.warning("arXiv retryable status, %s", note)
            if attempt_log is not None:
                attempt_log.append(note)
            if attempt == max_attempts:
                resp.raise_for_status()
                return resp
            _sleep_backoff(attempt, retry_after=_parse_retry_after(resp))
            continue

        return resp


    assert last_exc is not None
    raise last_exc


def _search_arxiv(query: str, limit: int, attempt_log: list[str] | None = None) -> list[dict]:
    """Search arXiv API. Returns list of paper dicts (possibly empty).

    Raises the underlying exception on non-recoverable failure so callers can
    surface a precise reason instead of a generic "API unavailable" message.
    """
    params = {
        "search_query": f"all:{query}",
        "start": 0,
        "max_results": limit,
        "sortBy": "relevance",
        "sortOrder": "descending",
    }
    resp = _http_get_with_retry(
        ARXIV_API, params=params, timeout=30, attempt_log=attempt_log
    )
    resp.raise_for_status()

    ns = {"atom": "http://www.w3.org/2005/Atom"}
    root = ET.fromstring(resp.text)
    entries = root.findall("atom:entry", ns)

    results: list[dict] = []
    for entry in entries:
        title = entry.findtext("atom:title", "", ns).strip().replace("\n", " ")
        summary = entry.findtext("atom:summary", "", ns).strip().replace("\n", " ")
        published = entry.findtext("atom:published", "", ns)[:4]
        link = entry.findtext("atom:id", "", ns).strip()
        pdf_link = ""
        for candidate in entry.findall("atom:link", ns):
            if candidate.attrib.get("type") == "application/pdf":
                pdf_link = candidate.attrib.get("href", "").strip()
                break
        if not pdf_link and "/abs/" in link:
            pdf_link = link.replace("/abs/", "/pdf/") + ".pdf"

        author_elements = entry.findall("atom:author/atom:name", ns)
        author_names = [a.text for a in author_elements if a.text]
        authors = ", ".join(author_names[:3])
        if len(author_names) > 3:
            authors += " et al."

        results.append({
            "title": title,
            "authors": authors,
            "year": published,
            "abstract": summary,
            "citations": "N/A",
            "url": link,
            "pdf_url": pdf_link,
            "source": "arXiv",
        })
    return results


def _format_papers(papers: list[dict]) -> str:
    """Format paper list into readable text."""
    source = papers[0].get("source", "")
    lines = []
    for i, paper in enumerate(papers, 1):
        abstract = paper["abstract"]
        if len(abstract) > 300:
            abstract = abstract[:300] + "..."
        pdf_line = f"    PDF: {paper['pdf_url']}\n" if paper.get("pdf_url") else ""
        lines.append(
            f"[{i}] {paper['title']} ({paper['year']})\n"
            f"    Authors: {paper['authors']}\n"
            f"    Citations: {paper['citations']}\n"
            f"    URL: {paper['url']}\n"
            f"{pdf_line}"
            f"    Abstract: {abstract}\n"
        )
    return f"Found {len(papers)} papers (via {source}):\n\n" + "\n".join(lines)


class SearchLiteratureTool(Tool):
    """Search for academic papers using arXiv.

    Every successful query is appended to ``<working_dir>/literature_cache.jsonl``
    (one paper per line). ``GenerateReportTool`` reads this file when the
    agent calls ``generate_report`` without explicit ``references=...``,
    so the agent can't accidentally produce a paper with an empty
    bibliography just because it forgot to forward the search results.
    """

    def __init__(
        self,
        max_results: int = 10,
        working_dir: str | None = None,
    ):
        self.max_results = max_results
        self.working_dir = working_dir



        self._quota: int = 0
        self._calls_used: int = 0

    def set_quota(self, quota: int) -> None:
        """Cap the number of search calls per run. ``quota <= 0`` disables the cap."""
        self._quota = max(0, int(quota or 0))
        self._calls_used = 0

    @property
    def name(self) -> str:
        return "search_literature"

    @property
    def description(self) -> str:
        return (
            "Search for academic papers via arXiv (free, ML/CS-heavy). Returns "
            "paper titles, authors, year, abstract, and an open-access PDF URL "
            "when available. Use this to find related work, understand "
            "baselines, or check if an approach already exists. Plan all your "
            "queries up front (Phase 1) — this tool has a per-run quota so "
            "reactive re-searching wastes budget. If you need to inspect the "
            "actual paper contents, follow up with read_paper_fulltext on a "
            "promising result."
        )

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query for finding papers.",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results to return (default 10).",
                    "default": 10,
                },
            },
            "required": ["query"],
        }

    def _persist_cache(self, query: str, source: str, papers: list[dict]) -> None:
        """Append every returned paper to ``literature_cache.jsonl``.

        Best-effort: failures here are swallowed so they never affect the
        agent's run. Each line is a self-contained JSON object so the cache
        file stays append-friendly across many search calls.
        """
        if not self.working_dir or not papers:
            return
        try:
            import json as _json
            from pathlib import Path as _P
            cache_path = _P(self.working_dir) / "literature_cache.jsonl"
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with cache_path.open("a", encoding="utf-8") as f:
                for paper in papers:
                    record = {**paper, "_query": query, "_source": source}
                    f.write(_json.dumps(record, ensure_ascii=False) + "\n")
        except Exception as exc:
            logger.debug("literature_cache.jsonl write skipped: %s", exc)

    def execute(self, query: str, max_results: int | None = None) -> ToolResult:
        limit = max_results or self.max_results




        if self._quota > 0 and self._calls_used >= self._quota:
            return ToolResult(
                output=(
                    f"search_literature quota exhausted ({self._calls_used}/"
                    f"{self._quota} calls used this run). The agent must work "
                    "with the papers already retrieved and saved to "
                    "literature_cache.jsonl. If a critical paper is missing, "
                    "either pivot the analysis to what's available or honestly "
                    "note the literature-search limitation in the report."
                ),
                success=False,
                metadata={"quota_exhausted": True, "calls_used": self._calls_used, "quota": self._quota},
            )

        self._calls_used += 1

        attempt_log: list[str] = []
        papers: list[dict] = []
        failure: Exception | None = None

        try:
            papers = _search_arxiv(query, limit, attempt_log=attempt_log)
        except Exception as exc:
            logger.warning("arXiv search exhausted retries: %s", exc)
            failure = exc

        if papers:
            kept, dropped = _filter_by_relevance(query, papers)
            relevance_note = ""
            if dropped:
                preview = "; ".join(
                    f"{title[:60]} (score={score:.2f})" for title, score in dropped[:3]
                )
                if len(dropped) > 3:
                    preview += f"; … (+{len(dropped) - 3} more)"
                relevance_note = (
                    f"\n\n[relevance filter] dropped {len(dropped)} of "
                    f"{len(papers)} hit(s) below threshold "
                    f"{RELEVANCE_THRESHOLD:.2f} for query {query!r}: {preview}"
                )
            if not kept:
                return ToolResult(
                    output=(
                        "All literature search hits fell below the relevance "
                        f"threshold ({RELEVANCE_THRESHOLD:.2f}) for query "
                        f"{query!r}. The upstream API over-recalled. "
                        "Refine the query with more specific terms (the dataset "
                        "name, the algorithm family, the venue) and try again."
                        + relevance_note
                    ),
                    success=False,
                    metadata={
                        "num_results": 0,
                        "num_dropped_low_relevance": len(dropped),
                        "source": "arxiv",
                        "calls_used": self._calls_used,
                        "quota": self._quota,
                    },
                )
            self._persist_cache(query, "arxiv", kept)
            quota_note = ""
            if self._quota > 0:
                quota_note = (
                    f"\n\n[search budget] {self._calls_used}/{self._quota} "
                    "search_literature calls used this run."
                )
            return ToolResult(
                output=_format_papers(kept) + relevance_note + quota_note,
                success=True,
                metadata={
                    "num_results": len(kept),
                    "num_dropped_low_relevance": len(dropped),
                    "source": "arxiv",
                    "calls_used": self._calls_used,
                    "quota": self._quota,
                },
            )

        if failure is not None:
            reason = f"{type(failure).__name__}: {failure}"
            detail = f"\n\nLast error: {reason}"
            if attempt_log:
                detail += "\n\nRetry log:\n- " + "\n- ".join(attempt_log)
            return ToolResult(
                output=(
                    "Literature search failed "
                    f"({reason}). This is usually a transient network issue — "
                    "try again, or proceed while explicitly noting the "
                    "literature-search limitation."
                    + detail
                ),
                success=False,
                metadata={"calls_used": self._calls_used, "quota": self._quota},
            )


        detail = ""
        if attempt_log:
            detail = "\n\nRetry log:\n- " + "\n- ".join(attempt_log)
        return ToolResult(
            output=(
                "Literature search returned no results for this query. "
                "Try a broader or differently phrased query, or proceed while "
                "explicitly noting that no related work was found."
                + detail
            ),
            success=False,
            metadata={"calls_used": self._calls_used, "quota": self._quota},
        )
