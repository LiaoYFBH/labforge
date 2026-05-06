"""
Full-text paper reader with PDF extraction and PaddleOCR-VL fallback.
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import mimetypes
import os
import re
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests

try:
    from bs4 import BeautifulSoup
except ImportError:  # pragma: no cover - optional dependency
    BeautifulSoup = None

try:
    from pypdf import PdfReader
except ImportError:  # pragma: no cover - optional dependency
    PdfReader = None

from ..sandbox import Sandbox
from .base import Tool, ToolResult

logger = logging.getLogger(__name__)

DEFAULT_CACHE_DIR = "papers"
DEFAULT_CHUNK_SIZE = 6000
DEFAULT_OCR_API_URL = "https://j5j557k6rbo1c6f4.aistudio-app.com/layout-parsing"
MIN_MACHINE_TEXT_CHARS = 1200
MIN_HTML_FULLTEXT_CHARS = 5000
HTTP_HEADERS = {"User-Agent": "LabForge/0.1"}
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
PDF_SUFFIXES = {".pdf"}


@dataclass
class ResolvedPaper:
    source_kind: str
    local_path: Path
    original_input: str
    resolved_url: str | None = None
    html_text: str = ""
    pdf_url: str = ""


def _slugify(value: str, max_len: int = 64) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", value).strip("-").lower()
    return slug[:max_len] or "paper"


def _hash_suffix(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()[:8]


def _collapse_whitespace(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def _decode_base64_blob(value: str) -> bytes:
    if "," in value and value.lstrip().startswith("data:"):
        value = value.split(",", 1)[1]
    return base64.b64decode(value)


def _get_env_first(*keys: str) -> str:
    for key in keys:
        value = os.environ.get(key, "").strip()
        if value:
            return value
    return ""


def _looks_like_pdf_url(url: str) -> bool:
    return url.lower().endswith(".pdf") or "/pdf/" in url.lower()


def _looks_like_image_url(url: str) -> bool:
    suffix = Path(urlparse(url).path).suffix.lower()
    return suffix in IMAGE_SUFFIXES


def _guess_suffix_from_response(resp: requests.Response, fallback_url: str) -> str:
    content_type = (resp.headers.get("content-type") or "").split(";")[0].strip().lower()
    if "pdf" in content_type:
        return ".pdf"
    if content_type.startswith("image/"):
        guessed = mimetypes.guess_extension(content_type)
        if guessed:
            return guessed
    suffix = Path(urlparse(fallback_url).path).suffix
    return suffix or ".bin"


def _normalize_arxiv_pdf_url(url: str) -> str | None:
    parsed = urlparse(url)
    if "arxiv.org" not in parsed.netloc:
        return None
    if "/pdf/" in parsed.path and parsed.path.endswith(".pdf"):
        return url
    if parsed.path.startswith("/abs/"):
        paper_id = parsed.path.split("/abs/", 1)[1]
        return f"https://arxiv.org/pdf/{paper_id}.pdf"
    return None


def _extract_pdf_url_from_html(base_url: str, html: str) -> str:
    if BeautifulSoup is not None:
        soup = BeautifulSoup(html, "html.parser")
        for meta_name in ("citation_pdf_url", "pdf_url"):
            meta = soup.find("meta", attrs={"name": meta_name})
            if meta and meta.get("content"):
                return urljoin(base_url, meta["content"].strip())
        for link in soup.find_all("a", href=True):
            href = link["href"].strip()
            if href.lower().endswith(".pdf") or "/pdf/" in href.lower():
                return urljoin(base_url, href)

    patterns = [
        r'content=["\']([^"\']+\.pdf(?:\?[^"\']*)?)["\']',
        r'href=["\']([^"\']+\.pdf(?:\?[^"\']*)?)["\']',
        r'href=["\']([^"\']+/pdf/[^"\']+)["\']',
    ]
    for pattern in patterns:
        match = re.search(pattern, html, flags=re.IGNORECASE)
        if match:
            return urljoin(base_url, match.group(1).strip())
    return ""


def _extract_html_text(html: str) -> str:
    if BeautifulSoup is not None:
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "noscript", "svg", "header", "footer", "nav"]):
            tag.decompose()
        root = soup.find("article") or soup.find("main") or soup.body or soup
        return _collapse_whitespace(root.get_text("\n"))

    text = re.sub(r"<script.*?>.*?</script>", " ", html, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"<style.*?>.*?</style>", " ", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"<[^>]+>", "\n", text)
    return _collapse_whitespace(text)


def _looks_like_fulltext_html(text: str) -> bool:
    lowered = text.lower()
    keyword_hits = sum(
        1
        for keyword in (
            "introduction",
            "related work",
            "method",
            "methods",
            "experiment",
            "results",
            "discussion",
            "conclusion",
            "references",
        )
        if keyword in lowered
    )
    return len(text) >= MIN_HTML_FULLTEXT_CHARS and keyword_hits >= 2


def _split_into_chunks(text: str, chunk_size: int) -> list[str]:
    chunks: list[str] = []
    remaining = text.strip()
    while remaining:
        if len(remaining) <= chunk_size:
            chunks.append(remaining)
            break
        split_at = remaining.rfind("\n\n", 0, chunk_size)
        if split_at < chunk_size // 2:
            split_at = remaining.rfind("\n", 0, chunk_size)
        if split_at < chunk_size // 2:
            split_at = chunk_size
        chunks.append(remaining[:split_at].strip())
        remaining = remaining[split_at:].lstrip()
    return chunks


class ReadPaperFullTextTool(Tool):
    """Read a paper's full text by downloading or OCR-ing its source."""

    def __init__(self, sandbox: Sandbox, cache_dir: str = DEFAULT_CACHE_DIR, ocr_enabled: bool = False):
        self.sandbox = sandbox
        self.cache_root = Path(self.sandbox.working_dir) / cache_dir
        self.cache_root.mkdir(parents=True, exist_ok=True)
        self.ocr_enabled = ocr_enabled

    @property
    def name(self) -> str:
        return "read_paper_fulltext"

    @property
    def description(self) -> str:
        return (
            "Download or load a paper PDF/image and extract full text for deep reading. "
            "Use this after search_literature when a paper looks important. The tool "
            "tries direct PDF text extraction first, then falls back to PaddleOCR-VL "
            "for scanned PDFs or images. It saves full text plus chunk files into the "
            "workspace so you can inspect the whole paper with file_read."
        )

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "Paper landing page, arXiv URL, or direct PDF/image URL.",
                },
                "local_path": {
                    "type": "string",
                    "description": "Local PDF/image path (relative to the workspace or absolute).",
                },
                "title": {
                    "type": "string",
                    "description": "Optional title hint used for naming saved files.",
                },
                "force_ocr": {
                    "type": "boolean",
                    "description": "Force PaddleOCR-VL even when direct PDF text extraction succeeds.",
                    "default": False,
                },
                "chunk_size_chars": {
                    "type": "integer",
                    "description": "Approximate size of each saved chunk file.",
                    "default": DEFAULT_CHUNK_SIZE,
                },
                "use_doc_orientation_classify": {
                    "type": "boolean",
                    "description": "PaddleOCR-VL option: whether to classify document orientation.",
                    "default": False,
                },
                "use_doc_unwarping": {
                    "type": "boolean",
                    "description": "PaddleOCR-VL option: whether to apply document unwarping.",
                    "default": False,
                },
                "use_chart_recognition": {
                    "type": "boolean",
                    "description": "PaddleOCR-VL option: whether to run chart recognition.",
                    "default": False,
                },
                "use_layout_detection": {
                    "type": "boolean",
                    "description": "PaddleOCR-VL option: whether to run layout detection and ordering.",
                },
                "layout_threshold": {
                    "type": "number",
                    "description": "Optional layout score threshold between 0 and 1.",
                },
                "layout_nms": {
                    "type": "boolean",
                    "description": "Optional PaddleOCR-VL layout NMS switch.",
                },
                "layout_unclip_ratio": {
                    "type": "number",
                    "description": "Optional layout box expansion ratio (>0).",
                },
                "layout_merge_bboxes_mode": {
                    "type": "string",
                    "description": "Optional layout merge mode: large, small, or union.",
                },
                "layout_shape_mode": {
                    "type": "string",
                    "description": "Optional layout geometry mode: rect, quad, poly, or auto.",
                },
                "prompt_label": {
                    "type": "string",
                    "description": "Optional prompt label when layout detection is disabled: ocr, formula, table, or chart.",
                },
                "repetition_penalty": {
                    "type": "number",
                    "description": "Optional repetition penalty for OCR decoding.",
                },
                "temperature": {
                    "type": "number",
                    "description": "Optional OCR decoding temperature.",
                },
                "top_p": {
                    "type": "number",
                    "description": "Optional OCR decoding top-p.",
                },
                "min_pixels": {
                    "type": "number",
                    "description": "Optional minimum image size parameter.",
                },
                "max_pixels": {
                    "type": "number",
                    "description": "Optional maximum image size parameter.",
                },
                "show_formula_number": {
                    "type": "boolean",
                    "description": "Whether Markdown output should include formula numbering.",
                },
                "restructure_pages": {
                    "type": "boolean",
                    "description": "Whether to reconstruct multi-page PDF results.",
                },
                "merge_tables": {
                    "type": "boolean",
                    "description": "Whether to merge cross-page tables when supported.",
                },
                "relevel_titles": {
                    "type": "boolean",
                    "description": "Whether to infer heading levels when supported.",
                },
                "prettify_markdown": {
                    "type": "boolean",
                    "description": "Whether to request beautified Markdown.",
                },
                "visualize": {
                    "type": "boolean",
                    "description": "Whether to ask the API to return visualization images.",
                },
            },
        }

    def execute(
        self,
        url: str = "",
        local_path: str = "",
        title: str = "",
        force_ocr: bool = False,
        chunk_size_chars: int = DEFAULT_CHUNK_SIZE,
        use_doc_orientation_classify: bool | None = None,
        use_doc_unwarping: bool | None = None,
        use_chart_recognition: bool | None = None,
        use_layout_detection: bool | None = None,
        layout_threshold: float | None = None,
        layout_nms: bool | None = None,
        layout_unclip_ratio: float | None = None,
        layout_merge_bboxes_mode: str = "",
        layout_shape_mode: str = "",
        prompt_label: str = "",
        repetition_penalty: float | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        min_pixels: float | None = None,
        max_pixels: float | None = None,
        show_formula_number: bool | None = None,
        restructure_pages: bool | None = None,
        merge_tables: bool | None = None,
        relevel_titles: bool | None = None,
        prettify_markdown: bool | None = None,
        visualize: bool | None = None,
    ) -> ToolResult:
        if not url and not local_path:
            return ToolResult(
                output="Paper read failed: provide either url or local_path.",
                success=False,
            )
        if url and local_path:
            return ToolResult(
                output="Paper read failed: provide only one of url or local_path.",
                success=False,
            )

        paper_dir = self._make_paper_dir(title=title, url=url, local_path=local_path)

        try:
            resolved = self._resolve_input(url=url, local_path=local_path, paper_dir=paper_dir)
        except Exception as exc:
            return ToolResult(output=f"Paper read failed during download/resolve: {exc}", success=False)

        extracted_text = ""
        extraction_method = ""
        notes: list[str] = []

        if resolved.source_kind == "html":
            extracted_text = resolved.html_text
            extraction_method = "html_text"
        elif resolved.source_kind == "pdf" and not force_ocr:
            extracted_text = self._extract_pdf_text(resolved.local_path)
            if extracted_text:
                extraction_method = "pypdf"

        if resolved.source_kind in {"pdf", "image"}:
            needs_ocr = force_ocr or resolved.source_kind == "image" or len(_collapse_whitespace(extracted_text)) < MIN_MACHINE_TEXT_CHARS
            if needs_ocr and not self.ocr_enabled:
                notes.append("OCR needed but disabled via settings. Enable PaddleOCR in the UI to use OCR extraction.")
                needs_ocr = False
            if needs_ocr:
                try:
                    ocr_result = self._run_paddleocr(
                        file_path=resolved.local_path,
                        file_type=0 if resolved.source_kind == "pdf" else 1,
                        paper_dir=paper_dir,
                        use_doc_orientation_classify=use_doc_orientation_classify,
                        use_doc_unwarping=use_doc_unwarping,
                        use_chart_recognition=use_chart_recognition,
                        use_layout_detection=use_layout_detection,
                        layout_threshold=layout_threshold,
                        layout_nms=layout_nms,
                        layout_unclip_ratio=layout_unclip_ratio,
                        layout_merge_bboxes_mode=layout_merge_bboxes_mode,
                        layout_shape_mode=layout_shape_mode,
                        prompt_label=prompt_label,
                        repetition_penalty=repetition_penalty,
                        temperature=temperature,
                        top_p=top_p,
                        min_pixels=min_pixels,
                        max_pixels=max_pixels,
                        show_formula_number=show_formula_number,
                        restructure_pages=restructure_pages,
                        merge_tables=merge_tables,
                        relevel_titles=relevel_titles,
                        prettify_markdown=prettify_markdown,
                        visualize=visualize,
                    )
                    if ocr_result["text"]:
                        extracted_text = ocr_result["text"]
                        extraction_method = "paddleocr_vl"
                except Exception as exc:
                    notes.append(f"OCR skipped/failed: {exc}")

        extracted_text = _collapse_whitespace(extracted_text)
        if not extracted_text:
            return ToolResult(
                output=(
                    "Paper read failed: no full text could be extracted. "
                    + (" ".join(notes) if notes else "")
                ).strip(),
                success=False,
            )

        if chunk_size_chars < 1000:
            chunk_size_chars = DEFAULT_CHUNK_SIZE

        title_line = title or self._derive_title(url=url, local_path=local_path, resolved=resolved)
        fulltext_relpath = self._write_fulltext_file(paper_dir, title_line, extracted_text)
        chunk_relpaths = self._write_chunk_files(
            paper_dir=paper_dir,
            title=title_line,
            text=extracted_text,
            chunk_size_chars=chunk_size_chars,
        )
        manifest_relpath = self._write_manifest(
            paper_dir=paper_dir,
            title=title_line,
            resolved=resolved,
            extraction_method=extraction_method,
            fulltext_relpath=fulltext_relpath,
            chunk_relpaths=chunk_relpaths,
            notes=notes,
        )

        preview = extracted_text[:1200]
        if len(extracted_text) > 1200:
            preview += "..."

        lines = [
            "Paper full text prepared successfully.",
            f"Title: {title_line}",
            f"Source kind: {resolved.source_kind}",
            f"Extraction method: {extraction_method or 'unknown'}",
            f"Manifest: {manifest_relpath}",
            f"Full text: {fulltext_relpath}",
            f"Chunks: {len(chunk_relpaths)}",
            "Chunk files:",
        ]
        lines.extend(f"  - {path}" for path in chunk_relpaths[:10])
        if len(chunk_relpaths) > 10:
            lines.append(f"  - ... ({len(chunk_relpaths) - 10} more)")
        if resolved.pdf_url:
            lines.append(f"Resolved PDF: {resolved.pdf_url}")
        if notes:
            lines.append("Notes:")
            lines.extend(f"  - {note}" for note in notes)
        lines.append("Preview:")
        lines.append(preview)

        return ToolResult(
            output="\n".join(lines),
            success=True,
            metadata={
                "paper_dir": str(paper_dir),
                "chunk_count": len(chunk_relpaths),
                "extraction_method": extraction_method,
            },
        )

    def _make_paper_dir(self, title: str, url: str, local_path: str) -> Path:
        base = title or url or local_path or "paper"
        folder_name = f"{_slugify(base)}-{_hash_suffix(base)}"
        paper_dir = self.cache_root / folder_name
        paper_dir.mkdir(parents=True, exist_ok=True)
        return paper_dir

    def _resolve_input(self, url: str, local_path: str, paper_dir: Path) -> ResolvedPaper:
        if local_path:
            return self._resolve_local_input(local_path=local_path, paper_dir=paper_dir)
        return self._resolve_remote_input(url=url, paper_dir=paper_dir)

    def _resolve_local_input(self, local_path: str, paper_dir: Path) -> ResolvedPaper:
        candidate = Path(local_path)
        if not candidate.is_absolute():
            workspace_candidate = Path(self.sandbox.working_dir) / local_path
            candidate = workspace_candidate if workspace_candidate.exists() else candidate.resolve()
        if not candidate.exists():
            raise FileNotFoundError(candidate)

        suffix = candidate.suffix.lower()
        source_name = "source" + (suffix or ".bin")
        target = paper_dir / source_name
        target.write_bytes(candidate.read_bytes())

        if suffix in PDF_SUFFIXES:
            source_kind = "pdf"
        elif suffix in IMAGE_SUFFIXES:
            source_kind = "image"
        else:
            raise ValueError(f"Unsupported local file type: {candidate.suffix or '(no suffix)'}")

        return ResolvedPaper(
            source_kind=source_kind,
            local_path=target,
            original_input=local_path,
        )

    def _resolve_remote_input(self, url: str, paper_dir: Path) -> ResolvedPaper:
        candidate_url = _normalize_arxiv_pdf_url(url) or url
        response = requests.get(candidate_url, headers=HTTP_HEADERS, timeout=60)
        response.raise_for_status()

        content_type = (response.headers.get("content-type") or "").lower()
        if "pdf" in content_type or _looks_like_pdf_url(candidate_url):
            return self._save_downloaded_binary(
                response=response,
                original_url=url,
                resolved_url=candidate_url,
                paper_dir=paper_dir,
                source_kind="pdf",
                pdf_url=candidate_url,
            )
        if content_type.startswith("image/") or _looks_like_image_url(candidate_url):
            return self._save_downloaded_binary(
                response=response,
                original_url=url,
                resolved_url=candidate_url,
                paper_dir=paper_dir,
                source_kind="image",
            )

        html = response.text
        html_path = paper_dir / "source.html"
        html_path.write_text(html, encoding="utf-8")

        pdf_url = _extract_pdf_url_from_html(candidate_url, html)
        if pdf_url:
            pdf_response = requests.get(pdf_url, headers=HTTP_HEADERS, timeout=60)
            pdf_response.raise_for_status()
            return self._save_downloaded_binary(
                response=pdf_response,
                original_url=url,
                resolved_url=pdf_url,
                paper_dir=paper_dir,
                source_kind="pdf",
                pdf_url=pdf_url,
            )

        html_text = _extract_html_text(html)
        if _looks_like_fulltext_html(html_text):
            return ResolvedPaper(
                source_kind="html",
                local_path=html_path,
                original_input=url,
                resolved_url=candidate_url,
                html_text=html_text,
            )

        raise RuntimeError("Could not find a downloadable PDF or trustworthy full-text HTML page.")

    def _save_downloaded_binary(
        self,
        response: requests.Response,
        original_url: str,
        resolved_url: str,
        paper_dir: Path,
        source_kind: str,
        pdf_url: str = "",
    ) -> ResolvedPaper:
        suffix = _guess_suffix_from_response(response, resolved_url)
        target = paper_dir / ("source" + suffix)
        target.write_bytes(response.content)
        return ResolvedPaper(
            source_kind=source_kind,
            local_path=target,
            original_input=original_url,
            resolved_url=resolved_url,
            pdf_url=pdf_url,
        )

    def _extract_pdf_text(self, pdf_path: Path) -> str:
        if PdfReader is None:
            return ""
        try:
            reader = PdfReader(str(pdf_path))
            page_texts = []
            for page in reader.pages:
                text = page.extract_text() or ""
                if text.strip():
                    page_texts.append(text.strip())
            return "\n\n".join(page_texts)
        except Exception as exc:
            logger.warning("Direct PDF text extraction failed for %s: %s", pdf_path, exc)
            return ""

    def _run_paddleocr(
        self,
        file_path: Path,
        file_type: int,
        paper_dir: Path,
        use_doc_orientation_classify: bool | None,
        use_doc_unwarping: bool | None,
        use_chart_recognition: bool | None,
        use_layout_detection: bool | None,
        layout_threshold: float | None,
        layout_nms: bool | None,
        layout_unclip_ratio: float | None,
        layout_merge_bboxes_mode: str,
        layout_shape_mode: str,
        prompt_label: str,
        repetition_penalty: float | None,
        temperature: float | None,
        top_p: float | None,
        min_pixels: float | None,
        max_pixels: float | None,
        show_formula_number: bool | None,
        restructure_pages: bool | None,
        merge_tables: bool | None,
        relevel_titles: bool | None,
        prettify_markdown: bool | None,
        visualize: bool | None,
    ) -> dict[str, str]:
        token = _get_env_first("PADDLEOCR_VL_TOKEN", "OCR_API_TOKEN")
        if not token:
            raise RuntimeError(
                "OCR token is not configured. Set PADDLEOCR_VL_TOKEN or OCR_API_TOKEN."
            )

        api_url = _get_env_first("PADDLEOCR_VL_API_URL", "OCR_API_URL") or DEFAULT_OCR_API_URL
        payload: dict[str, object] = {
            "file": base64.b64encode(file_path.read_bytes()).decode("ascii"),
            "fileType": file_type,
        }
        optional_payload = {
            "useDocOrientationClassify": use_doc_orientation_classify,
            "useDocUnwarping": use_doc_unwarping,
            "useChartRecognition": use_chart_recognition,
            "useLayoutDetection": use_layout_detection,
            "layoutThreshold": layout_threshold,
            "layoutNms": layout_nms,
            "layoutUnclipRatio": layout_unclip_ratio,
            "layoutMergeBboxesMode": layout_merge_bboxes_mode or None,
            "layoutShapeMode": layout_shape_mode or None,
            "promptLabel": prompt_label or None,
            "repetitionPenalty": repetition_penalty,
            "temperature": temperature,
            "topP": top_p,
            "minPixels": min_pixels,
            "maxPixels": max_pixels,
            "showFormulaNumber": show_formula_number,
            "restructurePages": restructure_pages,
            "mergeTables": merge_tables,
            "relevelTitles": relevel_titles,
            "prettifyMarkdown": prettify_markdown,
            "visualize": visualize,
        }
        payload.update({k: v for k, v in optional_payload.items() if v is not None})
        headers = {
            "Authorization": f"token {token}",
            "Content-Type": "application/json",
        }

        response = requests.post(api_url, json=payload, headers=headers, timeout=180)
        response.raise_for_status()
        body = response.json()
        if body.get("errorCode", 0) not in (0, None):
            raise RuntimeError(body.get("errorMsg") or f"PaddleOCR-VL error: {body['errorCode']}")
        result = body.get("result") or {}
        layout_results = result.get("layoutParsingResults") or []
        if not layout_results:
            raise RuntimeError("PaddleOCR-VL returned no layoutParsingResults.")

        page_dir = paper_dir / "ocr_pages"
        page_dir.mkdir(parents=True, exist_ok=True)
        markdown_pages: list[str] = []

        for idx, page in enumerate(layout_results, 1):
            markdown = page.get("markdown") or {}
            page_text = (markdown.get("text") or "").strip()
            if page_text:
                markdown_pages.append(page_text)
                (page_dir / f"page_{idx:03d}.md").write_text(page_text, encoding="utf-8")
            for relative_name, image_blob in (markdown.get("images") or {}).items():
                self._save_auxiliary_image(
                    paper_dir=paper_dir / "ocr_markdown_images",
                    relative_name=relative_name,
                    image_ref=image_blob,
                )
            for image_name, image_blob in (page.get("outputImages") or {}).items():
                self._save_auxiliary_image(
                    paper_dir=paper_dir / "ocr_output_images",
                    relative_name=f"{image_name}_{idx:03d}.jpg",
                    image_ref=image_blob,
                )

        return {"text": "\n\n".join(markdown_pages)}

    def _save_auxiliary_image(self, paper_dir: Path, relative_name: str, image_ref: str):
        try:
            if isinstance(image_ref, str) and image_ref.startswith(("http://", "https://")):
                response = requests.get(image_ref, headers=HTTP_HEADERS, timeout=60)
                response.raise_for_status()
                image_bytes = response.content
            else:
                image_bytes = _decode_base64_blob(image_ref)
        except Exception as exc:
            logger.warning("Skipping OCR image materialization for %s: %s", relative_name, exc)
            return
        target = paper_dir / relative_name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(image_bytes)

    def _derive_title(self, url: str, local_path: str, resolved: ResolvedPaper) -> str:
        if url:
            parsed = urlparse(resolved.resolved_url or url)
            name = Path(parsed.path).stem or parsed.netloc
            return name.replace("-", " ").replace("_", " ").strip() or "paper"
        return Path(local_path).stem.replace("-", " ").replace("_", " ").strip() or "paper"

    def _workspace_relpath(self, path: Path) -> str:
        return str(path.relative_to(Path(self.sandbox.working_dir)))

    def _write_fulltext_file(self, paper_dir: Path, title: str, text: str) -> str:
        fulltext_path = paper_dir / "fulltext.md"
        fulltext_path.write_text(f"# {title}\n\n{text}\n", encoding="utf-8")
        return self._workspace_relpath(fulltext_path)

    def _write_chunk_files(
        self,
        paper_dir: Path,
        title: str,
        text: str,
        chunk_size_chars: int,
    ) -> list[str]:
        chunk_dir = paper_dir / "chunks"
        chunk_dir.mkdir(parents=True, exist_ok=True)
        chunks = _split_into_chunks(text, chunk_size_chars)
        relpaths: list[str] = []
        total = len(chunks)
        for idx, chunk in enumerate(chunks, 1):
            chunk_path = chunk_dir / f"chunk_{idx:03d}.md"
            chunk_path.write_text(
                f"# {title}\n\nChunk {idx}/{total}\n\n{chunk}\n",
                encoding="utf-8",
            )
            relpaths.append(self._workspace_relpath(chunk_path))
        return relpaths

    def _write_manifest(
        self,
        paper_dir: Path,
        title: str,
        resolved: ResolvedPaper,
        extraction_method: str,
        fulltext_relpath: str,
        chunk_relpaths: list[str],
        notes: list[str],
    ) -> str:
        manifest = {
            "title": title,
            "original_input": resolved.original_input,
            "resolved_url": resolved.resolved_url,
            "source_kind": resolved.source_kind,
            "pdf_url": resolved.pdf_url,
            "extraction_method": extraction_method,
            "fulltext_path": fulltext_relpath,
            "chunk_paths": chunk_relpaths,
            "notes": notes,
        }
        manifest_path = paper_dir / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return self._workspace_relpath(manifest_path)
