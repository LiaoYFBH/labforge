"""PaddleOCR-VL API client for document parsing."""

from __future__ import annotations

import base64
import logging
from dataclasses import dataclass, field
from pathlib import Path

import requests

from .config import OCRConfig

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {
    ".pdf": 0,   # fileType 0 = PDF
    ".png": 1,   # fileType 1 = image
    ".jpg": 1,
    ".jpeg": 1,
    ".bmp": 1,
    ".tiff": 1,
    ".tif": 1,
    ".gif": 1,
}


@dataclass
class OCRResult:
    markdown_text: str = ""
    images: dict[str, bytes] = field(default_factory=dict)
    page_count: int = 0


def parse_document(file_path: str | Path, config: OCRConfig) -> OCRResult:
    """Parse a single document file via PaddleOCR-VL API."""
    file_path = Path(file_path)
    suffix = file_path.suffix.lower()

    if suffix not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type: {suffix}. "
            f"Supported: {', '.join(SUPPORTED_EXTENSIONS)}"
        )

    if not config.api_url or not config.token:
        raise ValueError("PaddleOCR API URL and Token must be configured.")

    file_data = base64.b64encode(file_path.read_bytes()).decode("ascii")
    file_type = SUPPORTED_EXTENSIONS[suffix]

    api_url = config.api_url.rstrip("/")
    if not api_url.endswith("/layout-parsing"):
        api_url += "/layout-parsing"

    headers = {
        "Authorization": f"token {config.token}",
        "Content-Type": "application/json",
    }

    payload = {
        "file": file_data,
        "fileType": file_type,
        "useDocOrientationClassify": config.use_doc_orientation_classify,
        "useDocUnwarping": config.use_doc_unwarping,
        "useChartRecognition": config.use_chart_recognition,
    }

    logger.info("Calling PaddleOCR API for %s (type=%d)", file_path.name, file_type)
    resp = requests.post(api_url, json=payload, headers=headers, timeout=300)
    resp.raise_for_status()

    data = resp.json()
    result_data = data.get("result", {})
    parsing_results = result_data.get("layoutParsingResults", [])

    all_markdown: list[str] = []
    all_images: dict[str, bytes] = {}

    for i, page_result in enumerate(parsing_results):
        md_section = page_result.get("markdown", {})
        md_text = md_section.get("text", "")
        if md_text:
            all_markdown.append(md_text)

        # Download/decode images referenced in this page
        images_dict = md_section.get("images", {})
        for img_name, img_src in images_dict.items():
            try:
                if img_src.startswith("http"):
                    img_bytes = requests.get(img_src, timeout=60).content
                else:
                    img_bytes = base64.b64decode(img_src)
                all_images[img_name] = img_bytes
            except Exception as e:
                logger.warning("Failed to fetch image %s: %s", img_name, e)

    return OCRResult(
        markdown_text="\n\n".join(all_markdown),
        images=all_images,
        page_count=len(parsing_results),
    )


def parse_multiple_documents(
    file_paths: list[str | Path],
    config: OCRConfig,
    progress_callback=None,
) -> OCRResult:
    """Parse multiple documents and merge results."""
    combined_markdown: list[str] = []
    combined_images: dict[str, bytes] = {}
    total_pages = 0

    for idx, fp in enumerate(file_paths):
        fp = Path(fp)
        if progress_callback:
            progress_callback(f"Parsing document {idx + 1}/{len(file_paths)}: {fp.name}")

        try:
            result = parse_document(fp, config)
            if result.markdown_text:
                combined_markdown.append(
                    f"<!-- Document: {fp.name} -->\n{result.markdown_text}"
                )
            # Prefix image names with doc index to avoid collision
            for img_name, img_data in result.images.items():
                unique_name = f"doc{idx}_{img_name}"
                combined_images[unique_name] = img_data
            total_pages += result.page_count
        except Exception as e:
            logger.error("Failed to parse %s: %s", fp.name, e)
            combined_markdown.append(
                f"<!-- Document: {fp.name} - PARSE FAILED: {e} -->"
            )

    return OCRResult(
        markdown_text="\n\n---\n\n".join(combined_markdown),
        images=combined_images,
        page_count=total_pages,
    )
