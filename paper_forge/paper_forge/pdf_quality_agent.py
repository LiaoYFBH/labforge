"""Post-render quality checks for PaperForge PDFs.

This is the deterministic first layer of the PaperForge QA agent. It inspects
the rendered LaTeX source and compiler log for issues that are visible in the
final PDF: overfull boxes, leaked Markdown fences, code snippets in prose,
language mismatch, and long math that was not constrained to column width.

The module is intentionally standalone and model-free. A later visual layer can
feed rendered page images through PaddleOCR-VL and pass the layout/text evidence
to an LLM reviewer without changing the public report schema below.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path


_CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]")
_LATIN_RE = re.compile(r"[A-Za-z]")
_CODE_FENCE_RE = re.compile(r"```")
_CODE_LINE_RE = re.compile(
    r"(^|\s)(import|from|def|class|assert|return|for\s+\w+\s+in|while\s+|if\s+)"
    r"|np\.|pd\.|plt\.|torch\.|sklearn\.|\.append\(|#\s*\w+|print\(",
    re.IGNORECASE,
)
_DISPLAY_MATH_RE = re.compile(r"\\\[(.*?)\\\]", re.DOTALL)
_OVERFULL_RE = re.compile(r"Overfull \\hbox \(([^)]+) too wide\)")


@dataclass
class PdfQualityIssue:
    severity: str
    kind: str
    message: str
    evidence: str = ""


@dataclass
class PdfQualityReport:
    passed: bool
    issues: list[PdfQualityIssue] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "issues": [asdict(issue) for issue in self.issues],
        }


def inspect_paper_artifacts(
    *,
    tex_path: str | Path | None = None,
    log_path: str | Path | None = None,
    target_language: str | None = None,
) -> PdfQualityReport:
    """Inspect rendered artifacts and return a QA report."""
    issues: list[PdfQualityIssue] = []
    tex_text = ""
    if tex_path is not None and Path(tex_path).exists():
        tex_text = Path(tex_path).read_text(encoding="utf-8", errors="replace")
        issues.extend(_inspect_tex(tex_text, target_language=target_language))

    if log_path is not None and Path(log_path).exists():
        log_text = Path(log_path).read_text(encoding="utf-8", errors="replace")
        issues.extend(_inspect_latex_log(log_text))

    blocking = any(issue.severity == "error" for issue in issues)
    return PdfQualityReport(passed=not blocking, issues=issues)


def write_quality_report(report: PdfQualityReport, output_path: str | Path) -> Path:
    """Write the QA report as JSON and return its path."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def format_quality_summary(report: PdfQualityReport, max_items: int = 5) -> str:
    if not report.issues:
        return "PDF QA passed: no obvious layout or prose-artifact issues detected."
    rendered = []
    for issue in report.issues[:max_items]:
        rendered.append(f"- {issue.severity.upper()} {issue.kind}: {issue.message}")
    if len(report.issues) > max_items:
        rendered.append(f"- ... {len(report.issues) - max_items} more issue(s)")
    return "PDF QA detected potential issues:\n" + "\n".join(rendered)


def _inspect_tex(tex_text: str, target_language: str | None) -> list[PdfQualityIssue]:
    issues: list[PdfQualityIssue] = []
    if _CODE_FENCE_RE.search(tex_text):
        issues.append(PdfQualityIssue(
            severity="error",
            kind="markdown_fence",
            message="Markdown code fences leaked into the LaTeX source.",
            evidence="```",
        ))

    for line in tex_text.splitlines():
        stripped = line.strip()
        if _CODE_LINE_RE.search(stripped):
            issues.append(PdfQualityIssue(
                severity="warning",
                kind="code_like_prose",
                message="Code-like syntax appears in the paper body.",
                evidence=stripped[:160],
            ))
            break

    for match in _DISPLAY_MATH_RE.finditer(tex_text):
        body = re.sub(r"\s+", "", match.group(1))
        if len(body) > 110 and "\\resizebox" not in match.group(0):
            issues.append(PdfQualityIssue(
                severity="warning",
                kind="long_math",
                message="A long display equation is not constrained to column width.",
                evidence=body[:160],
            ))
            break

    if target_language in ("zh", "en"):
        cjk_count = len(_CJK_RE.findall(tex_text))
        latin_count = len(_LATIN_RE.findall(tex_text))
        if target_language == "en" and cjk_count > 30:
            issues.append(PdfQualityIssue(
                severity="warning",
                kind="language_mismatch",
                message="English export still contains substantial Chinese text.",
                evidence=f"cjk_count={cjk_count}",
            ))
        elif target_language == "zh" and latin_count > max(250, cjk_count * 4):
            issues.append(PdfQualityIssue(
                severity="warning",
                kind="language_mismatch",
                message="Chinese export appears dominated by English prose.",
                evidence=f"latin_count={latin_count}, cjk_count={cjk_count}",
            ))
    return issues


def _inspect_latex_log(log_text: str) -> list[PdfQualityIssue]:
    issues: list[PdfQualityIssue] = []
    overfulls = _OVERFULL_RE.findall(log_text)
    if overfulls:
        issues.append(PdfQualityIssue(
            severity="warning",
            kind="overfull_hbox",
            message=f"LaTeX reported {len(overfulls)} overfull hbox warning(s).",
            evidence=", ".join(overfulls[:3]),
        ))
    return issues
