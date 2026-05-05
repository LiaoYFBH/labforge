"""Deterministic checks for obviously invalid experiment results.

LLM review is useful for scientific judgment, but values such as ``inf`` and
``NaN`` are machine-checkable. This module is intentionally small and
dependency-free so it can run inside every report / submit path.
"""

from __future__ import annotations

import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class ResultFinding:
    """One suspicious or invalid value found in an experiment artifact."""

    source: str
    location: str
    value: str
    reason: str
    severity: str = "error"


NUMERIC_ARTIFACT_EXTENSIONS = {".csv", ".tsv", ".json", ".jsonl", ".txt"}
SKIP_DIRS = {
    ".remote_images",
    "data",
    "literature_cache",
    "node_modules",
    "papers",
    "uploads",
}
SKIP_FILES = {
    "paperforge_bundle.json",
    "research_report.md",
}
MAX_SCAN_ROWS = 2000
MAX_FINDINGS = 80

_NONFINITE_WORD_RE = re.compile(
    r"(?<![A-Za-z0-9_])(?:[+-]?(?:nan|inf(?:inity)?)|[+-]?∞)(?![A-Za-z0-9_])",
    re.IGNORECASE,
)





_NONCONVERGENCE_RE = re.compile(
    r"\bConvergenceWarning\b"
    r"|\bmax_iter\s+was\s+reached\b"
    r"|\bMaximum\s+iterations\s+\([^)]*\)\s+reached\b"
    r"|\bSolver\s+terminated\s+early\b"
    r"|\b(?:lbfgs|saga|adam|sgd)\s+failed\s+to\s+converge\b"
    r"|\b(?:optimization|coef_)\s+(?:has(?:n['’]t)?|did)\s+not\s+converged?\b"
    r"|\bdid\s+not\s+converge\b",
    re.IGNORECASE,
)


def validate_workspace_results(working_dir: str | Path) -> list[ResultFinding]:
    """Scan generated result files and execution logs for blocking issues."""

    root = Path(working_dir)
    if not root.exists():
        return []

    findings: list[ResultFinding] = []

    for path in sorted(root.rglob("*")):
        if len(findings) >= MAX_FINDINGS:
            break
        if not path.is_file():
            continue
        try:
            rel = path.relative_to(root)
        except ValueError:
            continue
        if _should_skip(rel):
            continue

        suffix = path.suffix.lower()
        if rel.parts and rel.parts[0] == "logs" and suffix == ".log":
            findings.extend(_scan_execution_log(path, rel, MAX_FINDINGS - len(findings)))
        elif suffix in {".csv", ".tsv"}:
            findings.extend(_scan_delimited_file(path, rel, MAX_FINDINGS - len(findings)))
        elif suffix == ".json":
            findings.extend(_scan_json_file(path, rel, MAX_FINDINGS - len(findings)))
        elif suffix == ".jsonl":
            findings.extend(_scan_jsonl_file(path, rel, MAX_FINDINGS - len(findings)))
        elif suffix == ".txt":
            findings.extend(_scan_text_file(path, rel, MAX_FINDINGS - len(findings)))

    return findings[:MAX_FINDINGS]


def blocking_findings(findings: Iterable[ResultFinding]) -> list[ResultFinding]:
    """Return findings that must be fixed or explicitly disclosed."""

    return [finding for finding in findings if finding.severity == "error"]


def format_findings(findings: Iterable[ResultFinding], *, max_items: int = 10) -> str:
    """Render findings as a concise message for the agent / UI."""

    items = list(findings)
    if not items:
        return "No invalid numeric experiment results detected."

    lines = ["Invalid experiment result guardrail triggered:"]
    for finding in items[:max_items]:
        lines.append(
            f"- {finding.source} {finding.location}: {finding.reason} "
            f"(value={finding.value!r})"
        )
    if len(items) > max_items:
        lines.append(f"- ... and {len(items) - max_items} more finding(s).")
    return "\n".join(lines)


def report_text_discloses_findings(text: str, findings: Iterable[ResultFinding]) -> bool:
    """True when a report draft explicitly treats findings as invalid.

    Merely repeating ``MSE=inf`` is not enough; the text must also use a
    caution/failure marker so the paper cannot present broken numbers as
    successful results.
    """

    blocking = blocking_findings(findings)
    if not blocking:
        return True

    haystack = (text or "").casefold()
    if not haystack.strip():
        return False

    caution_markers = (
        "abnormal",
        "cannot conclude",
        "caveat",
        "diverge",
        "diverged",
        "error",
        "failed",
        "failure",
        "invalid",
        "issue",
        "not reliable",
        "problem",
        "rerun",
        "unstable",
        "warning",
        "不可靠",
        "不应",
        "不能",
        "修正",
        "发散",
        "失败",
        "异常",
        "无效",
        "有问题",
        "警告",
        "错误",
        "需",
        "需要",
    )
    has_caution = any(marker in haystack for marker in caution_markers)
    if not has_caution:
        return False

    for finding in blocking:
        reason = finding.reason.casefold()
        value = finding.value.casefold()
        if "non-finite" in reason:
            if value.startswith("nan"):
                markers = ("nan", "not a number", "非数", "缺失", "异常")
            else:
                markers = (
                    "inf",
                    "infinite",
                    "infinity",
                    "non-finite",
                    "nonfinite",
                    "overflow",
                    "发散",
                    "无穷",
                    "非有限",
                )
        elif "negative convergence" in reason:
            markers = (
                "negative convergence",
                "negative rate",
                "below zero",
                "负",
                "负值",
                "收敛",
            )
        else:
            markers = ("invalid", "abnormal", "异常", "无效")

        if not any(marker in haystack for marker in markers):
            return False

    return True


def _should_skip(rel: Path) -> bool:
    if rel.name in SKIP_FILES:
        return True
    return any(part in SKIP_DIRS for part in rel.parts)


def _scan_delimited_file(path: Path, rel: Path, budget: int) -> list[ResultFinding]:
    delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
    findings: list[ResultFinding] = []
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.reader(handle, delimiter=delimiter)
            rows = [row for _, row in zip(range(MAX_SCAN_ROWS + 1), reader)]
    except (OSError, UnicodeDecodeError, csv.Error):
        return findings

    if not rows:
        return findings

    headers = [str(cell).strip() for cell in rows[0]]
    for row_idx, row in enumerate(rows[1:], start=2):
        for col_idx, value in enumerate(row, start=1):
            if len(findings) >= budget:
                return findings
            header = headers[col_idx - 1] if col_idx - 1 < len(headers) else f"Col {col_idx}"
            location = f"row {row_idx}, column {header or col_idx}"
            findings.extend(_inspect_value(value, rel, location))
            if len(findings) >= budget:
                return findings
            findings.extend(
                _inspect_negative_convergence(value, rel, location, header)
            )
    return findings


def _scan_json_file(path: Path, rel: Path, budget: int) -> list[ResultFinding]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return []
    findings: list[ResultFinding] = []
    _scan_json_value(data, rel, "$", findings, budget)
    return findings


def _scan_jsonl_file(path: Path, rel: Path, budget: int) -> list[ResultFinding]:
    findings: list[ResultFinding] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError):
        return findings
    for line_idx, line in enumerate(lines[:MAX_SCAN_ROWS], start=1):
        if len(findings) >= budget:
            break
        line = line.strip()
        if not line:
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            findings.extend(
                _scan_text_for_nonfinite(line, rel, f"line {line_idx}", budget - len(findings))
            )
            continue
        _scan_json_value(data, rel, f"$[{line_idx}]", findings, budget)
    return findings


def _scan_json_value(
    value: Any,
    rel: Path,
    location: str,
    findings: list[ResultFinding],
    budget: int,
) -> None:
    if len(findings) >= budget:
        return
    if isinstance(value, dict):
        for key, item in value.items():
            _scan_json_value(item, rel, f"{location}.{key}", findings, budget)
            if len(findings) >= budget:
                return
    elif isinstance(value, list):
        for idx, item in enumerate(value):
            _scan_json_value(item, rel, f"{location}[{idx}]", findings, budget)
            if len(findings) >= budget:
                return
    else:
        findings.extend(_inspect_value(value, rel, location))


def _scan_text_file(path: Path, rel: Path, budget: int) -> list[ResultFinding]:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    return _scan_text_for_nonfinite(text, rel, "text", budget)


def _scan_execution_log(path: Path, rel: Path, budget: int) -> list[ResultFinding]:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    sections = _extract_log_output_sections(text)
    if not sections:
        sections = [("output", text)]

    findings: list[ResultFinding] = []
    for label, body in sections:
        if len(findings) >= budget:
            break
        findings.extend(
            _scan_text_for_nonfinite(body, rel, label, budget - len(findings))
        )
        if label == "stderr" and len(findings) < budget:
            findings.extend(
                _scan_text_for_nonconvergence(
                    body, rel, label, budget - len(findings)
                )
            )
    return findings


def _scan_text_for_nonconvergence(
    text: str,
    rel: Path,
    location_prefix: str,
    budget: int,
) -> list[ResultFinding]:
    """Surface sklearn/torch convergence warnings as blocking findings.

    The trajectory bug we are guarding against: an LR/SVM/MLP run with
    ``max_iter=3`` exits with ``exit_code=0`` but stderr is full of
    ``ConvergenceWarning``. Without this scan, the metrics CSV looks legit
    and the paper writer happily quotes "89.8% accuracy" as a settled result.
    We only report at most one finding per distinct warning sentence to
    avoid flooding the agent on a verbose stderr.
    """
    findings: list[ResultFinding] = []
    seen_evidence: set[str] = set()
    for line_idx, line in enumerate(text.splitlines()[:MAX_SCAN_ROWS], start=1):
        if len(findings) >= budget:
            break
        match = _NONCONVERGENCE_RE.search(line)
        if not match:
            continue
        evidence = match.group(0).casefold()
        if evidence in seen_evidence:
            continue
        seen_evidence.add(evidence)
        findings.append(
            ResultFinding(
                source=rel.as_posix(),
                location=f"{location_prefix} line {line_idx}",
                value=line.strip()[:160],
                reason=(
                    "training did not converge — raise max_iter, scale "
                    "features, or pick a stronger solver before reporting "
                    "the metrics as a real result"
                ),
            )
        )
    return findings


def _extract_log_output_sections(text: str) -> list[tuple[str, str]]:
    sections: list[tuple[str, str]] = []
    for marker in ("STDOUT", "STDERR"):
        pattern = re.compile(
            rf"---- {marker} ----\n(.*?)(?=\n---- [A-Z ]+ ----|\Z)",
            re.DOTALL,
        )
        match = pattern.search(text)
        if match:
            sections.append((marker.lower(), match.group(1)))
    return sections


def _scan_text_for_nonfinite(
    text: str,
    rel: Path,
    location_prefix: str,
    budget: int,
) -> list[ResultFinding]:
    findings: list[ResultFinding] = []
    for line_idx, line in enumerate(text.splitlines()[:MAX_SCAN_ROWS], start=1):
        if len(findings) >= budget:
            break
        if _line_denies_nonfinite(line):
            continue
        for match in _NONFINITE_WORD_RE.finditer(line):
            if len(findings) >= budget:
                break
            findings.append(
                ResultFinding(
                    source=rel.as_posix(),
                    location=f"{location_prefix} line {line_idx}",
                    value=match.group(0),
                    reason="non-finite numeric value",
                )
            )
    return findings


def _line_denies_nonfinite(line: str) -> bool:
    lowered = line.casefold()
    denial_patterns = (
        r"\bno\s+(?:nan|inf|infinity|non-finite|nonfinite)\b",
        r"\bwithout\s+(?:nan|inf|infinity|non-finite|nonfinite)\b",
        r"\b(?:has|have|contains?|any)\s+(?:nan|inf|infinity|non-finite|nonfinite)\s*[:?=]\s*false\b",
        r"\b(?:nan|inf|infinity|non-finite|nonfinite)\s*(?:present|detected)?\s*[:?=]\s*false\b",
    )
    return any(re.search(pattern, lowered) for pattern in denial_patterns)


def _inspect_value(value: Any, rel: Path, location: str) -> list[ResultFinding]:
    text = str(value).strip()
    if not text:
        return []

    normalized = (
        text.replace(",", "")
        .replace("−", "-")
        .replace("∞", "inf")
        .strip()
        .casefold()
    )
    if normalized in {"nan", "+nan", "-nan", "inf", "+inf", "-inf", "infinity", "+infinity", "-infinity"}:
        return [
            ResultFinding(
                source=rel.as_posix(),
                location=location,
                value=text,
                reason="non-finite numeric value",
            )
        ]

    try:
        numeric = float(normalized)
    except ValueError:
        return []
    if not math.isfinite(numeric):
        return [
            ResultFinding(
                source=rel.as_posix(),
                location=location,
                value=text,
                reason="non-finite numeric value",
            )
        ]
    return []


def _inspect_negative_convergence(
    value: Any,
    rel: Path,
    location: str,
    header: str,
) -> list[ResultFinding]:
    context = f"{rel.name} {header}".casefold()
    if "convergence" not in context or "rate" not in context:
        return []
    try:
        numeric = float(str(value).replace(",", "").replace("−", "-").strip())
    except ValueError:
        return []
    if numeric < 0:
        return [
            ResultFinding(
                source=rel.as_posix(),
                location=location,
                value=str(value),
                reason="negative convergence rate",
            )
        ]
    return []
