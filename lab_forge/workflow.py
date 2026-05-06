"""
Shared workflow helpers used by both the terminal CLI and the Gradio UI.

Keeps topic→task wrapping, per-run sandbox preparation, and workspace file
summarisation in one place so the two entry points behave identically.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_SANDBOX_ROOT = Path("/tmp/lab_forge_workspace")

# A single execute_code call is hard-capped at 300 s by the sandbox. The agent
# should plan well below that ceiling so a transient slowdown doesn't cost a
# whole turn. These targets are referenced inside the machine-profile prompt.
EXECUTE_CODE_TARGET_SECONDS = 60
EXECUTE_CODE_TIMEOUT_SECONDS = 300


def detect_machine_profile() -> dict[str, Any]:
    """Probe the current machine so the agent can pick a feasible experiment scale.

    Returns a dict with whatever could be discovered. Each field is independent
    and the function never raises — missing tools (no ``psutil``, no
    ``nvidia-smi``) just leave the corresponding entry as ``None``.
    """
    profile: dict[str, Any] = {
        "cpu_logical": os.cpu_count(),
        "cpu_physical": None,
        "ram_gb": None,
        "gpus": [],
        "disk_free_gb": None,
        "platform": None,
    }
    try:
        import platform as _platform
        profile["platform"] = f"{_platform.system()} {_platform.release()} ({_platform.machine()})"
    except Exception:
        pass

    try:
        import psutil  # type: ignore
        profile["cpu_physical"] = psutil.cpu_count(logical=False) or profile["cpu_logical"]
        profile["ram_gb"] = round(psutil.virtual_memory().total / (1024 ** 3), 1)
    except Exception:
        # psutil missing or call failed — leave physical core / RAM as None.
        pass

    # GPU detection via nvidia-smi. Quick timeout because if it's missing we
    # don't want to block run startup.
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=3,
        )
        if result.returncode == 0:
            for line in result.stdout.strip().splitlines():
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 2 and parts[0]:
                    try:
                        mem_mb = int(parts[1])
                    except ValueError:
                        mem_mb = 0
                    profile["gpus"].append({"name": parts[0], "memory_mb": mem_mb})
    except (FileNotFoundError, subprocess.TimeoutExpired, subprocess.SubprocessError):
        pass

    try:
        usage = shutil.disk_usage(str(DEFAULT_SANDBOX_ROOT.parent))
        profile["disk_free_gb"] = round(usage.free / (1024 ** 3), 1)
    except Exception:
        pass

    return profile


def format_machine_profile_for_prompt(profile: dict[str, Any]) -> str:
    """Render a probed machine profile into a compact prompt block with concrete
    experiment-scale guidance.

    The guidance is intentionally specific (named datasets, batch sizes, epoch
    counts, runtime targets) so the agent doesn't fall back to "default"
    setups like full CIFAR-10 + many epochs that always time out on CPU.
    """
    has_gpu = bool(profile.get("gpus"))
    cpu_logical = profile.get("cpu_logical") or "?"
    cpu_physical = profile.get("cpu_physical")
    ram_gb = profile.get("ram_gb")
    disk_gb = profile.get("disk_free_gb")
    platform = profile.get("platform")

    lines: list[str] = []
    lines.append(
        "EXECUTION ENVIRONMENT (probed at run start — your experiment scale "
        "MUST fit inside these limits, not a textbook default):"
    )
    if platform:
        lines.append(f"  - Platform: {platform}")
    cpu_line = f"  - CPU: {cpu_logical} logical cores"
    if cpu_physical and cpu_physical != cpu_logical:
        cpu_line += f" ({cpu_physical} physical)"
    lines.append(cpu_line)
    if ram_gb is not None:
        lines.append(f"  - RAM: {ram_gb} GB")
    if has_gpu:
        for idx, gpu in enumerate(profile["gpus"]):
            lines.append(f"  - GPU{idx}: {gpu['name']} ({gpu['memory_mb']} MB)")
    else:
        lines.append("  - GPU: none detected (CPU-only execution)")
    if disk_gb is not None:
        lines.append(f"  - Free disk: {disk_gb} GB at the sandbox root")
    lines.append(
        f"  - execute_code hard timeout: {EXECUTE_CODE_TIMEOUT_SECONDS}s per call "
        f"(target each call to finish in ~{EXECUTE_CODE_TARGET_SECONDS}s)"
    )

    lines.append("")
    if has_gpu:
        gpu_mb = max((g.get("memory_mb") or 0) for g in profile["gpus"])
        lines.append("Scale guidance (GPU available):")
        if gpu_mb >= 16000:
            lines.append("  - Mid-size datasets are OK: full CIFAR-10, GLUE tasks, ImageNet subsets ≤ 50 k images.")
            lines.append("  - Models up to ~300 M params, batch ≤ 256, epochs ≤ 20.")
        elif gpu_mb >= 6000:
            lines.append("  - Small/medium datasets: CIFAR-10 (full), Tiny-ImageNet, MNIST/Fashion-MNIST, GLUE single-task.")
            lines.append("  - Models up to ~50 M params, batch ≤ 128, epochs ≤ 10.")
        else:
            lines.append("  - Treat as low-VRAM: prefer MNIST / sklearn / synthetic data, models ≤ 5 M params, batch ≤ 32.")
        lines.append("  - Always start with a 1-epoch smoke run; only scale up after it succeeds.")
    else:
        lines.append("Scale guidance (CPU-only — this is the strict path):")
        lines.append("  - Use SMALL datasets only:")
        lines.append("      • sklearn.datasets (load_iris, load_digits, fetch_20newsgroups subset)")
        lines.append("      • MNIST or Fashion-MNIST")
        lines.append("      • CIFAR-10 SUBSET (≤ 2 000 images), NOT the full 60 k set")
        lines.append("      • Synthetic data via sklearn.make_classification / make_blobs / numpy.random")
        lines.append("  - Lightweight models only:")
        lines.append("      • Linear / logistic regression, decision trees, SVM with linear kernel")
        lines.append("      • Small MLPs (1–2 hidden layers, ≤ 256 units)")
        lines.append("      • Tiny ConvNets (≤ 2 conv blocks, ≤ 1 M params)")
        lines.append("  - batch_size ≤ 64, epochs ≤ 3, total samples per run ≤ 5 000.")
        lines.append("  - DO NOT download multi-GB datasets (full ImageNet, full CIFAR-10 = 170 MB; prefer sklearn / torchvision MNIST).")
        lines.append(
            f"  - First execute_code MUST be a smoke run that finishes in < 30 s. "
            "If it doesn't, halve the dataset/epochs/model size BEFORE retrying — "
            "do NOT re-submit the same configuration."
        )
        lines.append(
            "  - If you see a 300 s timeout, that is a SIGNAL that the configuration "
            "is too large for this machine; shrink it. Do not interpret it as a flaky run."
        )

    lines.append("")
    lines.append(
        "If you need to verify any of the numbers above (or check installed "
        "packages, GPU availability inside Python, etc.), call execute_code "
        "with a short probe before scaling up."
    )
    return "\n".join(lines)

# File categories used when summarising a run workspace for the user.
_CODE_EXTS = {".py", ".sh", ".ipynb"}
_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".svg", ".gif", ".bmp", ".tiff"}
_DATA_EXTS = {
    ".csv", ".tsv", ".parquet", ".npy", ".npz", ".pkl", ".pickle",
    ".h5", ".hdf5", ".xlsx", ".xls", ".json", ".jsonl", ".feather", ".arrow",
}
_REPORT_EXTS = {".md", ".txt", ".pdf", ".tex", ".bib", ".rst"}
_LOG_EXTS = {".log"}


# ---------------------------------------------------------------------------
# Topic mode
# ---------------------------------------------------------------------------
#
# LabForge is a *lightweight* research agent: it can run smoke-scale
# experiments and produce literature surveys, but it cannot actually invent
# a new algorithm or train a serious model end-to-end. The mode is now
# always picked explicitly by the user (UI radio / CLI ``--mode``):
#   - "survey":     no experiments. Just literature search → fulltext reads
#                   → generate_report.
#   - "experiment": the 5-phase loop with code execution.
#
# There is intentionally no auto-classifier; previous keyword-driven routing
# misfired on common phrasings ("做实验分析参数的作用" was misclassified as
# survey because the keyword list omitted "实验"). Forcing the user to
# choose keeps the contract honest.


def _build_experiment_task_desc(topic: str, detail: str) -> str:
    desc = (
        f"Research Topic: {topic}\n\n"
        "[MODE: EXPERIMENT — lightweight comparison study]\n\n"
        "Conduct a systematic computational investigation on this topic.\n\n"
        "Your workflow:\n"
        "1. Literature Review: search for key methods, baselines, datasets, and evaluation metrics.\n"
        "2. Plan: decide which methods to compare and how to evaluate them — "
        "pick a scale that fits the EXECUTION ENVIRONMENT block below.\n"
        "3. Implement: write and run code incrementally in Python.\n"
        "4. Analyze: save tables as CSV and charts as PNG/SVG, then verify the findings.\n"
        "5. Report: call generate_report with verified references plus the real figure/table files "
        "so the final PDF can reuse them directly.\n"
    )
    if detail.strip():
        desc += f"\nAdditional requirements:\n{detail.strip()}\n"
    return desc


def _build_survey_task_desc(topic: str, detail: str) -> str:
    """Survey-only prompt: NO experiments, NO custom-algorithm implementation.

    Used when the topic asks for novelty (which the agent cannot honestly
    deliver) or when intent is ambiguous (default safe path).
    """
    desc = (
        f"Research Topic: {topic}\n\n"
        "[MODE: SURVEY ONLY — DO NOT run experiments, DO NOT implement new algorithms]\n\n"
        "This is a literature-survey task. Your job is to map out what the "
        "research community has already proposed for this topic and write a "
        "synthesised review. You are NOT asked to invent a new method, NOT "
        "asked to beat baselines, and NOT asked to run training experiments.\n\n"
        "Your workflow:\n"
        "1. Identify candidate methods via search_literature (3-6 well-formed queries; "
        "respect the search quota).\n"
        "2. Read 5–10 representative papers in full via read_paper_fulltext "
        "to ground your synthesis in real evidence.\n"
        "3. (Optional) Use lookup_paper_code to record official code links.\n"
        "4. Write the survey by calling generate_report. The report must:\n"
        "   - cite ≥5 verified references that you actually opened with read_paper_fulltext "
        "or that came back from search_literature this run;\n"
        "   - clearly distinguish what each cited paper claims vs your own synthesis;\n"
        "   - explicitly state when the literature does NOT settle a question, "
        "rather than papering over it with invented numbers.\n"
        "5. submit_result with a short summary of the survey's key findings.\n\n"
        "Hard rules — violating any one is a task failure:\n"
        "  • Do NOT call execute_code or execute_bash for ML training / fitting / scoring.\n"
        "  • Do NOT fabricate metrics, comparison tables, or experimental numbers. "
        "If the topic seems to demand them, your job is to surface what the literature "
        "already reports, not to generate your own.\n"
        "  • Every numeric claim in the report must be quoted from a paper you actually "
        "opened (cite the paper) — do not transcribe it as if you measured it.\n"
        "  • Do NOT propose 'your own' algorithm or method. If the user asked for one, "
        "explain in the report that this agent only delivers literature surveys, then "
        "summarise what existing methods already address the gap.\n"
    )
    if detail.strip():
        desc += f"\nAdditional requirements:\n{detail.strip()}\n"
    return desc


def topic_to_task(
    topic: str,
    detail: str = "",
    mode: str = "survey",
) -> tuple[str, str, str]:
    """Convert a research topic + optional detail into a structured task prompt.

    Returns ``(task_description, expected_output, data_description)``.

    ``mode`` must be ``"survey"`` or ``"experiment"`` — the caller (UI
    radio / CLI ``--mode``) is responsible for picking it. Any other
    value silently falls back to ``"survey"`` (the safer branch: it can't
    fabricate experimental numbers).

    Both branches embed the probed machine profile so the agent picks a
    feasible configuration on step 0.
    """
    if mode not in ("survey", "experiment"):
        mode = "survey"
    if mode == "experiment":
        task_desc = _build_experiment_task_desc(topic, detail)
    else:
        task_desc = _build_survey_task_desc(topic, detail)

    try:
        machine_profile = detect_machine_profile()
        machine_block = format_machine_profile_for_prompt(machine_profile)
        task_desc += f"\n{machine_block}\n"
        logger.info(
            "topic_to_task: mode=%s, machine cpu=%s ram=%s gpus=%d disk=%s",
            mode,
            machine_profile.get("cpu_logical"),
            machine_profile.get("ram_gb"),
            len(machine_profile.get("gpus") or []),
            machine_profile.get("disk_free_gb"),
        )
    except Exception:
        logger.exception("Machine profile detection failed; continuing without it")

    if mode == "experiment":
        expected_output = (
            "A complete research package including: "
            "1) comparison tables, "
            "2) charts/visualizations, "
            "3) a structured report with verified references, "
            "4) a short summary of key findings."
        )
        data_description = (
            "Prefer publicly available datasets such as sklearn, torchvision, or HuggingFace datasets. "
            "Install any missing packages when needed. Pick dataset size based on the "
            "EXECUTION ENVIRONMENT block in the task description above."
        )
    else:  # survey
        expected_output = (
            "A literature survey delivered as research_report.md, citing ≥5 verified "
            "references actually opened during this run, with clear separation of "
            "what each paper claims vs the agent's synthesis. NO experimental tables "
            "or fabricated metrics."
        )
        data_description = (
            "No experimental data needed. The 'data' for this run is the body of "
            "literature reachable via search_literature + read_paper_fulltext. "
            "If a paper PDF is uploaded by the user, prioritise reading it first."
        )
    return task_desc, expected_output, data_description


def prepare_run_sandbox(task_id: str, sandbox_root: Path | None = None) -> Path:
    """Create a clean per-task sandbox directory and return it."""
    root = Path(sandbox_root) if sandbox_root else DEFAULT_SANDBOX_ROOT
    root.mkdir(parents=True, exist_ok=True)
    run_dir = root / task_id
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _categorise(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in _CODE_EXTS:
        return "code"
    if suffix in _IMAGE_EXTS:
        return "image"
    if suffix in _DATA_EXTS:
        return "data"
    if suffix in _LOG_EXTS:
        return "log"
    if suffix in _REPORT_EXTS:
        return "report"
    return "other"


def summarise_workspace(run_dir: Path) -> dict[str, list[Path]]:
    """Group files in ``run_dir`` by category for end-of-run reporting."""
    buckets: dict[str, list[Path]] = {
        "code": [], "image": [], "data": [], "report": [], "log": [], "other": [],
    }
    if not run_dir.exists():
        return buckets
    for candidate in sorted(run_dir.rglob("*")):
        if not candidate.is_file():
            continue
        buckets[_categorise(candidate)].append(candidate)
    return buckets


def format_workspace_summary(run_dir: Path, *, max_per_bucket: int = 10) -> str:
    """Plain-text summary of a run's workspace suitable for the terminal."""
    buckets = summarise_workspace(run_dir)
    lines: list[str] = []
    lines.append(f"Workspace: {run_dir}")
    headers = [
        ("code", "代码与脚本 (execute_code / execute_bash 持久化后的真实源代码)"),
        ("log", "执行日志 (stdout/stderr + 执行的脚本路径)"),
        ("data", "数据与结果文件 (由实验程序生成)"),
        ("image", "图表"),
        ("report", "报告"),
        ("other", "其他"),
    ]
    for key, header in headers:
        files = buckets[key]
        if not files:
            continue
        lines.append(f"\n{header}:")
        for path in files[:max_per_bucket]:
            lines.append(f"  - {path}")
        remaining = len(files) - max_per_bucket
        if remaining > 0:
            lines.append(f"  ... 还有 {remaining} 个文件")
    return "\n".join(lines)
