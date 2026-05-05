"""
Code execution sandbox.

Supports two backends:
  - subprocess: run code in a local subprocess (simple, for dev/debug)
  - docker: run code in an isolated Docker container (for production/safety)

Features:
  - Auto-dependency installation: detects ImportError and installs missing packages
  - Inspired by AutoResearchClaw's self-healing execution pattern
"""

from __future__ import annotations

import itertools
import logging
import os
import re
import subprocess
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from .config import SandboxConfig

logger = logging.getLogger(__name__)


_IMPORT_TO_PIP = {
    "sklearn": "scikit-learn",
    "cv2": "opencv-python",
    "PIL": "Pillow",
    "skimage": "scikit-image",
    "yaml": "pyyaml",
    "bs4": "beautifulsoup4",
    "attr": "attrs",
    "dotenv": "python-dotenv",
    "gi": "PyGObject",
}


@dataclass
class ExecutionResult:
    """Result of a code execution."""

    stdout: str
    stderr: str
    exit_code: int
    timed_out: bool = False


    script_path: str | None = None
    log_path: str | None = None

    @property
    def success(self) -> bool:
        return self.exit_code == 0 and not self.timed_out

    @property
    def output(self) -> str:
        """Combined output for the agent to see."""
        parts = []
        if self.stdout:
            parts.append(self.stdout)
        if self.stderr:
            parts.append(f"[STDERR]\n{self.stderr}")
        if self.timed_out:
            parts.append(f"[TIMEOUT] Execution exceeded {self.exit_code}s limit.")
        if self.script_path:
            parts.append(f"[Script saved at] {self.script_path}")
        if self.log_path:
            parts.append(f"[Execution log] {self.log_path}")
        if not parts:
            parts.append("[No output]")
        return "\n".join(parts)


class Sandbox:
    """Code execution sandbox."""

    def __init__(self, config: SandboxConfig):
        self.config = config
        self.working_dir = Path(config.working_dir)
        self.working_dir.mkdir(parents=True, exist_ok=True)

        self.python_cmd = config.python_executable or self._detect_python()




        self.logs_dir = self.working_dir / "logs"
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self._exec_counter = itertools.count(1)

    def _detect_python(self) -> str:
        """Find a Python interpreter, preferring the one running the agent.

        Probe order:
          1. ``sys.executable`` — the interpreter that imported lab_forge.
             If the user installed the project into a venv/conda env, this is
             almost always the correct one and will already have the project's
             scientific dependencies. Listed first so the open-source default
             (no ``python_executable`` in YAML) "just works".
          2. PATH lookups (``python3``/``python``).
          3. ``/usr/bin/python3`` as a last-resort system fallback.

        For each candidate we still verify that ``numpy``/``sklearn`` import
        cleanly, but we do NOT require it — if nothing has them we fall back
        to ``sys.executable`` so ``execute_code`` can still install missing
        packages on the fly via ``execute_bash`` + ``pip install``.
        """
        import shutil
        import sys
        candidates: list[str] = []
        if sys.executable:
            candidates.append(sys.executable)
        for name in ("python3", "python"):
            resolved = shutil.which(name)
            if resolved and resolved not in candidates:
                candidates.append(resolved)
        if "/usr/bin/python3" not in candidates:
            candidates.append("/usr/bin/python3")

        for py in candidates:
            if not py:
                continue
            try:
                result = subprocess.run(
                    [py, "-c", "import numpy; import sklearn; print('ok')"],
                    capture_output=True, text=True, timeout=10,
                )
                if result.returncode == 0:
                    logger.info("Detected Python with scientific packages: %s", py)
                    return py
            except Exception:
                continue

        fallback = sys.executable or "python3"
        logger.warning(
            "No Python with numpy+sklearn found; falling back to %s. "
            "execute_code can still install packages on the fly.",
            fallback,
        )
        return fallback

    def execute_code(
        self, code: str, language: str = "python", auto_install: bool = True
    ) -> ExecutionResult:
        """
        Execute code and return the result.

        If auto_install is True and an ImportError is detected, automatically
        install the missing package and retry once.
        """
        if self.config.backend == "subprocess":
            result = self._execute_subprocess(code, language)
        elif self.config.backend == "docker":
            result = self._execute_docker(code, language)
        else:
            raise ValueError(f"Unknown sandbox backend: {self.config.backend}")


        if auto_install and not result.success and language == "python":
            missing_pkg = self._detect_missing_import(result.stderr)
            if missing_pkg:
                logger.info("Auto-installing missing package: %s", missing_pkg)
                install_result = self.execute_bash(
                    f"{self.python_cmd} -m pip install {missing_pkg}"
                )
                if install_result.success:
                    logger.info("Package %s installed, retrying code", missing_pkg)
                    if self.config.backend == "subprocess":
                        result = self._execute_subprocess(code, language)
                    else:
                        result = self._execute_docker(code, language)

                    if result.success:
                        result.stdout = (
                            f"[Auto-installed: {missing_pkg}]\n" + result.stdout
                        )

        return result

    def _detect_missing_import(self, stderr: str) -> str | None:
        """Detect missing package from ImportError in stderr."""
        if not stderr:
            return None




        patterns = [
            r"No module named ['\"](\w+)['\"]",
            r"No module named (\w+)",
            r"ModuleNotFoundError:.*['\"](\w+)['\"]",
        ]
        for pattern in patterns:
            match = re.search(pattern, stderr)
            if match:
                import_name = match.group(1)

                pip_name = _IMPORT_TO_PIP.get(import_name, import_name)
                return pip_name
        return None

    def execute_bash(self, command: str) -> ExecutionResult:
        """Execute a bash command."""
        script_path, log_path = self._next_log_paths(kind="bash", suffix=".sh")


        script_path.write_text(command, encoding="utf-8")
        cmd = ["bash", "-c", command]
        result = self._run_command(cmd, cwd=str(self.working_dir))
        result.script_path = str(script_path)
        result.log_path = str(log_path)
        self._write_run_log(
            log_path=log_path,
            tool="execute_bash",
            command=cmd,
            cwd=str(self.working_dir),
            script_path=script_path,
            script_body=command,
            result=result,
        )
        return result








    def _next_log_paths(self, kind: str, suffix: str) -> tuple[Path, Path]:
        idx = next(self._exec_counter)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        stem = f"step_{idx:03d}_{ts}_{kind}"
        return self.logs_dir / f"{stem}{suffix}", self.logs_dir / f"{stem}.log"

    def _write_run_log(
        self,
        *,
        log_path: Path,
        tool: str,
        command: list[str],
        cwd: str,
        script_path: Path,
        script_body: str,
        result: "ExecutionResult",
    ) -> None:
        """Write a human-readable + machine-parseable execution log."""
        try:
            lines: list[str] = []
            lines.append("== LabForge Execution Log ==")
            lines.append(f"timestamp        : {datetime.now().isoformat(timespec='seconds')}")
            lines.append(f"tool             : {tool}")
            lines.append(f"python_executable: {self.python_cmd}")
            lines.append(f"cwd              : {cwd}")
            lines.append(f"script_path      : {script_path}")
            lines.append(f"command          : {' '.join(command)}")
            lines.append(f"exit_code        : {result.exit_code}")
            lines.append(f"timed_out        : {result.timed_out}")
            lines.append("")
            lines.append("---- SCRIPT ----")
            lines.append(script_body.rstrip("\n"))
            lines.append("---- STDOUT ----")
            lines.append((result.stdout or "").rstrip("\n"))
            lines.append("---- STDERR ----")
            lines.append((result.stderr or "").rstrip("\n"))
            lines.append("---- END ----")
            log_path.write_text("\n".join(lines), encoding="utf-8")
        except Exception:
            logger.exception("Failed to write execution log: %s", log_path)

    def _execute_subprocess(self, code: str, language: str) -> ExecutionResult:
        """Execute code via local subprocess."""
        suffix = ".py" if language == "python" else ".sh"
        cmd_prefix = [self.python_cmd] if language == "python" else ["bash"]




        script_path, log_path = self._next_log_paths(kind="code", suffix=suffix)
        script_path.write_text(code, encoding="utf-8")

        result = self._run_command(
            cmd_prefix + [str(script_path)],
            cwd=str(self.working_dir),
        )
        result.script_path = str(script_path)
        result.log_path = str(log_path)
        self._write_run_log(
            log_path=log_path,
            tool="execute_code" if language == "python" else "execute_script",
            command=cmd_prefix + [str(script_path)],
            cwd=str(self.working_dir),
            script_path=script_path,
            script_body=code,
            result=result,
        )
        return result

    def _execute_docker(self, code: str, language: str) -> ExecutionResult:
        """Execute code in a Docker container."""
        suffix = ".py" if language == "python" else ".sh"
        cmd_inside = f"python /workspace/code{suffix}" if language == "python" else f"bash /workspace/code{suffix}"

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=suffix, dir=str(self.working_dir), delete=False
        ) as f:
            f.write(code)
            f.flush()
            temp_path = f.name

        try:
            return self._run_command(
                [
                    "docker", "run", "--rm",
                    "--network=none",
                    "-v", f"{self.working_dir}:/workspace",
                    "-v", f"{temp_path}:/workspace/code{suffix}",
                    "-w", "/workspace",
                    self.config.docker_image,
                    "bash", "-c", cmd_inside,
                ],
                cwd=str(self.working_dir),
            )
        finally:
            os.unlink(temp_path)

    def _run_command(self, cmd: list[str], cwd: str) -> ExecutionResult:
        """Run a command with timeout and output capture."""
        try:





            proc = subprocess.run(
                cmd,
                cwd=cwd,
                capture_output=True,
                timeout=self.config.timeout,
            )
            stdout = proc.stdout.decode("utf-8", errors="replace")[: self.config.max_output_length]
            stderr = proc.stderr.decode("utf-8", errors="replace")[: self.config.max_output_length]
            return ExecutionResult(
                stdout=stdout,
                stderr=stderr,
                exit_code=proc.returncode,
            )
        except subprocess.TimeoutExpired:
            return ExecutionResult(
                stdout="",
                stderr=f"Execution timed out after {self.config.timeout}s",
                exit_code=-1,
                timed_out=True,
            )
        except Exception as e:
            return ExecutionResult(
                stdout="",
                stderr=str(e),
                exit_code=-1,
            )

    def write_file(self, path: str, content: str) -> str:
        """Write a file to the sandbox working directory."""
        full_path = self.working_dir / path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_text(content)
        return f"File written to {full_path}"

    def read_file(self, path: str) -> str:
        """Read a file from the sandbox working directory."""
        full_path = self.working_dir / path
        if not full_path.exists():
            return f"Error: File not found: {full_path}"




        return full_path.read_text(encoding="utf-8", errors="replace")[
            : self.config.max_output_length
        ]

    def list_files(self, path: str = ".") -> str:
        """List files in a directory."""
        full_path = self.working_dir / path
        if not full_path.exists():
            return f"Error: Directory not found: {full_path}"
        entries = sorted(full_path.iterdir())
        lines = []
        for e in entries:
            prefix = "d " if e.is_dir() else "f "
            lines.append(prefix + e.name)
        return "\n".join(lines) if lines else "(empty directory)"
