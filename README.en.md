# lab-forge

> A research agent for machine-learning topics that performs literature surveys and lightweight experimental validation.
> English ｜ [中文](README.md)

## Overview

Given a research topic, lab-forge automatically routes to either survey mode or experiment mode and produces a report with citations and reproducible artifacts.

- **Survey mode** — searches arXiv / Google Scholar, reads full text, and writes a structured survey.
- **Experiment mode** — five-stage pipeline: literature retrieval, demo implementation, result aggregation, report writing, and reviewer-side consistency checks.
- **Reviewer** — a separate reviewer LLM validates intermediate artifacts at two checkpoints.
- **Sandbox** — subprocess / docker backend with a 300 s per-call cap.

Mode-routing logic lives in `classify_topic_mode()` in [`lab_forge/workflow.py`](lab_forge/workflow.py).

## Environment recommendations

Linux is recommended. On Windows, use WSL2 with Ubuntu to install and run this project. PDF / LaTeX paper export requires a local LaTeX distribution such as TeX Live, and at least one of `latexmk` / `xelatex` / `pdflatex` must be available from the command line.

For Ubuntu / WSL2:

```bash
sudo apt-get update
sudo apt-get install -y latexmk texlive-xetex texlive-latex-recommended \
    texlive-latex-extra texlive-fonts-recommended texlive-lang-chinese \
    fonts-noto-cjk
```

## Quick start

```bash
git clone <this-repo>
cd lab-forge
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in an API key
```

### CLI

```bash
python -m lab_forge run --config configs/default.yaml \
    --topic "Survey recent GNN methods on long-range dependencies"

python -m lab_forge interactive --config configs/default.yaml
```

### Web UI

```bash
python ui-new.py        # default 0.0.0.0:7861
```

AI Studio (星河社区) deployment uses `Gradio.app.py` as the entry point.

## Configuration

`model` / `base_url` written explicitly in YAML take priority over env vars; empty fields fall back to env vars and finally to the built-in defaults. API keys are read from env vars only.

The default reviewer is DeepSeek-V3 on AI Studio; it can be replaced via `configs/default.yaml` or the UI with other active models on the same plan (ERNIE 4.5 Turbo, Qwen3 Coder, Kimi K2, ...).

## Layout

```
lab-forge/
├── lab_forge/              # core package
│   ├── agent.py            # LangChain agent entry
│   ├── workflow.py         # five-stage pipeline & topic routing
│   ├── reviewer.py         # separate reviewer
│   ├── sandbox.py          # subprocess / docker sandbox
│   ├── langchain_tools.py  # search / read / execute / report tools
│   ├── tools/
│   └── web.py              # Gradio backend
├── paper_forge/            # paper-export backend (also released as a standalone repo)
├── ui-new.py               # primary UI
├── configs/
└── tests/
```

## License

TBD.
