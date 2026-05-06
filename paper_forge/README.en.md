# paper-forge

> A tool that converts drafts, PDFs, or images into conference-style academic paper PDFs.
> English ｜ [中文](README.md)

## Overview

paper-forge provides an end-to-end pipeline from raw material to academic PDF:

1. **Input parsing** — accepts Markdown / plain text, or parses PDFs / images into Markdown via PaddleOCR-VL.
2. **Structural rewriting** — calls an OpenAI-compatible LLM to organize the content into a paper skeleton of title, abstract, sections, and figure captions.
3. **Template rendering** — renders the skeleton to LaTeX source via Jinja. Built-in templates: IEEE, NeurIPS, ICML, ACL, ACM SIGCONF, and a generic `article`.
4. **PDF compilation** — compiles directly when local `pdflatex` / `xelatex` is available; otherwise emits `.tex` for use in Overleaf or similar environments.

paper-forge runs standalone and also serves as the paper-export backend for upstream agents such as lab-forge.

## Environment recommendations

Linux is recommended. On Windows, use WSL2 with Ubuntu to install and run this project. PDF / LaTeX compilation requires a local LaTeX distribution such as TeX Live, and at least one of `latexmk` / `xelatex` / `pdflatex` must be available from the command line.

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
cd paper_forge
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python app.py            # default 0.0.0.0:7860
```

AI Studio (星河社区) deployment uses `Gradio.app.py` as the entry point.

Once the UI is open:

1. Pick a preset under **Model Settings** (recommended: `星河社区 · ERNIE 4.5 Turbo 128K`) or fill in a custom OpenAI-compatible endpoint.
2. Upload a PDF / image, or paste Markdown directly.
3. Pick a template (default: `article`) and generate the PDF.

## Configuration

API keys are read only from env vars or the UI form — they are never persisted to YAML:

| Provider | Env var |
|---|---|
| AI Studio (星河社区) | `AI_STUDIO_API_KEY` |
| MiniMax | `MINIMAX_API_KEY` |
| Generic OpenAI-compatible | `OPENAI_API_KEY` or `API_KEY` |

Keys entered in the UI live only in the current session.

## Layout

```
paper_forge/
├── app.py                  # Gradio UI
├── paper_forge/            # core library
│   ├── ocr_client.py       # PaddleOCR wrapper
│   ├── llm_client.py       # OpenAI-compatible client
│   ├── paper_writer.py     # document → structured paper bundle
│   ├── latex_renderer.py   # bundle → .tex
│   ├── pdf_compiler.py     # .tex → .pdf
│   └── templates_catalog.py
├── templates/              # Jinja LaTeX templates
├── configs/
└── tests/
```

## License

TBD.
