# paper-forge

> 将草稿、PDF 或图片转换为会议风格学术论文 PDF 的工具。
> [English](README.en.md) ｜ 中文

## 概述

paper-forge 提供一条从原始素材到学术论文 PDF 的端到端流程：

1. **输入解析**：接收 Markdown / 纯文本，或经由 PaddleOCR-VL 将 PDF / 图片解析为 Markdown。
2. **结构重组**：调用 OpenAI 兼容大模型将内容整理为标题、摘要、章节与图表说明组成的论文骨架。
3. **模板渲染**：通过 Jinja 将骨架渲染为 LaTeX 源码，内置 IEEE、NeurIPS、ICML、ACL、ACM SIGCONF 与通用 article 模板。
4. **PDF 编译**：本地具备 `pdflatex` / `xelatex` 时直接编译；否则保留 `.tex` 供 Overleaf 等环境使用。

paper-forge 既可独立运行，也可作为 lab-forge 等上游 Agent 的论文导出后端。

## 快速开始

```bash
git clone <this-repo>
cd paper_forge
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python app.py            # 默认 0.0.0.0:7860
```

星河社区部署同样使用 `app.py` 作为入口；如平台要求固定文件名，可将入口指向 `app:main`。

打开 UI 后：

1. 在「Model & OCR」中选择预设或填入自定义 OpenAI 兼容接口。
2. 上传 PDF / 图片，或直接粘贴 Markdown。
3. 选择模板（默认 `article`），生成 PDF。

## 配置

API key 仅从环境变量或 UI 表单读取，不写入 YAML：

| 来源 | 环境变量 |
|---|---|
| 星河社区 (AI Studio) | `AI_STUDIO_API_KEY` |
| MiniMax | `MINIMAX_API_KEY` |
| 通用 OpenAI 兼容接口 | `OPENAI_API_KEY` 或 `API_KEY` |

UI 中填入的 key 仅在当前 session 生效。

## 目录结构

```
paper_forge/
├── app.py                  # Gradio UI
├── paper_forge/            # 核心库
│   ├── ocr_client.py       # PaddleOCR 调用封装
│   ├── llm_client.py       # OpenAI 兼容客户端
│   ├── paper_writer.py     # 文档至 paper bundle 的结构化
│   ├── latex_renderer.py   # bundle 至 .tex
│   ├── pdf_compiler.py     # .tex 至 .pdf
│   └── templates_catalog.py
├── templates/              # Jinja LaTeX 模板
├── configs/
└── tests/
```

## 协议

待定。
