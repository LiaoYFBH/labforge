# lab-forge

> 面向机器学习方向的科研 Agent，自动完成文献调研与轻量实验验证。
> [English](README.en.md) ｜ 中文

## 概述

lab-forge 接收一个研究主题，自动判断走综述模式还是实验模式，并交付一份带引用与可复现产物的报告。

- **综述模式**：检索 arXiv，读取全文，撰写结构化综述。
- **实验模式**：五阶段轻量流程 —— 文献检索、demo 实现、结果汇总、报告撰写、审稿一致性检查。
- **审稿环节**：独立的 reviewer LLM 在两个 checkpoint 上对中间产物进行一致性校验。
- **沙箱执行**：subprocess / docker 后端，单次执行 300 s 上限。

模式判断逻辑见 [`lab_forge/workflow.py`](lab_forge/workflow.py) 的 `classify_topic_mode()`。

## 快速开始

```bash
git clone <this-repo>
cd lab-forge
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # 填写 API key
```

### 命令行

```bash
python -m lab_forge run --config configs/default.yaml \
    --topic "对图神经网络在长程依赖上的最新方法做综述"

python -m lab_forge interactive --config configs/default.yaml
```

### Web UI

```bash
python ui-new.py        # 默认 0.0.0.0:7861
```

星河社区部署使用 `Gradio.app.py` 作为入口。

## 配置

YAML 中显式填写的 `model` / `base_url` 优先级高于环境变量；留空时回退到环境变量，再回退到内置默认值。API key 仅从环境变量读取。

默认 reviewer 走星河社区 DeepSeek-V3，可在 `configs/default.yaml` 或 UI 中替换为同套餐其它现役模型（ERNIE 4.5 Turbo、Qwen3 Coder、Kimi K2 等）。

## 目录结构

```
lab-forge/
├── lab_forge/              # 核心包
│   ├── agent.py            # LangChain agent 入口
│   ├── workflow.py         # 五阶段实验流程与主题路由
│   ├── reviewer.py         # 独立 reviewer
│   ├── sandbox.py          # subprocess / docker 沙箱
│   ├── langchain_tools.py  # 检索 / 阅读 / 执行 / 报告工具
│   ├── tools/
│   └── web.py              # Gradio 后端
├── paper_forge/            # 论文导出后端（亦作为独立仓库发布）
├── ui-new.py               # 主 UI
├── configs/
└── tests/
```

## 协议

待定。
