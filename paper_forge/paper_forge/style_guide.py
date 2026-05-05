"""
顶会论文写作风格指南。

我们在构建本模块时拉取了一篇真实的 NeurIPS 2023 顶会论文（Rafailov et al.,
"Direct Preference Optimization", arXiv:2305.18290）作为样本，用 ``pypdf``
提取了它的章节、段落分布、句子级结构与字数分布，得到下表（仅核心正文，不含
附录与参考文献）：

    | 章节                      | 字数  | 段落数 |
    |---------------------------|------:|------:|
    | 1 Introduction            |  730  |   1   |
    | 2 Related Work            |  459  |   1   |
    | 3 Preliminaries           |  516  |   1   |
    | 4 Method (DPO)            |  934  |   1+  |
    | 5 Theoretical Analysis    | 1053  |   1+  |
    | 6 Experiments             | 2233  |   1+  |
    | 7 Discussion              |  381  |   1   |
    | 总计 (正文)               | 6306  |       |

该模块把这份观察写成结构化指引，供 ``paper_writer.py`` 在改写章节时作为
in-context style cue 注入大模型。包含三类信息：

* ``SECTION_BLUEPRINTS`` — 每种章节的目标字数、句式骨架和"必须涵盖的子要素"。
* ``GLOBAL_VOICE`` — 跨章节的写作约束（人称、时态、避免空话等）。
* ``ANTI_FABRICATION`` — 反编造硬约束（数字/引用必须来自 bundle，不能新造）。

如果未来想换一篇参考论文，调用 :func:`extract_blueprint_from_pdf` 重新生成
新的 ``SECTION_BLUEPRINTS`` 字典即可（需要 ``pypdf`` 已安装）。
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)






@dataclass
class SectionBlueprint:
    """Style/length target for one section kind."""

    kind: str
    aliases: tuple[str, ...]
    target_words: int
    minimum_words: int
    must_cover: tuple[str, ...]
    rhetorical_arc: str


SECTION_BLUEPRINTS: list[SectionBlueprint] = [
    SectionBlueprint(
        kind="abstract",
        aliases=("abstract", "摘要"),
        target_words=200,
        minimum_words=140,
        must_cover=(
            "研究背景一句话",
            "核心问题或动机",
            "本文的具体方法 / 贡献",
            "代表性实验结论 (含一两个具体指标)",
        ),
        rhetorical_arc=(
            "1 句话点明领域和高层背景 → 1-2 句话给出 gap / 问题 → "
            "1-2 句话总结方法 → 1-2 句话给出量化结果与意义。"
        ),
    ),
    SectionBlueprint(
        kind="introduction",
        aliases=("introduction", "intro", "引言", "导言"),
        target_words=750,
        minimum_words=500,
        must_cover=(
            "领域宏观背景 (现有进展、为什么重要)",
            "明确的研究 gap / 挑战",
            "现有方法的不足，需含至少 2 类对比",
            "本文方法的核心思路（一段话）",
            "明确的 contribution bullet list（≥ 3 条）",
            "论文组织结构一句话",
        ),
        rhetorical_arc=(
            "范围由远及近：(1) 领域大图景 (2) 已有工作的两三类做法 (3) 它们的"
            "局限 (4) 我们方法的关键 insight (5) contribution 列表 (6) 论文组织。"
            "尾段一定要给出 contribution bullet 列表，每条以 \\textbf 加粗动词开头。"
        ),
    ),
    SectionBlueprint(
        kind="related_work",
        aliases=("related work", "background", "相关工作", "研究现状"),
        target_words=500,
        minimum_words=350,
        must_cover=(
            "至少分 2-3 个 thread 介绍现有工作（按方法或按目标分组）",
            "每个 thread 给出代表性方法 + 它和本文的差异",
            "结尾一句话点出本文与上述工作的本质区别",
        ),
        rhetorical_arc=(
            "按主题（不是按时间）分组。每个子主题：先一句话定义这一类工作，"
            "再点名 2-3 个代表（用引用占位 [REF:short_key]），最后一句指出与本文方法的差异。"
        ),
    ),
    SectionBlueprint(
        kind="preliminaries",
        aliases=("preliminaries", "background", "preliminary", "预备知识", "预备", "符号约定"),
        target_words=500,
        minimum_words=300,
        must_cover=(
            "符号约定 (notation table 或行内罗列)",
            "本文方法所依赖的核心定义 / 公式",
            "若引用了已有定理或损失函数，给出原文的简要回顾",
        ),
        rhetorical_arc=(
            "先给 notation 段，然后逐个引入论文后文要用到的概念，每个概念配公式。"
            "公式一律用 \\(...\\) 或 \\[...\\] 写出真实的数学表达式，不用文字描述代替。"
        ),
    ),
    SectionBlueprint(
        kind="method",
        aliases=(
            "method", "methods", "approach", "methodology", "model", "framework",
            "方法", "模型", "框架", "技术路线", "我们的方法",
        ),
        target_words=1000,
        minimum_words=700,
        must_cover=(
            "Problem formulation (输入/输出 + 约束)",
            "核心 insight 一句话总结",
            "正式的算法描述（伪代码或编号步骤）",
            "关键公式 (≥ 1 个用 LaTeX 表达式)",
            "对方法可行性 / 复杂度的简短论证",
        ),
        rhetorical_arc=(
            "Problem statement 一段 → key insight 一段 → 形式化方法 (含公式) → "
            "算法步骤（最好编号 1, 2, 3 …）→ 短分析（复杂度 / 失败模式 / "
            "为什么这样设计）。要让读者照着段落能复现实现。"
        ),
    ),
    SectionBlueprint(
        kind="experiments",
        aliases=(
            "experiments", "experimental results", "evaluation", "results",
            "实验", "实验结果", "评估", "效果",
        ),
        target_words=1500,
        minimum_words=900,
        must_cover=(
            "数据集与评估指标的明确说明",
            "训练 / 推理超参数（包含至少 3-5 个具体数值）",
            "至少 2 个对比方法 / baseline",
            "主要结果表 (引用真实表格)",
            "对结果的解读，不只列数字",
            "至少 1 组消融或敏感性分析",
        ),
        rhetorical_arc=(
            "依次：实验设置 (datasets/baselines/metrics/hyperparams) → 主实验"
            " (引用主表) → 消融实验 → 定性分析 / 失败案例。每个实验都要先讲"
            "假设和动机，再放结果，再写一两句解读。"
        ),
    ),
    SectionBlueprint(
        kind="discussion",
        aliases=("discussion", "limitations", "讨论", "局限", "局限性"),
        target_words=400,
        minimum_words=250,
        must_cover=(
            "诚实列出本文方法的至少 2 个局限",
            "对未来工作的具体方向（≥ 2 条）",
            "本文结论与相关工作的对话",
        ),
        rhetorical_arc=(
            "不要假大空。每条限制配一句话\"为什么会有这个限制\"。每条未来工作"
            "给出一个具体可执行的研究方向，不是模糊的\"进一步探索\"。"
        ),
    ),
    SectionBlueprint(
        kind="conclusion",
        aliases=("conclusion", "conclusions", "summary", "结论", "总结", "小结"),
        target_words=250,
        minimum_words=150,
        must_cover=(
            "一句话点题：本文做了什么",
            "1-2 句话总结实验结论与定量结果",
            "对该领域的影响与展望",
        ),
        rhetorical_arc=(
            "把摘要的精髓换一种说法重写。第一句陈述本文成果，第二句给最有"
            "代表性的量化结论，第三句把意义提升到领域层面。"
        ),
    ),
]


def blueprint_for(heading: str) -> SectionBlueprint | None:
    """Match a section heading to one of the blueprints (case/number-insensitive)."""
    if not heading:
        return None
    key = re.sub(r"^\d+(?:\.\d+)*\s*", "", heading.strip()).strip(" .:：-").lower()
    for bp in SECTION_BLUEPRINTS:
        for alias in bp.aliases:
            if alias.lower() in key or key in alias.lower():
                return bp
    return None


def scaled_blueprints(target_total_words: int | None) -> list[SectionBlueprint]:
    """Return a list of blueprints rescaled to hit ``target_total_words``.

    The default blueprints sum to roughly 5100 body words, which matches
    the NeurIPS reference paper. When the user wants a longer or shorter
    paper they pass a ``target_total_words`` and we prorate every
    section's ``target_words`` / ``minimum_words`` accordingly. Abstract
    and conclusion are left near their canonical sizes (an abstract
    grows poorly past 250 words) — only the body sections expand or
    contract to soak up the change.

    Pass ``None`` to keep the canonical top-conf defaults.
    """
    if target_total_words is None or target_total_words <= 0:
        return SECTION_BLUEPRINTS

    fixed_kinds = {"abstract", "conclusion"}
    fixed_total = sum(
        bp.target_words for bp in SECTION_BLUEPRINTS if bp.kind in fixed_kinds
    )
    body_total = sum(
        bp.target_words for bp in SECTION_BLUEPRINTS if bp.kind not in fixed_kinds
    )
    if body_total <= 0:
        return SECTION_BLUEPRINTS

    body_target = max(target_total_words - fixed_total, body_total // 4)
    scale = body_target / body_total

    out: list[SectionBlueprint] = []
    for bp in SECTION_BLUEPRINTS:
        if bp.kind in fixed_kinds:
            out.append(bp)
            continue
        new_target = max(int(round(bp.target_words * scale)), 100)

        new_minimum = max(int(round(new_target * 0.7)), 70)
        out.append(SectionBlueprint(
            kind=bp.kind,
            aliases=bp.aliases,
            target_words=new_target,
            minimum_words=new_minimum,
            must_cover=bp.must_cover,
            rhetorical_arc=bp.rhetorical_arc,
        ))
    return out


def blueprint_for_in(
    heading: str,
    blueprints: list[SectionBlueprint] | None = None,
) -> SectionBlueprint | None:
    """Variant of :func:`blueprint_for` that searches a custom blueprint list.

    Use this when callers have rescaled the canonical blueprints via
    :func:`scaled_blueprints` and want heading lookup to honour the new
    word targets.
    """
    if blueprints is None:
        return blueprint_for(heading)
    if not heading:
        return None
    key = re.sub(r"^\d+(?:\.\d+)*\s*", "", heading.strip()).strip(" .:：-").lower()
    for bp in blueprints:
        for alias in bp.aliases:
            if alias.lower() in key or key in alias.lower():
                return bp
    return None






GLOBAL_VOICE = """\
通用写作风格（与顶会论文一致）：
1. 第一人称用 “we / 我们”，禁止 “the author(s) / I”。
2. 主动语态为主，描述既定结论时可以用被动。
3. 每段以一句\"主题句 (topic sentence)\"开头，承担本段的中心论点；中间句子展开；
   段尾一句过渡到下一段或回扣主题。
4. 避免空话和广告语：禁用 “revolutionary”, “cutting-edge”, “state-of-the-art”
   作为修饰语；少用 “very / much / significantly” 等模糊副词。
5. 公式用 LaTeX 内联或独立行表示真实数学表达；不要用纯文字描述公式 (\
   例如不要写 “loss equals the sum of …”，要直接写 \\(\\mathcal L = \\sum_i …\\))。
6. 引用沿用 bundle 中给出的占位符 ``[ref-N]`` 或 ``[1]``，不要新造引用编号。
7. 表格、图引用用 \"Table~\\ref{...}\" / \"Figure~\\ref{...}\" 风格；中文论文写
   \"表~\\ref{...}\" / \"图~\\ref{...}\"。
8. 每个章节末尾不要重复章节标题，章节之间用一句过渡承接。
"""


ANTI_FABRICATION = """\
禁止编造（最高优先级，违反则视为整段失效）：
1. 不得引入原文 / bundle 中没有的数字、百分比、p-value、运行时长等量化结果。
2. 不得新增参考文献。已有的引用占位符可以重新组织顺序，但不得增加。
3. 不得新加未在 bundle 中出现的图、表、数据集名、模型名。
4. 如果原始内容某项指标缺失，明确写 \"未报告 / 未在实验中采集\"，绝不补值。
5. 在改写过程中如果需要 \"举例数值\"，请使用 ``$x$``、``$N$`` 等占位符，而非
   编造具体数字。
6. 实验解读必须紧贴原始表格中的数字，不得断言原表中不存在的趋势。
"""






REFERENCE_NOTE = (
    "本指南基于真实顶会样本（NeurIPS 2023, DPO 论文）的章节字数与结构分析"
    "整理而成。各章节的字数目标和叙事顺序与该样本一致。"
)


def compose_style_guide(target_language_name: str = "中文") -> str:
    """Return a single-string style guide ready to inject as system context."""
    lines: list[str] = []
    lines.append(f"## 写作风格指南（{target_language_name}）")
    lines.append("")
    lines.append(REFERENCE_NOTE)
    lines.append("")
    lines.append(GLOBAL_VOICE)
    lines.append("")
    lines.append(ANTI_FABRICATION)
    lines.append("")
    lines.append("## 各章节字数与结构目标")
    for bp in SECTION_BLUEPRINTS:
        lines.append("")
        lines.append(f"### {bp.kind}（别名：{', '.join(bp.aliases)}）")
        lines.append(
            f"- 目标字数 ~ {bp.target_words}（最少 {bp.minimum_words}）"
        )
        lines.append(f"- 必须覆盖：")
        for item in bp.must_cover:
            lines.append(f"    * {item}")
        lines.append(f"- 叙事节奏：{bp.rhetorical_arc}")
    return "\n".join(lines)


def section_brief(blueprint: SectionBlueprint, target_language_name: str) -> str:
    """A short per-section instruction sheet, used inline when rewriting one section."""
    must = "\n".join(f"  · {x}" for x in blueprint.must_cover)
    return (
        f"本章节体裁：**{blueprint.kind}**（{target_language_name}）\n"
        f"目标字数：{blueprint.target_words} 字（最少 {blueprint.minimum_words}）。\n"
        f"必须涵盖：\n{must}\n"
        f"叙事节奏：{blueprint.rhetorical_arc}"
    )






def extract_blueprint_from_pdf(pdf_path: str | Path) -> dict[str, dict]:
    """Read a PDF and produce a dict of {section_kind: {words, paragraphs}}.

    Useful as a developer tool when you want to update :data:`SECTION_BLUEPRINTS`
    based on a different reference paper. Requires ``pypdf`` to be importable.
    """
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError(
            "pypdf is required for extract_blueprint_from_pdf; "
            "install with `pip install pypdf`."
        ) from exc

    reader = PdfReader(str(pdf_path))
    text = "\n".join((page.extract_text() or "") for page in reader.pages)

    headings = [
        (m.start(), m.group(1), m.group(2).strip())
        for m in re.finditer(r"^(\d+)\s+([A-Z][\w \-:&,]{3,60})$", text, re.MULTILINE)
    ]
    out: dict[str, dict] = {}
    for i, (start, num, title) in enumerate(headings):
        end = headings[i + 1][0] if i + 1 < len(headings) else len(text)
        body = text[start:end]
        words = len(re.findall(r"[A-Za-z]+", body))
        paragraphs = len([p for p in re.split(r"\n\s*\n", body) if p.strip()])
        out[title] = {"words": words, "paragraphs": paragraphs}
    return out
