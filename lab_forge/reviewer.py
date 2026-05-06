"""
LLM-based reviewer for hallucination detection at key checkpoints.

A separate (cheap) model reviews the agent's work at critical points to catch
hallucinations and unsupported claims.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field

from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)


@dataclass
class ReviewResult:
    """Result of a review checkpoint."""

    passed: bool
    score: float  # 0.0 to 1.0
    issues: list[str] = field(default_factory=list)
    suggestion: str = ""
    # True when this result reflects an infrastructure failure (LLM call
    # failed) rather than a genuine content review. Lets callers
    # distinguish "the reviewer rejected the work" from "we never
    # actually got a verdict because the API rejected our request".
    error: bool = False


# ---------------------------------------------------------------------------
# Review prompts
# ---------------------------------------------------------------------------

LITERATURE_REVIEW_PROMPT = """\
你是一个科研评审员。请检查以下文献证据和 agent 的总结是否一致。

## 实际文献证据
{search_results}

## Agent 的文献总结/引用
{agent_summary}

## 检查要点
1. Agent 是否引用了实际文献证据中不存在的论文？
2. Agent 对论文内容的描述是否与摘要或全文读取结果一致？
3. 如果 arXiv 检索失败，Agent 是否如实说明了这一限制？

硬性规则：实际文献证据既包括 search_literature 返回的 arXiv 论文，
也包括 read_paper_fulltext 对用户上传/本地 PDF 的 OCR 或全文抽取结果。
用户上传/本地全文读取成功的论文是有效来源，不能因为它不是 arXiv
检索返回项就判定为“未检索出来的幻觉引用”。

请用以下 JSON 格式回复：
{{"passed": true/false, "score": 0.0-1.0, "issues": ["问题1", "问题2"], "suggestion": "修改建议"}}
"""

EXPERIMENT_REVIEW_PROMPT = """\
你是一个科研评审员。请检查以下实验结果是否合理。

## 任务描述
{task_description}

## 代码执行输出
{code_outputs}

## Agent 声称的结果
{claimed_results}

## 检查要点
1. 声称的数值结果是否与实际代码输出一致？
2. 结果是否存在异常（NaN、inf、infinity、overflow、非有限值、全零、负收敛率等不合理数据）？
3. 实验方法是否合理（有无训练/测试集划分、随机种子等）？

硬性规则：只要实际输出或保存结果里出现 NaN/inf/infinity/非有限值，
而 Agent 没有明确把它当作失败、发散或无效结果处理，就必须返回
passed=false。不能把非有限数值解释成正常提升或有效结论。

请用以下 JSON 格式回复：
{{"passed": true/false, "score": 0.0-1.0, "issues": ["问题1", "问题2"], "suggestion": "修改建议"}}
"""

REPORT_REVIEW_PROMPT = """\
你是一个科研评审员。请检查以下研究报告是否存在幻觉或编造内容。

## 任务描述
{task_description}

## 实际工具执行历史（摘要）
{tool_history}

## 报告内容
{report_content}

## 检查要点
1. 报告中的数据和结论是否有工具执行结果作为支撑？
2. 引用的论文和作者是否在实际文献证据中出现过（包括 arXiv 检索结果
   和 read_paper_fulltext 成功读取的用户上传/本地论文）？
3. 是否存在编造的数据、引用或结论？
4. 如果 arXiv 文献检索失败，报告是否如实说明了限制？
5. 如果工具历史、CSV/JSON/日志或报告中出现 NaN、inf、infinity、
   overflow、非有限值或负收敛率，报告是否明确说明这是失败/发散/无效
   结果，并避免基于它做正向结论？

硬性规则：非有限实验值没有被明确披露为失败或未解决问题时，必须
返回 passed=false，即使报告其他部分写得完整。
硬性规则：用户上传/本地全文读取成功的论文是有效文献证据；不要把这类
论文误判为“主模型引用了未检索出来的论文”。

请用以下 JSON 格式回复：
{{"passed": true/false, "score": 0.0-1.0, "issues": ["问题1", "问题2"], "suggestion": "修改建议"}}
"""

SUBMISSION_REVIEW_PROMPT = """\
你是一个科研评审员。请对以下科研任务的最终提交做一个简要质量评审。

## 任务描述
{task_description}

## 执行摘要
{trajectory_summary}

## 检查要点
1. 任务目标是否基本完成？
2. 关键结果是否有实验支撑？
3. 是否存在明显的遗漏或错误？
4. 是否存在未处理的 NaN/inf/infinity/非有限实验结果？如果有且未
   说明为失败或未解决问题，必须判定不通过。

## 任务对齐硬性规则 (HARD — 任意一条违反必须 passed=false)

仔细阅读上面 ## 任务描述 的具体动词 / 名词组合，**逐项检查执行摘要里
是否有对应证据**。常见对齐项：

- 任务说"**提出 / 设计 / 创新 / propose / introduce / design** 一种
  新算法 (新方法 / 新模型)" → 执行摘要里必须出现自定义方法的实现代码
  (不只是调用 sklearn / torch 现成模型)，且实验结果中该自定义方法有
  独立的 metric 数字。如果只看到 K-means / DBSCAN / SVM 等现成方法
  的对比，没有自定义实现 → 必须 passed=false，issues 中明确写
  "**任务要求的"提出新方法"未实现**"。

- 任务说"**超过 / 优于 / outperform** 基线 (在大多数指标上)" → 执行
  摘要必须有**新方法 vs 基线**的并排数字，且新方法在多数指标上数值
  更优。如果新方法不存在或没赢 → 必须 passed=false，issues 中写
  "**未达成"超过基线"目标**"。

- 任务说"**对比 / 比较** 多种方法" → 至少有 2 个方法的实验结果。
  少于 2 个 → passed=false。

- 任务说"在 **真实数据集 / real-world dataset** 上验证" → 执行摘要
  必须出现公开数据集名 (CIFAR / MNIST / GLUE / 20newsgroups 等)，
  不能只用合成 / synthetic / make_classification 数据。否则
  passed=false。

- 任务说"**轻量实验**" 是约束放宽，不是免责 — 上面的核心动词检查
  仍然适用，只是数据规模可以小。

不要因为代码里出现"我们提出"、"our method"、"novel framework" 这些
**口号词**就以为目标完成了。**只看执行摘要里有没有真实的自定义方法
代码与对应的实验数字**。

请用以下 JSON 格式回复：
{{"passed": true/false, "score": 0.0-1.0, "issues": ["问题1", "问题2"], "suggestion": "修改建议"}}
"""


_AUTH_ERROR_MARKERS = (
    "invalid_model",
    "model not found",
    "modelnotfound",
    "model_not_found",
    "no permission",
    "permission denied",
    "unauthorized",
    "no access",
    "401",
    "403",
    "404",
)


def _looks_like_auth_or_model_error(exc: Exception) -> bool:
    """True if the exception looks like a credential / model-permission error.

    AI Studio retired several legacy ERNIE models in 2026 and rejects them
    with messages like ``invalid_model`` / ``no permission`` / 401. When that
    happens the reviewer can't recover by retrying the same call, but it
    *can* recover by falling back to the agent's own (working) LLM.
    """
    status_code = getattr(exc, "status_code", None)
    if status_code is None:
        status_code = getattr(getattr(exc, "response", None), "status_code", None)
    if status_code in (401, 403, 404):
        return True
    text = str(exc).lower()
    return any(marker in text for marker in _AUTH_ERROR_MARKERS)


class Reviewer:
    """LLM-based reviewer that checks agent work at key checkpoints."""

    def __init__(self, llm: ChatOpenAI, fallback_llm: ChatOpenAI | None = None):
        self.llm = llm
        # ``fallback_llm`` is consulted exactly once per call when the primary
        # reviewer LLM fails with an auth / model-permission style error
        # (typically AI Studio rejecting a retired ERNIE 3.5/Speed model).
        # Passing the agent's main LLM here is the recommended setup: if
        # the agent can talk to the API at all, the reviewer can too.
        self.fallback_llm = fallback_llm
        self._fallback_active = False
        self._fallback_warned = False

    def review_literature(
        self,
        search_results: list[str],
        agent_summary: str,
    ) -> ReviewResult:
        """Check if claimed papers/facts match actual literature evidence."""
        results_text = "\n\n---\n\n".join(search_results) if search_results else "(无成功的文献证据)"
        prompt = LITERATURE_REVIEW_PROMPT.format(
            search_results=results_text[:6000],
            agent_summary=agent_summary[:3000],
        )
        return self._call_and_parse(prompt)

    def review_experiment(
        self,
        task_description: str,
        code_outputs: list[str],
        claimed_results: str,
    ) -> ReviewResult:
        """Check if claimed numbers match actual execution outputs."""
        outputs_text = "\n\n---\n\n".join(code_outputs[-5:]) if code_outputs else "(无代码执行输出)"
        prompt = EXPERIMENT_REVIEW_PROMPT.format(
            task_description=task_description[:2000],
            code_outputs=outputs_text[:6000],
            claimed_results=claimed_results[:3000],
        )
        return self._call_and_parse(prompt)

    def review_report(
        self,
        task_description: str,
        tool_history: str,
        report_content: str,
    ) -> ReviewResult:
        """Full hallucination check on report draft vs. actual evidence."""
        prompt = REPORT_REVIEW_PROMPT.format(
            task_description=task_description[:2000],
            tool_history=tool_history[:6000],
            report_content=report_content[:6000],
        )
        return self._call_and_parse(prompt)

    def review_submission(
        self,
        task_description: str,
        trajectory_summary: str,
    ) -> ReviewResult:
        """Final quality check before submission."""
        prompt = SUBMISSION_REVIEW_PROMPT.format(
            task_description=task_description[:2000],
            trajectory_summary=trajectory_summary[:6000],
        )
        return self._call_and_parse(prompt)

    def _call_and_parse(self, prompt: str) -> ReviewResult:
        """Call the reviewer LLM and parse the JSON response.

        On infrastructure failure (network, 401/invalid_model, timeout, etc.)
        we MUST NOT silently pass-through. Returning ``passed=True`` here
        previously meant a broken reviewer effectively disabled itself and
        let hallucinated work ship.

        Recovery rules:
        - If a fallback LLM is configured AND the primary fails with an
          auth/model-permission style error, switch to the fallback for
          the rest of the run and retry once. Subsequent calls go through
          the fallback directly.
        - All other failures (or fallback also failing) surface as a
          blocking ``passed=False`` result with the actual error so the
          agent loop and UI make it visible.
        """
        # If we already migrated to the fallback this run, use it directly.
        active_llm = self.fallback_llm if self._fallback_active else self.llm
        try:
            response = active_llm.invoke(prompt)
            text = response.content if hasattr(response, "content") else str(response)
            return self._parse_review(text)
        except Exception as exc:
            # Try the fallback exactly once on auth/model errors.
            if (
                not self._fallback_active
                and self.fallback_llm is not None
                and _looks_like_auth_or_model_error(exc)
            ):
                primary_msg = self._format_call_error(exc)
                if not self._fallback_warned:
                    logger.warning(
                        "Reviewer primary LLM rejected (%s); falling back to "
                        "the agent's main LLM for the rest of this run.",
                        primary_msg,
                    )
                    self._fallback_warned = True
                self._fallback_active = True
                try:
                    response = self.fallback_llm.invoke(prompt)
                    text = response.content if hasattr(response, "content") else str(response)
                    result = self._parse_review(text)
                    # Record the silent fallback once so the UI can show why.
                    if not result.issues:
                        result.issues = []
                    return result
                except Exception as fb_exc:
                    error_msg = self._format_call_error(fb_exc)
                    logger.error(
                        "Reviewer fallback also failed after primary %s: %s",
                        primary_msg,
                        error_msg,
                    )
                    return ReviewResult(
                        passed=False,
                        score=0.0,
                        issues=[
                            f"评审模型调用失败：{primary_msg}",
                            f"备用模型也失败：{error_msg}",
                        ],
                        suggestion=(
                            "评审 API 主备模型都失败，无法判断产出是否可信。"
                            "请检查 API Key、Base URL，以及当前账户对所选模型的访问权限。"
                        ),
                        error=True,
                    )

            error_msg = self._format_call_error(exc)
            logger.error("Reviewer call failed: %s", error_msg)
            return ReviewResult(
                passed=False,
                score=0.0,
                issues=[f"评审模型调用失败：{error_msg}"],
                suggestion=(
                    "评审 API 调用失败，无法判断本阶段产出是否可信。"
                    "请检查评审模型的 model 名称、Base URL 与 API Key 配置后重试；"
                    "在评审恢复前禁止直接 submit_result。"
                ),
                error=True,
            )

    @staticmethod
    def _format_call_error(exc: Exception) -> str:
        """Build a compact, actionable error message for reviewer failures."""
        status_code = getattr(exc, "status_code", None)
        if status_code is None:
            status_code = getattr(getattr(exc, "response", None), "status_code", None)
        text = str(exc)
        if len(text) > 400:
            text = text[:400] + "…(truncated)"
        if status_code is not None:
            return f"HTTP {status_code} - {text}"
        return text

    def _parse_review(self, text: str) -> ReviewResult:
        """Parse the reviewer's JSON response, tolerating nested braces."""
        candidate = self._extract_json_object(text)
        if candidate is not None:
            try:
                data = json.loads(candidate)
                return ReviewResult(
                    passed=bool(data.get("passed", True)),
                    score=max(0.0, min(1.0, float(data.get("score", 0.5)))),
                    issues=list(data.get("issues", []) or []),
                    suggestion=str(data.get("suggestion", "") or ""),
                )
            except (json.JSONDecodeError, ValueError, TypeError):
                pass

        # Fallback: try to determine pass/fail from free-form text.
        lower = text.lower()
        passed = "pass" in lower or "true" in lower
        return ReviewResult(
            passed=passed,
            score=0.7 if passed else 0.3,
            issues=[],
            suggestion=text[:500],
        )

    @staticmethod
    def _extract_json_object(text: str) -> str | None:
        """Scan ``text`` for the first balanced ``{...}`` block.

        The previous `\\{[^{}]*\\}` regex bailed out on any nested braces — even
        innocuous ones like `{"issues": ["missing {key}"]}`. We walk the string
        instead so nested braces inside strings don't corrupt the match.
        """
        depth = 0
        start = -1
        in_string = False
        escape = False
        for idx, ch in enumerate(text):
            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
                continue
            if ch == "{":
                if depth == 0:
                    start = idx
                depth += 1
            elif ch == "}":
                if depth == 0:
                    continue
                depth -= 1
                if depth == 0 and start != -1:
                    return text[start : idx + 1]
        return None
