from langchain_core.messages import HumanMessage
from .markdown_validator import validate_markdown
import re

def normalize_markdown(text: str, llm=None) -> str:
    """Normalize markdown to enforce strict syntax subset."""
    violations = validate_markdown(text)
    if not violations:
        return text
    
    if not llm:
        # Fallback if no LLM: try simple cleaning
        text = re.sub(r'<[^>]+>', '', text)
        return text

    prompt = f"""
Fix the following markdown text to strictly adhere to the academic template format.
Violations found:
{chr(10).join(violations)}

STRICT RULES:
1. No HTML tags.
2. No LaTeX commands like \\section or \\textbf (only math $...$ and $$...$$).
3. No Markdown tables. Convert tables to descriptive text.
4. Use ONLY numeric references like [1] or [1, 2]. Do NOT use \\cite{{}} or [Author, Year].
5. Use **bold**, *italic*, and `code` natively.
6. Headers MUST be like '## 1. Introduction'.

TEXT:
{text}

Return ONLY the corrected markdown text.
"""
    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        return response.content.strip()
    except Exception:
        return text
