from typing import List, Dict
from .markdown_template import (
    HTML_TAG_PATTERN, LATEX_COMMAND_PATTERN, 
    INVALID_CITE_PATTERN, has_markdown_table
)

def validate_markdown(text: str) -> List[str]:
    violations = []
    if HTML_TAG_PATTERN.search(text):
        violations.append("Contains raw HTML tags.")
    if has_markdown_table(text):
        violations.append("Contains Markdown tables. Only CSV paths are allowed.")
    if INVALID_CITE_PATTERN.search(text):
        violations.append("Contains invalid citation styles (e.g., \\cite{} or Author-Year).")
    return violations
