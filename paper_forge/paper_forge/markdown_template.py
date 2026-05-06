"""
Markdown syntax subset template definitions.
Define allow-lists and deny-lists to ensure stable rendering to LaTeX.
"""
import re

# We enforce strict formatting for headings, references, etc.
# Valid Heading: "## 1. Introduction" or "## 2. Related Work"
VALID_HEADING_PATTERN = re.compile(r'^#{2,4} \d+(?:\.\d+)*\.?\s+.*$')

# Valid References: "[1]", "[1, 2]"
VALID_REFERENCE_PATTERN = re.compile(r'\[\s*\d+\s*(?:,\s*\d+\s*)*\]')

# Prohibited HTML tags
HTML_TAG_PATTERN = re.compile(r'<[^>]+>')

# Prohibited LaTeX commands in text (allow backslashes in math)
# A simple heuristic: check for \command{...} outside of math blocks.
LATEX_COMMAND_PATTERN = re.compile(r'\\[a-zA-Z]+\{')

# Prohibited markdown tables
MARKDOWN_TABLE_PATTERN = re.compile(r'^[|\s]*---[-|\s]*$')

# Forbidden citation styles like \cite{key} or [Author, Year]
INVALID_CITE_PATTERN = re.compile(r'\\cite\{|\[[a-zA-Z]+,?\s+\d{4}\]')

# Markdown table detector: A row that contains multiple pipes and is table-like.
def has_markdown_table(text: str) -> bool:
    for line in text.split('\n'):
        if MARKDOWN_TABLE_PATTERN.match(line):
            return True
    return False
