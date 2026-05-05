"""
Markdown syntax subset template definitions.
Define allow-lists and deny-lists to ensure stable rendering to LaTeX.
"""
import re



VALID_HEADING_PATTERN = re.compile(r'^#{2,4} \d+(?:\.\d+)*\.?\s+.*$')


VALID_REFERENCE_PATTERN = re.compile(r'\[\s*\d+\s*(?:,\s*\d+\s*)*\]')


HTML_TAG_PATTERN = re.compile(r'<[^>]+>')



LATEX_COMMAND_PATTERN = re.compile(r'\\[a-zA-Z]+\{')


MARKDOWN_TABLE_PATTERN = re.compile(r'^[|\s]*---[-|\s]*$')


INVALID_CITE_PATTERN = re.compile(r'\\cite\{|\[[a-zA-Z]+,?\s+\d{4}\]')


def has_markdown_table(text: str) -> bool:
    for line in text.split('\n'):
        if MARKDOWN_TABLE_PATTERN.match(line):
            return True
    return False
