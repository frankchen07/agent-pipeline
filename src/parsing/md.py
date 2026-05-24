"""Markdown file parser."""
import re
from pathlib import Path


def parse(filepath: Path) -> dict:
    text = filepath.read_text(encoding="utf-8", errors="replace")
    return {
        "text": text,
        "media_type": "markdown",
        "structure": _extract_structure(text),
    }


def _extract_structure(text: str) -> list[dict]:
    boundaries = []
    for m in re.finditer(r"^(#{1,4}\s+.+)$", text, re.MULTILINE):
        level = len(m.group(1)) - len(m.group(1).lstrip("#"))
        boundaries.append({
            "position": m.start(),
            "type": f"h{level}",
            "text": m.group(1).strip(),
        })
    return boundaries
