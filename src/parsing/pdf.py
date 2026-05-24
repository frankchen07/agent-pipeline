"""PDF parser using pymupdf4llm."""
from pathlib import Path


def parse(filepath: Path) -> dict:
    try:
        import pymupdf4llm
    except ImportError:
        raise RuntimeError("pymupdf4llm not installed. Run: pip install pymupdf4llm")

    md_text = pymupdf4llm.to_markdown(str(filepath))
    return {
        "text": md_text,
        "media_type": "pdf",
        "structure": _extract_structure(md_text),
    }


def _extract_structure(text: str) -> list[dict]:
    import re
    boundaries = []
    for m in re.finditer(r"^(#{1,4}\s+.+)$", text, re.MULTILINE):
        level = len(m.group(1)) - len(m.group(1).lstrip("#"))
        boundaries.append({
            "position": m.start(),
            "type": f"h{level}",
            "text": m.group(1).strip(),
        })
    return boundaries
