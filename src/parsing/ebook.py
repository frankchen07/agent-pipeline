"""Calibre-exported ebook txt files (YAML frontmatter + markdown-ish content)."""
import re
from pathlib import Path


def parse(filepath: Path) -> dict:
    raw = filepath.read_text(encoding="utf-8", errors="replace")
    text, frontmatter = _strip_frontmatter(raw)
    text = _strip_html(text)
    return {
        "text": text,
        "media_type": "ebook_txt",
        "frontmatter": frontmatter,
        "structure": _extract_structure(text),
    }


def _strip_frontmatter(raw: str) -> tuple[str, dict]:
    if not raw.startswith("---"):
        return raw, {}
    end = raw.find("\n---", 3)
    if end == -1:
        return raw, {}
    fm_block = raw[3:end].strip()
    body = raw[end + 4:].lstrip("\n")
    # Parse simple key: value pairs (good enough without a full YAML parser)
    fm = {}
    for line in fm_block.splitlines():
        if ":" in line:
            k, _, v = line.partition(":")
            fm[k.strip()] = v.strip().strip('"')
    return body, fm


def _strip_html(text: str) -> str:
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"\{[^}]*\}", "", text)  # calibre div attrs like {.class}
    text = re.sub(r":{2,}[^\n]*", "", text)  # ::: fenced divs
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _extract_structure(text: str) -> list[dict]:
    boundaries = []
    for m in re.finditer(r"^(#{1,3}\s+.+)$", text, re.MULTILINE):
        level = len(m.group(1)) - len(m.group(1).lstrip("#"))
        boundaries.append({
            "position": m.start(),
            "type": f"h{level}",
            "text": m.group(1).strip(),
        })
    return boundaries
