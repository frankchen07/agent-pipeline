"""Plain text and podcast transcript parser."""
import re
from pathlib import Path


def parse(filepath: Path) -> dict:
    text = filepath.read_text(encoding="utf-8", errors="replace")
    media_type = _detect_txt_subtype(text)
    return {
        "text": text,
        "media_type": media_type,
        "structure": _extract_structure(text, media_type),
    }


def _detect_txt_subtype(text: str) -> str:
    # Podcast transcript: has EPISODE N headers and [HH:MM:SS] timestamps
    if re.search(r"^EPISODE\s+\d+", text, re.MULTILINE) and re.search(r"\[\d{2}:\d{2}:\d{2}\]", text):
        return "podcast_transcript"
    # Plain text
    return "txt"


def _extract_structure(text: str, media_type: str) -> list[dict]:
    """Return a list of {position, type, text} boundary markers."""
    boundaries = []
    if media_type == "podcast_transcript":
        for m in re.finditer(r"^(EPISODE\s+\d+)", text, re.MULTILINE):
            boundaries.append({"position": m.start(), "type": "episode", "text": m.group(1)})
        for m in re.finditer(r"\[(\d{2}:\d{2}:\d{2})\]", text):
            boundaries.append({"position": m.start(), "type": "timestamp", "text": m.group(0)})
    else:
        for m in re.finditer(r"\n\n+", text):
            boundaries.append({"position": m.start(), "type": "paragraph", "text": ""})
    return sorted(boundaries, key=lambda x: x["position"])
