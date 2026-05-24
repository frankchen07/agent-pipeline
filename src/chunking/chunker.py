"""Boundary-aware semantic chunker."""
import re
from dataclasses import dataclass, field

import tiktoken


@dataclass
class Chunk:
    seq: int
    text: str
    token_count: int
    boundary_type: str
    boundary_text: str
    char_start: int
    char_end: int
    concepts: list[str] = field(default_factory=list)


def count_tokens(text: str, enc) -> int:
    return len(enc.encode(text))


def chunk_text(
    text: str,
    structure: list[dict],
    media_type: str,
    target: int = 500,
    min_tokens: int = 150,
    max_tokens: int = 650,
    tokenizer: str = "cl100k_base",
) -> list[Chunk]:
    enc = tiktoken.get_encoding(tokenizer)
    segments = _split_into_segments(text, structure, media_type)
    return _merge_segments(segments, enc, target, min_tokens, max_tokens)


def _split_into_segments(text: str, structure: list[dict], media_type: str) -> list[dict]:
    """Split text at boundary positions into raw segments."""
    boundaries = sorted(structure, key=lambda x: x["position"])

    # Add synthetic start boundary
    positions = [0] + [b["position"] for b in boundaries] + [len(text)]
    boundary_meta = [{"type": "start", "text": ""}] + boundaries + [{"type": "eof", "text": ""}]

    segments = []
    for i in range(len(positions) - 1):
        start = positions[i]
        end = positions[i + 1]
        content = text[start:end]
        if content.strip():
            segments.append({
                "content": content,
                "boundary_type": boundary_meta[i]["type"],
                "boundary_text": boundary_meta[i]["text"],
                "char_start": start,
                "char_end": end,
            })
    return segments


def _merge_segments(
    segments: list[dict],
    enc,
    target: int,
    min_tokens: int,
    max_tokens: int,
) -> list[Chunk]:
    chunks = []
    seq = 0
    buf_text = []
    buf_tokens = 0
    buf_start = 0
    buf_boundary_type = "start"
    buf_boundary_text = ""

    def flush(end_pos: int):
        nonlocal seq, buf_text, buf_tokens, buf_start, buf_boundary_type, buf_boundary_text
        combined = "".join(buf_text).strip()
        if not combined:
            return
        tok = count_tokens(combined, enc)
        if tok >= min_tokens:
            chunks.append(Chunk(
                seq=seq,
                text=combined,
                token_count=tok,
                boundary_type=buf_boundary_type,
                boundary_text=buf_boundary_text,
                char_start=buf_start,
                char_end=end_pos,
                concepts=_extract_concepts(combined),
            ))
            seq += 1
        buf_text = []
        buf_tokens = 0

    for seg in segments:
        seg_tokens = count_tokens(seg["content"], enc)

        if seg_tokens > max_tokens:
            # Flush current buffer first
            if buf_tokens >= min_tokens:
                flush(seg["char_start"])
            else:
                buf_text = []
                buf_tokens = 0
            # Split the oversized segment at sentence boundaries
            for sub in _split_at_sentences(seg["content"], enc, target, max_tokens):
                tok = count_tokens(sub, enc)
                chunks.append(Chunk(
                    seq=seq,
                    text=sub.strip(),
                    token_count=tok,
                    boundary_type=seg["boundary_type"],
                    boundary_text=seg["boundary_text"],
                    char_start=seg["char_start"],
                    char_end=seg["char_end"],
                    concepts=_extract_concepts(sub),
                ))
                seq += 1
            buf_start = seg["char_end"]
            buf_boundary_type = seg["boundary_type"]
            buf_boundary_text = seg["boundary_text"]
            continue

        if buf_tokens + seg_tokens > max_tokens and buf_tokens >= min_tokens:
            flush(seg["char_start"])
            buf_start = seg["char_start"]
            buf_boundary_type = seg["boundary_type"]
            buf_boundary_text = seg["boundary_text"]

        if not buf_text:
            buf_start = seg["char_start"]
            buf_boundary_type = seg["boundary_type"]
            buf_boundary_text = seg["boundary_text"]

        buf_text.append(seg["content"])
        buf_tokens += seg_tokens

        if buf_tokens >= target:
            flush(seg["char_end"])
            buf_start = seg["char_end"]

    if buf_text:
        combined = "".join(buf_text).strip()
        if combined and buf_tokens >= min_tokens:
            chunks.append(Chunk(
                seq=seq,
                text=combined,
                token_count=count_tokens(combined, enc),
                boundary_type=buf_boundary_type,
                boundary_text=buf_boundary_text,
                char_start=buf_start,
                char_end=len(combined),
                concepts=_extract_concepts(combined),
            ))

    return chunks


def _split_at_sentences(text: str, enc, target: int, max_tokens: int) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text)
    parts = []
    current = []
    current_tok = 0
    for sent in sentences:
        tok = count_tokens(sent, enc)
        if current_tok + tok > max_tokens and current:
            parts.append(" ".join(current))
            current = [sent]
            current_tok = tok
        else:
            current.append(sent)
            current_tok += tok
    if current:
        parts.append(" ".join(current))
    return [p for p in parts if p.strip()]


# Simple keyword extraction without LLM — good enough for concept tags
_FINANCE_KEYWORDS = [
    "rich life", "invisible script", "automation", "savings", "investing",
    "earning", "salary", "negotiate", "budget", "spending", "debt",
    "psychology", "behavior", "identity", "couples", "money",
    "retirement", "roth", "401k", "index fund", "frugality", "hustle",
    "income", "career", "raise", "freelance", "entrepreneur", "side hustle",
    "net worth", "credit card", "fee", "bank", "financial", "wealth",
]


def _extract_concepts(text: str) -> list[str]:
    text_lower = text.lower()
    found = [kw for kw in _FINANCE_KEYWORDS if kw in text_lower]
    return found[:5]
