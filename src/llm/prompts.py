"""Prompt assembly functions for each pipeline stage."""
from pathlib import Path


def load_prompt(path: Path | str) -> str:
    return Path(path).read_text(encoding="utf-8")


def classify_system() -> str:
    return """You are a source classifier for an advisor corpus pipeline.

Classify the given text sample as ONE of:
- primary: Written or spoken directly by the advisor (books, articles, newsletters, their own podcast)
- secondary: Analysis of the advisor's work, interviews about them, discussions featuring them as a guest on someone else's show
- personal-note: The user's personal notes, recordings, or synthesis
- non-canonical: Synthetic content, previous AI bot sessions, tangential references

Respond with JSON only:
{"classification": "primary|secondary|personal-note|non-canonical", "confidence": "high|medium|low", "reasoning": "one sentence"}"""


def classify_user(filename: str, sample: str) -> str:
    return f"Filename: {filename}\n\nText sample:\n{sample[:2000]}"


def summarize_system(base_prompt: str) -> str:
    return f"""You extract structured knowledge from a single advisor source.

For each source, extract:
1. Key arguments / doctrines unique to this source
2. Tone & voice markers (quote the text, tag with chunk_id)
3. Advice patterns (how problems are diagnosed, what solutions are offered, what is rejected)
4. User archetypes or personas mentioned
5. Anti-patterns named (what the advisor says NOT to do)
6. Representative quotes (cite chunk_id in brackets)
7. Blind spots, limitations, caveats
8. Internal contradictions or tensions

Rules:
- Do NOT generalize beyond what is in the source
- Tag every quote and claim with its chunk_id in brackets like [chk_xxx_042]
- Flag confidence: high (directly stated), medium (implied), low (inferred)
- Be ruthlessly concise; no filler

Output as markdown with clear section headers."""


def summarize_user(source_name: str, tier: str, chunks: list[dict]) -> str:
    chunks_text = "\n\n---\n\n".join(
        f"[{c['chunk_id']}] ({c['boundary_text'] or c['boundary_type']})\n{c['text']}"
        for c in chunks
    )
    return f"""Source: {source_name}
Tier: {tier}
Total chunks shown: {len(chunks)}

{chunks_text}"""


def doctrine_system(base_prompt: str, advisor_addendum: str, advisor_name: str) -> str:
    return f"""{base_prompt}

## Advisor-Specific Extraction Guidance for {advisor_name}

{advisor_addendum}

## Your Task

You are synthesizing a unified advisor persona from multiple source summaries.

For each claim you extract, you MUST:
- Tag with source tier: (primary), (secondary), or (non-canonical)
- Tag with provenance: [source-derived], [archive-derived], [advisor-inferred], or [thin/uncertain]
- Tag with confidence: high | medium | low
- List supporting chunk_ids like [chk_xxx_042, chk_yyy_007]

Organize output into these exact sections (use these exact headers):
# PERSONA CARD
# CORE FRAMEWORKS
# KEY CONCEPTS & PRINCIPLES
# VOICE & COMMUNICATION STYLE
# VOCABULARY & TERMINOLOGY
# RECURRING THEMES
# SIGNATURE ADVICE PATTERNS
# WHO THIS IS NOT FOR
# BLIND SPOTS & CRITIQUES
# CONTRADICTIONS & TENSIONS
# NOTABLE EXAMPLES & STORIES
# STYLE/VOICE SYNTHESIS
# PATTERNS & ARCHETYPES SYNTHESIS

The last two sections (STYLE/VOICE SYNTHESIS and PATTERNS & ARCHETYPES SYNTHESIS) will be extracted into separate files. Be comprehensive there.

Target: 12–18 pages. Ruthlessly concise. No filler. Preserve decision-relevant nuance, contradictions, and boundary conditions."""


def doctrine_user(summaries: list[dict]) -> str:
    primary = [s for s in summaries if s["tier"] in ("primary",)]
    secondary = [s for s in summaries if s["tier"] == "secondary"]
    non_canonical = [s for s in summaries if s["tier"] == "non-canonical"]

    parts = []
    if primary:
        parts.append("## PRIMARY SOURCES (highest weight)\n")
        for s in primary:
            parts.append(f"### {s['source_name']}\n{s['content']}\n")
    if secondary:
        parts.append("## SECONDARY SOURCES (interview/guest appearances — weight accordingly)\n")
        for s in secondary:
            parts.append(f"### {s['source_name']}\n{s['content']}\n")
    if non_canonical:
        parts.append("## NON-CANONICAL (synthetic / previous bot sessions — use cautiously, do not treat as primary)\n")
        for s in non_canonical:
            parts.append(f"### {s['source_name']}\n{s['content']}\n")

    return "\n".join(parts)


def runtime_context_system() -> str:
    return """You compress advisor doctrine into a short runtime context document.

The output will be injected into every Claude conversation as the "always-on" system context.

Requirements:
- Target: 2,500–3,500 tokens
- No filler, no biography, no preamble
- Behavioral primitives over descriptive prose
- Structure exactly as shown in the user message
- Every section must be actionable for an AI playing this advisor's role"""


def runtime_context_user(advisor_name: str, doctrine: str, style_voice: str, patterns: str) -> str:
    return f"""Compress the following advisor doctrine for {advisor_name} into a runtime context document.

Use EXACTLY these sections with these headers:

# {advisor_name} — Runtime Context

## Persona (5–8 bullets)
## Core Belief
## How to Diagnose (numbered steps)
## Key Frameworks (each: name, 2–4 bullet steps)
## Signature Language
## Common User Profiles (each: name, one-line description, how to respond)
## Anti-Patterns (what NOT to do/suggest)
## When the Framework Breaks (edge cases requiring different approach)
## Tone Guide (3–5 bullets)

---

CANONICAL DOCTRINE:
{doctrine[:8000]}

STYLE/VOICE:
{style_voice[:3000]}

PATTERNS/ARCHETYPES:
{patterns[:3000]}"""
