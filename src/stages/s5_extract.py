"""Stage 5: Cross-source doctrine extraction."""
import logging
import re
from pathlib import Path

from src.utils.jsonl import read_jsonl, write_jsonl
from src.llm.prompts import doctrine_system, doctrine_user, load_prompt

logger = logging.getLogger(__name__)

# Sections to split into style_voice.md
_STYLE_HEADERS = {"# VOICE & COMMUNICATION STYLE", "# VOCABULARY & TERMINOLOGY", "# STYLE/VOICE SYNTHESIS"}
# Sections to split into patterns_archetypes.md
_PATTERNS_HEADERS = {
    "# SIGNATURE ADVICE PATTERNS", "# WHO THIS IS NOT FOR",
    "# BLIND SPOTS & CRITIQUES", "# CONTRADICTIONS & TENSIONS",
    "# RECURRING THEMES", "# PATTERNS & ARCHETYPES SYNTHESIS",
}


def run(cfg: dict, force: bool = False) -> None:
    output_dir = Path(cfg["paths"]["output"])
    summaries_dir = output_dir / "summaries"

    doctrine_path = output_dir / "canonical_doctrine.md"
    style_path = output_dir / "style_voice.md"
    patterns_path = output_dir / "patterns_archetypes.md"

    if not force and doctrine_path.exists() and style_path.exists() and patterns_path.exists():
        logger.info("Stage 5: doctrine files already exist, skipping (use --force to re-run)")
        return

    records = read_jsonl(output_dir / "source_registry.jsonl")

    summaries = []
    for rec in records:
        tier = rec.get("classification", "non-canonical")
        summary_path = summaries_dir / f"summary_{rec['source_id']}.md"
        if summary_path.exists():
            summaries.append({
                "source_id": rec["source_id"],
                "source_name": rec["filename"],
                "tier": tier,
                "content": summary_path.read_text(encoding="utf-8"),
            })

    if not summaries:
        logger.error("No summaries found — run Stage 4 first")
        return

    advisor_name = cfg["advisor"]["name"]
    base_prompt = load_prompt(cfg["paths"]["base_prompt"])
    addendum = load_prompt(cfg["paths"]["advisor_addendum"]) if Path(cfg["paths"]["advisor_addendum"]).exists() else ""

    system = doctrine_system(base_prompt, addendum, advisor_name)
    user_msg = doctrine_user(summaries)

    logger.info(f"Extracting doctrine from {len(summaries)} source summaries...")

    from src.llm.client import call_with_cache, get_model
    response = call_with_cache(
        model=get_model(cfg, "sonnet_model"),
        system_blocks=[{
            "type": "text",
            "text": system,
            "cache_control": {"type": "ephemeral"},
        }],
        user_message=user_msg,
        max_tokens=cfg["llm"]["max_tokens_extraction"],
    )

    sections = _split_sections(response)
    full_doctrine = response

    style_sections = {k: v for k, v in sections.items() if k.upper() in _STYLE_HEADERS}
    patterns_sections = {k: v for k, v in sections.items() if k.upper() in _PATTERNS_HEADERS}

    doctrine_path.write_text(full_doctrine, encoding="utf-8")
    style_path.write_text(_sections_to_md(style_sections), encoding="utf-8")
    patterns_path.write_text(_sections_to_md(patterns_sections), encoding="utf-8")

    logger.info("Stage 5 complete: doctrine, style_voice, patterns_archetypes written")


def _split_sections(text: str) -> dict[str, str]:
    """Split markdown by top-level # headers into {header: content} dict."""
    pattern = re.compile(r"^(#\s+.+)$", re.MULTILINE)
    positions = [(m.start(), m.group(1)) for m in pattern.finditer(text)]
    sections = {}
    for i, (pos, header) in enumerate(positions):
        end = positions[i + 1][0] if i + 1 < len(positions) else len(text)
        sections[header.strip()] = text[pos:end].strip()
    return sections


def _sections_to_md(sections: dict[str, str]) -> str:
    return "\n\n".join(sections.values())
