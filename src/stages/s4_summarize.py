"""Stage 4: Per-source LLM summary (primary + secondary only)."""
import logging
from pathlib import Path

from src.utils.jsonl import read_jsonl, write_jsonl
from src.llm.prompts import summarize_system, summarize_user, load_prompt

logger = logging.getLogger(__name__)


def run(cfg: dict, force: bool = False) -> None:
    output_dir = Path(cfg["paths"]["output"])
    registry_path = output_dir / "source_registry.jsonl"
    chunk_index_path = output_dir / "chunk_index.jsonl"
    summaries_dir = output_dir / "summaries"
    summaries_dir.mkdir(parents=True, exist_ok=True)

    records = read_jsonl(registry_path)
    all_chunks = read_jsonl(chunk_index_path)

    chunks_by_source: dict[str, list[dict]] = {}
    for c in all_chunks:
        chunks_by_source.setdefault(c["source_id"], []).append(c)

    base_prompt = load_prompt(cfg["paths"]["base_prompt"])
    system = summarize_system(base_prompt)

    updated_records = []
    for rec in records:
        source_id = rec["source_id"]
        tier = rec.get("classification", "non-canonical")

        if tier == "non-canonical":
            logger.debug(f"Skipping non-canonical: {rec['filename']}")
            updated_records.append(rec)
            continue

        summary_path = summaries_dir / f"summary_{source_id}.md"
        if not force and rec.get("stage_completed", 0) >= 4 and summary_path.exists():
            logger.debug(f"Skipping already summarized: {rec['filename']}")
            updated_records.append(rec)
            continue

        source_chunks = chunks_by_source.get(source_id, [])
        if not source_chunks:
            logger.warning(f"No chunks for {rec['filename']}, skipping summary")
            updated_records.append(rec)
            continue

        sampled = _sample_chunks(source_chunks, cfg["chunking"]["max_sample_chunks"])
        logger.info(f"Summarizing {rec['filename']} ({len(sampled)}/{len(source_chunks)} chunks)...")

        user_msg = summarize_user(rec["filename"], tier, sampled)

        from src.llm.client import call_with_cache, get_model
        response = call_with_cache(
            model=get_model(cfg, "sonnet_model"),
            system_blocks=[{
                "type": "text",
                "text": system,
                "cache_control": {"type": "ephemeral"},
            }],
            user_message=user_msg,
            max_tokens=cfg["llm"]["max_tokens_summarize"],
        )

        header = f"# Source Summary: {rec['filename']}\n- source_id: {source_id}\n- tier: {tier}\n- chunks: {rec.get('chunk_count', '?')} | tokens: {rec.get('token_count', '?')}\n\n"
        summary_path.write_text(header + response, encoding="utf-8")

        rec["stage_completed"] = max(rec.get("stage_completed", 0), 4)
        updated_records.append(rec)
        logger.info(f"Summary written: {summary_path.name}")

    write_jsonl(registry_path, updated_records)
    logger.info("Stage 4 complete")


def _sample_chunks(chunks: list[dict], max_n: int) -> list[dict]:
    if len(chunks) <= max_n:
        return chunks
    step = len(chunks) / max_n
    return [chunks[int(i * step)] for i in range(max_n)]
