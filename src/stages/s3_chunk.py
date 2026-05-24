"""Stage 3: Chunk sources by semantic boundaries."""
import logging
from pathlib import Path

from src.chunking.chunker import chunk_text, Chunk
from src.utils.jsonl import read_jsonl, write_jsonl

logger = logging.getLogger(__name__)


def run(cfg: dict, force: bool = False) -> list[dict]:
    output_dir = Path(cfg["paths"]["output"])
    registry_path = output_dir / "source_registry.jsonl"
    chunk_index_path = output_dir / "chunk_index.jsonl"
    ingested_dir = output_dir / "ingested"

    records = read_jsonl(registry_path)
    chunk_cfg = cfg["chunking"]

    existing_chunks = read_jsonl(chunk_index_path)
    existing_by_source: dict[str, list[dict]] = {}
    for c in existing_chunks:
        existing_by_source.setdefault(c["source_id"], []).append(c)

    all_chunks: list[dict] = []
    updated_records = []

    for rec in records:
        source_id = rec["source_id"]

        if not force and rec.get("stage_completed", 0) >= 3 and source_id in existing_by_source:
            all_chunks.extend(existing_by_source[source_id])
            updated_records.append(rec)
            continue

        ingested_path = ingested_dir / f"{source_id}.txt"
        if not ingested_path.exists():
            logger.warning(f"No ingested text for {rec['filename']}, skipping chunking")
            updated_records.append(rec)
            continue

        text = ingested_path.read_text(encoding="utf-8")
        structure = _get_structure(rec, text)

        chunks = chunk_text(
            text=text,
            structure=structure,
            media_type=rec["media_type"],
            target=chunk_cfg["target_tokens"],
            min_tokens=chunk_cfg["min_tokens"],
            max_tokens=chunk_cfg["max_tokens"],
            tokenizer=chunk_cfg["tokenizer"],
        )

        chunk_records = []
        for c in chunks:
            chunk_records.append({
                "chunk_id": f"chk_{source_id}_{c.seq:04d}",
                "source_id": source_id,
                "seq": c.seq,
                "tier": rec["classification"],
                "boundary_type": c.boundary_type,
                "boundary_text": c.boundary_text,
                "text": c.text,
                "token_count": c.token_count,
                "char_start": c.char_start,
                "char_end": c.char_end,
                "provenance": "source-derived",
                "confidence": rec.get("confidence", "high"),
                "concepts": c.concepts,
            })

        all_chunks.extend(chunk_records)
        rec["chunk_count"] = len(chunk_records)
        rec["stage_completed"] = max(rec.get("stage_completed", 0), 3)
        logger.info(f"Chunked {rec['filename']}: {len(chunk_records)} chunks")
        updated_records.append(rec)

    write_jsonl(chunk_index_path, all_chunks)
    write_jsonl(registry_path, updated_records)
    logger.info(f"Stage 3 complete: {len(all_chunks)} total chunks")
    return all_chunks


def _get_structure(rec: dict, text: str) -> list[dict]:
    """Re-derive boundary structure from the ingested text based on media_type."""
    media_type = rec.get("media_type", "txt")
    if media_type == "ebook_txt":
        from src.parsing.ebook import _extract_structure
        return _extract_structure(text)
    if media_type == "markdown":
        from src.parsing.md import _extract_structure
        return _extract_structure(text)
    if media_type == "podcast_transcript":
        from src.parsing.txt import _extract_structure
        return _extract_structure(text, "podcast_transcript")
    if media_type == "pdf":
        from src.parsing.pdf import _extract_structure
        return _extract_structure(text)
    # plain txt
    from src.parsing.txt import _extract_structure
    return _extract_structure(text, "txt")
