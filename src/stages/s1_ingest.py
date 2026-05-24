"""Stage 1: Ingest source files, parse, hash, count tokens."""
import fnmatch
import logging
from datetime import datetime, timezone
from pathlib import Path

import tiktoken

from src.utils.hashing import file_hash, load_state, save_state, source_id_from_filename
from src.utils.jsonl import write_jsonl, read_jsonl

logger = logging.getLogger(__name__)


def run(cfg: dict, force: bool = False) -> list[dict]:
    sources_dir = Path(cfg["paths"]["sources"])
    output_dir = Path(cfg["paths"]["output"])
    output_dir.mkdir(parents=True, exist_ok=True)

    registry_path = output_dir / "source_registry.jsonl"
    ingested_dir = output_dir / "ingested"
    ingested_dir.mkdir(parents=True, exist_ok=True)

    state = load_state(output_dir)
    exclude_patterns = cfg.get("exclude_patterns", [])
    enc = tiktoken.get_encoding(cfg["chunking"]["tokenizer"])

    existing_registry = {r["source_id"]: r for r in read_jsonl(registry_path)}
    records = []

    source_files = sorted(sources_dir.iterdir())
    logger.info(f"Found {len(source_files)} files in {sources_dir}")

    for filepath in source_files:
        if not filepath.is_file():
            continue
        filename = filepath.name

        if _is_excluded(filename, exclude_patterns):
            logger.debug(f"Skipping excluded: {filename}")
            continue

        ext = filepath.suffix.lower()
        if ext not in (".txt", ".md", ".pdf", ".docx", ".html", ".json"):
            logger.debug(f"Skipping unsupported type: {filename}")
            continue

        source_id = source_id_from_filename(filename)
        current_hash = file_hash(filepath)

        # Incremental: skip if unchanged and already past stage 1
        src_state = state.get(source_id, {})
        if (
            not force
            and src_state.get("hash") == current_hash
            and src_state.get("stage_completed", 0) >= 1
            and source_id in existing_registry
        ):
            logger.debug(f"Skipping unchanged: {filename}")
            records.append(existing_registry[source_id])
            continue

        logger.info(f"Ingesting: {filename}")
        parsed = _parse_file(filepath, ext)
        text = parsed["text"]
        token_count = len(enc.encode(text))

        # Save raw text for downstream stages
        (ingested_dir / f"{source_id}.txt").write_text(text, encoding="utf-8")

        record = {
            "source_id": source_id,
            "filename": filename,
            "filepath": str(filepath),
            "media_type": parsed["media_type"],
            "classification": None,
            "confidence": None,
            "classification_method": None,
            "content_hash": current_hash,
            "token_count": token_count,
            "chunk_count": 0,
            "dedup_flag": False,
            "dedup_note": "",
            "ingest_ts": datetime.now(timezone.utc).isoformat(),
            "stage_completed": 1,
        }
        records.append(record)

        state[source_id] = {"hash": current_hash, "stage_completed": 1, "last_processed": record["ingest_ts"]}

    _flag_duplicates(records)
    write_jsonl(registry_path, records)
    save_state(output_dir, state)

    logger.info(f"Stage 1 complete: {len(records)} sources ingested")
    return records


def _is_excluded(filename: str, patterns: list[str]) -> bool:
    return any(fnmatch.fnmatch(filename, p) for p in patterns)


def _parse_file(filepath: Path, ext: str) -> dict:
    if ext == ".pdf":
        from src.parsing.pdf import parse
        return parse(filepath)
    if ext == ".md":
        from src.parsing.md import parse
        return parse(filepath)
    if ext == ".txt":
        # Detect ebook (YAML frontmatter) vs plain/podcast
        raw = filepath.read_text(encoding="utf-8", errors="replace")
        if raw.startswith("---"):
            from src.parsing.ebook import parse
            return parse(filepath)
        from src.parsing.txt import parse
        return parse(filepath)
    # Fallback: read as plain text
    return {"text": filepath.read_text(encoding="utf-8", errors="replace"), "media_type": "txt"}


def _flag_duplicates(records: list[dict]) -> None:
    seen_hashes: dict[str, str] = {}
    for rec in records:
        h = rec["content_hash"]
        if h in seen_hashes:
            rec["dedup_flag"] = True
            rec["dedup_note"] = f"Same content hash as {seen_hashes[h]}"
            logger.warning(f"Duplicate detected: {rec['filename']} == {seen_hashes[h]}")
        else:
            seen_hashes[h] = rec["filename"]
