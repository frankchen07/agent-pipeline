"""Stage 2: Classify sources by tier using filename patterns + LLM fallback."""
import fnmatch
import json
import logging
from pathlib import Path

from src.utils.jsonl import read_jsonl, write_jsonl

logger = logging.getLogger(__name__)


def run(cfg: dict, force: bool = False) -> list[dict]:
    output_dir = Path(cfg["paths"]["output"])
    registry_path = output_dir / "source_registry.jsonl"
    rules = cfg.get("classification_rules", [])

    records = read_jsonl(registry_path)
    updated = []

    for rec in records:
        if not force and rec.get("classification") and rec.get("stage_completed", 0) >= 2:
            updated.append(rec)
            continue

        filename = rec["filename"]
        tier, confidence, method = _classify_by_pattern(filename, rules)

        if tier is None:
            # LLM fallback
            tier, confidence = _classify_with_llm(cfg, rec)
            method = "llm"

        rec["classification"] = tier
        rec["confidence"] = confidence
        rec["classification_method"] = method
        rec["stage_completed"] = max(rec.get("stage_completed", 0), 2)
        logger.info(f"Classified {filename} → {tier} ({confidence}, {method})")
        updated.append(rec)

    write_jsonl(registry_path, updated)
    logger.info(f"Stage 2 complete: {len(updated)} sources classified")
    return updated


def _classify_by_pattern(filename: str, rules: list[dict]) -> tuple[str | None, str | None, str | None]:
    for rule in rules:
        pattern = rule.get("pattern", "")
        if fnmatch.fnmatch(filename, pattern):
            return rule["tier"], "high", "filename_pattern"
    return None, None, None


def _classify_with_llm(cfg: dict, rec: dict) -> tuple[str, str]:
    from src.llm.client import call_simple, get_model
    from src.llm.prompts import classify_system, classify_user

    ingested_path = Path(cfg["paths"]["output"]) / "ingested" / f"{rec['source_id']}.txt"
    if not ingested_path.exists():
        return "non-canonical", "low"

    sample = ingested_path.read_text(encoding="utf-8")[:2000]
    response = call_simple(
        model=get_model(cfg, "haiku_model"),
        system=classify_system(),
        user=classify_user(rec["filename"], sample),
        max_tokens=cfg["llm"]["max_tokens_classify"],
    )
    try:
        data = json.loads(response.strip())
        return data["classification"], data.get("confidence", "medium")
    except (json.JSONDecodeError, KeyError):
        logger.warning(f"LLM classification parse failed for {rec['filename']}: {response[:200]}")
        return "non-canonical", "low"
