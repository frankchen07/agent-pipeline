"""Stage 7: Generate assembly report with token budgets and quality metrics."""
import logging
from collections import Counter
from pathlib import Path

import tiktoken

from src.utils.jsonl import read_jsonl

logger = logging.getLogger(__name__)


def run(cfg: dict) -> None:
    output_dir = Path(cfg["paths"]["output"])
    report_path = output_dir / "assembly_report.md"
    enc = tiktoken.get_encoding(cfg["chunking"]["tokenizer"])

    records = read_jsonl(output_dir / "source_registry.jsonl")
    chunks = read_jsonl(output_dir / "chunk_index.jsonl")
    evidence = read_jsonl(output_dir / "evidence_index.jsonl")

    runtime_path = output_dir / "runtime_context.md"
    runtime_tokens = len(enc.encode(runtime_path.read_text(encoding="utf-8"))) if runtime_path.exists() else 0

    # Token budget
    total_ingested = sum(r.get("token_count", 0) for r in records)
    avg_chunk_tokens = sum(c["token_count"] for c in chunks) / len(chunks) if chunks else 0
    retrieval_budget_3 = runtime_tokens + 3 * avg_chunk_tokens
    retrieval_budget_6 = runtime_tokens + 6 * avg_chunk_tokens

    # Source tier distribution
    tier_counts = Counter(r.get("classification", "unknown") for r in records)

    # Confidence distribution in evidence index
    conf_counts = Counter(e.get("confidence", "unknown") for e in evidence)

    # Category distribution
    cat_counts = Counter(e.get("category", "unknown") for e in evidence)

    # Duplicate flags
    dupes = [r for r in records if r.get("dedup_flag")]

    # Chunks with no embedding
    no_embed = [e for e in evidence if e.get("embedding") is None]

    # Project upload files
    upload_dir = output_dir / "project_upload"
    upload_files = sorted(upload_dir.rglob("*.md")) if upload_dir.exists() else []

    lines = [
        f"# Assembly Report: {cfg['advisor']['name']}",
        "",
        "## Token Budget",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Total tokens ingested | {total_ingested:,} |",
        f"| runtime_context.md tokens | {runtime_tokens:,} |",
        f"| Avg chunk tokens | {avg_chunk_tokens:.0f} |",
        f"| Total chunks | {len(chunks)} |",
        f"| Runtime budget (3 chunks) | {retrieval_budget_3:.0f} tokens |",
        f"| Runtime budget (6 chunks) | {retrieval_budget_6:.0f} tokens |",
        "",
        "## Source Tier Distribution",
        f"| Tier | Count |",
        f"|------|-------|",
    ] + [f"| {tier} | {count} |" for tier, count in sorted(tier_counts.items())] + [
        "",
        "## Evidence Index",
        f"| Confidence | Count |",
        f"|------------|-------|",
    ] + [f"| {conf} | {count} |" for conf, count in sorted(conf_counts.items())] + [
        "",
        f"| Category | Count |",
        f"|----------|-------|",
    ] + [f"| {cat} | {count} |" for cat, count in sorted(cat_counts.items())] + [
        "",
        f"Chunks with embeddings: {len(evidence) - len(no_embed)}/{len(evidence)}",
        "",
        "## Flagged Issues",
    ]

    if dupes:
        lines.append(f"\n### Duplicate Sources ({len(dupes)})")
        for r in dupes:
            lines.append(f"- {r['filename']}: {r.get('dedup_note', '')}")
    else:
        lines.append("No duplicate sources detected.")

    lines += [
        "",
        "## Claude.ai Projects Upload Checklist",
        "",
        "Upload these files to your Claude Project in order:",
        "1. `project_upload/00_runtime_context.md` — set as **project instructions** or first doc",
        "2. `project_upload/01_canonical_doctrine.md`",
        "3. `project_upload/02_style_voice.md`",
        "4. `project_upload/03_patterns_archetypes.md`",
        "5. Select top chunks from `project_upload/chunks/` by topic relevance",
        "",
        "### Available Upload Files",
    ] + [f"- `{f.relative_to(output_dir)}`" for f in upload_files] + [
        "",
        "## Runtime Retrieval Strategy",
        "",
        "**Immediate (Claude.ai Projects):**",
        "- runtime_context.md always loaded as project context",
        "- Canonical doctrine + style/voice docs for deep reference",
        "- Manually include relevant chunks when asking domain-specific questions",
        "",
        "**Future (API bot):**",
        "- At query time: keyword match on `tags`/`category` in evidence_index.jsonl for recall",
        "- Cosine similarity on embeddings for ranking (top-6)",
        "- Inject: runtime_context.md + top-3 to 6 retrieved chunks",
        "- Filter: prefer `tier=primary` and `confidence=high`",
    ]

    report_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info(f"Stage 7 complete: assembly_report.md written")
    logger.info(f"  Total ingested: {total_ingested:,} tokens")
    logger.info(f"  Runtime budget (3 chunks): {retrieval_budget_3:.0f} tokens")
    logger.info(f"  Runtime budget (6 chunks): {retrieval_budget_6:.0f} tokens")
