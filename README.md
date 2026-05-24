# Agent Pipeline

A local Python pipeline that converts a mixed archive of source material into a compact, source-aware corpus for an advisor bot. Built for Ramit Sethi but generalizes to any advisor with a config swap.

**What it produces:**
- A `runtime_context.md` (~3k tokens) you can paste directly into any Claude.ai Project as permanent context
- A `project_upload/` folder ready to drag-and-drop into Claude.ai Projects
- An `evidence_index.jsonl` with vector embeddings for a future API-based retrieval bot — no re-run needed when you build that layer

The runtime budget: `runtime_context.md` (~3k tokens always present) + 3–6 retrieved chunks (~500 tokens each) = 5–7k total.

---

## How It Works

Seven discrete stages. Each stage reads from the previous stage's output files, so any stage can be re-run independently without redoing upstream work.

```
sources/          →  [S1 Ingest]  →  source_registry.jsonl
                                      output/ingested/{source_id}.txt

source_registry   →  [S2 Classify]  →  classification fields updated

ingested texts    →  [S3 Chunk]    →  chunk_index.jsonl

chunks            →  [S4 Summarize] →  summaries/{source_id}.md

summaries         →  [S5 Extract]  →  canonical_doctrine.md
                                       style_voice.md
                                       patterns_archetypes.md

doctrine files    →  [S6 Assemble] →  runtime_context.md
                                       evidence_index.jsonl
                                       project_upload/

all artifacts     →  [S7 Report]   →  assembly_report.md
```

### Stage 1 — Ingest

Walks `sources/`, parses every file, counts tokens, and records each source in `source_registry.jsonl`.

- Detects file type by extension and content (calibre ebook exports have YAML frontmatter; podcast transcripts have `EPISODE N` + `[HH:MM:SS]` markers)
- Strips HTML tags from calibre ebook exports automatically
- Computes SHA-256 content hash for incremental processing — unchanged sources are skipped on re-runs
- Flags exact-duplicate sources (same hash) without deleting them
- Supported types: `.txt`, `.md`, `.pdf`, `.docx`, `.html`, `.json`
- Files matching `exclude_patterns` in config are silently skipped

### Stage 2 — Classify

Assigns each source a tier: `primary`, `secondary`, or `non-canonical`.

- Pattern-matched by filename (fnmatch, config-driven) — no LLM needed for known sources
- Falls back to Claude Haiku for any filename that doesn't match a pattern — sends a 500-token sample for classification
- First matching pattern wins; order matters in config

Tiers affect everything downstream: primary sources are summarized first, dominate doctrine extraction, and are the only tier that gets chunks in `project_upload/`.

### Stage 3 — Chunk

Splits ingested text into 400–650 token chunks at semantic boundaries.

- **Ebooks/markdown**: splits at `##` / `###` headers, preserves header as `boundary_text`
- **Podcast transcripts**: splits at `EPISODE N` markers and `[HH:MM:SS]` timestamp blocks
- **Plain text**: splits at paragraph breaks (`\n\n`)
- Oversized segments are split further at sentence boundaries
- No overlap between chunks (research shows overlap adds token cost with no retrieval benefit)
- Each chunk gets 3–5 concept tags via keyword extraction (no LLM)
- Outputs `chunk_index.jsonl` with `chunk_id`, `source_id`, `tier`, `boundary_type`, `token_count`, `concepts`, `char_start/end`

### Stage 4 — Summarize

Generates a structured per-source summary for every primary and secondary source. Non-canonical sources are skipped.

- Samples up to 40 chunks evenly from each source (prevents context overflow for large files like full podcast dumps)
- Each summary extracts: key arguments, tone/voice markers, advice patterns, user archetypes, anti-patterns, representative quotes with `chunk_id` citations, blind spots, internal contradictions
- Uses Claude Sonnet with a cached system prompt (saves tokens on repeated runs)
- Writes one `summaries/summary_{source_id}.md` per source
- Incremental: skips sources already summarized unless `--force`

### Stage 5 — Extract Doctrine

Cross-source synthesis. Takes all per-source summaries and extracts the canonical advisor persona.

- System prompt = `prompts/context_compaction_prompt.md` + `prompts/{advisor}_addendum.md` (Anthropic prompt caching on both)
- Summaries are fed grouped by tier: primary first, secondary second, non-canonical last with an explicit label
- One Sonnet call produces the full doctrine, which is then split by section headers into three files:
  - `canonical_doctrine.md` — full output, every major claim
  - `style_voice.md` — voice, communication style, vocabulary sections only
  - `patterns_archetypes.md` — advice patterns, archetypes, blind spots, contradictions sections only

### Stage 6 — Assemble

Builds the two runtime artifacts.

**`runtime_context.md`** — compressed to ~3k tokens via Sonnet. Contains persona summary, diagnostic framework, 7-step conversational flow, key frameworks, signature moves, common archetypes, and when the framework breaks. Designed to be injected directly into any prompt.

**`evidence_index.jsonl`** — one record per chunk. Adds:
- `category` inferred by keyword heuristic: `framework`, `concept`, `anti-pattern`, `example`, `voice`, `archetype`, `general`
- `tags` from the chunk's concept list
- `embedding` via local sentence-transformers (`all-MiniLM-L6-v2`) for primary + secondary tiers

**`project_upload/`** — ready for Claude.ai Projects drag-and-drop:
```
project_upload/
├── 00_runtime_context.md       ← set as project instructions or first doc
├── 01_canonical_doctrine.md
├── 02_style_voice.md
├── 03_patterns_archetypes.md
└── chunks/                     ← top-N high-confidence primary chunks by category
```

### Stage 7 — Report

Writes `assembly_report.md` with:
- Token budget table (total ingested, runtime context size, avg chunk size, retrieval budget estimates)
- Source tier distribution
- Evidence confidence and category distributions
- Flagged duplicate sources
- Claude.ai Projects upload checklist
- Retrieval strategy notes for the future API bot

---

## Setup

**Prerequisites:** Python 3.11+

```bash
# Clone and create a virtual environment
git clone <repo>
cd agent-pipeline
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Add your Anthropic API key
cp .env.example .env
# Edit .env and set: ANTHROPIC_API_KEY=sk-ant-...
```

---

## Usage

```bash
# Activate the venv first
source .venv/bin/activate

# Full pipeline from scratch
python run_pipeline.py --config config/ramit-sethi.yaml

# Resume from a specific stage (useful after an API error in stage 4+)
python run_pipeline.py --config config/ramit-sethi.yaml --from-stage 4

# Run only one stage
python run_pipeline.py --config config/ramit-sethi.yaml --only-stage 5

# Force re-process even unchanged sources
python run_pipeline.py --config config/ramit-sethi.yaml --force
```

Stages 1–3 (ingest, classify, chunk) require no API key. You can validate your source setup before spending any tokens.

---

## Adding Sources

Drop files into `sources/`. Supported: `.txt`, `.md`, `.pdf`, `.docx`, `.html`, `.json`.

On the next run, only new or changed files are processed. Existing summaries and doctrine are preserved unless you pass `--force`.

---

## Using the Output

### Immediate — Claude.ai Projects

1. Open Claude.ai → Projects → New Project
2. Upload files from `output/{advisor}/project_upload/` in this order:
   - `00_runtime_context.md` — set as **project instructions** or pin as the first document
   - `01_canonical_doctrine.md`
   - `02_style_voice.md`
   - `03_patterns_archetypes.md`
   - Select relevant chunks from `project_upload/chunks/` by topic
3. Claude will reference the persona context in every conversation automatically

### Future — API Bot

`evidence_index.jsonl` is already built with embeddings. When you wire up an API bot:

1. At query time, keyword-match on `tags` / `category` fields for fast recall
2. Cosine similarity on `embedding` vectors for ranking (top 3–6)
3. Inject: `runtime_context.md` + retrieved chunks as context
4. Filter: prefer `tier=primary` and `confidence=high`

No re-run of the pipeline needed — the index is ready.

---

## Adding a New Advisor

1. Copy and edit the config:
   ```bash
   cp config/ramit-sethi.yaml config/jane-expert.yaml
   ```
   Update `advisor.name`, `advisor.slug`, `advisor.domain`, `classification_rules`, and `exclude_patterns`.

2. Optionally create `prompts/jane_expert_addendum.md` with advisor-specific extraction guidance. Set `paths.advisor_addendum` in the config to point to it.

3. Add sources to `sources/` (or point `paths.sources` to a different directory).

4. Run:
   ```bash
   python run_pipeline.py --config config/jane-expert.yaml
   ```

Output lands in `output/jane-expert/` — fully isolated from other advisors.

---

## Output Files Reference

| File | Stage | Description |
|------|-------|-------------|
| `source_registry.jsonl` | S1–S2 | One record per source: metadata, classification, token count, dedup flags |
| `output/ingested/{id}.txt` | S1 | Parsed, cleaned text per source |
| `chunk_index.jsonl` | S3 | All chunks: text, token count, boundary type, concepts, tier |
| `summaries/summary_{id}.md` | S4 | Per-source LLM summary with chunk_id citations |
| `canonical_doctrine.md` | S5 | Full cross-source doctrine extraction |
| `style_voice.md` | S5 | Voice, style, vocabulary sections |
| `patterns_archetypes.md` | S5 | Advice patterns, archetypes, blind spots |
| `runtime_context.md` | S6 | Compressed persona context (~3k tokens) for prompt injection |
| `evidence_index.jsonl` | S6 | All chunks with category, tags, and embeddings |
| `project_upload/` | S6 | Ready-to-upload folder for Claude.ai Projects |
| `assembly_report.md` | S7 | Token budgets, distributions, quality flags |

---

## Config Reference (`config/*.yaml`)

```yaml
advisor:
  name: "Ramit Sethi"
  slug: "ramit-sethi"           # used as output directory name
  domain: "Personal Finance"

paths:
  sources: "sources"            # input directory
  output: "output/ramit-sethi"  # all outputs land here
  base_prompt: "prompts/context_compaction_prompt.md"
  advisor_addendum: "prompts/ramit_addendum.md"

classification_rules:           # fnmatch patterns, first match wins
  - pattern: "text-book-*"
    tier: primary
  - pattern: "tim-ferriss-*"
    tier: secondary
  - pattern: "text-chatgptconvo-*"
    tier: non-canonical

exclude_patterns:               # skipped entirely, no processing
  - "text-miseenmoney-*"

llm:
  sonnet_model: "claude-sonnet-4-6-20250514"
  haiku_model: "claude-haiku-4-5-20251001"
  max_tokens_summarize: 4000
  max_tokens_extraction: 8000
  max_tokens_runtime_context: 4000

chunking:
  target_tokens: 500
  min_tokens: 150
  max_tokens: 650
  overlap_tokens: 0             # no overlap by design
  tokenizer: "cl100k_base"
  max_sample_chunks: 40         # max chunks sampled per source in Stage 4

embeddings:
  model: "all-MiniLM-L6-v2"    # local, no API cost
  tiers_to_embed: ["primary", "secondary"]

project_upload:
  top_chunks_per_category: 10   # how many chunks per category to export
  include_tiers: ["primary"]
  min_confidence: "high"
```
