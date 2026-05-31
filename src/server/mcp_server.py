"""MCP server exposing per-advisor RAG as queryable tools for OpenClaw."""
import logging
import os
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from fastmcp import FastMCP
from starlette.requests import Request
from starlette.responses import JSONResponse

from src.utils.embeddings import embed_text, warm_model
from src.utils.jsonl import read_jsonl

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_OUTPUT_DIR = Path(os.getenv("AGENT_PIPELINE_OUTPUT_DIR", "output/ramit-sethi"))
_PORT = int(os.getenv("MCP_PORT", "8000"))


def _load_index(output_dir: Path) -> tuple[list[dict], np.ndarray]:
    records = read_jsonl(output_dir / "evidence_index.jsonl")
    valid = [r for r in records if r.get("embedding")]
    if not valid:
        raise RuntimeError(f"No embedded chunks found in {output_dir}/evidence_index.jsonl — run Stage 6 first")
    embeddings = np.array([r["embedding"] for r in valid], dtype=np.float32)
    skipped = len(records) - len(valid)
    logger.info(f"Loaded {len(valid)} chunks" + (f" ({skipped} without embeddings skipped)" if skipped else ""))
    return valid, embeddings


def _load_runtime_context(output_dir: Path) -> str:
    path = output_dir / "runtime_context.md"
    if not path.exists():
        logger.warning(f"runtime_context.md not found at {path}")
        return ""
    return path.read_text(encoding="utf-8")


def _cosine_top_k(query_vec: np.ndarray, embeddings: np.ndarray, k: int) -> list[int]:
    # embeddings are already L2-normalized from sentence-transformers
    scores = embeddings @ query_vec
    return np.argsort(scores)[::-1][:k].tolist()


def _format_chunk(rec: dict) -> str:
    tags = ", ".join(rec.get("tags", []))
    meta = f"[{rec.get('category', 'general')}] {tags} | {rec.get('tier', '')} | {rec.get('confidence', '')}"
    return f"### {meta}\n{rec['text']}"


# --- Load index at startup ---
_records, _embeddings = _load_index(_OUTPUT_DIR)
_runtime_context = _load_runtime_context(_OUTPUT_DIR)
_context_seeded = False

mcp = FastMCP("agent-pipeline")


@mcp.tool(
    description=(
        "Retrieve Ramit Sethi's knowledge on personal finance, earning more, "
        "negotiation, psychology of money, career, and intentional life design. "
        "Always call this before answering questions in Ramit's domain."
    )
)
def query_ramit(query: str, top_k: int = 6) -> str:
    """Query Ramit Sethi's knowledge base. Returns core persona context + most relevant source chunks."""
    global _context_seeded
    logger.info(f"query_ramit called: {query!r} (context_seeded={_context_seeded})")
    q_vec = np.array(embed_text(query), dtype=np.float32)
    top_indices = _cosine_top_k(q_vec, _embeddings, top_k)
    chunks = "\n\n".join(_format_chunk(_records[i]) for i in top_indices)

    if not _context_seeded:
        _context_seeded = True
        return (
            "YOU ARE RAMIT SETHI. Speak exclusively in first person. "
            "Never say 'Ramit would say' or 'Ramit believes' or refer to yourself in third person. "
            "The content below is YOUR knowledge and perspective.\n\n"
            f"## Core Context\n{_runtime_context}\n\n## Relevant Knowledge\n{chunks}"
        )

    return f"## Relevant Knowledge\n{chunks}"


@mcp.custom_route("/.well-known/oauth-authorization-server", methods=["GET"])
async def oauth_metadata(request: Request) -> JSONResponse:
    return JSONResponse({"issuer": f"http://localhost:{_PORT}", "response_types_supported": []})


if __name__ == "__main__":
    logger.info(f"Starting MCP server on port {_PORT} (output dir: {_OUTPUT_DIR})")
    logger.info("Pre-warming embedding model...")
    warm_model()
    logger.info("Embedding model ready.")
    mcp.run(transport="http", path="/sse", host="0.0.0.0", port=_PORT)
