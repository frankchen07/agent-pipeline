#!/usr/bin/env python3
"""CLI wrapper so Ramit subagent can query the knowledge base without MCP."""
import sys
import os

sys.path.insert(0, "/Users/snappo/.openclaw/github/agent-pipeline")
os.chdir("/Users/snappo/.openclaw/github/agent-pipeline")

from src.server.mcp_server import query_ramit  # noqa: E402

query = sys.argv[1] if len(sys.argv) > 1 else ""
top_k = int(sys.argv[2]) if len(sys.argv) > 2 else 8
print(query_ramit(query, top_k))
