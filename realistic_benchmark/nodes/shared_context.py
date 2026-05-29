"""
Shared context node — runs once per seed before the 3 parallel tier sub-pipelines.

Builds a hard-negative candidate pool using chromadb vector similarity search.
"""

from __future__ import annotations

import os
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions

N_HARD_NEGATIVES = 13

_DEFAULT_EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "text-embedding-3-large")


# ---------------------------------------------------------------------------
# Retriever setup (called once at pipeline build time)
# ---------------------------------------------------------------------------

def build_retriever(
    all_tools: list[dict],
    cache_dir: str,
    embedding_model: str | None = None,
):
    """Build (or load from cache) a chromadb collection with tool embeddings.

    Embedding configuration (via .env or environment variables):
        OPENAI_API_KEY   — API key (falls back to LITELLM_API_KEY, then "unused")
        OPENAI_API_BASE  — Proxy URL for embeddings (falls back to LITELLM_BASE_URL)
        EMBEDDING_MODEL  — Embedding model name (default: text-embedding-3-large)
    """
    model = embedding_model or _DEFAULT_EMBEDDING_MODEL

    api_key = (
        os.environ.get("OPENAI_API_KEY", "").strip()
        or os.environ.get("LITELLM_API_KEY", "").strip()
        or "unused"
    )
    api_base = (
        os.environ.get("OPENAI_API_BASE", "").strip()
        or os.environ.get("LITELLM_BASE_URL", "").strip()
        or None
    )

    ef_kwargs: dict = {"api_key": api_key, "model_name": model}
    if api_base:
        ef_kwargs["api_base"] = api_base

    ef = embedding_functions.OpenAIEmbeddingFunction(**ef_kwargs)
    db_path = str(Path(cache_dir) / "chromadb")
    client = chromadb.PersistentClient(path=db_path)

    try:
        collection = client.get_collection("tools", embedding_function=ef)
        if collection.count() == len(all_tools):
            print(f"Using cached tool embeddings ({collection.count()} tools) from {db_path}")
            return collection
        print(f"Tool count mismatch ({collection.count()} cached vs {len(all_tools)} new). Rebuilding...")
        client.delete_collection("tools")
    except Exception:
        pass

    print(f"Indexing {len(all_tools)} tools into chromadb at {db_path} ...")
    collection = client.create_collection("tools", embedding_function=ef)

    batch_size = 100
    for i in range(0, len(all_tools), batch_size):
        batch = all_tools[i : i + batch_size]
        collection.add(
            ids=[t["tool_name"] for t in batch],
            documents=[t.get("tool_description", "") for t in batch],
            metadatas=[{"tool_name": t["tool_name"]} for t in batch],
        )
        print(f"  Indexed {min(i + batch_size, len(all_tools))}/{len(all_tools)} tools")

    return collection


# ---------------------------------------------------------------------------
# Hard-negative retrieval
# ---------------------------------------------------------------------------

def _get_hard_negative_names(retriever, seed_tool: dict, n: int = N_HARD_NEGATIVES) -> list[str]:
    results = retriever.query(
        query_texts=[seed_tool.get("tool_description", seed_tool["tool_name"])],
        n_results=n + 5,  # over-fetch to account for seed exclusion
    )
    names = results["ids"][0]
    return [name for name in names if name != seed_tool["tool_name"]][:n]


def _build_candidate_pool(
    seed_tool: dict,
    hard_negative_names: list[str],
    all_tools_by_name: dict[str, dict],
) -> list[dict]:
    pool = [{**seed_tool, "pool_tier": "gt"}]
    seen = {seed_tool["tool_name"]}

    for name in hard_negative_names:
        if len(pool) > N_HARD_NEGATIVES:
            break
        if name in seen:
            continue
        full_tool = all_tools_by_name.get(name)
        if full_tool is None:
            continue
        pool.append({**full_tool, "pool_tier": "hard_negative"})
        seen.add(name)

    return pool


# ---------------------------------------------------------------------------
# Shared context node
# ---------------------------------------------------------------------------

class SharedContextNode:
    """Builds the hard-negative candidate pool for a seed tool."""

    def __init__(self, all_tools: list[dict], retriever):
        self._retriever = retriever
        self._all_tools_by_name = {t["tool_name"]: t for t in all_tools}

    def __call__(self, state: dict) -> dict:
        seed_tool = state["seed_tool"]
        hard_negative_names = _get_hard_negative_names(self._retriever, seed_tool)
        candidate_pool = _build_candidate_pool(seed_tool, hard_negative_names, self._all_tools_by_name)
        return {"candidate_pool": candidate_pool}
