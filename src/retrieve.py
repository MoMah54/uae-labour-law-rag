"""
Search over the chunks table, with three modes you can compare in evaluation:

    vector        original baseline: the top_k most similar chunks (duplicates allowed)
    vector_dedup  vector search over more candidates, keeping the best chunk per article
    rerank        vector search for candidates, then a cross-encoder reranker re-orders them,
                  keeping the best chunk per article  (default: 10 candidates, 256 tokens each,
                  chosen by evaluation as the best accuracy and speed trade-off)

Choose with RETRIEVAL_MODE in .env, or --retrieval-mode in eval/run_eval.py.

Usage:
    python src/retrieve.py "How long do I have to file a claim for unpaid wages?"
"""

import os
import sys
from functools import lru_cache

from dotenv import load_dotenv

# Load .env BEFORE importing sentence_transformers, so settings like
# HF_HUB_OFFLINE=1 (use the cached model, skip checking Hugging Face) take effect.
load_dotenv()

import psycopg  # noqa: E402
from pgvector.psycopg import register_vector  # noqa: E402
from sentence_transformers import CrossEncoder, SentenceTransformer  # noqa: E402

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/labour_law")
MODEL_NAME = "BAAI/bge-m3"
RERANKER_NAME = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
DEFAULT_TOP_K = 6
CANDIDATES = int(os.getenv("RERANK_CANDIDATES", "10"))
RERANK_MAX_LENGTH = int(os.getenv("RERANK_MAX_LENGTH", "256"))  # tokens the reranker reads per chunk
MODES = ("vector", "vector_dedup", "rerank")


def current_mode() -> str:
    mode = os.getenv("RETRIEVAL_MODE", "rerank").lower()
    if mode not in MODES:
        raise ValueError(f"RETRIEVAL_MODE must be one of {MODES}, got '{mode}'")
    return mode


@lru_cache(maxsize=1)
def get_model() -> SentenceTransformer:
    """Load the embedding model once and reuse it (loading takes several seconds)."""
    model = SentenceTransformer(MODEL_NAME)
    model.max_seq_length = 1024
    return model


@lru_cache(maxsize=1)
def get_reranker() -> CrossEncoder:
    """Load the reranker once. First use downloads about 2 GB, then it is cached."""
    return CrossEncoder(RERANKER_NAME, max_length=RERANK_MAX_LENGTH)


def warm_up() -> None:
    """Load every model the current mode needs, so the first request isn't slow."""
    get_model()
    if current_mode() == "rerank":
        get_reranker()


def vector_search(question: str, limit: int) -> list[dict]:
    qvec = get_model().encode(question, normalize_embeddings=True)
    with psycopg.connect(DATABASE_URL) as conn:
        register_vector(conn)
        rows = conn.execute(
            """SELECT id, text, article_number, part, total_parts, language, page,
                      1 - (embedding <=> %s) AS similarity
               FROM chunks
               ORDER BY embedding <=> %s
               LIMIT %s""",
            (qvec, qvec, limit),
        ).fetchall()
    keys = ["id", "text", "article_number", "part", "total_parts", "language", "page", "similarity"]
    return [dict(zip(keys, row)) for row in rows]


def best_per_article(chunks: list[dict], top_k: int) -> list[dict]:
    """Chunks are already sorted best first; keep the first chunk seen for each article."""
    seen, kept = set(), []
    for chunk in chunks:
        if chunk["article_number"] not in seen:
            seen.add(chunk["article_number"])
            kept.append(chunk)
        if len(kept) == top_k:
            break
    return kept


def retrieve(question: str, top_k: int = DEFAULT_TOP_K) -> list[dict]:
    """Return up to top_k chunks, best first."""
    mode = current_mode()
    if mode == "vector":
        return vector_search(question, top_k)

    candidates = vector_search(question, CANDIDATES)
    if mode == "rerank":
        # The cross-encoder reads the question and each chunk TOGETHER, so it can tell
        # "leaving a job" (ending employment) apart from "leave" (holidays).
        scores = get_reranker().predict([(question, c["text"]) for c in candidates])
        for chunk, score in zip(candidates, scores):
            chunk["rerank_score"] = float(score)
        candidates.sort(key=lambda c: c["rerank_score"], reverse=True)
    return best_per_article(candidates, top_k)


if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) or "How many days of annual leave do I get?"
    print(f"Mode: {current_mode()}")
    for r in retrieve(question):
        preview = " ".join(r["text"].split())[:60]
        score = f"rerank {r['rerank_score']:.3f}" if "rerank_score" in r else f"sim {r['similarity']:.3f}"
        print(f"Art {r['article_number']:>3} [{r['language']}] {score}  {preview}")