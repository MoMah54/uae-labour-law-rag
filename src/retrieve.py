"""
Phase 3a: Semantic search over the chunks table.

Usage:
    python src/retrieve.py "How many days of annual leave do I get?"
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
from sentence_transformers import SentenceTransformer  # noqa: E402
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/labour_law")
MODEL_NAME = "BAAI/bge-m3"
DEFAULT_TOP_K = 6


@lru_cache(maxsize=1)
def get_model() -> SentenceTransformer:
    """Load the embedding model once and reuse it (loading takes several seconds)."""
    model = SentenceTransformer(MODEL_NAME)
    model.max_seq_length = 1024
    return model


def retrieve(question: str, top_k: int = DEFAULT_TOP_K) -> list[dict]:
    """Return the top_k most similar chunks, in both languages, best first."""
    qvec = get_model().encode(question, normalize_embeddings=True)
    with psycopg.connect(DATABASE_URL) as conn:
        register_vector(conn)
        rows = conn.execute(
            """SELECT id, text, article_number, part, total_parts, language, page,
                      1 - (embedding <=> %s) AS similarity
               FROM chunks
               ORDER BY embedding <=> %s
               LIMIT %s""",
            (qvec, qvec, top_k),
        ).fetchall()
    keys = ["id", "text", "article_number", "part", "total_parts", "language", "page", "similarity"]
    return [dict(zip(keys, row)) for row in rows]


if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) or "How many days of annual leave do I get?"
    for r in retrieve(question):
        preview = " ".join(r["text"].split())[:70]
        print(f"Art {r['article_number']:>3} [{r['language']}] {r['similarity']:.3f}  {preview}")