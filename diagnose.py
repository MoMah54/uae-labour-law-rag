"""
Times each step of search separately, to find which one is slow.

Usage (from the project root, with .venv active and the database running):
    python diagnose.py
"""

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
os.environ.setdefault("RETRIEVAL_MODE", "rerank")


def timed(label, fn):
    start = time.time()
    result = fn()
    print(f"{label:<42} {time.time() - start:7.2f}s")
    return result


print("Step                                          Time")
print("-" * 52)
import torch  # noqa: E402

print(f"{'torch threads':<42} {torch.get_num_threads():>7}")
retrieve = timed("import retrieve module", lambda: __import__("retrieve"))
print(f"{'database URL host':<42} {retrieve.DATABASE_URL.split('@')[-1]}")

model = timed("load embedding model (bge-m3)", retrieve.get_model)
reranker = timed("load reranker (bge-reranker-v2-m3)", retrieve.get_reranker)

question = "My employer is keeping my passport. Is that allowed?"
timed("embed question (1st time)", lambda: model.encode(question, normalize_embeddings=True))
timed("embed question (2nd time)", lambda: model.encode(question, normalize_embeddings=True))

import psycopg  # noqa: E402

timed("connect to database", lambda: psycopg.connect(retrieve.DATABASE_URL).close())
timed("connect to database (2nd time)", lambda: psycopg.connect(retrieve.DATABASE_URL).close())

candidates = timed("vector search (embed + query, 10 chunks)",
                   lambda: retrieve.vector_search(question, 10))
pairs = [(question, c["text"]) for c in candidates]
timed("rerank 10 chunks (1st time)", lambda: reranker.predict(pairs))
timed("rerank 10 chunks (2nd time)", lambda: reranker.predict(pairs))
timed("full retrieve() call", lambda: retrieve.retrieve(question))