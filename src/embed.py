"""
Phase 2: Embed every chunk with BAAI/bge-m3 and load it into PostgreSQL + pgvector.

Usage (from the project root, with .venv active and the database running):
    python src/embed.py           # embed chunks.jsonl and (re)load the table
    python src/embed.py --test    # run a few sample searches against the table

The first run downloads the bge-m3 model (about 2.3 GB), then caches it.
"""

import json
import os
import sys
import time
from pathlib import Path

import psycopg
from dotenv import load_dotenv
from pgvector.psycopg import register_vector
from sentence_transformers import SentenceTransformer

load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/labour_law")
CHUNKS_FILE = Path("data/processed/chunks.jsonl")
MODEL_NAME = "BAAI/bge-m3"
EMBEDDING_DIM = 1024
BATCH_SIZE = 8

SAMPLE_QUESTIONS = [
    "How many days of annual leave does an employee get?",
    "كم يوم إجازة سنوية يستحق العامل؟",
    "What is the maximum probation period?",
    "ما هي مدة فترة التجربة؟",
]


def load_model() -> SentenceTransformer:
    print(f"Loading {MODEL_NAME} (first time downloads about 2.3 GB)...")
    model = SentenceTransformer(MODEL_NAME)
    model.max_seq_length = 1024  # our chunks are well under this; shorter is faster on CPU
    return model


def connect() -> psycopg.Connection:
    conn = psycopg.connect(DATABASE_URL, autocommit=True)
    conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    register_vector(conn)
    return conn


def create_table(conn: psycopg.Connection) -> None:
    # Dropping and recreating keeps reruns simple while the schema is still changing.
    conn.execute("DROP TABLE IF EXISTS chunks")
    conn.execute(f"""
        CREATE TABLE chunks (
            id              TEXT PRIMARY KEY,
            text            TEXT NOT NULL,
            article_number  INTEGER NOT NULL,
            part            INTEGER NOT NULL,
            total_parts     INTEGER NOT NULL,
            language        TEXT NOT NULL,
            source_document TEXT NOT NULL,
            page            INTEGER,
            embedding       vector({EMBEDDING_DIM}) NOT NULL
        )
    """)


def load_chunks() -> list[dict]:
    if not CHUNKS_FILE.exists():
        sys.exit(f"{CHUNKS_FILE} not found. Run python src/ingest.py first.")
    with CHUNKS_FILE.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def embed_and_load() -> None:
    chunks = load_chunks()
    model = load_model()

    print(f"Embedding {len(chunks)} chunks on CPU...")
    start = time.time()
    # normalize_embeddings=True makes cosine similarity equal to a dot product,
    # which is what bge-m3 is trained for.
    vectors = model.encode(
        [c["text"] for c in chunks],
        batch_size=BATCH_SIZE,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    print(f"Done in {time.time() - start:.0f}s")

    conn = connect()
    create_table(conn)
    with conn.cursor() as cur:
        cur.executemany(
            """INSERT INTO chunks (id, text, article_number, part, total_parts,
                                   language, source_document, page, embedding)
               VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)""",
            [
                (c["id"], c["text"], c["article_number"], c["part"], c["total_parts"],
                 c["language"], c["source_document"], c.get("page"), vec)
                for c, vec in zip(chunks, vectors)
            ],
        )
    count = conn.execute("SELECT language, COUNT(*) FROM chunks GROUP BY language ORDER BY language").fetchall()
    print("Rows loaded:", ", ".join(f"{lang}: {n}" for lang, n in count))
    conn.close()


def test_search(top_k: int = 5) -> None:
    model = load_model()
    conn = connect()
    for question in SAMPLE_QUESTIONS:
        qvec = model.encode(question, normalize_embeddings=True)
        # <=> is pgvector's cosine distance; similarity = 1 - distance
        rows = conn.execute(
            """SELECT article_number, language, 1 - (embedding <=> %s) AS similarity, LEFT(text, 70)
               FROM chunks ORDER BY embedding <=> %s LIMIT %s""",
            (qvec, qvec, top_k),
        ).fetchall()
        print(f"\nQ: {question}")
        for article, lang, sim, preview in rows:
            preview = " ".join(preview.split())
            print(f"  Art {article:>3} [{lang}] {sim:.3f}  {preview}")
    conn.close()


if __name__ == "__main__":
    if "--test" in sys.argv:
        test_search()
    else:
        embed_and_load()
