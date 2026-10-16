"""
Phase 4: FastAPI service for the UAE Labour Law assistant.

Run (from the project root, with .venv active and the database running):
    uvicorn api:app --app-dir src --port 8000

Then open http://localhost:8000/docs to try it in the browser.
"""

import logging
import time
from contextlib import asynccontextmanager

import psycopg
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from generate import answer_question, get_provider
from retrieve import DATABASE_URL, get_model

log = logging.getLogger("api")

# Shared objects, created once at startup instead of on every request.
state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    start = time.time()
    get_model()                       # load bge-m3 into memory once (the slow part)
    state["provider"] = get_provider()
    state["started_at"] = time.time()
    log.info("Startup complete in %.1fs", time.time() - start)
    yield
    state.clear()


app = FastAPI(
    title="UAE Labour Law Assistant",
    description="Ask questions about UAE Federal Decree-Law No. 33 of 2021 in Arabic or English. "
                "Answers cite the articles they are based on. Not legal advice.",
    version="0.4.0",
    lifespan=lifespan,
)

# Lets a browser-based frontend on another port call this API during development.
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# ---------- Request and response models ----------

class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=500,
                          examples=["How many days of annual leave does an employee get?"])


class Source(BaseModel):
    article_number: int
    language: str
    part: int
    text: str
    similarity: float


class AskResponse(BaseModel):
    question: str
    language: str
    answer: str
    cited_articles: list[int]
    found: bool
    retrieved_articles: list[int]
    sources: list[Source]
    provider: str
    latency_seconds: float
    disclaimer: str = "This is general information from the UAE Labour Law, not legal advice."


class HealthResponse(BaseModel):
    status: str
    database: str
    chunks_indexed: int
    embedding_model_loaded: bool
    uptime_seconds: float


# ---------- Endpoints ----------

@app.get("/health", response_model=HealthResponse)
def health():
    """Checks the database is reachable and the embedding model is loaded."""
    try:
        with psycopg.connect(DATABASE_URL, connect_timeout=3) as conn:
            count = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        db_status = "ok"
    except Exception as error:
        log.error("Health check: database error: %s", error)
        count, db_status = 0, "unreachable"

    cache_info = getattr(get_model, "cache_info", None)
    model_loaded = bool(cache_info and cache_info().currsize > 0)
    healthy = db_status == "ok" and count > 0 and model_loaded
    return HealthResponse(
        status="ok" if healthy else "degraded",
        database=db_status,
        chunks_indexed=count,
        embedding_model_loaded=model_loaded,
        uptime_seconds=round(time.time() - state.get("started_at", time.time()), 1),
    )


# A plain "def" (not "async def") makes FastAPI run this in a worker thread,
# so slow model and LLM calls don't block other requests.
@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    """Answer a question in Arabic or English, with citations to the law's articles."""
    question = request.question.strip()
    if len(question) < 3:
        raise HTTPException(status_code=422, detail="Question is too short.")
    try:
        return answer_question(question, provider=state["provider"])
    except psycopg.OperationalError:
        log.exception("Database unavailable")
        raise HTTPException(status_code=503, detail="The database is unavailable. Is Docker running?")
    except Exception as error:
        code = getattr(error, "code", None)
        log.exception("Failed to answer question")
        if code in (429, 503):
            raise HTTPException(status_code=503,
                                detail="The language model is busy or rate limited. Try again in a minute.")
        raise HTTPException(status_code=500, detail="Something went wrong while answering.")