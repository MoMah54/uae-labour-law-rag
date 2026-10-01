"""
Phase 3b: Answer a question using only the retrieved law articles, with citations.

Usage:
    python src/generate.py "How many days of annual leave do I get?"
    python src/generate.py "كم يوم إجازة سنوية يستحق العامل؟"

The LLM provider is chosen with LLM_PROVIDER in .env (default: gemini).
Adding another provider (Claude, OpenAI) means writing one small class below.
"""

import json
import logging
import os
import random
import re
import sys
import time
from typing import Protocol

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from retrieve import retrieve

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("generate")
# Hide the per-request HTTP lines from libraries; keep our own logs.
for noisy in ("httpx", "huggingface_hub", "sentence_transformers", "google_genai"):
    logging.getLogger(noisy).setLevel(logging.WARNING)

NOT_FOUND = {
    "en": "I couldn't find this in the UAE Labour Law.",
    "ar": "لم أجد إجابة على هذا السؤال في قانون العمل الإماراتي.",
}

SYSTEM_PROMPT = """You answer questions about the UAE Labour Law (Federal Decree-Law No. 33 of 2021).

Rules:
1. Use ONLY the articles provided in the user message. Never use outside knowledge.
2. Cite the article number(s) your answer relies on in cited_articles.
3. Write the answer in the language named in the user message.
4. If the provided articles do not answer the question, set found to false,
   leave cited_articles empty, and say you could not find it in the law.
5. Be concise and factual. Quote exact numbers (days, percentages, periods) from the text.
6. The Arabic source text has some letters swapped by the PDF (for example
   "عالقات" means "علاقات"). Read past this, and write correct Arabic in your answer."""


class Answer(BaseModel):
    """The structured output we require from the LLM."""
    answer: str = Field(description="The answer to the question")
    cited_articles: list[int] = Field(description="Article numbers the answer relies on")
    found: bool = Field(description="False if the provided articles do not answer the question")


# ---------- Providers ----------

class LLMProvider(Protocol):
    name: str
    def generate(self, system: str, user: str) -> Answer: ...


class GeminiProvider:
    name = "gemini"

    def __init__(self) -> None:
        from google import genai
        from google.genai import types
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            sys.exit("GEMINI_API_KEY is missing from .env")
        self.client = genai.Client(api_key=api_key)
        self.types = types
        # GEMINI_MODEL can list several models, comma separated, best first.
        # If one is overloaded (503) or retired (404), the next one is tried.
        models = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
        self.models = [m.strip() for m in models.split(",") if m.strip()]

    def _call(self, model: str, system: str, user: str) -> Answer:
        response = self.client.models.generate_content(
            model=model,
            contents=user,
            config=self.types.GenerateContentConfig(
                system_instruction=system,
                response_mime_type="application/json",
                response_schema=Answer,
                temperature=0.1,
                # We don't use tools, so switch off automatic function calling
                automatic_function_calling=self.types.AutomaticFunctionCallingConfig(disable=True),
            ),
        )
        if response.parsed is not None:
            return response.parsed
        return Answer.model_validate(json.loads(response.text))

    def generate(self, system: str, user: str) -> Answer:
        last_error = None
        for model in self.models:
            try:
                result = self._call(model, system, user)
                self.name = f"gemini:{model}"
                return result
            except Exception as error:
                code = getattr(error, "code", None)
                if code in (404, 503) and model != self.models[-1]:
                    log.warning("Model %s unavailable (%s). Falling back to next model.", model, code)
                    last_error = error
                    continue
                raise
        raise last_error

    def list_models(self) -> list[str]:
        names = []
        for m in self.client.models.list():
            actions = getattr(m, "supported_actions", None) or []
            if not actions or "generateContent" in actions:
                names.append(m.name.removeprefix("models/"))
        return names


def get_provider() -> LLMProvider:
    provider = os.getenv("LLM_PROVIDER", "gemini").lower()
    if provider == "gemini":
        return GeminiProvider()
    sys.exit(f"Unknown LLM_PROVIDER '{provider}'. Supported: gemini")


# ---------- Retry with exponential backoff ----------

def is_retryable(error: Exception) -> bool:
    """Rate limits (429) and temporary server errors (5xx) are worth retrying."""
    code = getattr(error, "code", None) or getattr(error, "status_code", None)
    if isinstance(code, int):
        return code == 429 or code >= 500
    return isinstance(error, (TimeoutError, ConnectionError))


MAX_WAIT_SECONDS = 65


def server_retry_delay(error: Exception) -> float | None:
    """Rate-limit errors often say how long to wait, e.g. 'Please retry in 56.8s'."""
    match = re.search(r"retry in ([\d.]+)\s*s", str(error), re.IGNORECASE)
    return float(match.group(1)) if match else None


def short_error(error: Exception) -> str:
    code = getattr(error, "code", None) or getattr(error, "status_code", None)
    message = str(error).split(".", 1)[0]
    return message if not code or message.startswith(str(code)) else f"{code} {message}"


def with_retries(fn, max_attempts: int = 4, base_delay: float = 5.0):
    for attempt in range(1, max_attempts + 1):
        try:
            return fn()
        except Exception as error:
            if attempt == max_attempts or not is_retryable(error):
                raise
            # 5s, 10s, 20s ... plus random jitter so parallel clients don't retry in sync.
            # If the server tells us how long to wait (rate limits), respect that instead.
            delay = base_delay * 2 ** (attempt - 1) + random.uniform(0, 1)
            requested = server_retry_delay(error)
            if requested:
                delay = max(delay, requested + 1)
            delay = min(delay, MAX_WAIT_SECONDS)
            log.warning("Attempt %d failed (%s). Retrying in %.0fs", attempt, short_error(error), delay)
            time.sleep(delay)


# ---------- Pipeline ----------

def detect_language(text: str) -> str:
    return "ar" if re.search(r"[؀-ۿ]", text) else "en"


def build_prompt(question: str, chunks: list[dict], language: str) -> str:
    context = "\n\n".join(
        f"[Article {c['article_number']} | {c['language']}]\n{c['text']}" for c in chunks
    )
    language_name = "Arabic" if language == "ar" else "English"
    return f"Answer language: {language_name}\n\nArticles:\n{context}\n\nQuestion: {question}"


def answer_question(question: str, provider: LLMProvider | None = None) -> dict:
    start = time.time()
    provider = provider or get_provider()
    language = detect_language(question)

    chunks = retrieve(question)
    retrieved_articles = sorted({c["article_number"] for c in chunks})

    result = with_retries(lambda: provider.generate(SYSTEM_PROMPT, build_prompt(question, chunks, language)))

    # Guard against invented citations: keep only articles we actually gave the model.
    cited = [a for a in result.cited_articles if a in retrieved_articles]
    dropped = set(result.cited_articles) - set(cited)
    if dropped:
        log.warning("Dropped citations not in retrieved context: %s", sorted(dropped))

    found = result.found and bool(cited)
    output = {
        "question": question,
        "language": language,
        "answer": result.answer if found else NOT_FOUND[language],
        "cited_articles": cited if found else [],
        "found": found,
        "retrieved_articles": retrieved_articles,
        "provider": provider.name,
        "latency_seconds": round(time.time() - start, 2),
    }
    log.info("q=%r retrieved=%s cited=%s found=%s latency=%.2fs",
             question[:60], retrieved_articles, output["cited_articles"], found, output["latency_seconds"])
    return output


if __name__ == "__main__":
    if "--list-models" in sys.argv:
        for name in GeminiProvider().list_models():
            print(name)
        sys.exit()
    q = " ".join(sys.argv[1:]) or "How many days of annual leave does an employee get?"
    out = answer_question(q)
    print("\n" + out["answer"])
    print(f"\nCited articles: {out['cited_articles']}  |  Retrieved: {out['retrieved_articles']}")