# UAE Labour Law Assistant (Arabic / English RAG)

Ask questions about the UAE Labour Law in Arabic or English and get answers that cite the exact articles they are based on.

This is a retrieval-augmented generation (RAG) system over **Federal Decree-Law No. 33 of 2021 Regulating Labour Relations**. It answers only from the text of the law, cites its sources, refuses questions the law does not cover, and is measured on a bilingual evaluation set.

> **Not legal advice.** This project is for learning and demonstration. Always check the official text or a qualified professional.

---

## Highlights

- **Bilingual:** Arabic and English questions, answered in the language they were asked in
- **Cross-lingual search:** an Arabic question can retrieve the English article and vice versa
- **Article-level citations:** every answer lists the articles it relies on, with the source text
- **Grounded answers:** a citation guard drops any article the model cites that was not actually retrieved, and out-of-scope questions are refused instead of answered from general knowledge
- **Two-stage retrieval:** vector search (bge-m3 + pgvector) followed by a cross-encoder reranker (bge-reranker-v2-m3)
- **Measured:** 76 evaluation questions, with every change compared against a recorded baseline
- **Resilient LLM layer:** structured output, a model fallback chain, a per-request time limit, and retries with exponential backoff that respect rate limits
- **Chat interface:** Streamlit app with right-to-left Arabic, cited articles, and an expandable panel showing the source text
- **One-command setup:** database, API and interface run as three Docker containers with health checks

---

## Results

Final configuration: reranker over 10 vector-search candidates (256 tokens each), Gemini for answer generation.

| Metric | Standard set (50 questions) | Hard set (26 questions) |
|---|---|---|
| Retrieval hit rate (top 6) | 100% | 100% |
| Correct article ranked first | 95.5% | 90.5% |
| Mean reciprocal rank | 0.973 | 0.926 |
| Answer rate (in-scope questions) | 100% | 100% |
| **Citation accuracy** | **100%** | **100%** |
| Citation precision | 93.6% | 100% |
| **Refusal accuracy (out-of-scope questions)** | **100%** | **100%** |
| Cross-lingual agreement (EN/AR pairs) | 100% | n/a |
| Average answer time (laptop CPU) | 16.4s | 21.2s |

**Effect of the reranker and prompt fix on the hard set:**

| Metric | Baseline (vector search only) | Final |
|---|---|---|
| Retrieval hit rate | 95.2% | 100% |
| Correct article ranked first | 81.0% | 90.5% |
| Citation accuracy | 90.5% | 100% |

### The two test sets

- **Standard set (`eval/questions.jsonl`):** 22 topics, each asked in English and in Arabic (44 questions), plus 6 clearly out-of-scope questions. Used to check consistency between languages.
- **Hard set (`eval/questions_hard.jsonl`):** questions written the way people actually ask, in four categories:
  - **Scenarios (17):** for example *"My employer is keeping my passport. Is that allowed?"* (Article 13) or *"عقدي يتضمن شرط عدم منافسة لمدة 3 سنوات، هل هذا قانوني؟"* (Article 10, maximum two years)
  - **Scope rules (2):** government employees and domestic workers, who are excluded from this law (Article 3)
  - **Number traps (2):** *"What is the minimum wage in dirhams?"* The law gives no amount, so the correct answer says so and names who sets it, instead of inventing a figure
  - **Near-scope refusals (5):** work visa fees, free zone company setup, income tax, Emirates ID renewal. These sound employment-related but are not in this law

### Metrics

| Metric | Meaning |
|---|---|
| Hit rate | An expected article appears in the retrieved results |
| Ranked first | The expected article is the top result |
| MRR | Mean reciprocal rank of the first expected article (1.0 = always first) |
| Citation accuracy | The answer cites at least one expected article |
| Citation precision | Share of all citations that are expected articles |
| Refusal accuracy | Out-of-scope questions are refused |
| Cross-lingual agreement | The English and Arabic versions of a question cite a common article |

---

## Architecture

```mermaid
flowchart LR
    Q[Question<br/>Arabic or English] --> E[Embed with<br/>bge-m3]
    E --> V[(PostgreSQL<br/>+ pgvector)]
    V -->|top 10 chunks| R[Rerank with<br/>bge-reranker-v2-m3]
    R -->|best chunk per article,<br/>top 6| P[Prompt with<br/>retrieved articles]
    P --> L[Gemini<br/>structured JSON output]
    L --> G[Citation guard<br/>and refusal check]
    G --> A[Answer + cited articles<br/>+ source text]
```

**Ingestion (offline):** official PDFs are split **by article** rather than by fixed size, embedded with bge-m3, and stored in pgvector with their article number, language and page.

**Deployment:** three containers started by Docker Compose. The interface calls the API over HTTP, and the API reaches the database by its service name. Each container waits for the previous one to report healthy, and the models are stored in a volume so they download only once.

```mermaid
flowchart LR
    U[Browser] -->|:8501| UI[ui<br/>Streamlit]
    UI -->|HTTP /ask| API[api<br/>FastAPI + models]
    API -->|SQL| DB[(db<br/>PostgreSQL + pgvector)]
    API -->|HTTPS| G[Gemini API]
```

**Query time:** the question is embedded and the 10 nearest chunks are retrieved. A cross-encoder reranker reads the question together with each chunk and re-orders them, then the best chunk per article is kept (top 6). Gemini answers using only those articles and returns JSON (`answer`, `cited_articles`, `found`). Citations not present in the retrieved articles are dropped.

---

## Design decisions and lessons learned

**Chunking by article.** Legal text has natural boundaries, and article-level chunks make citations exact. Headings are only matched at the start of a line, so references like "as per Article (5)" inside a sentence do not split an article.

**Handling defects in the official PDFs.** Text extraction revealed several problems that had to be handled explicitly:
- Brackets reversed in headings: `Article )3 (` instead of `Article (3)`
- Arabic headings split across lines or broken by brackets: `املا( دة27`
- Arabic letter pairs swapped throughout the text by the font encoding: `عالقات` instead of `علاقات`

All 74 articles are recovered in both languages. The letter swaps did not measurably hurt retrieval (Arabic scored the same as English), and the model writes correct Arabic in its answers.

**Why a reranker, and why a small candidate pool.** On the hard set, vector search confused *"leaving a job"* with *leave* in the holiday sense and missed Article 54 entirely. A cross-encoder reranker fixed this, but the first configuration (20 candidates, 512 tokens) took about 55 seconds per question and lost some questions it previously answered, because it promoted "answer-shaped" passages such as penalty articles full of AED amounts. Reducing to **10 candidates and 256 tokens** was both faster (about 15s) and more accurate: vector search provides a sensible shortlist, and the reranker orders it. Going further, to 128 tokens per chunk, lowered accuracy on the standard set (ranked first fell from 95.5% to 90.9%), so 256 was kept.

| Hard set | Vector only | Rerank 20 × 512 | Rerank 10 × 256 |
|---|---|---|---|
| Hit rate | 95.2% | 95.2% | 100% |
| Ranked first | 81.0% | 85.7% | 90.5% |
| Search time per question | under 1s | about 55s | about 15s |

**Refuse, but don't over-refuse.** A strict "refuse if the articles don't answer" rule caused the model to refuse *"What is the minimum wage in dirhams?"* even though Article 27 covers the topic. The prompt now separates "topic not in the law" (refuse) from "topic covered but the specific detail is not" (answer, cite, and say who determines the detail). The change was checked against the near-scope questions to confirm refusals still hold.

**Finding where the time goes.** Answers sometimes took over two minutes, so every answer now reports its search time and model time separately, and `diagnose.py` times each step of search on its own. This found one clear cause and closed off a likely second one:
- **`localhost` versus `127.0.0.1` on Windows (confirmed).** Embedding (0.2s) and reranking (about 9 to 15s) were fine, but every database connection took **130 seconds**. On Windows, `localhost` can resolve to the IPv6 address first, and the connection to the Docker container hung until it timed out and fell back to IPv4. Connecting to `127.0.0.1` directly removed the delay entirely.
- **Slow model responses (prevented).** A slow but successful Gemini response could previously hold up an answer with no limit. Each call now has a 25-second limit, after which the next model in the chain is tried, and repeated questions are served instantly from an answer cache.

**Free-tier constraints.** The Gemini free tier allows a few requests per minute per model, and newer models often return 503 under load. The client uses a fallback chain of models, honours the server's requested retry delay, and the evaluation script paces requests and can resume an interrupted run.

---

## Tech stack

| Part | Choice |
|---|---|
| Language | Python 3.11+ |
| PDF extraction | PyMuPDF |
| Embeddings | `BAAI/bge-m3` (local, multilingual, 1024 dimensions) |
| Reranker | `BAAI/bge-reranker-v2-m3` (local cross-encoder) |
| Vector database | PostgreSQL + pgvector, in Docker |
| LLM | Google Gemini (provider layer designed for adding others) |
| API | FastAPI + Pydantic |
| Interface | Streamlit |
| Packaging | Docker Compose (db, api, ui), CPU-only PyTorch |
| Tests | pytest (no database, model or API key needed) |

---

## Project structure

```
uae-labour-law-rag/
├── data/
│   ├── raw/                 # official PDFs (not committed)
│   └── processed/           # chunks.jsonl
├── src/
│   ├── ingest.py            # PDF extraction and article-based chunking
│   ├── embed.py             # embeddings and loading into pgvector
│   ├── retrieve.py          # vector search, deduplication, reranking
│   ├── generate.py          # prompt, Gemini call, citation guard, retries
│   └── api.py               # FastAPI app: /ask and /health
├── ui/
│   └── app.py               # Streamlit chat interface
├── eval/
│   ├── questions.jsonl      # standard set (50)
│   ├── questions_hard.jsonl # hard set (26)
│   ├── run_eval.py          # evaluation script
│   └── results/             # saved runs and reports
├── tests/
│   └── test_api.py
├── docker-compose.yml       # db, api and ui containers
├── Dockerfile.api           # API image
├── Dockerfile.ui            # interface image
├── requirements.txt         # full local development environment
├── requirements-api.txt     # API container dependencies
├── requirements-ui.txt      # interface container dependencies
└── diagnose.py              # times each step of search
```

---

## Getting started

**Requirements:** Docker Desktop, Python 3.11+ (to build the search index), and a free Gemini API key from [Google AI Studio](https://aistudio.google.com).

**1. Clone and install**
```bash
git clone https://github.com/MoMah54/uae-labour-law-rag.git
cd uae-labour-law-rag
python -m venv .venv
# Windows: .\.venv\Scripts\Activate.ps1    macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
```

**2. Configure:** create a `.env` file in the project root:
```
GEMINI_API_KEY=your-key
GEMINI_MODEL=model-a,model-b          # comma-separated fallback chain
DATABASE_URL=postgresql://postgres:postgres@127.0.0.1:5432/labour_law   # use 127.0.0.1, not localhost
RETRIEVAL_MODE=rerank                 # or vector for faster, less accurate search
```
Run `python src/generate.py --list-models` to see which Gemini models your key can use.

**3. Get the data:** download the law from the [UAE Legislation portal](https://uaelegislation.gov.ae/en/legislations/1541) in English and Arabic, and save the files as `data/raw/labour_law_en.pdf` and `data/raw/labour_law_ar.pdf`.

**4. Build the search index** (once)
```bash
docker compose up -d db       # start only the database
python src/ingest.py          # extract and chunk by article
python src/embed.py           # embed and load (first run downloads bge-m3)
python src/embed.py --test    # sample searches in both languages
```

**5. Run the whole app**
```bash
docker compose up -d --build
```
The first start builds the images and downloads the models (about 4.5 GB), which takes a while. Follow it with `docker compose logs -f api`, and check with `docker compose ps` that the containers are healthy.

- Chat interface: http://localhost:8501
- API documentation: http://localhost:8000/docs

Afterwards, `docker compose up -d` starts the app and `docker compose down` stops it. Data and models are kept.

### Running without Docker (for development)

With the database running (`docker compose up -d db`), start the API and the interface in two terminals:
```bash
uvicorn api:app --app-dir src --port 8000
python -m streamlit run ui/app.py
```
Or ask from the command line:
```bash
python src/generate.py "How many days of annual leave does an employee get?"
python src/generate.py "كم يوم إجازة سنوية يستحق العامل؟"
```

---

## Testing and evaluation

```bash
python -m pytest -q                                      # API tests with fakes, runs in seconds

python eval/run_eval.py --retrieval-only --label check   # search only, no LLM calls
python eval/run_eval.py --label final                    # full pipeline, standard set
python eval/run_eval.py --questions eval/questions_hard.jsonl --label hard_final
python eval/run_eval.py --retrieval-mode vector --retrieval-only --label compare
```
Options include `--retrieval-mode` (`vector`, `vector_dedup`, `rerank`), `--ids` to re-run selected questions, and `--resume` to continue an interrupted run. Reports are saved to `eval/results/`.

---

## Limitations

- **Speed:** a typical answer takes about 20 seconds on a laptop CPU: about 15 seconds for the reranker and about 4 for Gemini. On a GPU the reranker would take well under a second. `RETRIEVAL_MODE=vector` is much faster but less accurate on hard questions, and repeated questions are answered instantly from the cache.
- **Test set size:** 76 questions in total. On the hard set, one question changes a score by about 5 percentage points.
- **Tuning bias:** the reranker settings were chosen using the hard set and confirmed on the standard set, so the hard-set score is somewhat optimistic.
- **Expected citations are narrow:** some "imprecise" citations are related articles (for example citing the definitions article alongside the juveniles article), so citation precision understates answer quality.
- **Arabic source text:** quoted Arabic sources contain the PDF's letter-order defects, although answers are written in correct Arabic.
- **Scope:** only the Decree-Law itself is indexed. The Executive Regulation (Cabinet Resolution No. 1 of 2022), which fills in many details such as Ramadan working hours, is not yet included.

---

## Roadmap

- [x] Article-based ingestion of Arabic and English PDFs
- [x] Cross-lingual vector search with pgvector
- [x] Cited answer generation with refusals and fallback
- [x] FastAPI service with automated tests
- [x] Bilingual evaluation (standard and hard sets) and reranker
- [x] Chat interface showing answers next to their sources
- [x] Full application in Docker (database, API, interface)
- [ ] CI with GitHub Actions
- [ ] Cloud deployment
- [ ] Repair of the Arabic source text for display
- [ ] Index the Executive Regulation

---

## Author

**Mohamed Ashraf Mahmoud**, Computer Engineering graduate, American University of Sharjah
[GitHub](https://github.com/MoMah54)