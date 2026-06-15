# Agentic Career Copilot

[![Tests](https://github.com/Snehagh/agentic-career-copilot/actions/workflows/test.yml/badge.svg)](https://github.com/Snehagh/agentic-career-copilot/actions/workflows/test.yml)

A multi-agent AI system that matches resumes to job descriptions, scores fit, identifies skill gaps, and delivers coaching recommendations — all via a REST API.

**Current mode: local demo.** The RAG pipeline uses real vector search (ChromaDB + sentence-transformers, no API key needed). The agent orchestration layer uses mock LLM responses that mirror a real CrewAI + OpenAI pipeline. See [Upgrade to Production](#upgrade-to-production) to swap in live LLMs.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    FastAPI REST API                      │
│  POST /upload/resume   POST /upload/job                 │
│  POST /analyze         POST /search   POST /evaluate    │
└───────────────────────┬─────────────────────────────────┘
                        │
          ┌─────────────▼──────────────┐
          │     RAG Pipeline           │
          │  ┌──────────────────────┐  │
          │  │  Document Ingestion  │  │
          │  │  pypdf / TextLoader  │  │
          │  │  chunk → embed       │  │
          │  └──────────┬───────────┘  │
          │             │              │
          │  ┌──────────▼───────────┐  │
          │  │  ChromaDB (local)    │  │
          │  │  all-MiniLM-L6-v2   │  │
          │  │  sentence-transformers│ │
          │  └──────────┬───────────┘  │
          └─────────────┼──────────────┘
                        │ top-k chunks
          ┌─────────────▼──────────────┐
          │  Multi-Agent Orchestrator  │
          │                            │
          │  [1] Resume Analyst        │  extracts skills & achievements
          │       ↓ context            │
          │  [2] JD Analyst            │  extracts requirements
          │       ↓ context            │
          │  [3] Match Scorer          │  0-100 score + gap list
          │       ↓ context            │
          │  [4] Career Coach          │  actionable recommendations
          │                            │
          │  LLM layer: mock (demo)    │
          │           → OpenAI (prod)  │
          └────────────────────────────┘
                        │
          ┌─────────────▼──────────────┐
          │  LLM-as-Judge Evaluator    │
          │  answer_relevancy          │
          │  faithfulness              │
          │  heuristic (demo) →        │
          │  OpenAI judge (prod)       │
          └────────────────────────────┘
```

---

## Tech Stack

| Layer | Demo mode | Production mode |
|---|---|---|
| API | FastAPI + Pydantic | same |
| Vector store | ChromaDB (local persistent) | same |
| Embeddings | `all-MiniLM-L6-v2` via sentence-transformers | OpenAI `text-embedding-3-small` |
| Agent orchestration | Custom sequential orchestrator (mock LLM) | CrewAI + OpenAI GPT-4o |
| LLM evaluation | Keyword-overlap heuristic | LLM-as-judge via OpenAI |
| Document parsing | pypdf, python-docx | same |

---

## Quickstart (no API key needed)

```bash
# 1. Clone and enter the repo
git clone <repo-url> && cd agentic-career-copilot

# 2. Create virtual environment
python3 -m venv .venv && source .venv/bin/activate

# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 4. Seed ChromaDB with sample resume and job description
python scripts/seed.py

# 5. Start the API
uvicorn app.main:app --reload
```

Open **http://localhost:8000/docs** for the interactive Swagger UI.

---

## API Endpoints

### Upload documents
```bash
# Upload a resume (PDF or TXT)
curl -X POST http://localhost:8000/api/v1/upload/resume \
  -F "file=@data/resumes/sample_resume.txt"

# Upload a job description
curl -X POST http://localhost:8000/api/v1/upload/job \
  -F "file=@data/jobs/ai_engineer_jd.txt"
```

### Run the multi-agent analysis
```bash
curl -X POST http://localhost:8000/api/v1/analyze \
  -H "Content-Type: application/json" \
  -d '{"job_title": "AI Engineer"}'
```

Returns:
```json
{
  "resume_analysis": "...",
  "job_analysis": "...",
  "match_score": "{\"score\": 91, \"matched_skills\": [...], \"gaps\": [...]}",
  "recommendations": "..."
}
```

### Semantic search over indexed documents
```bash
curl -X POST http://localhost:8000/api/v1/search \
  -H "Content-Type: application/json" \
  -d '{"query": "machine learning AWS", "collection": "resumes", "n_results": 3}'
```

### Evaluate a response (LLM-as-judge)
```bash
curl -X POST http://localhost:8000/api/v1/evaluate \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are the candidate Python skills?",
    "actual_output": "5 years of Python with FastAPI and LangChain.",
    "retrieval_context": ["Python (5 years): FastAPI, LangChain, CrewAI"]
  }'
```

---

## Project Structure

```
agentic-career-copilot/
├── app/
│   ├── main.py              # FastAPI app entry point
│   ├── config.py            # Pydantic settings (12-factor)
│   ├── agents/
│   │   └── crew.py          # Multi-agent orchestrator
│   ├── rag/
│   │   └── embedder.py      # Ingestion + ChromaDB retrieval
│   ├── eval/
│   │   └── evaluator.py     # LLM-as-judge scoring
│   └── api/
│       └── routes.py        # REST route definitions
├── data/
│   ├── resumes/             # Sample resume (TXT/PDF)
│   └── jobs/                # Sample job descriptions
├── scripts/
│   └── seed.py              # One-shot ChromaDB seeder
├── tests/
│   ├── test_rag.py
│   └── test_api.py
└── requirements.txt
```

---

## Run Tests

```bash
pytest tests/ -v
```

---

## Upgrade to Production

Two functions handle all LLM interaction. Replace them to go live:

**1. Agent LLM — [`app/agents/crew.py`](app/agents/crew.py)**

Replace `_mock_llm_call` with a real OpenAI call:
```python
from openai import OpenAI
client = OpenAI(api_key=settings.openai_api_key)

def _mock_llm_call(role, task, rag_context, prior_context):
    response = client.chat.completions.create(
        model=settings.openai_model,
        messages=[
            {"role": "system", "content": f"You are a {role}. {prior_context}"},
            {"role": "user", "content": f"{task}\n\nContext:\n{rag_context}"},
        ]
    )
    return response.choices[0].message.content
```

Then add `crewai>=1.14.7` and `openai>=2.30.0,<3` to requirements.txt.

**2. Embeddings — [`app/rag/embedder.py`](app/rag/embedder.py)**

Replace `SentenceTransformerEmbeddingFunction` with:
```python
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
_embed_fn = OpenAIEmbeddingFunction(
    api_key=settings.openai_api_key,
    model_name="text-embedding-3-small"
)
```

**3. Set env vars**
```bash
cp .env.example .env
# Add: OPENAI_API_KEY=sk-...
```

---

## CI — Automated Testing

Tests run automatically on every push and pull request via GitHub Actions ([`.github/workflows/test.yml`](.github/workflows/test.yml)).

The workflow:
1. Checks out the code on `ubuntu-latest`
2. Sets up Python 3.13 with pip caching
3. Installs all dependencies from `requirements.txt`
4. Runs `pytest tests/ -v` — the workflow fails if any test fails

To see live results, go to the **Actions** tab of the repository on GitHub.

---

## Skills Demonstrated

- **Agentic AI** — sequential multi-agent pipeline (Resume Analyst → JD Analyst → Match Scorer → Career Coach) with context propagation between agents
- **RAG** — full pipeline: document ingestion, chunking, local vector embeddings, semantic retrieval from ChromaDB
- **LLM Evaluation** — LLM-as-judge pattern scoring answer relevancy and faithfulness
- **FastAPI** — async REST API with Pydantic v2 request/response models, file upload, OpenAPI docs
- **Python / Software Engineering** — 12-factor config, modular package structure, mocked unit tests, clean upgrade seams
