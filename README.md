# Agentic Career Copilot
![Tests](https://github.com/Snehagh/agentic-career-copilot/actions/workflows/test.yml/badge.svg)

A multi-agent AI system that matches resumes to job descriptions, scores fit, identifies skill gaps, and delivers coaching recommendations — all via a REST API.

**How it runs:** The RAG pipeline uses real vector search (ChromaDB + sentence-transformers). The multi-agent orchestration runs on **CrewAI + OpenAI**, and the evaluation layer uses a real **LLM-as-judge**, both activate automatically when an `OPENAI_API_KEY` is set. Without a key, the project falls back to deterministic mock responses so it can be cloned, run, and tested fully offline (no key required).

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

## Tech Stack
| Layer | Without API key (offline) | With API key (live) |
| --- | --- | --- |
| API | FastAPI + Pydantic | same |
| Vector store | ChromaDB (local persistent) | same |
| Embeddings | all-MiniLM-L6-v2 via sentence-transformers | OpenAI text-embedding-3-small |
| Agent orchestration | Deterministic mock fallback | CrewAI + OpenAI |
| LLM evaluation | Keyword-overlap heuristic | LLM-as-judge via OpenAI |
| Document parsing | pypdf, python-docx | same |

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
Open http://localhost:8000/docs for the interactive Swagger UI.

## Enable live LLMs (optional)
```bash
cp .env.example .env
# Add your key:  OPENAI_API_KEY=sk-...
```
With a valid key, agent orchestration (CrewAI + OpenAI), OpenAI embeddings, and the LLM-as-judge evaluator activate automatically. No code changes required.

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

## Project Structure

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


## Run Tests
```bash
pytest tests/ -v
```

## CI — Automated Testing
Tests run automatically on every push and pull request via GitHub Actions (`.github/workflows/test.yml`):
- Checks out the code on `ubuntu-latest`
- Sets up Python 3.13 with pip caching
- Installs all dependencies from `requirements.txt`
- Runs `pytest tests/ -v` — the workflow fails if any test fails

See live results in the **Actions** tab of the repository.

## Skills Demonstrated
- **Agentic AI** — sequential multi-agent pipeline (Resume Analyst → JD Analyst → Match Scorer → Career Coach) with context propagation between agents
- **RAG** — full pipeline: document ingestion, chunking, local vector embeddings, semantic retrieval from ChromaDB
- **LLM Evaluation** — LLM-as-judge pattern scoring answer relevancy and faithfulness
- **FastAPI** — async REST API with Pydantic v2 request/response models, file upload, OpenAPI docs
- **Python / Software Engineering** — 12-factor config, modular package structure, unit tests, clean offline/live seams

## Author
**Sneha Ghosh** — [Portfolio](https://snehagh.github.io/Snehagh-Portfolio/) · [LinkedIn](https://www.linkedin.com/in/sneha-ghosh08/)
