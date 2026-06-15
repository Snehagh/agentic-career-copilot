"""FastAPI application entry point."""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router

app = FastAPI(
    title="Agentic Career Copilot",
    description=(
        "Multi-agent AI system for resume-job matching, gap analysis, and career coaching. "
        "**Demo mode:** real ChromaDB RAG with local sentence-transformer embeddings; "
        "agent orchestration uses a mock LLM layer. "
        "See /docs for upgrade path to OpenAI + CrewAI production mode."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api/v1")


@app.get("/", tags=["Health"])
def root():
    return {"status": "ok", "service": "Agentic Career Copilot"}


@app.get("/health", tags=["Health"])
def health():
    return {"status": "ok"}
