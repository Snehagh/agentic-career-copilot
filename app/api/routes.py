"""FastAPI route definitions."""
import tempfile
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from app.agents.crew import run_career_crew
from app.eval.evaluator import evaluate_response
from app.rag.embedder import ingest_document, query

router = APIRouter()


# ---------------------------------------------------------------------------
# Upload endpoints
# ---------------------------------------------------------------------------

class IngestResponse(BaseModel):
    collection: str
    chunks_stored: int


@router.post("/upload/resume", response_model=IngestResponse, tags=["Upload"])
async def upload_resume(file: UploadFile = File(...)):
    """Upload a resume (PDF or TXT) and index it into ChromaDB."""
    return await _ingest(file, "resumes")


@router.post("/upload/job", response_model=IngestResponse, tags=["Upload"])
async def upload_job(file: UploadFile = File(...)):
    """Upload a job description (PDF or TXT) and index it into ChromaDB."""
    return await _ingest(file, "jobs")


async def _ingest(file: UploadFile, collection: str) -> IngestResponse:
    suffix = Path(file.filename).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name
    try:
        count = ingest_document(tmp_path, collection)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return IngestResponse(collection=collection, chunks_stored=count)


# ---------------------------------------------------------------------------
# Analysis endpoint
# ---------------------------------------------------------------------------

class AnalysisRequest(BaseModel):
    job_title: str


class AnalysisResponse(BaseModel):
    resume_analysis: str
    job_analysis: str
    match_score: str
    recommendations: str


@router.post("/analyze", response_model=AnalysisResponse, tags=["Analysis"])
async def analyze(request: AnalysisRequest):
    """Run the full CrewAI pipeline for resume-job matching."""
    try:
        result = run_career_crew(request.job_title)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return AnalysisResponse(**{k: result[k] for k in AnalysisResponse.model_fields})


# ---------------------------------------------------------------------------
# RAG search endpoint
# ---------------------------------------------------------------------------

class SearchRequest(BaseModel):
    query: str
    collection: str = "resumes"
    n_results: int = 5


class SearchResponse(BaseModel):
    results: list[str]


@router.post("/search", response_model=SearchResponse, tags=["RAG"])
async def search(request: SearchRequest):
    """Semantic search over a ChromaDB collection."""
    results = query(request.query, request.collection, request.n_results)
    return SearchResponse(results=results)


# ---------------------------------------------------------------------------
# Evaluation endpoint
# ---------------------------------------------------------------------------

class EvalRequest(BaseModel):
    query: str
    actual_output: str
    retrieval_context: list[str]


class EvalResponse(BaseModel):
    answer_relevancy: float
    faithfulness: float
    relevancy_reason: str
    faithfulness_reason: str


@router.post("/evaluate", response_model=EvalResponse, tags=["Evaluation"])
async def evaluate(request: EvalRequest):
    """Run LLM-as-judge evaluation on a response."""
    try:
        scores = evaluate_response(
            request.query, request.actual_output, request.retrieval_context
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return EvalResponse(**scores)
