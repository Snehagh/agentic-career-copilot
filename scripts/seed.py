"""Seed ChromaDB with sample resume and job description."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.rag.embedder import ingest_document

print("Ingesting sample resume...")
n = ingest_document("data/resumes/sample_resume.txt", "resumes")
print(f"  -> {n} chunks stored in 'resumes' collection")

print("Ingesting sample job description...")
n = ingest_document("data/jobs/ai_engineer_jd.txt", "jobs")
print(f"  -> {n} chunks stored in 'jobs' collection")

print("Done. ChromaDB is seeded and ready.")
