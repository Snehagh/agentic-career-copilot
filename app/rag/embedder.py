"""RAG pipeline: document ingestion and semantic retrieval via local embeddings."""
import hashlib
from pathlib import Path

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader

from app.config import settings

_embed_fn = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
_client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)


def _collection(name: str) -> chromadb.Collection:
    return _client.get_or_create_collection(name, embedding_function=_embed_fn)


def ingest_document(file_path: str, collection_name: str) -> int:
    """Load, chunk, embed and store a document. Returns chunk count."""
    path = Path(file_path)
    loader = PyPDFLoader(str(path)) if path.suffix.lower() == ".pdf" else TextLoader(str(path))
    chunks = _splitter.split_documents(loader.load())

    col = _collection(collection_name)
    texts = [c.page_content for c in chunks]
    ids = [hashlib.md5(f"{i}{t}".encode()).hexdigest() for i, t in enumerate(texts)]
    col.upsert(documents=texts, ids=ids)
    return len(chunks)


def query(text: str, collection_name: str, n_results: int = 5) -> list[str]:
    """Return the top-n most semantically similar chunks."""
    col = _collection(collection_name)
    if col.count() == 0:
        return []
    results = col.query(query_texts=[text], n_results=min(n_results, col.count()))
    return results["documents"][0] if results["documents"] else []
