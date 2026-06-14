"""FastAPI backend for RAG query endpoint."""

from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from exchange_connectivity_hub.config import get_config
from exchange_connectivity_hub.retrieval.rag_chain import create_rag_chain


# Request/Response models
class QueryRequest(BaseModel):
    """Request body for the /query endpoint."""

    question: str = Field(..., description="Question to answer")
    exchange_filter: str | None = Field(None, description="Optional exchange filter")


class SourceInfo(BaseModel):
    """Source document metadata returned with each answer."""

    filename: str
    page_number: int | None
    exchange: str
    ingested_at: str | None


class QueryResponse(BaseModel):
    """Response body for the /query endpoint."""

    answer: str
    sources: list[SourceInfo]
    staleness_warning: str | None


class HealthResponse(BaseModel):
    """Response body for the /health endpoint."""

    status: str
    collection_count: int | None
    version: str


# FastAPI app
app = FastAPI(
    title="Exchange Connectivity Hub API",
    description="RAG-based Q&A for Asian exchange documentation",
    version="0.1.0",
)

# RAG chain (lazy loaded)
_rag_chain = None


def get_rag_chain():
    """Get or create RAG chain singleton."""
    global _rag_chain
    if _rag_chain is None:
        _rag_chain = create_rag_chain()
    return _rag_chain


@app.get("/health", response_model=HealthResponse)
def health() -> dict[str, Any]:
    """Health check endpoint."""
    # Try to get collection count
    try:
        import chromadb

        config = get_config()
        chroma = chromadb.PersistentClient(
            path=config["vector_store"]["persist_directory"],
        )
        collection = chroma.get_collection(name=config["vector_store"]["collection_name"])
        count = collection.count()
    except Exception:
        count = None

    return {
        "status": "ok",
        "collection_count": count,
        "version": "0.1.0",
    }


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest) -> dict[str, Any]:
    """Query the RAG system."""
    if not req.question or not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    try:
        chain = get_rag_chain()
        result = chain.invoke(
            {
                "question": req.question,
                "exchange_filter": req.exchange_filter,
            }
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
