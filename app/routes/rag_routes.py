from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from ..services.rag_service import RAGService
from ..models.document import Document
from typing import Dict, Any
from pydantic import BaseModel

router = APIRouter()
rag_service = RAGService()

class QueryRequest(BaseModel):
    query: str

@router.get("/status")
async def get_status() -> Dict[str, Any]:
    """Get the initialization status of the RAG service."""
    return rag_service.get_initialization_status()

@router.post("/initialize")
async def initialize_service(background_tasks: BackgroundTasks) -> Dict[str, str]:
    """Initialize the RAG service asynchronously."""
    if not rag_service.is_initialized:
        background_tasks.add_task(rag_service.initialize)
        return {"status": "Initialization started"}
    return {"status": "Already initialized"}

@router.post("/documents/")
async def upload_document(file: UploadFile = File(...)) -> Document:
    """Upload and process a document."""
    if not rag_service.is_initialized:
        raise HTTPException(
            status_code=503,
            detail="Service is initializing. Please check /status endpoint for progress."
        )
    return await rag_service.process_document(file)

@router.get("/documents/{document_id}/summary")
async def get_document_summary(document_id: str):
    """Get a summary of a specific document."""
    if not rag_service.is_initialized:
        raise HTTPException(
            status_code=503,
            detail="Service is initializing. Please check /status endpoint for progress."
        )
    summary, sources = await rag_service.summarize_document(document_id)
    return {"summary": summary, "sources": sources}

@router.post("/documents/{document_id}/query")
async def query_document(document_id: str, request: QueryRequest):
    """Query a specific document."""
    if not rag_service.is_initialized:
        raise HTTPException(
            status_code=503,
            detail="Service is initializing. Please check /status endpoint for progress."
        )
    response, sources = await rag_service.process_query(document_id, request.query)
    return {"response": response, "sources": sources} 