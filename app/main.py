from fastapi import FastAPI, UploadFile, File, Form, Request, Body
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
import os
import logging
from pathlib import Path
from typing import List, Dict, Any
from pydantic import BaseModel
from .services.rag_service import RAGService
from .models.document import Document

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QueryRequest(BaseModel):
    query: str

app = FastAPI(title="Local RAG System")

# Initialize templates
templates = Jinja2Templates(directory="app/templates")

# Initialize the RAG service
rag_service = RAGService()

# Mount static files
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# Ensure data directory exists
data_dir = Path("data")
data_dir.mkdir(exist_ok=True)

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Render the home page."""
    documents = rag_service.list_documents()
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "documents": documents}
    )

@app.get("/status")
async def get_status() -> Dict[str, Any]:
    """Get the current initialization status."""
    return rag_service.get_initialization_status()

@app.post("/initialize")
async def initialize_system():
    """Start the system initialization process."""
    if not rag_service.is_initialized:
        await rag_service.initialize()
    return {"status": "Initialization complete" if rag_service.is_initialized else "Initialization started"}

@app.post("/documents/")
async def upload_document(file: UploadFile = File(...)):
    """Upload and process a document."""
    try:
        document = await rag_service.process_document(file)
        return document
    except Exception as e:
        logger.error(f"Error processing document: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

@app.get("/documents/{document_id}/summary")
async def get_document_summary(document_id: str):
    """Generate a summary for a specific document."""
    try:
        summary, sources = await rag_service.summarize_document(document_id)
        return {"summary": summary, "sources": sources}
    except Exception as e:
        logger.error(f"Error generating summary: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

@app.post("/documents/{document_id}/query")
async def query_document(document_id: str, request: QueryRequest):
    """Process a query for a specific document."""
    try:
        logger.info(f"Processing query for document {document_id}: {request.query}")
        if not rag_service.is_initialized:
            raise Exception("Service is not initialized yet. Please wait for initialization to complete.")
            
        result = await rag_service.process_query(document_id, request.query)
        return result  # Already in the correct format: {"response": response_text, "sources": sources}
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

@app.get("/documents/{document_id}/history")
async def get_conversation_history(document_id: str):
    """Get the conversation history for a specific document."""
    try:
        history = rag_service.get_conversation_history(document_id)
        return {"history": history}
    except Exception as e:
        logger.error(f"Error retrieving conversation history: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        ) 