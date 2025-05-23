from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import os
from pathlib import Path
from typing import List
from .services.rag_service import RAGService
from .models.document import Document

app = FastAPI(title="Local RAG System")
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

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """Upload and process a document."""
    try:
        document = await rag_service.process_document(file)
        return {"message": "Document uploaded successfully", "document_id": document.id}
    except Exception as e:
        return {"error": str(e)}

@app.get("/documents")
async def list_documents():
    """List all processed documents."""
    return rag_service.list_documents()

@app.post("/chat/{document_id}")
async def chat(document_id: str, query: str = Form(...)):
    """Process a chat query for a specific document."""
    try:
        response = await rag_service.process_query(document_id, query)
        return {"response": response}
    except Exception as e:
        return {"error": str(e)}

@app.get("/summarize/{document_id}")
async def summarize(document_id: str):
    """Generate a summary for a specific document."""
    try:
        summary = await rag_service.summarize_document(document_id)
        return {"summary": summary}
    except Exception as e:
        return {"error": str(e)} 