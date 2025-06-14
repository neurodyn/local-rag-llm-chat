from fastapi import FastAPI, UploadFile, File, Form, Request, Body, BackgroundTasks
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
    document_ids: List[str]

class SearchEngineConfig(BaseModel):
    search_engine: str

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
async def initialize_system(background_tasks: BackgroundTasks):
    """Start the system initialization process."""
    if not rag_service.is_initialized:
        background_tasks.add_task(rag_service.initialize)
        return {"status": "Initialization started"}
    return {"status": "Already initialized"}

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

@app.post("/documents/query")
async def query_documents(request: QueryRequest):
    """Process a query across multiple documents."""
    try:
        logger.info(f"Processing query across documents {request.document_ids}: {request.query}")
        if not rag_service.is_initialized:
            raise Exception("Service is not initialized yet. Please wait for initialization to complete.")
            
        result = await rag_service.process_query_multi(request.document_ids, request.query)
        return result
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

@app.delete("/documents/{document_id}")
async def delete_document(document_id: str):
    """Delete a document from the system."""
    try:
        await rag_service.delete_document(document_id)
        return {"status": "success", "message": f"Document {document_id} deleted successfully"}
    except Exception as e:
        logger.error(f"Error deleting document: {str(e)}")
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

@app.post("/config/search-engine")
async def set_search_engine(config: SearchEngineConfig):
    """Set the search engine configuration."""
    try:
        logger.info(f"Changing search engine from {rag_service.search_engine} to {config.search_engine}")
        rag_service.search_engine = config.search_engine
        # Reinitialize search tool with new engine
        rag_service.search_tool = None
        await rag_service.ensure_initialized()
        logger.info(f"Search engine successfully changed to {config.search_engine}")
        return {"status": "success", "message": f"Search engine set to {config.search_engine}"}
    except Exception as e:
        logger.error(f"Error setting search engine: {e}")
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)}
        ) 