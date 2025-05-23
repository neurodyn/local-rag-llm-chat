import pytest
from pathlib import Path
from fastapi import UploadFile
from app.services.rag_service import RAGService
from app.models.document import Document

@pytest.fixture
def rag_service():
    """Create a RAG service instance for testing."""
    service = RAGService()
    yield service
    # Cleanup after tests
    for file in Path("data/documents").glob("*"):
        file.unlink()
    for file in Path("data/vectorstore").glob("*"):
        file.unlink()

@pytest.fixture
def sample_document():
    """Create a sample document for testing."""
    content = b"This is a test document for RAG system testing."
    file = UploadFile(
        filename="test.txt",
        file=None,
        content_type="text/plain"
    )
    file._file = content
    return file

async def test_process_document(rag_service, sample_document):
    """Test document processing."""
    document = await rag_service.process_document(sample_document)
    assert isinstance(document, Document)
    assert document.filename == "test.txt"
    assert document.content_type == "text/plain"
    assert document.vector_store_id is not None

def test_list_documents(rag_service):
    """Test document listing."""
    documents = rag_service.list_documents()
    assert isinstance(documents, list)
    for doc in documents:
        assert isinstance(doc, Document)

async def test_process_query(rag_service, sample_document):
    """Test query processing."""
    document = await rag_service.process_document(sample_document)
    response = await rag_service.process_query(document.id, "What is this document about?")
    assert isinstance(response, str)
    assert len(response) > 0

async def test_summarize_document(rag_service, sample_document):
    """Test document summarization."""
    document = await rag_service.process_document(sample_document)
    summary = await rag_service.summarize_document(document.id)
    assert isinstance(summary, str)
    assert len(summary) > 0 