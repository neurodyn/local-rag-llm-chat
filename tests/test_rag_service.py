import pytest
from pathlib import Path
from fastapi import UploadFile
from app.services.rag_service import RAGService
from app.models.document import Document
import os
import io
from typing import BinaryIO

class MockFile(BinaryIO):
    def __init__(self, content: bytes):
        self._content = io.BytesIO(content)
        self._size = len(content)
        self._position = 0

    def read(self, size: int = -1) -> bytes:
        return self._content.read(size)

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        return self._content.seek(offset, whence)

    def tell(self) -> int:
        return self._content.tell()

    def close(self) -> None:
        self._content.close()

    # Required methods for BinaryIO
    def write(self, s: bytes) -> int:
        return self._content.write(s)

    def writelines(self, lines) -> None:
        self._content.writelines(lines)

    def readable(self) -> bool:
        return True

    def writable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def fileno(self) -> int:
        raise OSError("Not a real file")

    def flush(self) -> None:
        self._content.flush()

@pytest.fixture
def rag_service():
    """Create a RAG service instance for testing."""
    # Ensure directories exist with proper permissions
    data_dir = Path("data")
    docs_dir = data_dir / "documents"
    vector_dir = data_dir / "vectorstore"
    
    data_dir.mkdir(exist_ok=True)
    docs_dir.mkdir(exist_ok=True)
    vector_dir.mkdir(exist_ok=True)
    
    service = RAGService()
    yield service
    
    # Cleanup after tests
    try:
        for file in docs_dir.glob("*"):
            file.unlink(missing_ok=True)
        for file in vector_dir.glob("*"):
            file.unlink(missing_ok=True)
        if Path("data/init_state.json").exists():
            Path("data/init_state.json").unlink()
    except Exception as e:
        print(f"Cleanup error: {e}")

@pytest.fixture
def sample_document():
    """Create a sample document for testing."""
    content = b"This is a test document for RAG system testing."
    mock_file = MockFile(content)
    
    # Create headers dictionary
    headers = {
        'content-type': 'text/plain'
    }
    
    # Create UploadFile with headers
    file = UploadFile(
        filename="test.txt",
        file=mock_file,
        headers=headers
    )
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
    assert isinstance(response, dict)
    assert "response" in response
    assert isinstance(response["response"], str)
    assert len(response["response"]) > 0 