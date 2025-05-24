# Local Privacy Preserving RAG System with MLX

A web-based RAG (Retrieval Augmented Generation) system that runs a LLM locally to provide privacy protected inference.  The system allows users to upload documents, process them for RAG, and query multiple documents simultaneously using a chat interface. The system uses MLX for accelerated LLM inference on Apple M-series computers.

## Features

- Document upload and processing with multi-document selection and querying
- MLX-accelerated LLM inference using Mistral-7B-Instruct-v0.3-4bit
- Vector database storage using ChromaDB with all-MiniLM-L6-v2 embeddings
- Interactive chat interface with conversation history and source attribution
- FastAPI-based web interface with real-time updates
- Comprehensive error handling and logging

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Requirements

### System Dependencies
- macOS (for MLX support)
- Python 3.9+
- Hugging Face account (for model access)
- Conda package manager
- Poppler (for PDF processing)
  ```bash
  # macOS
  brew install poppler
  # Linux
  sudo apt-get install poppler-utils
  # Windows: Download from http://blog.alivate.com.au/poppler-windows/
  ```
- Tesseract OCR (for text extraction)
  ```bash
  # macOS
  brew install tesseract
  # Linux
  sudo apt-get install tesseract-ocr
  # Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki
  ```
  Note: Ensure Tesseract is in your system PATH

### Python Dependencies
Key packages (full list in `requirements.txt`):
- LangChain and LangChain Community for RAG implementation
- FastAPI and Uvicorn for web server
- MLX and MLX-LM for Apple Silicon optimized inference
- ChromaDB for vector storage
- Sentence Transformers for embeddings
- Document processing tools (PyPDF2, python-docx, unstructured)

## Setup

1. Clone and setup environment:
```bash
git clone <repository-url>
cd local-rag
conda create -n mlx_p39 python=3.9
conda activate mlx_p39
```

2. Install dependencies:
```bash
conda install -c conda-forge mlx
pip install -r requirements.txt --ignore-installed
pip install langchain-core
```

3. Configure Hugging Face:
- Get token from [Access Tokens](https://huggingface.co/settings/tokens)
- Create `.env` file:
```bash
HUGGINGFACE_TOKEN=your_token_here
```

4. Run the application:
```bash
# Local development
uvicorn app.main:app --reload

# Network access
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Access at `http://localhost:8000` (local) or `http://<your-ip-address>:8000` (network)

## Usage

1. Open the web interface
2. Upload documents
3. Select documents using checkboxes
4. Click "Start Chat with Selected Documents"
5. Query documents and view responses with source attribution

## Environment Configuration

```bash
# Required Settings
HUGGINGFACE_TOKEN=your_token_here

# Model Configuration
MODEL_NAME=mlx-community/Mistral-7B-Instruct-v0.3-4bit
EMBEDDING_MODEL=all-MiniLM-L6-v2

# Server Configuration
HOST=0.0.0.0
PORT=8000
LOG_LEVEL=INFO

# Storage Configuration
VECTOR_STORE_DIR=data/vectorstore
DOCUMENTS_DIR=data/documents

# RAG Configuration
CHUNK_SIZE=1000            # Characters per chunk
CHUNK_OVERLAP=200         # Overlap between chunks
MAX_TOKENS=512           # Maximum response tokens
TEMPERATURE=0.7          # Response randomness (0.0-1.0)
```

## Security Notes

1. Never commit `.env` file or tokens to version control
2. Keep Hugging Face token secure
3. Use appropriate file permissions
4. Consider using secrets manager for production

## Project Structure

```
local-rag/
├── app/
│   ├── main.py            # FastAPI application
│   ├── models/            # Data models
│   ├── services/          # Business logic
│   └── templates/         # HTML templates
├── tests/                 # Test cases
├── data/                  # Storage
│   ├── documents/         # Uploaded documents
│   └── vectorstore/       # ChromaDB storage
├── requirements.txt       # Dependencies
└── README.md             # Documentation
```

## Testing

```bash
pytest tests/
```