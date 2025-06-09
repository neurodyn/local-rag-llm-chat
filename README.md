# Local Privacy-Preserving RAG System with MLX

A web-based RAG (Retrieval Augmented Generation) system that runs a LLM locally to provide privacy protected inference.  The system allows users to upload documents, process them for RAG, and query multiple documents simultaneously using a chat interface. The system uses MLX for accelerated LLM inference on Apple M-series computers.

## Features

- Document upload and processing with multi-document selection and querying
- MLX-accelerated LLM inference using Mistral-7B-Instruct-v0.3-4bit
- Vector database storage using ChromaDB with all-MiniLM-L6-v2 embeddings
- Interactive chat interface with conversation history and source attribution
- FastAPI-based web interface with real-time updates
- Comprehensive error handling and logging
- Web search integration with multiple search engines:
  - SerpAPI (recommended, requires API key)
  - DuckDuckGo (working)
- Combined search results from both uploaded documents and web searches

## Usage

1. Open the web interface
2. Upload documents
3. Select documents using checkboxes
4. Click "Start Chat with Selected Documents"
5. Query documents and view responses with source attribution
6. To search the web, include phrases like:
   - "search online for..."
   - "search the web for..."
   - "look up online..."
   - "search internet for..."
   The system will automatically combine web search results with document context.

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
- Web search tools (duckduckgo-search, google-search-results)

### Checking MLX Version
You can check your MLX version in several ways:
```bash
# Method 1: Using pip
pip show mlx

# Method 2: Using Python
python -c "import mlx; print(mlx.__version__)"

# Method 3: Check all ML-related package versions
pip list | grep -i mlx
```
Make sure you have MLX version 0.25.2 installed.

## Setup

1. Clone and setup environment:
```bash
git clone <repository-url>
cd local-rag
conda create -n mlx_p39 python=3.9
conda activate mlx_p39
```

2. Install dependencies (make sure we have the right mlx version):
```bash
pip install -r requirements.txt
```

3. Configure environment variables:
- Create `.env` file with the following variables:
```bash
# Required for model access
HUGGINGFACE_TOKEN=your_token_here

# Required for web search (recommended)
SERPAPI_API_KEY=your_serpapi_key_here
```
- Get Hugging Face token from [Access Tokens](https://huggingface.co/settings/tokens)
- Get SerpAPI key from [SerpAPI](https://serpapi.com/) (recommended for reliable web search)

4. Run the application:
```bash
# Local development
uvicorn app.main:app --reload

# Network access
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Access at `http://localhost:8000` (local) or `http://<your-ip-address>:8000` (network)

## Environment Configuration

```bash
# Required Settings
HUGGINGFACE_TOKEN=your_token_here

# Web Search Settings
SERPAPI_API_KEY=your_serpapi_key_here  # Recommended for reliable web search
```

## Security Notes

1. Never commit `.env` file or tokens to version control
2. Keep Hugging Face and SerpAPI tokens secure
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