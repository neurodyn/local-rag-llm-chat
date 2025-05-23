# Local RAG System with MLX

A web-based RAG (Retrieval Augmented Generation) system that uses MLX for accelerated LLM inference on Mac laptops. The system allows users to upload documents, process them for RAG, and query multiple documents simultaneously using a chat interface.

## Features

- Document upload and processing
- Multi-document selection and querying
- Vector database storage using ChromaDB
- MLX-accelerated LLM inference using Mistral-7B
- Interactive chat interface with conversation history
- Source attribution for answers
- FastAPI-based web interface
- Modern UI with real-time updates
- Comprehensive error handling and logging

## Requirements

- macOS (for MLX support)
- Python 3.9+
- Hugging Face account (for model access)
- Conda package manager

## System Requirements

### System Dependencies
- Python 3.9+
- Poppler (for PDF processing)
  - macOS: `brew install poppler`
  - Linux: `sudo apt-get install poppler-utils`
  - Windows: Download from [poppler releases](http://blog.alivate.com.au/poppler-windows/)
- Tesseract OCR (for text extraction from PDFs and images)
  - macOS: `brew install tesseract`
  - Linux: `sudo apt-get install tesseract-ocr`
  - Windows: Download installer from [UB Mannheim](https://github.com/UB-Mannheim/tesseract/wiki)
  - Note: After installation, ensure Tesseract is in your system PATH

### Python Dependencies
All Python dependencies are listed in `requirements.txt` and include:
- LangChain and LangChain Community packages for RAG implementation
- FastAPI and Uvicorn for web server
- MLX and MLX-LM for Apple Silicon optimized inference
- ChromaDB for vector storage
- Sentence Transformers for embeddings
- Document processing:
  - PyPDF2 and pdf2image for PDF handling
  - python-docx for Word documents
  - unstructured and unstructured-inference for enhanced document parsing
  - python-magic for file type detection
  - pytesseract for OCR capabilities

## Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd local-rag
```

2. Create and activate the conda environment:
```bash
conda create -n mlx_p39 python=3.9
conda activate mlx_p39
```

3. Install MLX:
```bash
conda install -c conda-forge mlx
```

4. Install other dependencies:
```bash
pip install -r requirements.txt --ignore-installed
pip install langchain-core
```

5. Create a `.env` file and add your Hugging Face token:
```bash
HUGGINGFACE_TOKEN=your_token_here
```

6. Run the application:
```bash
uvicorn app.main:app --reload
```

The application will be available at `http://localhost:8000`

## Project Structure

```
local-rag/
├── app/
│   ├── main.py            # FastAPI application
│   ├── models/            # Data models
│   ├── services/          # Business logic
│   └── templates/         # HTML templates
├── tests/                 # Test cases
├── data/                  # Storage for vector database
│   ├── documents/         # Uploaded documents
│   └── vectorstore/       # ChromaDB storage
├── requirements.txt       # Project dependencies
└── README.md             # This file
```

## Usage

1. Open the web interface at `http://localhost:8000`
2. Upload documents using the upload interface
3. Select one or more documents using the checkboxes
4. Click "Start Chat with Selected Documents" to begin querying
5. Use the chat interface to:
   - Query across all selected documents
   - View conversation history
   - See source attributions for answers

## Features in Detail

### Multi-Document RAG Implementation
- Supports querying across multiple documents simultaneously
- Dynamic retrieval scaling based on number of documents
- Ensures balanced context from all selected documents
- Document coverage tracking for comprehensive answers
- Unified conversation history across document sets

### RAG Implementation
- Uses LangChain's RetrievalQA chains
- Document chunking with overlap for context preservation
- Semantic search using HuggingFace embeddings
- Retriever configuration:
  - Automatic scaling of retrieved chunks based on document count
  - Minimum of 2 chunks per document for context
  - Document coverage tracking to ensure all selected documents are represented

### Conversation Management
- Maintains conversation history per document set
- Context-aware responses using conversation memory
- Full conversation history retrieval
- Synchronized history across multiple documents

### Source Attribution
- Shows which parts of each document were used for answers
- Includes metadata about document chunks
- Helps verify answer accuracy
- Tracks document coverage in multi-document queries

### Error Handling and Logging
- Comprehensive error handling
- Detailed logging for troubleshooting
- HTTP error responses with meaningful messages
- Document coverage logging for debugging

## Testing

Run the tests using:
```bash
pytest tests/
```

## Model Information

This system uses the Mistral-7B-Instruct-v0.3-4bit model from MLX community, optimized for Apple Silicon:
- Model: mlx-community/Mistral-7B-Instruct-v0.3-4bit
- Accelerated using MLX for optimal performance on Mac
- Integrated with LangChain for RAG capabilities

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a new Pull Request 

## Environment Configuration

Create a `.env` file in the root directory with the following variables:

```bash
# Required Settings
HUGGINGFACE_TOKEN=your_token_here  # Get this from https://huggingface.co/settings/tokens

# Model Configuration
MODEL_NAME=mlx-community/Mistral-7B-Instruct-v0.3-4bit  # The MLX model to use
EMBEDDING_MODEL=all-MiniLM-L6-v2                        # Model for text embeddings

# Server Configuration
HOST=0.0.0.0                # Server host
PORT=8000                   # Server port
LOG_LEVEL=INFO             # Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)

# Vector Store Configuration
VECTOR_STORE_DIR=data/vectorstore    # Directory for ChromaDB
DOCUMENTS_DIR=data/documents         # Directory for uploaded documents

# RAG Configuration
CHUNK_SIZE=1000            # Size of text chunks for processing
CHUNK_OVERLAP=200          # Overlap between chunks
MAX_TOKENS=512             # Maximum tokens for model response
TEMPERATURE=0.7            # Model temperature (0.0 - 1.0)
```

### Environment Variables Explained

1. **Authentication**
   - `HUGGINGFACE_TOKEN`: Your Hugging Face API token for accessing the MLX model
     - Required for downloading and using the model
     - Get it from your Hugging Face account settings

2. **Model Settings**
   - `MODEL_NAME`: The MLX model to use for text generation
   - `EMBEDDING_MODEL`: The model used for creating text embeddings
     - Default is all-MiniLM-L6-v2 for good performance on Mac

3. **Server Configuration**
   - `HOST`: The server host address
   - `PORT`: The port number for the web interface
   - `LOG_LEVEL`: Controls the verbosity of logging
     - DEBUG: Detailed information for debugging
     - INFO: General information about operation
     - WARNING: Unexpected or notable events
     - ERROR: Serious issues that need attention
     - CRITICAL: Critical issues that need immediate attention

4. **Storage Configuration**
   - `VECTOR_STORE_DIR`: Location for the vector database
   - `DOCUMENTS_DIR`: Location for uploaded document storage
     - Both directories are created automatically if they don't exist

5. **RAG Configuration**
   - `CHUNK_SIZE`: Number of characters per text chunk
     - Larger chunks provide more context but use more memory
   - `CHUNK_OVERLAP`: Number of overlapping characters between chunks
     - Helps maintain context across chunk boundaries
   - `MAX_TOKENS`: Maximum number of tokens in model responses
   - `TEMPERATURE`: Controls response randomness
     - Lower values (e.g., 0.3) for more focused responses
     - Higher values (e.g., 0.7) for more creative responses

### Security Notes

1. Never commit your `.env` file to version control
2. Keep your Hugging Face token secure
3. Use appropriate file permissions for the data directories
4. Consider using a secrets manager for production deployments 

## Hugging Face Setup

### Getting Your Token

1. Create a Hugging Face account at [huggingface.co](https://huggingface.co) if you don't have one
2. Go to your [Access Tokens](https://huggingface.co/settings/tokens) page
3. Click "New token" and create a token with `read` access
4. Copy your token for use in the next steps

### Setting Up Token Access

There are two ways to configure your Hugging Face token:

1. **Using the CLI (Recommended)**:
   ```bash
   # Install huggingface_cli if not already installed
   pip install -U "huggingface_hub[cli]"
   
   # Login using the CLI
   huggingface-cli login
   # Enter your token when prompted
   ```

2. **Using Environment Variables**:
   - Add to your `.env` file:
     ```bash
     HUGGINGFACE_TOKEN=your_token_here
     ```
   - Or export in your shell:
     ```bash
     export HUGGINGFACE_TOKEN=your_token_here
     ```

### Accessing Mistral MLX Model

The system uses the quantized Mistral-7B model optimized for MLX:
```
mlx-community/Mistral-7B-Instruct-v0.3-4bit
```

To ensure you have access:

1. Visit the [model page](https://huggingface.co/mlx-community/Mistral-7B-Instruct-v0.3-4bit)
2. Accept the model's terms of use on the Hugging Face website
3. The system will automatically download the model on first use

### Troubleshooting Token Issues

If you encounter authentication issues:

1. Verify your token is correct:
   ```bash
   huggingface-cli whoami
   ```

2. Check token permissions:
   ```bash
   huggingface-cli list-models --token your_token_here
   ```

3. Common issues and solutions:
   - "Token not found": Ensure your token is properly set in `.env` or environment
   - "Permission denied": Make sure you've accepted the model's terms of use
   - "Token has insufficient permissions": Create a new token with `read` access
   - "Token expired": Generate a new token on Hugging Face website

4. If using the CLI doesn't work, try:
   ```bash
   # Clear any existing token
   huggingface-cli logout
   
   # Login again
   huggingface-cli login
   ```

### Security Best Practices

1. Never commit your token to version control
2. Rotate your token periodically
3. Use read-only tokens for deployment
4. Set appropriate environment variables in your deployment environment
5. Consider using a secrets manager for production deployments 

## Installation

1. Install system dependencies:
```bash
# macOS
brew install poppler

# Linux
sudo apt-get install poppler-utils

# Windows
# Download and install poppler from http://blog.alivate.com.au/poppler-windows/
```

2. Install Python dependencies:
```bash
pip install -r requirements.txt
```