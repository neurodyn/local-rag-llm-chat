# Local RAG System with MLX

A web-based RAG (Retrieval Augmented Generation) system that uses MLX for accelerated LLM inference on Mac laptops. The system allows users to upload documents, process them for RAG, and query them using a chat interface.

## Features

- Document upload and processing
- Vector database storage using ChromaDB
- MLX-accelerated LLM inference using Mistral-7B
- Interactive chat interface with conversation history
- Document summarization
- Source attribution for answers
- FastAPI-based web interface
- Modern UI with real-time updates
- Comprehensive error handling and logging

## Requirements

- macOS (for MLX support)
- Python 3.9+
- Hugging Face account (for model access)
- Conda package manager

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
3. Select a document from the list of uploaded documents
4. Use the chat interface to:
   - Query the document
   - Request document summaries
   - View conversation history
   - See source attributions for answers

## Features in Detail

### RAG Implementation
- Uses LangChain's RetrievalQA chains
- Document chunking with overlap for context preservation
- Semantic search using HuggingFace embeddings
- Different chain types for different tasks:
  - 'stuff' chain for regular queries
  - 'map_reduce' chain for document summarization

### Conversation Management
- Maintains conversation history per document
- Context-aware responses using conversation memory
- Full conversation history retrieval

### Source Attribution
- Shows which parts of the document were used for answers
- Includes metadata about document chunks
- Helps verify answer accuracy

### Error Handling and Logging
- Comprehensive error handling
- Detailed logging for troubleshooting
- HTTP error responses with meaningful messages

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