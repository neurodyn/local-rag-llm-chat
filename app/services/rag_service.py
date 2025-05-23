import os
import logging
from datetime import datetime
from typing import List, Optional, Dict, Tuple, Any
import uuid
from pathlib import Path
import magic
from docx import Document as DocxDocument
from PyPDF2 import PdfReader
from huggingface_hub import hf_hub_download, HfApi
from tqdm.auto import tqdm
from langchain_community.llms.mlx_pipeline import MLXPipeline
from langchain_community.chat_models.mlx import ChatMLX
from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import HumanMessage, AIMessage
from fastapi import UploadFile, HTTPException
from ..models.document import Document

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGService:
    def __init__(self):
        """Initialize the RAG service with basic setup."""
        self.is_initialized = False
        self.initialization_error = None
        self.initialization_status = "Not started"
        self.initialization_progress = {
            "current_step": "",
            "total_steps": 4,
            "current_step_number": 0,
            "download_progress": 0,
            "detailed_status": ""
        }
        self.documents_dir = Path("data/documents")
        self.vector_store_dir = Path("data/vectorstore")
        self.documents_dir.mkdir(parents=True, exist_ok=True)
        self.vector_store_dir.mkdir(parents=True, exist_ok=True)
        self.llm = None
        self.chat_model = None
        self.embeddings = None
        self.vector_store = None
        self.conversations = {}
        
        # Supported file types
        self.supported_mimetypes = {
            'application/pdf': self._extract_pdf_text,
            'application/vnd.openxmlformats-officedocument.wordprocessingml.document': self._extract_docx_text,
            'text/plain': self._extract_text_file,
        }

    def _extract_pdf_text(self, file_path: Path) -> str:
        """Extract text from a PDF file."""
        try:
            reader = PdfReader(file_path)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
            return text
        except Exception as e:
            logger.error(f"Error extracting text from PDF {file_path}: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Error processing PDF file: {str(e)}"
            )

    def _extract_docx_text(self, file_path: Path) -> str:
        """Extract text from a Word document."""
        try:
            doc = DocxDocument(file_path)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text
        except Exception as e:
            logger.error(f"Error extracting text from DOCX {file_path}: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Error processing Word document: {str(e)}"
            )

    def _extract_text_file(self, file_path: Path) -> str:
        """Extract text from a plain text file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except UnicodeDecodeError:
            # Try with different encodings if UTF-8 fails
            try:
                with open(file_path, 'r', encoding='latin-1') as f:
                    return f.read()
            except Exception as e:
                logger.error(f"Error reading text file {file_path}: {e}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Error processing text file: {str(e)}"
                )

    def update_status(self, step: str, progress: int = 0, details: str = ""):
        """Update initialization status with progress information."""
        self.initialization_progress["current_step"] = step
        self.initialization_progress["current_step_number"] += 1
        self.initialization_progress["download_progress"] = progress
        self.initialization_progress["detailed_status"] = details
        self.initialization_status = f"{step} - {details}"
        logger.info(self.initialization_status)

    async def initialize(self):
        """Asynchronously initialize the MLX model and vector store."""
        try:
            if self.is_initialized:
                return

            # Step 1: Check model cache and download if needed
            self.update_status(
                "Checking model files",
                0,
                "Verifying model cache and downloading if needed..."
            )

            model_id = "mlx-community/Mistral-7B-Instruct-v0.3-4bit"
            embedding_model_id = "all-MiniLM-L6-v2"

            # Get model file sizes for progress tracking
            api = HfApi()
            try:
                model_files = api.list_repo_files(model_id)
                total_size = sum(
                    api.get_repo_info(model_id).siblings[i].size 
                    for i, file in enumerate(model_files) 
                    if file.endswith('.bin') or file.endswith('.json')
                )
            except Exception as e:
                logger.warning(f"Could not get model file sizes: {e}")
                total_size = None

            # Download and initialize MLX model
            self.update_status(
                "Downloading Mistral model",
                0,
                "Starting model download..."
            )
            
            try:
                # Initialize MLX model through LangChain
                self.llm = MLXPipeline.from_model_id(
                    model_id,
                    pipeline_kwargs={"max_tokens": 512, "temp": 0.7}
                )
                
                self.update_status(
                    "Initializing MLX model",
                    50,
                    "Model downloaded, setting up MLX pipeline..."
                )
                
                # Create ChatMLX model for better prompt handling
                self.chat_model = ChatMLX(llm=self.llm)
                
            except Exception as e:
                self.update_status(
                    "Error downloading model",
                    0,
                    f"Failed to download model: {str(e)}"
                )
                raise

            # Initialize embeddings
            self.update_status(
                "Setting up embeddings",
                75,
                "Downloading and initializing embedding model..."
            )
            
            self.embeddings = HuggingFaceEmbeddings(
                model_name=embedding_model_id,
                model_kwargs={'device': 'cpu'}
            )
            
            # Initialize vector store
            self.update_status(
                "Configuring vector store",
                90,
                "Setting up document storage..."
            )
            
            # Initialize or load existing vector store
            self.vector_store = Chroma(
                persist_directory=str(self.vector_store_dir),
                embedding_function=self.embeddings
            )
            
            self.update_status(
                "Initialization complete",
                100,
                "System is ready to use"
            )
            
            self.is_initialized = True
            
        except Exception as e:
            self.initialization_error = str(e)
            self.initialization_status = f"Initialization failed: {str(e)}"
            self.initialization_progress["detailed_status"] = str(e)
            logger.error(self.initialization_status)
            raise

    def get_initialization_status(self) -> Dict[str, any]:
        """Get the current initialization status with detailed progress."""
        return {
            "is_initialized": self.is_initialized,
            "status": self.initialization_status,
            "error": self.initialization_error,
            "progress": self.initialization_progress
        }

    async def ensure_initialized(self):
        """Ensure the service is initialized before use."""
        if not self.is_initialized:
            await self.initialize()

    async def process_document(self, file: UploadFile) -> Document:
        """Process an uploaded document and add it to the vector store."""
        await self.ensure_initialized()
        try:
            logger.info(f"Processing document: {file.filename}")
            
            # Generate unique ID for the document
            doc_id = str(uuid.uuid4())
            
            # Save the file
            file_path = self.documents_dir / f"{doc_id}_{file.filename}"
            content = await file.read()
            file_path.write_bytes(content)
            
            # Detect file type
            mime_type = magic.from_file(str(file_path), mime=True)
            
            if mime_type not in self.supported_mimetypes:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unsupported file type: {mime_type}. Supported types are PDF, Word (docx), and text files."
                )
            
            # Extract text using appropriate method
            text = self.supported_mimetypes[mime_type](file_path)
            
            # Create text chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            texts = text_splitter.split_text(text)
            
            # Add to vector store
            self.vector_store.add_texts(
                texts,
                metadatas=[{
                    "source": str(file_path),
                    "doc_id": doc_id,
                    "chunk_id": i,
                    "total_chunks": len(texts)
                } for i in range(len(texts))],
                ids=[f"{doc_id}_{i}" for i in range(len(texts))]
            )
            
            # Initialize conversation history as a list instead of ConversationBufferMemory
            self.conversations[doc_id] = []
            
            # Create and return document metadata
            document = Document(
                id=doc_id,
                filename=file.filename,
                upload_time=datetime.now(),
                content_type=mime_type,
                vector_store_id=doc_id
            )
            
            logger.info(f"Document processed successfully: {doc_id}")
            return document
            
        except HTTPException as e:
            raise e
        except Exception as e:
            logger.error(f"Error processing document {file.filename}: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")
    
    def list_documents(self) -> List[Document]:
        """List all processed documents."""
        try:
            documents = []
            for file_path in self.documents_dir.glob("*"):
                if file_path.is_file():
                    doc_id = file_path.name.split("_")[0]
                    filename = "_".join(file_path.name.split("_")[1:])
                    documents.append(
                        Document(
                            id=doc_id,
                            filename=filename,
                            upload_time=datetime.fromtimestamp(file_path.stat().st_mtime),
                            content_type="application/octet-stream",
                            vector_store_id=doc_id
                        )
                    )
            return documents
        except Exception as e:
            logger.error(f"Error listing documents: {str(e)}")
            raise HTTPException(status_code=500, detail="Error listing documents")
    
    async def process_query(self, document_id: str, query: str) -> Dict[str, Any]:
        """Process a query for a specific document using RAG."""
        await self.ensure_initialized()
        try:
            logger.info(f"Processing query for document {document_id}: {query}")
            
            # Create a document-specific retriever
            retriever = self.vector_store.as_retriever(
                search_kwargs={
                    "filter": {"doc_id": document_id},
                    "k": 4
                }
            )
            
            # Get relevant documents
            relevant_docs = retriever.get_relevant_documents(query)
            logger.info(f"Number of relevant documents retrieved: {len(relevant_docs)}")
            
            # Log each document's content and metadata
            for i, doc in enumerate(relevant_docs):
                logger.info(f"Document {i + 1} metadata: {doc.metadata}")
                logger.info(f"Document {i + 1} content: {doc.page_content[:200]}...")  # First 200 chars
            
            # Create context from relevant documents
            context = "\n\n".join([doc.page_content for doc in relevant_docs])
            logger.info(f"Combined context length: {len(context)} characters")
            
            # Create prompt
            prompt = f"""Use the following pieces of context to answer the question. If you cannot find the answer in the context, say "I don't have enough information to answer that."

Context:
{context}

Question: {query}

Answer:"""
            
            # Get response from model
            response = self.chat_model.invoke(prompt)
            logger.info(f"Raw model response type: {type(response)}")
            logger.info(f"Raw model response: {response}")
            
            # Ensure response is a string
            if isinstance(response, dict) and 'content' in response:
                response_text = response['content']
                logger.info("Using response['content']")
            elif isinstance(response, (list, tuple)) and len(response) > 0:
                response_text = str(response[0])
                logger.info("Using response[0]")
            elif hasattr(response, 'content'):
                response_text = response.content
                logger.info("Using response.content")
            elif hasattr(response, 'text'):
                response_text = response.text
                logger.info("Using response.text")
            else:
                response_text = str(response)
                logger.info("Using str(response)")
            
            # Clean up response text
            response_text = response_text.strip()
            logger.info(f"Final response text: {response_text}")
            
            # Extract source information
            sources = []
            for doc in relevant_docs:
                sources.append({
                    "content": doc.page_content,
                    "metadata": doc.metadata
                })
            
            # Update conversation history using the proper method
            if document_id not in self.conversations:
                self.conversations[document_id] = []
            
            # Add messages to conversation history as a list instead
            self.conversations[document_id].extend([
                {"role": "user", "content": query},
                {"role": "assistant", "content": response_text}
            ])
            
            logger.info(f"Query processed successfully for document {document_id}")
            
            # Return structured response
            return {
                "response": response_text,
                "sources": sources
            }
            
        except Exception as e:
            logger.error(f"Error processing query for document {document_id}: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")
    
    def get_conversation_history(self, document_id: str) -> List[Dict]:
        """Get the conversation history for a specific document."""
        try:
            return self.conversations.get(document_id, [])
        except Exception as e:
            logger.error(f"Error retrieving conversation history for document {document_id}: {str(e)}")
            raise HTTPException(status_code=500, detail="Error retrieving conversation history")
    
    async def summarize_document(self, document_id: str) -> Tuple[str, List[Dict]]:
        """Generate a summary for a specific document."""
        await self.ensure_initialized()
        try:
            logger.info(f"Generating summary for document {document_id}")
            
            # Get all chunks for the document
            retriever = self.vector_store.as_retriever(
                search_kwargs={
                    "filter": {"doc_id": document_id},
                    "k": 100  # Get more chunks for summarization
                }
            )
            
            # Create a summarization prompt
            summary_query = "Please provide a comprehensive summary of this document."
            
            # Use the same QA chain for summarization
            qa_chain = ConversationalRetrievalChain.from_llm(
                llm=self.chat_model,
                retriever=retriever,
                return_source_documents=True
            )
            
            # Generate summary
            response = qa_chain.invoke({"query": summary_query})
            
            # Extract source information
            sources = []
            for doc in response["source_documents"]:
                sources.append({
                    "content": doc.page_content,
                    "metadata": doc.metadata
                })
            
            logger.info(f"Summary generated successfully for document {document_id}")
            return response["result"], sources
            
        except Exception as e:
            logger.error(f"Error generating summary for document {document_id}: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error generating summary: {str(e)}") 