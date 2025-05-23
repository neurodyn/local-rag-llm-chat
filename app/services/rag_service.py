import os
import logging
from datetime import datetime
from typing import List, Optional, Dict, Tuple
import uuid
from pathlib import Path
from langchain_community.llms.mlx_pipeline import MLXPipeline
from langchain_community.chat_models.mlx import ChatMLX
from langchain.chains import RetrievalQA
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
        """Initialize the RAG service with MLX model and vector store."""
        try:
            # Initialize MLX model through LangChain
            logger.info("Initializing MLX model...")
            self.llm = MLXPipeline.from_model_id(
                "mlx-community/Mistral-7B-Instruct-v0.3-4bit",
                pipeline_kwargs={"max_tokens": 512, "temp": 0.7}
            )
            
            # Create ChatMLX model for better prompt handling
            self.chat_model = ChatMLX(llm=self.llm)
            
            # Setup document storage
            self.documents_dir = Path("data/documents")
            self.documents_dir.mkdir(parents=True, exist_ok=True)
            
            # Initialize vector store with embeddings
            logger.info("Initializing vector store...")
            self.embeddings = HuggingFaceEmbeddings(
                model_name="all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'}
            )
            self.vector_store = Chroma(
                persist_directory="data/vectorstore",
                embedding_function=self.embeddings
            )
            
            # Initialize conversation memories for each document
            self.conversations: Dict[str, ConversationBufferMemory] = {}
            
            logger.info("RAG service initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize RAG service: {str(e)}")
            raise
        
    async def process_document(self, file: UploadFile) -> Document:
        """Process an uploaded document and add it to the vector store."""
        try:
            logger.info(f"Processing document: {file.filename}")
            
            # Generate unique ID for the document
            doc_id = str(uuid.uuid4())
            
            # Save the file
            file_path = self.documents_dir / f"{doc_id}_{file.filename}"
            content = await file.read()
            file_path.write_bytes(content)
            
            # Create text chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            texts = text_splitter.split_text(content.decode())
            
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
            
            # Initialize conversation memory for this document
            self.conversations[doc_id] = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True
            )
            
            # Create and return document metadata
            document = Document(
                id=doc_id,
                filename=file.filename,
                upload_time=datetime.now(),
                content_type=file.content_type,
                vector_store_id=doc_id
            )
            
            logger.info(f"Document processed successfully: {doc_id}")
            return document
            
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
    
    async def process_query(self, document_id: str, query: str) -> Tuple[str, List[Dict]]:
        """Process a query for a specific document using RAG."""
        try:
            logger.info(f"Processing query for document {document_id}: {query}")
            
            # Create a document-specific retriever
            retriever = self.vector_store.as_retriever(
                search_kwargs={
                    "filter": {"doc_id": document_id},
                    "k": 4
                }
            )
            
            # Get conversation memory for this document
            memory = self.conversations.get(document_id)
            if not memory:
                memory = ConversationBufferMemory(
                    memory_key="chat_history",
                    return_messages=True
                )
                self.conversations[document_id] = memory
            
            # Update the chain with document-specific retriever and memory
            qa_chain = RetrievalQA.from_chain_type(
                llm=self.chat_model,
                chain_type="stuff",
                retriever=retriever,
                memory=memory,
                return_source_documents=True
            )
            
            # Execute the query
            response = qa_chain.invoke({"query": query})
            
            # Extract source information
            sources = []
            for doc in response["source_documents"]:
                sources.append({
                    "content": doc.page_content,
                    "metadata": doc.metadata
                })
            
            # Update conversation history
            memory.chat_memory.add_user_message(query)
            memory.chat_memory.add_ai_message(response["result"])
            
            logger.info(f"Query processed successfully for document {document_id}")
            return response["result"], sources
            
        except Exception as e:
            logger.error(f"Error processing query for document {document_id}: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")
    
    async def summarize_document(self, document_id: str) -> Tuple[str, List[Dict]]:
        """Generate a summary for a specific document."""
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
            qa_chain = RetrievalQA.from_chain_type(
                llm=self.chat_model,
                chain_type="map_reduce",  # Use map_reduce for better handling of large documents
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
            
    def get_conversation_history(self, document_id: str) -> List[Dict]:
        """Get the conversation history for a specific document."""
        try:
            memory = self.conversations.get(document_id)
            if not memory:
                return []
            
            history = []
            for message in memory.chat_memory.messages:
                if isinstance(message, HumanMessage):
                    history.append({"role": "user", "content": message.content})
                elif isinstance(message, AIMessage):
                    history.append({"role": "assistant", "content": message.content})
            
            return history
        except Exception as e:
            logger.error(f"Error retrieving conversation history for document {document_id}: {str(e)}")
            raise HTTPException(status_code=500, detail="Error retrieving conversation history") 