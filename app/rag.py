import sqlite3
from typing import Dict

from app.document_loader import DocumentLoader
from app.text_chunker import TextChunker
from app.embeddings_manager import EmbeddingManager
from app.vector_db import VectorDatabase
from app.reranker import Reranker
from app.model import QwenAPI
from app.logger import get_logger
from app.constants import DB_PATH, MODEL, RAG_TOP_K
logger = get_logger(__name__)

class RAGSystem:
    """Main RAG system orchestrator"""
    
    def __init__(self, directory_path: str, db_path: str = DB_PATH, 
                 qwen_base_url: str = MODEL):
        self.directory_path = directory_path
        self.loader = DocumentLoader()
        self.chunker = TextChunker()
        self.embedding_manager = EmbeddingManager()
        self.vector_db = VectorDatabase(db_path)
        self.reranker = Reranker()
        self.qwen_api = QwenAPI(qwen_base_url)
        
        self.is_indexed = False
    
    def build_index(self):
        """Build the RAG index from documents"""
        logger.info("Starting document indexing...")
        
        # Load documents
        documents = self.loader.load_directory(self.directory_path)
        logger.info(f"Loaded {len(documents)} documents")
        
        if not documents:
            logger.warning("No documents found!")
            return
        
        # Chunk documents
        chunks = self.chunker.chunk_documents(documents)
        logger.info(f"Created {len(chunks)} chunks")
        
        # Generate embeddings
        chunks_with_embeddings = self.embedding_manager.encode_chunks(chunks)
        
        # Store in vector database
        self.vector_db.store_chunks(chunks_with_embeddings)
        
        self.is_indexed = True
        logger.info("Indexing completed!")
    
    def query(self, question: str, top_k: int = RAG_TOP_K) -> str:
        """Query the RAG system"""
        if not self.is_indexed:
            # Try to load existing index
            try:
                self.vector_db._load_index()
                self.is_indexed = True
            except:
                return "Please build the index first using build_index() method."
        
        # Generate query embedding
        query_embedding = self.embedding_manager.encode_query(question)
        
        # Search for relevant chunks
        search_results = self.vector_db.search(query_embedding, k=top_k * 2)
        
        if not search_results:
            return "I couldn't find any relevant information to answer your question."
        
        # Rerank results
        reranked_results = self.reranker.rerank(question, search_results, top_k)
        
        # Prepare context from top chunks
        context_chunks = []
        for chunk, score in reranked_results:
            context_chunks.append(f"Source: {chunk.metadata.get('filename', 'Unknown')}\n{chunk.content}")
        
        context = "\n\n---\n\n".join(context_chunks)
        
        # Create prompt for Qwen
        prompt = self._create_prompt(question, context)
        
        # Generate response
        response = self.qwen_api.generate_response(prompt)
        
        return response
    
    def _create_prompt(self, question: str, context: str) -> str:
        """Create prompt for the language model"""
        return f"""You are a helpful assistant that answers questions based on the provided context. Use the context to answer the question accurately and concisely.

Context:
{context}

Question: {question}

Answer: """

    def get_stats(self) -> Dict:
        """Get statistics about the indexed documents"""
        if not self.is_indexed:
            return {"error": "Index not built yet"}
        
        conn = sqlite3.connect(self.vector_db.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT COUNT(*) FROM chunks')
        chunk_count = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(*) FROM documents')
        doc_count = cursor.fetchone()[0]
        
        conn.close()
        
        return {
            "total_documents": doc_count,
            "total_chunks": chunk_count,
            "embedding_dimension": self.embedding_manager.dimension
        }
