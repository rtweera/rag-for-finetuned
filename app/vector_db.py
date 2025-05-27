import sqlite3
from typing import List, Tuple
import json
import faiss
import numpy as np

from app.types import Chunk
from app.logger import get_logger

logger = get_logger(__name__)

class VectorDatabase:
    """Vector database using FAISS and SQLite"""
    
    def __init__(self, db_path: str = "rag_database.db"):
        self.db_path = db_path
        self.index = None
        self.chunks = []
        self._init_database()
    
    def _init_database(self):
        """Initialize SQLite database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS chunks (
                id TEXT PRIMARY KEY,
                content TEXT,
                document_id TEXT,
                metadata TEXT,
                embedding BLOB
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                file_path TEXT,
                metadata TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def store_chunks(self, chunks: List[Chunk]):
        """Store chunks in database and FAISS index"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Store in SQLite
        for chunk in chunks:
            cursor.execute('''
                INSERT OR REPLACE INTO chunks 
                (id, content, document_id, metadata, embedding)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                chunk.id,
                chunk.content,
                chunk.document_id,
                json.dumps(chunk.metadata),
                chunk.embedding.tobytes()
            ))
        
        conn.commit()
        conn.close()
        
        # Build FAISS index
        self._build_faiss_index(chunks)
        logger.info(f"Stored {len(chunks)} chunks in database")
    
    def _build_faiss_index(self, chunks: List[Chunk]):
        """Build FAISS index for vector similarity search"""
        embeddings = np.array([chunk.embedding for chunk in chunks])
        
        # Create FAISS index
        self.index = faiss.IndexFlatIP(embeddings.shape[1])  # Inner product for cosine similarity
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)
        
        # Store chunks for retrieval
        self.chunks = chunks
    
    def search(self, query_embedding: np.ndarray, k: int = 10) -> List[Tuple[Chunk, float]]:
        """Search for similar chunks"""
        if self.index is None:
            self._load_index()
        
        # Normalize query embedding
        query_embedding = query_embedding.reshape(1, -1)
        faiss.normalize_L2(query_embedding)
        
        # Search
        scores, indices = self.index.search(query_embedding, k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0:  # Valid index
                results.append((self.chunks[idx], float(score)))
        
        return results
    
    def _load_index(self):
        """Load chunks from database and rebuild index"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT id, content, document_id, metadata, embedding FROM chunks')
        rows = cursor.fetchall()
        
        chunks = []
        embeddings = []
        
        for row in rows:
            chunk_id, content, doc_id, metadata_str, embedding_bytes = row
            embedding = np.frombuffer(embedding_bytes, dtype=np.float32)
            
            chunk = Chunk(
                id=chunk_id,
                content=content,
                document_id=doc_id,
                metadata=json.loads(metadata_str),
                embedding=embedding
            )
            chunks.append(chunk)
            embeddings.append(embedding)
        
        conn.close()
        
        if chunks:
            self.chunks = chunks
            self._build_faiss_index(chunks)
