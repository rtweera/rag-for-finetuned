from sentence_transformers import SentenceTransformer
from typing import List
import numpy as np

from app.types import Chunk
from app.logger import get_logger
from app.constants import EMBEDDING_MODEL

logger = get_logger(__name__)

class EmbeddingManager:
    """Manage embeddings using sentence transformers"""
    
    def __init__(self, model_name: str = EMBEDDING_MODEL):
        """Initialize with a free embedding model"""
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        logger.info(f"Loaded embedding model: {model_name}, dimension: {self.dimension}")
    
    def encode_chunks(self, chunks: List[Chunk]) -> List[Chunk]:
        """Generate embeddings for all chunks"""
        texts = [chunk.content for chunk in chunks]
        embeddings = self.model.encode(texts, show_progress_bar=True)
        
        for chunk, embedding in zip(chunks, embeddings):
            chunk.embedding = embedding
        
        return chunks
    
    def encode_query(self, query: str) -> np.ndarray:
        """Generate embedding for query"""
        return self.model.encode([query])[0]