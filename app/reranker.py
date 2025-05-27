from typing import List,Tuple
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from app.types import Chunk 

class Reranker:
    """Simple reranking based on keyword matching and semantic similarity"""
    
    def __init__(self):
        self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    
    def rerank(self, query: str, results: List[Tuple[Chunk, float]], top_k: int = 5) -> List[Tuple[Chunk, float]]:
        """Rerank results based on semantic similarity and keyword matching"""
        if not results:
            return []
        
        # Extract chunks and their vector similarity scores
        chunks = [result[0] for result in results]
        vector_scores = [result[1] for result in results]
        
        # Calculate semantic similarity scores
        query_embedding = self.model.encode([query])
        chunk_embeddings = [chunk.embedding for chunk in chunks]
        semantic_scores = cosine_similarity(query_embedding, chunk_embeddings)[0]
        
        # Calculate keyword matching scores
        keyword_scores = [self._calculate_keyword_score(query, chunk.content) for chunk in chunks]
        
        # Combine scores (weighted)
        final_scores = []
        for i in range(len(chunks)):
            combined_score = (
                0.4 * vector_scores[i] +
                0.4 * semantic_scores[i] +
                0.2 * keyword_scores[i]
            )
            final_scores.append(combined_score)
        
        # Sort by combined score
        ranked_results = list(zip(chunks, final_scores))
        ranked_results.sort(key=lambda x: x[1], reverse=True)
        
        return ranked_results[:top_k]
    
    def _calculate_keyword_score(self, query: str, text: str) -> float:
        """Calculate keyword matching score"""
        query_words = set(query.lower().split())
        text_words = set(text.lower().split())
        
        if not query_words:
            return 0.0
        
        overlap = len(query_words.intersection(text_words))
        return overlap / len(query_words)
