import re
from nltk.tokenize import sent_tokenize
import nltk
from typing import List

from app.types import Document, Chunk
from app.constants import CHUNK_SIZE, CHUNK_OVERLAP
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

class TextChunker:
    """Chunk text into smaller pieces for processing"""
    
    def __init__(self, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def chunk_documents(self, documents: List[Document]) -> List[Chunk]:
        """Chunk all documents"""
        chunks = []
        for doc in documents:
            doc_chunks = self._chunk_document(doc)
            chunks.extend(doc_chunks)
        return chunks
    
    def _chunk_document(self, document: Document) -> List[Chunk]:
        """Chunk a single document"""
        # Clean the text
        text = self._clean_text(document.content)
        
        # Try sentence-based chunking first
        sentences = sent_tokenize(text)
        chunks = []
        current_chunk = ""
        chunk_count = 0
        
        for sentence in sentences:
            # If adding this sentence would exceed chunk size, create a new chunk
            if len(current_chunk) + len(sentence) > self.chunk_size and current_chunk:
                chunk_id = f"{document.id}_{chunk_count}"
                chunks.append(Chunk(
                    id=chunk_id,
                    content=current_chunk.strip(),
                    document_id=document.id,
                    metadata={
                        **document.metadata,
                        'chunk_index': chunk_count,
                        'total_chunks': 0  # Will be updated later
                    }
                ))
                chunk_count += 1
                
                # Start new chunk with overlap
                current_chunk = self._get_overlap_text(current_chunk) + " " + sentence
            else:
                current_chunk += " " + sentence
        
        # Add the last chunk
        if current_chunk.strip():
            chunk_id = f"{document.id}_{chunk_count}"
            chunks.append(Chunk(
                id=chunk_id,
                content=current_chunk.strip(),
                document_id=document.id,
                metadata={
                    **document.metadata,
                    'chunk_index': chunk_count,
                    'total_chunks': chunk_count + 1
                }
            ))
        
        # Update total_chunks in metadata
        for chunk in chunks:
            chunk.metadata['total_chunks'] = len(chunks)
        
        return chunks
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)]', ' ', text)
        return text.strip()
    
    def _get_overlap_text(self, text: str) -> str:
        """Get overlap text from the end of current chunk"""
        words = text.split()
        if len(words) <= self.overlap:
            return text
        return " ".join(words[-self.overlap:])
