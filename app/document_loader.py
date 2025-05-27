from typing import List, Tuple, Optional
import hashlib 
import logging
from app.types import Document
from os import Path
import markdown
from bs4 import BeautifulSoup

from app.logger import get_logger

logger = get_logger(__name__)

class DocumentLoader:
    """Load and parse documents from various formats"""
    
    def __init__(self):
        self.supported_extensions = {'.txt', '.md', '.html', '.py', '.js', '.json', '.csv'}
    
    def load_directory(self, directory_path: str) -> List[Document]:
        """Recursively load all supported documents from directory"""
        documents = []
        directory_path = Path(directory_path)
        
        for file_path in directory_path.rglob('*'):
            if file_path.is_file() and file_path.suffix.lower() in self.supported_extensions:
                try:
                    doc = self._load_single_file(file_path)
                    if doc:
                        documents.append(doc)
                        logger.info(f"Loaded: {file_path}")
                except Exception as e:
                    logger.error(f"Error loading {file_path}: {e}")
        
        return documents
    
    def _load_single_file(self, file_path: Path) -> Optional[Document]:
        """Load a single file and return Document object"""
        try:
            content = ""
            extension = file_path.suffix.lower()
            
            if extension == '.html':
                content = self._extract_html(file_path)
            elif extension == '.md':
                content = self._extract_markdown(file_path)
            else:
                # For text files, code files, etc.
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
            
            if not content.strip():
                return None
            
            # Generate document ID
            doc_id = hashlib.md5(str(file_path).encode()).hexdigest()
            
            metadata = {
                'filename': file_path.name,
                'extension': extension,
                'size': file_path.stat().st_size,
                'modified': file_path.stat().st_mtime,
                'path': str(file_path)
            }
            
            return Document(
                id=doc_id,
                content=content,
                metadata=metadata,
                file_path=str(file_path)
            )
            
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
            return None
    
    
    def _extract_html(self, file_path: Path) -> str:
        """Extract text from HTML"""
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            soup = BeautifulSoup(f.read(), 'html.parser')
            return soup.get_text()
    
    def _extract_markdown(self, file_path: Path) -> str:
        """Extract text from Markdown"""
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            md_content = f.read()
            html = markdown.markdown(md_content)
            soup = BeautifulSoup(html, 'html.parser')
            return soup.get_text()
