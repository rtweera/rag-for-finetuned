from dataclasses import dataclass
from typing import Dict, Optional
import numpy as np

@dataclass
class Document:
    """Document representation"""
    id: str
    content: str
    metadata: Dict
    file_path: str
    chunk_id: Optional[str] = None

@dataclass
class Chunk:
    """Text chunk representation"""
    id: str
    content: str
    document_id: str
    metadata: Dict
    embedding: Optional[np.ndarray] = None