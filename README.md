# RAG System for Fine-tuned Qwen Models

A comprehensive Retrieval-Augmented Generation (RAG) system designed to work with fine-tuned Qwen2.5 3B models. This system enables question-answering capabilities over custom document collections by combining document retrieval with a fine-tuned language model for accurate, context-aware responses.

## 📋 Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
  - [Building the Index](#building-the-index)
  - [Querying the System](#querying-the-system)
  - [Interactive Mode](#interactive-mode)
- [Configuration](#configuration)
- [Project Structure](#project-structure)
- [Components](#components)
- [How It Works](#how-it-works)
- [Requirements](#requirements)
- [Contributing](#contributing)
- [Future Improvements](#future-improvements)
- [License](#license)

## ✨ Features

- **Multi-format Document Support**: Loads and processes documents in multiple formats:
  - Plain text (.txt)
  - Markdown (.md)
  - HTML (.html)
  - Code files (.py, .js)
  - JSON (.json)
  - CSV (.csv)

- **Intelligent Text Chunking**: 
  - Sentence-based text chunking with configurable size and overlap
  - Preserves context through overlapping chunks
  - Metadata enrichment with source file information

- **Efficient Embeddings**:
  - Uses open-source sentence-transformers for embedding generation
  - All-MiniLM-L6-v2 model (384-dimensional embeddings)
  - Fast batch processing with progress tracking

- **Vector Database**:
  - FAISS-based similarity search for efficient retrieval
  - SQLite backend for persistent storage
  - Scalable architecture supporting large document collections

- **Intelligent Reranking**:
  - Multi-factor reranking strategy combining:
    - Vector similarity scores (40% weight)
    - Semantic similarity scores (40% weight)
    - Keyword matching scores (20% weight)
  - Improves retrieval quality and result relevance

- **Fine-tuned Model Integration**:
  - Seamless integration with Qwen2.5 3B fine-tuned models
  - Customizable API endpoints
  - Context-aware response generation

- **Interactive & Batch Modes**:
  - Interactive CLI for real-time querying
  - Batch processing with single queries
  - Index building and statistics reporting

## 🏗️ Architecture

The system follows a modular architecture with clear separation of concerns:

```
┌─────────────────────────────────────┐
│         User Interface              │
│      (CLI / Interactive Mode)       │
└─────────┬───────────────────────────┘
          │
┌─────────▼───────────────────────────┐
│       RAG System Orchestrator       │
│    (Main coordination logic)        │
└─────────┬───────────────────────────┘
          │
    ┌─────┴──────┬──────────┬─────────────┬────────────┐
    │            │          │             │            │
    ▼            ▼          ▼             ▼            ▼
┌────────┐ ┌─────────┐ ┌─────────┐ ┌──────────┐ ┌──────────┐
│Document│ │  Text   │ │Embedding│ │  Vector  │ │Reranker &│
│ Loader │ │Chunker  │ │ Manager │ │ Database │ │Qwen API  │
└────────┘ └─────────┘ └─────────┘ └──────────┘ └──────────┘
    │            │          │             │            │
    └─────┬──────┴──────────┴─────────────┴────────────┘
          │
    ┌─────▼────────────────────┐
    │   Persistent Storage     │
    │ (SQLite + FAISS Index)   │
    └──────────────────────────┘
```

## 📦 Installation

### Prerequisites

- Python 3.8+
- pip package manager

### Step 1: Clone the Repository

```bash
git clone https://github.com/rtweera/rag-for-finetuned.git
cd rag-for-finetuned
```

### Step 2: Create Virtual Environment (Recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Download Required Models

The system will automatically download the sentence-transformers model on first run:

```bash
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"
```

### Step 5: Set Up Your Qwen Model

Before using the system, ensure your fine-tuned Qwen2.5 3B model is running at the specified endpoint. The default configuration expects the model to be accessible at `http://localhost:8000`.

You can customize the model URL by modifying `constants.py` or passing it as a command-line argument.

## 🚀 Quick Start

### 1. Prepare Your Documents

Create a directory with your documents:

```bash
mkdir my_documents
# Add your .txt, .md, .html, .py, .js, .json, or .csv files
```

### 2. Build the Index

```bash
python main.py --directory ./my_documents --build-index
```

Output:
```
Index built successfully!
Documents: 15
Chunks: 342
```

### 3. Query the System

```bash
python main.py --directory ./my_documents --query "What is the main topic?"
```

### 4. Interactive Mode

```bash
python main.py --directory ./my_documents
```

Then enter your questions:

```
Interactive RAG System
Type 'quit' to exit

Enter your question: What are the key concepts?
Answer: [Response from Qwen model with relevant context]

Enter your question: quit
```

## 📖 Usage

### Building the Index

The index must be built before querying. This process:

1. Loads all supported documents from the specified directory
2. Chunks the documents into manageable pieces
3. Generates embeddings for each chunk
4. Stores chunks and embeddings in the vector database

```bash
python main.py --directory ./my_documents --build-index
```

**Options:**
- `--directory`: Path to directory containing documents (required)
- `--db-path`: Custom database path (default: `rag_database.db`)

### Querying the System

Query after building the index:

```bash
python main.py --directory ./my_documents --query "Your question here"
```

**Options:**
- `--directory`: Path to document directory (required)
- `--query`: Your question (required for batch mode)
- `--qwen-url`: Qwen model API URL (default: `http://localhost:8000`)
- `--db-path`: Custom database path

### Interactive Mode

Launch interactive mode without any additional flags:

```bash
python main.py --directory ./my_documents
```

**Interactive Mode Features:**
- Type questions and get immediate responses
- Type `quit` or `exit` to end the session
- Maintains conversation context through document retrieval
- All responses are based on indexed documents

## ⚙️ Configuration

### Default Configuration (app/constants.py)

```python
# Embedding Model Configuration
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # 384-dim embeddings

# Text Chunking
CHUNK_SIZE = 512          # Characters per chunk
CHUNK_OVERLAP = 50        # Character overlap between chunks

# Database
DB_PATH = "rag_database.db"

# Model Endpoints
MODEL = "http://localhost:8000"

# Retrieval Configuration
RAG_TOP_K = 5             # Number of top results for reranking
```

### Customizing Configuration

#### Option 1: Modify constants.py

Edit `app/constants.py` directly:

```python
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"  # Different model
CHUNK_SIZE = 256          # Smaller chunks
RAG_TOP_K = 10            # Retrieve more results
```

#### Option 2: Command-line Arguments

```bash
python main.py \
  --directory ./my_documents \
  --db-path ./custom_db.db \
  --qwen-url "http://your-server:8000"
```

### Fine-tuning the Reranker

The reranker combines three scoring methods. Adjust weights in `app/reranker.py`:

```python
combined_score = (
    0.4 * vector_scores[i] +       # Vector similarity (40%)
    0.4 * semantic_scores[i] +     # Semantic similarity (40%)
    0.2 * keyword_scores[i]        # Keyword matching (20%)
)
```

## 📁 Project Structure

```
rag-for-finetuned/
├── README.md                      # This file
├── main.py                        # Entry point and CLI
├── app/
│   ├── __init__.py               # Package initialization
│   ├── constants.py              # Configuration constants
│   ├── types.py                  # Type definitions (Document, Chunk)
│   ├── logger.py                 # Logging configuration
│   ├── rag.py                    # Main RAG orchestrator
│   ├── document_loader.py        # Document loading and parsing
│   ├── text_chunker.py           # Text chunking logic
│   ├── embeddings_manager.py     # Embedding generation
│   ├── vector_db.py              # Vector database (FAISS + SQLite)
│   ├── reranker.py               # Result reranking logic
│   └── model.py                  # Qwen API interface
├── sandbox/
│   └── demo-chroma.py            # Demo scripts and experiments
└── .gitignore                    # Git ignore patterns
```

## 🔧 Components

### 1. Document Loader (`app/document_loader.py`)

Loads and parses documents in multiple formats.

**Supported Formats:**
- Plain text (.txt)
- Markdown (.md)
- HTML (.html)
- Code (.py, .js)
- Data formats (.json, .csv)

**Features:**
- Recursive directory traversal
- Metadata extraction (filename, size, modification time)
- Error handling for corrupted files
- HTML/Markdown text extraction

### 2. Text Chunker (`app/text_chunker.py`)

Intelligently chunks documents for processing.

**Algorithm:**
- Sentence-based chunking
- Configurable chunk size and overlap
- Metadata preservation per chunk
- Batch processing for multiple documents

**Parameters:**
- `CHUNK_SIZE`: 512 characters (default)
- `CHUNK_OVERLAP`: 50 characters (default)

### 3. Embedding Manager (`app/embeddings_manager.py`)

Generates vector embeddings for text.

**Features:**
- Uses sentence-transformers library
- all-MiniLM-L6-v2 model (384-dimensional)
- Batch encoding with progress tracking
- Separate query encoding

### 4. Vector Database (`app/vector_db.py`)

Stores and retrieves vectors efficiently.

**Components:**
- **SQLite**: Persistent metadata storage
- **FAISS**: High-performance vector similarity search
- Normalized embeddings for cosine similarity
- Tables:
  - `chunks`: Chunk content and metadata
  - `documents`: Document information

**Operations:**
- Store chunks with embeddings
- Similarity search
- Index persistence and loading

### 5. Reranker (`app/reranker.py`)

Improves retrieval quality through intelligent reranking.

**Ranking Factors:**
1. **Vector Similarity** (40%): FAISS index similarity scores
2. **Semantic Similarity** (40%): Sentence transformer similarity
3. **Keyword Matching** (20%): Query keyword overlap

**Result:** Top K most relevant chunks

### 6. RAG Orchestrator (`app/rag.py`)

Coordinates all components.

**Functions:**
- `build_index()`: Create vector database from documents
- `query(question)`: Answer questions using RAG pipeline
- `get_stats()`: Retrieve indexing statistics

**Query Pipeline:**
1. Generate query embedding
2. Search vector database (FAISS)
3. Rerank results
4. Build context from top chunks
5. Generate prompt for Qwen
6. Return AI response

### 7. Qwen API Interface (`app/model.py`)

Interfaces with fine-tuned Qwen models.

**Current Status:** Template requiring implementation

**To Implement:**
```python
def generate_response(self, prompt: str, max_tokens: int = 512) -> str:
    """Generate response using Qwen model"""
    # Implement based on your model's API format
    # Supports OpenAI-compatible endpoints or custom APIs
```

## 🔄 How It Works

### Index Building Process

```
Input Documents
      │
      ▼
Document Loader
(Parse & Extract Text)
      │
      ▼
Text Chunker
(Split into chunks)
      │
      ▼
Embedding Manager
(Generate vectors)
      │
      ▼
Vector Database
(Store in SQLite + FAISS)
      │
      ▼
Index Ready ✓
```

### Query Processing Pipeline

```
User Question
      │
      ▼
Generate Query Embedding
      │
      ▼
Vector Search (FAISS)
      │
      ▼
Rerank Results
(Vector + Semantic + Keyword)
      │
      ▼
Build Context
(Top K chunks)
      │
      ▼
Create Prompt
      │
      ▼
Qwen Model
(Generate Response)
      │
      ▼
Return Answer to User
```

### Example Query Flow

**User Question:**
```
"What is the main purpose of this project?"
```

**Processing:**
1. Query embedding is generated (384-dimensional vector)
2. FAISS searches for top 10 similar chunks
3. Reranker scores them using three methods
4. Top 5 chunks are selected
5. Context is built from these chunks:
   ```
   Source: README.md
   [Chunk content 1]
   
   Source: specification.txt
   [Chunk content 2]
   
   ... (up to 5 chunks)
   ```
6. Prompt is created:
   ```
   You are a helpful assistant...
   
   Context:
   [Combined chunk content]
   
   Question: What is the main purpose of this project?
   
   Answer:
   ```
7. Qwen model generates response based on context
8. Response is returned to user

## 📋 Requirements

### Core Dependencies

- **sentence-transformers**: For embedding generation
- **faiss-cpu**: For vector similarity search (CPU version)
  - Use `faiss-gpu` for GPU support
- **sqlite3**: Included with Python
- **BeautifulSoup4**: For HTML parsing
- **markdown**: For Markdown parsing
- **nltk**: For sentence tokenization
- **scikit-learn**: For reranking calculations
- **numpy**: Numerical computing

### Python Version

- Python 3.8 or higher

### Qwen Model

- Qwen2.5 3B (or compatible fine-tuned model)
- Running on accessible endpoint (default: localhost:8000)

## 🤝 Contributing

Contributions are welcome! Here's how to contribute:

1. **Fork** the repository
2. **Create a feature branch**: `git checkout -b feature/your-feature`
3. **Make changes** and test thoroughly
4. **Commit** with clear messages: `git commit -am 'Add your feature'`
5. **Push** to your fork: `git push origin feature/your-feature`
6. **Open a Pull Request** with detailed description

### Areas for Contribution

- [ ] Implement Qwen model integration
- [ ] Add support for more document formats
- [ ] Optimize embedding generation
- [ ] Improve reranking strategies
- [ ] Add unit and integration tests
- [ ] Expand documentation
- [ ] Performance optimization
- [ ] GPU acceleration support

## 🚀 Future Improvements

### Planned Features

1. **Advanced Reranking**
   - Cross-encoder based reranking
   - Learning-to-rank approaches
   - Query expansion

2. **Document Management**
   - Incremental indexing (update without rebuilding)
   - Document versioning
   - Deletion and modification support

3. **Enhanced Retrieval**
   - Hybrid search (dense + sparse)
   - Multi-hop reasoning
   - Query reformulation

4. **Performance**
   - GPU acceleration for embeddings
   - Quantization for embeddings
   - Index compression

5. **Scalability**
   - Distributed indexing
   - Streaming document processing
   - Cloud storage integration

6. **Observability**
   - Query logging
   - Performance metrics
   - Relevance feedback

7. **Testing**
   - Unit tests
   - Integration tests
   - Performance benchmarks

8. **API & Integration**
   - REST API endpoint
   - Web UI
   - Integration with LangChain
   - Streaming responses

## 📝 Implementation Notes

### Current Status

- ✅ Core RAG pipeline implemented
- ✅ Document loading and chunking
- ✅ Embedding generation
- ✅ Vector database with FAISS
- ✅ Reranking logic
- ✅ CLI interface
- ⚠️ Qwen API integration (template provided, needs implementation)

### Known Limitations

1. **Qwen Model Integration**: Currently requires implementation based on your model's API format
2. **Memory Usage**: For large document collections, consider:
   - Using `faiss-gpu` for GPU acceleration
   - Implementing batch processing
   - Using embedding quantization
3. **Model Selection**: all-MiniLM-L6-v2 is a good general-purpose model but can be replaced with domain-specific embeddings

### Troubleshooting

**Issue: "No documents found"**
- Ensure documents exist in the specified directory
- Check file extensions are supported (.txt, .md, .html, .py, .js, .json, .csv)

**Issue: "Please build the index first"**
- Run `python main.py --directory ./path --build-index` before querying

**Issue: "Connection refused to Qwen model"**
- Ensure your Qwen model is running at the specified endpoint
- Check the `MODEL` configuration in `constants.py`
- Implement the QwenAPI class in `app/model.py`

**Issue: Slow embedding generation**
- Use `faiss-gpu` for GPU acceleration
- Reduce `CHUNK_SIZE` for smaller chunks
- Consider batch processing

## 📄 License

This project is open source. Please check the LICENSE file for details.

## 📞 Support

For issues, questions, or suggestions:

1. **GitHub Issues**: Create an issue in the repository
2. **Documentation**: Refer to inline code comments and this README
3. **Code Examples**: Check the `sandbox/` directory for example usage

## 🎯 Quick Reference Commands

```bash
# Build index from documents
python main.py --directory ./my_documents --build-index

# Query with a single question
python main.py --directory ./my_documents --query "Your question"

# Interactive mode
python main.py --directory ./my_documents

# Custom database and model
python main.py --directory ./docs --db-path ./custom.db --qwen-url "http://custom-host:8000"
```

---

**Happy Querying! 🚀**
