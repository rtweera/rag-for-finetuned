from app.rag import RAGSystem
from app.constants import DB_PATH, MODEL
def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="RAG System for Document Q&A")
    parser.add_argument("--directory", required=True, help="Directory containing documents")
    parser.add_argument("--build-index", action="store_true", help="Build the document index")
    parser.add_argument("--query", type=str, help="Query the RAG system")
    parser.add_argument("--qwen-url", default=MODEL, help="Qwen API base URL")
    parser.add_argument("--db-path", default=DB_PATH, help="Database path")
    
    args = parser.parse_args()
    
    # Initialize RAG system
    rag = RAGSystem(args.directory, args.db_path, args.qwen_url)
    
    if args.build_index:
        rag.build_index()
        stats = rag.get_stats()
        print(f"Index built successfully!")
        print(f"Documents: {stats['total_documents']}")
        print(f"Chunks: {stats['total_chunks']}")
    
    if args.query:
        response = rag.query(args.query)
        print(f"\nQuestion: {args.query}")
        print(f"Answer: {response}")
    
    # Interactive mode
    if not args.build_index and not args.query:
        print("Interactive RAG System")
        print("Type 'quit' to exit")
        
        while True:
            question = input("\nEnter your question: ").strip()
            if question.lower() in ['quit', 'exit']:
                break
            
            if question:
                response = rag.query(question)
                print(f"Answer: {response}")

if __name__ == "__main__":
    main()
