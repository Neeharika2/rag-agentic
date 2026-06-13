import os
from src.parsing.docling_parser import DoclingParser
from src.vectorstore.chroma_store import ChromaStore
from src.pipeline import IngestionPipeline

def main():
    """
    Main entry point for building and indexing the vector store.
    Instantiates concrete parsing and storage structures, orchestrates the 
    ingestion pipeline, and performs a simple verification lookup query.
    """
    # 1. Establish filesystem paths for target dataset and local vector store
    base_dir = os.path.dirname(os.path.abspath(__file__))
    pdf_path = os.path.join(base_dir, "uploads", "Placement_RAG_Dataset_Enhanced.pdf")
    persist_dir = os.path.join(base_dir, "chroma_db")
    
    # 2. Instantiate concrete dependencies (satisfying Parser and Vector Store Interfaces)
    parser = DoclingParser()
    vector_store = ChromaStore(persist_dir=persist_dir)
    
    # 3. Inject concrete dependencies into the central orchestration pipeline
    pipeline = IngestionPipeline(
        parser=parser,
        vector_store=vector_store
    )
    
    # 4. Execute parsing, dynamic chunking, and database indexing
    result = pipeline.run(pdf_path)
    print("\n[*] Verification Ingestion Statistics:")
    print(f"    - Parsed raw layout items: {result['parsed_elements_count']}")
    print(f"    - Created indexable chunks: {result['chunks_count']}")
    
    # 5. Run a simple keyword query on the populated database to verify retrieval
    print("\n[*] Performing verification query on vector database...")
    query = "TCS CGPA requirement"
    search_results = vector_store.search(query, limit=2)
    
    print(f"\nQuery: '{query}'")
    print(f"Found {len(search_results)} matching documents in vector store:")
    for idx, doc in enumerate(search_results):
        print(f"[{idx+1}] ID: {doc['id']}")
        print(f"    Text: {doc['text']}")
        print(f"    Metadata: {doc['metadata']}")

if __name__ == "__main__":
    main()
