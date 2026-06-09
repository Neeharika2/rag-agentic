import os
from src.parsing.docling_parser import DoclingParser
from src.serialization.row_serializer import RowSerializer
from src.vectorstore.chroma_store import ChromaStore
from src.pipeline import IngestionPipeline

def main():
    # Define paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    pdf_path = os.path.join(base_dir, "uploads", "IPL_LangGraph_RAG_Dataset.pdf")
    persist_dir = os.path.join(base_dir, "chroma_db")
    
    # 1. Instantiate the concrete dependencies (SOLID - DIP Dependency Injection)
    parser = DoclingParser()
    serializer = RowSerializer()
    vector_store = ChromaStore(persist_dir=persist_dir)
    
    # 2. Inject dependencies into the orchestrator pipeline
    pipeline = IngestionPipeline(
        parser=parser,
        serializer=serializer,
        vector_store=vector_store
    )
    
    # 3. Execute the ingestion pipeline
    result = pipeline.run(pdf_path)
    print("\n[*] Verification Ingestion Statistics:")
    print(f"    - Parsed raw layout items: {result['parsed_elements_count']}")
    print(f"    - Created indexable chunks: {result['chunks_count']}")
    
    # 4. Perform a simple verification search query
    print("\n[*] Performing verification query on vector database...")
    query = "Virat Kohli career runs"
    search_results = vector_store.search(query, limit=2)
    
    print(f"\nQuery: '{query}'")
    print(f"Found {len(search_results)} matching documents in vector store:")
    for idx, doc in enumerate(search_results):
        print(f"[{idx+1}] ID: {doc['id']}")
        print(f"    Text: {doc['text']}")
        print(f"    Metadata: {doc['metadata']}")

if __name__ == "__main__":
    main()
