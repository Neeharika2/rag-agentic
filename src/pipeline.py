from typing import Dict, Any, List
from pathlib import Path
from src.parsing.parser_interface import PDFParserInterface
from src.serialization.serializer_interface import SerializerInterface
from src.vectorstore.store_interface import VectorStoreInterface

class IngestionPipeline:
    """
    Orchestration pipeline for document ingestion.
    Fulfills Dependency Inversion Principle (DIP) by receiving interfaces
    in its constructor rather than importing concrete parsers, serializers, or stores.
    """

    def __init__(
        self,
        parser: PDFParserInterface,
        serializer: SerializerInterface,
        vector_store: VectorStoreInterface
    ):
        self.parser = parser
        self.serializer = serializer
        self.vector_store = vector_store

    def run(self, pdf_path: str) -> Dict[str, Any]:
        """
        Executes the three stages of document ingestion:
        1. Parsing: Extract structured layout paragraphs and tables.
        2. Serialization: Convert parsed elements into metadata-enriched chunks.
        3. Indexing: Load chunks into the vector store.
        """
        print(f"[*] Starting Ingestion Pipeline for file: {pdf_path}")
        
        # Stage 1: Parsing
        print("[1/3] Parsing PDF layout with Docling...")
        parsed_elements = self.parser.parse(pdf_path)
        print(f"    - Extracted {len(parsed_elements)} raw elements (paragraphs/tables)")
        
        # Stage 2: Chunk Ingestion / Serialization
        print("[2/3] Serializing rows and extracting metadata...")
        chunks = self.serializer.serialize(parsed_elements)
        print(f"    - Generated {len(chunks)} chunks with targeted metadata schema")
        
        # Stage 3: Loading into Vector Store
        print("[3/3] Loading documents into Vector Store...")
        self.vector_store.add_documents(chunks)
        print(f"[+] Ingestion completed successfully! Total chunks indexed: {len(chunks)}")
        
        return {
            "status": "success",
            "parsed_elements_count": len(parsed_elements),
            "chunks_count": len(chunks)
        }
