import os
import hashlib
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.utils import embedding_functions
from src.vectorstore.store_interface import VectorStoreInterface

class ChromaStore(VectorStoreInterface):
    """
    Concrete implementation of VectorStoreInterface using ChromaDB.
    Handles embedding generation, document storage, indexing, and metadata-filtered search.
    Idempotent document ingestion is achieved using MD5 hash IDs.
    """

    def __init__(
        self, 
        persist_dir: str = "./chroma_db", 
        collection_name: str = "ipl_assistant",
        embedding_function: Optional[Any] = None
    ):
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        
        # Initialize persistent client
        self.client = chromadb.PersistentClient(path=self.persist_dir)
        
        # Use default Chroma embedding function (all-MiniLM-L6-v2) if not specified
        if embedding_function is None:
            self.embedding_function = embedding_functions.DefaultEmbeddingFunction()
        else:
            self.embedding_function = embedding_function
            
        # Create or fetch existing collection
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self.embedding_function
        )

    def _generate_id(self, text: str, metadata: Dict[str, Any]) -> str:
        """Generates a stable, idempotent ID based on md5 hash of content and section metadata."""
        section = metadata.get("section", "general")
        hash_input = f"{section}:{text}".encode("utf-8")
        return hashlib.md5(hash_input).hexdigest()

    def add_documents(self, chunks: List[Dict[str, Any]]) -> None:
        if not chunks:
            return
            
        documents = []
        metadatas = []
        ids = []
        
        for chunk in chunks:
            text = chunk["text"]
            meta = chunk["metadata"]
            
            # Clean metadata: ChromaDB metadata values must be simple types (str, int, float, bool)
            cleaned_meta = {}
            for k, v in meta.items():
                if isinstance(v, list):
                    # ChromaDB metadata does not support lists directly.
                    # We store it as a comma-separated string for compatibility.
                    cleaned_meta[k] = ",".join(str(item) for item in v)
                elif isinstance(v, (str, int, float, bool)):
                    cleaned_meta[k] = v
                else:
                    cleaned_meta[k] = str(v)
            
            doc_id = self._generate_id(text, cleaned_meta)
            
            documents.append(text)
            metadatas.append(cleaned_meta)
            ids.append(doc_id)
            
        # Ingest documents in batches to avoid network limits if scale changes
        batch_size = 100
        for i in range(0, len(documents), batch_size):
            self.collection.upsert(
                documents=documents[i:i+batch_size],
                metadatas=metadatas[i:i+batch_size],
                ids=ids[i:i+batch_size]
            )

    def search(self, query: str, limit: int = 5, filter_dict: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        # Map flat filter_dict to ChromaDB's where format
        where_filter = None
        if filter_dict:
            cleaned_filter = {}
            for k, v in filter_dict.items():
                if isinstance(v, list):
                    # For list containment (e.g. searching a list of teams)
                    # We query if string column contains the search value
                    # In Chroma, this can be mapped using "$contains" if supported, or handled in post-processing.
                    cleaned_filter[k] = ",".join(str(item) for item in v)
                else:
                    cleaned_filter[k] = v
            
            if len(cleaned_filter) == 1:
                key, val = list(cleaned_filter.items())[0]
                where_filter = {key: val}
            elif len(cleaned_filter) > 1:
                where_filter = {"$and": [{k: v} for k, v in cleaned_filter.items()]}

        results = self.collection.query(
            query_texts=[query],
            n_results=limit,
            where=where_filter
        )
        
        # Format the query output
        docs = []
        if results and "documents" in results and results["documents"]:
            raw_docs = results["documents"][0]
            raw_metas = results["metadatas"][0]
            raw_ids = results["ids"][0]
            
            for i in range(len(raw_docs)):
                docs.append({
                    "id": raw_ids[i],
                    "text": raw_docs[i],
                    "metadata": raw_metas[i]
                })
                
        return docs
