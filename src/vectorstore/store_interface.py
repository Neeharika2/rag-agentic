from abc import ABC, abstractmethod
from typing import List, Dict, Any

class VectorStoreInterface(ABC):
    """
    Interface for vector store databases.
    Adheres to the Single Responsibility Principle (SRP) for storage operations.
    """

    @abstractmethod
    def add_documents(self, chunks: List[Dict[str, Any]]) -> None:
        """
        Ingests a list of serialized chunks with their metadata into the vector store.
        Each chunk is a dictionary containing "text" and "metadata".
        """
        pass

    @abstractmethod
    def search(self, query: str, limit: int = 5, filter_dict: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Searches the vector store using similarity search and optional metadata filtering.
        """
        pass
