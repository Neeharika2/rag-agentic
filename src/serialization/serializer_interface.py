from abc import ABC, abstractmethod
from typing import List, Dict, Any

class SerializerInterface(ABC):
    """
    Interface for serializing extracted PDF elements into database chunks.
    Adheres to the Single Responsibility Principle (SRP) for chunk formatting.
    """

    @abstractmethod
    def serialize(self, parsed_elements: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Processes parsed document elements and returns a list of chunks,
        where each chunk is a dictionary with "text" (serialized string content)
        and "metadata" (associated fields like team, player, year, etc.).
        """
        pass
