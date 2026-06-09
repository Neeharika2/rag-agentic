from abc import ABC, abstractmethod
from typing import List, Dict, Any, Union
from pathlib import Path

class PDFParserInterface(ABC):
    """
    Interface for parsing PDF documents.
    Adheres to the Single Responsibility Principle (SRP) for document parsing
    and Dependency Inversion Principle (DIP) for pipeline decoupling.
    """

    @abstractmethod
    def parse(self, file_path: Union[str, Path]) -> List[Dict[str, Any]]:
        """
        Parses a PDF file and returns a list of raw structured elements.
        Each element is represented as a dictionary containing its type 
        (e.g., 'table', 'text'), content, and layout position or headers.
        """
        pass
