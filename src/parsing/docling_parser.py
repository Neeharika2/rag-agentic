import os
from pathlib import Path
from typing import List, Dict, Any, Union
from docling.document_converter import DocumentConverter
from src.parsing.parser_interface import PDFParserInterface

class DoclingParser(PDFParserInterface):
    """
    Concrete implementation of PDFParserInterface using IBM's Docling.
    Responsible solely for converting a PDF into structured layout items
    (text paragraphs and tabular rows), maintaining the document's flow and headers.
    """
    
    def __init__(self):
        # Initialize the default document converter from Docling
        self.converter = DocumentConverter()

    def parse(self, file_path: Union[str, Path]) -> List[Dict[str, Any]]:
        file_path = str(file_path)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"PDF file not found at: {file_path}")

        # Convert the document using Docling
        conversion_result = self.converter.convert(file_path)
        doc = conversion_result.document
        
        elements = []
        current_section = "general"
        
        # Iterate through layout items in reading order
        for element, _ in doc.iterate_items():
            element_type = type(element).__name__
            
            # Detect section headings to contextualize downstream paragraphs/tables
            if element_type in ["SectionHeaderItem", "HeadingItem"] or (element_type == "TextItem" and getattr(element, "label", "") == "heading"):
                current_section = element.text.strip()
                continue
                
            if element_type == "TableItem":
                try:
                    # Docling provides export_to_dataframe() for structured tables
                    df = element.export_to_dataframe()
                    headers = [str(col).strip() for col in df.columns]
                    rows = []
                    for _, row in df.iterrows():
                        row_dict = {headers[i]: str(val).strip() for i, val in enumerate(row)}
                        rows.append(row_dict)
                    
                    elements.append({
                        "type": "table",
                        "section": current_section,
                        "headers": headers,
                        "data": rows
                    })
                except Exception as e:
                    # Fallback to plain text table representation if dataframe conversion fails
                    elements.append({
                        "type": "table_raw",
                        "section": current_section,
                        "text": element.text.strip() if hasattr(element, 'text') else ""
                    })
            elif element_type == "TextItem":
                text_content = element.text.strip() if hasattr(element, 'text') else ""
                if text_content:
                    elements.append({
                        "type": "text",
                        "section": current_section,
                        "text": text_content
                    })
                
        return elements
