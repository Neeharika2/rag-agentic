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
        """
        Initializes the default DocumentConverter from Docling.
        On first invocation, this converter will download local OCR and layout model weights
        from Hugging Face (such as layout detector and table structure models).
        """
        self.converter = DocumentConverter()

    def parse(self, file_path: Union[str, Path]) -> List[Dict[str, Any]]:
        """
        Parses the target PDF document using Docling.
        Preserves reading flow order, sections, and structured tables.

        Parameters:
        - file_path (Union[str, Path]): Path to the PDF document to be parsed.

        Returns:
        - List[Dict[str, Any]]: A list of dictionaries representing document items.
                                Each item contains:
                                - "type": 'text', 'table', or 'table_raw'
                                - "section": The nearest heading section string
                                - "text" or "headers"/"data" fields depending on the item type.
        """
        file_path = str(file_path)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"PDF file not found at: {file_path}")

        # Convert the document using Docling converter
        conversion_result = self.converter.convert(file_path)
        doc = conversion_result.document
        
        elements = []
        # Keep track of current section/heading to tag downstream text blocks contextually
        current_section = "general"
        
        # Iterate through layout items in reading order
        for element, _ in doc.iterate_items():
            element_type = type(element).__name__
            
            # Detect section headings to contextualize downstream paragraphs/tables
            if element_type in ["SectionHeaderItem", "HeadingItem"] or (element_type == "TextItem" and getattr(element, "label", "") == "heading"):
                current_section = element.text.strip()
                continue
                
            # Handle table element blocks
            if element_type == "TableItem":
                try:
                    # Docling provides export_to_dataframe() for structured tables
                    # We convert this dataframe into row-wise dictionaries
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
            # Handle narrative paragraphs
            elif element_type == "TextItem":
                text_content = element.text.strip() if hasattr(element, 'text') else ""
                if text_content:
                    elements.append({
                        "type": "text",
                        "section": current_section,
                        "text": text_content
                    })
                
        return elements
