import os
import fitz  # PyMuPDF
import re
from typing import List, Dict, Any, Tuple
import numpy as np
from dataclasses import dataclass
from pathlib import Path

@dataclass
class Section:
    document: str
    section_title: str
    text: str
    page_number: int
    headings: List[str] = None
    
    def __post_init__(self):
        if self.headings is None:
            self.headings = []

class DocumentProcessor:
    """
    Handles PDF document processing, including:
    - Loading PDF documents
    - Extracting text and metadata
    - Identifying sections and subsections
    - Preprocessing text for embedding
    """
    
    def __init__(self, data_dir: str):
        """
        Initialize the DocumentProcessor.
        
        Args:
            data_dir: Directory containing PDF documents
        """
        self.data_dir = Path(data_dir)
        
        # Also check PDFs directory if data_dir is not already PDFs
        self.alt_dir = Path("PDFs") if self.data_dir.name != "PDFs" else Path("input")
        
        self.heading_patterns = [
            r'^(?:Section|Chapter|Part)\s+\d+[.:]\s*(.+)$',  # Section 1: Title
            r'^(?:Section|Chapter|Part)\s+\d+\s+(.+)$',       # Section 1 Title
            r'^(?:\d+\.)+\s+(.+)$',                          # 1.2.3 Title
            r'^(?:[A-Z]\.)+\s+(.+)$',                         # A.B. Title
            r'^[IVX]+\.\s+(.+)$',                            # IV. Title
            r'^(?:[A-Z][A-Za-z\s]+:)$',                      # Heading:
            r'^(?:\d+\s+[A-Z][A-Za-z\s]+)$',                 # 1 Title
            r'^(?:[A-Z][A-Za-z\s]+)$'                        # TITLE
        ]

    def load_documents(self, doc_files: List[str]) -> Dict[str, Any]:
        """
        Load multiple PDF documents.
        
        Args:
            doc_files: List of PDF filenames to load
            
        Returns:
            Dictionary mapping filenames to document content
        """
        docs = {}
        for doc_file in doc_files:
            try:
                # Try primary directory first
                primary_path = self.data_dir / doc_file
                alt_path = self.alt_dir / doc_file
                
                if primary_path.exists():
                    docs[doc_file] = self.load_document(str(primary_path))
                    print(f"Loaded {doc_file} from {self.data_dir}")
                elif alt_path.exists():
                    docs[doc_file] = self.load_document(str(alt_path))
                    print(f"Loaded {doc_file} from {self.alt_dir}")
                else:
                    print(f"Warning: Document {doc_file} not found in either {self.data_dir} or {self.alt_dir}")
            except Exception as e:
                print(f"Error loading document {doc_file}: {e}")
        
        return docs
    
    def load_document(self, file_path: str) -> fitz.Document:
        """
        Load a PDF document using PyMuPDF.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            PyMuPDF Document object
        """
        return fitz.open(file_path)
    
    def extract_sections(self, doc_file: str, document: fitz.Document) -> List[Section]:
        """
        Extract sections from a document.
        
        Args:
            doc_file: Document filename
            document: PyMuPDF Document object
            
        Returns:
            List of Section objects
        """
        sections = []
        current_section = None
        current_text = []
        
        for page_num, page in enumerate(document):
            text = page.get_text()
            lines = text.split('\n')
            
            for line in lines:
                line = line.strip()
                
                # Check if this line is a heading
                if self._is_heading(line):
                    # Save the current section if it exists
                    if current_section is not None and current_text:
                        current_section.text = '\n'.join(current_text)
                        sections.append(current_section)
                    
                    # Start a new section
                    current_section = Section(
                        document=doc_file,
                        section_title=line,
                        text='',
                        page_number=page_num + 1  # 1-indexed page numbers
                    )
                    current_text = []
                elif current_section is not None:
                    # Add the line to the current section
                    current_text.append(line)
                else:
                    # If no section has been started yet, use a default section
                    if not sections:
                        current_section = Section(
                            document=doc_file,
                            section_title=f"Introduction",
                            text='',
                            page_number=page_num + 1
                        )
                        current_text = [line]
            
            # End of page, if we're in a section, add a newline
            if current_section is not None and current_text:
                current_text.append('\n')
        
        # Don't forget the last section
        if current_section is not None and current_text:
            current_section.text = '\n'.join(current_text)
            sections.append(current_section)
        
        return sections

    def _is_heading(self, text: str) -> bool:
        """
        Determine if a line of text is a section heading.
        
        Args:
            text: Line of text to check
            
        Returns:
            True if the line is a heading, False otherwise
        """
        if not text or len(text) < 3:
            return False
        
        # Check if the text matches any of the heading patterns
        for pattern in self.heading_patterns:
            if re.match(pattern, text):
                return True
        
        # Check for characteristics of a heading
        if text.isupper() and len(text) < 50:
            return True
        
        if text.endswith(':') and len(text) < 50:
            return True
            
        return False
    
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess text for embedding.
        
        Args:
            text: Text to preprocess
            
        Returns:
            Preprocessed text
        """
        # Replace multiple whitespace with a single space
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters
        text = re.sub(r'[^\w\s]', '', text)
        
        # Convert to lowercase
        text = text.lower()
        
        return text.strip()
    
    def extract_all_sections(self, doc_files: List[str]) -> List[Section]:
        """
        Extract sections from multiple documents.
        
        Args:
            doc_files: List of document filenames
            
        Returns:
            List of sections from all documents
        """
        all_sections = []
        docs = self.load_documents(doc_files)
        
        for doc_file, document in docs.items():
            sections = self.extract_sections(doc_file, document)
            all_sections.extend(sections)
        
        return all_sections 