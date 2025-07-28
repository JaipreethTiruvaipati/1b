import os
import json
import time
from typing import List, Dict, Any
from datetime import datetime
from pathlib import Path

from .document_processor import DocumentProcessor, Section
from .retrieval_system import RetrievalSystem


class DocumentAnalyzer:
    """
    Analyzer class that orchestrates the document analysis workflow.
    """
    
    def __init__(self, data_dir: str, output_dir: str):
        """
        Initialize the document analyzer.
        
        Args:
            data_dir: Directory containing PDF documents
            output_dir: Directory to write output files
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Also look for PDFs in a separate PDFs directory
        self.pdfs_dir = Path("PDFs") if self.data_dir != Path("PDFs") else Path("input")
        
        # Initialize components
        self.doc_processor = DocumentProcessor(data_dir)
        self.retrieval_system = RetrievalSystem()
        
    def analyze(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze documents based on the persona and job-to-be-done.
        
        Args:
            input_data: Input data containing challenge info, documents, persona, and job-to-be-done
            
        Returns:
            Dictionary containing the analysis results
        """
        # Extract inputs
        challenge_info = input_data.get("challenge_info", {})
        documents_info = input_data.get("documents", [])
        persona = input_data.get("persona", {})
        job_to_be_done = input_data.get("job_to_be_done", {})
        
        # Extract document filenames
        document_filenames = [doc.get("filename") for doc in documents_info]
        
        # Process documents
        start_time = time.time()
        all_sections = self.doc_processor.extract_all_sections(document_filenames)
        print(f"Extracted {len(all_sections)} sections from {len(document_filenames)} documents")
        
        # Index sections
        self.retrieval_system.index_sections(all_sections)
        
        # Generate query from persona and job-to-be-done
        query = self._generate_query(persona, job_to_be_done)
        print(f"Generated query: {query}")
        
        # Retrieve relevant sections
        top_sections = self.retrieval_system.search(query, k=20)
        
        # Create output JSON
        output = self._create_output(document_filenames, persona, job_to_be_done, all_sections, top_sections)
        
        # Calculate processing time
        processing_time = time.time() - start_time
        print(f"Processing completed in {processing_time:.2f} seconds")
        
        return output
    
    def _generate_query(self, persona: Dict[str, Any], job_to_be_done: Dict[str, Any]) -> str:
        """
        Generate a query from the persona and job-to-be-done.
        
        Args:
            persona: Persona information
            job_to_be_done: Job-to-be-done information
            
        Returns:
            Query string
        """
        role = persona.get("role", "")
        task = job_to_be_done.get("task", "")
        
        query = f"As a {role}, I need to {task}"
        return query
    
    def _create_output(
        self, 
        document_filenames: List[str],
        persona: Dict[str, Any],
        job_to_be_done: Dict[str, Any],
        all_sections: List[Section],
        top_sections: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Create the output JSON structure.
        
        Args:
            document_filenames: List of document filenames
            persona: Persona information
            job_to_be_done: Job-to-be-done information
            all_sections: List of all sections from all documents
            top_sections: List of top sections based on relevance
            
        Returns:
            Output JSON structure
        """
        # Create metadata
        metadata = {
            "input_documents": document_filenames,
            "persona": persona.get("role", ""),
            "job_to_be_done": job_to_be_done.get("task", ""),
            "processing_timestamp": datetime.now().isoformat()
        }
        
        # Create extracted_sections list
        extracted_sections = []
        for i, section_info in enumerate(top_sections[:5]):
            section_idx = section_info["index"]
            if section_idx < len(all_sections):
                section = all_sections[section_idx]
                extracted_section = {
                    "document": section.document,
                    "section_title": section.section_title,
                    "importance_rank": i + 1,
                    "page_number": section.page_number
                }
                extracted_sections.append(extracted_section)
        
        # Create subsection_analysis list
        subsection_analysis = []
        for section_info in top_sections[:5]:
            section_idx = section_info["index"]
            if section_idx < len(all_sections):
                section = all_sections[section_idx]
                refined_text = self._refine_text(section.text)
                subsection = {
                    "document": section.document,
                    "refined_text": refined_text,
                    "page_number": section.page_number
                }
                subsection_analysis.append(subsection)
        
        # Create output structure
        output = {
            "metadata": metadata,
            "extracted_sections": extracted_sections,
            "subsection_analysis": subsection_analysis
        }
        
        return output
        
    def _refine_text(self, text: str) -> str:
        """
        Refine the text by removing unnecessary whitespace and formatting.
        
        Args:
            text: Text to refine
            
        Returns:
            Refined text
        """
        if not text:
            return ""
        
        # Normalize whitespace
        text = " ".join(text.split())
        
        return text
        
    def save_output(self, output: Dict[str, Any], output_file: str = None):
        """
        Save the output to a JSON file.
        
        Args:
            output: Output data to save
            output_file: Output file path (optional)
        """
        if output_file:
            # Use the specified output file
            output_path = Path(output_file)
        else:
            # Generate filename based on timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = self.output_dir / f"output_{timestamp}.json"
        
        # Ensure parent directory exists
        output_path.parent.mkdir(exist_ok=True, parents=True)
        
        # Write output to file
        with open(output_path, "w") as f:
            json.dump(output, f, indent=4)
        
        print(f"Output saved to {output_path}")
        
    def process_input_file(self, input_file: str, output_file: str = None):
        """
        Process an input file and save the results.
        
        Args:
            input_file: Input JSON file path
            output_file: Output JSON file path (optional)
        """
        # Read input file
        with open(input_file, "r") as f:
            input_data = json.load(f)
        
        # Analyze documents
        output = self.analyze(input_data)
        
        # Save output
        self.save_output(output, output_file) 