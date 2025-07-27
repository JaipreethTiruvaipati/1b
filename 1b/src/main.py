import argparse
import json
import os
import sys
from pathlib import Path

from src.document_analyzer import DocumentAnalyzer

def main():
    """Main entry point for the document analysis application."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Analyze documents based on persona and job-to-be-done"
    )
    parser.add_argument(
        "--input", "-i", 
        default="challenge1b_input.json",
        help="Input JSON file containing the challenge info, documents, persona, and job-to-be-done"
    )
    parser.add_argument(
        "--output", "-o",
        default="challenge1b_output.json",
        help="Output JSON file (default: challenge1b_output.json)"
    )
    parser.add_argument(
        "--data-dir", "-d",
        default="PDFs",
        help="Directory containing PDF documents (default: PDFs)"
    )
    parser.add_argument(
        "--output-dir", "-od",
        default="output",
        help="Directory to write output files (default: output)"
    )
    
    args = parser.parse_args()
    
    # Ensure input file exists
    input_file = Path(args.input)
    if not input_file.exists():
        print(f"Error: Input file '{input_file}' does not exist")
        sys.exit(1)
    
    # Ensure output directory exists
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Read input JSON
    try:
        with open(input_file, "r") as f:
            input_data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in input file: {e}")
        sys.exit(1)
    
    # Validate input data
    if not validate_input(input_data):
        print("Error: Invalid input data format")
        sys.exit(1)
    
    # Initialize document analyzer
    analyzer = DocumentAnalyzer(args.data_dir, args.output_dir)
    
    # Process input
    output = analyzer.analyze(input_data)
    
    # Save output
    analyzer.save_output(output, args.output)
    
    print("Document analysis completed successfully!")

def validate_input(input_data):
    """
    Validate input data structure.
    
    Args:
        input_data: Input data dictionary
        
    Returns:
        True if valid, False otherwise
    """
    # Check for required keys
    required_keys = ["documents", "persona", "job_to_be_done"]
    if not all(key in input_data for key in required_keys):
        print(f"Missing required keys. Required: {required_keys}")
        return False
    
    # Check documents
    documents = input_data.get("documents", [])
    if not isinstance(documents, list) or not documents:
        print("Invalid or empty documents list")
        return False
    
    for doc in documents:
        if not isinstance(doc, dict) or "filename" not in doc:
            print("Document missing required 'filename' field")
            return False
    
    # Check persona
    persona = input_data.get("persona", {})
    if not isinstance(persona, dict) or "role" not in persona:
        print("Invalid persona or missing 'role' field")
        return False
    
    # Check job_to_be_done
    job = input_data.get("job_to_be_done", {})
    if not isinstance(job, dict) or "task" not in job:
        print("Invalid job_to_be_done or missing 'task' field")
        return False
    
    return True

if __name__ == "__main__":
    main() 