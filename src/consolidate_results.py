#!/usr/bin/env python3
"""
Script to consolidate individual document analysis results into a single output.json file.
"""

import os
import json
import argparse
from datetime import datetime
from pathlib import Path


def consolidate_results(output_dir):
    """
    Consolidate individual JSON result files into a single output.json file.
    
    Args:
        output_dir: Directory containing individual JSON result files
    """
    output_dir = Path(output_dir)
    
    # Get all JSON files in the output directory
    json_files = list(output_dir.glob("*.json"))
    
    # Skip output.json if it already exists in the list
    json_files = [f for f in json_files if f.name != "output.json"]
    
    if not json_files:
        print("No JSON result files found in output directory")
        return
    
    print(f"Found {len(json_files)} JSON result files")
    
    # Initialize consolidated data with exactly the expected structure
    consolidated = {
        "metadata": {
            "input_documents": [],
            "persona": "Document Analyst",
            "job_to_be_done": "Extract and analyze key information from documents.",
            "processing_timestamp": datetime.now().isoformat()
        },
        "extracted_sections": [],
        "subsection_analysis": []
    }
    
    # Process each JSON file
    for json_file in json_files:
        try:
            with open(json_file, "r") as f:
                data = json.load(f)
            
            # Extract document name from filename
            doc_name = json_file.stem
            
            # Add to input documents list if not already present
            doc_list = data.get("metadata", {}).get("input_documents", [])
            for doc in doc_list:
                if doc not in consolidated["metadata"]["input_documents"]:
                    consolidated["metadata"]["input_documents"].append(doc)
            
            # Add extracted sections
            sections = data.get("extracted_sections", [])
            for section in sections:
                # Add document name context if needed
                if doc_name not in section.get("document", ""):
                    section["document"] = f"{doc_name}: {section['document']}"
                consolidated["extracted_sections"].append(section)
            
            # Add subsection analysis
            subsections = data.get("subsection_analysis", [])
            for subsection in subsections:
                # Add document name context if needed
                if doc_name not in subsection.get("document", ""):
                    subsection["document"] = f"{doc_name}: {subsection['document']}"
                consolidated["subsection_analysis"].append(subsection)
            
            print(f"Processed {json_file.name}")
            
        except Exception as e:
            print(f"Error processing {json_file.name}: {e}")
    
    # Sort extracted sections by importance rank
    consolidated["extracted_sections"] = sorted(
        consolidated["extracted_sections"], 
        key=lambda x: x.get("importance_rank", 999)
    )
    
    # Cap at maximum 10 sections
    consolidated["extracted_sections"] = consolidated["extracted_sections"][:10]
    
    # Cap at maximum 10 subsections
    consolidated["subsection_analysis"] = consolidated["subsection_analysis"][:10]
    
    # Write consolidated output
    output_file = output_dir / "output.json"
    with open(output_file, "w") as f:
        json.dump(consolidated, f, indent=4)
    
    print(f"Consolidated results written to {output_file}")
    
    return consolidated


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Consolidate individual document analysis results"
    )
    parser.add_argument(
        "--output-dir", "-o",
        default="output",
        help="Directory containing individual JSON result files (default: output)"
    )
    
    args = parser.parse_args()
    
    consolidate_results(args.output_dir) 