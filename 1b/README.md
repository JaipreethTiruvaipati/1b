# Document Analyst: Intelligent PDF Analysis System

This project implements an intelligent document analyst system that extracts and prioritizes relevant sections from a collection of documents based on a specific persona and their job-to-be-done.

## Overview

The system accepts:
1. A collection of 3-10 related PDF documents
2. A persona definition (role with specific expertise and focus areas)
3. A job-to-be-done (concrete task the persona needs to accomplish)

The system analyzes the documents to extract and rank sections based on their relevance to the persona and task, returning results in a structured JSON format.

## File Structure

The system follows this directory structure:
- `PDFs/` - Contains all PDF documents for analysis
- `challenge1b_input.json` - Configuration file defining documents, persona, and job-to-be-done
- `challenge1b_output.json` - Generated output from the analysis

## Input Format

The `challenge1b_input.json` file specifies which PDF files to analyze, along with the persona and job-to-be-done:

```json
{
  "challenge_info": {
    "challenge_id": "round_1b",
    "test_case_name": "document_analysis",
    "description": "Document Analysis Challenge"
  },
  "documents": [
    {
      "filename": "document1.pdf",
      "title": "Document 1 Title"
    },
    {
      "filename": "document2.pdf",
      "title": "Document 2 Title"
    }
  ],
  "persona": {
    "role": "Document Analyst"
  },
  "job_to_be_done": {
    "task": "Extract and analyze key information from these documents."
  }
}
```

## Architecture

The system uses a retrieve-and-rerank approach with the following components:

- **Retrieval Model**: INT8-quantized ONNX version of sentence-transformers/all-MiniLM-L6-v2
- **Re-ranking Model**: INT8-quantized ONNX version of cross-encoder/ms-marco-MiniLM-L6-v2
- **Vector Index**: FAISS IndexIVFPQ for fast approximate nearest neighbor search
- **Inference Engine**: ONNX Runtime with CPU Execution Provider

## Docker Deployment

The simplest way to run the system is using Docker:

```bash
# Build and run the Docker container
./docker-run.sh
```

Or manually:

```bash
# Build the Docker image
docker build --platform linux/amd64 -t document-analyst:latest .

# Run the container
docker run --rm -v $(pwd)/input:/app/input -v $(pwd)/output:/app/output --network none document-analyst:latest
```

Place either:
- Your `challenge1b_input.json` file and referenced PDF files in the `input` directory, or
- Just your PDF files in the `input` directory (a default `challenge1b_input.json` will be generated)

The analysis results will be written to `output/challenge1b_output.json`.

## Manual Installation

If you prefer to run the system directly without Docker:

```bash
# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the analysis script
./run_demo.sh
```

## Usage

For direct Python usage:

```bash
python -m src.main --input challenge1b_input.json --output challenge1b_output.json --data-dir PDFs
```

The input JSON should contain the challenge info, documents, persona, and job-to-be-done as described in the sample input format.

## Output

The system generates a JSON output containing:
- Metadata (input documents, persona, job-to-be-done, timestamp)
- Extracted sections with rankings
- Sub-section analysis with refined text

## Performance

- Processing time: < 60 seconds for 3-5 documents
- CPU-only operation
- Model size: < 1GB 