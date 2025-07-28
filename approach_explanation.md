# Document Analyst: Approach Explanation

## Overview

Our document analysis system employs a retrieve-and-rerank architecture to efficiently extract and prioritize relevant sections from PDFs based on the persona and job-to-be-done. This approach optimizes both accuracy and performance while staying within the constraints of CPU-only execution, offline operation, and a 1GB model size limit.

## Architecture Components

### Retrieval Model: All-MiniLM-L6-v2

We selected the lightweight yet powerful all-MiniLM-L6-v2 model (sentence-transformers) for initial document section retrieval. This model:
- Has only 80MB size (INT8-quantized ONNX version)
- Generates 384-dimensional embeddings
- Offers an excellent balance between accuracy and inference speed

### Re-ranking Model: MS-Marco-MiniLM-L6-v2

For precision in ranking relevant sections, we employ the cross-encoder/ms-marco-MiniLM-L6-v2 model that:
- Weighs ~85MB after INT8 quantization
- Specializes in passage ranking based on relevance
- Dramatically improves the quality of top-K retrieved results

### Vector Index: FAISS IndexIVFPQ

FAISS IndexIVFPQ provides efficient approximate nearest neighbor search that:
- Scales effectively with growing document collections
- Offers logarithmic-time searches, vital for meeting the 60-second processing time constraint
- Requires minimal memory footprint through quantization

### Inference Engine: ONNX Runtime

We use the ONNX Runtime with CPU Execution Provider to:
- Accelerate inference without GPU requirements
- Enable vectorized operations optimized for modern CPUs
- Reduce memory usage through graph optimization

## Workflow

1. **Document Parsing**: PyMuPDF extracts text and maintains structural information from PDFs, preserving section hierarchy and page references.

2. **Section Extraction**: Rule-based heuristics identify section boundaries and hierarchies, preserving the document's logical structure.

3. **Query Generation**: The system converts the persona and job-to-be-done into an effective query for retrieval.

4. **Two-Stage Retrieval**:
   - First stage: Generate embeddings for all sections and perform efficient semantic search using FAISS
   - Second stage: Re-rank the top candidates using the cross-encoder model to prioritize the most relevant sections

5. **Result Consolidation**: The system merges analysis from multiple documents, ensuring the most relevant sections appear first while maintaining diversity of information.

## Optimizations

Several optimizations ensure the system meets the 60-second time constraint:

1. **Model Quantization**: INT8 quantization reduces model size by ~75% and improves inference speed
2. **Batched Processing**: Document sections are processed in batches to maximize CPU utilization
3. **Prioritized Analysis**: The system focuses detailed analysis on the most promising sections first
4. **Efficient Text Processing**: Text normalization and chunking strategies balance information retention with processing speed

## Container Implementation

The Docker container performs offline batch processing by:
1. Iterating through all PDFs in the input directory
2. Processing each document individually with tailored analysis
3. Producing per-document JSON outputs and a consolidated output.json
4. Enforcing network isolation to ensure offline execution

This retrieve-and-rerank approach provides high-quality document analysis within tight constraints by leveraging efficient models and optimized algorithms that scale effectively with document collection size. 