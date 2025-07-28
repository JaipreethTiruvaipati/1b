# Document Analyst: Execution Instructions

This document provides instructions for building and running the Document Analyst solution.

## System Requirements

- Docker installed (recent version)
- At least 4GB of RAM
- x86_64/AMD64 architecture

## Build Instructions

Build the Docker image with:

```bash
docker build --platform linux/amd64 -t document-analyst:latest .
```

## Run Instructions

1. Create directories for input and output (if they don't exist):

```bash
mkdir -p input output
```

2. Place PDF files for analysis in the `input` directory.

3. Run the container with:

```bash
docker run --rm -v $(pwd)/input:/app/input -v $(pwd)/output:/app/output --network none document-analyst:latest
```

## Output

The system produces:

1. Individual JSON files in the `output` directory, one for each PDF file processed
2. A consolidated `output.json` file containing the combined analysis results

## Expected Output Format

The `output.json` file contains:

1. **Metadata**:
   - Input documents list
   - Persona information
   - Job-to-be-done description
   - Processing timestamp

2. **Extracted Sections**:
   - Document source
   - Page number
   - Section title
   - Importance ranking

3. **Sub-section Analysis**:
   - Document source
   - Refined text content
   - Page number

## Troubleshooting

- Ensure PDF files are properly formatted and readable
- Check disk space and permissions
- Verify Docker is running with sufficient resources 