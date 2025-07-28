#!/bin/bash

# Script for building and running the document analysis Docker container

# Set environment
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Build the Docker image
echo "Building Docker image..."
docker build --platform linux/amd64 -t document-analyst:latest .

# Check if build was successful
if [ $? -ne 0 ]; then
    echo "Error: Docker build failed"
    exit 1
fi

echo "Docker image built successfully"

# Create input and output directories if they don't exist
mkdir -p input output

echo "Please place in the 'input' directory:"
echo "  - Either a challenge1b_input.json file with document specifications"
echo "  - Or your PDF files directly (a challenge1b_input.json will be generated)"
echo ""
echo "The analysis output will be saved to 'output/challenge1b_output.json'."

# Run the Docker container
echo "Running Docker container..."
docker run --rm -v "$(pwd)/input:/app/input" -v "$(pwd)/output:/app/output" --network none document-analyst:latest

echo "Docker container execution completed" 