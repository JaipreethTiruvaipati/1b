#!/bin/bash

# Run the document analysis system with sample input

# Set environment
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Check if Python is installed
if ! command -v python3 &> /dev/null
then
    echo "Error: Python 3 is not installed"
    exit 1
fi

# Create a virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install requirements if they're not already installed
echo "Installing requirements..."
pip install -r requirements.txt

# Run the document analysis system
echo "Running document analysis system..."
python3 -m src.main --input data/input.json --output output/result.json

# Check if the analysis was successful
if [ $? -eq 0 ]; then
    echo "Analysis completed successfully!"
    echo "Output saved to output/result.json"
else
    echo "Error: Analysis failed"
    exit 1
fi

# Optionally run the performance profiler
if [ "$1" = "--profile" ]; then
    echo "Running performance profiler..."
    python3 -m src.profiler --full-profile
fi

# Deactivate virtual environment
deactivate

echo "Done!" 