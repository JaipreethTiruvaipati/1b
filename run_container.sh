#!/bin/bash
# Container entry point script for document analysis

echo "Starting document analysis..."

# Create symbolic links between input and PDFs directory for compatibility
echo "Setting up directory structure..."
mkdir -p /app/PDFs
ln -sf /app/input/* /app/PDFs/ 2>/dev/null || true

# Check if challenge1b_input.json already exists
if [ -f "/app/challenge1b_input.json" ]; then
    echo "Using existing challenge1b_input.json file in /app directory"
    input_file="/app/challenge1b_input.json"
elif [ -f "/app/input/challenge1b_input.json" ]; then
    echo "Using existing challenge1b_input.json file from input directory"
    cp /app/input/challenge1b_input.json /app/
    input_file="/app/challenge1b_input.json"
else
    # If no input file exists, generate one from PDFs as fallback
    echo "No challenge1b_input.json found, generating one from available PDFs..."
    
    # Find all PDFs in the input directory
    pdf_files=$(find /app/input -name "*.pdf" -type f)
    pdf_count=$(echo "$pdf_files" | wc -l)
    
    if [ -z "$pdf_files" ]; then
        echo "Error: No PDF files found in /app/input directory and no challenge1b_input.json provided."
        exit 1
    fi
    
    echo "Found $pdf_count PDF files to process."
    
    # Create a challenge1b_input.json file with all PDFs
    echo "Generating challenge1b_input.json with all PDFs..."
    
    # Start JSON structure
    cat > /app/challenge1b_input.json << EOF
{
    "challenge_info": {
        "challenge_id": "round_1b",
        "test_case_name": "document_analysis",
        "description": "Document Analysis Challenge"
    },
    "documents": [
EOF
    
    # Add each PDF file to the documents array
    first=true
    for pdf_file in $pdf_files; do
        filename=$(basename "$pdf_file")
        name_without_ext="${filename%.pdf}"
        
        # Add comma for all but the first item
        if [ "$first" = true ]; then
            first=false
        else
            echo "," >> /app/challenge1b_input.json
        fi
        
        # Add document entry
        cat >> /app/challenge1b_input.json << EOF
        {
            "filename": "$filename",
            "title": "$name_without_ext"
        }
EOF
    done
    
    # Complete the JSON structure
    cat >> /app/challenge1b_input.json << EOF
    ],
    "persona": {
        "role": "Document Analyst"
    },
    "job_to_be_done": {
        "task": "Extract and analyze key information from these documents."
    }
}
EOF
    
    echo "Generated challenge1b_input.json"
    input_file="/app/challenge1b_input.json"
fi

# Process using the challenge1b_input.json
echo "Analyzing documents..."
python -m src.main --input "$input_file" --output /app/output/challenge1b_output.json --data-dir /app/PDFs

echo "Document analysis completed!" 