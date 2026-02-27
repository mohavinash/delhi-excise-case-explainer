#!/bin/bash
# =============================================================================
# Setup and run the Delhi Excise Policy Judgment Analysis Pipeline
# =============================================================================
# Prerequisites:
#   - Python 3.10+
#   - Gemini API key
# =============================================================================

set -e

echo "=== Delhi Excise Policy Judgment - DocETL Pipeline ==="
echo ""

# Check for API key
if [ -z "$GEMINI_API_KEY" ]; then
    echo "ERROR: GEMINI_API_KEY not set."
    echo "Please set it: export GEMINI_API_KEY=your_key_here"
    echo "Or create a .env file in this directory with: GEMINI_API_KEY=your_key_here"
    exit 1
fi

# Navigate to project root
cd "$(dirname "$0")/.."

# Install docetl if not installed
if ! command -v docetl &> /dev/null; then
    echo "Installing DocETL..."
    pip install docetl
fi

# Prepare input data
echo "Preparing input data..."
python3 docetl-pipeline/prepare_input.py

# Run the pipeline
echo ""
echo "Running DocETL pipeline..."
echo "Model: gemini/gemini-2.0-flash"
echo "Input: 55 sections from 549-page judgment"
echo ""
docetl run docetl-pipeline/pipeline.yaml

echo ""
echo "=== Pipeline Complete ==="
echo "Output: docetl-pipeline/output/judgment_analysis.json"
echo "Intermediates: docetl-pipeline/output/intermediates/"
