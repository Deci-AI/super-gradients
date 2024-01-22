#!/bin/bash

# Script to run and convert a single notebook
NOTEBOOK_PATH="$1"
ENV_NAME="$2"  # Second argument is the environment name
OUTPUT_DIR="documentation/source/"

# Ensure the virtual environment is activated
source "${ENV_NAME}/bin/activate"  # Updated to use the variable

# Convert the notebook
jupyter nbconvert --to markdown --execute --output-dir="$OUTPUT_DIR" "$NOTEBOOK_PATH"

# Check for errors
if [ $? -ne 0 ]; then
    echo "Error processing $NOTEBOOK_PATH"
    exit 1
fi

echo "$NOTEBOOK_PATH processed successfully."
