#!/bin/bash

# Script to run and convert a single notebook
NOTEBOOK_PATH="$1"
ENV_NAME="$2"  # Second argument is the environment name
OUTPUT_DIR="documentation/source/"

echo "processing $NOTEBOOK_PATH"

# Check if the notebook path is empty
if [ -z "$NOTEBOOK_PATH" ]; then
    echo "No notebook path provided."
    exit 0  # Exit successfully as there's nothing to do
fi

# Check if the notebook file exists
if [ ! -f "$NOTEBOOK_PATH" ]; then
    echo "Notebook path does not exist: $NOTEBOOK_PATH"
    exit 1  # Exit with an error as the notebook path is invalid
fi

# Ensure the virtual environment is activated
source "${ENV_NAME}/bin/activate"

# Convert the notebook
jupyter nbconvert --to markdown --execute --output-dir="$OUTPUT_DIR" "$NOTEBOOK_PATH"

# Check for errors
if [ $? -ne 0 ]; then
    echo "Error processing $NOTEBOOK_PATH"
    exit 1
fi

echo "$NOTEBOOK_PATH processed successfully."
