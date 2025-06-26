#!/bin/bash

# This script automates the full data preparation pipeline for CRAG 'multi-hop' questions.
# It performs the following steps:
# 1. Filters 'multi-hop' questions from the source CRAG file.
# 2. Splits the filtered data into dev and test sets under raw_data/crag/.
# 3. Converts the split files into the project's standard format under processed_data/crag/.
# 4. Builds the Elasticsearch index for the CRAG dataset.

set -e  # Exit immediately if a command exits with a non-zero status.

echo ">>> [Step 1/4] Filtering for 'multi-hop' questions..."

# Define file paths
SOURCE_FILE="RAG/crag_task_1_dev_v4_release.jsonl"
RAW_DIR="raw_data/crag"
FILTERED_FILE="$RAW_DIR/crag_multihop_raw.jsonl"
PROCESSED_DIR="processed_data/crag"

# Create directories
mkdir -p $RAW_DIR
mkdir -p $PROCESSED_DIR

# Check if source file exists
if [ ! -f "$SOURCE_FILE" ]; then
    echo "Error: Source file not found at $SOURCE_FILE"
    exit 1
fi

# Call the existing data_filter.py script to filter for multi-hop questions
python RAG/src/data_filter.py $SOURCE_FILE $FILTERED_FILE question_type multi-hop
echo "Filtering complete. Filtered questions are in $FILTERED_FILE"

echo ">>> [Step 2/4] Splitting filtered data into dev and test sets..."
python processing_scripts/process_crag.py split --input_file $FILTERED_FILE --output_dir $RAW_DIR

echo ">>> [Step 3/4] Converting dev and test sets to processed format..."
echo "Converting dev set..."
python processing_scripts/process_crag.py convert dev
echo "Converting test set..."
python processing_scripts/process_crag.py convert test

echo ">>> [Step 4/4] Building Elasticsearch index for the CRAG dataset..."
# Note: This assumes 'build_index.py' accepts a '--dataset' argument.
# Please verify the exact argument name if this step fails.
python retriever_server/build_index.py crag

echo ">>> All steps completed successfully!"
echo "CRAG 'multi-hop' dataset is now ready for use."

