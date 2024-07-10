#!/bin/bash

# Default values for environment variables
# REPO_DIR=${REPO_DIR:-"./"}
MODEL_DIR=${MODEL_DIR:-"./models"}
DUMP_FILE=${DUMP_FILE:-"./test.ttl"}
WIKTIONARY_FILE=${WIKTIONARY_FILE:-"./wiktionary.tsv"}
EXAMPLES_FILE=${EXAMPLES_FILE:-"./wiktionary_examples.tsv"}
PREDS_FILE=${PREDS_FILE:-"./wiktionary_preds.tsv"}
ENRICHED_FILE=${ENRICHED_FILE:-"./enriched_wiktionary.tsv"}


# Google Drive file links and names
DEF_MODEL_FILE_ID="1J9PAVP74KSNCG9PX0OaL6Zzc8WV6E7st"
DEF_MODEL_FILE_NAME="def_lem_clf.params"
EX_MODEL_FILE_ID="1ZM2Nlp5oZQJv0f0xRZvKIwRXtJb2OkMQ"
EX_MODEL_FILE_NAME="ex_clf.params"
DUMP_FILE_ID="1QKZjcYVqFkFWwup3zBlmoswAkB9l0Gkr"
DUMP_FILE_NAME="test.ttl"

# Create model directory if it does not exist
mkdir -p "$MODEL_DIR"

# Download Wiki dump file from Google Drive
echo "Downloading Wiki dump file from Google Drive..."
gdown "https://drive.google.com/uc?id=$DEF_MODEL_FILE_ID" -O "$DUMP_FILE"
if [ $? -ne 0 ]; then
    echo "Error downloading Wiki dump file from Google Drive"
    exit 1
fi


# Step 1: Extract wiktionary.tsv from the .ttl dump file
echo "Starting step 1: Extracting wiktionary.tsv from the .ttl dump file"
python3 "./extract_wiki.py" --input "$DUMP_FILE" --output "$WIKTIONARY_FILE"
if [ $? -ne 0 ]; then
    echo "Error in step 1: extract_wiki.py failed"
    exit 1
fi


echo "Pipeline completed successfully"

