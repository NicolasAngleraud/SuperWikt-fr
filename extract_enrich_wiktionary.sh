#!/bin/bash

# Default values for environment variables
REPO_DIR=${REPO_DIR:-"./"}
MODEL_DIR=${MODEL_DIR:-"$REPO_DIR/models"}
DUMP_FILE=${DUMP_FILE:-"$REPO_DIR/test.ttl"}
WIKTIONARY_FILE=${WIKTIONARY_FILE:-"$REPO_DIR/wiktionary.tsv"}
EXAMPLES_FILE=${EXAMPLES_FILE:-"$REPO_DIR/wiktionary_examples.tsv"}
PREDS_FILE=${PREDS_FILE:-"$REPO_DIR/wiktionary_preds.tsv"}
ENRICHED_FILE=${ENRICHED_FILE:-"$REPO_DIR/enriched_wiktionary.tsv"}


# Google Drive file links and names
DEF_MODEL_FILE_LINK="https://drive.google.com/file/d/1J9PAVP74KSNCG9PX0OaL6Zzc8WV6E7st/view?usp=drive_link"
DEF_MODEL_FILE_NAME="def_lem_clf.params"
EX_MODEL_FILE_LINK="https://drive.google.com/file/d/1ZM2Nlp5oZQJv0f0xRZvKIwRXtJb2OkMQ/view?usp=drive_link"
EX_MODEL_FILE_NAME="ex_clf.params"
DUMP_FILE_LINK="https://drive.google.com/file/d/1QKZjcYVqFkFWwup3zBlmoswAkB9l0Gkr/view?usp=drive_link"

# Create model directory if it does not exist
mkdir -p "$MODEL_DIR"

# Download DEF model from Google Drive
echo "Downloading DEF model from Google Drive..."
gdown "$DEF_MODEL_FILE_LINK" -O "$MODEL_DIR/$DEF_MODEL_FILE_NAME"
if [ $? -ne 0 ]; then
    echo "Error downloading DEF model from Google Drive"
    exit 1
fi

# Download EX model from Google Drive
echo "Downloading EX model from Google Drive..."
gdown "$EX_MODEL_FILE_LINK" -O "$MODEL_DIR/$EX_MODEL_FILE_NAME"
if [ $? -ne 0 ]; then
    echo "Error downloading EX model from Google Drive"
    exit 1
fi

# Download Wiki dump file from Google Drive
echo "Downloading Wiki dump file from Google Drive..."
gdown "$DUMP_FILE_LINK" -O "$DUMP_FILE"
if [ $? -ne 0 ]; then
    echo "Error downloading Wiki dump file from Google Drive"
    exit 1
fi




# Step 1: Extract wiktionary.tsv from the .ttl dump file
echo "Starting step 1: Extracting wiktionary.tsv from the .ttl dump file"
python3 "$REPO_DIR/extract_wiki.py" --input "$DUMP_FILE" --output "$WIKTIONARY_FILE"
if [ $? -ne 0 ]; then
    echo "Error in step 1: extract_wiki.py failed"
    exit 1
fi

'''

# Step 2: Generate wiktionary_examples.tsv from wiktionary.tsv
echo "Starting step 2: Generating wiktionary_examples.tsv from wiktionary.tsv"
python3 "$REPO_DIR/process_examples.py" --input "$WIKTIONARY_FILE" --output "$EXAMPLES_FILE"
if [ $? -ne 0 ]; then
    echo "Error in step 2: process_examples.py failed"
    exit 1
fi

# Step 3: Generate wiktionary_preds.tsv using wiktionary.tsv and wiktionary_examples.tsv
echo "Starting step 3: Generating wiktionary_preds.tsv using wiktionary.tsv and wiktionary_examples.tsv"
python3 "$REPO_DIR/get_preds.py" --input_wiktionary "$WIKTIONARY_FILE" --input_examples "$EXAMPLES_FILE" --output "$PREDS_FILE" --model_dir "$MODEL_DIR"
if [ $? -ne 0 ]; then
    echo "Error in step 3: get_preds.py failed"
    exit 1
fi

# Step 4: Generate enriched_wiktionary.tsv using wiktionary.tsv and wiktionary_preds.tsv
echo "Starting step 4: Generating enriched_wiktionary.tsv using wiktionary.tsv and wiktionary_preds.tsv"
python3 "$REPO_DIR/enrich_wiktionary.py" --input_wiktionary "$WIKTIONARY_FILE" --input_preds "$PREDS_FILE" --output "$ENRICHED_FILE"
if [ $? -ne 0 ]; then
    echo "Error in step 4: enrich_wiktionary.py failed"
    exit 1
fi

'''

echo "Pipeline completed successfully."

