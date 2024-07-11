#!/bin/bash

# Default values for environment variables
SCRIPT_DIR=$(dirname "$0")
REPO_DIR=${REPO_DIR:-"$SCRIPT_DIR"}
PARENT_DIR=$(dirname "$SCRIPT_DIR")
MODEL_DIR=${MODEL_DIR:-"${REPO_DIR}/models"}
BZ2_FILE="${REPO_DIR}/fr_dbnary_ontolex_20240501.ttl.bz2"
DUMP_FILE=${DUMP_FILE:-"${REPO_DIR}/wiktionary.ttl"}
WIKTIONARY_FILE=${WIKTIONARY_FILE:-"${REPO_DIR}/wiktionary.tsv"}
EXAMPLES_FILE=${EXAMPLES_FILE:-"${REPO_DIR}/wiktionary_examples.tsv"}
PREDS_FILE=${PREDS_FILE:-"${REPO_DIR}/wiktionary_preds.tsv"}
ENRICHED_FILE=${ENRICHED_FILE:-"${REPO_DIR}/enriched_wiktionary.tsv"}
VENV_DIR=${VENV_DIR:-"$PARENT_DIR/venv"}



# Google Drive file IDs and names
DEF_MODEL_FILE_ID="1J9PAVP74KSNCG9PX0OaL6Zzc8WV6E7st"
DEF_MODEL_FILE_NAME="def_lem_clf.params"
EX_MODEL_FILE_ID="1ZM2Nlp5oZQJv0f0xRZvKIwRXtJb2OkMQ"
EX_MODEL_FILE_NAME="ex_clf.params"

# Create model directory if it does not exist
mkdir -p "$MODEL_DIR"

# Download DEF model from Google Drive
echo "Downloading DEF model from Google Drive..."
gdown "https://drive.google.com/uc?id=$DEF_MODEL_FILE_ID" -O "$MODEL_DIR/$DEF_MODEL_FILE_NAME"
if [ $? -ne 0 ]; then
    echo "Error downloading DEF model from Google Drive"
    exit 1
fi

# Download EX model from Google Drive
echo "Downloading EX model from Google Drive..."
gdown "https://drive.google.com/uc?id=$EX_MODEL_FILE_ID" -O "$MODEL_DIR/$EX_MODEL_FILE_NAME"
if [ $? -ne 0 ]; then
    echo "Error downloading EX model from Google Drive"
    exit 1
fi


# Check if the .bz2 file exists
if [ -f "$BZ2_FILE" ]; then
    echo "Extracting TTL file from $BZ2_FILE..."

    # Extract .ttl file from .bz2 file
    bunzip2 -c "$BZ2_FILE" > "$DUMP_FILE"

    echo "Extraction complete. TTL file saved to $DUMP_FILE"
else
    echo "Error: $BZ2_FILE not found."
    exit 1
fi

# Step 1: Extract wiktionary.tsv from the .ttl dump file
echo "Starting step 1: Extracting wiktionary.tsv from the .ttl dump file"
python3 "${REPO_DIR}/extract_wiki.py" --input "$DUMP_FILE" --output "$WIKTIONARY_FILE"
if [ $? -ne 0 ]; then
    echo "Error in step 1: extract_wiki.py failed"
    exit 1
fi


# Step 2: Generate wiktionary_examples.tsv from wiktionary.tsv
echo "Starting step 2: Generating wiktionary_examples.tsv from wiktionary.tsv"
# Function to download Spacy model if not already present
download_spacy_model() {
    model=$1
    python3 -c "import spacy; spacy.cli.download('${model}')"
}

# Download the required Spacy model
SPACY_MODEL="fr_core_news_lg"
echo "Downloading Spacy model: $SPACY_MODEL"
download_spacy_model $SPACY_MODEL

python3 "${REPO_DIR}/process_examples.py" --input "$WIKTIONARY_FILE" --output "$EXAMPLES_FILE"
if [ $? -ne 0 ]; then
    echo "Error in step 2: process_examples.py failed"
    exit 1
fi


echo "Pipeline completed successfully"

