#!/bin/bash

# Default values for environment variables
REPO_DIR=$(dirname "$0")
# customize this output dir if needed
# all files created by this script are stored in $OUT
OUT=$REPO_DIR/out
MODEL_DIR=$OUT/models
SENSE_DATA_FILE=${REPO_DIR}/sense_data.tsv
EX_DATA_FILE=${REPO_DIR}/ex_data.tsv

# Set a default base value
DEFAULT_DEVICE_ID="cpu"

# Check if a specific value is provided as an argument or environment variable
if [ -n "$SPECIFIC_DEVICE_ID" ]; then
    DEVICE_ID="$SPECIFIC_DEVICE_ID"
else
    DEVICE_ID="$DEFAULT_DEVICE_ID"
fi

if [ "$DEVICE_ID" != "cpu" ]; then
    echo "Selected DEVICE_ID: cuda:$DEVICE_ID"
else
    echo "Selected DEVICE_ID: $DEVICE_ID"
fi


# Create model directory if it does not exist
mkdir -p $OUT
mkdir -p $MODEL_DIR


# Encoding of annotated data for FlauBERT large processing and training of definition and example classifiers
echo "Encoding of annotated data for FlauBERT large processing and training of definition and example classifiers"
python3 "$REPO_DIR/train_def_ex_lex_clf.py" --sense_data_file "$SENSE_DATA_FILE" --ex_data_file "$EX_DATA_FILE" --out "$OUT"  --device_id "$DEVICE_ID" --model_dir "$MODEL_DIR"
if [ $? -ne 0 ]; then
    echo "Error: train_def_ex_lex_clf.py failed"
    exit 1
fi

echo "Pipeline completed successfully"


