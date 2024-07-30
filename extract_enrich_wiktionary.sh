#!/bin/bash


REPO_DIR=$(dirname "$0")

OUT=$REPO_DIR/out
MODEL_DIR=$OUT/models
BZ2_FILE=${REPO_DIR}/fr_dbnary_ontolex_20240501.ttl.bz2
DUMP_FILE=${OUT}/wiktionary.ttl
WIKTIONARY_FILE=${OUT}/wiktionary.tsv
EXAMPLES_FILE=${OUT}/wiktionary_examples.tsv
PREDS_FILE=${OUT}/wiktionary_preds.tsv
ENRICHED_FILE=${OUT}/enriched_wiktionary.tsv


DEFAULT_DEVICE_ID="cpu"


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


URL="http://www.linguist.univ-paris-diderot.fr/~mcandito/nangleraud_wikt_supersenses/models/"

DEF_MODEL_FILE_NAME="def_lem_clf.params"
EX_MODEL_FILE_NAME="ex_clf.params"


mkdir -p $OUT
mkdir -p $MODEL_DIR


echo "Downloading DEF model ..."
wget $URL/$DEF_MODEL_FILE_NAME -O "$MODEL_DIR/$DEF_MODEL_FILE_NAME"
if [ $? -ne 0 ]; then
    echo "Error downloading DEF model"
    exit 1
fi


echo "Downloading EX model ..."
wget $URL/$EX_MODEL_FILE_NAME -O "$MODEL_DIR/$EX_MODEL_FILE_NAME"
if [ $? -ne 0 ]; then
    echo "Error downloading EX model"
    exit 1
fi


if [ -f "$BZ2_FILE" ]; then
    echo "Extracting TTL file from $BZ2_FILE..."

    bunzip2 -c "$BZ2_FILE" > "$DUMP_FILE"

    echo "Extraction complete. TTL file saved to $DUMP_FILE"
else
    echo "Error: $BZ2_FILE not found."
    exit 1
fi


echo "WIKTIONARY EXTRACTION..."
python3 "$REPO_DIR/extract_wiki.py" --input "$DUMP_FILE" --output "$WIKTIONARY_FILE"
if [ $? -ne 0 ]; then
    echo "Error in step 1: extract_wiki.py failed"
    exit 1
fi



echo "WIKTIONARY EXAMPLES PROCESSING..."
download_spacy_model() {
    model=$1
    python3 -c "import spacy; spacy.cli.download('${model}')"
}

SPACY_MODEL="fr_core_news_lg"
echo "Downloading Spacy model: $SPACY_MODEL"
download_spacy_model $SPACY_MODEL

python3 "$REPO_DIR/process_examples.py" --input "$WIKTIONARY_FILE" --output "$EXAMPLES_FILE"
if [ $? -ne 0 ]; then
    echo "Error in step 2: process_examples.py failed"
    exit 1
fi


echo "RESOURCE'S LEXICAL SENSES SUPERSENSES PREDICTIONS..."
python3 "$REPO_DIR/get_preds.py" --input_wiktionary "$WIKTIONARY_FILE" --input_examples "$EXAMPLES_FILE" --output "$PREDS_FILE" --model_dir "$MODEL_DIR" --device_id "$DEVICE_ID"
if [ $? -ne 0 ]; then
    echo "Error in step 3: get_preds.py failed"
    exit 1
fi



echo "WIKTIONARY RESOURCE ENRICHMENT..."
python3 "$REPO_DIR/enrich_wiktionary.py" --input_wiktionary "$WIKTIONARY_FILE" --input_preds "$PREDS_FILE" --output "$ENRICHED_FILE"
if [ $? -ne 0 ]; then
    echo "Error in step 4: enrich_wiktionary.py failed"
    exit 1
fi

echo "Resource production pipeline completed successfully."

