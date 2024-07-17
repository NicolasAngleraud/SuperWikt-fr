# StageM2
This repository contains a shell script (extract_enrich_wiktionary.sh) designed to automate the extraction and enrichment of Wiktionary data using Python scripts. This README provides an overview of the script's purpose and the steps involved. The repository contains as well content for the supervised training of classifiers able to do coarse-grained semantic classification of lexical senses from the Wiktionary or of noun occurences in context, which can take the form of code scripts (lexicalClf.py, dataEncoder.py, main.py, test_arllm.py) or annotated data (data.xlsx).

Purpose

The extract_enrich_wiktionary.sh script facilitates the extraction of lexical data from a Wiktionary dump file, processes it using Python scripts, and enriches it with semantic classes predictions.

Steps

- Step 1: Extract Wiktionary Data
	
	script : extract_wiki.py. The script starts by extracting Wiktionary data from a specified dump file (wiktionary.ttl obtained by extracting it from a bz2 archive). The extracted data is saved as wiktionary.tsv.
     TODO: define what is kept exactly
     
- Step 2: Process Examples
	
	script: process_examples.py. Next, the script processes examples from the wiktionary.tsv file to create wiktionary_examples.tsv. This step involves another Python script (), which prepares example data for subsequent analysis.
     TODO: add what "process" mean : tabulated format , output columns
     Spacy is used to tokenize, lemmatization to identify the rank of the token in the exemplar sentence, whose sense is illustrated
    
- Step 3: Generate Predictions
	
	script : get_preds.py. Using both wiktionary.tsv and wiktionary_examples.tsv, the script generates predictions (wiktionary_preds.tsv). It applies the classifier of definitions and the classifier of exemplar sentences, and combines it using a weighted sum of scores. The classifiers are downloaded from a url and stored locally to $OUT/models
    
- Step 4: Enrich Wiktionary Data
	
	script: enrich_wiktionary.py. Finally, the script enriches the Wiktionary data by combining wiktionary.tsv and wiktionary_preds.tsv. This step enhances the dataset with additional semantic information, creating enriched_wiktionary.tsv.
    
Usage

To execute the script and perform the above steps:

- Clone the Repository: Clone the repository containing extract_enrich_wiktionary.sh, the bz2 archive fr_dbnary_ontolex_20240501.ttl.bz2 (can be replaced with a more recent one) containing the Wiktionary ttl dump file and the required Python scripts (extract_wiki.py, process_examples.py, get_preds.py, enrich_wiktionary.py, dataEncoder.py, lexicalClf.py).

- Set Up Environment: Ensure Python 3.x and necessary packages are installed. Optionally, create a virtual environment (venv or virtualenv) to manage dependencies.

- Run the Shell Script: Execute extract_enrich_wiktionary.sh from the command line: ./extract_enrich_wiktionary.sh. This will initiate the process of extracting, processing, and enriching Wiktionary data as described in the steps above.

Requirements

    Python 3.x
    Required Python packages (specified in requirements.txt or installed manually):
        pandas
        argparse
        torch
        sacremoses
        numpy
        transformers
        matplotlib
        spacy
        wget

Notes

Customize file paths and script parameters (DUMP_FILE, WIKTIONARY_FILE, etc.) in extract_enrich_wiktionary.sh as per your environment and dataset location. Ensure adequate permissions (chmod +x extract_enrich_wiktionary.sh) to execute the shell script.

Contributors

Nicolas Angleraud
