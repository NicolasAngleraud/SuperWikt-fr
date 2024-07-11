# StageM2
This repository contains a shell script (extract_enrich_wiktionary.sh) designed to automate the extraction and enrichment of Wiktionary data using Python scripts. This README provides an overview of the script's purpose and the steps involved.

Purpose

    The extract_enrich_wiktionary.sh script facilitates the extraction of lexical data from a Wiktionary dump file, processes it using Python scripts, and enriches it with semantic classes predictions.

Steps

    Step 1: Extract Wiktionary Data
     The script starts by extracting Wiktionary data from a specified dump file (wiktionary.ttl obtained by extracting it from a bz2 archive) using a Python script (extract_wiki.py). The extracted data is saved as wiktionary.tsv.
     
    Step 2: Process Examples
     Next, the script processes examples from the wiktionary.tsv file to create wiktionary_examples.tsv. This step involves another Python script (process_examples.py), which prepares example data for subsequent analysis.
    
    Step 3: Generate Predictions
     Using both wiktionary.tsv and wiktionary_examples.tsv, the script generates predictions (wiktionary_preds.tsv). This step employs get_preds.py, a Python script that utilizes pre-trained models and algorithms to predict semantic classes for each lexical sense of the wiktioanry based on the processed data.
    
    Step 4: Enrich Wiktionary Data
     Finally, the script enriches the Wiktionary data by combining wiktionary.tsv and wiktionary_preds.tsv. This step enhances the dataset with additional semantic information, creating enriched_wiktionary.tsv.
    
Usage

To execute the script and perform the above steps:

    Clone the Repository: Clone the repository containing extract_enrich_wiktionary.sh and the required Python scripts (extract_wiki.py, process_examples.py, get_preds.py, enrich_wiktionary.py).

    Set Up Environment: Ensure Python 3.x and necessary packages are installed. Optionally, create a virtual environment (venv or virtualenv) to manage dependencies.

    Run the Shell Script: Execute extract_enrich_wiktionary.sh from the command line:

    bash

    ./extract_enrich_wiktionary.sh

    This will initiate the process of extracting, processing, and enriching Wiktionary data as described in the steps above.

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
        gdown

Notes

    Customize file paths and script parameters (DUMP_FILE, WIKTIONARY_FILE, etc.) in extract_enrich_wiktionary.sh as per your environment and dataset location.
    Ensure adequate permissions (chmod +x extract_enrich_wiktionary.sh) to execute the shell script.

Contributors

    Nicolas Angleraud
