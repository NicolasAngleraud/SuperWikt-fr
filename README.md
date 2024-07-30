# StageM2
This repository contains files aiming at the production of a lexical resource for french nouns based on a partial extraction of the Wiktionary enriched with semantic information through trained classifiers able to classify a lexical sense into coarse semantic classes called supersenses (such as Person, Animal, Feeling, Cognition, Act, ...). It contains the following files:

- extract_enrich_wiktionary.sh: Shell script implementing a pipeline meant to partialy extract information of Wiktionary from bz2 archive of a dump file and enrich the extracted resource with supersenses using trained models based on FlauBERT large.

- train_new_def_ex_model.sh: Shell script implementing a pipepline meant to train new definition and examlple supersense classifiers using new sense annotated data. It keeps the same hyperparameters as the ones used to get current classifiers.

- 

- 

- 

- 

- 

- 

- 

- 

- 

- 

- 

- 


There are two main typical pipelines:

## 1. Predict supersenses

The extract_enrich_wiktionary.sh script facilitates the extraction of lexical data from a Wiktionary dump file, processes it using Python scripts, and enriches it with semantic classes predictions.

**Steps**

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

## 2. Retraining models

indiquer les scripts pour l'entraînement des modèles


## TO BE NOTED

**Usage**

To execute the scripts and perform the above steps:

- Clone the Repository

- Set Up Environment: Ensure Python 3.x and necessary packages are installed. Optionally, create a virtual environment (venv or virtualenv) to manage dependencies.

- Run the desired Shell Script: Execute extract_enrich_wiktionary.sh from the command lines: ./extract_enrich_wiktionary.sh and ./train_new_def_ex_model.sh.

**Requirements**

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

**Notes**

Customize file paths and script parameters (DUMP_FILE, WIKTIONARY_FILE, etc.) in extract_enrich_wiktionary.sh as per your environment and dataset location. Ensure adequate permissions (chmod +x extract_enrich_wiktionary.sh) to execute the shell script.

**Contributors**

Nicolas Angleraud
