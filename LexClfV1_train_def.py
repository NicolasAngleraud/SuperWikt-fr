# EXP4LR2DP0HL1 exemples
# EXP3LR1DP0HL1 définitions

import pandas as pd
import argparse
import torch
import spacy
import mono_target_unique_rank_classifier as clf
from transformers import AutoModel, AutoTokenizer
import datetime


SUPERSENSES = ['act', 'animal', 'artifact', 'attribute', 'body', 'cognition',
               'communication', 'event', 'feeling', 'food', 'institution', 'act*cognition',
               'object', 'possession', 'person', 'phenomenon', 'plant', 'artifact*cognition',
               'quantity', 'relation', 'state', 'substance', 'time', 'groupxperson']

HYPERSENSES = {"dynamic_situation": ["act", "event", "phenomenon", "act*cognition"],
               "stative_situation": ["attribute", "state", "feeling", "relation"],
               "animate_entity": ["animal", "person", "groupxperson"],
               "inanimate_entity": ["artifact", "food", "body", "object", "plant", "substance", "artifact*cognition"],
               "informational_object": ["cognition", "communication", "act*cognition", "artifact*cognition"],
               "quantification": ["quantity", "part", "group", "groupxperson"],
               "other": ["institution", "possession", "time"]
               }
               
LPARAMETERS = {
	"nb_epochs": 100,
	"batch_size": 16,
	"hidden_layer_size": 768,
	"patience": 1,
	"lr": 0.00001,
	"frozen": False,
	"dropout": 0.1,
	"max_seq_length": 100
}

params_keys = ["nb_epochs", "batch_size", "hidden_layer_size", "patience", "lr", "frozen", "dropout", "max_seq_length"]

def create_unique_id(unique_mark):
	return f"{datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')}-{unique_mark}"

def flatten_list(lst):
    return [item for sublist in lst for item in (flatten_list(sublist) if isinstance(sublist, list) else [sublist])]

def get_parser_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("-device_id", choices=['0', '1', '2', '3'], help="Id of the GPU.")
	parser.add_argument("-data_file", default="./donnees.xlsx", help="The excel file containing all the annotated sense data from wiktionary.")
	parser.add_argument("-batch_size", choices=['2', '4', '8', '16', '32', '64'], help="batch size for the classifier.")
	parser.add_argument('-v', "--trace", action="store_true", help="Toggles the verbose mode. Default=False")
	args = parser.parse_args()
	return args


if __name__ == '__main__':
	args = get_parser_args()
	

	# DEVICE setup
	device_id = args.device_id
	if torch.cuda.is_available():
		DEVICE = torch.device("cuda:" + args.device_id)
	
	MODEL_NAME = "flaubert/flaubert_large_cased"
	tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

	patience = 1
	batch_size = int(args.batch_size)
	frozen = False
	max_seq_length = 100
	lr_def = 0.000005
	dropout = 0.1
	hidden_layer_size = 768
	token_rank = 1	
	
	train_defs, train_supersenses, train_lemmas, train_senses_ids = clf.encoded_examples(datafile=args.data_file, dataset='train')
	
	freq_dev_defs, freq_dev_supersenses, freq_dev_lemmas, freq_dev_senses_ids = clf.encoded_examples(datafile=args.data_file, dataset='freq-dev')
	
	rand_dev_defs, rand_dev_supersenses, rand_dev_lemmas, rand_dev_senses_ids = clf.encoded_examples(datafile=args.data_file, dataset='rand-dev')
	
	clf_file = f"./clfs/clf_{device_id}.params"

	max_perf = 0
	path = './clfs/LexClfV1_def.params'

	for run in range(10):
		
		print()
		print()
		print("RUN ", run+1)
		print()

		params = {key: value for key, value in LPARAMETERS.items()}
		params['lr'] = lr_def
		params['patience'] = patience
		params['max_seq_length'] = max_seq_length
		params['frozen'] = frozen
		params['batch_size'] = batch_size
		params['dropout'] = dropout
		params['token_rank'] = token_rank
		params['hidden_layer_size'] = hidden_layer_size

		classifier = clf.SupersenseTagger(params, DEVICE)
		clf.training(params, train_defs, train_supersenses, freq_dev_defs, freq_dev_supersenses, rand_dev_defs, rand_dev_supersenses, classifier, DEVICE, clf_file)
		
		print(f"CLASSIFIER TRAINED ON {len(train_defs)} EXAMPLES...")
		
		classifier = clf.SupersenseTagger(params, DEVICE)
		classifier.load_state_dict(torch.load(clf_file))
		
		max_perf = clf.save_best_clf(max_perf, freq_dev_defs, freq_dev_supersenses, rand_dev_defs, rand_dev_supersenses, classifier, parameters, path, DEVICE)
		
	print(f"BEST DEFINITION CLASSIFIER (mean accuracy = {max_perf}) SAVED...")

	print("PROCESS DONE.\n")

