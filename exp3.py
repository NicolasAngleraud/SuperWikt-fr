import pandas as pd
import argparse
import torch
import contextual_classifier_wiki as cclfw
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
	"hidden_layer_size": 512,
	"patience": 2,
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
	parser.add_argument("-run", choices=['1', '2', '3', '4', '5'], help="number of the run for an experiment.")
	parser.add_argument('-v', "--trace", action="store_true", help="Toggles the verbose mode. Default=False")
	args = parser.parse_args()
	return args


if __name__ == '__main__':
	args = get_parser_args()
	
	df_eval = []

	# DEVICE setup
	device_id = args.device_id
	if torch.cuda.is_available():
		DEVICE = torch.device("cuda:" + args.device_id)
	
	clf_file = f"./clfs/clf_{device_id}.params"
		

	run = int(args.run)
	patience = 2
	batch_size = int(args.batch_size)
	frozen = False
	lrs = [0.00001, 0.000005, 0.000001, 0.0000005]
	dropouts = [0.1, 0.3]
	hidden_layer_sizes = [512, 768]
	
	params_ids = flatten_list([[[f"CCLFDEXP3LR{k}DP{i}HL{j}" for i in range(len(dropouts))] for j in range(len(hidden_layer_sizes))] for k in range(len(lrs))])
	
	train_examples = encoded_examples(datafile=args.data_file, 'train', max_length=100)
	freq_dev_examples = encoded_examples(datafile=args.data_file, 'freq-dev', max_length=100)
	rand_dev_examples = encoded_examples(datafile=args.data_file, 'rand-dev', max_length=100)
	freq_test_examples = encoded_examples(datafile=args.data_file, 'freq-test', max_length=100)
	rand_test_examples = encoded_examples(datafile=args.data_file, 'rand-test', max_length=100)
	
	
	
	
	
	"""
	for lr in lrs:
		for hidden_layer_size in hidden_layer_sizes:
			for dropout in dropouts:
			
				params_id = params_ids.pop(0)
				clf_id = params_id + f"-{run}"
				eval_data = {}
				eval_data['params_id'] = params_id
				eval_data['clf_id'] = clf_id
				
				print()
				print(f"run {run} : lr = {lr}")
				print(f"dropout :  {dropout} ; hidden layer size : {hidden_layer_size}; batch_size : {batch_size}")
				print()


				params = {key: value for key, value in LPARAMETERS.items()}
				params['lr'] = lr
				params['patience'] = patience
				params['frozen'] = frozen
				params['batch_size'] = batch_size
				params['dropout'] = dropout
				params['hidden_layer_size'] = hidden_layer_size
				eval_data["run"] = run
	"""
