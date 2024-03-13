import pandas as pd
import argparse
import torch
import spacy
import contextual_classifier_wiki as cclf
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
	
	nlp = spacy.load("fr_core_news_lg")
	
	df_eval = []

	# DEVICE setup
	device_id = args.device_id
	if torch.cuda.is_available():
		DEVICE = torch.device("cuda:" + args.device_id)
	
	clf_file = f"./clfs/clf_{device_id}.params"
	
	MODEL_NAME = "flaubert/flaubert_large_cased"
	tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

	run = int(args.run)
	patience = 2
	batch_size = int(args.batch_size)
	frozen = False
	max_seq_length = 100
	lrs = [0.00001]
	dropouts = [0.3]
	hidden_layer_sizes = [512]
	
	params_ids = flatten_list([[[f"CCLFDEXP3LR{k}DP{i}HL{j}" for i in range(len(dropouts))] for j in range(len(hidden_layer_sizes))] for k in range(len(lrs))])
	

	
	train_examples = cclf.encoded_definitions(datafile=args.data_file, nlp=nlp, set_='train', max_length=max_seq_length)
	freq_dev_examples = cclf.encoded_definitions(datafile=args.data_file, nlp=nlp, set_='freq-dev', max_length=max_seq_length)
	rand_dev_examples = cclf.encoded_definitions(datafile=args.data_file, nlp=nlp, set_='rand-dev', max_length=max_seq_length)
	freq_test_examples = cclf.encoded_definitions(datafile=args.data_file, nlp=nlp, set_='freq-test', max_length=max_seq_length)
	rand_test_examples = cclf.encoded_definitions(datafile=args.data_file, nlp=nlp, set_='rand-test', max_length=max_seq_length)
	
	print(len(list(train_examples)))
	print(len(list(freq_dev_examples)))
	print(len(list(rand_dev_examples)))
	print(len(list(freq_test_examples)))
	print(len(list(rand_test_examples)))
	
	
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

				classifier = cclf.SupersenseTagger(params, DEVICE)
				cclf.training(params, train_examples, freq_dev_examples, rand_dev_examples, classifier, DEVICE, eval_data, clf_file)
				
				classifier = cclf.SupersenseTagger(params, DEVICE)
				classifier.load_state_dict(torch.load(clf_file))
				
				cclf.evaluation(train_examples, classifier, params, DEVICE, "train", eval_data, "exp3")
				cclf.evaluation(freq_dev_examples, classifier, params, DEVICE, "freq-dev", eval_data, "exp3")
				cclf.evaluation(rand_dev_examples, classifier, params, DEVICE, "rand-dev", eval_data, "exp3")

				print(f"CLASSIFIER TRAINED ON {len(train_examples)} DEFINITIONS...")

				sequoia_baseline = cclf.MostFrequentSequoia()
				train_baseline = cclf.MostFrequentTrainingData()
				wiki_baseline = cclf.MostFrequentWiktionary()

				sequoia_baseline.training()
				train_baseline.training()
				wiki_baseline.training()
				
				eval_data["train-sequoia_baseline"] = sequoia_baseline.evaluation(train_examples)
				eval_data["train-train_baseline"] =train_baseline.evaluation(train_examples)
				eval_data["train-wiki_baseline"] = wiki_baseline.evaluation(train_examples)

				eval_data["freq_dev-sequoia_baseline"] = sequoia_baseline.evaluation(freq_dev_examples)
				eval_data["freq_dev-train_baseline"] =train_baseline.evaluation(freq_dev_examples)
				eval_data["freq_dev-wiki_baseline"] = wiki_baseline.evaluation(freq_dev_examples)

				eval_data["rand_dev-sequoia_baseline"] = sequoia_baseline.evaluation(rand_dev_examples)
				eval_data["rand_dev-train_baseline"] =train_baseline.evaluation(rand_dev_examples)
				eval_data["rand_dev-wiki_baseline"] = wiki_baseline.evaluation(rand_dev_examples)

				print("BASELINES COMPUTED...")
				
				cclf.evaluation(freq_test_examples, classifier, params, DEVICE, "freq-test", eval_data, "exp3")
				cclf.evaluation(rand_test_examples, classifier, params, DEVICE, "rand-test", eval_data, "exp3")

				df_eval.append(eval_data)

	print("CREATION OF THE EVALUATION FILE...")
	df = pd.DataFrame(df_eval)
	excel_filename = f'./exp3/contextual_classifier_definitions_wiki_results-run{run}.xlsx'
	df.to_excel(excel_filename, index=False)
	
	print("PROCESS DONE.\n")
