import pandas as pd
import argparse
from collections import Counter, defaultdict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sacremoses
from random import shuffle
import os
import numpy as np
from transformers import AutoModel, AutoTokenizer, AutoConfig
from matplotlib import pyplot as plt
import warnings
import lexicalClf as clf
import subprocess
import dataEncoder as data
warnings.filterwarnings("ignore")


SUPERSENSES = ['act', 'animal', 'artifact', 'attribute', 'body', 'cognition',
               'communication', 'event', 'feeling', 'food', 'institution', 'act*cognition',
               'object', 'possession', 'person', 'phenomenon', 'plant', 'artifact*cognition',
               'quantity', 'relation', 'state', 'substance', 'time', 'groupxperson']

HYPERSENSES = {"dynamic_situation": ["act", "event", "phenomenon"],
               "stative_situation": ["attribute", "state", "feeling", "relation"],
               "animate_entity": ["animal", "person"],
               "inanimate_entity": ["artifact", "food", "body", "object", "plant", "substance"],
               "informational_object": ["cognition", "communication"],
               "quantification": ["quantity", "part", "group"],
               "other": ["institution", "possession", "time"]
               }
                      
supersense2i = {supersense: i for i, supersense in enumerate(SUPERSENSES)}
MODEL_NAME = "flaubert/flaubert_large_cased"

def percentage(decimal):
    percentage = decimal * 100
    return f"{percentage:.2f}%"


def get_parser_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("--device_id", choices=['cpu', '0', '1', '2', '3'], help="Id of the device used for computation.")
	parser.add_argument("--sense_data_file", default="./sense_data.tsv", help="The tsv file containing all the annotated sense data from Wiktionary.")
	parser.add_argument("--ex_data_file", default="./ex_data.tsv", help="The tsv file containing all the annotated examples data from Wiktionary.")
	parser.add_argument('-v', "--trace", action="store_true", help="Toggles the verbose mode. Default=False")
	args = parser.parse_args()
	return args


if __name__ == '__main__':
	args = get_parser_args()
	
	device_id = args.device_id
	if device_id != 'cpu':
		if torch.cuda.is_available():
			DEVICE = torch.device("cuda:" + args.device_id)
	else:
		DEVICE = 'cpu'
	
	
	def_lem_clf_file = f'./out/models/NEW_def_lem_clf_{device_id}.params'
	ex_clf_file = f'./out/models/NEW_ex_clf_{device_id}.params'
	
	params_def = {
		"nb_epochs": 100,
		"batch_size": 16,
		"hidden_layer_size": 768,
		"patience": 2,
		"lr": 0.000005,
		"weight_decay": 0.001,
		"frozen": False,
		"max_seq_length": 100
		}
	
	
	params_ex = {
		"nb_epochs": 100,
		"batch_size": 16,
		"hidden_layer_size": 768,
		"patience": 2,
		"lr": 0.000005,
		"weight_decay": 0.001,
		"frozen": False,
		"max_seq_length": 100
		}
	
	
	tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
	"""
	print('ENCODING DEFINITIONS DATA...\n')
	train_definitions_encoder = data.definitionEncoder(args.sense_data_file, args.ex_data_file, "train", tokenizer, remove_demonyms=False, use_sample=False)
	train_definitions_encoder.encode()
	train_definitions_encoder.shuffle_data()
	
	freq_dev_definitions_encoder = data.definitionEncoder(args.sense_data_file, args.ex_data_file, "freq-dev", tokenizer, remove_demonyms=False, use_sample=False)
	freq_dev_definitions_encoder.encode()
	
	rand_dev_definitions_encoder = data.definitionEncoder(args.sense_data_file, args.ex_data_file, "rand-dev", tokenizer, remove_demonyms=False, use_sample=False)
	rand_dev_definitions_encoder.encode()
	"""
	"""
	freq_test_definitions_encoder = data.definitionEncoder(args.sense_data_file, args.ex_data_file, "freq-test", tokenizer, remove_demonyms=False, use_sample=False)
	freq_test_definitions_encoder.encode()
	
	rand_test_definitions_encoder = data.definitionEncoder(args.sense_data_file, args.ex_data_file, "rand-test", tokenizer, remove_demonyms=False, use_sample=False)
	rand_test_definitions_encoder.encode()
	
	train_encoder_2000 = train_definitions_encoder.clone()
	train_encoder_2000.truncate_senses(k=2000)
	
	train_encoder_4000 = train_definitions_encoder.clone()
	train_encoder_4000.truncate_senses(k=4000)
	
	train_encoder_6000 = train_definitions_encoder.clone()
	train_encoder_6000.truncate_senses(k=6000)
	
	train_encoder_8000 = train_definitions_encoder.clone()
	train_encoder_8000.truncate_senses(k=8000)
	"""
	"""
	print('DEFINITIONS DATA ENCODED.\n')
	
	results = []

	
	for nb, enc in zip([10117], [train_definitions_encoder]):
	
		for run in range(5):
		
			print()
			print("RUN", run+1, "NB TRAINING EXAMPLES", nb)
			print()

			print('TRAINING DEFINITION CLASSIFIER...\n')
			def_clf = clf.monoRankClf(params_def, DEVICE, use_lemma=True, bert_model_name=MODEL_NAME)
			def_clf.train_clf(enc, freq_dev_definitions_encoder, rand_dev_definitions_encoder, def_lem_clf_file)
			print('DEFINITION CLASSIFIER TRAINED.\n')
			print('LOADING BEST DEFINITION CLASSIFIER...\n')
			def_clf = clf.monoRankClf(params_def, DEVICE, use_lemma=True, bert_model_name=MODEL_NAME)
			def_clf.load_clf(def_lem_clf_file)
			print('BEST DEFINITION CLASSIFIER LOADED.\n')


			train_accuracy = def_clf.evaluate(enc)

			freq_dev_accuracy = def_clf.evaluate(freq_dev_definitions_encoder)
			rand_dev_accuracy = def_clf.evaluate(rand_dev_definitions_encoder)

			#freq_test_accuracy = def_clf.evaluate(freq_test_definitions_encoder)
			#rand_test_accuracy = def_clf.evaluate(rand_test_definitions_encoder)

			print("train def accurcay = ", percentage(train_accuracy))
			print("freq dev def accurcay = ", percentage(freq_dev_accuracy))
			print("rand dev def accurcay = ", percentage(rand_dev_accuracy))
			#print("freq test def accurcay = ", percentage(freq_test_accuracy))
			#print("rand test def accurcay = ", percentage(rand_test_accuracy))
			print()
			
			comb = {"number_training_examples": nb,
					"run": run+1,
					"train_accuracy": train_accuracy,
					"freq_dev_accuracy": freq_dev_accuracy,
					"rand_dev_accuracy": rand_dev_accuracy}#,
					#"freq_test_accuracy": freq_test_accuracy,
					#"rand_test_accuracy": rand_test_accuracy}
			results.append(comb)
			
			freq_dev_predictions = def_clf.predict(freq_dev_definitions_encoder)
			rand_dev_predictions = def_clf.predict(rand_dev_definitions_encoder)
			#freq_test_predictions = def_clf.predict(freq_test_definitions_encoder)
			#rand_test_predictions = def_clf.predict(rand_test_definitions_encoder)
			
			
			freq_dev_df = pd.DataFrame(freq_dev_predictions)
			freq_dev_df.to_csv(f'./out/training_curve/freq_dev_preds_{nb}_{run+1}_{device_id}.tsv', sep='\t', index=False, encoding='utf-8')

			rand_dev_df = pd.DataFrame(rand_dev_predictions)
			rand_dev_df.to_csv(f'./out/training_curve/rand_dev_preds_{nb}_{run+1}_{device_id}.tsv', sep='\t', index=False, encoding='utf-8')

			#freq_test_df = pd.DataFrame(freq_test_predictions)
			#freq_test_df.to_csv(f'./out/training_curve/freq_test_preds_{nb}_{run+1}_{device_id}.tsv', sep='\t', index=False, encoding='utf-8')

			#rand_test_df = pd.DataFrame(rand_test_predictions)
			#rand_test_df.to_csv(f'./out/training_curve/rand_test_preds_{nb}_{run+1}_{device_id}.tsv', sep='\t', index=False, encoding='utf-8')
			
	
	df = pd.DataFrame(results)
	df.to_csv(f'./out/training_curve/training_curve_{device_id}.tsv', sep='\t', index=False, encoding='utf-8')
	"""
	
	
	print('ENCODING EXAMPLES DATA...\n')
	train_examples_encoder = data.exampleEncoder(args.sense_data_file, args.ex_data_file, "train", tokenizer, use_sample=False, sub_corpus="wiki")
	train_examples_encoder.encode()
	freq_dev_examples_encoder = data.exampleEncoder(args.sense_data_file, args.ex_data_file, "freq-dev", tokenizer, use_sample=False, sub_corpus="wiki")
	freq_dev_examples_encoder.encode()
	rand_dev_examples_encoder = data.exampleEncoder(args.sense_data_file, args.ex_data_file, "rand-dev", tokenizer, use_sample=False, sub_corpus="wiki")
	rand_dev_examples_encoder.encode()
	
	
	#print(train_examples_encoder.length)
	#print(freq_dev_examples_encoder.length)
	#print(rand_dev_examples_encoder.length)
	#freq_dev_sense_encoder = senseEncoder(sense_datafile, ex_datafile, "freq-dev", tokenizer, use_sample=False)
	#rand_dev_sense_encoder = senseEncoder(sense_datafile, ex_datafile, "rand-dev", tokenizer, use_sample=False)
	print('EXAMPLES DATA ENCODED.\n')
	
	results = []
	
	
	for run in range(2):
		
		print()
		print("RUN", run+1)
		print()
	
		print('TRAINING EXAMPLE CLASSIFIER...\n')
		ex_clf = clf.multiRankClf(params_ex, DEVICE, dropout_input=0, dropout_hidden=0.3, bert_model_name=MODEL_NAME)
		ex_clf.train_clf(train_examples_encoder, freq_dev_examples_encoder, rand_dev_examples_encoder, ex_clf_file)
		print('EXAMPLE CLASSIFIER TRAINED.\n')
		print('LOADING BEST EXAMPLE CLASSIFIER...\n')
		ex_clf = clf.multiRankClf(params_ex, DEVICE, dropout_input=0, dropout_hidden=0.3, bert_model_name=MODEL_NAME)
		ex_clf.load_clf(ex_clf_file)
		print('BEST EXAMPLE CLASSIFIER LOADED.\n')
		
		train_accuracy = ex_clf.evaluate(train_examples_encoder)
		freq_dev_accuracy = ex_clf.evaluate(freq_dev_examples_encoder)
		rand_dev_accuracy = ex_clf.evaluate(rand_dev_examples_encoder)
		
		freq_dev_predictions = ex_clf.predict(freq_dev_examples_encoder)
		rand_dev_predictions = ex_clf.predict(rand_dev_examples_encoder)

		
		print("train accurcay = ", percentage(train_accuracy))
		print("freq dev accurcay = ", percentage(freq_dev_accuracy))
		print("rand dev accurcay = ", percentage(rand_dev_accuracy))

		print()
		
		comb = {
			"run": run+1,
			"train_accuracy": train_accuracy,
			"freq_dev_accuracy": freq_dev_accuracy,
			"rand_dev_accuracy": rand_dev_accuracy
			}
		results.append(comb)
		
		freq_dev_ex_df = pd.DataFrame(freq_dev_predictions)
		freq_dev_ex_df.to_csv(f'./out/training_curve/ex_freq_dev_preds_{run+1}_{device_id}.tsv', sep='\t', index=False, encoding='utf-8')
		
		rand_dev_ex_df = pd.DataFrame(rand_dev_predictions)
		rand_dev_ex_df.to_csv(f'./out/training_curve/ex_rand_dev_preds_{run+1}_{device_id}.tsv', sep='\t', index=False, encoding='utf-8')
		
	
	df = pd.DataFrame(results)
	df.to_csv(f'./out/training_curve/training_curve_{device_id}.tsv', sep='\t', index=False, encoding='utf-8')
	print('EXAMPLES MODELS TRAINED.\n')
	
	
	
