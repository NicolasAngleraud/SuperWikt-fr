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
	parser.add_argument("--device_id", choices=['cpu', '0', '1', '2', '3'], help="Id of the GPU.")
	parser.add_argument("--sense_data_file", default="./sense_data.tsv", help="The tsv file containing all the annotated sense data from wiktionary.")
	parser.add_argument("--ex_data_file", default="./ex_data.tsv", help="The tsv file containing all the annotated examples data from wiktionary.")
	parser.add_argument('--out', required=True, help='Path to the output folder where to save the predictions from trained models.')
	parser.add_argument('--model_dir', required=True, help='Path to the folder where to save the trained models.')
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
	
	
	def_lem_clf_file = args.model_dir + '/NEW_def_lem_clf.params'
	ex_clf_file = args.model_dir + '/NEW_ex_clf.params'

	
	params_def = {
	"nb_epochs": 100,
	"batch_size": 16,
	"hidden_layer_size": 768,
	"patience": 2,
	"lr": 0.001,
	"weight_decay": 0.000005,
	"frozen": True,
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

	print('ENCODING DEFINITIONS DATA...\n')
	train_definitions_encoder = data.definitionEncoder(args.sense_data_file, args.ex_data_file, "train", tokenizer, use_sample=False)
	train_definitions_encoder.encode()
	freq_dev_definitions_encoder = data.definitionEncoder(args.sense_data_file, args.ex_data_file, "freq-dev", tokenizer, use_sample=False)
	freq_dev_definitions_encoder.encode()
	rand_dev_definitions_encoder = data.definitionEncoder(args.sense_data_file, args.ex_data_file, "rand-dev", tokenizer, use_sample=False)
	rand_dev_definitions_encoder.encode()
	freq_test_definitions_encoder = data.definitionEncoder(args.sense_data_file, args.ex_data_file, "freq-test", tokenizer, use_sample=False)
	freq_test_definitions_encoder.encode()
	rand_test_definitions_encoder = data.definitionEncoder(args.sense_data_file, args.ex_data_file, "rand-test", tokenizer, use_sample=False)
	rand_test_definitions_encoder.encode()
	print('DEFINITIONS DATA ENCODED.\n')
	
	print('TRAINING DEFINITION CLASSIFIER...\n')
	def_clf = clf.monoRankClf(params_def, DEVICE, use_lemma=True, bert_model_name=MODEL_NAME)
	def_clf.train_clf(train_definitions_encoder, freq_dev_definitions_encoder, rand_dev_definitions_encoder, def_lem_clf_file)
	print('DEFINITION CLASSIFIER TRAINED.\n')
	print('LOADING BEST DEFINITION CLASSIFIER...\n')
	def_clf = clf.monoRankClf(params_def, DEVICE, use_lemma=True, bert_model_name=MODEL_NAME)
	def_clf.load_clf(def_lem_clf_file)
	print('BEST DEFINITION CLASSIFIER LOADED.\n')
	
	
	train_accuracy = def_clf.evaluate(train_definitions_encoder)
	
	freq_dev_accuracy = def_clf.evaluate(freq_dev_definitions_encoder)
	rand_dev_accuracy = def_clf.evaluate(rand_dev_definitions_encoder)
	
	freq_dev_predictions = def_clf.predict(freq_dev_definitions_encoder)
	rand_dev_predictions = def_clf.predict(rand_dev_definitions_encoder)
	
	freq_test_accuracy = def_clf.evaluate(freq_test_definitions_encoder)
	rand_test_accuracy = def_clf.evaluate(rand_test_definitions_encoder)
	
	freq_test_predictions = def_clf.predict(freq_test_definitions_encoder)
	rand_test_predictions = def_clf.predict(rand_test_definitions_encoder)
	
	print("train def accurcay = ", percentage(train_accuracy))
	print("freq dev def accurcay = ", percentage(freq_dev_accuracy))
	print("rand dev def accurcay = ", percentage(rand_dev_accuracy))
	print("freq test def accurcay = ", percentage(freq_test_accuracy))
	print("rand test def accurcay = ", percentage(rand_test_accuracy))
	print()
	
	freq_dev_def_df = pd.DataFrame(freq_dev_predictions)
	freq_dev_def_df.to_csv(args.out+'/def_freq_dev_preds.tsv', sep='\t', index=False, encoding='utf-8')
	
	rand_dev_def_df = pd.DataFrame(rand_dev_predictions)
	rand_dev_def_df.to_csv(args.out+'/def_rand_dev_preds.tsv', sep='\t', index=False, encoding='utf-8')
	
	freq_test_def_df = pd.DataFrame(freq_test_predictions)
	freq_test_def_df.to_csv(args.out+'/def_freq_test_preds.tsv', sep='\t', index=False, encoding='utf-8')
	
	rand_test_def_df = pd.DataFrame(rand_test_predictions)
	rand_test_def_df.to_csv(args.out+'/def_rand_test_preds.tsv', sep='\t', index=False, encoding='utf-8')
	
	
	
	
	print('ENCODING EXAMPLES DATA...\n')
	train_examples_encoder = data.exampleEncoder(args.sense_data_file, args.ex_data_file, "train", tokenizer, use_sample=False, sub_corpus="wiki")
	train_examples_encoder.encode()
	freq_dev_examples_encoder = data.exampleEncoder(args.sense_data_file, args.ex_data_file, "freq-dev", tokenizer, use_sample=False, sub_corpus="wiki")
	freq_dev_examples_encoder.encode()
	rand_dev_examples_encoder = data.exampleEncoder(args.sense_data_file, args.ex_data_file, "rand-dev", tokenizer, use_sample=False, sub_corpus="wiki")
	rand_dev_examples_encoder.encode()
	freq_test_examples_encoder = data.exampleEncoder(args.sense_data_file, args.ex_data_file, "freq-test", tokenizer, use_sample=False, sub_corpus="wiki")
	freq_test_examples_encoder.encode()
	rand_test_examples_encoder = data.exampleEncoder(args.sense_data_file, args.ex_data_file, "rand-test", tokenizer, use_sample=False, sub_corpus="wiki")
	rand_test_examples_encoder.encode()
	print('EXAMPLES DATA ENCODED.\n')
	
	print('TRAINING EXAMPLE CLASSIFIER...\n')
	ex_clf = clf.multiRankClf(params, DEVICE, dropout_input=0.1, dropout_hidden=0.3, bert_model_name=MODEL_NAME)
	ex_clf.train_clf(train_examples_encoder, freq_dev_examples_encoder, rand_dev_examples_encoder, ex_clf_file)
	print('EXAMPLE CLASSIFIER TRAINED.\n')
	print('LOADING BEST EXAMPLE CLASSIFIER...\n')
	ex_clf.load_clf(ex_clf_file)
	print('BEST EXAMPLE CLASSIFIER LOADED.\n')
	
	train_accuracy = ex_clf.evaluate(train_examples_encoder)
	freq_dev_accuracy = ex_clf.evaluate(freq_dev_examples_encoder)
	rand_dev_accuracy = ex_clf.evaluate(rand_dev_examples_encoder)
	
	freq_dev_predictions = ex_clf.predict(freq_dev_examples_encoder)
	rand_dev_predictions = ex_clf.predict(rand_dev_examples_encoder)
	
	freq_test_accuracy = ex_clf.evaluate(freq_test_examples_encoder)
	rand_test_accuracy = ex_clf.evaluate(rand_test_examples_encoder)
	
	freq_test_predictions = ex_clf.predict(freq_test_examples_encoder)
	rand_test_predictions = ex_clf.predict(rand_test_examples_encoder)
	
	print("train accurcay = ", percentage(train_accuracy))
	print("freq dev accurcay = ", percentage(freq_dev_accuracy))
	print("rand dev accurcay = ", percentage(rand_dev_accuracy))
	print("freq test accurcay = ", percentage(freq_test_accuracy))
	print("rand test accurcay = ", percentage(rand_test_accuracy))
	print()
	
	freq_dev_ex_df = pd.DataFrame(freq_dev_predictions)
	freq_dev_ex_df.to_csv(args.out+'/ex_freq_dev_preds.tsv', sep='\t', index=False, encoding='utf-8')
	
	rand_dev_ex_df = pd.DataFrame(rand_dev_predictions)
	rand_dev_ex_df.to_csv(args.out+'/ex_rand_dev_preds.tsv', sep='\t', index=False, encoding='utf-8')
	
	freq_test_ex_df = pd.DataFrame(freq_test_predictions)
	freq_test_ex_df.to_csv(args.out+'/ex_freq_test_preds.tsv', sep='\t', index=False, encoding='utf-8')
	
	rand_test_ex_df = pd.DataFrame(rand_test_predictions)
	rand_test_ex_df.to_csv(args.out+'/ex_rand_test_preds.tsv', sep='\t', index=False, encoding='utf-8')
	
	print('DEFINITIONS AND EXAMPLES MODELS TRAINED.\n')
