import pandas as pd
import argparse
from collections import Counter, defaultdict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sacremoses
from random import shuffle
import numpy as np
from transformers import AutoModel, AutoTokenizer
from matplotlib import pyplot as plt
import warnings
import lexicalClf as clf
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
	parser.add_argument("-device_id", choices=['0', '1', '2', '3'], help="Id of the GPU.")
	parser.add_argument("-data_file", default="./data.xlsx", help="The excel file containing all the annotated sense data from wiktionary.")
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
		
	tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
	
	def_lem_clf_file = './def_lem_clf.params'
	def_clf_file = './def_clf.params'
	ex_clf_file = './ex_clf.params'
	
	freq_dev_def_lem_pred_file = './freq_dev_def_lem_clf.xlsx'
	freq_dev_def_pred_file = './freq_dev_def_clf.xlsx'
	freq_dev_ex_pred_file = './freq_dev_ex_clf.xlsx'
	
	rand_dev_def_lem_pred_file = './rand_dev_def_lem_clf.xlsx'
	rand_dev_def_pred_file = './rand_dev_def_clf.xlsx'
	rand_dev_ex_pred_file = './rand_dev_ex_clf.xlsx'
	
	params = {
	"nb_epochs": 100,
	"batch_size": 16,
	"hidden_layer_size": 768,
	"patience": 1,
	"lr": 0.00001,
	"frozen": False,
	"dropout": 0.1,
	"max_seq_length": 100
	}
	
	coeff_def = 1
	coeff_ex = 1
	
	train_definitions_encoder = data.definitionEncoder(args.data_file, "train", tokenizer, use_sample=False)
	train_definitions_encoder.encode()
	freq_dev_definitions_encoder = data.definitionEncoder(args.data_file, "freq-dev", tokenizer, use_sample=False)
	freq_dev_definitions_encoder.encode()
	rand_dev_definitions_encoder = data.definitionEncoder(args.data_file, "rand-dev", tokenizer, use_sample=False)
	rand_dev_definitions_encoder.encode()

	"""
	def_lem_clf = clf.monoRankClf(params, DEVICE, use_lemma=True, bert_model_name=MODEL_NAME)
	def_lem_clf.train_clf(train_definitions_encoder, freq_dev_definitions_encoder, rand_dev_definitions_encoder, def_lem_clf_file)
	def_lem_clf = clf.monoRankClf(params, DEVICE, use_lemma=True, bert_model_name=MODEL_NAME)
	def_lem_clf.load_clf(def_lem_clf_file)
	
	train_accuracy = def_lem_clf.evaluate(train_definitions_encoder)
	freq_dev_accuracy = def_lem_clf.evaluate(freq_dev_definitions_encoder)
	rand_dev_accuracy = def_lem_clf.evaluate(rand_dev_definitions_encoder)
	
	freq_dev_predictions = def_lem_clf.predict(freq_dev_definitions_encoder)
	rand_dev_predictions = def_lem_clf.predict(rand_dev_definitions_encoder)
	
	print("train dev def lem accurcay = ", percentage(train_accuracy))
	print("freq dev def lem accurcay = ", percentage(freq_dev_accuracy))
	print("rand dev def lem accurcay = ", percentage(rand_dev_accuracy))
	print()
	
	freq_dev_def_lem_df = pd.DataFrame(freq_dev_predictions)
	freq_dev_def_lem_df.to_excel(freq_dev_def_lem_pred_file, index=False)
	
	rand_dev_def_lem_df = pd.DataFrame(rand_dev_predictions)
	rand_dev_def_lem_df.to_excel(rand_dev_def_lem_pred_file, index=False)
	
	
	"""
	def_clf = clf.monoRankClf(params, DEVICE, use_lemma=False, bert_model_name=MODEL_NAME)
	def_clf.train_clf(train_definitions_encoder, freq_dev_definitions_encoder, rand_dev_definitions_encoder, def_clf_file)
	def_clf = clf.monoRankClf(params, DEVICE, use_lemma=False, bert_model_name=MODEL_NAME)
	def_clf.load_clf(def_clf_file)
	
	train_accuracy = def_clf.evaluate(train_definitions_encoder)
	freq_dev_accuracy = def_clf.evaluate(freq_dev_definitions_encoder)
	rand_dev_accuracy = def_clf.evaluate(rand_dev_definitions_encoder)
	
	freq_dev_predictions = def_clf.predict(freq_dev_definitions_encoder)
	rand_dev_predictions = def_clf.predict(rand_dev_definitions_encoder)
	
	print("train dev def accurcay = ", percentage(train_accuracy))
	print("freq dev def accurcay = ", percentage(freq_dev_accuracy))
	print("rand dev def accurcay = ", percentage(rand_dev_accuracy))
	print()
	
	freq_dev_def_df = pd.DataFrame(freq_dev_predictions)
	freq_dev_def_df.to_excel(freq_dev_def_pred_file, index=False)
	
	rand_dev_def_df = pd.DataFrame(rand_dev_predictions)
	rand_dev_def_df.to_excel(rand_dev_def_pred_file, index=False)
	
	
	
	"""
	train_examples_encoder = data.exampleEncoder(args.data_file, "train", tokenizer, use_sample=True)
	train_examples_encoder.encode()
	freq_dev_examples_encoder = data.exampleEncoder(args.data_file, "freq-dev", tokenizer, use_sample=True)
	freq_dev_examples_encoder.encode()
	rand_dev_examples_encoder = data.exampleEncoder(args.data_file, "rand-dev", tokenizer, use_sample=True)
	rand_dev_examples_encoder.encode()
	
	ex_clf = clf.multiRankClf(params, DEVICE, use_lemma=True, dropout_rate=0.1, bert_model_name=MODEL_NAME)
	ex_clf.train_clf(train_examples_encoder, freq_dev_examples_encoder, rand_dev_examples_encoder, ex_clf_file)
	ex_clf.load_clf(ex_clf_file)
	
	train_accuracy = ex_clf.evaluate(train_examples_encoder)
	freq_dev_accuracy = ex_clf.evaluate(freq_dev_examples_encoder)
	rand_dev_accuracy = ex_clf.evaluate(rand_dev_examples_encoder)
	
	freq_dev_predictions = ex_clf.predict(freq_dev_examples_encoder)
	rand_dev_predictions = ex_clf.predict(rand_dev_examples_encoder)
	
	print("train dev accurcay = ", percentage(train_accuracy))
	print("freq dev accurcay = ", percentage(freq_dev_accuracy))
	print("rand dev accurcay = ", percentage(rand_dev_accuracy))
	print()
	
	print(freq_dev_predictions)
	print(rand_dev_predictions)
	"""
