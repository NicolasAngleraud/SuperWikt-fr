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
	
	train_senses_encoder = data.senseEncoder(args.data_file, "train", tokenizer)
	
	i = 0
	for definitions_with_lemma_encoded, definitions_without_lemma_encoded, bert_input_examples, tg_trks_examples, supersense_encoded, sense_id, lemma in train_senses_encoder.encoded_senses(device=DEVICE):
	
		if i <= 1:
			print(definitions_with_lemma_encoded)
			print()
			print(definitions_without_lemma_encoded)
			print()
			for bert_input_example in bert_input_examples: print(bert_input_example)
			print()
			print(tg_trks_examples)
			print()
			print(supersense_encoded)
			print()
			print(sense_id)
			print()
			print(lemma)
			print()
			print()
			i+=1
		else: break
	
