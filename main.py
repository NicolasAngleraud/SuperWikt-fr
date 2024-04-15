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
	
	train_examples_encoder = data.exampleEncoder(args.data_file, "train", tokenizer)
	train_examples_encoder.encode()
	
	freq_dev_examples_encoder = data.exampleEncoder(args.data_file, "freq-dev", tokenizer)
	freq_dev_examples_encoder.encode()
	
	rand_dev_examples_encoder = data.exampleEncoder(args.data_file, "rand-dev", tokenizer)
	rand_dev_examples_encoder.encode()
	
	i = 0
	for b_bert_input, b_tg_trks, b_supersenses_encoded, b_senses_ids, b_lemmas in train_examples_encoder.make_batches(batch_size=2, device=DEVICE, shuffle_data=True):
	
		if i <= 1:
			print(b_bert_input)
			print()
			print(b_tg_trks)
			print()
			print(b_supersenses_encoded)
			print()
			print(b_senses_ids)
			print()
			print(b_lemmas)
			print()
			print()
			i+=1
		else: break
	
