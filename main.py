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
from huggingface_hub import HfApi, login
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
flaubert_fr_sem = "nangleraud/flaubert-fr-sem-nom-def"

def percentage(decimal):
    percentage = decimal * 100
    return f"{percentage:.2f}%"

def lr_id(lr):
	lr = str(lr)
	if 'e-' in lr:
		return lr[0] + "_e_minus_" + lr.split('e-')[1].strip('0')
	return lr.split('.')[1][-1] + "_e_minus_" + str(len(lr.split('.')[1]))

def get_parser_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("-device_id", choices=['cpu', '0', '1', '2', '3'], help="Id of the GPU.")
	parser.add_argument("-data_file", default="./data.xlsx", help="The excel file containing all the annotated sense data from wiktionary.")
	parser.add_argument("-batch_size", choices=['1', '2', '4', '8', '16', '32', '64'], help="batch size for the classifier.")
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
	
	API_TOKEN = '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'
	
	def_lem_clf_file = './def_lem_clf.params'
	#def_clf_file = './def_clf.params'
	ex_clf_file = './ex_clf.params'
	#corpus_clf_file = './corpus_clf.params'
	
	params_def = {
	"nb_epochs": 100,
	"batch_size": 1,
	"hidden_layer_size": 768,
	"patience": 2,
	"lr": 0.000005,
	"weight_decay": 0.001,
	"frozen": False,
	"max_seq_length": 100
	}
	
	params_ex = {
	"nb_epochs": 100,
	"batch_size": 1,
	"hidden_layer_size": 768,
	"patience": 2,
	"lr": 0.000005,
	"weight_decay": 0.001,
	"frozen": False,
	"max_seq_length": 100
	}
	
	"""
	# Load your fine-tuned model
	def_lem_clf = clf.monoRankClf(params_def, DEVICE, use_lemma=True, bert_model_name=MODEL_NAME)
	def_lem_clf.load_clf(def_lem_clf_file)
	bert_model = def_lem_clf.bert_model
	tokenizer = AutoTokenizer.from_pretrained('nangleraud/flaubert-fr-sem-nom-def')

	# Set BERT to evaluation mode and disable gradients
	bert_model.eval()
	for param in bert_model.parameters():
		param.requires_grad = False

	def get_embeddings(texts, tokenizer, model, batch_size=32):
		all_embeddings = []
		with torch.no_grad():
		    for i in range(0, len(texts), batch_size):
		        batch_texts = texts[i:i+batch_size]
		        encoded_inputs = tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt").to(DEVICE)
		        input_ids = encoded_inputs['input_ids']
		        attention_mask = encoded_inputs['attention_mask']
		        
		        # Get the last hidden states from BERT
		        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
		        last_hidden_states = outputs.last_hidden_state  # Shape: (batch_size, sequence_length, hidden_size)
		        
		        # Use the [CLS] token representation as the embedding
		        cls_embeddings = last_hidden_states[:, 0, :]  # Shape: (batch_size, hidden_size)
		        
		        all_embeddings.append(cls_embeddings)

		all_embeddings = torch.cat(all_embeddings, dim=0).to(DEVICE)
		return all_embeddings
	
	datafile = "./data.xlsx"
	df_definitions = pd.read_excel(datafile, sheet_name='senses', engine='openpyxl')
	df_definitions = df_definitions[df_definitions['supersense'].isin(SUPERSENSES)]
	df_definitions = df_definitions[(df_definitions['definition'] != "") & (df_definitions['definition'].notna())]
	df_definitions['lemma'] = df_definitions['lemma'].str.replace('_', ' ')
	
	# Filter definitions based on set column values
	train_definitions = df_definitions[df_definitions['set'] == 'train']
	test_definitions = df_definitions[df_definitions['set'].isin(['freq-dev', 'rand-dev'])]

	# Format lemma: definition for each set
	def format_definitions(df):
		formatted_definitions = []
		for index, row in df.iterrows():
		    lemma = row['lemma']
		    definition = row['definition']
		    formatted_definitions.append(f"{lemma}: {definition}")
		return formatted_definitions

	# Create X_train and X_test with lemma: definition format
	X_train = format_definitions(train_definitions)
	X_test = format_definitions(test_definitions)

	# Get embeddings
	train_embeddings = get_embeddings(X_train, tokenizer, bert_model)
	test_embeddings = get_embeddings(X_test, tokenizer, bert_model)
	
	# Save embeddings as safetensors
	torch.save(train_embeddings, './train_embeddings.pt')
	torch.save(test_embeddings, './test_embeddings.pt')
	"""

	# Load embeddings from safetensors
	train_embeddings_loaded = torch.load('./train_embeddings.pt')
	test_embeddings_loaded = torch.load('./test_embeddings.pt')
	
	print(train_embeddings_loaded.shape)
	print(test_embeddings_loaded.shape)
		
	"""
	freq_dev_def_lem_pred_file = './freq_dev_def_lem_clf.xlsx'
	freq_dev_def_pred_file = './freq_dev_def_clf.xlsx'
	freq_dev_ex_pred_file = './freq_dev_ex_clf.xlsx'
	corpus_dev_pred_file = './dev_corpus_clf.xlsx'
	freq_dev_def_ex_pred_file = './freq_dev_def_ex_clf.xlsx'
	
	rand_dev_def_lem_pred_file = './rand_dev_def_lem_clf.xlsx'
	rand_dev_def_pred_file = './rand_dev_def_clf.xlsx'
	rand_dev_ex_pred_file = './rand_dev_ex_clf.xlsx'
	rand_dev_def_ex_pred_file = './rand_dev_def_ex_clf.xlsx'
	"""
	
	#freq_dev_sense_encoder = data.senseEncoder(args.data_file, "freq-dev", tokenizer, use_sample=False)

	#rand_dev_sense_encoder = data.senseEncoder(args.data_file, "rand-dev", tokenizer, use_sample=False)
	
	'''
	wiki_def_file = "./wiktionnaire.xlsx"
	wiki_example_file = "./wiktionnaire_exemples.xlsx"
	
	wiki_pred_file = "./wiktionary_predictions.xlsx"
	
	wiki_encoder = data.wikiEncoder(def_datafile=wiki_def_file, ex_datafile=wiki_example_file, tokenizer=tokenizer, use_sample=False, sample_size=1000)
	'''
	"""
	for definition_with_lemma_encoded, bert_input_examples, tg_trks_examples, sense_id, lemma in wiki_encoder.encoded_senses(DEVICE):
		print(lemma)
		print(sense_id)
		if definition_with_lemma_encoded is not None: print(tokenizer.decode(definition_with_lemma_encoded[0], skip_special_tokens=True))
		for i, ex in enumerate(bert_input_examples): print(tokenizer.decode(ex.squeeze()[tg_trks_examples[i]]))
		print()
		print()
	"""
	

	'''
	coeff_ex = 0.68
	
	coeff_def = 0.80
	
	lex_clf = clf.lexicalClf_V1(params_def, params_ex, DEVICE, coeff_ex, coeff_def)
	lex_clf.load_clf(def_lem_clf_file, ex_clf_file)
	
	wiktionary_predictions = lex_clf.predict_wiki(wiki_encoder)
	
	wiki_df = pd.DataFrame(wiktionary_predictions)
	wiki_df.to_excel(wiki_pred_file, index=False)
	
	
	
	
	
	#freq_dev_predictions = lex_clf.predict(freq_dev_sense_encoder)
	#rand_dev_predictions = lex_clf.predict(rand_dev_sense_encoder)
	
	#freq_dev_def_ex_df = pd.DataFrame(freq_dev_predictions)
	#freq_dev_def_ex_df.to_excel(freq_dev_def_ex_pred_file, index=False)
	
	#rand_dev_def_ex_df = pd.DataFrame(rand_dev_predictions)
	#rand_dev_def_ex_df.to_excel(rand_dev_def_ex_pred_file, index=False)
	'''
	"""
	params = {
	"nb_epochs": 100,
	"batch_size": 16,
	"hidden_layer_size": 768,
	"patience": 2,
	"lr": 0.000005,
	"weight_decay": 0.001,
	"frozen": False,
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
	
	print("train def lem accurcay = ", percentage(train_accuracy))
	print("freq dev def lem accurcay = ", percentage(freq_dev_accuracy))
	print("rand dev def lem accurcay = ", percentage(rand_dev_accuracy))
	print()
	
	freq_dev_def_lem_df = pd.DataFrame(freq_dev_predictions)
	freq_dev_def_lem_df.to_excel(freq_dev_def_lem_pred_file, index=False)
	
	rand_dev_def_lem_df = pd.DataFrame(rand_dev_predictions)
	rand_dev_def_lem_df.to_excel(rand_dev_def_lem_pred_file, index=False)
	
	
	
	def_clf = clf.monoRankClf(params, DEVICE, use_lemma=False, bert_model_name=MODEL_NAME)
	def_clf.train_clf(train_definitions_encoder, freq_dev_definitions_encoder, rand_dev_definitions_encoder, def_clf_file)
	def_clf = clf.monoRankClf(params, DEVICE, use_lemma=False, bert_model_name=MODEL_NAME)
	def_clf.load_clf(def_clf_file)
	
	train_accuracy = def_clf.evaluate(train_definitions_encoder)
	freq_dev_accuracy = def_clf.evaluate(freq_dev_definitions_encoder)
	rand_dev_accuracy = def_clf.evaluate(rand_dev_definitions_encoder)
	
	freq_dev_predictions = def_clf.predict(freq_dev_definitions_encoder)
	rand_dev_predictions = def_clf.predict(rand_dev_definitions_encoder)
	
	print("train def accurcay = ", percentage(train_accuracy))
	print("freq dev def accurcay = ", percentage(freq_dev_accuracy))
	print("rand dev def accurcay = ", percentage(rand_dev_accuracy))
	print()
	
	freq_dev_def_df = pd.DataFrame(freq_dev_predictions)
	freq_dev_def_df.to_excel(freq_dev_def_pred_file, index=False)
	
	rand_dev_def_df = pd.DataFrame(rand_dev_predictions)
	rand_dev_def_df.to_excel(rand_dev_def_pred_file, index=False)
	"""
	
	
	"""
	params = {
	"nb_epochs": 100,
	"batch_size": 16,
	"hidden_layer_size": 768,
	"patience": 2,
	"lr": 0.00001,
	"weight_decay": 0.001,
	"frozen": False,
	"max_seq_length": 100
	}
	
	
	train_examples_encoder = data.exampleEncoder(args.data_file, "train", tokenizer, use_sample=False, sub_corpus="wiki")
	train_examples_encoder.encode()
	freq_dev_examples_encoder = data.exampleEncoder(args.data_file, "freq-dev", tokenizer, use_sample=False, sub_corpus="wiki")
	freq_dev_examples_encoder.encode()
	rand_dev_examples_encoder = data.exampleEncoder(args.data_file, "rand-dev", tokenizer, use_sample=False, sub_corpus="wiki")
	rand_dev_examples_encoder.encode()
	
	ex_clf = clf.multiRankClf(params, DEVICE, dropout_input=0.1, dropout_hidden=0.3, bert_model_name=MODEL_NAME)
	ex_clf.train_clf(train_examples_encoder, freq_dev_examples_encoder, rand_dev_examples_encoder, ex_clf_file)
	ex_clf.load_clf(ex_clf_file)
	
	train_accuracy = ex_clf.evaluate(train_examples_encoder)
	freq_dev_accuracy = ex_clf.evaluate(freq_dev_examples_encoder)
	rand_dev_accuracy = ex_clf.evaluate(rand_dev_examples_encoder)
	
	freq_dev_predictions = ex_clf.predict(freq_dev_examples_encoder)
	rand_dev_predictions = ex_clf.predict(rand_dev_examples_encoder)
	
	print("train accurcay = ", percentage(train_accuracy))
	print("freq dev accurcay = ", percentage(freq_dev_accuracy))
	print("rand dev accurcay = ", percentage(rand_dev_accuracy))
	print()
	
	freq_dev_ex_df = pd.DataFrame(freq_dev_predictions)
	freq_dev_ex_df.to_excel(freq_dev_ex_pred_file, index=False)
	
	rand_dev_ex_df = pd.DataFrame(rand_dev_predictions)
	rand_dev_ex_df.to_excel(rand_dev_ex_pred_file, index=False)
	"""
	
	
	
	"""
	train_examples_encoder = data.corpusEncoder(args.data_file, "train", tokenizer, "frsemcor", use_sample=False)
	train_examples_encoder.encode()
	dev_examples_encoder = data.corpusEncoder(args.data_file, "protect-frsemcor-dev", tokenizer, "frsemcor", use_sample=False)
	dev_examples_encoder.encode()
	
	for run in range(1):
		for i, lr in enumerate([0.00005, 0.00001]):
	
			params = {
			"nb_epochs": 100,
			"batch_size": 16,
			"hidden_layer_size": 768,
			"patience": 2,
			"lr": lr,
			"weight_decay": 0.001,
			"frozen": False,
			"max_seq_length": 100
			}
			
			
			print(f"LR {lr} - RUN {run+1}")
			print()
			
			corpus_clf = clf.multiRankClf(params, DEVICE, dropout_input=0.1, dropout_hidden=0.3, bert_model_name=MODEL_NAME)
			corpus_clf.train_contextual_clf(train_examples_encoder, dev_examples_encoder, corpus_clf_file)
			corpus_clf.load_clf(corpus_clf_file)
			
			train_accuracy = corpus_clf.evaluate(train_examples_encoder)
			dev_accuracy = corpus_clf.evaluate(dev_examples_encoder)
			
			dev_predictions = corpus_clf.predict(dev_examples_encoder)
			
			print("train accurcay = ", percentage(train_accuracy))
			print("dev accurcay = ", percentage(dev_accuracy))
			print()
			print()
			
			dev_df = pd.DataFrame(dev_predictions)
			dev_df.to_excel(corpus_dev_pred_file.replace('.xlsx', f"_lr_{lr_id(lr)}_run_{run+1}.xlsx"), index=False)
	"""
			
	'''
	# Load your fine-tuned model
	def_lem_clf = clf.monoRankClf(params_def, DEVICE, use_lemma=True, bert_model_name=MODEL_NAME)
	def_lem_clf.load_clf(def_lem_clf_file)

	# Log in to Hugging Face
	login(token=API_TOKEN)

	# Initialize the API
	api = HfApi()

	# Repository name
	repo_name = "nangleraud/flaubert-fr-sem-nom-def"

	# Create a new repository or use an existing one
	api.create_repo(repo_id=repo_name, exist_ok=True)

	def_lem_clf.bert_model.push_to_hub(repo_name)
	tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
	tokenizer.push_to_hub(repo_name)
	'''
