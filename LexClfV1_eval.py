import pandas as pd
import argparse
import torch
import spacy
import mono_target_multi_rank_classifier as ex_clf
import mono_target_unique_rank_classifier as def_clf
from transformers import AutoModel, AutoTokenizer
import datetime
import warnings
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

max_seq_length = 100
max_length = 100

def get_parser_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("-device_id", choices=['0', '1', '2', '3', 'cpu'], help="Id of the GPU.")
	parser.add_argument("-data_file", default="./donnees.xlsx", help="The excel file containing all the annotated sense data from wiktionary.")
	parser.add_argument('-v', "--trace", action="store_true", help="Toggles the verbose mode. Default=False")
	args = parser.parse_args()
	return args
	
def word_to_token_rank(lst, index):
	count = 0
	for i in range(index):
		count += len(lst[i])
	return count
	

def flatten_list(lst):

	return [item for sublist in lst for item in (sublist if isinstance(sublist, list) else [sublist])]


def pad_batch(sentences, pad_id=2, max_length=100):

	max_length = max_length - 2
	pad_lengths = [ max_length - len(sent) if max_length >= len(sent) else 0 for sent in sentences ]

	padded_sentences = [ [el for el in sent] + pad_lengths[i] * [pad_id] for i, sent in enumerate(sentences) ]


	return padded_sentences



def add_special_tokens_batch(sentence, cls_id=0, sep_id=1):

	sentence_with_special_tokens = [cls_id] + [tok for tok in sentence] + [sep_id]

	return sentence_with_special_tokens


def truncate_batch_ex(sent, target_index, max_length=100):
	# Adjust max_length to account for potential special tokens
	max_length = max_length - 2

	if len(sent) <= max_length:
		return sent, target_index
	else:
		# Calculate the number of tokens to keep before and after the target_index
		half_max_length = max_length // 2
		start_index = max(0, min(len(sent) - max_length, target_index - half_max_length))
		end_index = start_index + max_length

		# Truncate the sentence
		trunc_sent = sent[start_index:end_index]

		# Adjust the target index based on truncation
		new_target_index = target_index - start_index
		# Ensure the new target index does not exceed the bounds of the truncated sentence
		new_target_index = max(0, min(new_target_index, max_length-1))

	return trunc_sent, new_target_index


def encoded_senses(dataset, datafile):

	# DEFINITIONS
	df_senses = pd.read_excel(datafile, sheet_name='senses', engine='openpyxl')
	df_senses = df_senses[df_senses['supersense'].isin(SUPERSENSES)]
	df_senses = df_senses[(df_senses['definition'] != "") & (df_senses['definition'].notna())]
	df_senses = df_senses[df_senses['set']==dataset]
	df_senses['definition_encoded'] = df_senses.apply(lambda row: tokenizer.encode(text=f"{row['lemma']} : {row['definition']}", add_special_tokens=True), axis=1)
	df_senses['supersense_encoded'] = df_senses['supersense'].apply(lambda supersense: supersense2i[supersense])
	

	# EXAMPLES
	df_examples = pd.read_excel(datafile, sheet_name='examples', engine='openpyxl')
	df_examples = df_examples[df_examples['supersense'].isin(SUPERSENSES)]
	df_examples = df_examples[df_examples['word_rank'] >= 0]
	df_examples = df_examples[(df_examples['example'] != "") & (df_examples['example'].notna())]
	df_examples = df_examples[df_examples['set']==dataset]
	df_examples['example_encoded'] = df_examples['example'].apply(lambda x: x.split(' '))
	df_examples['example_encoded'] = df_examples['example_encoded'].apply(lambda x: [word.replace('##', ' ') for word in x])
	df_examples['example_encoded'] = df_examples['example_encoded'].apply(lambda example: [tokenizer(word, add_special_tokens=False)['input_ids'] for word in example])
	df_examples['token_rank'] = df_examples.apply(lambda row: word_to_token_rank(row['example_encoded'], row['word_rank']), axis=1)
	df_examples['example_encoded'] = df_examples['example_encoded'].apply(lambda x: flatten_list(x))
	df_examples['example_encoded'], df_examples['token_rank'] = zip(*df_examples.apply(lambda row: truncate_batch_ex(row['example_encoded'], row['token_rank'], max_length=100), axis=1))
	# df_examples['example_encoded'] = df_examples['example_encoded'].apply(lambda example: pad_batch(example, pad_id=2, max_length=max_length))
	df_examples['example_encoded'] = df_examples['example_encoded'].apply(lambda example: add_special_tokens_batch(example))
	df_examples['token_rank'] = df_examples['token_rank'].apply(lambda rk: rk+1)
	df_examples['supersense_encoded'] = df_examples['supersense'].apply(lambda supersense: supersense2i[supersense])
	
	# SENSES
	senses_ids = df_senses['sense_id'].tolist()
	
	return senses_ids, df_senses.head, df_examples.head
	


if __name__ == '__main__':
	args = get_parser_args()
	
	# DEVICE setup
	device_id = args.device_id
	
	if args.device_id == 'cpu': DEVICE = torch.device(args.device_id)
	else:
		if torch.cuda.is_available():
			DEVICE = torch.device("cuda:" + args.device_id)
	
	def_clf_file = "./clfs/LexClfV1_def.params"
	ex_clf_file = "./clfs/LexClfV1_ex.params"
	
	MODEL_NAME = "flaubert/flaubert_large_cased"
	tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
	datafile = args.data_file

	patience = 1
	frozen = False
	max_seq_length = 100
	lr_ex = 0.00001
	dropout = 0.1
	hidden_layer_size = 768
	token_rank = 1
	
	examples = pd.read_excel(datafile, sheet_name='examples', engine='openpyxl')
	senses = pd.read_excel(datafile, sheet_name='senses', engine='openpyxl')
	

	for def_weight in [0.5, 0.6, 0.7, 0.8, 0.9, 1]:
		
		eval_data = []
		ex_weight = 1 - def_weight
		
		freq_dev_senses_ids, freq_dev_df_senses, freq_dev_df_examples = encoded_senses(dataset='freq-dev', datafile=args.data_file)
		# rand_dev_senses_ids, rand_dev_df_senses, rand_dev_df_examples = encoded_senses(dataset='rand-dev', datafile=args.data_file)
		
		print()
		print()
		print("DEF WEIGHT : ", def_weight)
		print("EX WEIGHT : ", ex_weight)
		print()

		params = {key: value for key, value in LPARAMETERS.items()}
		params['lr'] = lr_ex
		params['patience'] = patience
		params['max_seq_length'] = max_seq_length
		params['frozen'] = frozen
		params['dropout'] = dropout
		params['hidden_layer_size'] = hidden_layer_size
		params['token_rank'] = token_rank
		
		def_classifier = def_clf.SupersenseTagger(params, DEVICE)
		def_classifier.load_state_dict(torch.load(def_clf_file))
		
		ex_classifier = ex_clf.SupersenseTagger(params, DEVICE)
		ex_classifier.load_state_dict(torch.load(ex_clf_file))
		
		# FREQ-DEV
		freq_dev_df_senses['probs'] = freq_dev_df_senses['definition_encoded'].apply(def_classifier.forward_encoding)
		freq_dev_df_examples['probs'] = freq_dev_df_examples.apply(lambda row: ex_classifier.forward_encoding(row['example_encoded'], row['token_rank']), axis=1)
		
		freq_dev_df_senses['probs'] = freq_dev_df_senses['probs'].apply(lambda x: x.squeeze())
		freq_dev_df_examples['probs'] = freq_dev_df_examples['probs'].apply(lambda x: x.squeeze())
		
		
		# RAND-DEV
		# rand_dev_df_senses['probs'] = rand_dev_df_senses['definition_encoded'].apply(lambda x: def_classifier.forward_encoding(x))
		# rand_dev_df_examples['probs'] = rand_dev_df_examples.apply(lambda row: ex_classifier.forward_encoding(row['example_encoded'], row['token_rank']), axis=1)
		
		freq_dev_df_examples.reset_index(drop=True, inplace=True)
		freq_dev_df_senses.reset_index(drop=True, inplace=True)
		
		# SENSES PRED
		for sense_id in freq_dev_senses_ids:
			sense_eval_data = {}
			sense_eval_data['sense_id'] = sense_id
			sense_eval_data['dataset'] = 'freq-dev'
			sense_eval_data['gold'] = freq_dev_df_senses[freq_dev_df_senses['sense_id'] == sense_id]['supersense']

			if not freq_dev_df_examples[freq_dev_df_examples['sense_id'] == sense_id]['probs'].dropna().empty:
				example_score = torch.mean(torch.stack(freq_dev_df_examples[freq_dev_df_examples['sense_id'] == sense_id]['probs'].dropna().tolist()), dim=0).detach()
			else:
				example_score = 0
				
			definition_score = torch.tensor([prob.item() for prob in freq_dev_df_senses[freq_dev_df_senses['sense_id'] == sense_id]['probs'].iloc[0]])
			
			sense_eval_data['pred'] = SUPERSENSES[torch.argmax(torch.add(def_weight * definition_score, ex_weight * example_score), dim=1) ]
			
			sense_eval_data['definition'] = freq_dev_df_senses[freq_dev_df_senses['sense_id'] == sense_id]['definition']
			for i in range(23): sense_eval_data[f'example_{i+1}'] =  freq_dev_df_senses[freq_dev_df_senses['sense_id'] == sense_id][f'example_{i+1}']
			
			eval_data.append(sense_eval_data)
			
		
		
		"""
		for sense_id in rand_dev_senses_ids:
			sense_eval_data = {}
			sense_eval_data['sense_id'] = sense_id
			sense_eval_data['dataset'] = 'rand-dev'
			sense_eval_data['gold'] = rand_dev_df_senses[rand_dev_df_senses['sense_id'] == sense_id]['supersense']
			
			rand_dev_df_examples.reset_index(drop=True, inplace=True)
			rand_dev_df_senses.reset_index(drop=True, inplace=True)
			
			if not rand_dev_df_examples[rand_dev_df_examples['sense_id'] == sense_id]['probs'].dropna().empty:
				example_score = torch.mean(torch.stack(rand_dev_df_examples[rand_dev_df_examples['sense_id'] == sense_id]['probs'].tolist()), dim=0)
			else:
				example_score = 0
			definition_score = (rand_dev_df_senses[rand_dev_df_senses['sense_id'] == sense_id]['probs'])
			sense_eval_data['pred'] = SUPERSENSES[torch.argmax(def_weight * definition_score + ex_weight * example_score, dim=1) ]
			sense_eval_data['definition'] = rand_dev_df_senses[rand_dev_df_senses['sense_id'] == sense_id]['definition']
			for i in range(23): sense_eval_data[f'example_{i+1}'] =  rand_dev_df_senses[rand_dev_df_senses['sense_id'] == sense_id][f'example_{i+1}']
			eval_data.append(sense_eval_data)
		"""
		
		df_eval = pd.DataFrame(eval_data)
		df_eval.to_excel(f'./LexClfV1_Wdef_{def_weight*100}_freq_dev.xlsx', index=False)
		
		
	print("PROCESS DONE.\n")
