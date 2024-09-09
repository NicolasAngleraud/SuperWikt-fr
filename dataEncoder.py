import pandas as pd
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
NB_CLASSES = len(supersense2i)
MODEL_NAME = "flaubert/flaubert_large_cased"
PADDING_TOKEN_ID = 2
MAX_LENGTH = 100


def flatten_list(lst):
    return [item for sublist in lst for item in (sublist if isinstance(sublist, list) else [sublist])]

def token_rank(lst, index):
	count = 0
	for i in range(index):
		count += len(lst[i])
	return count


class Encoder:
	
	def __init__(self, sense_datafile, ex_datafile, dataset, tokenizer, use_sample=False, sample_size=32):
	
		self.tokenizer = tokenizer
		self.sense_datafile = sense_datafile
		self.ex_datafile = ex_datafile
		
		self.df_definitions = pd.read_csv(sense_datafile, sep='\t')
		self.df_definitions = self.df_definitions[self.df_definitions['supersense'].isin(SUPERSENSES)]
		self.df_definitions = self.df_definitions[(self.df_definitions['definition'] != "") & (self.df_definitions['definition'].notna())]
		self.df_definitions['lemma'] = self.df_definitions['lemma'].str.replace('_', ' ')
		
		self.df_examples = pd.read_csv(ex_datafile, sep='\t')
		self.df_examples = self.df_examples[self.df_examples['supersense'].isin(SUPERSENSES)]
		self.df_examples = self.df_examples[self.df_examples['word_rank'] >= 0]
		self.df_examples = self.df_examples[(self.df_examples['example'] != "") & (self.df_examples['example'].notna())]
		self.df_examples['lemma'] = self.df_examples['lemma'].str.replace('_', ' ')
		
		self.df_definitions = self.df_definitions[self.df_definitions['set']==dataset]
		self.df_examples = self.df_examples[self.df_examples['set']==dataset]
		
		if use_sample:
			self.df_definitions = self.df_definitions[self.df_definitions['set']==dataset].sample(sample_size)
			self.df_examples = self.df_examples[self.df_examples['set']==dataset].sample(sample_size)
	
	def truncate(self, sentences, word_ranks=[], max_length=100):

		max_length = max_length - 2

		trunc_sentences = []
		new_word_ranks = []
		
		if len(word_ranks) == 0: word_ranks = [0]*len(sentences)

		for sent, target_index in zip(sentences, word_ranks):
			if len(sent) <= max_length:

				trunc_sentences.append(sent)
				new_word_ranks.append(target_index) 
			else:

				half_max_length = max_length // 2
				start_index = max(0, min(len(sent) - max_length, target_index - half_max_length))
				end_index = start_index + max_length


				trunc_sent = sent[start_index:end_index]
				trunc_sentences.append(trunc_sent)


				new_target_index = target_index - start_index

				new_target_index = max(0, min(new_target_index, max_length-1))
				new_word_ranks.append(new_target_index)

		return trunc_sentences, new_word_ranks



	def pad(self, sentences, pad_id=2, max_length=100):

		max_length = max_length - 2
		pad_lengths = [ max_length - len(sent) if max_length >= len(sent) else 0 for sent in sentences ]

		padded_sentences = [ [el for el in sent] + pad_lengths[i] * [pad_id] for i, sent in enumerate(sentences) ]
		
		return padded_sentences
		
	
	
	def add_special_tokens(self, sentences, ranks=[], cls_id=0, sep_id=1):
	
		sentences_with_special_tokens = [ [cls_id] + [tok for tok in sent] + [sep_id] for sent in sentences ]
		if ranks: rks = [rk + 1 for rk in ranks]
		else: rks = []

		return sentences_with_special_tokens, rks
	
	
	def encode(self):
		pass
	
	
	def make_batches(self):
		pass
	
	
	def shuffle_data(self):
		pass
	

class definitionEncoder(Encoder):
	
	def __init__(self, sense_datafile, ex_datafile, dataset, tokenizer, use_sample=False, sample_size=32):
		super().__init__(sense_datafile, ex_datafile, dataset, tokenizer, use_sample, sample_size)

		
	
	def encode(self):
		df_definitions = self.df_definitions
		df_examples = self.df_examples
		
		tokenizer = self.tokenizer
		
		definitions = df_definitions['definition'].tolist()
		supersenses = df_definitions['supersense'].tolist()
		lemmas = df_definitions['lemma'].tolist()
		senses_ids = df_definitions['sense_id'].tolist()
		
		self.length = len(supersenses)
		
		definitions_with_lemma_encoded = [tokenizer.encode(text=f"{lemma.replace('_',' ')} : {definition}", add_special_tokens=False) for definition, lemma in zip(definitions, lemmas)]
		definitions_without_lemma_encoded = [tokenizer.encode(text=definition, add_special_tokens=False) for definition, lemma in zip(definitions, lemmas)]
		
		definitions_with_lemma_encoded, _ = self.truncate(definitions_with_lemma_encoded)
		definitions_with_lemma_encoded = self.pad(definitions_with_lemma_encoded, pad_id=2)
		definitions_with_lemma_encoded, _ = self.add_special_tokens(definitions_with_lemma_encoded, cls_id=0, sep_id=1)
		
		definitions_without_lemma_encoded, _ = self.truncate(definitions_without_lemma_encoded)
		definitions_without_lemma_encoded = self.pad(definitions_without_lemma_encoded, pad_id=2)
		definitions_without_lemma_encoded, _ = self.add_special_tokens(definitions_without_lemma_encoded, cls_id=0, sep_id=1)
		
		supersenses_encoded = [supersense2i[supersense] for supersense in supersenses]
		
		self.definitions_with_lemma_encoded = definitions_with_lemma_encoded
		self.definitions_without_lemma_encoded = definitions_without_lemma_encoded
		self.supersenses_encoded = supersenses_encoded
		self.lemmas = lemmas
		self.senses_ids = senses_ids
	
		
	def shuffle_data(self):
	
		data = zip(self.definitions_with_lemma_encoded, self.definitions_without_lemma_encoded, self.supersenses_encoded, self.senses_ids, self.lemmas)
		data = list(data)
		shuffle(data)
		definitions_with_lemma_encoded, definitions_without_lemma_encoded, supersenses_encoded, senses_ids, lemmas = zip(*data)
		
		self.definitions_with_lemma_encoded = definitions_with_lemma_encoded
		self.definitions_without_lemma_encoded = definitions_without_lemma_encoded
		self.supersenses_encoded = supersenses_encoded
		self.lemmas = lemmas
		self.senses_ids = senses_ids
	
	
	def truncate_senses(self, k):
		self.definitions_with_lemma_encoded = self.definitions_with_lemma_encoded[:k]
		self.definitions_without_lemma_encoded = self.definitions_without_lemma_encoded[:k]
		self.supersenses_encoded = self.supersenses_encoded[:k]
		self.lemmas = self.lemmas[:k]
		self.senses_ids = self.senses_ids[:k]
		
	def make_batches(self, batch_size, device, shuffle_data=False):
		if shuffle_data: self.shuffle_data()

		k = 0
		while k < len(self.supersenses_encoded):

			start_idx = k
			end_idx = k+batch_size if k+batch_size <= len(self.supersenses_encoded) else len(self.supersenses_encoded)
			k += batch_size

			b_definitions_with_lemma_encoded = self.definitions_with_lemma_encoded[start_idx:end_idx]
			b_definitions_without_lemma_encoded = self.definitions_without_lemma_encoded[start_idx:end_idx]
			b_supersenses_encoded = self.supersenses_encoded[start_idx:end_idx]
			b_senses_ids = self.senses_ids[start_idx:end_idx]
			b_lemmas = self.lemmas[start_idx:end_idx]

			b_definitions_with_lemma_encoded = torch.tensor(b_definitions_with_lemma_encoded).to(device)
			b_definitions_without_lemma_encoded = torch.tensor(b_definitions_without_lemma_encoded).to(device)
			b_supersenses_encoded = torch.tensor(b_supersenses_encoded).to(device)
			

			yield b_definitions_with_lemma_encoded, b_definitions_without_lemma_encoded, b_supersenses_encoded, b_senses_ids, b_lemmas
		


class exampleEncoder(Encoder):
	def __init__(self, sense_datafile, ex_datafile, dataset, tokenizer, use_sample=False, sample_size=32, sub_corpus=None):
		super().__init__(sense_datafile, ex_datafile, dataset, tokenizer, use_sample, sample_size)
		
				
		if sub_corpus:
			if sub_corpus == "wiki":
				self.df_definitions = self.df_definitions[~self.df_definitions["sense_id"].str.contains('frsemcor')]
				self.df_examples = self.df_examples[~self.df_examples["sense_id"].str.contains('frsemcor')]
			if sub_corpus == "frsemcor":
				self.df_definitions = self.df_definitions[self.df_definitions["sense_id"].str.contains('frsemcor')]
				self.df_examples = self.df_examples[self.df_examples["sense_id"].str.contains('frsemcor')]
		
	
	def encode(self):
		df_definitions = self.df_definitions
		df_examples = self.df_examples
		
		tokenizer = self.tokenizer
		
		examples = [ x.split(' ') for x in df_examples['example'].tolist() ]
		for example in examples:
			for x in example:
				x = x.replace('##', ' ')
		
		supersenses = df_examples['supersense'].tolist()
		senses_ids = df_examples['sense_id'].tolist()
		lemmas = df_examples['lemma'].tolist()
		ranks = df_examples['word_rank'].tolist()
		
		self.length = len(supersenses)
		
		sents_encoded = [ tokenizer(word, add_special_tokens=False)['input_ids'] for word in examples ]
		
		tg_trks = [token_rank(sent, rank) for sent, rank in zip(sents_encoded, ranks)]
		bert_input_raw = [ flatten_list(sent) for sent in sents_encoded ]
		bert_input_raw, tg_trks = self.truncate(bert_input_raw, tg_trks)
		bert_input_raw = self.pad(bert_input_raw, pad_id=2)
		bert_input, tg_trks = self.add_special_tokens(bert_input_raw, tg_trks, cls_id=0, sep_id=1)
		supersenses_encoded = [supersense2i[supersense] for supersense in supersenses]

		self.bert_input = bert_input
		self.tg_trks = tg_trks
		self.supersenses_encoded = supersenses_encoded
		self.senses_ids = senses_ids
		self.lemmas = lemmas
		
	def shuffle_data(self):
	
		data = zip(self.bert_input, self.tg_trks, self.supersenses_encoded, self.senses_ids, self.lemmas)
		data = list(data)
		shuffle(data)
		bert_input, tg_trks, supersenses_encoded, senses_ids, lemmas = zip(*data)
		
		self.bert_input = bert_input
		self.tg_trks = tg_trks
		self.supersenses_encoded = supersenses_encoded
		self.senses_ids = senses_ids
		self.lemmas = lemmas
		
		
	def make_batches(self, batch_size, device, shuffle_data=False):
		if shuffle_data: self.shuffle_data()

		k = 0
		while k < len(self.supersenses_encoded):

			start_idx = k
			end_idx = k+batch_size if k+batch_size <= len(self.supersenses_encoded) else len(self.supersenses_encoded)
			k += batch_size

			b_bert_input = self.bert_input[start_idx:end_idx]
			b_tg_trks = self.tg_trks[start_idx:end_idx]
			b_supersenses_encoded = self.supersenses_encoded[start_idx:end_idx]
			b_senses_ids = self.senses_ids[start_idx:end_idx]
			b_lemmas = self.lemmas[start_idx:end_idx]

			b_bert_input = torch.tensor(b_bert_input).to(device)
			b_tg_trks = torch.tensor(b_tg_trks).to(device)
			b_supersenses_encoded = torch.tensor(b_supersenses_encoded).to(device)

			yield b_bert_input, b_tg_trks, b_supersenses_encoded, b_senses_ids, b_lemmas



class senseEncoder(Encoder):
	def __init__(self, sense_datafile, ex_datafile, dataset, tokenizer, use_sample=False, sample_size=32):
		super().__init__(sense_datafile, ex_datafile, dataset, tokenizer, use_sample, sample_size)
		self.senses_ids = self.df_definitions['sense_id'].tolist()
	
	def encoded_senses(self, device):
		
		df_definitions = self.df_definitions
		df_examples = self.df_examples
		
		tokenizer = self.tokenizer
		
		for sense_id in self.senses_ids:
			
			definition = df_definitions[df_definitions['sense_id'] == sense_id]["definition"].iloc[0]
			lemma = df_definitions[df_definitions['sense_id'] == sense_id]["lemma"].iloc[0]
			supersense = df_definitions[df_definitions['sense_id'] == sense_id]["supersense"].iloc[0]
			
			definition_with_lemma_encoded = tokenizer.encode(text=f"{lemma.replace('_',' ')} : {definition}", add_special_tokens=True, return_tensors='pt')
			
			definition_without_lemma_encoded = tokenizer.encode(text=definition, add_special_tokens=True, return_tensors='pt')
			
			examples = df_examples[df_examples['sense_id'] == sense_id]['example'].tolist()
			word_ranks = df_examples[df_examples['sense_id'] == sense_id]['word_rank'].tolist()
			
			examples = [ x.split(' ') for x in examples ]
			for example in examples:
				for x in example:
					x = x.replace('##', ' ')

			sents_encoded = [ tokenizer(word, add_special_tokens=False)['input_ids'] for word in examples ]
			
			tg_trks = [token_rank(sent, rank) for sent, rank in zip(sents_encoded, word_ranks)]
			bert_input_raw = [ flatten_list(sent) for sent in sents_encoded ]
			bert_input_examples, tg_trks_examples = self.add_special_tokens(bert_input_raw, tg_trks, cls_id=0, sep_id=1)
			
			definition_with_lemma_encoded = torch.tensor(definition_with_lemma_encoded).to(device)
			definition_without_lemma_encoded = torch.tensor(definition_without_lemma_encoded).to(device)
			tg_trks_examples = torch.tensor(tg_trks_examples).to(device)
			bert_input_examples = [torch.tensor(bert_input).unsqueeze(0).to(device) for bert_input in bert_input_examples]
			
			yield definition_with_lemma_encoded, definition_without_lemma_encoded, bert_input_examples, tg_trks_examples, supersense, sense_id, lemma



class corpusEncoder(exampleEncoder):
	def __init__(self, datafile, dataset, tokenizer, ann_stage, use_sample=False, sample_size=32):
		super().__init__(datafile, dataset, tokenizer, use_sample=use_sample, sample_size=sample_size)
		self.df_definitions = self.df_definitions[self.df_definitions['ann_stage']==ann_stage]
		self.df_examples = self.df_examples[self.df_examples['ann']==ann_stage]


class wikiEncoder():
	def __init__(self, def_datafile, ex_datafile, tokenizer, use_sample=False, sample_size=32):
		self.tokenizer = tokenizer
		
		self.def_datafile = def_datafile
		self.ex_datafile = ex_datafile
		
		self.df_definitions = pd.read_csv(def_datafile, sep='\t')
		self.df_definitions['lemma'] = self.df_definitions['lemma'].str.replace('_', ' ')
		
		self.df_examples = pd.read_csv(ex_datafile, sep='\t')
		self.df_examples = self.df_examples[self.df_examples['word_rank'] >= 0]
		self.df_examples['lemma'] = self.df_examples['lemma'].str.replace('_', ' ')
		
		self.senses_ids = self.df_definitions['sense_id'].tolist()
		
		if use_sample:
			self.df_definitions = self.df_definitions.sample(sample_size)
			self.senses_ids = self.df_definitions['sense_id'].tolist()
	
	def truncate(self, sentences, ranks=[], max_length=100):

		max_length = max_length - 2

		trunc_sentences = []
		new_ranks = []
		
		if len(ranks) == 0: ranks = [0]*len(sentences)

		for sent, target_index in zip(sentences, ranks):
			if len(sent) <= max_length:

				trunc_sentences.append(sent)
				new_ranks.append(target_index)  
			else:

				half_max_length = max_length // 2
				start_index = max(0, min(len(sent) - max_length, target_index - half_max_length))
				end_index = start_index + max_length


				trunc_sent = sent[start_index:end_index]
				trunc_sentences.append(trunc_sent)


				new_target_index = target_index - start_index

				new_target_index = max(0, min(new_target_index, max_length-1))
				new_ranks.append(new_target_index)

		return trunc_sentences, new_ranks



	def pad(self, sentences, pad_id=2, max_length=100):

		max_length = max_length - 2
		pad_lengths = [ max_length - len(sent) if max_length >= len(sent) else 0 for sent in sentences ]

		padded_sentences = [ [el for el in sent] + pad_lengths[i] * [pad_id] for i, sent in enumerate(sentences) ]
		
		return padded_sentences
		
	
	
	def add_special_tokens(self, sentences, ranks=[], cls_id=0, sep_id=1):
	
		sentences_with_special_tokens = [ [cls_id] + [tok for tok in sent] + [sep_id] for sent in sentences ]
		if ranks: rks = [rk + 1 for rk in ranks]
		else: rks = []

		return sentences_with_special_tokens, rks
	
	def encoded_senses(self, device):
		
		df_definitions = self.df_definitions
		df_examples = self.df_examples
		
		tokenizer = self.tokenizer
		
		for sense_id in self.senses_ids:
			
			definition = df_definitions[df_definitions['sense_id'] == sense_id]["definition"].iloc[0]
			if pd.isna(definition) or definition == '': definition = None
			
			lemma = df_definitions[df_definitions['sense_id'] == sense_id]["lemma"].iloc[0]
			lemma = str(lemma)

			
			if definition: 
				definition_with_lemma_encoded = tokenizer.encoded_input = tokenizer.encode_plus(
																					text=f"{lemma.replace('_',' ')} : {definition}",
																					add_special_tokens=True,
																					max_length=100,  
																					padding='max_length',  
																					truncation=True,        
																					return_tensors='pt'     
																								)['input_ids']
				
			else:
				definition_with_lemma_encoded = None
			
			examples = df_examples[df_examples['sense_id'] == sense_id]['example'].tolist()
			word_ranks = df_examples[df_examples['sense_id'] == sense_id]['word_rank'].tolist()
			
			
			examples = [ x.split(' ') for x in examples ]
			for example in examples:
				for x in example:
					x = x.replace('##', ' ')
			

			sents_encoded = [ tokenizer(word, add_special_tokens=False)['input_ids'] for word in examples ]
			
			tg_trks = [token_rank(sent, rank) for sent, rank in zip(sents_encoded, word_ranks)]
			

			bert_input_raw = [ flatten_list(sent) for sent in sents_encoded ]
			
			bert_input_raw, tg_trks = self.truncate(bert_input_raw, tg_trks)
			bert_input_raw = self.pad(bert_input_raw, pad_id=2)
			
			bert_input_examples, tg_trks_examples = self.add_special_tokens(bert_input_raw, tg_trks, cls_id=0, sep_id=1)

			
			if definition: definition_with_lemma_encoded = torch.tensor(definition_with_lemma_encoded).to(device)
			tg_trks_examples = torch.tensor(tg_trks_examples).to(device)
			bert_input_examples = [torch.tensor(bert_input).unsqueeze(0).to(device) for bert_input in bert_input_examples]
			
			
			yield definition_with_lemma_encoded, bert_input_examples, tg_trks_examples, sense_id, lemma
