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


class monoRankClf(nn.Module):

	def __init__(self, params, DEVICE, use_lemma=False, dropout_rate=0.1, bert_model_name=MODEL_NAME):
		super(monoRankClf, self).__init__()

		self.bert_model = AutoModel.from_pretrained(bert_model_name).to(DEVICE)

		if params["frozen"]:
			for param in self.bert_model.parameters():
				param.requires_grad = False

		self.embedding_layer_size = self.bert_model.config.hidden_size
		
		self.use_lemma = use_lemma

		self.hidden_layer_size = params['hidden_layer_size']
		
		self.token_rank = params['token_rank']

		self.output_size = NB_CLASSES
		
		self.device = DEVICE
		
		self.params = params

		self.linear_1 = nn.Linear(self.embedding_layer_size, self.hidden_layer_size).to(DEVICE)

		self.linear_2 = nn.Linear(self.hidden_layer_size, self.output_size).to(DEVICE)

		self.dropout = nn.Dropout(params['dropout']).to(DEVICE)

		self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
		

	def forward(self, padded_encodings):

		bert_output = self.bert_model(padded_encodings, return_dict=True) # SHAPE [len(definitions), max_length, embedding_size]

		batch_contextual_embeddings = bert_output.last_hidden_state[:,self.token_rank,:] # from [batch_size , max_seq_length, plm_emb_size] to [batch_size, plm_emb_size]

		out = self.linear_1(batch_contextual_embeddings) # SHAPE [len(definitions), hidden_layer_size]
		
		out = self.dropout(out)

		out = torch.relu(out) # SHAPE [len(definitions), hidden_layer_size]

		out = self.linear_2(out) # SHAPE [len(definitions), nb_classes]

		return F.log_softmax(out, dim=1)
		


	def training(self, train_encoder, dev_encoder):
		self.train()
		pass
	

	def evaluate(self, data_encoder):
		self.eval()
		with torch.no_grad():
			pass
	
	def predict(self, data_encoder):
		self.eval()
		with torch.no_grad():
			pass
	
	def evaluate_and_predict(self, data_encoder):
		self.eval()
		with torch.no_grad():
			pass
		



class multiRankClf(nn.Module):

	def __init__(self, params, DEVICE, use_lemma=False, dropout_rate=0.1, bert_model_name=MODEL_NAME):
		super(multiRankClf, self).__init__()

		self.bert_model = AutoModel.from_pretrained(bert_model_name).to(DEVICE)

		if params["frozen"]:
			for param in self.bert_model.parameters():
				param.requires_grad = False

		self.embedding_layer_size = self.bert_model.config.hidden_size

		self.hidden_layer_size = params['hidden_layer_size']

		self.output_size = NB_CLASSES
		
		self.use_lemma = use_lemma

		self.linear_1 = nn.Linear(self.embedding_layer_size, self.hidden_layer_size).to(DEVICE)

		self.linear_2 = nn.Linear(self.hidden_layer_size, self.output_size).to(DEVICE)

		self.dropout = nn.Dropout(params['dropout']).to(DEVICE)

		self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
		
		self.params = params
		
		self.device = DEVICE

	def forward(self, X_input, X_rank):
	
		batch_size = self.params['batch_size']
		max_length = self.params['max_seq_length']
		bert_emb_size = self.embedding_layer_size

		bert_tok_embeddings = self.bert_model(input_ids=X_input).last_hidden_state # [batch_size, max_length, bert_emb_size]

		selected_tensors = [bert_tok_embeddings[idx, X_rank[idx], :] for idx in range(bert_tok_embeddings.size(0))]

		bert_target_word_embeddings = torch.stack(selected_tensors, dim=0)
		
		out = self.linear_1(bert_target_word_embeddings) # SHAPE [len(definitions), hidden_layer_size]
		
		out = self.dropout(out)

		out = torch.relu(out) # SHAPE [len(definitions), hidden_layer_size]

		out = self.linear_2(out) # SHAPE [len(definitions), nb_classes]

		return F.log_softmax(out, dim=1)


	def training(self, train_encoder, dev_encoder):
		self.train()
		pass
		

	def evaluate(self, data_encoder):
		self.eval()
		with torch.no_grad():
			pass
	
	def predict(self, data_encoder):
		self.eval()
		with torch.no_grad():
			pass
	
	def evaluate_and_predict(self, data_encoder):
		self.eval()
		with torch.no_grad():
			pass


class lexicalClf_V1():

	def __init__(self, params, DEVICE, multi_clf=False, dropout_rate=0.1, bert_model_name=MODEL_NAME):
	
		if multi_clf: self.def_clf = monoRankClf(params, DEVICE, dropout_rate=dropout_rate, bert_model_name=bert_model_name)
		self.def_lem_clf = monoRankClf(params, DEVICE, use_lemma=True, dropout_rate=dropout_rate, bert_model_name=bert_model_name)
		self.ex_clf = multiRankClf(params, DEVICE, dropout_rate=dropout_rate, bert_model_name=bert_model_name)
		self.multi_clf = multi_clf

	def training(self, train_encoder, dev_encoder):
		if self.multi_clf: self.def_clf.train(train_encoder, dev_encoder)
		self.def_lem_clf.train(train_encoder, dev_encoder)
		self.ex_clf.train(train_encoder, dev_encoder)

	def evaluate(self, sense_encoder):
		pass
		
	def predict(self, sense_encoder):
		pass
		
	def evaluate_and_predict(self, sense_encoder):
		pass



class lexicalClf_V2(nn.Module):

	def __init__(self, params, DEVICE, multi_clf=False, dropout_rate=0.1, bert_model_name=MODEL_NAME):
		if multi_clf: self.clf = multiRankClf(params, DEVICE, dropout_rate=0.1, bert_model_name=MODEL_NAME)
		self.clf_lem = multiRankClf(params, DEVICE, use_lemma=True, dropout_rate=0.1, bert_model_name=MODEL_NAME)
		self.multi_clf = multi_clf
		

	def training(self, train_encoder, dev_encoder):
		pass
	
	def evaluate(self, sense_encoder):
		pass
		
	def predict(self, sense_encoder):
		pass
		
	def evaluate_and_predict(self, sense_encoder):
		pass



class Baseline:

	def __init__(self):
		self.most_frequent_supersense = ''
	
	def training(self):
		pass

	def evaluation(self, eval_examples):
		return sum([int(supersense == supersense2i[self.most_frequent_supersense]) for _, supersense in eval_examples])/len(eval_examples)



class MostFrequentSequoia(Baseline):
	def __init__(self):
		super().__init__()

	def training(self):
		self.most_frequent_supersense = 'act'


class MostFrequentWiktionary(Baseline):
	def __init__(self):
		super().__init__()
	def training(self):
		self.most_frequent_supersense = 'person'


class MostFrequentTrainingData(Baseline):
	def __init__(self):
		super().__init__()

	def training(self):
		self.most_frequent_supersense = 'artifact'
