import pandas as pd
from collections import Counter, defaultdict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sacremoses
from random import shuffle
import numpy as np
from transformers import AutoModel, AutoTokenizer, AutoConfig
from matplotlib import pyplot as plt
import warnings
warnings.filterwarnings("ignore")
from kan import KAN



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

	def __init__(self, params, DEVICE, use_lemma=True, dropout_hidden=0.3, dropout_input=0, bert_model_name=MODEL_NAME):
		super(monoRankClf, self).__init__()

		self.bert_model = AutoModel.from_pretrained(bert_model_name).to(DEVICE)

		if params["frozen"]:
			for param in self.bert_model.parameters():
				param.requires_grad = False

		self.embedding_layer_size = self.bert_model.config.hidden_size
		
		self.use_lemma = use_lemma

		self.hidden_layer_size = params['hidden_layer_size']

		self.output_size = NB_CLASSES
		
		self.device = DEVICE
		
		self.params = params

		self.linear_1 = nn.Linear(self.embedding_layer_size, self.hidden_layer_size).to(DEVICE)

		self.linear_2 = nn.Linear(self.hidden_layer_size, self.output_size).to(DEVICE)
		
		self.dropout_input = nn.Dropout(dropout_input).to(DEVICE)

		self.dropout_hidden = nn.Dropout(dropout_hidden).to(DEVICE)

		self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
		

	def forward(self, padded_encodings):

		bert_output = self.bert_model(padded_encodings, return_dict=True) # SHAPE [len(definitions), max_length, embedding_size]

		batch_contextual_embeddings = bert_output.last_hidden_state[:,0,:] # from [batch_size , max_seq_length, plm_emb_size] to [batch_size, plm_emb_size]
		
		# out = self.dropout_input(batch_contextual_embeddings)

		out = self.linear_1(batch_contextual_embeddings) # SHAPE [len(definitions), hidden_layer_size]
		
		out = self.dropout_hidden(out)
		
		out = torch.relu(out) # SHAPE [len(definitions), hidden_layer_size]

		out = self.linear_2(out) # SHAPE [len(definitions), nb_classes]

		return F.log_softmax(out, dim=1)
		


	def train_clf(self, train_encoder, freq_dev_encoder, rand_dev_encoder, clf_file):
		self.train()
		
		train_losses = []
		mean_dev_losses = []
		mean_dev_accuracies = []
		freq_dev_losses = []
		freq_dev_accuracies = []
		rand_dev_losses = []
		rand_dev_accuracies = []
		
		use_lemma = self.use_lemma
		
		params = self.params
		
		loss_function = nn.NLLLoss()
		
		patience = params["patience"]
		max_mean_dev_accuracy = 0
		min_mean_dev_loss = 10000000000
		
		optimizer = optim.AdamW(self.parameters(), lr=params["lr"], weight_decay=params["weight_decay"])
		

		for epoch in range(params["nb_epochs"]):
			print("epoch: ", epoch+1)
			
			epoch_loss = 0
			train_epoch_accuracy = 0
			freq_dev_epoch_loss = 0
			freq_dev_epoch_accuracy = 0
			rand_dev_epoch_loss = 0
			rand_dev_epoch_accuracy = 0
			
			for b_definitions_with_lemma_encoded, b_definitions_without_lemma_encoded, b_supersenses_encoded, _, _ in train_encoder.make_batches(device=self.device, batch_size=params['batch_size'], shuffle_data=True):
				
				if use_lemma: b_def_encoded = b_definitions_with_lemma_encoded
				else: b_def_encoded = b_definitions_without_lemma_encoded
				
				self.zero_grad()
				
				log_probs = self.forward(b_def_encoded)
				
				loss = loss_function(log_probs, b_supersenses_encoded)
				loss.backward()
				optimizer.step()

				epoch_loss += loss.item()/params["batch_size"]

			train_losses.append(epoch_loss)
			
			with torch.no_grad():
			
				for b_definitions_with_lemma_encoded, b_definitions_without_lemma_encoded, b_supersenses_encoded, _, _ in freq_dev_encoder.make_batches(device=self.device, batch_size=params['batch_size'], shuffle_data=False):
					
					if use_lemma: b_def_encoded = b_definitions_with_lemma_encoded
					else: b_def_encoded = b_definitions_without_lemma_encoded
					
					freq_dev_log_probs = self.forward(b_def_encoded)

					predicted_indices = torch.argmax(freq_dev_log_probs, dim=1)
					freq_dev_epoch_accuracy += torch.sum((predicted_indices == b_supersenses_encoded).int()).item()

					freq_dev_loss = loss_function(freq_dev_log_probs, b_supersenses_encoded)
					freq_dev_epoch_loss += freq_dev_loss.item()

				freq_dev_losses.append(freq_dev_epoch_loss / freq_dev_encoder.length)
				freq_dev_accuracies.append(freq_dev_epoch_accuracy / freq_dev_encoder.length)
				
				
				for b_definitions_with_lemma_encoded, b_definitions_without_lemma_encoded, b_supersenses_encoded, _, _ in rand_dev_encoder.make_batches(device=self.device, batch_size=params['batch_size'], shuffle_data=False):
					
					if use_lemma: b_def_encoded = b_definitions_with_lemma_encoded
					else: b_def_encoded = b_definitions_without_lemma_encoded
					
					rand_dev_log_probs = self.forward(b_def_encoded)

					predicted_indices = torch.argmax(rand_dev_log_probs, dim=1)
					rand_dev_epoch_accuracy += torch.sum((predicted_indices == b_supersenses_encoded).int()).item()

					rand_dev_loss = loss_function(rand_dev_log_probs, b_supersenses_encoded)
					rand_dev_epoch_loss += rand_dev_loss.item()

				rand_dev_losses.append(rand_dev_epoch_loss / rand_dev_encoder.length)
				rand_dev_accuracies.append(rand_dev_epoch_accuracy / rand_dev_encoder.length)


			mean_dev_losses.append( (freq_dev_epoch_loss/freq_dev_encoder.length + rand_dev_epoch_loss/rand_dev_encoder.length ) / 2)
			mean_dev_accuracies.append( (freq_dev_epoch_accuracy/freq_dev_encoder.length + rand_dev_epoch_accuracy/rand_dev_encoder.length ) / 2)
			
			if epoch >= params["patience"]:
			
				if mean_dev_losses[epoch] < min_mean_dev_loss:
					min_mean_dev_loss = mean_dev_losses[epoch]
					torch.save(self.state_dict(), clf_file)
					patience = params["patience"]
					
				else:
					patience = patience - 1
				
				if patience == 0:
					print("EARLY STOPPING : epoch ", epoch+1)
					break
			else:
				if mean_dev_losses[epoch] < min_mean_dev_loss:
					min_mean_dev_loss = mean_dev_losses[epoch]
				torch.save(self.state_dict(), clf_file)
	
	def save_clf(self, clf_save_file):
		torch.save(self.state_dict(), clf_save_file)
	
	def load_clf(self, clf_save_file):
		self.load_state_dict(torch.load(clf_save_file))
		

	def evaluate(self, data_encoder):
		self.eval()
		accuracy = 0
		with torch.no_grad():
			for b_definitions_with_lemma_encoded, b_definitions_without_lemma_encoded, b_supersenses_encoded, _, _ in data_encoder.make_batches(device=self.device, batch_size=self.params['batch_size'], shuffle_data=False):
				
				if self.use_lemma: b_def_encoded = b_definitions_with_lemma_encoded
				else: b_def_encoded = b_definitions_without_lemma_encoded
				
				log_probs = self.forward(b_def_encoded)
				predicted_indices = torch.argmax(log_probs, dim=1)
				accuracy += torch.sum((predicted_indices == b_supersenses_encoded).int()).item()
				
			return accuracy / data_encoder.length
			
	
	def predict(self, data_encoder):
		self.eval()
		predictions = {"lemma":[], "sense_id":[], "gold":[], "pred":[], "definition":[]}
		with torch.no_grad():
			for b_definitions_with_lemma_encoded, b_definitions_without_lemma_encoded, b_supersenses_encoded, b_senses_ids, b_lemmas in data_encoder.make_batches(device=self.device, batch_size=self.params['batch_size'], shuffle_data=False):
				
				if self.use_lemma: b_def_encoded = b_definitions_with_lemma_encoded
				else: b_def_encoded = b_definitions_without_lemma_encoded
				
				log_probs = self.forward(b_def_encoded)
				predicted_indices = torch.argmax(log_probs, dim=1).tolist()
				
				pred = [SUPERSENSES[i] for i in predicted_indices]
				gold = [SUPERSENSES[i] for i in b_supersenses_encoded.tolist()]
				definitions = [self.tokenizer.decode(token_ids.tolist(), skip_special_tokens=True) for token_ids in b_def_encoded]
				
				predictions['lemma'].extend(b_lemmas)
				predictions['sense_id'].extend(b_senses_ids)
				predictions['gold'].extend(gold)
				predictions['pred'].extend(pred)
				predictions['definition'].extend(definitions)
				
			return predictions
	
	def evaluate_and_predict(self, data_encoder):
	
		accuracy = self.evaluate(data_encoder)
		predictions = self.predict(data_encoder)
		
		return accuracy, predictions	
		



class multiRankClf(nn.Module):

	def __init__(self, params, DEVICE, dropout_input=0.1, dropout_hidden=0.5, bert_model_name=MODEL_NAME):
		super(multiRankClf, self).__init__()

		self.bert_model = AutoModel.from_pretrained(bert_model_name).to(DEVICE)

		if params["frozen"]:
			for param in self.bert_model.parameters():
				param.requires_grad = False

		self.embedding_layer_size = self.bert_model.config.hidden_size

		self.hidden_layer_size = params['hidden_layer_size']

		self.output_size = NB_CLASSES

		self.linear_1 = nn.Linear(self.embedding_layer_size, self.hidden_layer_size).to(DEVICE)

		self.linear_2 = nn.Linear(self.hidden_layer_size, self.output_size).to(DEVICE)

		self.dropout_input = nn.Dropout(dropout_input).to(DEVICE)
		
		self.dropout_hidden = nn.Dropout(dropout_hidden).to(DEVICE)

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
		
		# out = self.dropout_input(bert_target_word_embeddings)
		
		out = self.linear_1(bert_target_word_embeddings) # SHAPE [len(definitions), hidden_layer_size]
		
		out = self.dropout_hidden(out)

		out = torch.relu(out) # SHAPE [len(definitions), hidden_layer_size]

		out = self.linear_2(out) # SHAPE [len(definitions), nb_classes]

		return F.log_softmax(out, dim=1)


	def train_clf(self, train_encoder, freq_dev_encoder, rand_dev_encoder, clf_file):
		self.train()
		
		train_losses = []
		mean_dev_losses = []
		mean_dev_accuracies = []
		freq_dev_losses = []
		freq_dev_accuracies = []
		rand_dev_losses = []
		rand_dev_accuracies = []
		
		
		params = self.params
		
		loss_function = nn.NLLLoss()
		
		patience = params["patience"]
		max_mean_dev_accuracy = 0
		min_mean_dev_loss = 10000000000
		
		optimizer = optim.AdamW(self.parameters(), lr=params["lr"], weight_decay=params["weight_decay"])
		

		for epoch in range(params["nb_epochs"]):
			print("epoch: ", epoch+1)
			
			epoch_loss = 0
			train_epoch_accuracy = 0
			freq_dev_epoch_loss = 0
			freq_dev_epoch_accuracy = 0
			rand_dev_epoch_loss = 0
			rand_dev_epoch_accuracy = 0
			
			for b_bert_encodings, b_target_ranks, b_supersenses_encoded, _, _ in train_encoder.make_batches(device=self.device, batch_size=params['batch_size'], shuffle_data=True):
				
				self.zero_grad()
				
				log_probs = self.forward(b_bert_encodings, b_target_ranks)
				
				loss = loss_function(log_probs, b_supersenses_encoded)
				loss.backward()
				optimizer.step()

				epoch_loss += loss.item()/params["batch_size"]

			train_losses.append(epoch_loss)
			
			with torch.no_grad():
			
				for b_bert_encodings, b_target_ranks, b_supersenses_encoded, _, _ in freq_dev_encoder.make_batches(device=self.device, batch_size=params['batch_size'], shuffle_data=False):
					
					freq_dev_log_probs = self.forward(b_bert_encodings, b_target_ranks)

					predicted_indices = torch.argmax(freq_dev_log_probs, dim=1)
					freq_dev_epoch_accuracy += torch.sum((predicted_indices == b_supersenses_encoded).int()).item()

					freq_dev_loss = loss_function(freq_dev_log_probs, b_supersenses_encoded)
					freq_dev_epoch_loss += freq_dev_loss.item()

				freq_dev_losses.append(freq_dev_epoch_loss / freq_dev_encoder.length)
				freq_dev_accuracies.append(freq_dev_epoch_accuracy / freq_dev_encoder.length)
				
				
				for b_bert_encodings, b_target_ranks, b_supersenses_encoded, _, _ in rand_dev_encoder.make_batches(device=self.device, batch_size=params['batch_size'], shuffle_data=False):
					
					rand_dev_log_probs = self.forward(b_bert_encodings, b_target_ranks)

					predicted_indices = torch.argmax(rand_dev_log_probs, dim=1)
					rand_dev_epoch_accuracy += torch.sum((predicted_indices == b_supersenses_encoded).int()).item()

					rand_dev_loss = loss_function(rand_dev_log_probs, b_supersenses_encoded)
					rand_dev_epoch_loss += rand_dev_loss.item()

				rand_dev_losses.append(rand_dev_epoch_loss / rand_dev_encoder.length)
				rand_dev_accuracies.append(rand_dev_epoch_accuracy / rand_dev_encoder.length)


			mean_dev_losses.append( (freq_dev_epoch_loss/freq_dev_encoder.length + rand_dev_epoch_loss/rand_dev_encoder.length ) / 2)
			mean_dev_accuracies.append( (freq_dev_epoch_accuracy/freq_dev_encoder.length + rand_dev_epoch_accuracy/rand_dev_encoder.length ) / 2)
			
			if epoch >= params["patience"]:
			
				if mean_dev_losses[epoch] < min_mean_dev_loss:
					min_mean_dev_loss = mean_dev_losses[epoch]
					torch.save(self.state_dict(), clf_file)
					patience = params["patience"]
					
				else:
					patience = patience - 1
				
				if patience == 0:
					print("EARLY STOPPING : epoch ", epoch+1)
					break
			else:
				if mean_dev_losses[epoch] < min_mean_dev_loss:
					min_mean_dev_loss = mean_dev_losses[epoch]
				torch.save(self.state_dict(), clf_file)
				
				
	def train_contextual_clf(self, train_encoder, dev_encoder, clf_file):
		self.train()
		
		train_losses = []
		dev_losses = []
		dev_accuracies = []
		
		params = self.params
		
		loss_function = nn.NLLLoss()
		
		patience = params["patience"]
		max_dev_accuracy = 0
		min_dev_loss = 10000000000
		
		optimizer = optim.AdamW(self.parameters(), lr=params["lr"], weight_decay=params["weight_decay"])
		

		for epoch in range(params["nb_epochs"]):
			print("epoch: ", epoch+1)
			
			epoch_loss = 0
			train_epoch_accuracy = 0
			dev_epoch_loss = 0
			dev_epoch_accuracy = 0
			
			for b_bert_encodings, b_target_ranks, b_supersenses_encoded, _, _ in train_encoder.make_batches(device=self.device, batch_size=params['batch_size'], shuffle_data=True):
				
				self.zero_grad()
				
				log_probs = self.forward(b_bert_encodings, b_target_ranks)
				
				loss = loss_function(log_probs, b_supersenses_encoded)
				loss.backward()
				optimizer.step()

				epoch_loss += loss.item()/params["batch_size"]

			train_losses.append(epoch_loss)
			
			with torch.no_grad():
			
				for b_bert_encodings, b_target_ranks, b_supersenses_encoded, _, _ in dev_encoder.make_batches(device=self.device, batch_size=params['batch_size'], shuffle_data=False):
					
					dev_log_probs = self.forward(b_bert_encodings, b_target_ranks)

					predicted_indices = torch.argmax(dev_log_probs, dim=1)
					dev_epoch_accuracy += torch.sum((predicted_indices == b_supersenses_encoded).int()).item()

					dev_loss = loss_function(dev_log_probs, b_supersenses_encoded)
					dev_epoch_loss += dev_loss.item()

				dev_losses.append(dev_epoch_loss / dev_encoder.length)
				dev_accuracies.append(dev_epoch_accuracy / dev_encoder.length)

			
			if epoch >= params["patience"]:
			
				if dev_losses[epoch] < min_dev_loss:
					min_dev_loss = dev_losses[epoch]
					torch.save(self.state_dict(), clf_file)
					patience = params["patience"]
					
				else:
					patience = patience - 1
				
				if patience == 0:
					print("EARLY STOPPING : epoch ", epoch+1)
					break
			else:
				if dev_losses[epoch] < min_dev_loss:
					min_dev_loss = dev_losses[epoch]
				torch.save(self.state_dict(), clf_file)
	
	
	def save_clf(self, clf_save_file):
		torch.save(self.state_dict(), clf_save_file)
		
	def load_clf(self, clf_save_file):
		self.load_state_dict(torch.load(clf_save_file))
		

	def evaluate(self, data_encoder):
		self.eval()
		accuracy = 0
		with torch.no_grad():
			for b_bert_encodings, b_target_ranks, b_supersenses_encoded, _, _ in data_encoder.make_batches(device=self.device, batch_size=self.params['batch_size'], shuffle_data=False):
				
				log_probs = self.forward(b_bert_encodings, b_target_ranks)
				predicted_indices = torch.argmax(log_probs, dim=1)
				accuracy += torch.sum((predicted_indices == b_supersenses_encoded).int()).item()
				
			return accuracy / data_encoder.length
			
	
	def predict(self, data_encoder):
		self.eval()
		predictions = {"lemma":[], "sense_id":[], "gold":[], "pred":[], "sentence":[]}
		with torch.no_grad():
			for b_bert_encodings, b_target_ranks, b_supersenses_encoded, b_senses_ids, b_lemmas in data_encoder.make_batches(device=self.device, batch_size=self.params['batch_size'], shuffle_data=False):
				
				log_probs = self.forward(b_bert_encodings, b_target_ranks)
				predicted_indices = torch.argmax(log_probs, dim=1).tolist()
				
				pred = [SUPERSENSES[i] for i in predicted_indices]
				gold = [SUPERSENSES[i] for i in b_supersenses_encoded.tolist()]
				sentences = [self.tokenizer.decode(token_ids.tolist(), skip_special_tokens=True) for token_ids in b_bert_encodings]
				
				predictions['lemma'].extend(b_lemmas)
				predictions['sense_id'].extend(b_senses_ids)
				predictions['gold'].extend(gold)
				predictions['pred'].extend(pred)
				predictions['sentence'].extend(sentences)
				
			return predictions
	
	def evaluate_and_predict(self, data_encoder):
		
		accuracy = self.evaluate(data_encoder)
		predictions = self.predict(data_encoder)
		
		return accuracy, predictions	





class lexicalClf_V1():

	def __init__(self, params_def, params_ex, DEVICE, coeff_ex, coeff_def,  dropout_hidden=0.3, dropout_input=0, bert_model_name=MODEL_NAME):

		self.def_lem_clf = monoRankClf(params_def, DEVICE, use_lemma=True, dropout_hidden=dropout_hidden, dropout_input=dropout_input, bert_model_name=bert_model_name)
		self.ex_clf = multiRankClf(params_ex, DEVICE, dropout_hidden=dropout_hidden, dropout_input=dropout_input, bert_model_name=bert_model_name)
		self.coeff_ex = coeff_ex
		self.coeff_def = coeff_def
		self.device = DEVICE
		self.tokenizer = AutoTokenizer.from_pretrained(bert_model_name)

	def training(self, train_encoder, freq_dev_encoder, rand_dev_encoder):
	
		self.def_lem_clf.train(train_encoder, freq_dev_encoder, rand_dev_encoder)
		self.ex_clf.train(train_encoder, freq_dev_encoder, rand_dev_encoder)
	
	def load_clf(self, clf_def_lem_file, clf_ex_file):
	
		self.def_lem_clf.load_state_dict(torch.load(clf_def_lem_file))
		self.ex_clf.load_state_dict(torch.load(clf_ex_file))
	
	# TODO
	"""
	def evaluate(self, sense_encoder):
		self.eval()
		accuracy = 0
		with torch.no_grad():
			for definition_with_lemma_encoded, definition_without_lemma_encoded, bert_input_examples, tg_trks_examples, supersense, sense_id, lemma in sense_encoder.make_batches(device=self.device, batch_size=self.params['batch_size'], shuffle_data=False):
				
				def_log_probs = self.def_lem_clf.forward(definition_with_lemma_encoded)
				
				ex_log_probs = [self.ex_clf.forward(bert_input_examples, tg_trks_examples)]
				ex_log_probs = torch.stack(ex_log_probs)
				ex_log_probs = torch.mean(stacked_tensors, dim=0)
				
				log_probs = self.coeff_def * def_log_probs + self.coeff_ex * ex_log_probs
				predicted_indices = torch.argmax(log_probs, dim=1)
				accuracy += torch.sum((predicted_indices == b_supersenses_encoded).int()).item()
				
			return accuracy / data_encoder.length
	"""
	
	def predict(self, sense_encoder):
		self.def_lem_clf.eval()
		self.ex_clf.eval()
		predictions = {"lemma":[], "sense_id":[], "gold":[], "pred":[], "sentence":[]}
		with torch.no_grad():
			for definition_with_lemma_encoded, definition_without_lemma_encoded, bert_input_examples, tg_trks_examples, supersense, sense_id, lemma in sense_encoder.encoded_senses(device=self.device):
				
				def_log_probs = self.def_lem_clf.forward(definition_with_lemma_encoded)
				
				if bert_input_examples:
					ex_log_probs = [self.ex_clf.forward(input_, torch.tensor(tg_trk).unsqueeze(0)) for input_, tg_trk in zip(bert_input_examples, tg_trks_examples)]
					ex_log_probs = torch.stack(ex_log_probs)
					ex_log_probs = torch.mean(ex_log_probs, dim=0)
				
					log_probs = self.coeff_def * def_log_probs + self.coeff_ex * ex_log_probs
					predicted_index = torch.argmax(log_probs, dim=1).item()
				else:
					predicted_index = torch.argmax(def_log_probs, dim=1).item()
				
				pred = SUPERSENSES[predicted_index]
				gold = supersense

				sentence = self.tokenizer.decode(definition_with_lemma_encoded.squeeze(), skip_special_tokens=True)
				
				predictions['lemma'].append(lemma)
				predictions['sense_id'].append(sense_id)
				predictions['gold'].append(gold)
				predictions['pred'].append(pred)
				predictions['sentence'].append(sentence)
				
		return predictions
		
	def evaluate_and_predict(self, sense_encoder):
		accuracy = self.evaluate(sense_encoder)
		predictions = self.predict(sense_encoder)
		
		return accuracy, predictions
		
	def predict_wiki(self, wiki_encoder):
		self.def_lem_clf.eval()
		self.ex_clf.eval()
		predictions = {"lemma":[], "sense_id":[], "pred":[]}
		with torch.no_grad():
			for definition_with_lemma_encoded, bert_input_examples, tg_trks_examples, sense_id, lemma in wiki_encoder.encoded_senses(device=self.device):
				
				if definition_with_lemma_encoded is not None: 
					def_log_probs = self.def_lem_clf.forward(definition_with_lemma_encoded)
					def_weight = self.coeff_def
				else:
					def_log_probs = torch.zeros(len(SUPERSENSES)).to(self.device)
					def_weight = 0
				
				if bert_input_examples:
					ex_log_probs = [self.ex_clf.forward(input_, torch.tensor(tg_trk).unsqueeze(0)) for input_, tg_trk in zip(bert_input_examples, tg_trks_examples)]
					ex_log_probs = torch.stack(ex_log_probs)
					ex_log_probs = torch.mean(ex_log_probs, dim=0)
					ex_weight = self.coeff_ex
				
				else:
					ex_log_probs = torch.zeros(len(SUPERSENSES)).to(self.device)
					ex_weight = 0
				
				log_probs = def_weight * def_log_probs + ex_weight * ex_log_probs
				
				if torch.all(log_probs == 0).item():
					pred = ''
					
				else:
					predicted_index = torch.argmax(log_probs, dim=1).item()
					pred = SUPERSENSES[predicted_index]
				
				predictions['lemma'].append(lemma)
				predictions['sense_id'].append(sense_id)
				predictions['pred'].append(pred)
				
		return predictions



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



class KANClf():

	def __init__(self, params, bert_model_name, DEVICE):

		# Load model configuration
		self.config = AutoConfig.from_pretrained(bert_model_name)

		# Set embedding layer size
		self.embedding_layer_size = self.config.hidden_size

		self.hidden_layer_size = params['hidden_layer_size']

		self.output_size = NB_CLASSES
		
		self.device = DEVICE
		
		self.kan = KAN(width=[self.embedding_layer_size, self.hidden_layer_size, self.output_size], grid=params['grid'], k=params['k'], device=self.device)
		
		self.params = params

		self.tokenizer = AutoTokenizer.from_pretrained(bert_model_name)
		

	def train(self):

		params = self.params

		X_train = torch.load('./train_embeddings.pt').to(self.device)
		y_train = torch.load('./train_supersenses.pt').to(self.device)
		
		X_test = torch.load('./test_embeddings.pt').to(self.device)
		y_test = torch.load('./test_supersenses.pt').to(self.device)
		
		dataset = {}
		dataset['train_input'] = X_train
		dataset['test_input'] = X_test
		dataset['train_label'] = y_train
		dataset['test_label'] = y_test
		
		def train_acc():
			batch_size = 500  # Set your desired batch size
			num_samples = X_train.shape[0]
			num_batches = (num_samples + batch_size - 1) // batch_size  # Calculate the number of batches

			total_correct = 0
			total_samples = 0

			for i in range(num_batches):
				start_idx = i * batch_size
				end_idx = min((i + 1) * batch_size, num_samples)

				# Get batch
				X_batch = X_train[start_idx:end_idx]
				y_batch = y_train[start_idx:end_idx]

				# Get predictions for the batch
				predictions = torch.argmax(self.kan(X_batch), dim=1)

				# Calculate accuracy for the batch
				total_correct += torch.sum(predictions == y_batch)
				total_samples += y_batch.size(0)

			# Calculate overall accuracy
			accuracy = total_correct / total_samples
			return accuracy

		def test_acc():
			batch_size = 500  # Set your desired batch size
			num_samples = X_test.shape[0]
			num_batches = (num_samples + batch_size - 1) // batch_size  # Calculate the number of batches

			total_correct = 0
			total_samples = 0

			for i in range(num_batches):
				start_idx = i * batch_size
				end_idx = min((i + 1) * batch_size, num_samples)

				# Get batch
				X_batch = X_test[start_idx:end_idx]
				y_batch = y_test[start_idx:end_idx]

				# Get predictions for the batch
				predictions = torch.argmax(self.kan(X_batch), dim=1)

				# Calculate accuracy for the batch
				total_correct += torch.sum(predictions == y_batch)
				total_samples += y_batch.size(0)

			# Calculate overall accuracy
			accuracy = total_correct / total_samples
			return accuracy

		results = self.kan.train(
							dataset, 
							opt="LBFGS",
							steps=params['nb_epochs'],
							metrics=(train_acc, test_acc),
							batch=params['batch_size'],
							loss_fn=torch.nn.CrossEntropyLoss())
		
		return results
		
	def predict(self):
		pass

