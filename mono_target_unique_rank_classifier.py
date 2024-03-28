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

HYPERSENSES = {"dynamic_situation": ["act", "event", "phenomenon", "act*cognition"],
               "stative_situation": ["attribute", "state", "feeling", "relation"],
               "animate_entity": ["animal", "person", "groupxperson"],
               "inanimate_entity": ["artifact", "food", "body", "object", "plant", "substance", "artifact*cognition"],
               "informational_object": ["cognition", "communication", "act*cognition", "artifact*cognition"],
               "quantification": ["quantity", "part", "group", "groupxperson"],
               "other": ["institution", "possession", "time"]
               }

supersense2i = {supersense: i for i, supersense in enumerate(SUPERSENSES)}


NB_CLASSES = len(supersense2i)
MODEL_NAME = "flaubert/flaubert_large_cased"
PADDING_TOKEN_ID = 2


def encoded_examples(datafile, dataset):
	df_senses = pd.read_excel(datafile, sheet_name='senses', engine='openpyxl')
	df_senses = df_senses[df_senses['supersense'].isin(SUPERSENSES)]
	df_senses = df_senses[(df_senses['definition'] != "") & (df_senses['definition'].notna())]
	df_senses = df_senses[df_senses['set']==dataset]

	tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
	
	definitions = df_senses['definition'].tolist()
	supersenses = df_senses['supersense'].tolist()
	lemmas = df_senses['lemma'].tolist()
	senses_ids = df_senses['sense_id'].tolist()
	definitions_encoded = [tokenizer.encode(text=f"{lemma} : {definition}", add_special_tokens=True) for definition, lemma in zip(definitions, lemmas)]
	supersenses_encoded = [supersense2i[supersense] for supersense in supersenses]
	
	return definitions_encoded, supersenses_encoded, lemmas, senses_ids



def pad_batch(encodings_batch, padding_token_id=2, max_seq_length=100):
	padding_size = max(len(sublist) for sublist in encodings_batch)
	padding_size = min(padding_size, max_seq_length)
	for sentence_encoding in encodings_batch:
		if len(sentence_encoding) < padding_size:
			while len(sentence_encoding) < padding_size:
				sentence_encoding.append(padding_token_id)
		else:
			while len(sentence_encoding) > padding_size:
				sentence_encoding.pop(-2)
	return torch.tensor(encodings_batch, dtype=torch.long)


class SupersenseTagger(nn.Module):

	def __init__(self, params, DEVICE, dropout_rate=0.1, bert_model_name=MODEL_NAME):
		super(SupersenseTagger, self).__init__()

		self.bert_model = AutoModel.from_pretrained(bert_model_name).to(DEVICE)

		if params["frozen"]:
			for param in self.bert_model.parameters():
				param.requires_grad = False

		self.embedding_layer_size = self.bert_model.config.hidden_size

		self.hidden_layer_size = params['hidden_layer_size']
		
		self.token_rank = params['token_rank']

		self.output_size = NB_CLASSES
		
		self.device = DEVICE

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
		
	def forward_encoding(self, encoding):
		encoding = torch.tensor(encoding).to(self.device)
		bert_output = self.bert_model(encoding.unsqueeze(0), return_dict=True)
		contextual_embeddings = bert_output.last_hidden_state[self.token_rank,:]
		out = self.linear_1(contextual_embeddings)
		out = torch.relu(out)
		out = self.linear_2(out)
		return F.softmax(out, dim=1)

	def predict(self, definitions_batch_encodings):
		self.eval()
		with torch.no_grad():
			log_probs = self.forward(definitions_batch_encodings)
			predicted_indices = torch.argmax(log_probs, dim=1).tolist()
		return [SUPERSENSES[i] for i in predicted_indices]

	def evaluate(self, examples_batch_encodings, DEVICE, dataset, predictions):
		self.eval()
		with torch.no_grad():
			X, Y = zip(*examples_batch_encodings)
			X = pad_batch(X, padding_token_id=PADDING_TOKEN_ID).to(DEVICE)
			Y_gold = torch.tensor(Y).to(DEVICE)
			Y_pred = torch.argmax(self.forward(X), dim=1)
			
			definitions = self.tokenizer.batch_decode(X, skip_special_tokens=True)
			gold = Y_gold.tolist()
			pred = Y_pred.tolist()
			
			predictions.extend(list(zip(definitions, pred, gold)))

		return torch.sum((Y_pred == Y_gold).int()).item()
		
	def evaluate_perf(self, X_input, Y, DEVICE):
		self.eval()
		with torch.no_grad():
			
			X_input = torch.tensor(X_input).to(DEVICE)
			Y_gold = torch.tensor(Y).to(DEVICE)
			Y_pred = torch.argmax(self.forward(X_input), dim=1)

		return torch.sum((Y_pred == Y_gold).int()).item()
		

def training(parameters, train_defs, train_supersenses, freq_dev_defs, freq_dev_supersenses, rand_dev_defs, rand_dev_supersenses, classifier, DEVICE, clf_file):

	my_supersense_tagger = classifier
	train_losses = []
	train_accuracies = []
	mean_dev_losses = []
	mean_dev_accuracies = []
	freq_dev_losses = []
	freq_dev_accuracies = []
	rand_dev_losses = []
	rand_dev_accuracies = []
	loss_function = nn.NLLLoss()
	patience = parameters["patience"]
	max_mean_dev_accuracy = 0
	
	optimizer = optim.Adam(my_supersense_tagger.parameters(), lr=parameters["lr"])

	for epoch in range(parameters["nb_epochs"]):
		print("epoch: ", epoch+1)
		
		epoch_loss = 0
		train_epoch_accuracy = 0
		freq_dev_epoch_loss = 0
		freq_dev_epoch_accuracy = 0
		rand_dev_epoch_loss = 0
		rand_dev_epoch_accuracy = 0
		
		
		train_examples = zip(train_defs, train_supersenses)
		train_examples = list(train_examples)
		shuffle(train_examples)
		train_defs, train_supersenses, = zip(*train_examples)

		shuffle(train_examples)
		i = 0
		j = 0

		my_supersense_tagger.train()
		while i < len(train_defs):
			X_train = train_defs[i: i + parameters["batch_size"]]
			Y_train = train_supersenses[i: i + parameters["batch_size"]]

			i += parameters["batch_size"]

			padded_encodings = pad_batch(X_train, padding_token_id=PADDING_TOKEN_ID).to(DEVICE)
			Y_train = torch.tensor(Y_train, dtype=torch.long).to(DEVICE)

			my_supersense_tagger.zero_grad()
			log_probs = my_supersense_tagger(padded_encodings)

			predicted_indices = torch.argmax(log_probs, dim=1)

			loss = loss_function(log_probs, Y_train)
			loss.backward()
			optimizer.step()

			epoch_loss += loss.item()/parameters["batch_size"]

		train_losses.append(epoch_loss)
		
		"""
		j = 0
		my_supersense_tagger.eval()
		with torch.no_grad():
			while j < len(train_defs):
				X_train = train_defs[i: i + parameters["batch_size"]]
				Y_train = train_supersenses[i: i + parameters["batch_size"]]
				
				j += parameters["batch_size"]

				train_padded_encodings = pad_batch(X_train, padding_token_id=PADDING_TOKEN_ID).to(DEVICE)
				Y_train = torch.tensor(Y_train, dtype=torch.long).to(DEVICE)
				train_log_probs = my_supersense_tagger(train_padded_encodings)

				predicted_indices = torch.argmax(train_log_probs, dim=1)
				train_epoch_accuracy += torch.sum((predicted_indices == Y_train).int()).item()

			train_accuracies.append(train_epoch_accuracy / len(train_examples))
		"""

		j = 0
		my_supersense_tagger.eval()
		with torch.no_grad():
			while j < len(freq_dev_defs):
			
				X_freq_dev = freq_dev_defs[j: j + parameters["batch_size"]]
				Y_freq_dev = freq_dev_supersenses[j: j + parameters["batch_size"]]
				
				j += parameters["batch_size"]

				freq_dev_padded_encodings = pad_batch(X_freq_dev, padding_token_id=PADDING_TOKEN_ID).to(DEVICE)
				Y_freq_dev = torch.tensor(Y_freq_dev, dtype=torch.long).to(DEVICE)
				freq_dev_log_probs = my_supersense_tagger(freq_dev_padded_encodings)

				predicted_indices = torch.argmax(freq_dev_log_probs, dim=1)
				freq_dev_epoch_accuracy += torch.sum((predicted_indices == Y_freq_dev).int()).item()

				freq_dev_loss = loss_function(freq_dev_log_probs, Y_freq_dev)
				freq_dev_epoch_loss += freq_dev_loss.item()

			freq_dev_losses.append(freq_dev_epoch_loss / len(freq_dev_defs))
			freq_dev_accuracies.append(freq_dev_epoch_accuracy / len(freq_dev_defs))
		
		j = 0
		my_supersense_tagger.eval()
		with torch.no_grad():
			while j < len(rand_dev_defs):
				
				X_rand_dev = rand_dev_defs[j: j + parameters["batch_size"]]
				Y_rand_dev = rand_dev_supersenses[j: j + parameters["batch_size"]]
				
				j += parameters["batch_size"]
				
				rand_dev_padded_encodings = pad_batch(X_rand_dev, padding_token_id=PADDING_TOKEN_ID).to(DEVICE)
				Y_rand_dev = torch.tensor(Y_rand_dev, dtype=torch.long).to(DEVICE)
				rand_dev_log_probs = my_supersense_tagger(rand_dev_padded_encodings)

				predicted_indices = torch.argmax(rand_dev_log_probs, dim=1)
				rand_dev_epoch_accuracy += torch.sum((predicted_indices == Y_rand_dev).int()).item()

				rand_dev_loss = loss_function(rand_dev_log_probs, Y_rand_dev)
				rand_dev_epoch_loss += rand_dev_loss.item()

			rand_dev_losses.append(rand_dev_epoch_loss / len(rand_dev_defs))
			rand_dev_accuracies.append(rand_dev_epoch_accuracy / len(rand_dev_defs))

		mean_dev_losses.append( (freq_dev_epoch_loss/len(freq_dev_defs) + rand_dev_epoch_loss/len(rand_dev_defs) ) / 2)
		mean_dev_accuracies.append( (freq_dev_epoch_accuracy/len(freq_dev_defs) + rand_dev_epoch_accuracy/len(rand_dev_defs) ) / 2)
		
		if epoch >= parameters["patience"]:
		
			if mean_dev_accuracies[epoch] > max_mean_dev_accuracy:
				max_mean_dev_accuracy = mean_dev_accuracies[epoch]
				torch.save(my_supersense_tagger.state_dict(), clf_file)
				patience = parameters["patience"]
				
			else:
				patience = patience - 1
			
			if patience == 0:
				print("EARLY STOPPING : epoch ", epoch+1)
				break
		else:
			if mean_dev_accuracies[epoch] > max_mean_dev_accuracy:
				max_mean_dev_accuracy = mean_dev_accuracies[epoch]
			torch.save(my_supersense_tagger.state_dict(), clf_file)



def evaluation(examples, classifier, parameters, DEVICE, dataset, data, exp):
	batch_size = parameters['batch_size']
	predictions_file = f'./{exp}/{data["clf_id"]}_{dataset}_predictions.txt'
	predictions = []
	i = 0
	nb_good_preds = 0
	while i < len(examples):
		evaluation_batch = examples[i: i + batch_size]
		i += batch_size
		partial_nb_good_preds= classifier.evaluate(evaluation_batch, DEVICE, dataset, predictions)
		nb_good_preds += partial_nb_good_preds

	data[f"{dataset}_accuracy"] = nb_good_preds/len(examples)
	
	with open(predictions_file, 'w', encoding='utf-8') as f:
		f.write("definition\tpred\tgold\n")

		for definition, pred, gold in predictions:
			f.write(f"{definition}\t{SUPERSENSES[pred]}\t{SUPERSENSES[gold]}\n")


def save_best_clf(max_perf, freq_inputs, freq_supersenses, rand_inputs, rand_supersenses, classifier, parameters, path, DEVICE):
	batch_size = parameters['batch_size']
	i = 0
	nb_good_preds_freq = 0
	
	while i < len(freq_inputs):

		X_freq_input = freq_inputs[i: i + batch_size]
		Y_freq = freq_supersenses[i: i + batch_size]

		i += batch_size
		
		partial_nb_good_preds_freq = classifier.evaluate_perf(X_freq_input, Y_freq, DEVICE)
		nb_good_preds_freq += partial_nb_good_preds_freq
	
	j = 0
	nb_good_preds_rand = 0
	
	while j < len(rand_inputs):

		X_rand_input = rand_inputs[j: j + batch_size]
		Y_rand = rand_supersenses[j: j + batch_size]

		j += batch_size
		
		partial_nb_good_preds_rand = classifier.evaluate_perf(X_rand_input, Y_rand, DEVICE)
		nb_good_preds_rand += partial_nb_good_preds_rand
	
	
	
	current_perf = 0.6*(nb_good_preds_freq / len(freq_inputs)) + 0.4*(nb_good_preds_rand / len(rand_inputs)) 
	
	if current_perf > max_perf:
		torch.save(classifier.state_dict(), path)
		max_perf = current_perf
		
	return max_perf




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


