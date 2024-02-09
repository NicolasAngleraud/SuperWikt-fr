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


# Definition of the supersenses and index structure
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
MODEL_NAME = "flaubert/flaubert_base_cased"
PADDING_TOKEN_ID = 2


def encoded_examples(datafile, eval_prefix):
	df_senses = pd.read_excel(datafile, sheet_name='senses', engine='openpyxl')
	df_senses = df_senses[df_senses['supersense'].isin(SUPERSENSES)]
	tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
	
	dev = f"{eval_prefix}-dev"
	test = f"{eval_prefix}-test"
	
	train_definitions = df_senses[df_senses['set']=='train']['definition'].tolist()
	train_supersenses = df_senses[df_senses['set']=='train']['supersense'].tolist()
	train_lemmas = df_senses[df_senses['set']=='train']['lemma'].tolist()
	train_definitions_encoded = [tokenizer.encode(text=f"{lemma}: {definition}", add_special_tokens=True) for definition, lemma in zip(train_definitions, train_lemmas)]
	train_supersenses_encoded = [supersense2i[supersense] for supersense in train_supersenses]
	train_examples = list(zip(train_definitions_encoded, train_supersenses_encoded))

	dev_definitions = df_senses[df_senses['set']==dev]['definition'].tolist()
	dev_supersenses = df_senses[df_senses['set']==dev]['supersense'].tolist()
	dev_lemmas = df_senses[df_senses['set']==dev]['lemma'].tolist()
	dev_definitions_encoded = [tokenizer.encode(text=f"{lemma}: {definition}", add_special_tokens=True) for definition, lemma in zip(dev_definitions, dev_lemmas)]
	dev_supersenses_encoded = [supersense2i[supersense] for supersense in dev_supersenses]
	dev_examples = list(zip(dev_definitions_encoded, dev_supersenses_encoded))
	
	test_definitions = df_senses[df_senses['set']==test]['definition'].tolist()
	test_supersenses = df_senses[df_senses['set']==test]['supersense'].tolist()
	test_lemmas = df_senses[df_senses['set']==test]['lemma'].tolist()
	test_definitions_encoded = [tokenizer.encode(text=f"{lemma}: {definition}", add_special_tokens=True) for definition, lemma in zip(test_definitions, test_lemmas)]
	test_supersenses_encoded = [supersense2i[supersense] for supersense in test_supersenses]
	test_examples = list(zip(test_definitions_encoded, test_supersenses_encoded))
	
	return train_examples, dev_examples, test_examples


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

	def __init__(self, params, DEVICE, dropout_rate=0.5, bert_model_name=MODEL_NAME):
		super(SupersenseTagger, self).__init__()

		self.bert_model = AutoModel.from_pretrained(bert_model_name, output_attentions=True).to(DEVICE)

		if params["frozen"]:
			for param in self.bert_model.parameters():
				param.requires_grad = False

		self.embedding_layer_size = self.bert_model.config.hidden_size

		self.hidden_layer_size = params['hidden_layer_size']

		self.output_size = NB_CLASSES

		self.linear_1 = nn.Linear(self.embedding_layer_size, self.hidden_layer_size).to(DEVICE)

		self.linear_2 = nn.Linear(self.hidden_layer_size, self.output_size).to(DEVICE)

		self.dropout = nn.Dropout(dropout_rate).to(DEVICE)

		self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

	def forward(self, padded_encodings):

		bert_output = self.bert_model(padded_encodings, return_dict=True) # SHAPE [len(definitions), max_length, embedding_size]

		batch_contextual_embeddings = bert_output.last_hidden_state[:,0,:] # from [batch_size , max_seq_length, plm_emb_size] to [batch_size, plm_emb_size]

		out = self.linear_1(batch_contextual_embeddings) # SHAPE [len(definitions), hidden_layer_size]
		
		out = self.dropout(out)

		out = torch.relu(out) # SHAPE [len(definitions), hidden_layer_size]

		out = self.linear_2(out) # SHAPE [len(definitions), nb_classes]

		out = self.dropout(out)

		return F.log_softmax(out, dim=1)

	def predict(self, definitions_batch_encodings):
		self.eval()
		with torch.no_grad():
			log_probs = self.forward(definitions_batch_encodings)
			predicted_indices = torch.argmax(log_probs, dim=1).tolist()
		return [SUPERSENSES[i] for i in predicted_indices]

	def evaluate(self, examples_batch_encodings, DEVICE, run, dataset):
		self.eval()
		with torch.no_grad():
			X, Y = zip(*examples_batch_encodings)
			X = pad_batch(X, padding_token_id=PADDING_TOKEN_ID).to(DEVICE)
			Y_gold = torch.tensor(Y).to(DEVICE)
			Y_pred = torch.argmax(self.forward(X), dim=1)

		return torch.sum((Y_pred == Y_gold).int()).item()

def training(parameters, train_examples, dev_examples, classifier, DEVICE, dev_data, test_data):

	for param, value in parameters.items():
		dev_data[param] = value
		test_data[param] = value

	my_supersense_tagger = classifier

	train_losses = []
	train_accuracies = []

	dev_losses = []
	dev_accuracies = []
	loss_function = nn.NLLLoss()

	optimizer = optim.Adam(my_supersense_tagger.parameters(), lr=parameters["lr"])

	for epoch in range(parameters["nb_epochs"]):
		epoch_loss = 0
		dev_epoch_loss = 0

		train_epoch_accuracy = 0
		dev_epoch_accuracy = 0

		shuffle(train_examples)
		i = 0
		j = 0

		my_supersense_tagger.train()
		while i < len(train_examples):
			train_batch = train_examples[i: i + parameters["batch_size"]]

			i += parameters["batch_size"]

			X_train, Y_train = zip(*train_batch)

			padded_encodings = pad_batch(X_train, padding_token_id=PADDING_TOKEN_ID).to(DEVICE)
			Y_train = torch.tensor(Y_train, dtype=torch.long).to(DEVICE)

			my_supersense_tagger.zero_grad()
			log_probs = my_supersense_tagger(padded_encodings)

			predicted_indices = torch.argmax(log_probs, dim=1)

			loss = loss_function(log_probs, Y_train)
			loss.backward()
			optimizer.step()

			epoch_loss += loss.item()

		train_losses.append(epoch_loss)
		
		
		my_supersense_tagger.eval()
		with torch.no_grad():
			while j < len(train_examples):
				train_batch = train_examples[j: j + parameters["batch_size"]]
				j += parameters["batch_size"]
				X_train, Y_train = zip(*train_batch)
				train_padded_encodings = pad_batch(X_train, padding_token_id=PADDING_TOKEN_ID).to(DEVICE)
				Y_train = torch.tensor(Y_train, dtype=torch.long).to(DEVICE)
				train_log_probs = my_supersense_tagger(train_padded_encodings)

				predicted_indices = torch.argmax(train_log_probs, dim=1)
				train_epoch_accuracy += torch.sum((predicted_indices == Y_train).int()).item()

			train_accuracies.append(train_epoch_accuracy / len(train_examples))
		
		j=0
		my_supersense_tagger.eval()
		with torch.no_grad():
			while j < len(dev_examples):
				dev_batch = dev_examples[j: j + parameters["batch_size"]]
				j += parameters["batch_size"]
				X_dev, Y_dev = zip(*dev_batch)
				dev_padded_encodings = pad_batch(X_dev, padding_token_id=PADDING_TOKEN_ID).to(DEVICE)
				Y_dev = torch.tensor(Y_dev, dtype=torch.long).to(DEVICE)
				dev_log_probs = my_supersense_tagger(dev_padded_encodings)

				predicted_indices = torch.argmax(dev_log_probs, dim=1)
				dev_epoch_accuracy += torch.sum((predicted_indices == Y_dev).int()).item()

				dev_loss = loss_function(dev_log_probs, Y_dev)
				dev_epoch_loss += dev_loss.item()

			dev_losses.append(dev_epoch_loss)
			dev_accuracies.append(dev_epoch_accuracy / len(dev_examples))

			if epoch > parameters["patience"]:
				if all(dev_losses[i] > dev_losses[i - 1] for i in range(-1, -parameters["patience"]-1, -1)):
					dev_data["early_stopping"] = epoch
					test_data["early_stopping"] = epoch
					break

	dev_data["train_losses"] = [round(train_loss, 2) for train_loss in train_losses]
	test_data["train_losses"] = [round(train_loss, 2) for train_loss in train_losses]

	dev_data["dev_losses"] = [round(dev_loss, 2) for dev_loss in dev_losses]
	test_data["dev_losses"] = [round(dev_loss, 2) for dev_loss in dev_losses]

	dev_data["train_accuracies"] = [round(train_accuracy, 2) for train_accuracy in train_accuracies]
	test_data["train_accuracies"] = [round(train_accuracy, 2) for train_accuracy in train_accuracies]

	dev_data["dev_accuracies"] = [round(dev_accuracy, 2) for dev_accuracy in dev_accuracies]
	test_data["dev_accurcies"] = [round(dev_accuracy, 2) for dev_accuracy in dev_accuracies]


def evaluation(examples, classifier, parameters, DEVICE, run, dataset, data):
	batch_size = parameters['batch_size']
	i = 0
	nb_good_preds = 0
	while i < len(examples):
		evaluation_batch = examples[i: i + batch_size]
		i += batch_size
		partial_nb_good_preds= classifier.evaluate(evaluation_batch, DEVICE, run, dataset)
		nb_good_preds += partial_nb_good_preds

	data["accuracy"] = nb_good_preds/len(examples)

	



def inference(inference_data_set, classifier, DEVICE):
    pass


class Baseline:

	def __init__(self):
		self.most_frequent_supersense = None
	
	def training(self):
		pass

	def evaluation(self, eval_examples):
		correct_pred = 0
		nb_examples = 0
		i = 0
		
		for _, supersense in eval_examples:
			if i < 10: print(supersense, self.most_frequent_supersense)
			i += 1
			nb_examples += 1
			if supersense == self.most_frequent_supersense:
				correct_pred += 1

			return correct_pred / nb_examples


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


