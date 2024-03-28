import pandas as pd
from collections import Counter, defaultdict
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sacremoses
from random import shuffle
import numpy as np
import spacy
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



def flatten_list(lst):

	return [item for sublist in lst for item in (sublist if isinstance(sublist, list) else [sublist])]



def truncate_batch_def(sentences, word_ranks, max_length=100):

	max_length = max_length - 2
	trunc_sentences = [[tok for i, tok in enumerate(sent) if i<max_length] for sent in sentences]

	return trunc_sentences

def truncate_batch_ex(sentences, word_ranks, max_length=100):
	# Adjust max_length to account for potential special tokens
	max_length = max_length - 2

	trunc_sentences = []
	new_word_ranks = []

	for sent, target_index in zip(sentences, word_ranks):
		if len(sent) <= max_length:
			# No truncation needed
			trunc_sentences.append(sent)
			new_word_ranks.append(target_index)  # The target index remains the same
		else:
			# Calculate the number of tokens to keep before and after the target_index
			half_max_length = max_length // 2
			start_index = max(0, min(len(sent) - max_length, target_index - half_max_length))
			end_index = start_index + max_length

			# Truncate the sentence
			trunc_sent = sent[start_index:end_index]
			trunc_sentences.append(trunc_sent)

			# Adjust the target index based on truncation
			new_target_index = target_index - start_index
			# Ensure the new target index does not exceed the bounds of the truncated sentence
			new_target_index = max(0, min(new_target_index, max_length-1))
			new_word_ranks.append(new_target_index)

	return trunc_sentences, new_word_ranks



def pad_batch(sentences, pad_id=2, max_length=100):

	max_length = max_length - 2
	pad_lengths = [ max_length - len(sent) if max_length >= len(sent) else 0 for sent in sentences ]

	padded_sentences = [ [el for el in sent] + pad_lengths[i] * [pad_id] for i, sent in enumerate(sentences) ]


	return padded_sentences



def add_special_tokens_batch(sentences, ranks, cls_id=0, sep_id=1):

	sentences_with_special_tokens = [ [cls_id] + [tok for tok in sent] + [sep_id] for sent in sentences ]
	rks = [rk + 1 for rk in ranks]

	return sentences_with_special_tokens, rks

def token_rank(lst, index):
	count = 0
	for i in range(index):
		count += len(lst[i])
	return count


def encoded_examples(datafile, set_, max_length=100):
	df_examples = pd.read_excel(datafile, sheet_name='examples', engine='openpyxl')
	df_examples = df_examples[df_examples['supersense'].isin(SUPERSENSES)]
	df_examples = df_examples[df_examples['word_rank'] >= 0]
	df_examples = df_examples[(df_examples['example'] != "") & (df_examples['example'].notna())]

	tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
	
	examples = df_examples[df_examples['set']==set_]['example'].tolist()

	examples = [ x.split(' ') for x in examples ]
	for example in examples:
		for x in example:
			x = x.replace('##', ' ')
	
	supersenses = df_examples[df_examples['set']==set_]['supersense'].tolist()
	senses_ids = df_examples[df_examples['set']==set_]['sense_id'].tolist()
	lemmas = df_examples[df_examples['set']==set_]['lemma'].tolist()
	ranks = df_examples[df_examples['set']==set_]['word_rank'].tolist()
	
	sents_encoded = [ tokenizer(word, add_special_tokens=False)['input_ids'] for word in examples ]
	
	tg_trks = [token_rank(sent, rank) for sent, rank in zip(sents_encoded, ranks)]
	bert_input_raw = [ flatten_list(sent) for sent in sents_encoded ]
	bert_input_raw, tg_trks = truncate_batch_ex(bert_input_raw, tg_trks, max_length)
	bert_input_raw= pad_batch(bert_input_raw, pad_id=2, max_length=max_length)
	bert_input, tg_trks = add_special_tokens_batch(bert_input_raw, tg_trks, cls_id=0, sep_id=1)
	supersenses_encoded = [supersense2i[supersense] for supersense in supersenses]

	return bert_input, tg_trks, supersenses_encoded, senses_ids, lemmas


def encoded_definitions(datafile, nlp, set_, max_length=100):
	df_senses = pd.read_excel(datafile, sheet_name='senses', engine='openpyxl')
	df_senses = df_senses[df_senses['supersense'].isin(SUPERSENSES)]
	df_senses = df_senses[(df_senses['definition'] != "") & (df_senses['definition'].notna())]

	tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
	lemmas = df_senses[df_senses['set']==set_]['lemma'].tolist()
	
	definitions = df_senses[df_senses['set']==set_]['definition'].tolist()
	
	definitions = [ [lemma]+[' : ']+[token.text for token in nlp(x)] for x, lemma in zip(definitions, lemmas) ]
	
	supersenses = df_senses[df_senses['set']==set_]['supersense'].tolist()
	senses_ids = df_senses[df_senses['set']==set_]['sense_id'].tolist()
	
	ranks = [0]*len(definitions)
	
	sentences_wrks = [ [sent[:max_length], ranks[i]] for i, sent in enumerate(definitions) ]

	sentences = [inner[0] for inner in sentences_wrks]
	tg_wrks = [inner[1] for inner in sentences_wrks]
	sents_encoded = [ tokenizer(word, add_special_tokens=False)['input_ids'] for word in sentences ]
	bert_input_raw = [ flatten_list(sent) for sent in sents_encoded ]
	bert_input_raw = truncate_batch_def(bert_input_raw, tg_wrks, max_length)
	bert_input_raw = pad_batch(bert_input_raw, pad_id=2, max_length=max_length)
	bert_input, tg_wrks = add_special_tokens_batch(bert_input_raw, tg_wrks, cls_id=0, sep_id=1)
	supersenses_encoded = [supersense2i[supersense] for supersense in supersenses]

	return bert_input, tg_wrks, supersenses_encoded, senses_ids, lemmas



class SupersenseTagger(nn.Module):

	def __init__(self, params, DEVICE, dropout_rate=0.1, bert_model_name=MODEL_NAME):
		super(SupersenseTagger, self).__init__()

		self.bert_model = AutoModel.from_pretrained(bert_model_name).to(DEVICE)

		if params["frozen"]:
			for param in self.bert_model.parameters():
				param.requires_grad = False

		self.embedding_layer_size = self.bert_model.config.hidden_size

		self.hidden_layer_size = params['hidden_layer_size']

		self.output_size = NB_CLASSES

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
		
	def forward_encoding(self, encoding, trk):
		encoding = torch.tensor(encoding).to(self.device)
		bert_output = self.bert_model(encoding.unsqueeze(0), return_dict=True)
		contextual_embeddings = bert_output.last_hidden_state[trk,:]
		out = self.linear_1(contextual_embeddings)
		out = torch.relu(out)
		out = self.linear_2(out)
		return F.softmax(out, dim=1)

	def predict(self, examples_batch_encodings):
		self.eval()
		with torch.no_grad():
			log_probs = self.forward(examples_batch_encodings)
			predicted_indices = torch.argmax(log_probs, dim=1).tolist()
		return [SUPERSENSES[i] for i in predicted_indices]

	def evaluate(self, X_input, X_rank, Y, senses_ids, lemmas, DEVICE, dataset, predictions):
		self.eval()
		with torch.no_grad():
			
			X_input = torch.tensor(X_input).to(DEVICE)
			X_rank = torch.tensor(X_rank).to(DEVICE)
			Y_gold = torch.tensor(Y).to(DEVICE)
			Y_pred = torch.argmax(self.forward(X_input, X_rank), dim=1)
			
			examples = self.tokenizer.batch_decode(X_input, skip_special_tokens=True)
			gold = Y_gold.tolist()
			pred = Y_pred.tolist()
			
			predictions.extend(list(zip(examples, pred, gold, senses_ids, lemmas)))

		return torch.sum((Y_pred == Y_gold).int()).item()
	
	def evaluate_perf(self, X_input, X_rank, Y, DEVICE):
		self.eval()
		with torch.no_grad():
			
			X_input = torch.tensor(X_input).to(DEVICE)
			X_rank = torch.tensor(X_rank).to(DEVICE)
			Y_gold = torch.tensor(Y).to(DEVICE)
			Y_pred = torch.argmax(self.forward(X_input, X_rank), dim=1)

		return torch.sum((Y_pred == Y_gold).int()).item()

		

def training(parameters, train_inputs, train_ranks, train_supersenses, train_senses_ids, train_lemmas, freq_dev_inputs, freq_dev_ranks, freq_dev_supersenses, freq_dev_senses_ids, freq_dev_lemmas, rand_dev_inputs, rand_dev_ranks, rand_dev_supersenses, rand_dev_senses_ids, rand_dev_lemmas, classifier, DEVICE, clf_file):


	my_supersense_tagger = classifier
	train_losses = []
	# train_accuracies = []
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
		
		train_examples = zip(train_inputs, train_ranks, train_supersenses, train_senses_ids, train_lemmas)
		train_examples = list(train_examples)
		shuffle(train_examples)
		train_input, train_rank, train_supersense, train_sense_id, train_lemma = zip(*train_examples)
		
		
		i = 0
		j = 0

		my_supersense_tagger.train()
		while i < len(train_input):
			X_train_input = torch.tensor(train_input[i: i + parameters["batch_size"]]).to(DEVICE)
			X_train_rank = torch.tensor(train_rank[i: i + parameters["batch_size"]]).to(DEVICE)
			Y_train = torch.tensor(train_supersense[i: i + parameters["batch_size"]]).to(DEVICE)
			
			i += parameters["batch_size"]

			my_supersense_tagger.zero_grad()
			log_probs = my_supersense_tagger(X_train_input, X_train_rank)

			# predicted_indices = torch.argmax(log_probs, dim=1)

			loss = loss_function(log_probs, Y_train)
			loss.backward()
			optimizer.step()

			epoch_loss += loss.item()/parameters["batch_size"]

		train_losses.append(epoch_loss)
		
		"""
		j = 0
		my_supersense_tagger.eval()
		with torch.no_grad():
			while j < len(train_input):
				X_train_input = torch.tensor(train_input[j: j + parameters["batch_size"]]).to(DEVICE)
				X_train_rank = torch.tensor(train_rank[j: j + parameters["batch_size"]]).to(DEVICE)
				Y_train = torch.tensor(train_supersense[j: j + parameters["batch_size"]]).to(DEVICE)
				
				j += parameters["batch_size"]
				
				train_log_probs = my_supersense_tagger(X_train_input, X_train_rank)

				predicted_indices = torch.argmax(train_log_probs, dim=1)
				train_epoch_accuracy += torch.sum((predicted_indices == Y_train).int()).item()

			train_accuracies.append(train_epoch_accuracy / len(train_input))
		"""


		j = 0
		my_supersense_tagger.eval()
		with torch.no_grad():
			while j < len(freq_dev_inputs):
				X_freq_dev_input = torch.tensor(freq_dev_inputs[j: j + parameters["batch_size"]]).to(DEVICE)
				X_freq_dev_rank = torch.tensor(freq_dev_ranks[j: j + parameters["batch_size"]]).to(DEVICE)
				Y_freq_dev = torch.tensor(freq_dev_supersenses[j: j + parameters["batch_size"]]).to(DEVICE)
				
				j += parameters["batch_size"]
				
				freq_dev_log_probs = my_supersense_tagger(X_freq_dev_input, X_freq_dev_rank)

				predicted_indices = torch.argmax(freq_dev_log_probs, dim=1)
				freq_dev_epoch_accuracy += torch.sum((predicted_indices == Y_freq_dev).int()).item()

				freq_dev_loss = loss_function(freq_dev_log_probs, Y_freq_dev)
				freq_dev_epoch_loss += freq_dev_loss.item()

			freq_dev_losses.append(freq_dev_epoch_loss / len(freq_dev_inputs))
			freq_dev_accuracies.append(freq_dev_epoch_accuracy / len(freq_dev_inputs))
		
		j = 0
		my_supersense_tagger.eval()
		with torch.no_grad():
			while j < len(rand_dev_inputs):
				X_rand_dev_input = torch.tensor(rand_dev_inputs[j: j + parameters["batch_size"]]).to(DEVICE)
				X_rand_dev_rank = torch.tensor(rand_dev_ranks[j: j + parameters["batch_size"]]).to(DEVICE)
				Y_rand_dev = torch.tensor(rand_dev_supersenses[j: j + parameters["batch_size"]]).to(DEVICE)
				
				j += parameters["batch_size"]
				
				rand_dev_log_probs = my_supersense_tagger(X_rand_dev_input, X_rand_dev_rank)

				predicted_indices = torch.argmax(rand_dev_log_probs, dim=1)
				rand_dev_epoch_accuracy += torch.sum((predicted_indices == Y_rand_dev).int()).item()

				rand_dev_loss = loss_function(rand_dev_log_probs, Y_rand_dev)
				rand_dev_epoch_loss += rand_dev_loss.item()

			rand_dev_losses.append(rand_dev_epoch_loss / len(rand_dev_inputs))
			rand_dev_accuracies.append(rand_dev_epoch_accuracy / len(rand_dev_inputs))

		mean_dev_losses.append( (freq_dev_epoch_loss/len(freq_dev_inputs) + rand_dev_epoch_loss/len(rand_dev_inputs) ) / 2)
		mean_dev_accuracies.append( (freq_dev_epoch_accuracy/len(freq_dev_inputs) + rand_dev_epoch_accuracy/len(rand_dev_inputs) ) / 2)
		
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


def evaluation(eval_inputs, eval_ranks, eval_supersenses, eval_senses_ids, eval_lemmas, classifier, parameters, DEVICE, dataset, data, exp):
	batch_size = parameters['batch_size']
	predictions_file = f'./{exp}/{data["clf_id"]}_{dataset}_predictions.text'
	predictions = []
	i = 0
	nb_good_preds = 0
	
	while i < len(eval_inputs):

		X_input = eval_inputs[i: i + batch_size]
		X_rank = eval_ranks[i: i + batch_size]
		Y = eval_supersenses[i: i + batch_size]
		senses_ids = eval_senses_ids[i: i + batch_size]
		lemmas = eval_lemmas[i: i + batch_size]
		
		i += batch_size
		partial_nb_good_preds= classifier.evaluate(X_input, X_rank, Y, senses_ids, lemmas, DEVICE, dataset, predictions)
		nb_good_preds += partial_nb_good_preds

	data[f"{dataset}_accuracy"] = nb_good_preds/len(eval_inputs)
	
	with open(predictions_file, 'w', encoding='utf-8') as f:
		f.write("example\tpred\tgold\tsense_id\tlemma\n")

		for example, pred, gold, sense_id, lemma in predictions:
			f.write(f"{example}\t{SUPERSENSES[pred]}\t{SUPERSENSES[gold]}\t{sense_id}\t{lemma}\n")


def save_best_clf(max_perf, freq_inputs, freq_ranks, freq_supersenses, rand_inputs, rand_ranks, rand_supersenses, classifier, parameters, path, DEVICE):
	batch_size = parameters['batch_size']
	i = 0
	nb_good_preds_freq = 0
	
	while i < len(freq_inputs):

		X_freq_input = freq_inputs[i: i + batch_size]
		X_freq_rank = freq_ranks[i: i + batch_size]
		Y_freq = freq_supersenses[i: i + batch_size]

		i += batch_size
		
		partial_nb_good_preds_freq = classifier.evaluate_perf(X_freq_input, X_freq_rank, Y_freq, DEVICE)
		nb_good_preds_freq += partial_nb_good_preds_freq
	
	j = 0
	nb_good_preds_rand = 0
	
	while j < len(rand_inputs):

		X_rand_input = rand_inputs[j: j + batch_size]
		X_rand_rank = rand_ranks[j: j + batch_size]
		Y_rand = rand_supersenses[j: j + batch_size]

		j += batch_size
		
		partial_nb_good_preds_rand = classifier.evaluate_perf(X_rand_input, X_rand_rank, Y_rand, DEVICE)
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

	def evaluation(self, eval_supersenses):
		return sum([int(supersense == supersense2i[self.most_frequent_supersense]) for supersense in eval_supersenses])/len(eval_supersenses)



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


