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



def truncate_batch(sentences, word_ranks, index_map, max_length=100):

	max_length = max_length - 2
	trunc_sentences = [[tok for i, tok in enumerate(sent) if i<max_length] for sent in sentences]
	trunc_index_map = [[id for i, id in enumerate(sent) if i<max_length] for sent in index_map]

	return trunc_sentences, trunc_index_map



def pad_batch(sentences, index_map, pad_id=2, max_length=100):

	max_length = max_length - 2
	pad_lengths = [ max_length - len(sent) if max_length >= len(sent) else 0 for sent in sentences ]

	padded_sentences = [ [el for el in sent] + pad_lengths[i] * [pad_id] for i, sent in enumerate(sentences) ]
	padded_index_map = [ [el for el in sent] + pad_lengths[i] * [0] for i, sent in enumerate(index_map) ]

	return padded_sentences, padded_index_map



def add_special_tokens_batch(sentences,index_map, cls_id=0, sep_id=1):

	sentences_with_special_tokens = [ [cls_id] + [tok for tok in sent] + [sep_id] for sent in sentences ]
	index_map_with_special_tokens = [ [0] + [tok for tok in sent] + [0] for sent in index_map ]

	return sentences_with_special_tokens, index_map_with_special_tokens



def encoded_examples(datafile, set_, max_length=100):
	df_examples = pd.read_excel(datafile, sheet_name='examples', engine='openpyxl')
	df_examples = df_examples[df_examples['supersense'].isin(SUPERSENSES)]
	df_examples = df_examples[(df_examples['example'] != "") & (df_examples['example'].notna())]

	tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
	
	examples = df_examples[df_examples['set']==set_]['example'].tolist()
	# examples = [example.replace('{ { exemple|lang = fr|', '').replace('{ { exemple|', '').replace('{ { exemple', '').replace('<br', '').replace('lang = fr', '') for example in examples]
	examples = [ x.split(' ') for x in examples ]
	for example in examples:
		for x in example:
			x = x.replace('##', ' ')
	
	supersenses = df_examples[df_examples['set']==set_]['supersense'].tolist()
	senses_ids = df_examples[df_examples['set']==set_]['sense_id'].tolist()
	lemmas = df_examples[df_examples['set']==set_]['lemma'].tolist()
	ranks = df_examples[df_examples['set']==set_]['word_rank'].tolist()
	ranks = [rank + 1 for rank in ranks]
		
	sentences_wrks = [ [sent[:max_length], ranks[i]] if ranks[i]<max_length//2
		    else [sent[-max_length:], ranks[i]-(len(sent)-max_length+1) -1*(max_length%2)] if ranks[i]>len(sent)-(max_length//2)
		    else [sent[ranks[i]-(max_length//2):ranks[i]+(max_length//2)], (max_length//2)]
		    for i, sent in enumerate(examples) ]

	sentences = [inner[0] for inner in sentences_wrks]
	tg_wrks = [inner[1] for inner in sentences_wrks]
	sents_encoded = [ tokenizer(word, add_special_tokens=False)['input_ids'] for word in sentences ]
	index_map_raw = [ flatten_list([len(word_toks)*[(i+1)] for i, word_toks in enumerate(sent)]) for sent in sents_encoded ]
	bert_input_raw = [ flatten_list(sent) for sent in sents_encoded ]
	bert_input_raw, index_map_raw = truncate_batch(bert_input_raw, tg_wrks, index_map_raw, max_length)
	bert_input_raw, index_map_raw = pad_batch(bert_input_raw, index_map_raw, pad_id=2, max_length=max_length)
	bert_input, index_map = add_special_tokens_batch(bert_input_raw, index_map_raw, cls_id=0, sep_id=1)
	supersenses_encoded = [supersense2i[supersense] for supersense in supersenses]

	return zip(bert_input, tg_wrks, index_map, supersenses_encoded, senses_ids, lemmas, ranks)


def encoded_definitions(datafile, nlp, set_, max_length=100):
	df_senses = pd.read_excel(datafile, sheet_name='senses', engine='openpyxl')
	df_senses = df_senses[df_senses['supersense'].isin(SUPERSENSES)]
	df_senses = df_senses[(df_senses['definition'] != "") & (df_senses['definition'].notna())]

	tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
	lemmas = df_senses[df_senses['set']==set_]['lemma'].tolist()
	
	definitions = df_senses[df_senses['set']==set_]['definition'].tolist()
	# examples = [example.replace('{ { exemple|lang = fr|', '').replace('{ { exemple|', '').replace('{ { exemple', '').replace('<br', '').replace('lang = fr', '') for example in examples]
	
	definitions = [ [lemma]+[' : ']+[token.text for token in nlp(x)] for x, lemma in zip(definitions, lemmas) ]
	
	supersenses = df_senses[df_senses['set']==set_]['supersense'].tolist()
	senses_ids = df_senses[df_senses['set']==set_]['sense_id'].tolist()
	
	ranks = [1]*len(definitions)
	
	sentences_wrks = [ [sent[:max_length], ranks[i]] for i, sent in enumerate(definitions) ]

	sentences = [inner[0] for inner in sentences_wrks]
	tg_wrks = [inner[1] for inner in sentences_wrks]
	sents_encoded = [ tokenizer(word, add_special_tokens=False)['input_ids'] for word in sentences ]
	index_map_raw = [ flatten_list([len(word_toks)*[(i+1)] for i, word_toks in enumerate(sent)]) for sent in sents_encoded ]
	bert_input_raw = [ flatten_list(sent) for sent in sents_encoded ]
	bert_input_raw, index_map_raw = truncate_batch(bert_input_raw, tg_wrks, index_map_raw, max_length)
	bert_input_raw, index_map_raw = pad_batch(bert_input_raw, index_map_raw, pad_id=2, max_length=max_length)
	bert_input, index_map = add_special_tokens_batch(bert_input_raw, index_map_raw, cls_id=0, sep_id=1)
	supersenses_encoded = [supersense2i[supersense] for supersense in supersenses]

	return zip(bert_input, tg_wrks, index_map, supersenses_encoded, senses_ids, lemmas)



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

	def forward(self, X_input, X_rank, X_idxmap):
	
		batch_size = self.params['batch_size']
		max_length = self.params['max_length']
		bert_emb_size = self.embedding_layer_size

		bert_tok_embeddings = self.bert_model(input_ids=X_input).last_hidden_state # [batch_size, max_length, bert_emb_size]
		bert_word_embeddings = bert_tok_embeddings.new_zeros((batch_size, max_length, bert_emb_size)).to(self.device) # [batch_size, max_length, bert_emb_size]
		bert_word_embeddings = scatter_mean(bert_tok_embeddings, X_idxmap, out=bert_word_embeddings, dim=1) # [batch_size, max_length, bert_emb_size]
		bert_target_word_embeddings = bert_word_embeddings[torch.arange(bert_word_embeddings.size(0)), X_rank] # [batch_size, bert_emb_size]

		out = self.linear_1(bert_target_word_embeddings) # SHAPE [len(definitions), hidden_layer_size]
		
		out = self.dropout(out)

		out = torch.relu(out) # SHAPE [len(definitions), hidden_layer_size]

		out = self.linear_2(out) # SHAPE [len(definitions), nb_classes]

		return F.log_softmax(out, dim=1)

	def predict(self, examples_batch_encodings):
		self.eval()
		with torch.no_grad():
			log_probs = self.forward(examples_batch_encodings)
			predicted_indices = torch.argmax(log_probs, dim=1).tolist()
		return [SUPERSENSES[i] for i in predicted_indices]

	def evaluate(self, X_input, X_rank, X_idxmap, Y, senses_ids, lemmas, DEVICE, dataset, predictions):
		self.eval()
		with torch.no_grad():
			
			X_input = torch.tensor(X_input).to(DEVICE)
			X_rank = torch.tensor(X_rank).to(DEVICE)
			X_idxmap = torch.tensor(X_idxmap).to(DEVICE)
			Y_gold = torch.tensor(Y).to(DEVICE)
			Y_pred = torch.argmax(self.forward(X_input, X_rank, X_idxmap), dim=1)
			
			examples = self.tokenizer.batch_decode(X_input, skip_special_tokens=True)
			gold = Y_gold.tolist()
			pred = Y_pred.tolist()
			
			predictions.extend(list(zip(examples, pred, gold, senses_ids, lemmas)))

		return torch.sum((Y_pred == Y_gold).int()).item()

def training(parameters, train_examples, freq_dev_examples, rand_dev_examples, classifier, DEVICE, eval_data, clf_file):

	for param, value in parameters.items():
		eval_data[param] = value

	my_supersense_tagger = classifier
	eval_data["early_stopping"] = 0
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
	
	freq_dev_examples = list(freq_dev_examples)
	freq_dev_input, freq_dev_rank, freq_dev_idxmap, freq_dev_supersense, _, _ = zip(*freq_dev_examples)
	rand_dev_examples = list(rand_dev_examples)
	rand_dev_input, rand_dev_rank, rand_dev_idxmap, rand_dev_supersense, _, _ = zip(*rand_dev_examples)

	for epoch in range(parameters["nb_epochs"]):
		print("epoch: ", epoch+1)
		
		epoch_loss = 0
		train_epoch_accuracy = 0
		freq_dev_epoch_loss = 0
		freq_dev_epoch_accuracy = 0
		rand_dev_epoch_loss = 0
		rand_dev_epoch_accuracy = 0
		
		train_examples = list(train_examples)
		shuffle(train_examples)
		train_input, train_rank, train_idxmap, train_supersense, _, _ = zip(*train_examples)
		
		
		i = 0
		j = 0

		my_supersense_tagger.train()
		while i < len(train_input):
			X_train_input = torch.tensor(train_input[i: i + parameters["batch_size"]]).to(DEVICE)
			X_train_rank = torch.tensor(train_rank[i: i + parameters["batch_size"]]).to(DEVICE)
			X_train_idxmap = torch.tensor(train_idxmap[i: i + parameters["batch_size"]]).to(DEVICE)
			Y_train = torch.tensor(train_supersense[i: i + parameters["batch_size"]]).to(DEVICE)
			
			i += parameters["batch_size"]

			my_supersense_tagger.zero_grad()
			log_probs = my_supersense_tagger(X_train_input, X_train_rank, X_train_idxmap)

			# predicted_indices = torch.argmax(log_probs, dim=1)

			loss = loss_function(log_probs, Y_train)
			loss.backward()
			optimizer.step()

			epoch_loss += loss.item()/parameters["batch_size"]

		train_losses.append(epoch_loss)
		
		j = 0
		my_supersense_tagger.eval()
		with torch.no_grad():
			while j < len(train_input):
				X_train_input = torch.tensor(train_input[j: j + parameters["batch_size"]]).to(DEVICE)
				X_train_rank = torch.tensor(train_rank[j: j + parameters["batch_size"]]).to(DEVICE)
				X_train_idxmap = torch.tensor(train_idxmap[j: j + parameters["batch_size"]]).to(DEVICE)
				Y_train = torch.tensor(train_supersense[j: j + parameters["batch_size"]]).to(DEVICE)
				
				j += parameters["batch_size"]
				
				train_log_probs = my_supersense_tagger(X_train_input, X_train_rank, X_train_idxmap)

				predicted_indices = torch.argmax(train_log_probs, dim=1)
				train_epoch_accuracy += torch.sum((predicted_indices == Y_train).int()).item()

			train_accuracies.append(train_epoch_accuracy / len(train_input))


		j = 0
		my_supersense_tagger.eval()
		with torch.no_grad():
			while j < len(freq_dev_input):
				X_freq_dev_input = torch.tensor(freq_dev_input[j: j + parameters["batch_size"]]).to(DEVICE)
				X_freq_dev_rank = torch.tensor(freq_dev_rank[j: j + parameters["batch_size"]]).to(DEVICE)
				X_freq_dev_idxmap = torch.tensor(freq_dev_idxmap[j: j + parameters["batch_size"]]).to(DEVICE)
				Y_freq_dev = torch.tensor(freq_dev_supersense[j: j + parameters["batch_size"]]).to(DEVICE)
				
				j += parameters["batch_size"]
				
				freq_dev_log_probs = my_supersense_tagger(X_freq_dev_input, X_freq_dev_rank, X_freq_dev_idxmap)

				predicted_indices = torch.argmax(freq_dev_log_probs, dim=1)
				freq_dev_epoch_accuracy += torch.sum((predicted_indices == Y_freq_dev).int()).item()

				freq_dev_loss = loss_function(freq_dev_log_probs, Y_freq_dev)
				freq_dev_epoch_loss += freq_dev_loss.item()

			freq_dev_losses.append(freq_dev_epoch_loss / len(freq_dev_input))
			freq_dev_accuracies.append(freq_dev_epoch_accuracy / len(freq_dev_input))
		
		j = 0
		my_supersense_tagger.eval()
		with torch.no_grad():
			while j < len(rand_dev_input):
				X_rand_dev_input = torch.tensor(rand_dev_input[j: j + parameters["batch_size"]]).to(DEVICE)
				X_rand_dev_rank = torch.tensor(rand_dev_rank[j: j + parameters["batch_size"]]).to(DEVICE)
				X_rand_dev_idxmap = torch.tensor(rand_dev_idxmap[j: j + parameters["batch_size"]]).to(DEVICE)
				Y_rand_dev = torch.tensor(rand_dev_supersense[j: j + parameters["batch_size"]]).to(DEVICE)
				
				j += parameters["batch_size"]
				
				rand_dev_log_probs = my_supersense_tagger(X_rand_dev_input, X_rand_dev_rank, X_rand_dev_idxmap)

				predicted_indices = torch.argmax(rand_dev_log_probs, dim=1)
				rand_dev_epoch_accuracy += torch.sum((predicted_indices == Y_rand_dev).int()).item()

				rand_dev_loss = loss_function(rand_dev_log_probs, Y_rand_dev)
				rand_dev_epoch_loss += rand_dev_loss.item()

			rand_dev_losses.append(rand_dev_epoch_loss / len(rand_dev_input))
			rand_dev_accuracies.append(rand_dev_epoch_accuracy / len(rand_dev_input))

		mean_dev_losses.append( (freq_dev_epoch_loss/len(freq_dev_input) + rand_dev_epoch_loss/len(rand_dev_input) ) / 2)
		mean_dev_accuracies.append( (freq_dev_epoch_accuracy/len(freq_dev_input) + rand_dev_epoch_accuracy/len(rand_dev_input) ) / 2)
		
		if epoch >= parameters["patience"]:
		
			if mean_dev_accuracies[epoch] > max_mean_dev_accuracy:
				max_mean_dev_accuracy = mean_dev_accuracies[epoch]
				torch.save(my_supersense_tagger.state_dict(), clf_file)
				patience = parameters["patience"]
				
			else:
				patience = patience - 1
			
			if patience == 0:
				eval_data["early_stopping"] = epoch+1
				break
		else:
			if mean_dev_accuracies[epoch] > max_mean_dev_accuracy:
				max_mean_dev_accuracy = mean_dev_accuracies[epoch]
			torch.save(my_supersense_tagger.state_dict(), clf_file)
		
	eval_data["train_losses"] = [train_loss for train_loss in train_losses]
	eval_data["train_accuracies"] = [train_accuracy for train_accuracy in train_accuracies ]
	
	eval_data["mean_dev_losses"] = [mean_dev_loss for mean_dev_loss in mean_dev_losses]
	eval_data["mean_dev_accuracies"] = [mean_dev_accuracy for mean_dev_accuracy in mean_dev_accuracies]
	
	eval_data["freq_dev_losses"] = [freq_dev_loss for freq_dev_loss in freq_dev_losses]
	eval_data["freq_dev_accuracies"] = [freq_dev_accuracy for freq_dev_accuracy in freq_dev_accuracies]
	
	eval_data["rand_dev_losses"] = [rand_dev_loss for rand_dev_loss in rand_dev_losses]
	eval_data["rand_dev_accuracies"] = [rand_dev_accuracy for rand_dev_accuracy in rand_dev_accuracies]


def evaluation(examples, classifier, parameters, DEVICE, dataset, data, exp):
	batch_size = parameters['batch_size']
	predictions_file = f'./{exp}/{data["clf_id"]}_{dataset}_predictions.text'
	predictions = []
	i = 0
	nb_good_preds = 0
	examples = list(examples)
	eval_input, eval_rank, eval_idxmap, eval_supersenses, eval_senses_ids, eval_lemmas = zip(*examples)
	
	while i < len(examples):

		X_input = eval_input[i: i + batch_size]
		X_rank = eval_rank[i: i + batch_size]
		X_idxmap = eval_idxmap[i: i + batch_size]
		Y = eval_supersenses[i: i + batch_size]
		senses_ids = eval_senses_ids[i: i + batch_size]
		lemmas = eval_lemmas[i: i + batch_size]
		
		i += batch_size
		partial_nb_good_preds= classifier.evaluate(X_input, X_rank, X_idxmap, Y, senses_ids, lemmas, DEVICE, dataset, predictions)
		nb_good_preds += partial_nb_good_preds

	data[f"{dataset}_accuracy"] = nb_good_preds/len(examples)
	
	with open(predictions_file, 'w', encoding='utf-8') as f:
		f.write("example\tpred\tgold\tsense_id\tlemma\n")

		for example, pred, gold, sense_id, lemma in predictions:
			f.write(f"{example}\t{SUPERSENSES[pred]}\t{SUPERSENSES[gold]}\t{sense_id}\t{lemma}\n")



class Baseline:

	def __init__(self):
		self.most_frequent_supersense = ''
	
	def training(self):
		pass

	def evaluation(self, eval_examples):
		eval_examples = list(eval_examples)
		_, _, _, Y_eval, _, _ = zip(*eval_examples)
		return sum([int(supersense == supersense2i[self.most_frequent_supersense]) for supersense in Y_eval])/len(eval_examples)



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


