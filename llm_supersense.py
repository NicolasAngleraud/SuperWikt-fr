import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from random import shuffle
from transformers import AutoTokenizer, AutoModelForCausalLM
from dotenv import load_dotenv
import os



SUPERSENSES_EN = ['act', 'animal', 'artifact', 'attribute', 'body', 'cognition',
               'communication', 'event', 'feeling', 'food', 'institution', 'act*cognition',
               'object', 'possession', 'person', 'phenomenon', 'plant', 'artifact*cognition',
               'quantity', 'relation', 'state', 'substance', 'time', 'groupxperson']


SUPERSENSES = ['Action', 'Animal', 'Objet', 'Attribut', 'Corps', 'Pensée',
               'Communication', 'Evènement', 'Sentiment', 'Nourriture', 'Institution', 'Opération',
               'Nature', 'Possession', 'Personne', 'Phénomène', 'Plante', 'Document',
               'Quantité', 'Relation', 'Etat', 'Substance', 'Temps', 'Groupe']


HYPERSENSES = {"dynamic_situation": ["act", "event", "phenomenon"],
               "stative_situation": ["attribute", "state", "feeling", "relation"],
               "animate_entity": ["animal", "person"],
               "inanimate_entity": ["artifact", "food", "body", "object", "plant", "substance"],
               "informational_object": ["cognition", "communication"],
               "quantification": ["quantity", "part", "group"],
               "other": ["institution", "possession", "time"]
               }
                      
supersense2i = {supersense: i for i, supersense in enumerate(SUPERSENSES_EN)}
NB_CLASSES = len(supersense2i)
#supersenses_tok = [tokenizer.encode(supersense, add_special_tokens=False)[0] for supersense in SUPERSENSES]
#id2ss = {id_tok: SUPERSENSES[i] for i, id_tok in enumerate(supersenses_tok)}



def pretty_print(prompt, pred, gold):
	print()
	print()
	print("*********************************************************")
	print("PROMPT")
	print("*********************************************************")
	print()
	
	print(prompt)
	
	print()
	print("*********************************************************")
	print("ANSWER")
	print("*********************************************************")
	print()
	
	print(SUPERSENSES_EN[SUPERSENSES.index(id2ss[pred])])
	
	print()
	print("*********************************************************")
	print("GOLD SUPERSENSE")
	print("*********************************************************")
	print()
	
	print(gold)
	
	print()
	print()
	print("*********************************************************")



def def_to_prompt(definition):
	return f"""###INSTRUCTION : Parmi les classes sémantiques Action, Animal, Objet, Attribut, Corps, Pensée, Communication, Evènement, Sentiment, Nourriture, Institution, Opération, Nature, Possession, Personne, Phénomène, Plante, Document, Quantité, Relation, Etat, Substance, Temps, Groupe, quelle est la classe sémantique la plus adaptée pour décrire la définition suivante ?
	
	###DEFINITION : {definition}
	
	###TYPE SEMANTIQUE : """


class promptEncoder:
	
	def __init__(self, data_file, tokenizer, device):
		self.data_file = data_file
		self.tokenizer = tokenizer
		self.device = device
	
	
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
		
		df_definitions = pd.read_csv(self.data_file, sep='\t', low_memory=False).astype(str)
		df_definitions = df_definitions[df_definitions['supersense'].isin(SUPERSENSES_EN)]
		df_definitions = df_definitions[(df_definitions['definition'] != "") & (df_definitions['definition'].notna())]
		df_definitions['lemma'] = df_definitions['lemma'].str.replace('_', ' ')
		
		tokenizer = self.tokenizer
		
		definitions = df_definitions['definition'].tolist()
		supersenses = df_definitions['supersense'].tolist()
		lemmas = df_definitions['lemma'].tolist()
		senses_ids = df_definitions['sense_id'].tolist()
		
		definitions_with_lemma = [f"{lemma.replace('_',' ')} = {definition}" for definition, lemma in zip(definitions, lemmas)]
		
		prompts = [def_to_prompt(definition) for definition in definitions_with_lemma]
		prompts_encoded = [tokenizer(prompt, add_special_tokens=True, return_tensors='pt') for prompt in prompts]
		
		#prompts_encoded, _ = self.truncate(prompts_encoded)
		#prompts_encoded = self.pad(prompts_encoded, pad_id=2)
		#prompts_encoded, _ = self.add_special_tokens(prompts_encoded, cls_id=0, sep_id=1)

		supersenses_encoded = [supersense2i[supersense] for supersense in supersenses]
		
		self.prompts_encoded = prompts_encoded
		self.supersenses_encoded = supersenses_encoded
		self.lemmas = lemmas
		self.senses_ids = senses_ids

		
	def shuffle_data(self):
	
		data = zip(self.prompts_encoded, self.supersenses_encoded, self.senses_ids, self.lemmas)
		data = list(data)
		shuffle(data)
		prompts_encoded, supersenses_encoded, senses_ids, lemmas = zip(*data)
		
		self.prompts_encoded = prompts_encoded
		self.supersenses_encoded = supersenses_encoded
		self.lemmas = lemmas
		self.senses_ids = senses_ids
		
		
	def make_batches(self, batch_size=1, shuffle_data=False):
		device = self.device
		if shuffle_data: self.shuffle_data()

		k = 0
		while k < len(self.supersenses_encoded):

			start_idx = k
			end_idx = k+batch_size if k+batch_size <= len(self.supersenses_encoded) else len(self.supersenses_encoded)
			k += batch_size

			b_prompts_encoded = self.prompts_encoded[start_idx:end_idx]
			b_supersenses_encoded = self.supersenses_encoded[start_idx:end_idx]
			b_senses_ids = self.senses_ids[start_idx:end_idx]
			b_lemmas = self.lemmas[start_idx:end_idx]

			b_prompts_encoded = torch.tensor(b_prompts_encoded).to(device)
			b_supersenses_encoded = torch.tensor(b_supersenses_encoded).to(device)
			

			yield b_prompts_encoded, b_supersenses_encoded, b_senses_ids, b_lemmas



class LlamaSupersenseClf(nn.Module):
	
	def __init__(self):
		super(LlamaSupersenseClf, self).__init__()
		
	def forward(self):
		pass
		
	def train_clf(self):
		pass
		
	def save_clf(self):
		pass
		
	def load_clf(self):
		pass
		
	def predict(self):
		pass
		
	def predict_and_evaluate(self):
		pass
	
'''
n = 0
for definition, gold in zip(definitions, gold_labels):
	
	prompt = f"""###INSTRUCTION : Parmi les classes sémantiques Action, Animal, Objet, Attribut, Corps, Pensée, Communication, Evènement, Sentiment, Nourriture, Institution, Opération, Nature, Possession, Personne, Phénomène, Plante, Document, Quantité, Relation, Etat, Substance, Temps, Groupe, quel est la classe sémantique la plus adaptée pour décrire la définition suivante ?
	
	###DEFINITION: {definition}
	
	###TYPE SEMANTIQUE: 
	"""

	inputs = tokenizer(prompt, return_tensors="pt")
	input_ids = inputs['input_ids']
	

	with torch.no_grad():
		outputs = model(input_ids, use_cache=False)
		logits = outputs.logits

	logits_first_token = logits[0, -1, :]
	probs = torch.nn.functional.softmax(logits_first_token, dim=-1)
	probs = probs.cpu().numpy()
	
	supersense_probs = [probs[ss_tok] for ss_tok in supersenses_tok]

	best_index = np.argmax(supersense_probs)
	gen_tok = supersenses_tok[best_index]
	
	pretty_print(prompt, gen_tok, gold)
	
	n += 1
	if n>50: break
	
#NB_CLASSES = 30
#model.lm_head = torch.nn.Linear(in_features=model.lm_head.in_features, out_features=NB_CLASSES)
#print(model)
'''

if __name__ == '__main__':
	
	device = 'cpu'
	
	data_file = "./sense_data.tsv"
	
	model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
	
	load_dotenv()
	hf_token = os.getenv("HUGGINGFACE_TOKEN")

	if hf_token is None:
		raise ValueError("HUGGINGFACE_TOKEN environment variable is not set.")

	tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
	#model = AutoModelForCausalLM.from_pretrained(model_name, token=hf_token)

	tokenizer.pad_token_id = tokenizer.eos_token_id
	
	data_encoder = promptEncoder(data_file=data_file, tokenizer=tokenizer, device=device)
	
	data_encoder.encode()
	
	print(data_encoder.prompts_encoded[0])
	print(data_encoder.supersenses_encoded[0])
	print(data_encoder.lemmas[0])
	print(data_encoder.senses_ids[0])
	print(tokenizer.decode(data_encoder.prompts_encoded[0]['input_ids'].squeeze(), skip_special_tokens=True))
	
	data_encoder.shuffle_data()
	
	print(data_encoder.prompts_encoded[0])
	print(data_encoder.supersenses_encoded[0])
	print(data_encoder.lemmas[0])
	print(data_encoder.senses_ids[0])
	print(tokenizer.decode(data_encoder.prompts_encoded[0]['input_ids'].squeeze(), skip_special_tokens=True))
	
	print("Process done.")
