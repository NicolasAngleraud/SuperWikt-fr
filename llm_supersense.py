import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from random import shuffle
from transformers import AutoTokenizer, AutoModelForCausalLM
from dotenv import load_dotenv
import os


np.random.seed(42)

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


def enss2frss(enss):
	mapping = dict(zip(SUPERSENSES_EN, SUPERSENSES))
	return mapping.get(enss, "Element not found")
                      
                      
supersense2i = {supersense: i for i, supersense in enumerate(SUPERSENSES_EN)}
NB_CLASSES = len(supersense2i)




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



def def_to_prompt(definition, few_shot_examples=None, use_lemma=True):
	
	if few_shot_examples:
		if use_lemma:
			few_shot_prompt = '\n'.join([f"###DEFINITION : {example[2]} = {example[0]} --> ###CLASSE SEMANTIQUE : {example[1]}" for example in few_shot_examples])
		
		else:
			few_shot_prompt = '\n'.join([f"###DEFINITION : {example[0]} --> ###CLASSE SEMANTIQUE : {example[1]}" for example in few_shot_examples])
			
		return f"""###INSTRUCTION : Parmi les classes sémantiques Action, Animal, Objet, Attribut, Corps, Pensée, Communication, Evènement, Sentiment, Nourriture, Institution, Opération, Nature, Possession, Personne, Phénomène, Plante, Document, Quantité, Relation, Etat, Substance, Temps, Groupe, quelle est la classe sémantique la plus adaptée pour décrire la définition suivante ?
{few_shot_prompt}
###DEFINITION : {definition} --> ###CLASSE SEMANTIQUE : """

	else:
		return f"""###INSTRUCTION : Parmi les classes sémantiques Action, Animal, Objet, Attribut, Corps, Pensée, Communication, Evènement, Sentiment, Nourriture, Institution, Opération, Nature, Possession, Personne, Phénomène, Plante, Document, Quantité, Relation, Etat, Substance, Temps, Groupe, quelle est la classe sémantique la plus adaptée pour décrire la définition suivante ?
	
###DEFINITION : {definition}
	
###CLASSE SEMANTIQUE : """


class promptEncoder:
	
	def __init__(self, data_file, tokenizer, device, dataset, use_sample=False, sample_size=32):
		self.data_file = data_file
		self.tokenizer = tokenizer
		self.device = device
		self.dataset = dataset
		self.use_sample = use_sample
		self.sample_size = sample_size
	
	
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
		
		
	def encode(self, use_lemma=True):
		
		df_definitions = pd.read_csv(self.data_file, sep='\t', low_memory=False).astype(str)
		df_definitions = df_definitions[df_definitions['supersense'].isin(SUPERSENSES_EN)]
		df_definitions = df_definitions[(df_definitions['definition'] != "") & (df_definitions['definition'].notna())]
		df_definitions['lemma'] = df_definitions['lemma'].str.replace('_', ' ')
		
		train_df = df_definitions[df_definitions['set']=='train']
		df_definitions = df_definitions[df_definitions['set']==self.dataset]
		
		few_shot_examples = train_df.groupby('supersense', group_keys=False).apply(lambda x: x.sample(1))
		few_shot_examples = [(definition, enss2frss(supersense), lemma.replace('_',' ')) for definition, supersense, lemma in zip(few_shot_examples['definition'].tolist(), few_shot_examples['supersense'].tolist(), few_shot_examples['lemma'].tolist())]
		
		if self.use_sample: df_definitions = df_definitions.sample(self.sample_size)
		
		self.length = len(df_definitions['definition'].tolist())
		
		tokenizer = self.tokenizer
		
		definitions = df_definitions['definition'].tolist()
		supersenses = df_definitions['supersense'].tolist()
		lemmas = df_definitions['lemma'].tolist()
		senses_ids = df_definitions['sense_id'].tolist()
		
		if use_lemma: definitions = [f"{lemma.replace('_',' ')} = {definition}" for definition, lemma in zip(definitions, lemmas)]

		prompts = [def_to_prompt(definition, few_shot_examples, use_lemma) for definition in definitions]
		
		print(prompts[0])
		prompts_encoded = [tokenizer(prompt, add_special_tokens=True) for prompt in prompts]
		
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

			b_prompts_encoded = [prompt_encoded['input_ids'] for prompt_encoded in self.prompts_encoded[start_idx:end_idx]]
			b_attention_masks = [prompt_encoded['attention_mask'] for prompt_encoded in self.prompts_encoded[start_idx:end_idx]]
			b_supersenses_encoded = self.supersenses_encoded[start_idx:end_idx]
			b_senses_ids = self.senses_ids[start_idx:end_idx]
			b_lemmas = self.lemmas[start_idx:end_idx]

			b_prompts_encoded = torch.tensor(b_prompts_encoded).to(device)
			b_attention_masks = torch.tensor(b_attention_masks).to(device)
			b_supersenses_encoded = torch.tensor(b_supersenses_encoded).to(device)
			

			yield b_prompts_encoded, b_supersenses_encoded, b_attention_masks, b_senses_ids, b_lemmas



class LlamaSupersenseClfLM(nn.Module):
	
	def __init__(self, params, tokenizer, hf_token, device):
		super(LlamaSupersenseClfLM, self).__init__()
		self.tokenizer = tokenizer
		self.device = device
		self.params = params
		self.llm = AutoModelForCausalLM.from_pretrained(model_name, token=hf_token).to(self.device)
		
		
	def forward(self, prompt_enc, mask):
		outputs = self.llm(prompt_enc, attention_mask=mask, use_cache=False)
		logits = outputs.logits
		
		#logits_first_token = logits[0, -1, :]
		#log_probs = torch.nn.functional.log_softmax(logits_first_token, dim=-1)
		
		logits_last_token = logits[:, -1, :]  # shape: (batch_size, vocab_size)
		log_probs = torch.nn.functional.log_softmax(logits_last_token, dim=-1)  # shape: (batch_size, vocab_size)
		
		# gen_toks = supersenses_tok[best_indices]
		# probs = probs.cpu().numpy()
		# supersense_probs = [probs[ss_tok] for ss_tok in supersenses_tok]
		# best_index = np.argmax(supersense_probs)
		# gen_tok = supersenses_tok[best_index]
		return log_probs
		
	def train_clf(self):
		pass
		
	def save_clf(self):
		pass
		
	def load_clf(self):
		pass
		
	def evaluate(self, data_encoder, supersenses_tok):
		self.eval()
		accuracy = 0
		k = 0
		with torch.no_grad():
			for b_prompts_encoded, b_supersenses_encoded, b_attention_masks, b_senses_ids, b_lemmas in data_encoder.make_batches(batch_size=self.params['batch_size'], shuffle_data=False):
				k+=1
				print("BATCH ", k, b_senses_ids)
				
				log_probs = self.forward(b_prompts_encoded, b_attention_masks)
				supersense_probs = log_probs[:, supersenses_tok]  # shape: (batch_size, num_classes)
				predicted_indices = torch.argmax(supersense_probs, dim=-1)  # shape: (batch_size,)
				accuracy += torch.sum((predicted_indices == b_supersenses_encoded).int()).item()
				
			return accuracy / data_encoder.length
			
	
	def predict(self, data_encoder, supersenses_tok, id2ss):
		self.eval()
		predictions = {"lemma":[], "sense_id":[], "gold":[], "pred":[], "definition":[]}
		with torch.no_grad():
			for b_prompts_encoded, b_supersenses_encoded, b_attention_masks, b_senses_ids, b_lemmas in data_encoder.make_batches(batch_size=self.params['batch_size'], shuffle_data=False):
				
				
				log_probs = self.forward(b_prompts_encoded, b_attention_masks)
				supersense_probs = log_probs[:, supersenses_tok]  # shape: (batch_size, num_classes)
				predicted_indices = torch.argmax(supersense_probs, dim=-1).tolist()  # shape: (batch_size,)
				
				pred = [SUPERSENSES_EN[SUPERSENSES.index(id2ss[i])] for i in predicted_indices]
				gold = [SUPERSENSES_EN[i] for i in b_supersenses_encoded.tolist()]
				prompts = [self.tokenizer.decode(token_ids.tolist(), skip_special_tokens=True) for token_ids in b_prompts_encoded]
				definitions = [prompt.split('###DEFINITION : ')[1].split('###TYPE SEMANTIQUE :')[0].strip()  for prompt in prompts]
				
				predictions['lemma'].extend(b_lemmas)
				predictions['sense_id'].extend(b_senses_ids)
				predictions['gold'].extend(gold)
				predictions['pred'].extend(pred)
				predictions['definition'].extend(definitions)
				
			return predictions
	
	def evaluate_and_predict(self, data_encoder, supersenses_tok, id2ss):
	
		accuracy = self.evaluate(data_encoder, supersenses_tok)
		predictions = self.predict(data_encoder, supersenses_tok, id2ss)
		
		return accuracy, predictions	
		
		

class LlamaSupersenseClf(nn.Module):
	
	def __init__(self):
		super(LlamaSupersenseClf, self).__init__()
		#NB_CLASSES = 30
		#model.lm_head = torch.nn.Linear(in_features=model.lm_head.in_features, out_features=NB_CLASSES)
		#print(model)
		
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
		
		




if __name__ == '__main__':
	
	device = 'cpu'
	
	data_file = "./sense_data.tsv"
	
	model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
	
	params = {
		"batch_size": 1
	}
	
	load_dotenv()
	hf_token = os.getenv("HUGGINGFACE_TOKEN")

	if hf_token is None:
		raise ValueError("HUGGINGFACE_TOKEN environment variable is not set.")

	tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
	tokenizer.pad_token_id = tokenizer.eos_token_id
	
	model = LlamaSupersenseClfLM(params, tokenizer, hf_token, device)
	
	supersenses_tok = [tokenizer.encode(supersense, add_special_tokens=False)[0] for supersense in SUPERSENSES]
	id2ss = {id_tok: SUPERSENSES[i] for i, id_tok in enumerate(supersenses_tok)}
	
	'''
	freq_dev_encoder = promptEncoder(data_file=data_file, tokenizer=tokenizer, device=device, dataset='freq-dev')
	freq_dev_encoder.encode()
	accuracy_freq_dev = model.evaluate(freq_dev_encoder, supersenses_tok)
	print("FREQ DEV ACCURACY = ", accuracy_freq_dev)
	'''
	
	rand_dev_encoder = promptEncoder(data_file=data_file, tokenizer=tokenizer, device=device, dataset='rand-dev')
	rand_dev_encoder.encode()
	accuracy_rand_dev = model.evaluate(rand_dev_encoder, supersenses_tok)
	print("RAND DEV ACCURACY = ", accuracy_rand_dev)	
	
	print("Process done.")
