from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import torch.nn as nn
import argparse
import dataEncoder as data
import random
import pandas as pd
from peft import get_peft_model, PromptTuningConfig, TaskType, PromptEmbedding


## MODELS
#lightblue/suzume-llama-3-8B-multilingual
#meta-llama/Meta-Llama-3-8B
#meta-llama/Meta-Llama-3-8B-Instruct
#mistralai/Mistral-7B-Instruct-v0.2

## TEST SMALL MODEL
#bigscience/bloom-1b7


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


API_TOKEN = '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'

ss2classe = {
	'act': 'action',
	'animal': 'animal',
	'artifact': 'objet',
	'attribute': 'propriété',
	'body': 'anatomie',
	'cognition': 'pensée',
	'communication': 'langage',
	'event': 'événement',
	'feeling': 'sentiment',
	'food': 'nourriture',
	'institution': 'institution',
	'act*cognition': 'discours',
	'object': 'nature',
	'possession': 'possession',
	'person': 'personne',
	'phenomenon': 'phénomène',
	'plant': 'plante',
	'artifact*cognition': 'document',
	'quantity': 'quantité',
	'relation': 'relation',
	'state': 'état',
	'substance': 'substance',
	'time': 'temps',
	'groupxperson': 'collectif'}

fr_supersenses = [ss2classe[c] for c in SUPERSENSES]

def get_parser_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("-device_id", choices=['cpu','0', '1', '2', '3'], default='cpu', help="Id of the GPU.")
	parser.add_argument("-data_file", default="./data.xlsx", help="The excel file containing all the annotated sense data from wiktionary.")
	parser.add_argument("-batch_size", choices=['2', '4', '8', '16', '32', '64'], help="batch size for the classifier.")
	parser.add_argument('-v', "--trace", action="store_true", help="Toggles the verbose mode. Default=False")
	args = parser.parse_args()
	return args


if __name__ == '__main__':
	args = get_parser_args()

	# DEVICE setup
	device_id = args.device_id
	if device_id == "cpu": DEVICE = "cpu"
	else:
		if torch.cuda.is_available(): DEVICE = torch.device("cuda:" + args.device_id)
	
	tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct", use_auth_token=API_TOKEN)
		
	peft_config = PromptTuningConfig(
										peft_type="PROMPT_TUNING", 
										task_type=TaskType.CAUSAL_LM, 
										inference_mode=False,
										num_virtual_tokens=20,
										token_dim=768,
										num_transformer_submodules=1,
										num_attention_heads=12,
										num_layers=12,
										prompt_tuning_init="TEXT",
										prompt_tuning_init_text="Predict if the semantic type of the definition is person or animal",
										tokenizer_name_or_path="Meta-Llama-3-8B-Instruct")
	
	model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct", use_auth_token=API_TOKEN).to(DEVICE)
	model = get_peft_model(model, peft_config)
	
	model.print_trainable_parameters()
	
	
	
	################################################################################################################################
	
	'''
	for c in ss2classe:
		classe = ss2classe[c]
		print(classe, tokenizer.convert_ids_to_tokens(tokenizer(classe)['input_ids']))
	
	for c in fr_supersenses: print(c)
	
	'''
	
	'''
	df_definitions = pd.read_excel(args.data_file, sheet_name='senses', engine='openpyxl')
	df_definitions = df_definitions[df_definitions['supersense'].isin(SUPERSENSES)]
	df_definitions = df_definitions[(df_definitions['definition'] != "") & (df_definitions['definition'].notna())]
	df_definitions['lemma'] = df_definitions['lemma'].str.replace('_', ' ')
	
	eval_df = []
	
	num_indices = 25
	indices = random.sample(range(len(df_definitions)), num_indices)
	for i, index in enumerate(indices):
		print(i+1)
		row = df_definitions.iloc[index]
		definition = row["definition"]
		sense_id = row["sense_id"]
		dataset = row["set"]
		lemma = row["lemma"]
		gold = row["supersense"]

		prompt = """<s>[INST]Choisis la classe sémantique décrivant le mieux la définition donnée par la suite parmi les classes suivantes: act, animal, artifact, attribute, body, cognition, communication, event, feeling, food, institution, object, possession, person, phenomenon, plant, quantity, relation, state, substance, time. Donne simplement en réponse la classe choisie après 'classe sémantique: ' et ne rajoute aucune autre information. [/INST] </s>
		définition: {BODY} --> classe sémantique: """.format(BODY=definition)
		
		inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
		output = model.generate(**inputs, max_length=inputs.input_ids.size(1) + 50, num_return_sequences=1, temperature=0.2)

		generated_classification = tokenizer.decode(output[0], skip_special_tokens=True)

		# print("Generated Classification:", generated_classification)
		
		eval_df.append({"lemma": lemma, "sense_id": sense_id, "set": dataset, "definition": definition, "gold": gold, "prompt":prompt,"answer":generated_classification})
		
	eval_df = pd.DataFrame(eval_df)
	eval_df.to_excel("./eval_sample_def_zero_shot_prompting_llama3.xlsx", index=False)
	'''
