from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import torch.nn as nn
import argparse
import dataEncoder as data
import random
import pandas as pd


## MODELS
#lightblue/suzume-llama-3-8B-multilingual
#meta-llama/Meta-Llama-3-8B
#meta-llama/Meta-Llama-3-8B-Instruct
#mistralai/Mistral-7B-Instruct-v0.2


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
	
	tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B", use_auth_token=API_TOKEN)
	model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B", use_auth_token=API_TOKEN).to(DEVICE)
	
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

		prompt = """<s>[INST]Choisis la classe sémantique décrivant le mieux la définition suivante parmi les vingt quatre classes suivantes: act, animal, artifact, attribute, body, cognition, communication, event, feeling, food, institution, act*cognition, object, possession, person, phenomenon, plant, artifact*cognition, quantity, relation, state, substance, time, groupxperson. Donne simplement en réponse la classe choisie après 'classe sémantique: ' et ne rajoute aucune autre information. [/INST] </s>
		définition: {BODY} --> classe sémantique: """.format(BODY=definition)
		
		inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
		output = model.generate(**inputs, max_length=inputs.input_ids.size(1) + 50, num_return_sequences=1, temperature=0.2)

		generated_classification = tokenizer.decode(output[0], skip_special_tokens=True)

		# print("Generated Classification:", generated_classification)
		
		eval_df.append({"lemma": lemma, "sense_id": sense_id, "set": dataset, "definition": definition, "gold": gold, "prompt":prompt,"answer":generated_classification})
		
	eval_df = pd.DataFrame(eval_df)
	eval_df.to_excel("./eval_sample_def_zero_shot_prompting_llama3.xlsx", index=False)
