from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import torch.nn as nn
import argparse


## MODELS
#lightblue/suzume-llama-3-8B-multilingual
#meta-llama/Meta-Llama-3-8B
#meta-llama/Meta-Llama-3-8B-Instruct


API_TOKEN = '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'


def get_parser_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("-device_id", choices=['0', '1', '2', '3'], help="Id of the GPU.")
	parser.add_argument("-data_file", default="./data.xlsx", help="The excel file containing all the annotated sense data from wiktionary.")
	parser.add_argument("-batch_size", choices=['2', '4', '8', '16', '32', '64'], help="batch size for the classifier.")
	parser.add_argument('-v', "--trace", action="store_true", help="Toggles the verbose mode. Default=False")
	args = parser.parse_args()
	return args


if __name__ == '__main__':
	args = get_parser_args()

	# DEVICE setup
	device_id = args.device_id
	if torch.cuda.is_available():
		DEVICE = torch.device("cuda:" + args.device_id)
		
	
	def_ = "Mammifère domestique, ongulé de l’ordre des suidés ; porc."

	prompt = """Choisis la classe sémantique décrivant le mieux la définition suivant parmi les quatre classes suivantes: person, animal, mineral, plant.

	définition: {BODY} --> classe sémantique: """.format(BODY=def_)
	
	tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B", use_auth_token=API_TOKEN)
	model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B", use_auth_token=API_TOKEN).to(DEVICE)

	inputs = tokenizer(prompt, return_tensors="pt")
	output = model.generate(**inputs, max_length=inputs.input_ids.size(1) + 50, num_return_sequences=1, temperature=0.2)

	generated_classification = tokenizer.decode(output[0], skip_special_tokens=True)


	print("Generated Classification:", generated_classification)
