import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM




def pretty_print(prompt, generated_text, gold):
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
	
	print(generated_text[len(prompt)-1:])
	
	print()
	print("*********************************************************")
	print("GOLD SUPERSENSE")
	print("*********************************************************")
	print()
	
	print(gold)
	
	print()
	print()
	print("*********************************************************")



model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
token = "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"

tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
model = AutoModelForCausalLM.from_pretrained(model_name, token=token)

if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

df = pd.read_csv("./sense_data.tsv", sep='\t')
definitions = df["definition"].tolist()
gold_labels = df["supersense"].tolist()

for definition, gold in zip(definitions, gold_labels):

	prompt = f"""INSTRUCTION : Quelle est le supersense de l'entité décrite par la définition suivante ?
	
	SUPERSENSES: 'act', 'animal', 'artifact', 'attribute', 'body', 'cognition', 'communication', 'event', 'feeling', 'food', 'institution', 'act*cognition', 'object', 'possession', 'person', 'phenomenon', 'plant', 'artifact*cognition', 'quantity', 'relation', 'state', 'substance', 'time', 'groupxperson'.
	
	DEFINITION: {definition}
	SUPERSENSE:
	"""

	inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)

	outputs = model.generate(
		inputs['input_ids'], 
		attention_mask=inputs['attention_mask'],
		max_length=200,
		temperature=0.1, 
		num_return_sequences=1
	)

	generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

	pretty_print(prompt, generated_text, gold)

