import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM




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
                      
supersense2i = {supersense: i for i, supersense in enumerate(SUPERSENSES)}
NB_CLASSES = len(supersense2i)



def pretty_print(prompt, gen_tok, gold):
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
	
	print(id2ss[gen_tok])
	
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

supersenses_tok = [tokenizer.encode(supersense, add_special_tokens=False)[0] for supersense in SUPERSENSES]

for ss_toks in [tokenizer.encode(supersense, add_special_tokens=False) for supersense in SUPERSENSES]:
	print(ss_toks)
	
id2ss = {idx: SUPERSENSES[i] for i, idx in enumerate(supersenses_tok)}

if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

df = pd.read_csv("./sense_data.tsv", sep='\t')
definitions = df["definition"].tolist()
gold_labels = df["supersense"].tolist()


n = 0
for definition, gold in zip(definitions, gold_labels):

	prompt = f"""INSTRUCTION : Quelle est le supersense de l'entité décrite par la définition suivante ?
	
	SUPERSENSES: 'act', 'animal', 'artifact', 'attribute', 'body', 'cognition', 'communication', 'event', 'feeling', 'food', 'institution', 'act*cognition', 'object', 'possession', 'person', 'phenomenon', 'plant', 'artifact*cognition', 'quantity', 'relation', 'state', 'substance', 'time', 'groupxperson'.
	
	DEFINITION: {definition}
	SUPERSENSE:
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
	





####################################################################################################################################

"""


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


outputs = model.generate(
	inputs['input_ids'], 
	attention_mask=inputs['attention_mask'],
	max_length=200,
	temperature=0.1, 
	num_return_sequences=1
)


generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

pretty_print(prompt, generated_text, gold)

"""
