import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM




def pretty_print(prompt, generated_text):
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
	
	print(generated_text[len(prompt):])
	
	print()
	print()
	print("*********************************************************")



model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
token = "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"

tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
model = AutoModelForCausalLM.from_pretrained(model_name, token=token)

if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

prompt = """Quelle est le type sémantique de l'entité associé à la définition suivante ?
Types possibles: Animal, Plant, Act, Cognition, Feeling.
Définition: Sentiment éprouvé lors d'une situation difficile.
Type sémantique:
"""

inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)

outputs = model.generate(
    inputs['input_ids'], 
    attention_mask=inputs['attention_mask'],
    max_length=100, 
    num_return_sequences=1
)

generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

pretty_print(prompt, generated_text)

