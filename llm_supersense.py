import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM


model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
token = "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"

tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
model = AutoModelForCausalLM.from_pretrained(model_name, token=token)

if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

prompt = "Le chinois est une langue"

inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)

outputs = model.generate(
    inputs['input_ids'], 
    attention_mask=inputs['attention_mask'],
    max_length=50, 
    num_return_sequences=1
)

generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(generated_text)

