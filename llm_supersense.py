import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
#from huggingface_hub import login


model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"

token = "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"
#login(token)

tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
model = AutoModelForCausalLM.from_pretrained(model_name, token=token)

prompt = "Once upon a time"

inputs = tokenizer(prompt, return_tensors="pt")

outputs = model.generate(inputs['input_ids'], max_length=50, num_return_sequences=1)

generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(generated_text)

