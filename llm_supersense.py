import pandas as pd


model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

prompt = "Once upon a time"

inputs = tokenizer(prompt, return_tensors="pt")

outputs = model.generate(inputs['input_ids'], max_length=50, num_return_sequences=1)

generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

print(generated_text)

