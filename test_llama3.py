from transformers import AutoTokenizer, AutoModelForCausalLM

"""
# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("lightblue/suzume-llama-3-8B-multilingual")
model = AutoModelForCausalLM.from_pretrained("lightblue/suzume-llama-3-8B-multilingual")

# Define prompt
prompt = "'chien : individu de la race des canidés'. Donne un seul mot correspondant à la classe sémantique de cette définition entre 'person' et 'animal'."

# Tokenize prompt
input_ids = tokenizer.encode(prompt, return_tensors="pt")

# Generate text based on prompt
output = model.generate(input_ids, max_length=100, num_return_sequences=1, temperature=0.7)

# Decode generated output
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

print("Generated Text:")
print(generated_text)
"""



# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B")

# Define prompt
prompt = "'chien : individu de la race des canidés'. Donne un seul mot correspondant à la classe sémantique de cette définition entre 'person' et 'animal'."

# Tokenize prompt
input_ids = tokenizer.encode(prompt, return_tensors="pt")

# Generate text based on prompt
output = model.generate(input_ids, max_length=100, num_return_sequences=1, temperature=0.7)

# Decode generated output
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

print("Generated Text:")
print(generated_text)
